//! What shaders wrote into copies in the card's memory, put into title
//! memory when something touches it, not when it comes back
//! (`VICTORY_LAZY_BACK=1`; off by default).
//!
//! After every submission the pieces of copies in the card's memory that
//! its shaders changed are fetched and merged into title memory at once
//! (`Shadow::card_merge`, `Shadow::card_merge_relay`): 27 to 30 ms and some
//! 55 MB a frame in the first rooms. `VICTORY_BACK_READS` counted who
//! reads those bytes before the card writes them again: the title less
//! than half a percent of them; 72% only our own refreshes of copies.
//!
//! With the switch, a fetched piece whose title memory nobody wrote since
//! the copy last took it is not merged. The bytes the card changed (from
//! the first that differs from the copy's CPU side to the last, as the
//! merge would write them) are kept here, by 64 KiB page of title memory,
//! and the 4 KiB pages they lie in are made `PAGE_NOACCESS`. The first
//! touch of such a page, from any thread and of any kind, resolves it: our
//! vectored handler (`crate::watch::on_write`, also asked by the fast
//! copies' handlers) merges every piece kept for that 64 KiB page into
//! title memory, byte by byte as the merge on landing would have
//! (`Owner::merge`), gives the pages their access back, and the touch goes
//! on. Our own readers of title memory ask first ([`need`]): the copies'
//! refreshes, the title's copies and fills of a list, textures, the kernel
//! reads the runtime makes for the title (`crate::watch::writing`). Their
//! reads through the kernel would fail on such a page instead of faulting
//! into the handler.
//!
//! What is told at once, as if the bytes had landed: the journal of
//! written pages (`crate::watch::wrote_own`), so that every OTHER copy of
//! the same memory takes the page again at its next refresh (and that
//! refresh resolves it); the copy that wrote notes itself current, as
//! `VICTORY_BACK_OWN_VIEW` does, so it does not read back its own bytes;
//! and the placed textures there (`crate::descriptor::host::remapped`).
//!
//! Pieces of the same copy fetched again replace what is kept for their
//! bytes: the copy's CPU side is not touched while a piece waits, so the
//! next fetch compares with the same bytes and holds every change since.
//! A fetch that differs nowhere drops what was kept under it.
//!
//! Merged on landing as before ("double dirty", and everything that is
//! not plain): a page the copy never took or the journal has written since
//! it did (the caller looks, `Shadow::lazy_offer`); a 4 KiB page whose
//! kernel write-watch bit is set (written since the last asking, by the
//! title or by a merge of ours), looked at AFTER the protection is on, so
//! that no write slips between; memory that is not write-watched (views of
//! physical pages, which the guards hold, and anything else); a protection
//! that was not `PAGE_READWRITE`; bytes another copy has a piece of; and
//! more than `VICTORY_LAZY_MB` (256) kept in all.
//!
//! The handler does memory work only: no Direct3D, no fence, no log line.
//! A thread that is resolving or merging is marked ([`busy`]): a fault
//! there is left to the other handlers (a fast copy then fails as for a
//! page that is not there) and its own asks return at once, so nothing
//! waits on itself.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard};

use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READWRITE, VirtualProtect};

use crate::watch::PAGE;

/// The kernel's page: the unit of protection and of the write-watch bits.
const KERNEL_PAGE: usize = 4096;
/// Kernel pages in one 64 KiB page, one bit each in [`Page::mask`].
const SUBPAGES: usize = PAGE / KERNEL_PAGE;
const _: () = assert!(SUBPAGES <= 16);
/// Pages resolved lately, for a fault that raced with another thread's
/// resolution of its page ([`fault`]).
const RECENT: usize = 4096;
/// The longest a thread waits for another's resolution of a page before it
/// gives up ([`waited_too_long`]): a resolution is microseconds; one that
/// never ends is a fault of ours, and a wait for it must not spin for good.
const WAIT_LIMIT: std::time::Duration = std::time::Duration::from_secs(2);

/// What a kept piece goes back into: a copy in the card's memory.
pub(crate) trait Owner: Send + Sync {
    /// `gpu` is what the card's side holds for title memory at `at` on,
    /// where it differed from the copy's CPU side when it was fetched;
    /// `after` is the relay's "fetched after" (`Shadow::card_merge_relay`),
    /// `None` for a merge of a waited-for submission (`Shadow::card_merge`).
    /// Title memory there is readable and writable now: merge into it as
    /// the merge on landing would have.
    fn merge(&self, at: usize, gpu: &[u8], after: Option<u64>);
}

/// A kept piece: inside one 64 KiB page of title memory.
struct Piece {
    owner: Arc<dyn Owner>,
    at: usize,
    bytes: Vec<u8>,
    after: Option<u64>,
}

impl Piece {
    fn end(&self) -> usize {
        self.at + self.bytes.len()
    }

    fn of(&self, owner: *const u8) -> bool {
        Arc::as_ptr(&self.owner).cast::<u8>() == owner
    }
}

/// What is kept for one 64 KiB page of title memory.
#[derive(Default)]
struct Page {
    /// In the order they came back (the merges run in it).
    pieces: Vec<Piece>,
    /// Its kernel pages made `PAGE_NOACCESS` by us.
    mask: u16,
    /// A thread took the pieces and is merging them: others wait.
    resolving: bool,
}

struct Store {
    pages: BTreeMap<usize, Page>,
    /// Bytes kept in all.
    bytes: usize,
    /// Pages resolved lately, oldest first, and the same as a set.
    recent: VecDeque<usize>,
    recent_set: BTreeSet<usize>,
}

static STORE: Mutex<Store> = Mutex::new(Store {
    pages: BTreeMap::new(),
    bytes: 0,
    recent: VecDeque::new(),
    recent_set: BTreeSet::new(),
});

/// Pages kept ([`Store::pages`]), for the asks that find none with no lock.
static PAGES: AtomicUsize = AtomicUsize::new(0);
/// Pages in [`Store::recent`].
static RECENT_PAGES: AtomicUsize = AtomicUsize::new(0);

/// Copies whose last hold was a resolved piece: let go on a thread of
/// ours ([`offer`]), not in the handler (letting a copy go releases its
/// Direct3D buffers).
static GRAVE: Mutex<Vec<Arc<dyn Owner>>> = Mutex::new(Vec::new());

fn store() -> MutexGuard<'static, Store> {
    STORE.lock().unwrap_or_else(|e| e.into_inner())
}

thread_local! {
    /// This thread is resolving pages or merging a piece on landing.
    static BUSY: Cell<bool> = const { Cell::new(false) };
}

/// Marks the thread as resolving or merging ([`busy`]) while held.
pub(crate) struct Busy(bool);

impl Drop for Busy {
    fn drop(&mut self) {
        BUSY.with(|b| b.set(self.0));
    }
}

/// While the result is held, this thread's own asks ([`need`]) return at
/// once and a fault of its own is not resolved: what it touches it has
/// resolved itself, or is merging on landing (`Shadow::card_merge`, which
/// asks for its range first).
pub(crate) fn busy() -> Busy {
    Busy(BUSY.with(|b| b.replace(true)))
}

fn is_busy() -> bool {
    BUSY.with(Cell::get)
}

/// For the tests: on whatever the environment says.
#[cfg(test)]
pub(crate) static FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether pieces are kept (`VICTORY_LAZY_BACK=1`).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCE.load(Relaxed) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_LAZY_BACK").is_some_and(|v| v == "1" || v == "alt");
        if on {
            tracing::info!(
                "what shaders wrote into copies in the card's memory goes into title memory when something touches it, not when it comes back (VICTORY_LAZY_BACK=1; at most {} MB kept)",
                limit() >> 20
            );
        }
        on
    })
}

/// Whether a piece landing now may be kept: with [`on`], always, or with
/// `VICTORY_LAZY_BACK=alt` twenty seconds yes, twenty no, in turn (an A/B
/// within one run). Pieces kept stay kept and are put in when touched in
/// either phase: only the keeping of new ones stops.
pub(crate) fn keeping() -> bool {
    #[cfg(test)]
    if FORCE.load(Relaxed) {
        return true;
    }
    static ALT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ALT.get_or_init(|| std::env::var_os("VICTORY_LAZY_BACK").is_some_and(|v| v == "alt")) {
        crate::host::alternate(3, "VICTORY_LAZY_BACK")
    } else {
        on()
    }
}

/// The most bytes kept (`VICTORY_LAZY_MB`, 256): past it pieces merge on
/// landing.
fn limit() -> usize {
    static MB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MB.get_or_init(|| std::env::var("VICTORY_LAZY_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256)) << 20
}

/// Why a piece was merged on landing ([`Offer::No`]), for the counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// The copy never took the page, or it was written since (the journal).
    Written,
    /// Bytes of the copy's CPU side the card had not received (relay).
    Sending,
    /// A kernel write-watch bit of its pages is set.
    Bits,
    /// Not write-watched, or not protectable as plain read-write memory.
    Plain,
    /// Another copy has a piece of these bytes kept.
    Others,
    /// [`limit`] reached.
    Full,
    /// Nothing differs and nothing was kept there.
    Nothing,
}

/// What [`offer`] did with a piece.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Offer {
    /// Kept: title memory `[from, to)` waits (the caller tells the
    /// journal and the textures).
    Kept(usize, usize),
    /// Nothing differed: what was kept of these bytes for the same copy is
    /// dropped (the card holds the CPU side's bytes again there).
    Dropped,
    /// Not kept: the caller merges now, as without the switch.
    No(Why),
}

/// Since the last line: pieces kept and their bytes; pieces merged on
/// landing by [`Why`] (Written, Sending, Bits, Plain, Others, Full); pages
/// resolved by a fault, their bytes and microseconds; pages resolved by
/// our own readers, their bytes and microseconds; pages whose pieces were
/// dropped because the memory changed under them; pieces dropped by a
/// fetch that differed nowhere.
static COUNT: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
const KEPT: usize = 0;
const KEPT_BYTES: usize = 1;
const NOW: usize = 2; // ..NOW + 6
const FAULTS: usize = 8;
const FAULT_BYTES: usize = 9;
const FAULT_US: usize = 10;
const NEEDS: usize = 11;
const NEED_BYTES: usize = 12;
const NEED_US: usize = 13;
const LOST: usize = 14;
const DROPPED: usize = 15;

/// A piece was merged on landing because of `why` (the caller's own
/// reasons, before [`offer`]).
pub(crate) fn merged_now(why: Why) {
    let i = match why {
        Why::Written => 0,
        Why::Sending => 1,
        Why::Bits => 2,
        Why::Plain => 3,
        Why::Others => 4,
        Why::Full => 5,
        Why::Nothing => return,
    };
    COUNT[NOW + i].fetch_add(1, Relaxed);
}

/// The kernel pages of `[from, to)` that lie in page `page`, as bits.
fn subpages(page: usize, from: usize, to: usize) -> u16 {
    let (lo, hi) = (from.max(page * PAGE), to.min((page + 1) * PAGE));
    if lo >= hi {
        return 0;
    }
    let (a, b) = ((lo - page * PAGE) / KERNEL_PAGE, (hi - page * PAGE).div_ceil(KERNEL_PAGE));
    (a..b).fold(0u16, |m, i| m | 1 << i)
}

/// The runs of set bits of `mask` in page `page`, as (address, bytes).
fn runs(page: usize, mask: u16) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < SUBPAGES {
        if mask & 1 << i == 0 {
            i += 1;
            continue;
        }
        let first = i;
        while i < SUBPAGES && mask & 1 << i != 0 {
            i += 1;
        }
        out.push((page * PAGE + first * KERNEL_PAGE, (i - first) * KERNEL_PAGE));
    }
    out
}

/// Sets the protection of `[at, at + len)`; the one it had, or `None` if
/// the call failed (nothing committed there).
fn protect(at: usize, len: usize, protection: PAGE_PROTECTION_FLAGS) -> Option<PAGE_PROTECTION_FLAGS> {
    let mut old = 0;
    // SAFETY: changes a protection; the pages are title memory or nothing.
    (unsafe { VirtualProtect(at as *const _, len, protection, &mut old) } != 0).then_some(old)
}

/// Whether the kernel pages of `[from, to)` are write-watched and none was
/// written since its last asking. The bits are only read, not cleared: the
/// journal's askings (`crate::watch::writes`) still find what they find.
fn unwritten(from: usize, to: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let (from, to) = (from & !(KERNEL_PAGE - 1), to.next_multiple_of(KERNEL_PAGE));
    let mut found = [0usize; SUBPAGES * 4];
    let mut count = found.len();
    let mut granularity = 0;
    if to - from > found.len() * KERNEL_PAGE {
        return false;
    }
    // SAFETY: the answer has room for `count` entries.
    let failed =
        unsafe { GetWriteWatch(0, from as *const _, to - from, found.as_mut_ptr().cast(), &mut count, &mut granularity) } != 0;
    !failed && count == 0
}

/// A copy in the card's memory offers the piece it would merge now: `gpu`,
/// what the card holds for title memory at `at` on, and `held`, the copy's
/// CPU side for the same bytes. Kept (and its pages protected) if nothing
/// speaks against it ([`Offer`]); the caller has looked at its own reasons
/// first (the copy took the pages and nobody wrote them since).
///
/// Waits while another thread resolves a page of the range, so that
/// `held` is compared as the merge left it.
pub(crate) fn offer(owner: &Arc<dyn Owner>, at: usize, gpu: &[u8], held: &[u8], after: Option<u64>) -> Offer {
    let end = at + gpu.len();
    if gpu.is_empty() || held.len() != gpu.len() {
        return Offer::No(Why::Nothing);
    }
    crate::watch::install();
    drop(std::mem::take(&mut *GRAVE.lock().unwrap_or_else(|e| e.into_inner())));
    let (first, last) = (at / PAGE, (end - 1) / PAGE);
    let me = Arc::as_ptr(owner).cast::<u8>();
    let mut s = store();
    let waiting = std::time::Instant::now();
    while s.pages.range(first..=last).any(|(_, p)| p.resolving) {
        drop(s);
        if waited_too_long(waiting) {
            return Offer::No(Why::Plain);
        }
        std::thread::yield_now();
        s = store();
    }
    let differs = crate::shadow::first_difference(gpu, held)
        .map(|f| (f, crate::shadow::last_difference(gpu, held).unwrap_or(f)));
    let Some((f, l)) = differs else {
        let ours = s
            .pages
            .range(first..=last)
            .any(|(_, p)| p.pieces.iter().any(|q| q.of(me) && q.at < end && q.end() > at));
        if !ours {
            return Offer::No(Why::Nothing);
        }
        let n = take_out(&mut s, me, at, end, None);
        COUNT[DROPPED].fetch_add(n as u64, Relaxed);
        return Offer::Dropped;
    };
    let (from, to) = (at + f, at + l + 1);
    if s.bytes + (to - from) > limit() {
        COUNT[NOW + 5].fetch_add(1, Relaxed);
        return Offer::No(Why::Full);
    }
    let (lo, hi) = (from / PAGE, (to - 1) / PAGE);
    if s
        .pages
        .range(lo..=hi)
        .any(|(_, p)| p.pieces.iter().any(|q| !q.of(me) && q.at < to && q.end() > from))
    {
        COUNT[NOW + 4].fetch_add(1, Relaxed);
        return Offer::No(Why::Others);
    }
    // The pages are protected first and the write-watch bits read after:
    // a write before the protection shows in its bit, none can come after.
    let mut newly: Vec<(usize, usize)> = Vec::new();
    let undo = |newly: &[(usize, usize)]| {
        for &(a, n) in newly {
            protect(a, n, PAGE_READWRITE);
        }
    };
    for page in lo..=hi {
        let had = s.pages.get(&page).map_or(0, |p| p.mask);
        for (a, n) in runs(page, subpages(page, from, to) & !had) {
            match protect(a, n, PAGE_NOACCESS) {
                Some(PAGE_READWRITE) => newly.push((a, n)),
                Some(other) => {
                    protect(a, n, other);
                    undo(&newly);
                    COUNT[NOW + 3].fetch_add(1, Relaxed);
                    return Offer::No(Why::Plain);
                }
                None => {
                    undo(&newly);
                    COUNT[NOW + 3].fetch_add(1, Relaxed);
                    return Offer::No(Why::Plain);
                }
            }
        }
    }
    if !unwritten(from, to) {
        undo(&newly);
        // Not write-watched at all, or written: told apart for the counts.
        let watched = {
            use windows_sys::Win32::System::Memory::GetWriteWatch;
            let mut found = [0usize; 1];
            let (mut count, mut granularity) = (1, 0);
            let a = from & !(KERNEL_PAGE - 1);
            // SAFETY: room for one entry.
            unsafe { GetWriteWatch(0, a as *const _, KERNEL_PAGE, found.as_mut_ptr().cast(), &mut count, &mut granularity) == 0 }
        };
        let why = if watched { Why::Bits } else { Why::Plain };
        merged_now(why);
        return Offer::No(why);
    }
    // Kept: what this copy had kept of the fetched bytes goes (the pages
    // the new piece goes into stay as they are, protected), the new piece
    // goes in, page by page.
    take_out(&mut s, me, at, end, Some((lo, hi)));
    for page in lo..=hi {
        let (a, b) = (from.max(page * PAGE), to.min((page + 1) * PAGE));
        let entry = s.pages.entry(page).or_insert_with(|| {
            PAGES.fetch_add(1, Relaxed);
            Page::default()
        });
        entry.mask |= subpages(page, from, to);
        entry.pieces.push(Piece {
            owner: owner.clone(),
            at: a,
            bytes: gpu[a - at..b - at].to_vec(),
            after,
        });
    }
    s.bytes += to - from;
    COUNT[KEPT].fetch_add(1, Relaxed);
    COUNT[KEPT_BYTES].fetch_add((to - from) as u64, Relaxed);
    Offer::Kept(from, to)
}

/// Takes what `owner` kept of `[from, to)` out of the store; pages left
/// with nothing get their access back, but those of `keep` (first and
/// last page number), which the caller fills again. Returns the pieces cut
/// or dropped.
fn take_out(s: &mut Store, owner: *const u8, from: usize, to: usize, keep: Option<(usize, usize)>) -> usize {
    let (first, last) = (from / PAGE, (to - 1) / PAGE);
    let mut cut = 0;
    let mut freed = 0;
    let mut empty = Vec::new();
    for (&page, entry) in s.pages.range_mut(first..=last) {
        let mut kept = Vec::with_capacity(entry.pieces.len());
        for q in entry.pieces.drain(..) {
            if !q.of(owner) || q.at >= to || q.end() <= from {
                kept.push(q);
                continue;
            }
            cut += 1;
            freed += q.bytes.len();
            // What lies outside the fetched bytes stays.
            if q.at < from {
                let bytes = q.bytes[..from - q.at].to_vec();
                freed -= bytes.len();
                kept.push(Piece { owner: q.owner.clone(), at: q.at, bytes, after: q.after });
            }
            if q.end() > to {
                let bytes = q.bytes[to - q.at..].to_vec();
                freed -= bytes.len();
                kept.push(Piece { owner: q.owner.clone(), at: to, bytes, after: q.after });
            }
        }
        entry.pieces = kept;
        if entry.pieces.is_empty() && !keep.is_some_and(|(a, b)| (a..=b).contains(&page)) {
            empty.push(page);
        }
    }
    s.bytes -= freed;
    for page in empty {
        if let Some(entry) = s.pages.remove(&page) {
            PAGES.fetch_sub(1, Relaxed);
            for (a, n) in runs(page, entry.mask) {
                protect(a, n, PAGE_READWRITE);
            }
        }
    }
    cut
}

/// Merges the pieces taken from page `page` (its kept protection `mask`):
/// access back, then every piece in order. If the memory is no longer what
/// we protected (decommitted, or its protection changed by someone else)
/// the pieces are dropped. Returns the bytes merged.
fn resolve(page: usize, pieces: Vec<Piece>, mask: u16) -> usize {
    /// Takes the page out of the store when the resolution ends, also by a
    /// panic in a merge: nobody waits for it after that.
    struct Done(usize, usize);
    impl Drop for Done {
        fn drop(&mut self) {
            let mut s = store();
            if s.pages.remove(&self.0).is_some() {
                PAGES.fetch_sub(1, Relaxed);
            }
            s.bytes = s.bytes.saturating_sub(self.1);
            if s.recent_set.insert(self.0) {
                s.recent.push_back(self.0);
                if s.recent.len() > RECENT
                    && let Some(old) = s.recent.pop_front()
                {
                    s.recent_set.remove(&old);
                }
                RECENT_PAGES.store(s.recent.len(), Relaxed);
            }
        }
    }
    let _busy = busy();
    let bytes: usize = pieces.iter().map(|p| p.bytes.len()).sum();
    let _done = Done(page, bytes);
    let mut ours = true;
    for (a, n) in runs(page, mask) {
        match protect(a, n, PAGE_READWRITE) {
            Some(PAGE_NOACCESS) => {}
            Some(other) => {
                protect(a, n, other);
                ours = false;
            }
            None => ours = false,
        }
    }
    if ours {
        for p in &pieces {
            p.owner.merge(p.at, &p.bytes, p.after);
        }
    } else {
        COUNT[LOST].fetch_add(1, Relaxed);
    }
    for p in pieces {
        if Arc::strong_count(&p.owner) == 1 {
            GRAVE.lock().unwrap_or_else(|e| e.into_inner()).push(p.owner);
        }
    }
    if ours { bytes } else { 0 }
}

/// Takes the pages of `[first, last]` (64 KiB page numbers) that nobody is
/// resolving; whether another thread is resolving one of them.
fn take(first: usize, last: usize) -> (Vec<(usize, Vec<Piece>, u16)>, bool) {
    let mut s = store();
    let mut taken = Vec::new();
    let mut busy = false;
    for (&page, entry) in s.pages.range_mut(first..=last) {
        if entry.resolving {
            busy = true;
            continue;
        }
        entry.resolving = true;
        taken.push((page, std::mem::take(&mut entry.pieces), entry.mask));
    }
    (taken, busy)
}

/// Before our own read or write of title memory `[from, to)`: every page
/// of it that holds kept pieces is resolved (merged, access back), waiting
/// for a page another thread is resolving. Nothing if none is kept, or if
/// this thread is resolving or merging itself ([`busy`]).
pub(crate) fn need(from: usize, to: usize) {
    if PAGES.load(Relaxed) == 0 || from >= to || is_busy() {
        return;
    }
    let (first, last) = (from / PAGE, (to - 1) / PAGE);
    let waiting = std::time::Instant::now();
    loop {
        let (taken, busy) = take(first, last);
        if !taken.is_empty() {
            let started = std::time::Instant::now();
            let n = taken.len();
            let bytes: usize = taken.into_iter().map(|(page, pieces, mask)| resolve(page, pieces, mask)).sum();
            COUNT[NEEDS].fetch_add(n as u64, Relaxed);
            COUNT[NEED_BYTES].fetch_add(bytes as u64, Relaxed);
            COUNT[NEED_US].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        }
        if !busy || waited_too_long(waiting) {
            return;
        }
        std::thread::yield_now();
    }
}

/// Whether a wait that began at `since` for another thread's resolution
/// has gone on past [`WAIT_LIMIT`]; counted, and the waiter gives up.
fn waited_too_long(since: std::time::Instant) -> bool {
    let long = since.elapsed() > WAIT_LIMIT;
    if long {
        STUCK.fetch_add(1, Relaxed);
    }
    long
}

/// Waits given up since the start ([`waited_too_long`]).
static STUCK: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The address this thread's last fault was let go on as resolved by
    /// another thread ([`fault`]): the same fault again is not ours.
    static LET_GO: Cell<usize> = const { Cell::new(0) };
}

/// From the vectored handler: a read (or with `write` a write) of `at`
/// faulted. True if it was a page of ours, now resolved (the access may go
/// on); false for anything else. Memory work only: protections, copies,
/// our locks. Never a wait without end: a page another thread resolves is
/// waited for [`WAIT_LIMIT`] at most, and a fault that comes back at the
/// same address after it was let go as resolved is not ours.
pub(crate) fn fault(at: usize, write: bool) -> bool {
    if (PAGES.load(Relaxed) == 0 && RECENT_PAGES.load(Relaxed) == 0) || is_busy() {
        return false;
    }
    let page = at / PAGE;
    let waiting = std::time::Instant::now();
    loop {
        let mut s = store();
        match s.pages.get_mut(&page) {
            Some(entry) if !entry.resolving => {
                entry.resolving = true;
                let (pieces, mask) = (std::mem::take(&mut entry.pieces), entry.mask);
                drop(s);
                let started = std::time::Instant::now();
                let bytes = resolve(page, pieces, mask);
                COUNT[FAULTS].fetch_add(1, Relaxed);
                COUNT[FAULT_BYTES].fetch_add(bytes as u64, Relaxed);
                COUNT[FAULT_US].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
                return true;
            }
            Some(_) => {
                drop(s);
                if waited_too_long(waiting) {
                    return false;
                }
                std::thread::yield_now();
            }
            None => {
                // Another thread resolved it between the fault and now: the
                // access goes on if the page allows it now; once (the same
                // fault again is not ours, and would come back for good).
                let lately = s.recent_set.contains(&page);
                drop(s);
                if !lately || LET_GO.with(|l| l.replace(at)) == at {
                    LET_GO.with(|l| l.set(0));
                    return false;
                }
                return accessible(at, write);
            }
        }
    }
}

/// Whether `at` is committed memory that can be read (`write`: written)
/// now.
fn accessible(at: usize, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_WRITECOPY,
        VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    let writable = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    n != 0
        && info.State == MEM_COMMIT
        && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0
        && info.Protect != 0
        && (!write || info.Protect & writable != 0)
}

/// What is kept now: pages and bytes.
#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn kept() -> (usize, usize) {
    let s = store();
    (s.pages.len(), s.bytes)
}

/// With the frame line (diagnostics builds): what was kept and resolved
/// per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let c: Vec<f64> = COUNT.iter().map(|c| c.swap(0, Relaxed) as f64 / frames).collect();
    let (pages, bytes) = kept();
    tracing::info!(
        "copy-back kept for later per frame (VICTORY_LAZY_BACK=1): {:.0} pieces {:.2} MB kept; merged on landing: {:.1} written since taken, {:.1} still to go to the card, {:.1} with a write-watch bit set, {:.1} not plain watched memory, {:.1} beside another copy's piece, {:.1} over the limit; \
         resolved by a touch (fault): {:.1} pages {:.2} MB {:.2} ms; by our own readers: {:.1} pages {:.2} MB {:.2} ms; {:.2} pages dropped (memory changed), {:.1} pieces dropped (fetched again unchanged); kept now: {pages} pages {:.1} MB; waits for another thread's resolution given up since the start: {}",
        c[KEPT],
        c[KEPT_BYTES] / 1e6,
        c[NOW],
        c[NOW + 1],
        c[NOW + 2],
        c[NOW + 3],
        c[NOW + 4],
        c[NOW + 5],
        c[FAULTS],
        c[FAULT_BYTES] / 1e6,
        c[FAULT_US] / 1e3,
        c[NEEDS],
        c[NEED_BYTES] / 1e6,
        c[NEED_US] / 1e3,
        c[LOST],
        c[DROPPED],
        bytes as f64 / 1e6,
        STUCK.load(Relaxed)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, ResetWriteWatch, VirtualAlloc, VirtualFree,
        VirtualQuery,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    const LEN: usize = 4 * PAGE;

    /// A copy's CPU side, merged into as `Shadow::card_merge` merges.
    struct Fake {
        start: usize,
        held: Mutex<Vec<u8>>,
        merges: AtomicUsize,
    }

    impl Owner for Fake {
        fn merge(&self, at: usize, gpu: &[u8], _after: Option<u64>) {
            self.merges.fetch_add(1, Relaxed);
            let mut held = self.held.lock().unwrap();
            let off = at - self.start;
            for (i, &g) in gpu.iter().enumerate() {
                // SAFETY: title memory of the test, accessible now.
                let t = unsafe { ((at + i) as *const u8).read_volatile() };
                let o = held[off + i];
                let merged = if g != o { g } else { t };
                // SAFETY: as above.
                unsafe { ((at + i) as *mut u8).write_volatile(merged) };
                held[off + i] = merged;
            }
        }
    }

    /// Write-watched title memory with a pattern, its bits clear, and a
    /// copy that holds it.
    fn title() -> (usize, Arc<Fake>) {
        // SAFETY: a fresh write-watched allocation (freed by `free`).
        let p = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(p, 0);
        let pattern: Vec<u8> = (0..LEN).map(|i| (i * 7 + 3) as u8).collect();
        // SAFETY: our pages.
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), p as *mut u8, LEN);
            assert_eq!(ResetWriteWatch(p as *const _, LEN), 0);
        }
        (p, Arc::new(Fake { start: p, held: Mutex::new(pattern), merges: AtomicUsize::new(0) }))
    }

    fn free(p: usize) {
        need(p, p + LEN);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    fn protection(at: usize) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer.
        assert_ne!(unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) }, 0);
        info.Protect
    }

    /// The card's side: the copy's bytes with `changes` (offset, value;
    /// each a change).
    fn card(fake: &Fake, at: usize, len: usize, changes: &[(usize, u8)]) -> (Vec<u8>, Vec<u8>) {
        let held = fake.held.lock().unwrap()[at - fake.start..at - fake.start + len].to_vec();
        let mut gpu = held.clone();
        for &(i, v) in changes {
            assert_ne!(gpu[i], v, "the test's change at {i} changes nothing");
            gpu[i] = v;
        }
        (gpu, held)
    }

    fn offer_of(fake: &Arc<Fake>, at: usize, gpu: &[u8], held: &[u8]) -> Offer {
        let owner: Arc<dyn Owner> = fake.clone();
        offer(&owner, at, gpu, held, None)
    }

    fn read_on_another_thread(at: usize) -> u8 {
        // SAFETY: title memory of the test; a kept page faults into the
        // handler, which resolves it.
        std::thread::spawn(move || unsafe { (at as *const u8).read_volatile() }).join().unwrap()
    }

    /// A piece is kept: its kernel page is no-access, the title's bytes
    /// are as before; a read from another thread faults, is resolved, and
    /// sees the card's byte; the page is plain read-write again.
    #[test]
    fn a_kept_piece_is_merged_by_a_read_from_another_thread() {
        let (p, fake) = title();
        let at = p + PAGE;
        let (gpu, held) = card(&fake, at, PAGE, &[(100, 0xaa), (5000, 0x5b)]);
        assert_eq!(offer_of(&fake, at, &gpu, &held), Offer::Kept(at + 100, at + 5001));
        assert_eq!(protection(at), PAGE_NOACCESS);
        assert_eq!(protection(at + 4096), PAGE_NOACCESS);
        assert_eq!(protection(at + 8192), PAGE_READWRITE, "only the kernel pages of the changed bytes");
        assert_eq!(fake.merges.load(Relaxed), 0);
        assert_eq!(read_on_another_thread(at + 5000), 0x5b);
        assert_eq!(protection(at), PAGE_READWRITE);
        assert_eq!(protection(at + 4096), PAGE_READWRITE);
        // SAFETY: readable again.
        assert_eq!(unsafe { ((at + 100) as *const u8).read() }, 0xaa);
        assert_eq!(fake.held.lock().unwrap()[PAGE + 100], 0xaa, "the copy's side took the merge");
        assert_eq!(fake.merges.load(Relaxed), 1);
        free(p);
    }

    /// A title write to a kept page merges first, then lands.
    #[test]
    fn a_title_write_to_a_kept_page_merges_first() {
        let (p, fake) = title();
        let at = p + 2 * PAGE;
        let (gpu, held) = card(&fake, at, PAGE, &[(10, 0x11), (11, 0x22)]);
        assert!(matches!(offer_of(&fake, at, &gpu, &held), Offer::Kept(..)));
        // The title writes byte 11 (the card's too) and byte 12.
        std::thread::spawn(move || {
            // SAFETY: title memory of the test.
            unsafe {
                ((at + 11) as *mut u8).write_volatile(0x33);
                ((at + 12) as *mut u8).write_volatile(0x44);
            }
        })
        .join()
        .unwrap();
        // SAFETY: readable again.
        let now: Vec<u8> = (10..13).map(|i| unsafe { ((at + i) as *const u8).read() }).collect();
        assert_eq!(now, [0x11, 0x33, 0x44], "the card's byte merged, the title's writes after it");
        free(p);
    }

    /// A page the title wrote since its bits were cleared merges at once
    /// (no protection is left on it).
    #[test]
    fn a_page_written_before_landing_is_not_kept() {
        let (p, fake) = title();
        let at = p;
        // SAFETY: our pages.
        unsafe { ((at + 9000) as *mut u8).write_volatile(1) };
        let (gpu, held) = card(&fake, at, PAGE, &[(9001, 0x99)]);
        assert_eq!(offer_of(&fake, at, &gpu, &held), Offer::No(Why::Bits));
        assert_eq!(protection(at + 9000), PAGE_READWRITE);
        // A write to another kernel page of the same 64 KiB page does not
        // keep this one from being kept.
        let (gpu, held) = card(&fake, at, PAGE, &[(100, 0x99)]);
        assert!(matches!(offer_of(&fake, at, &gpu, &held), Offer::Kept(..)));
        free(p);
    }

    /// Memory that is not write-watched is merged at once.
    #[test]
    fn memory_not_watched_is_not_kept() {
        let mut plain = vec![0u8; 3 * PAGE];
        let at = (plain.as_mut_ptr() as usize).next_multiple_of(PAGE);
        let fake = Arc::new(Fake { start: at, held: Mutex::new(vec![0; PAGE]), merges: AtomicUsize::new(0) });
        let (gpu, held) = card(&fake, at, PAGE, &[(1, 1)]);
        assert_eq!(offer_of(&fake, at, &gpu, &held), Offer::No(Why::Plain));
        assert_eq!(protection(at), PAGE_READWRITE);
    }

    /// Our own reader asks first: the page is resolved without a fault.
    #[test]
    fn need_resolves_without_a_fault() {
        let (p, fake) = title();
        let at = p + 3 * PAGE;
        let (gpu, held) = card(&fake, at, PAGE, &[(PAGE - 1, 0x77)]);
        assert!(matches!(offer_of(&fake, at, &gpu, &held), Offer::Kept(..)));
        let faults = COUNT[FAULTS].load(Relaxed);
        need(at + PAGE - 8, at + PAGE);
        assert_eq!(COUNT[FAULTS].load(Relaxed), faults);
        assert_eq!(protection(at + PAGE - 1), PAGE_READWRITE);
        // SAFETY: readable again.
        assert_eq!(unsafe { ((at + PAGE - 1) as *const u8).read() }, 0x77);
        assert_eq!(kept().0, 0);
        free(p);
    }

    /// The card writes the same piece again before anything touched it:
    /// the second fetch, compared with the same CPU side, holds both
    /// writes and replaces the first; a third that differs nowhere (the
    /// card put the bytes back) drops what was kept.
    #[test]
    fn repeated_card_writes_add_up() {
        let (p, fake) = title();
        let at = p + PAGE;
        let (gpu1, held) = card(&fake, at, PAGE, &[(200, 0x01)]);
        assert!(matches!(offer_of(&fake, at, &gpu1, &held), Offer::Kept(..)));
        // The card's side still has the first write; the second adds one.
        let mut gpu2 = gpu1.clone();
        gpu2[30_000] = 0x02;
        assert_eq!(offer_of(&fake, at, &gpu2, &held), Offer::Kept(at + 200, at + 30_001));
        assert_eq!(kept(), (1, 30_001 - 200), "the first piece was replaced");
        assert_eq!(protection(at + 30_000), PAGE_NOACCESS);
        assert_eq!(read_on_another_thread(at + 200), 0x01);
        // SAFETY: readable again.
        assert_eq!(unsafe { ((at + 30_000) as *const u8).read() }, 0x02);
        assert_eq!(fake.merges.load(Relaxed), 1);
        // The merge's own write set the page's write-watch bit: until an
        // asking clears it (the next refresh), a piece there merges at once.
        let (gpu3, held3) = card(&fake, at, PAGE, &[(300, 0x03)]);
        assert_eq!(offer_of(&fake, at, &gpu3, &held3), Offer::No(Why::Bits));
        // SAFETY: our pages; as the refresh's asking does.
        assert_eq!(unsafe { ResetWriteWatch(at as *const _, PAGE) }, 0);
        // Kept again, then fetched with nothing different: dropped, and
        // the title keeps what it had.
        assert!(matches!(offer_of(&fake, at, &gpu3, &held3), Offer::Kept(..)));
        assert_eq!(offer_of(&fake, at, &held3, &held3), Offer::Dropped);
        assert_eq!(kept().0, 0);
        assert_eq!(protection(at + 300), PAGE_READWRITE);
        // SAFETY: our pages.
        assert_eq!(unsafe { ((at + 300) as *const u8).read() }, held3[300]);
        free(p);
    }

    /// A piece of another copy over the same bytes: the second merges at
    /// once (its caller resolves the first before it).
    #[test]
    fn another_copys_piece_is_not_kept_over() {
        let (p, fake) = title();
        let other = Arc::new(Fake { start: p, held: Mutex::new(fake.held.lock().unwrap().clone()), merges: AtomicUsize::new(0) });
        let at = p;
        let (gpu, held) = card(&fake, at, PAGE, &[(64, 0x10)]);
        assert!(matches!(offer_of(&fake, at, &gpu, &held), Offer::Kept(..)));
        let (gpu, held) = card(&other, at, PAGE, &[(64, 0x20)]);
        assert_eq!(offer_of(&other, at, &gpu, &held), Offer::No(Why::Others));
        need(at, at + PAGE);
        // SAFETY: readable again.
        assert_eq!(unsafe { ((at + 64) as *const u8).read() }, 0x10);
        free(p);
    }
}
