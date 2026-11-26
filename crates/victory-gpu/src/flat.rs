//! One mirror of title memory in the card's memory (`VICTORY_FLAT`; off by
//! default), instead of a host copy per title buffer ([`crate::shadow`]).
//!
//! Every title buffer had a host copy of its own, made on first use and
//! brought up to date range by range before each submission: some 4,400
//! copies and 4.2 GB, and some 40,000 refreshes a frame (120-150 MB,
//! 50-60 ms) of ranges that overlap from one copy to the next. Here the
//! title's GPU address space is covered by reserved buffers of 2 GiB
//! ([`window_bits`]), one per 2 GiB of addresses used ("windows"): a title
//! address is the same offset in its window, whichever buffer it belongs
//! to.
//!
//! `VICTORY_FLAT=census` changes nothing: it counts, with the frame line,
//! what such a mirror would hold and send (the distinct 64 KiB pages the
//! refreshes and the declared shader writes name, the 4 GiB slots they lie
//! in, the pages the CPU and the GPU both write in one submission, the
//! views on the copies of their own bytes).
//!
//! `VICTORY_FLAT=1`, the mirror, stage one: title memory to the card only.
//! - A window is made on the first ask in its 2 GiB, and kept in a table
//!   indexed by `address >> 31`, read without a lock ([`Space::window`]).
//!   Its pages get tiles of a heap of ours ([`crate::tiles::Pool`]) on
//!   first use, at the ask ([`Space::find`]); the `UpdateTileMappings` of
//!   all the pages given since are made on the queue at the next
//!   submission, ahead of its lists ([`Space::upload`]). Tiles are never
//!   given back. When the pool's budget (`VICTORY_FLAT_MB`, 1536: run
//!   trace2 had some 2.3 GB of the card free) is used up, no page is given
//!   any more and asks of pages without tiles go to the copies, as before.
//! - Only asks that read go to the mirror: constant buffers, shader
//!   resource views, vertex and index buffers, root views that are not
//!   unordered access (`crate::shadow::find_read`). A page that shaders
//!   are declared to write, or that an unordered access view, a root
//!   unordered access view or a copy names, leaves the mirror for good
//!   ([`gpu_wrote`]): readers and writers of it then share one copy, as
//!   before. Slots of view heaps that show the mirror over such a page are
//!   made again ([`evictions`], `HostHeap::mirror`). Also left to the
//!   copies: asks that cross a window's line, views off their copy's elements
//!   (`crate::shadow::exact`), pages where a title buffer does not start
//!   on 256 bytes (there a copy's offsets and the mirror's differ modulo
//!   256, and constant buffers' rows and raw views' bias are computed from
//!   either), and shader-side raw pointers (`crate::memory`).
//! - The card only ever sees the mirror change through copies of ours on
//!   its queue: at each submission the pages its refreshes name that the
//!   title wrote since they were last sent (the write watch's journal, or
//!   the guards, as for the copies; memory neither watches every time) are
//!   copied, whole pages joined into runs, into upload memory of that
//!   submission's own, and from there into the windows by a list executed
//!   ahead of the submission's lists. Upload memory is used again only once
//!   the host's fence has passed the submission. So a submission still on
//!   the card reads the bytes sent before it, whatever the title writes
//!   meanwhile (a constant ring the CPU wrote in place under a submission
//!   in the air made black frames, `VICTORY_CBV_SPLIT_IN_PLACE`).
//! - What the runtime maps or unmaps, or commits, is sent again at the next
//!   refresh ([`remapped`]). Physical pages mapped at several addresses
//!   are mirrored once per address.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

use crate::shadow::Shadow;

/// The page counted and mirrored: the card's tile and the console's page.
pub(crate) const PAGE: u64 = 64 << 10;

/// The bits of a title address below a 4 GiB slot, the most a raw view's
/// 32-bit offsets reach (what the census counts in).
pub(crate) const SLOT_BITS: u32 = 32;

/// The bits of a title address below a window of the mirror: 2 GiB, or
/// `VICTORY_FLAT_WINDOW_GB` (1, 2 or 4). Under 4 GiB a raw view's offsets
/// reach; 2 GiB keeps the offsets of copies into a window under 2^31,
/// which the software adapter needs (it removed the device for any copy
/// ending at 2 GiB or past it).
fn window_bits() -> u32 {
    static BITS: OnceLock<u32> = OnceLock::new();
    *BITS.get_or_init(|| match std::env::var("VICTORY_FLAT_WINDOW_GB").ok().and_then(|v| v.trim().parse::<u32>().ok()) {
        Some(1) => 30,
        Some(4) => 32,
        _ => 31,
    })
}

/// `VICTORY_FLAT`: 0 (unset) off; 1 `census`; 2 `1`, the mirror.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_FLAT").as_deref().map(str::trim) {
        Ok("census") => 1,
        Ok("1") => 2,
        _ => 0,
    })
}

/// Whether the census is on (`VICTORY_FLAT=census`).
pub(crate) fn census() -> bool {
    mode() == 1
}

/// Whether the mirror is on (`VICTORY_FLAT=1`): not with
/// `VICTORY_INPLACE`, whose tile mappings into heaps opened on title memory
/// removed the device.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if mode() != 2 {
            return false;
        }
        if crate::inplace::level() != 0 {
            tracing::warn!("VICTORY_FLAT=1 is refused with VICTORY_INPLACE set: title memory is copied per buffer as before");
            return false;
        }
        true
    })
}

/// A page's state bits ([`Page::state`]): it has tiles (or they are
/// given at the next submission); shaders may write it, so it is the
/// copies'; a refresh named it since the last frame line.
const MAPPED: u8 = 1;
const GPU: u8 = 2;
const USED: u8 = 4;

/// [`Page::layout`]: the note is there; every title buffer on the page
/// starts on 256 bytes; no buffer starts or ends inside the page, and the
/// low bits are the end of the buffer an address on it is in (0: none).
const KNOWN: u64 = 1 << 63;
const ALIGNED: u64 = 1 << 62;
const EVEN: u64 = 1 << 61;
const END_BITS: u64 = (1 << 48) - 1;

/// The cell memory outside buffers is cut on for an ask without a length,
/// as the copies' windows are (`crate::shadow`, 4 MiB).
const CELL: u64 = 4 << 20;

/// One 64 KiB page of a window.
#[derive(Default)]
struct Page {
    state: AtomicU8,
    /// The write watch's asking after which the page was last sent (0:
    /// never, or to be sent again).
    sent: AtomicU64,
    /// What the title buffers on it look like (`crate::shadow::page_layout`;
    /// 0: not known, asked again).
    layout: AtomicU64,
}

/// A window's title addresses and their reserved buffer.
pub(crate) struct Window {
    shadow: Arc<Shadow>,
    resource: ID3D12Resource,
    /// The first title address.
    base: u64,
    pages: Box<[Page]>,
    /// Held while pages are given tiles.
    giving: Mutex<()>,
}

impl Window {
    fn page(&self, address: u64) -> usize {
        ((address - self.base) / PAGE) as usize
    }
}

/// Pages given tiles at an ask, mapped on the queue at the next
/// submission: the window, its first page, the heap and its runs of tiles
/// (first, tiles; as many tiles as pages in all).
struct Given {
    resource: ID3D12Resource,
    first: u32,
    count: u32,
    heap: ID3D12Heap,
    runs: Vec<(u32, u32)>,
}

/// A submission's upload: its list and upload memory, used again once the
/// host's fence has passed `fence`.
struct Slot {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    staging: Option<(ID3D12Resource, *mut u8, u64)>,
    fence: u64,
}

/// Since the last frame line (the ones marked so: since the start).
#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum S {
    /// Asks the mirror took.
    Served,
    /// Asks left to the copies: a page shaders may write; a page whose
    /// buffers do not all start on 256 bytes; across a window's line; no
    /// tiles left.
    ToCopiesWritten,
    ToCopiesUnaligned,
    ToCopiesCrossing,
    ToCopiesNoTiles,
    /// Pages sent, their bytes, the runs they went in, and the time it took
    /// the submitting thread (asking, copying, recording; microseconds).
    PagesSent,
    BytesSent,
    Runs,
    SendUs,
    /// Pages refreshes named, and of them pages without tiles (left as
    /// they were).
    PagesNamed,
    PagesWithoutTiles,
    /// `UpdateTileMappings` calls and their microseconds.
    MapCalls,
    MapUs,
    /// Mapped pages that became the copies' (shaders may write them), and
    /// shader writes declared on the mirror itself (lost).
    Evicted,
    WrittenThrough,
    /// Submissions that sent something.
    Sends,
}

const STATS_LEN: usize = S::Sends as usize + 1;

/// The mirror of one device.
pub(crate) struct Space {
    device: ID3D12Device,
    windows: Box<[OnceLock<Option<Arc<Window>>>]>,
    pool: crate::tiles::Pool,
    /// The most tiles the mirror takes, and those it took (since the start).
    budget: u64,
    /// The bytes of a window's buffer: `1 << bits`, or 64 KiB less on a
    /// card whose resources hold no more address bits than that (the
    /// software adapter has 32): the window's last page is then the
    /// copies'.
    window_len: u64,
    /// The bits of a title address below a window.
    bits: u32,
    taken: AtomicU64,
    /// No tile is given any more: the budget or the card ran out (for
    /// good, so that an ask the copies took once is never the mirror's
    /// later, and a slot made on a copy shows what is brought up to date).
    exhausted: AtomicBool,
    given: Mutex<Vec<Given>>,
    ring: Mutex<Vec<Slot>>,
    stats: [AtomicU64; STATS_LEN],
}

// SAFETY: D3D12 objects are free-threaded; the upload mappings are written
// only by the submitting thread (under `ring`'s lock).
unsafe impl Send for Space {}
// SAFETY: as above.
unsafe impl Sync for Space {}
// SAFETY: as above.
unsafe impl Send for Window {}
// SAFETY: as above.
unsafe impl Sync for Window {}

/// Raised each time a page that had tiles leaves the mirror: view heaps
/// make their slots that show a window again (`HostHeap::mirror`).
static EVICTIONS: AtomicU64 = AtomicU64::new(0);

/// See [`EVICTIONS`].
pub(crate) fn evictions() -> u64 {
    EVICTIONS.load(Ordering::Acquire)
}

/// The mirror of the host device, made on first use with the switch on.
fn space() -> Option<&'static Space> {
    if !on() {
        return None;
    }
    SPACE
        .get_or_init(|| {
            let host = crate::host::host()?;
            let mb = std::env::var("VICTORY_FLAT_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1536);
            let space = Space::new(&host.device, (mb << 20) / PAGE);
            tracing::info!(
                made = space.is_some(),
                budget_mb = mb,
                "title memory read by shaders is mirrored in the card's memory (VICTORY_FLAT=1; VICTORY_FLAT_MB, VICTORY_FLAT_WINDOW_GB)"
            );
            space
        })
        .as_ref()
}

static SPACE: OnceLock<Option<Space>> = OnceLock::new();

/// The mirror if it was made already: what only notes, or runs with the
/// runtime's locks held, does not make one.
fn made() -> Option<&'static Space> {
    SPACE.get()?.as_ref()
}

/// The window of the mirror that holds `[address, address + len)` (`len`
/// 0 or 1: an ask without a length, to the end of its buffer), if the
/// mirror takes the ask; `None` sends it to the copies.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if !on() {
        return None;
    }
    space()?.find(address, len)
}

/// Bytes from `address` to the end of what an ask without a length
/// covers (see [`Space::left`]).
pub(crate) fn left(address: u64) -> u64 {
    made().map_or(0, |space| space.left(address))
}

/// Shaders may write `[address, address + len)`: its pages are the
/// copies' from now on.
pub(crate) fn gpu_wrote(address: u64, len: u64) {
    if let Some(space) = space() {
        space.gpu_wrote(address, len);
    }
}

/// Shaders were given a window of the mirror to write: what they write
/// never reaches the title. Counted (and said once), and the pages leave
/// the mirror.
pub(crate) fn written_through(address: u64, len: u64) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            address = format_args!("{address:#x}"),
            len,
            "shaders were given the mirror of title memory to write (VICTORY_FLAT=1): what they write there is lost"
        );
    });
    if let Some(space) = made() {
        space.add(S::WrittenThrough, 1);
        space.gpu_wrote(address, len);
    }
}

/// A title buffer `[start, start + len)` was registered or released: the
/// notes of its pages are asked again. Called with the buffer table's lock
/// held (`crate::shadow::register`).
pub(crate) fn buffers_changed(start: u64, len: u64) {
    if let Some(space) = made() {
        space.each_page(start, len.max(1), |page| page.layout.store(0, Ordering::Release));
    }
}

/// The runtime mapped, unmapped or committed `[at, at + len)`: its pages
/// are sent again when a refresh next names them.
pub(crate) fn remapped(at: usize, len: usize) {
    if let Some(space) = made() {
        space.each_page(at as u64, len.max(1) as u64, |page| page.sent.store(0, Ordering::Release));
    }
}

/// At submission, after its lists' steps ran: gives the pages given tiles
/// since the last submission their tiles on `queue`, and sends the pages of
/// `ranges` (title address, bytes: the refreshes of windows) the title
/// wrote since, ahead of the lists. `next`: the value the host's `fence`
/// is signalled with behind this submission's lists.
pub(crate) fn submit(queue: &ID3D12CommandQueue, fence: &ID3D12Fence, next: u64, ranges: Vec<(usize, usize)>) {
    if let Some(space) = made() {
        space.upload(queue, fence, next, &ranges);
    }
}

impl Space {
    /// The mirror on `device`, taking at most `budget` tiles. `None` if the
    /// card does not read zeros from tiles without memory (tiled resources
    /// tier 2) or cannot place a window in one resource.
    pub(crate) fn new(device: &ID3D12Device, budget: u64) -> Option<Space> {
        Space::with_bits(device, budget, window_bits())
    }

    /// [`Space::new`] with windows of `1 << bits` bytes (30 to 32).
    pub(crate) fn with_bits(device: &ID3D12Device, budget: u64, bits: u32) -> Option<Space> {
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        let mut va = D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT::default();
        // SAFETY: plain queries into structs of their own sizes.
        unsafe {
            device
                .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, (&raw mut options).cast(), size_of_val(&options) as u32)
                .ok()?;
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT,
                    (&raw mut va).cast(),
                    size_of_val(&va) as u32,
                )
                .ok()?;
        }
        let tier = options.TiledResourcesTier.0;
        if tier < 2 || va.MaxGPUVirtualAddressBitsPerResource < bits {
            tracing::warn!(
                tier,
                bits_per_resource = va.MaxGPUVirtualAddressBitsPerResource,
                "the card cannot hold the mirror of title memory (tiled resources tier 2 and resources of a window's size needed): copies as before"
            );
            return None;
        }
        // Title addresses are user-mode addresses: 47 bits.
        let slots = 1usize << (47 - bits);
        Some(Space {
            device: device.clone(),
            windows: (0..slots).map(|_| OnceLock::new()).collect(),
            pool: crate::tiles::Pool::for_buffers(),
            budget,
            window_len: if va.MaxGPUVirtualAddressBitsPerResource > bits { 1 << bits } else { (1 << bits) - PAGE },
            bits,
            taken: AtomicU64::new(0),
            exhausted: AtomicBool::new(false),
            given: Mutex::new(Vec::new()),
            ring: Mutex::new(Vec::new()),
            stats: [const { AtomicU64::new(0) }; STATS_LEN],
        })
    }

    fn add(&self, s: S, n: u64) {
        self.stats[s as usize].fetch_add(n, Ordering::Relaxed);
    }

    /// The window of `address`, made on first use (`None` if it could not
    /// be made, for good).
    pub(crate) fn window(&self, address: u64) -> Option<&Arc<Window>> {
        let slot = self.windows.get((address >> self.bits) as usize)?;
        slot.get_or_init(|| self.make_window(address >> self.bits << self.bits)).as_ref()
    }

    /// The window of `address` if it was made.
    fn window_made(&self, address: u64) -> Option<&Arc<Window>> {
        self.windows.get((address >> self.bits) as usize)?.get()?.as_ref()
    }

    fn make_window(&self, base: u64) -> Option<Arc<Window>> {
        let len = self.window_len;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: len,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: a plain creation; the out pointer is ours.
        let result = unsafe { self.device.CreateReservedResource(&raw const desc, D3D12_RESOURCE_STATE_COMMON, None, &mut made) };
        let resource = match (result, made) {
            (Ok(()), Some(resource)) => resource,
            (result, _) => {
                #[cfg(test)]
                eprintln!("flat: no window at {base:#x}: {result:?}");
                tracing::warn!(base = format_args!("{base:#x}"), ?result, "no window of the mirror of title memory: its asks go to the copies");
                return None;
            }
        };
        tracing::info!(base = format_args!("{base:#x}"), "a window of the mirror of title memory");
        // Every page of the window's addresses has its notes, also one
        // past `len`.
        let pages = ((1u64 << self.bits) / PAGE) as usize;
        Some(Arc::new(Window {
            shadow: Arc::new(Shadow::flat_window(resource.clone(), base, len)),
            resource,
            base,
            pages: (0..pages).map(|_| Page::default()).collect(),
            giving: Mutex::new(()),
        }))
    }

    /// Calls `f` for each page of `[address, address + len)` in windows
    /// made already.
    fn each_page(&self, address: u64, len: u64, mut f: impl FnMut(&Page)) {
        let end = address.saturating_add(len);
        let mut at = address / PAGE * PAGE;
        while at < end {
            let next_window = (at >> self.bits).saturating_add(1) << self.bits;
            match self.window_made(at) {
                Some(window) => {
                    let last = end.min(next_window);
                    for page in &window.pages[window.page(at)..window.page(last - 1) + 1] {
                        f(page);
                    }
                }
                None if (at >> self.bits) as usize >= self.windows.len() => return,
                None => {}
            }
            at = next_window;
        }
    }

    /// The note of a page's title buffers, asked for if not known.
    fn layout(&self, window: &Window, page: usize) -> u64 {
        let note = &window.pages[page].layout;
        let known = note.load(Ordering::Acquire);
        if known & KNOWN != 0 {
            return known;
        }
        let from = window.base + page as u64 * PAGE;
        let mut value = 0;
        crate::shadow::page_layout(from, from + PAGE, |aligned, end| {
            value = KNOWN | if aligned { ALIGNED } else { 0 } | end.map_or(0, |end| EVEN | (end & END_BITS));
            note.store(value, Ordering::Release);
        });
        value
    }

    /// Bytes from `address` to the end of the title buffer that holds it,
    /// as its copy says ([`Shadow::left`]); outside buffers, to the end of
    /// its 4 MiB cell, as a window outside buffers says; never past the
    /// window.
    pub(crate) fn left(&self, address: u64) -> u64 {
        let window_end = (address >> self.bits << self.bits) + self.window_len;
        let Some(window) = self.window(address) else {
            return 0;
        };
        let note = self.layout(window, window.page(address));
        let end = if note & EVEN != 0 { note & END_BITS } else { crate::shadow::buffer_end(address).unwrap_or(0) };
        let end = if end > address { end } else { (address / CELL + 1) * CELL };
        end.min(window_end).saturating_sub(address)
    }

    /// See [`find`].
    pub(crate) fn find(&self, address: u64, len: u64) -> Option<Arc<Shadow>> {
        if address == 0 {
            return None;
        }
        let window = self.window(address)?;
        if address >= window.base + self.window_len {
            self.add(S::ToCopiesCrossing, 1);
            return None;
        }
        let first = window.page(address);
        // A copy would start on 256 bytes too (see `ALIGNED`): the copy
        // `lookup` finds holds the address, so it lies on this page.
        if self.layout(window, first) & ALIGNED == 0 {
            self.add(S::ToCopiesUnaligned, 1);
            return None;
        }
        let end = if len <= 1 { address + self.left(address) } else { address.checked_add(len)? };
        if end > window.base + self.window_len {
            self.add(S::ToCopiesCrossing, 1);
            return None;
        }
        let last = window.page(end - 1);
        if window.pages[first..=last].iter().any(|p| p.state.load(Ordering::Acquire) & GPU != 0) {
            self.add(S::ToCopiesWritten, 1);
            return None;
        }
        if !self.give(window, first, last) {
            self.add(S::ToCopiesNoTiles, 1);
            return None;
        }
        self.add(S::Served, 1);
        Some(window.shadow.clone())
    }

    /// Gives pages `first..=last` of `window` tiles where they have none;
    /// false if the tiles ran out (the pages given meanwhile keep theirs).
    fn give(&self, window: &Window, first: usize, last: usize) -> bool {
        let mapped = |p: &Page| p.state.load(Ordering::Acquire) & MAPPED != 0;
        if window.pages[first..=last].iter().all(mapped) {
            return true;
        }
        let _giving = window.giving.lock().unwrap_or_else(|e| e.into_inner());
        let mut page = first;
        while page <= last {
            if mapped(&window.pages[page]) {
                page += 1;
                continue;
            }
            let mut end = page + 1;
            while end <= last && end - page < crate::tiles::HEAP_TILES as usize && !mapped(&window.pages[end]) {
                end += 1;
            }
            let count = (end - page) as u32;
            if self.exhausted.load(Ordering::Relaxed) {
                return false;
            }
            let taken = self.taken.load(Ordering::Relaxed);
            let tiles = if taken + u64::from(count) <= self.budget { self.pool.take(&self.device, count) } else { None };
            let Some((heap, _, runs)) = tiles else {
                if !self.exhausted.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        taken_mb = (taken * PAGE) >> 20,
                        "the mirror of title memory has no tiles left (VICTORY_FLAT_MB): asks of pages without tiles go to the copies from now on"
                    );
                }
                return false;
            };
            self.taken.fetch_add(u64::from(count), Ordering::Relaxed);
            // Noted for the queue first, then marked: an ask that sees the
            // pages mapped is submitted behind their mapping.
            self.given.lock().unwrap_or_else(|e| e.into_inner()).push(Given {
                resource: window.resource.clone(),
                first: page as u32,
                count,
                heap,
                runs,
            });
            for p in &window.pages[page..end] {
                p.sent.store(0, Ordering::Relaxed);
                p.state.fetch_or(MAPPED, Ordering::Release);
            }
            page = end;
        }
        true
    }

    /// See [`gpu_wrote`].
    pub(crate) fn gpu_wrote(&self, address: u64, len: u64) {
        let end = address.saturating_add(len.max(1));
        let mut at = address;
        let mut evicted = 0u64;
        while at < end {
            let next_window = (at >> self.bits).saturating_add(1) << self.bits;
            let Some(window) = self.window(at) else {
                return;
            };
            let to = end.min(next_window);
            for page in &window.pages[window.page(at)..=window.page(to - 1)] {
                let state = page.state.load(Ordering::Relaxed);
                if state & GPU == 0 && page.state.fetch_or(GPU, Ordering::AcqRel) & (GPU | MAPPED) == MAPPED {
                    evicted += 1;
                }
            }
            at = to;
        }
        if evicted > 0 {
            self.add(S::Evicted, evicted);
            EVICTIONS.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// The windows' buffers, as raw interface pointers.
    fn window_raws(&self) -> Vec<usize> {
        use windows::core::Interface;
        self.windows.iter().filter_map(|w| w.get()?.as_ref()).map(|w| w.resource.as_raw() as usize).collect()
    }

    /// See [`submit`]. Returns the pages sent.
    pub(crate) fn upload(&self, queue: &ID3D12CommandQueue, fence: &ID3D12Fence, next: u64, ranges: &[(usize, usize)]) -> u64 {
        let started = std::time::Instant::now();
        self.map_given(queue);
        // The pages named, in order, none touching another, and cut at the
        // windows' lines.
        let mut named: Vec<(u64, u64)> = Vec::new();
        for (a, e) in page_runs(ranges.iter().map(|&(a, n)| (a as u64, (a + n) as u64)).collect()) {
            let mut at = a;
            while at < e {
                let line = ((at * PAGE) >> self.bits).saturating_add(1) << self.bits;
                let to = e.min(line / PAGE);
                named.push((at, to));
                at = to;
            }
        }
        if named.is_empty() {
            return 0;
        }
        self.add(S::PagesNamed, run_pages(&named));
        // One asking for runs close together (`crate::watch::ask_ahead`).
        let bytes: Vec<(usize, usize)> = named.iter().map(|&(a, e)| ((a * PAGE) as usize, (e * PAGE) as usize)).collect();
        crate::watch::ask_ahead(&bytes);
        // The pages to send, as runs of title addresses, each in one window.
        let mut send: Vec<(Arc<Window>, u64, u64)> = Vec::new();
        for &(a, e) in &named {
            let Some(window) = self.window_made(a * PAGE).cloned() else {
                self.add(S::PagesWithoutTiles, e - a);
                continue;
            };
            let answer = crate::watch::writes((a * PAGE) as usize, (e * PAGE) as usize, true);
            let mut written = answer.as_ref().map_or(&[][..], |(_, w)| &w[..]).iter().peekable();
            let now = answer.as_ref().map_or(0, |(now, _)| *now);
            for number in a..e {
                let last_write = match written.peek() {
                    Some(&&(p, w)) if p as u64 == number => {
                        written.next();
                        w
                    }
                    _ => 0,
                };
                let page = &window.pages[window.page(number * PAGE)];
                let state = page.state.fetch_or(USED, Ordering::Relaxed);
                if state & MAPPED == 0 {
                    self.add(S::PagesWithoutTiles, 1);
                    continue;
                }
                let sent = page.sent.load(Ordering::Acquire);
                // Unwatched memory (`None`): every time, and it stays due.
                if answer.is_some() && sent != 0 && last_write <= sent {
                    continue;
                }
                page.sent.store(now, Ordering::Release);
                let at = number * PAGE;
                match send.last_mut() {
                    Some(last) if Arc::ptr_eq(&last.0, &window) && last.2 == at => last.2 = at + PAGE,
                    _ => send.push((window.clone(), at, at + PAGE)),
                }
            }
        }
        crate::watch::ask_ahead_done();
        if send.is_empty() {
            self.add(S::SendUs, started.elapsed().as_micros() as u64);
            return 0;
        }
        let total: u64 = send.iter().map(|r| r.2 - r.1).sum();
        let sent = self.send(queue, fence, next, &send, total);
        self.add(S::SendUs, started.elapsed().as_micros() as u64);
        if sent {
            self.add(S::PagesSent, total / PAGE);
            self.add(S::BytesSent, total);
            self.add(S::Runs, send.len() as u64);
            self.add(S::Sends, 1);
            total / PAGE
        } else {
            // Not sent: due again next time.
            for (window, a, e) in &send {
                for page in &window.pages[window.page(*a)..window.page(*e - 1) + 1] {
                    page.sent.store(0, Ordering::Release);
                }
            }
            0
        }
    }

    /// The `UpdateTileMappings` of the pages given tiles since the last
    /// submission, on `queue`: one call per window and heap.
    fn map_given(&self, queue: &ID3D12CommandQueue) {
        let given = std::mem::take(&mut *self.given.lock().unwrap_or_else(|e| e.into_inner()));
        if given.is_empty() {
            return;
        }
        use windows::core::Interface;
        let mut groups: Vec<(usize, usize, Vec<&Given>)> = Vec::new();
        for g in &given {
            let key = (g.resource.as_raw() as usize, g.heap.as_raw() as usize);
            match groups.iter_mut().find(|(r, h, _)| (*r, *h) == key) {
                Some(group) => group.2.push(g),
                None => groups.push((key.0, key.1, vec![g])),
            }
        }
        for (_, _, group) in groups {
            let began = std::time::Instant::now();
            let coordinates: Vec<D3D12_TILED_RESOURCE_COORDINATE> = group
                .iter()
                .map(|g| D3D12_TILED_RESOURCE_COORDINATE { X: g.first, Y: 0, Z: 0, Subresource: 0 })
                .collect();
            let sizes: Vec<D3D12_TILE_REGION_SIZE> =
                group.iter().map(|g| D3D12_TILE_REGION_SIZE { NumTiles: g.count, ..Default::default() }).collect();
            let runs: Vec<(u32, u32)> = group.iter().flat_map(|g| g.runs.iter().copied()).collect();
            let flags = vec![D3D12_TILE_RANGE_FLAG_NONE; runs.len()];
            let starts: Vec<u32> = runs.iter().map(|r| r.0).collect();
            let counts: Vec<u32> = runs.iter().map(|r| r.1).collect();
            // SAFETY: a live reserved buffer and a live heap of ours; the
            // regions' tiles add up to the ranges' (each region's runs were
            // taken for its tiles, in order), arrays that live through the
            // call.
            unsafe {
                queue.UpdateTileMappings(
                    &group[0].resource,
                    coordinates.len() as u32,
                    Some(coordinates.as_ptr()),
                    Some(sizes.as_ptr()),
                    &group[0].heap,
                    runs.len() as u32,
                    Some(flags.as_ptr()),
                    Some(starts.as_ptr()),
                    Some(counts.as_ptr()),
                    D3D12_TILE_MAPPING_FLAG_NONE,
                );
            }
            self.add(S::MapCalls, 1);
            self.add(S::MapUs, began.elapsed().as_micros() as u64);
        }
    }

    /// Copies the runs (title addresses) into upload memory of a slot the
    /// card is done with, and executes a list that copies them into the
    /// windows. False if no slot could be had.
    fn send(&self, queue: &ID3D12CommandQueue, fence: &ID3D12Fence, next: u64, runs: &[(Arc<Window>, u64, u64)], total: u64) -> bool {
        let mut ring = self.ring.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query.
        let done = unsafe { fence.GetCompletedValue() };
        let at = match ring.iter().position(|slot| slot.fence <= done) {
            Some(at) => at,
            None => {
                let Some(slot) = self.make_slot() else {
                    return false;
                };
                ring.push(slot);
                ring.len() - 1
            }
        };
        let slot = &mut ring[at];
        // Upload memory of at least `total` bytes; a large one is not kept
        // for small sends.
        let fits = slot.staging.as_ref().is_some_and(|s| s.2 >= total && (s.2 <= 64 << 20 || total > 16 << 20));
        if !fits {
            slot.staging = None;
            let size = total.next_power_of_two().max(4 << 20);
            let Some(staging) = self.upload_buffer(size) else {
                return false;
            };
            slot.staging = Some(staging);
        }
        let Some((staging, data, _)) = slot.staging.clone() else {
            return false;
        };
        let mut offset = 0u64;
        for (_, a, e) in runs {
            // SAFETY: inside the upload mapping (`total` bytes at least).
            let to = unsafe { data.add(offset as usize) };
            crate::host::copy_mapped(to, *a as *const u8, (*e - *a) as usize, false);
            offset += *e - *a;
        }
        // SAFETY: the slot's list ran (the fence passed it); live buffers,
        // the runs lie in them.
        let recorded = unsafe {
            (|| -> windows::core::Result<()> {
                slot.allocator.Reset()?;
                slot.list.Reset(&slot.allocator, None)?;
                let mut offset = 0u64;
                for (window, a, e) in runs {
                    slot.list.CopyBufferRegion(&window.resource, *a - window.base, &staging, offset, *e - *a);
                    offset += *e - *a;
                }
                slot.list.Close()?;
                queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&slot.list)?)]);
                Ok(())
            })()
        };
        if let Err(e) = recorded {
            tracing::warn!(error = %e, "the mirror of title memory could not be sent to the card");
            return false;
        }
        slot.fence = next;
        true
    }

    fn make_slot(&self) -> Option<Slot> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
            let list: ID3D12GraphicsCommandList =
                self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
            list.Close().ok()?;
            Some(Slot { allocator, list, staging: None, fence: 0 })
        }
    }

    /// Upload memory of `size` bytes, mapped.
    fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8, u64)> {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and a map of the whole buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut buffer)
                .and_then(|()| buffer.as_ref().ok_or_else(windows::core::Error::empty)?.Map(0, None, Some(&mut data)))
        };
        match made {
            Ok(()) => {
                crate::host::name(buffer.as_ref()?, || "upload memory of the mirror of title memory".into());
                Some((buffer?, data.cast(), size))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload memory for the mirror of title memory");
                None
            }
        }
    }

    /// Since the last call: the stats (see [`S`]), taken.
    fn take_stats(&self) -> [u64; STATS_LEN] {
        std::array::from_fn(|i| self.stats[i].swap(0, Ordering::Relaxed))
    }

    /// Of the windows made: pages with tiles, pages that are the copies',
    /// pages named since the last call (their mark taken).
    fn pages(&self) -> [u64; 3] {
        let mut out = [0u64; 3];
        for window in self.windows.iter().filter_map(|w| w.get()?.as_ref()) {
            for page in window.pages.iter() {
                let state = page.state.load(Ordering::Relaxed);
                if state == 0 {
                    continue;
                }
                out[0] += u64::from(state & MAPPED != 0);
                out[1] += u64::from(state & GPU != 0);
                if state & USED != 0 {
                    out[2] += 1;
                    page.state.fetch_and(!USED, Ordering::Relaxed);
                }
            }
        }
        out
    }
}

/// The buffers of the host mirror's windows, as raw interface pointers
/// (`HostHeap::mirror` makes the slots that show one again after
/// [`evictions`] moved).
pub(crate) fn window_raws() -> Vec<usize> {
    made().map_or_else(Vec::new, Space::window_raws)
}

/// The mirror's line, with the frame-rate line (`frames` since the last).
fn mirror_line(frames: f64) {
    let Some(space) = made() else {
        return;
    };
    let s = space.take_stats();
    let [mapped, copies, used] = space.pages();
    let windows = space.windows.iter().filter(|w| w.get().is_some_and(Option::is_some)).count();
    let mb = |bytes: u64| bytes as f64 / f64::from(1 << 20);
    let f = |i: S| s[i as usize] as f64 / frames;
    tracing::info!(
        "flat mirror (VICTORY_FLAT=1) per frame: {:.0} asks taken, to the copies {:.1} written by shaders, {:.1} unaligned, \
         {:.1} across a window, {:.1} without tiles; sent {:.0} pages ({:.1} MB) in {:.0} runs over {:.1} submissions, {:.2} ms; \
         {:.0} pages named, {:.1} without tiles; UpdateTileMappings {:.1} calls {:.0} us; {:.1} pages left to the copies, \
         {:.2} shader writes into the mirror (lost); now: {windows} windows, {:.0} MB with tiles ({:.0} MB of heaps, budget \
         {:.0} MB{}), {:.0} MB the copies', {:.0} MB named since the last line",
        f(S::Served),
        f(S::ToCopiesWritten),
        f(S::ToCopiesUnaligned),
        f(S::ToCopiesCrossing),
        f(S::ToCopiesNoTiles),
        f(S::PagesSent),
        mb(s[S::BytesSent as usize]) / frames,
        f(S::Runs),
        f(S::Sends),
        f(S::SendUs) / 1000.0,
        f(S::PagesNamed),
        f(S::PagesWithoutTiles),
        f(S::MapCalls),
        f(S::MapUs),
        f(S::Evicted),
        f(S::WrittenThrough),
        mb(mapped * PAGE),
        mb(space.pool.heap_bytes.load(Ordering::Relaxed)),
        mb(space.budget * PAGE),
        if space.exhausted.load(Ordering::Relaxed) { ", used up" } else { "" },
        mb(copies * PAGE),
        mb(used * PAGE),
    );
}

/// What the census gathers between two frame lines, and for one
/// submission.
#[derive(Default)]
struct Census {
    /// Title ranges (address, end) refreshed, and declared written by
    /// shaders, since the last line.
    refreshed: Vec<(u64, u64)>,
    written: Vec<(u64, u64)>,
    /// Asks of each kind since the last line, and of them those that cross
    /// a 4 GiB line.
    asks: [u64; 2],
    crossing: u64,
    /// Views on copies of their own bytes (`crate::shadow::exact`) since the
    /// last line, by (address, bytes), and the asks.
    exact: HashSet<(u64, u64)>,
    exact_asks: u64,
    /// The submission being prepared: title ranges its refreshes copied
    /// (the CPU wrote them since the last copy) and ranges its shaders were
    /// declared to write.
    copied: Vec<(u64, u64)>,
    declared: Vec<(u64, u64)>,
    /// Since the last line: submissions, the pages their refreshes copied
    /// (whole 64 KiB pages, what a mirror sends), the bytes those
    /// refreshes copied, and pages both copied and declared written in one
    /// submission ("double dirty").
    submissions: u64,
    copied_pages: u64,
    copied_bytes: u64,
    double_dirty: u64,
    /// Since the start: every page refreshed, and every page declared
    /// written (what a mirror that never lets a page go holds).
    ever_read: HashSet<u64>,
    ever_written: HashSet<u64>,
}

static CENSUS: Mutex<Option<Census>> = Mutex::new(None);

fn with_census(f: impl FnOnce(&mut Census)) {
    let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
    f(census.get_or_insert_with(Census::default));
}

/// `[address, address + len)`, at least one byte, as (address, end).
fn span(address: u64, len: u64) -> (u64, u64) {
    (address, address.saturating_add(len.max(1)))
}

fn crosses(address: u64, end: u64) -> bool {
    address >> SLOT_BITS != (end - 1) >> SLOT_BITS
}

/// Census: a host copy's range is refreshed for a list (`Shadow::refresh`).
pub(crate) fn census_refresh(address: u64, len: u64) {
    if !census() {
        return;
    }
    let (a, e) = span(address, len);
    with_census(|c| {
        c.refreshed.push((a, e));
        c.asks[0] += 1;
        c.crossing += u64::from(crosses(a, e));
    });
}

/// Census: shaders of a list may write a host copy's range
/// (`Shadow::gpu_writes`).
pub(crate) fn census_gpu_writes(address: u64, len: u64) {
    if !census() {
        return;
    }
    let (a, e) = span(address, len);
    with_census(|c| {
        c.written.push((a, e));
        c.asks[1] += 1;
        c.crossing += u64::from(crosses(a, e));
    });
}

/// Census: a buffer view got a copy of its own bytes
/// (`crate::shadow::exact`), which a mirror leaves on the old path.
pub(crate) fn census_exact(address: u64, len: u64) {
    if !census() {
        return;
    }
    with_census(|c| {
        c.exact_asks += 1;
        if c.exact.len() < 1 << 16 {
            c.exact.insert((address, len));
        }
    });
}

/// Census: a refresh copied title memory `[from, to)` into a host copy (the
/// title wrote it since the copy last took it).
pub(crate) fn census_copied(from: usize, to: usize) {
    if !census() || from >= to {
        return;
    }
    with_census(|c| {
        c.copied.push((from as u64, to as u64));
        c.copied_bytes += (to - from) as u64;
    });
}

/// Census: at submission, shaders are declared to write a range
/// (`Shadow::writing`).
pub(crate) fn census_writing(address: usize, len: usize) {
    if !census() || len == 0 {
        return;
    }
    with_census(|c| c.declared.push((address as u64, (address + len) as u64)));
}

/// Census: the submission's refreshes are done and its snapshots taken:
/// what it copied and declared is counted and forgotten.
pub(crate) fn census_submitted() {
    if !census() {
        return;
    }
    with_census(|c| {
        let copied = page_runs(std::mem::take(&mut c.copied));
        let declared = page_runs(std::mem::take(&mut c.declared));
        c.submissions += 1;
        c.copied_pages += run_pages(&copied);
        c.double_dirty += overlap_pages(&copied, &declared);
    });
}

/// The page runs `[first, end)` (page numbers) the ranges touch, in order,
/// none touching another.
pub(crate) fn page_runs(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    for r in &mut ranges {
        *r = (r.0 / PAGE, r.1.div_ceil(PAGE));
    }
    ranges.retain(|r| r.0 < r.1);
    ranges.sort_unstable();
    let mut runs: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (a, e) in ranges {
        match runs.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(e),
            _ => runs.push((a, e)),
        }
    }
    runs
}

fn run_pages(runs: &[(u64, u64)]) -> u64 {
    runs.iter().map(|r| r.1 - r.0).sum()
}

/// Pages in both lists of runs (each in order, none touching another).
fn overlap_pages(a: &[(u64, u64)], b: &[(u64, u64)]) -> u64 {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        let (from, to) = (a[i].0.max(b[j].0), a[i].1.min(b[j].1));
        if from < to {
            n += to - from;
        }
        if a[i].1 < b[j].1 { i += 1 } else { j += 1 }
    }
    n
}

/// The census's line, with the frame-rate line (`frames` since the last).
pub(crate) fn line(frames: f64) {
    if census() {
        census_line(frames);
    }
    if on() {
        mirror_line(frames);
    }
}

fn census_line(frames: f64) {
    let taken = {
        let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(c) = census.as_mut() else {
            return;
        };
        (
            std::mem::take(&mut c.refreshed),
            std::mem::take(&mut c.written),
            std::mem::take(&mut c.asks),
            std::mem::take(&mut c.crossing),
            std::mem::take(&mut c.exact),
            std::mem::take(&mut c.exact_asks),
            [
                std::mem::take(&mut c.submissions),
                std::mem::take(&mut c.copied_pages),
                std::mem::take(&mut c.copied_bytes),
                std::mem::take(&mut c.double_dirty),
            ],
        )
    };
    let (refreshed, written, asks, crossing, exact, exact_asks, [submissions, copied_pages, copied_bytes, double_dirty]) = taken;
    // The merging and the sets outside the lock: the title's threads keep
    // asking meanwhile.
    let read = page_runs(refreshed);
    let wrote = page_runs(written);
    let both = page_runs(read.iter().chain(&wrote).map(|r| (r.0 * PAGE, r.1 * PAGE)).collect());
    let mut slots: Vec<u64> = Vec::new();
    for r in &both {
        for slot in (r.0 * PAGE) >> SLOT_BITS..=((r.1 * PAGE - 1) >> SLOT_BITS) {
            if slots.last() != Some(&slot) {
                slots.push(slot);
            }
        }
    }
    let (ever_read, ever_written, ever_only_read) = {
        let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        let c = census.get_or_insert_with(Census::default);
        for r in &read {
            c.ever_read.extend(r.0..r.1);
        }
        for r in &wrote {
            c.ever_written.extend(r.0..r.1);
        }
        let only = c.ever_read.iter().filter(|p| !c.ever_written.contains(p)).count() as u64;
        (c.ever_read.len() as u64, c.ever_written.len() as u64, only)
    };
    let mb = |pages: u64| (pages * PAGE) as f64 / f64::from(1 << 20);
    let f = |n: u64| n as f64 / frames;
    tracing::info!(
        "flat census (VICTORY_FLAT=census) per frame: {:.0} refreshes over {} distinct 64 KiB pages ({:.0} MB), \
         {:.0} declared shader writes over {} pages ({:.0} MB), {} pages in all ({:.0} MB) in {} 4 GiB slots {slots:x?}, \
         {:.1} asks crossing a 4 GiB line; per frame {:.1} submissions, {:.0} pages a mirror would send ({:.1} MB; the copies \
         took {:.1} MB), {:.1} pages written by the CPU and declared written by shaders in the same submission; views on \
         copies of their own bytes: {:.1} asks, {} distinct; since the start: {} pages read ({:.0} MB), {} declared written \
         ({:.0} MB), {} read and never declared written ({:.0} MB: what FLAT=1 would mirror)",
        f(asks[0]),
        run_pages(&read),
        mb(run_pages(&read)),
        f(asks[1]),
        run_pages(&wrote),
        mb(run_pages(&wrote)),
        run_pages(&both),
        mb(run_pages(&both)),
        slots.len(),
        f(crossing),
        f(submissions),
        f(copied_pages),
        mb(copied_pages) / frames,
        copied_bytes as f64 / f64::from(1 << 20) / frames,
        f(double_dirty),
        f(exact_asks),
        exact.len(),
        ever_read,
        mb(ever_read),
        ever_written,
        mb(ever_written),
        ever_only_read,
        mb(ever_only_read),
    );
}

#[cfg(test)]
mod census_tests {
    use super::*;

    #[test]
    fn ranges_become_page_runs_that_do_not_touch() {
        let runs = page_runs(vec![(PAGE * 3 + 5, PAGE * 4 + 1), (0, 1), (PAGE * 5, PAGE * 6), (PAGE - 1, PAGE + 1)]);
        assert_eq!(runs, vec![(0, 2), (3, 6)]);
        assert_eq!(run_pages(&runs), 5);
    }

    #[test]
    fn overlapping_pages_are_counted_once() {
        let a = vec![(0, 4), (10, 12)];
        let b = vec![(2, 11), (20, 30)];
        assert_eq!(overlap_pages(&a, &b), 2 + 1);
        assert_eq!(overlap_pages(&a, &[]), 0);
    }

    #[test]
    fn an_ask_across_a_4_gib_line_is_seen() {
        let line = 1u64 << SLOT_BITS;
        assert!(crosses(line - 4, line + 4));
        assert!(!crosses(line - 4, line));
        assert!(!crosses(line, line + 1));
    }
}

#[cfg(test)]
mod mirror_tests {
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree};

    use super::*;

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Windows of 1 GiB: the software adapter removes the device for a
    /// copy into a buffer that ends at 2 GiB or past it.
    const BITS: u32 = 30;

    /// A device of its own on the software adapter, a queue, a fence.
    struct Rig {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        fence: ID3D12Fence,
        event: HANDLE,
        value: u64,
    }

    impl Rig {
        fn new() -> Option<Rig> {
            // SAFETY: plain object creation.
            unsafe {
                // (No debug layer: it would hold for every device made in
                // the process after it, the host device of later tests too.)
                let adapter = crate::host::warp_adapter().ok()?;
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                Some(Rig { device, queue, fence, event, value: 0 })
            }
        }

        /// Signals the next value behind what the queue has, and waits for it.
        fn finish(&mut self) {
            self.value += 1;
            // SAFETY: our own queue, fence and event.
            unsafe {
                self.queue.Signal(&self.fence, self.value).unwrap();
                self.fence.SetEventOnCompletion(self.value, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
            }
            self.check();
        }

        /// Stops if the device went, with the messages the device kept.
        fn check(&self) {
            // SAFETY: plain queries.
            unsafe {
                let Err(why) = self.device.GetDeviceRemovedReason() else {
                    return;
                };
                let mut said = Vec::new();
                if let Ok(queue) = windows::core::Interface::cast::<ID3D12InfoQueue>(&self.device) {
                    for i in 0..queue.GetNumStoredMessages() {
                        let mut len = 0;
                        let _ = queue.GetMessage(i, None, &mut len);
                        let mut bytes = vec![0u8; len];
                        let message = bytes.as_mut_ptr().cast::<D3D12_MESSAGE>();
                        if queue.GetMessage(i, Some(message), &mut len).is_ok() {
                            said.push(String::from_utf8_lossy(std::slice::from_raw_parts((*message).pDescription, (*message).DescriptionByteLength)).into_owned());
                        }
                    }
                }
                panic!("the device went ({why:?}): {said:#?}");
            }
        }

        /// The mirror's upload of `ranges`, then a signal: the pages sent.
        fn upload(&mut self, space: &Space, ranges: &[(usize, usize)]) -> u64 {
            let sent = space.upload(&self.queue, &self.fence, self.value + 1, ranges);
            self.check();
            self.finish();
            sent
        }

        fn readback(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                self.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer)
                    .unwrap();
            }
            buffer.unwrap()
        }

        /// Records a list that copies `len` bytes of `from` at `offset` into
        /// a new read-back buffer, and executes it (not waited for).
        fn copy_out(&self, from: &ID3D12Resource, offset: u64, len: u64) -> (ID3D12Resource, ID3D12CommandAllocator, ID3D12GraphicsCommandList) {
            let readback = self.readback(len);
            // SAFETY: plain creation; live buffers, the range lies in both.
            unsafe {
                let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                let list: ID3D12GraphicsCommandList =
                    self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).unwrap();
                list.CopyBufferRegion(&readback, 0, from, offset, len);
                list.Close().unwrap();
                self.queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&list).unwrap())]);
                (readback, allocator, list)
            }
        }

        /// The bytes a read-back buffer holds (after the card is done).
        fn bytes(readback: &ID3D12Resource, len: usize) -> Vec<u8> {
            let mut data = std::ptr::null_mut();
            // SAFETY: a read-back buffer of `len` bytes, the card is done.
            unsafe {
                readback.Map(0, None, Some(&mut data)).unwrap();
                let out = std::slice::from_raw_parts(data.cast::<u8>(), len).to_vec();
                readback.Unmap(0, None);
                out
            }
        }

        /// What `space`'s window shows of `[address, address + len)`.
        fn shown(&mut self, space: &Space, address: u64, len: u64) -> Vec<u8> {
            let window = space.window(address).unwrap().clone();
            let (readback, _a, _l) = self.copy_out(&window.resource, address - window.base, len);
            self.finish();
            Rig::bytes(&readback, len as usize)
        }
    }

    /// Write-watched "title memory" of `len` bytes that crosses a 4 GiB
    /// line in its middle, filled with a pattern; `None` if no such place
    /// was free.
    fn across_a_line(len: usize) -> Option<*mut u8> {
        for k in 2u64..256 {
            let at = (k << BITS) - len as u64 / 2;
            // SAFETY: a plain allocation at a place of our choosing.
            let made = unsafe { VirtualAlloc(at as *const _, len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) };
            if !made.is_null() {
                let bytes = made.cast::<u8>();
                for i in 0..len {
                    // SAFETY: inside the allocation.
                    unsafe { bytes.add(i).write((i as u32).wrapping_mul(2_654_435_761).rotate_right(13) as u8) };
                }
                return Some(bytes);
            }
        }
        None
    }

    fn title(at: *mut u8, offset: usize, len: usize) -> Vec<u8> {
        // SAFETY: inside the test's allocation.
        unsafe { std::slice::from_raw_parts(at.add(offset), len).to_vec() }
    }

    #[test]
    fn two_windows_show_title_memory_and_only_changed_pages_go_again() {
        const LEN: usize = 2 << 20;
        let Some(mut rig) = Rig::new() else {
            eprintln!("flat mirror_tests: skipped, no software adapter");
            return;
        };
        let space = Space::with_bits(&rig.device, 1 << 16, BITS).expect("the software adapter holds the mirror");
        let memory = across_a_line(LEN).expect("room across a 4 GiB line");
        let base = memory as u64;
        let line = ((base >> BITS) + 1) << BITS;
        // The low window may end a page short of the line (`window_len`).
        let low_end = line - (1 << BITS) + space.window_len;
        let (low_len, high_len) = (low_end - base, base + LEN as u64 - line);
        let high_at = (line - base) as usize;
        if low_end < line {
            assert!(space.find(low_end, 16).is_none(), "the page past a short window is the copies'");
            space.take_stats();
        }
        // Across the line: the copies' (as a crossing title buffer is).
        assert!(space.find(base, LEN as u64).is_none());
        assert_eq!(space.take_stats()[S::ToCopiesCrossing as usize], 1);
        // Each side alone: two windows.
        let low = space.find(base, low_len).expect("the low side");
        let high = space.find(line, high_len).expect("the high side");
        assert!(!Arc::ptr_eq(&low, &high));
        assert!(low.is_flat() && high.is_flat());
        assert_eq!(low.offset(low_end - 4), space.window_len - 4);
        assert_eq!(high.offset(line), 0);
        assert_eq!(space.window_raws().len(), 2);
        let all = [(base as usize, low_len as usize), (line as usize, high_len as usize)];
        assert_eq!(rig.upload(&space, &all), (low_len + high_len) / PAGE, "every page the first time");
        assert_eq!(space.take_stats()[S::MapCalls as usize], 2, "one mapping call per window");
        assert_eq!(rig.shown(&space, base, low_len), title(memory, 0, low_len as usize));
        assert_eq!(rig.shown(&space, line, high_len), title(memory, high_at, high_len as usize));
        // Nothing written: nothing sent.
        assert_eq!(rig.upload(&space, &all), 0);
        // Writes into three pages (one on each side, one inside a page),
        // 4 KiB each, of random bytes.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for offset in [0x1000usize, high_at + 0x3_0000, 0x8_8000] {
            for i in 0..4096 {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                // SAFETY: inside the allocation.
                unsafe { memory.add(offset + i).write(seed as u8) };
            }
        }
        assert_eq!(rig.upload(&space, &all), 3, "only the three pages written");
        let stats = space.take_stats();
        assert_eq!(stats[S::PagesSent as usize], 3);
        assert_eq!(stats[S::BytesSent as usize], 3 * PAGE);
        assert_eq!(rig.shown(&space, base, low_len), title(memory, 0, low_len as usize));
        assert_eq!(rig.shown(&space, line, high_len), title(memory, high_at, high_len as usize));
        // A refresh of part of the memory sends only what it names.
        // SAFETY: inside the allocation.
        unsafe {
            memory.add(0x10).write(0xAB);
            memory.add(0x5_0010).write(0xCD);
        }
        assert_eq!(rig.upload(&space, &[(base as usize, 0x100)]), 1);
        assert_eq!(rig.shown(&space, base, 0x100), title(memory, 0, 0x100));
        assert_eq!(rig.upload(&space, &all), 1, "the other written page is sent when it is named");
        assert_eq!(rig.shown(&space, base, low_len), title(memory, 0, low_len as usize));
        // Remapped memory is sent again.
        space.each_page(base, 1, |page| page.sent.store(0, Ordering::Release));
        assert_eq!(rig.upload(&space, &all), 1);
        // SAFETY: our allocation.
        unsafe { VirtualFree(memory.cast(), 0, MEM_RELEASE) };
    }

    #[test]
    fn asks_shaders_write_and_asks_past_the_tiles_go_to_the_copies() {
        let Some(rig) = Rig::new() else {
            eprintln!("flat mirror_tests: skipped, no software adapter");
            return;
        };
        // Four tiles in all.
        let space = Space::with_bits(&rig.device, 4, BITS).expect("the software adapter holds the mirror");
        let len = 8 * PAGE as usize;
        // SAFETY: a plain allocation.
        let memory = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as u64;
        assert_ne!(memory, 0);
        let first = memory.div_ceil(PAGE) * PAGE;
        assert!(space.find(first, 3 * PAGE).is_some());
        // Two more pages: one tile is left.
        assert!(space.find(first + 3 * PAGE, 2 * PAGE).is_none());
        assert_eq!(space.take_stats()[S::ToCopiesNoTiles as usize], 1);
        // For good, even for one page: an ask the copies took stays theirs.
        assert!(space.find(first + 3 * PAGE, 4).is_none());
        // What has tiles is still the mirror's.
        assert!(space.find(first + PAGE, PAGE).is_some());
        // Shaders write a page with tiles: it leaves the mirror, and the
        // view heaps are told.
        let before = evictions();
        space.gpu_wrote(first + PAGE + 16, 4);
        assert_eq!(evictions(), before + 1);
        assert!(space.find(first + PAGE, 4).is_none());
        assert!(space.find(first, 3 * PAGE).is_none());
        assert!(space.find(first, 16).is_some(), "the page beside it stays the mirror's");
        let stats = space.take_stats();
        assert_eq!(stats[S::ToCopiesWritten as usize], 2);
        assert_eq!(stats[S::Evicted as usize], 1);
        // A page without tiles written by shaders: no slot shows it, no
        // eviction to tell.
        space.gpu_wrote(first + 6 * PAGE, 4);
        assert_eq!(evictions(), before + 1);
        // SAFETY: our allocation.
        unsafe { VirtualFree(memory as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn a_submission_on_the_card_reads_the_bytes_sent_before_it() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("flat mirror_tests: skipped, no software adapter");
            return;
        };
        let space = Space::with_bits(&rig.device, 64, BITS).expect("the software adapter holds the mirror");
        let len = 2 * PAGE as usize;
        // SAFETY: a plain allocation.
        let memory = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!memory.is_null());
        let page = (memory as u64).div_ceil(PAGE) * PAGE;
        let at = page as *mut u32;
        // SAFETY: inside the allocation.
        unsafe { at.write(0xAAAA_AAAA) };
        assert!(space.find(page, 256).is_some());
        assert_eq!(rig.upload(&space, &[(page as usize, 256)]), 1);
        let window = space.window(page).unwrap().clone();
        // Submission A: held on the card behind a fence the CPU signals
        // later, then reads the page.
        // SAFETY: plain creation.
        let gate: ID3D12Fence = unsafe { rig.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        // SAFETY: our queue and fence.
        unsafe { rig.queue.Wait(&gate, 1) }.unwrap();
        let (seen_by_a, _a1, _a2) = rig.copy_out(&window.resource, page - window.base, 4);
        rig.value += 1;
        // SAFETY: our queue and fence.
        unsafe { rig.queue.Signal(&rig.fence, rig.value) }.unwrap();
        // The CPU changes the page; submission B sends it and reads it.
        // SAFETY: inside the allocation.
        unsafe { at.write(0xBBBB_BBBB) };
        assert_eq!(space.upload(&rig.queue, &rig.fence, rig.value + 1, &[(page as usize, 256)]), 1);
        let (seen_by_b, _b1, _b2) = rig.copy_out(&window.resource, page - window.base, 4);
        // A has not run yet: the queue waits for the gate.
        // SAFETY: a plain query.
        assert!(unsafe { rig.fence.GetCompletedValue() } < rig.value);
        // SAFETY: our fence.
        unsafe { gate.Signal(1) }.unwrap();
        rig.finish();
        assert_eq!(Rig::bytes(&seen_by_a, 4), 0xAAAA_AAAAu32.to_le_bytes(), "submission A read the bytes sent after it");
        assert_eq!(Rig::bytes(&seen_by_b, 4), 0xBBBB_BBBBu32.to_le_bytes());
        // SAFETY: our allocation.
        unsafe { VirtualFree(memory.cast(), 0, MEM_RELEASE) };
    }

    /// The same refreshes through the copies (`crate::shadow`, on the host
    /// device) and through the mirror: the card sees the same bytes. Only
    /// on the software adapter (`VICTORY_D3D12_WARP=1`, diagnostics).
    #[test]
    fn the_mirror_and_the_copies_show_the_same_bytes() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let software = host
            .adapter
            .as_ref()
            // SAFETY: a plain query.
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!(
                "flat mirror_tests: the differential test is skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1)"
            );
            return;
        }
        const LEN: usize = 512 << 10;
        // SAFETY: a plain allocation.
        let memory = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!memory.is_null());
        let start = memory as u64;
        crate::shadow::register(0xF1A7_0001, start, LEN as u64);
        let copy = crate::shadow::find(start, LEN as u64).expect("a host copy");
        assert!(!copy.is_flat());
        let space = Space::with_bits(&host.device, 1 << 12, BITS).expect("the software adapter holds the mirror");
        // SAFETY: plain creation.
        let fence: ID3D12Fence = unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let mut value = 0u64;
        let mut seed = 0x0123_4567_89AB_CDEFu64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let read = |from: &ID3D12Resource, offset: u64, len: u64| -> Vec<u8> {
            let readback = host.read_back_buffer(len).unwrap();
            let list = host.new_list().unwrap();
            // SAFETY: live buffers, the range lies in both.
            unsafe { list.list.CopyBufferRegion(&readback, 0, from, offset, len) };
            list.close();
            host.execute(&[&list]);
            Rig::bytes(&readback, len as usize)
        };
        for round in 0..6 {
            // The title writes some bytes here and there.
            for _ in 0..40 {
                let at = (next() as usize) % LEN;
                // SAFETY: inside the allocation.
                unsafe { memory.add(at).write(next() as u8) };
            }
            // Some ranges are refreshed, through both.
            let ranges: Vec<(usize, usize)> = (0..5)
                .map(|_| {
                    let at = (next() as usize) % (LEN - 4096);
                    (start as usize + at, 1 + (next() as usize) % (LEN - at).min(96 << 10))
                })
                .collect();
            let list = host.new_list().unwrap();
            for &(a, n) in &ranges {
                copy.refresh(&list, a as u64, n as u64);
                assert!(space.find(a as u64, n as u64).is_some(), "round {round}: the mirror takes the read");
            }
            list.close();
            host.execute(&[&list]);
            space.upload(host.queue(), &fence, value + 1, &ranges);
            value += 1;
            // SAFETY: the host's queue and our fence.
            unsafe { host.queue().Signal(&fence, value) }.unwrap();
            // SAFETY: a plain query.
            while unsafe { fence.GetCompletedValue() } < value {
                std::thread::yield_now();
            }
            let window = space.window(start).unwrap().clone();
            for &(a, n) in &ranges {
                let a = a as u64;
                let old = read(&copy.buffer, copy.offset(a), n as u64);
                let new = read(&window.resource, a - window.base, n as u64);
                assert_eq!(old, new, "round {round}: {a:#x}+{n:#x}");
                assert_eq!(new, title(memory, (a - start) as usize, n), "round {round}: title memory");
            }
        }
        // A range shaders write: the mirror leaves it to the copy, which
        // both then read.
        space.gpu_wrote(start + 0x100, 0x40);
        assert!(space.find(start + 0x100, 4).is_none());
        crate::shadow::unregister(0xF1A7_0001, start);
        drop(copy);
        // SAFETY: our allocation; the copy is gone.
        unsafe { VirtualFree(memory.cast(), 0, MEM_RELEASE) };
    }
}
