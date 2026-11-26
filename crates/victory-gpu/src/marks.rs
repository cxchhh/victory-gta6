//! View marks: which slots of the view heap the frame's shaders read, as
//! the shaders themselves note them (`victory_shader::emit::SLOT_MARKS`).
//!
//! The title's view heap names every texture it may draw with (up to some
//! 720,000 slots; 20,000 to 30,000 textures the title fills), and a host
//! copy of a texture was made, kept and uploaded for every view the heap
//! held. On a 12 GB card the copies filled the budget, and the textures of
//! what is on screen waited behind ones nothing drew: in one frame of run
//! take2 the tables bound 442 textures while 20,180 copies were alive and
//! 10,992 textures waited for room; 13% of the frame's pixels had a
//! G-buffer colour of exactly 0 (their texture was one of those waiting).
//! A console reads a texture's memory where it lies and pays nothing for a
//! view it does not sample. Here the shaders say which views they read,
//! and the host serves those first and lets the others go.
//!
//! Every shader that picks a view from the heap stores 1 at dword
//! `slot + 1` of one buffer on the card (bound as the root UAV
//! `u<SLOT_MARKS>`). At each present the host copies that buffer into a
//! read-back buffer and clears it (in its own list for the present), and
//! once the copy has run it notes, for every marked slot, the number of
//! the present it was marked before.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;

/// The marks buffer on the card and a buffer of zeros to clear it with.
struct Card {
    marks: ID3D12Resource,
    zero: ID3D12Resource,
    gpu: u64,
    bytes: u64,
}

// SAFETY: D3D12 resources may be used from any thread.
unsafe impl Send for Card {}
// SAFETY: as above; nothing in `Card` changes after it is made.
unsafe impl Sync for Card {}

static CARD: OnceLock<Option<Card>> = OnceLock::new();

fn card() -> Option<&'static Card> {
    CARD.get_or_init(|| {
        let limit = victory_shader::emit::view_heap_limit();
        if limit == 0 || !victory_shader::emit::view_marks() {
            return None;
        }
        let host = crate::host::host()?;
        // A dword per slot up to the null slot `limit`, and dword 0.
        let slots = limit + 1;
        let bytes = (u64::from(slots) + 1) * 4;
        let marks = host.uav_buffer(bytes)?;
        // Committed on the default heap without `CREATE_NOT_ZEROED`: zeros.
        let zero = host.uav_buffer(bytes)?;
        crate::host::name(&marks, || "view marks".into());
        crate::host::name(&zero, || "view marks: zeros".into());
        // SAFETY: a live buffer.
        let gpu = unsafe { marks.GetGPUVirtualAddress() };
        LAST.get_or_init(|| (0..slots).map(|_| AtomicU32::new(0)).collect());
        tracing::info!(
            slots,
            kb = bytes >> 10,
            "view marks: shaders note the views of the heap they read (VICTORY_VIEW_MARKS=0 turns it off)"
        );
        Some(Card { marks, zero, gpu, bytes })
    })
    .as_ref()
}

/// The address to bind as the root UAV `u<SLOT_MARKS>` of a shader that
/// has it. `None` only if the buffer could not be made: such a shader must
/// not run then (it would write through a null address).
pub(crate) fn address() -> Option<u64> {
    card().map(|c| c.gpu)
}

/// Per slot, the number of the present before which it was last marked
/// (0: never).
static LAST: OnceLock<Vec<AtomicU32>> = OnceLock::new();
/// Presents whose marks were recorded for reading, and the newest one read.
static PRESENTS: AtomicU32 = AtomicU32::new(0);
static READ: AtomicU32 = AtomicU32::new(0);
/// Of the newest present read: slots marked.
static MARKED: AtomicU64 = AtomicU64::new(0);

/// A read-back buffer and where its copy is: the submission that copies
/// into it (`FREE`: none; `RECORDED`: recorded, not sent yet) and the
/// present it is of.
struct Back {
    buffer: ID3D12Resource,
    submission: u64,
    present: u32,
}

// SAFETY: as for `Card`.
unsafe impl Send for Back {}

const FREE: u64 = u64::MAX;
const RECORDED: u64 = 0;

static BACKS: Mutex<Vec<Back>> = Mutex::new(Vec::new());

/// A copy of the marks recorded by [`record`], to hand to [`sent`].
pub(crate) struct Pending(usize);

/// At a present, into the host's own list for it (`Host::present_now`):
/// the marks of the frame are copied out for reading and cleared. Reads
/// the copies whose submission has finished first. `None` with marks off,
/// or with three copies still in flight (the marks then add up into the
/// next frame's).
pub(crate) fn record(host: &crate::host::Host, list: &ID3D12GraphicsCommandList) -> Option<Pending> {
    let card = card()?;
    collect();
    let mut backs = BACKS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match backs.iter().position(|b| b.submission == FREE) {
        Some(i) => i,
        None if backs.len() < 3 => {
            let buffer = host.read_back_buffer(card.bytes)?;
            crate::host::name(&buffer, || "view marks: read back".into());
            backs.push(Back { buffer, submission: FREE, present: 0 });
            backs.len() - 1
        }
        None => return None,
    };
    // SAFETY: an open list; the buffers live as long as the process. The
    // marks buffer is in the common state between submissions (a buffer
    // decays to it): its first copy promotes it to a copy source.
    unsafe {
        list.CopyBufferRegion(&backs[i].buffer, 0, &card.marks, 0, card.bytes);
        list.ResourceBarrier(&[crate::host::transition(
            &card.marks,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        list.CopyBufferRegion(&card.marks, 0, &card.zero, 0, card.bytes);
    }
    backs[i].submission = RECORDED;
    backs[i].present = PRESENTS.fetch_add(1, Relaxed) + 1;
    Some(Pending(i))
}

/// The list [`record`] recorded into went out as submission `submission`
/// (one already waited for, too).
pub(crate) fn sent(pending: Option<Pending>, submission: u64) {
    let Some(Pending(i)) = pending else { return };
    let mut backs = BACKS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(b) = backs.get_mut(i) {
        b.submission = submission.max(1);
    }
}

/// Reads every copy whose submission has finished.
fn collect() {
    let Some(card) = CARD.get().and_then(Option::as_ref) else { return };
    let finished = crate::host::finished();
    let mut backs = BACKS.lock().unwrap_or_else(|e| e.into_inner());
    let mut ready: Vec<usize> = (0..backs.len())
        .filter(|&i| backs[i].submission != FREE && backs[i].submission != RECORDED && backs[i].submission <= finished)
        .collect();
    ready.sort_by_key(|&i| backs[i].present);
    for i in ready {
        read(card, &backs[i]);
        backs[i].submission = FREE;
    }
}

fn read(card: &Card, back: &Back) {
    let Some(last) = LAST.get() else { return };
    let mut data = std::ptr::null_mut();
    let range = D3D12_RANGE { Begin: 0, End: card.bytes as usize };
    // SAFETY: a read-back buffer whose copy has finished; mapped for its
    // whole size and unmapped below, nothing written.
    if unsafe { back.buffer.Map(0, Some(&range), Some(&mut data)) }.is_err() || data.is_null() {
        return;
    }
    // SAFETY: `bytes` bytes mapped, dword aligned.
    let words = unsafe { std::slice::from_raw_parts(data.cast::<u32>(), card.bytes as usize / 4) };
    let now = back.present;
    let mut marked = 0u64;
    for (slot, &w) in words[1..].iter().enumerate() {
        if w != 0
            && let Some(l) = last.get(slot)
        {
            l.store(now, Relaxed);
            marked += 1;
        }
    }
    // SAFETY: mapped above.
    unsafe { back.buffer.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 })) };
    READ.fetch_max(now, Relaxed);
    MARKED.store(marked, Relaxed);
    log(last, now);
}

/// Every five seconds: slots marked in the newest frame read, and in the
/// last 60 presents.
fn log(last: &[AtomicU32], now: u32) {
    static AT: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let mut at = AT.lock().unwrap_or_else(|e| e.into_inner());
    if at.is_some_and(|t| t.elapsed().as_secs() < 5) {
        return;
    }
    *at = Some(std::time::Instant::now());
    let window = now.saturating_sub(60);
    let recent = last.iter().filter(|l| l.load(Relaxed) > window).count();
    let ever = last.iter().filter(|l| l.load(Relaxed) != 0).count();
    tracing::info!(
        present = now,
        frame = MARKED.load(Relaxed),
        last_60 = recent,
        ever,
        evicted = crate::draw::VIEW_MARKS_EVICTED.load(Relaxed),
        back = crate::draw::VIEW_MARKS_BACK.load(Relaxed),
        cbv_not_in_place = crate::draw::CBV_NOT_IN_PLACE.load(Relaxed),
        "view marks: slots the shaders read; texture views made null for room, and made again when read"
    );
}

/// The newest present whose marks were read (0: none yet).
pub(crate) fn now() -> u32 {
    READ.load(Relaxed)
}

/// Whether slot `slot` was marked in the last `presents` presents read.
pub(crate) fn recent(slot: u32, presents: u32) -> bool {
    let Some(last) = LAST.get() else { return false };
    let l = last.get(slot as usize).map_or(0, |l| l.load(Relaxed));
    l != 0 && now().saturating_sub(l) < presents
}

/// Whether marks are being read at all (a shader marked, and a copy of
/// them was read).
pub(crate) fn on() -> bool {
    now() != 0
}

