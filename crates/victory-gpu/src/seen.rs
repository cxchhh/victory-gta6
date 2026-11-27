//! Read feedback (`VICTORY_READ_FEEDBACK=1`, `victory_shader::emit::SLOT_READS`):
//! which view-heap slots the card read lately through an index a shader
//! computed. A translated shader that picks a texture from the resource
//! heap stores the frame's stamp at the slot's dword of a buffer in the
//! PC's memory, which the host reads as it is (a stamp read late or early
//! by a frame is no matter here). Slots read through tables are known to
//! the host already (`HostHeap::tabled`).
//!
//! What it is for: the host copy of a texture a slot shows is kept for as
//! long as the slot shows it, and the bindless heap shows nearly every
//! texture the title has mapped, so the card's memory filled with textures
//! not drawn for minutes while new ones waited for room (the texture
//! queue's "no room, new textures wait", black and late surfaces in a new
//! area). With the stamps, the slots not read for a while give their
//! texture up (`HostHeap::texture_slots_evict`) and get it back when a
//! shader reads them again.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use victory_shader::emit::SLOT_STRIDES_COUNT;

struct Reads {
    /// In the card's memory: what shaders store into (`gpu`).
    card: ID3D12Resource,
    /// In the PC's memory: the copy the host reads (`stamps`), made from
    /// `card` at the end of every other frame ([`copy_back`]).
    back: ID3D12Resource,
    stamps: *const AtomicU32,
    gpu: u64,
}

// SAFETY: the mapping lives as long as the process; the stamps are read
// as atomics.
unsafe impl Send for Reads {}
// SAFETY: as above.
unsafe impl Sync for Reads {}

static READS: OnceLock<Option<Reads>> = OnceLock::new();

/// Whether shaders store read feedback.
pub(crate) fn on() -> bool {
    victory_shader::emit::read_feedback()
}

fn reads() -> Option<&'static Reads> {
    READS
        .get_or_init(|| {
            use windows::Win32::Graphics::Direct3D12::*;
            use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
            let host = crate::host::host()?;
            let bytes = u64::from(SLOT_STRIDES_COUNT) * 4;
            let desc = |flags| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let make = |kind, flags, state| {
                let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
                let mut made: Option<ID3D12Resource> = None;
                // SAFETY: plain creation of a buffer.
                unsafe {
                    host.device
                        .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc(flags), state, None, &mut made)
                        .ok()?;
                }
                made
            };
            // (Zeroed by the system: a committed resource's memory is.)
            let card = make(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON)?;
            let back = make(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)?;
            crate::host::name(&card, || "read feedback: the frame each heap slot was last read in".into());
            crate::host::name(&back, || "read feedback: the host's copy".into());
            let mut data = std::ptr::null_mut();
            // SAFETY: a readback buffer, mapped for the life of the process.
            unsafe { back.Map(0, None, Some(&mut data)).ok()? };
            // SAFETY: a live buffer.
            let gpu = unsafe { card.GetGPUVirtualAddress() };
            tracing::info!(
                "read feedback (VICTORY_READ_FEEDBACK={}): shaders note the heap slots they pick textures from, {} MB in the card's memory, copied back every other frame",
                victory_shader::emit::read_feedback_mode(),
                bytes >> 20
            );
            Some(Reads { card, back, stamps: data.cast::<AtomicU32>().cast_const(), gpu })
        })
        .as_ref()
}

/// At the end of a frame, on the list that presents it: every other frame,
/// the card's stamps are copied into the host's copy. (The stamps of draws
/// still running are caught by the next copy; a stamp read late by two
/// frames is no matter.)
pub(crate) fn copy_back(list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList, frame: u64) {
    if !on() || frame % 2 != 0 {
        return;
    }
    let Some(r) = reads() else { return };
    // SAFETY: an open list; both buffers live as long as the process. The
    // card's buffer is in the common state between lists (buffers decay
    // to it) and moves to the copy source implicitly.
    unsafe { r.list_copy(list) };
}

impl Reads {
    unsafe fn list_copy(&self, list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList) {
        // SAFETY: as the caller says.
        unsafe { list.CopyResource(&self.back, &self.card) };
    }
}

/// The GPU address of the stamps, for the root UAV `SLOT_READS`; 0 without
/// them (the store then goes nowhere: the root view is left unset only if
/// this is 0, which `draw.rs` does not bind).
pub(crate) fn gpu() -> u64 {
    if !on() {
        return 0;
    }
    reads().map_or(0, |r| r.gpu)
}

/// The stamp of frame `frame`, as shaders store it (never 0: 0 is "never
/// read").
pub(crate) fn stamp_of(frame: u64) -> u32 {
    (frame as u32).wrapping_add(1).max(1)
}

/// The stamp of the frame being drawn.
pub(crate) fn stamp() -> u32 {
    stamp_of(crate::frame::current())
}

/// The stamp slot `index` was last read in by a computed index (0: never).
pub(crate) fn last_read(index: u32) -> u32 {
    if index >= SLOT_STRIDES_COUNT {
        return 0;
    }
    match reads() {
        // SAFETY: inside the mapping.
        Some(r) => unsafe { (*r.stamps.add(index as usize)).load(Ordering::Relaxed) },
        None => 0,
    }
}

/// Frames from stamp `then` to stamp `now` (u32 stamps wrap).
pub(crate) fn frames_since(now: u32, then: u32) -> u32 {
    now.wrapping_sub(then)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamps_are_never_zero_and_wrap() {
        assert_eq!(stamp_of(0), 1);
        assert_eq!(stamp_of(u64::from(u32::MAX)), 1);
        assert_eq!(frames_since(5, 3), 2);
        assert_eq!(frames_since(1, u32::MAX), 2);
    }
}
