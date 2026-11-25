//! `VICTORY_TEXTURE_BACK=1`: small textures that compute shaders write go
//! back to title memory once the list that wrote them has run.
//!
//! The console's processor reads graphics memory as it is: a title may
//! let a compute shader write a small texture (a time step, a reduction's
//! result) and read the texels with the processor a frame later. Here the
//! shaders write the texture's host copy only, and title memory keeps what
//! it had (the shallow water simulation then reads a time step of zero and
//! never integrates: boats sink into the pit their hulls push into the
//! water). A texture bound for unordered access in a list is noted; when
//! the list closes, its level 0 (slice 0) is copied to a read-back buffer;
//! once the list has run, the texels are tiled into the texture's title
//! memory in the console's layout.
//!
//! shortcut: level 0 and slice 0 only, textures of at most
//! `VICTORY_TEXTURE_BACK_MAX` bytes (64 KiB) of title memory, no XOR
//! swizzle modes, no depth textures; texels a title reads before the list
//! lands are the old ones, as without this.

use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::HostList;
use crate::object::This;

/// A texture's texels on their way back: the read-back buffer, the title
/// memory, its layout, row pitch, rows and depth of the copied level.
pub(crate) struct Back {
    buffer: ID3D12Resource,
    memory: usize,
    layout: victory_formats::texture::Layout,
    pitch: u32,
    rows: u32,
    depth: u32,
}

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_TEXTURE_BACK").is_some_and(|v| v == "1");
        if on {
            tracing::info!(most = most(), "VICTORY_TEXTURE_BACK: small textures compute shaders write go back to title memory");
        }
        on
    })
}

fn most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_BACK_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64 << 10)
    })
}

/// At list close: records the copy of each noted texture's level 0 into a
/// read-back buffer.
pub(crate) fn record(list: &HostList) {
    for resource in list.take_texture_backs_noted() {
        if let Some(back) = one(list, resource as This) {
            list.push_texture_back(back);
        }
    }
}

fn one(list: &HostList, resource: This) -> Option<Back> {
    const DEPTH_STENCIL: i32 = 0x2;
    const COPY_SOURCE: u32 = 0x800;
    let texture = crate::resource::host_resource(resource)?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if d.Flags.0 & DEPTH_STENCIL != 0 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        return None;
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    if memory.is_null() {
        return None;
    }
    let layout = crate::resource::texture_layout(&desc)?.planes.first()?.layout.clone();
    if layout.desc.swizzle.xor() || layout.size > most() {
        return None;
    }
    let host = crate::host::host()?;
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut total) = (0u32, 0u64);
    // SAFETY: outputs for one subresource.
    unsafe {
        host.device.GetCopyableFootprints(&d, 0, 1, 0, Some(&mut footprint), Some(&mut rows), None, Some(&mut total));
    }
    if total == 0 || rows == 0 {
        return None;
    }
    let buffer = host.read_back_buffer(total)?;
    crate::list::ensure_state(list, resource, COPY_SOURCE);
    // SAFETY: borrowed references for the call; neither copy location
    // outlives the call, so no reference is taken or released.
    unsafe {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
    }
    COPIED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(Back {
        buffer,
        memory: memory as usize,
        layout,
        pitch: footprint.Footprint.RowPitch,
        rows,
        depth: footprint.Footprint.Depth.max(1),
    })
}

static COPIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LANDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// After the list has run: the texels into title memory.
pub(crate) fn land(backs: &[Back]) {
    for b in backs {
        let mut data = std::ptr::null_mut();
        let len = b.pitch as usize * b.rows as usize * b.depth as usize;
        // SAFETY: our read-back buffer, written by the list that has run;
        // the texture's title memory holds `layout.size` bytes (it was made
        // with this layout), still the texture's while the list lives.
        unsafe {
            if b.buffer.Map(0, None, Some(&mut data)).is_err() || data.is_null() {
                continue;
            }
            let linear = std::slice::from_raw_parts(data.cast::<u8>().cast_const(), len);
            let tiled = std::slice::from_raw_parts_mut(b.memory as *mut u8, b.layout.size as usize);
            let pitch = b.pitch as usize;
            if let Err(e) = b.layout.tile(tiled, 0, 0, linear, pitch, pitch * b.rows as usize) {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| tracing::warn!(?e, "VICTORY_TEXTURE_BACK: texels not tiled into title memory"));
            } else {
                let n = LANDED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!(
                        landed = n,
                        copied = COPIED.load(std::sync::atomic::Ordering::Relaxed),
                        "VICTORY_TEXTURE_BACK: textures written by shaders put back into title memory"
                    );
                }
            }
            b.buffer.Unmap(0, None);
        }
    }
}
