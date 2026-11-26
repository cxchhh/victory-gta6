//! Title textures detiled on the card (`VICTORY_GPU_DETILE`, stage 5 of
//! the memory rewrite, `docs/fork-memory-rewrite.md`).
//!
//! Without the switch a texture the title writes itself is detiled on the
//! CPU, whole, on the submitting thread, into its staging buffer, which
//! the lists that sample it copy into the host texture ([`crate::texture`]).
//! With it:
//!
//! * the refresh that finds the memory changed copies the *tiled* bytes as
//!   they are into an upload buffer of the texture's ([`Gpu::take_tiled`]):
//!   a plain copy instead of the detiling's gather;
//! * a list that records the texture's upload copies from a region of a
//!   small pool of card memory ([`region`]), held until the list is reset
//!   (every run of it is done by then), not from the staging buffer;
//! * at each submission of that list, after the texture's refresh, a job
//!   is queued that fills the region: a compute pass of ours detiles the
//!   tiled bytes into it ([`SOURCE`]), with the same address arithmetic as
//!   `victory_formats::texture::Layout::detile` (byte for byte: the unit
//!   tests compare both on WARP and on the card);
//! * the jobs go to the card in a list of ours executed on the same queue
//!   just ahead of the submission's lists ([`before_lists`]).
//!
//! The CPU never writes memory the card may still read: a tiled upload
//! buffer still named by a job not known finished is not written, the
//! refresh takes another ([`Gpu::take_tiled`]); jobs hold what they read
//! until the fence of their submission has passed ([`reclaim`]); a region
//! is given back only when the list it was taken for is reset.
//!
//! `VICTORY_GPU_DETILE=alt`: twenty seconds each way (`crate::host::alternate`,
//! index 6). Off, a texture that has card regions is detiled on the CPU into
//! its staging buffer as before, and the job copies that into the region
//! (so the CPU side is today's; the card does one more copy).
//! `VICTORY_GPU_DETILE_MB`: the card pool (256 unless said); uploads past it
//! take a buffer of their own, released with the list.
//! Layouts the pass cannot address (or sizes past 2 GiB) stay on the CPU path.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use victory_formats::texture::{Dimension, Layout, Swizzle};

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// 0 unset, 1 on, 2 in turn.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_GPU_DETILE").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether textures get what the card path needs (the switch is set at
/// all, and our pass compiled).
pub(crate) fn wanted() -> bool {
    mode() != 0 && pipeline().is_some()
}

/// Whether a refresh now copies the tiled bytes for the card to detile
/// (else it detiles on the CPU).
pub(crate) fn on_card_now() -> bool {
    match mode() {
        0 => false,
        1 => true,
        _ => crate::host::alternate(6, "VICTORY_GPU_DETILE"),
    }
}

/// The detiling pass. One thread a 32-bit word of the linear output; the
/// grid covers each subresource's footprint whole (rows `row_pitch` apart,
/// `rows` of them, its depth), and what lies outside the mip's elements is
/// written as zeros, as the CPU path's zeroed staging buffer holds there.
/// `width` 0: zeros only (levels not the title's, regions with nothing
/// detiled yet).
pub(crate) const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=23, b0), SRV(t0), UAV(u0)"
cbuffer Pass : register(b0) {
    uint out_at;      // the subresource's offset in the region
    uint row_pitch;   // bytes
    uint depth_pitch; // bytes
    uint words;       // row_pitch / 4
    uint width;       // the mip in elements (0: zeros)
    uint height;
    uint depth;
    uint n_log2;      // log2 of the element bytes
    uint linear_mode; // the console layout is linear
    uint mip_at;      // the mip's offset in a slice
    uint slice_size;
    uint first;       // the array slice (3D: 0)
    uint slab;        // depth slices a slice holds (thick: the block's depth)
    uint bw_log2;     // block width, height in elements (log2)
    uint bh_log2;
    uint bz_count;    // log2 of `slab`
    uint blocks_row;  // blocks a row of the mip (linear: bytes a row)
    uint block_log2;  // log2 of the block's bytes
    uint tail_x;      // where the mip starts in the tail block
    uint tail_y;
    uint bits_x;      // the address bit of each coordinate bit, 4 bits each
    uint bits_y;
    uint bits_z;
};
ByteAddressBuffer tiled : register(t0);
RWByteAddressBuffer linear : register(u0);

uint spread(uint c, uint bits, uint count) {
    uint r = 0;
    for (uint i = 0; i < count; i++) {
        r |= ((c >> i) & 1u) << ((bits >> (4u * i)) & 15u);
    }
    return r;
}

uint element_at(uint x, uint y, uint z) {
    uint zz = first + z;
    uint s = zz / slab;
    uint inner = zz - s * slab;
    uint base = s * slice_size + mip_at;
    if (linear_mode != 0) {
        return base + y * blocks_row + (x << n_log2);
    }
    uint xx = x + tail_x;
    uint yy = y + tail_y;
    uint row = (((yy >> bh_log2) * blocks_row) << block_log2)
        + (spread(yy & ((1u << bh_log2) - 1u), bits_y, bh_log2) | spread(inner, bits_z, bz_count));
    uint col = ((xx >> bw_log2) << block_log2) + spread(xx & ((1u << bw_log2) - 1u), bits_x, bw_log2);
    return base + row + col;
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint w = id.x;
    uint y = id.y;
    uint z = id.z;
    if (w >= words) {
        return;
    }
    uint b = 4u * w;
    uint value = 0;
    if (y < height && z < depth) {
        if (n_log2 >= 2) {
            if ((b >> n_log2) < width) {
                value = tiled.Load(element_at(b >> n_log2, y, z) + (b & ((1u << n_log2) - 1u)));
            }
        } else {
            uint per = 4u >> n_log2;
            uint mask = n_log2 == 0 ? 0xffu : 0xffffu;
            for (uint k = 0; k < per; k++) {
                uint x = (b >> n_log2) + k;
                if (x < width) {
                    uint a = element_at(x, y, z);
                    uint v = (tiled.Load(a & ~3u) >> ((a & 3u) * 8u)) & mask;
                    value |= v << (k * (8u << n_log2));
                }
            }
        }
    }
    linear.Store(out_at + z * depth_pitch + y * row_pitch + b, value);
}
"#;

const CONSTANTS: usize = 23;
/// Places in the constants.
const C_WIDTH: usize = 4;

/// The pass's root signature and pipeline.
pub(crate) struct Pipeline {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pipeline {}
// SAFETY: as above.
unsafe impl Sync for Pipeline {}

/// The pass compiled for `device`.
pub(crate) fn make_pipeline(device: &ID3D12Device) -> windows::core::Result<Pipeline> {
    let code = crate::hlsl::compile_compute("texture detile", SOURCE)?;
    // SAFETY: our own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = code.as_slice();
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Pipeline { root, pso })
    }
}

fn pipeline() -> Option<&'static Pipeline> {
    static PIPELINE: OnceLock<Option<Pipeline>> = OnceLock::new();
    PIPELINE
        .get_or_init(|| {
            let host = crate::host::host()?;
            match make_pipeline(&host.device) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no texture detile pass: VICTORY_GPU_DETILE has no effect");
                    None
                }
            }
        })
        .as_ref()
}

/// One subresource's pass: its constants and its groups.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pass {
    c: [u32; CONSTANTS],
    groups: [u32; 3],
    mip: u32,
}

/// How a texture is detiled on the card: a pass a subresource, in the
/// order of the host's subresources (mip within array slice).
#[derive(Debug)]
pub(crate) struct Plan {
    passes: Vec<Pass>,
    /// The linear bytes (the footprints' total).
    pub(crate) total: u64,
    /// The tiled bytes (the layout's size).
    pub(crate) tiled: u64,
}

/// Packs up to 8 bit numbers below 16 into one word, 4 bits each.
fn pack(bits: &[u32]) -> Option<u32> {
    if bits.len() > 8 || bits.iter().any(|&b| b >= 16) {
        return None;
    }
    Some(bits.iter().enumerate().map(|(i, &b)| b << (4 * i)).sum())
}

impl Plan {
    /// The passes for `layout` detiled into the host footprints given
    /// (`rows` each, `mips` levels an array slice), as
    /// `crate::texture::Staging::detile` does on the CPU; an error names
    /// why the pass cannot do it (the CPU path then does).
    ///
    /// Every subresource is checked as `Layout::detile` checks it, levels
    /// held or not: one the CPU would refuse makes the whole texture stay
    /// on the CPU, where it is refused the same way.
    pub(crate) fn new(
        layout: &Layout,
        footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
        rows: &[u32],
        mips: u32,
        total: u64,
    ) -> Result<Plan, &'static str> {
        const MOST: u64 = 1 << 31;
        let d = &layout.desc;
        if !layout.addressable() {
            return Err("a layout without addressing (XOR or MSAA)");
        }
        if layout.size > MOST || total > MOST || layout.slice_size > MOST {
            return Err("over 2 GiB");
        }
        if !d.element_bytes.is_power_of_two() || d.element_bytes > 16 || mips == 0 || footprints.len() != rows.len() {
            return Err("element size");
        }
        let n = d.element_bytes;
        let n_log2 = n.trailing_zeros();
        let linear = d.swizzle == Swizzle::LINEAR;
        let three_d = d.dimension == Dimension::Tex3d;
        let slab = if layout.thick { layout.block[2] } else { 1 };
        let (bits, logs) = if linear {
            ([0u32; 3], [0u32; 3])
        } else {
            let all = layout.coordinate_bits();
            let logs = [layout.block[0], layout.block[1], slab].map(|b| if b.is_power_of_two() { b.trailing_zeros() } else { 99 });
            if (0..3).any(|a| all[a].len() as u32 != logs[a]) {
                return Err("a block whose pattern is not its size");
            }
            let packed = [pack(&all[0]), pack(&all[1]), pack(&all[2])];
            let [Some(x), Some(y), Some(z)] = packed else {
                return Err("more than 8 bits of a coordinate in a block");
            };
            ([x, y, z], logs)
        };
        let block_log2 = layout.alignment.trailing_zeros();
        let mut passes = Vec::with_capacity(footprints.len());
        for (i, (footprint, &rows)) in footprints.iter().zip(rows).enumerate() {
            let (mip, slice) = (i as u32 % mips, i as u32 / mips);
            let f = footprint.Footprint;
            let pitch = u64::from(f.RowPitch);
            let len = pitch * u64::from(rows) * u64::from(f.Depth);
            let m = layout.mips.get(mip as usize).ok_or("a level the layout has not")?;
            // `Layout::walk`'s checks.
            let first = if three_d {
                if slice != 0 {
                    return Err("a 3D slice");
                }
                0
            } else {
                slice
            };
            if first >= layout.slices {
                return Err("an array slice the layout has not");
            }
            let bytes = u64::from(m.width) * u64::from(n);
            let need = u64::from(m.depth - 1) * pitch * u64::from(rows) + u64::from(m.height - 1) * pitch + bytes;
            if need > len || pitch < bytes || m.height > rows || m.depth > f.Depth || !pitch.is_multiple_of(4) {
                return Err("rows the footprint has no room for");
            }
            if footprint.Offset + len > total {
                return Err("a footprint past the total");
            }
            let words = (pitch / 4) as u32;
            let groups = [words.div_ceil(64), rows, f.Depth];
            if groups.iter().any(|&g| g > 65535) {
                return Err("too many groups");
            }
            let (tail_x, tail_y) = m.tail.map_or((0, 0), |t| (t.x, t.y));
            let blocks_row = if linear { m.pitch * n } else { m.pitch / layout.block[0] };
            let c = [
                footprint.Offset as u32,
                f.RowPitch,
                (pitch * u64::from(rows)) as u32,
                words,
                m.width,
                m.height,
                m.depth,
                n_log2,
                u32::from(linear),
                m.offset as u32,
                layout.slice_size as u32,
                first,
                slab,
                logs[0],
                logs[1],
                logs[2],
                blocks_row,
                block_log2,
                tail_x,
                tail_y,
                bits[0],
                bits[1],
                bits[2],
            ];
            passes.push(Pass { c, groups, mip });
        }
        Ok(Plan { passes, total, tiled: layout.size })
    }

    /// The passes, levels not in `held` (a bit a level) as zeros.
    fn passes(&self, held: u32) -> impl Iterator<Item = Pass> + '_ {
        self.passes.iter().map(move |p| {
            let mut p = *p;
            if p.mip < 32 && held & (1 << p.mip) == 0 {
                p.c[C_WIDTH] = 0;
            }
            p
        })
    }
}

/// Records the passes of `plan` (levels not in `held` as zeros), reading
/// the tiled bytes at `tiled` and writing the region at `out` (both host
/// GPU addresses), which is in the unordered access state.
///
/// # Safety
/// An open list; both addresses lie in live buffers of the plan's sizes.
pub(crate) unsafe fn record(list: &ID3D12GraphicsCommandList, pipe: &Pipeline, plan: &Plan, held: u32, tiled: u64, out: u64) {
    // SAFETY: as the caller says.
    unsafe {
        list.SetComputeRootSignature(&pipe.root);
        list.SetPipelineState(&pipe.pso);
        list.SetComputeRootShaderResourceView(1, tiled);
        list.SetComputeRootUnorderedAccessView(2, out);
        for p in plan.passes(held) {
            list.SetComputeRoot32BitConstants(0, CONSTANTS as u32, p.c.as_ptr().cast(), 0);
            list.Dispatch(p.groups[0], p.groups[1], p.groups[2]);
        }
    }
}

/// Records zeros over `len` bytes (a multiple of 64 KiB) at `out`, which
/// is in the unordered access state; `any` is a readable buffer address
/// (the pass reads nothing for zeros).
///
/// # Safety
/// An open list; `out` holds `len` bytes.
pub(crate) unsafe fn record_zeros(list: &ID3D12GraphicsCommandList, pipe: &Pipeline, any: u64, out: u64, len: u64) {
    const ROW: u32 = 64 << 10;
    let mut c = [0u32; CONSTANTS];
    c[1] = ROW;
    c[3] = ROW / 4;
    c[12] = 1;
    // SAFETY: as the caller says.
    unsafe {
        list.SetComputeRootSignature(&pipe.root);
        list.SetPipelineState(&pipe.pso);
        list.SetComputeRootShaderResourceView(1, any);
        list.SetComputeRootUnorderedAccessView(2, out);
        list.SetComputeRoot32BitConstants(0, CONSTANTS as u32, c.as_ptr().cast(), 0);
        list.Dispatch(ROW / 4 / 64, (len / u64::from(ROW)) as u32, 1);
    }
}

// ---------------------------------------------------------------------
// Upload buffers for tiled bytes.

/// A mapped upload buffer holding a texture's tiled bytes for the card.
/// Given back to the spare ones when the last holder lets go (a texture,
/// or a job until its submission has run).
pub(crate) struct Upload {
    buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Its bytes (a power of two up to [`UPLOAD_POOLED`]), or 0: not
    /// pooled.
    class: u64,
}

// SAFETY: a D3D12 buffer and its mapping, written only by the one refresh
// that holds it alone.
unsafe impl Send for Upload {}
// SAFETY: as above.
unsafe impl Sync for Upload {}

const UPLOAD_POOLED: u64 = 64 << 20;
const UPLOAD_SPARE_MOST: u64 = 256 << 20;

struct Spare {
    by_class: std::collections::BTreeMap<u64, Vec<(ID3D12Resource, usize, u64)>>,
    bytes: u64,
}

// SAFETY: buffers nobody uses while they are spare.
unsafe impl Send for Spare {}

static SPARE: Mutex<Spare> = Mutex::new(Spare { by_class: std::collections::BTreeMap::new(), bytes: 0 });

impl Drop for Upload {
    fn drop(&mut self) {
        UPLOAD_BYTES.fetch_sub(if self.class != 0 { self.class } else { 0 }, Relaxed);
        if self.class == 0 {
            return;
        }
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if spare.bytes + self.class <= UPLOAD_SPARE_MOST {
            spare.bytes += self.class;
            spare.by_class.entry(self.class).or_default().push((self.buffer.clone(), self.data as usize, self.gpu));
        }
    }
}

/// Upload buffers for tiled bytes alive (pooled ones; not counting spare).
static UPLOAD_BYTES: AtomicU64 = AtomicU64::new(0);

fn take_upload(bytes: u64) -> Option<Upload> {
    let class = bytes.max(64 << 10).next_power_of_two();
    if class <= UPLOAD_POOLED {
        let taken = {
            let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
            let taken = spare.by_class.get_mut(&class).and_then(Vec::pop);
            if taken.is_some() {
                spare.bytes -= class;
            }
            taken
        };
        if let Some((buffer, data, gpu)) = taken {
            UPLOAD_BYTES.fetch_add(class, Relaxed);
            return Some(Upload { buffer, data: data as *mut u8, gpu, class });
        }
    }
    let host = crate::host::host()?;
    let size = if class <= UPLOAD_POOLED { class } else { bytes.next_multiple_of(64 << 10) };
    let (buffer, data) = host.upload_buffer(size)?;
    crate::host::name_frame(&buffer, || format!("tiled bytes for the card's detile {size:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let class = if class <= UPLOAD_POOLED { class } else { 0 };
    UPLOAD_BYTES.fetch_add(class, Relaxed);
    Some(Upload { buffer, data, gpu, class })
}

// ---------------------------------------------------------------------
// The card pool of regions.

/// Bytes of a slab of the pool.
const SLAB: u64 = 32 << 20;
/// Regions start on this.
const GRAIN: u64 = 64 << 10;

struct Slab {
    buffer: ID3D12Resource,
    gpu: u64,
    /// Free ranges (start, length), by start.
    free: Vec<(u64, u64)>,
}

struct Pool {
    slabs: Vec<Slab>,
}

// SAFETY: device-local buffers lists copy from; the ranges are ours.
unsafe impl Send for Pool {}

static POOL: Mutex<Pool> = Mutex::new(Pool { slabs: Vec::new() });

/// Of the pool and regions of their own: bytes in use now, the most since
/// the last line and since the start; regions of their own made since the
/// start (the pool was full or they were larger than a slab); uploads put
/// off because no card memory could be had.
static REGIONS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// The pool's bytes at most (`VICTORY_GPU_DETILE_MB`).
fn budget() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_GPU_DETILE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map_or(256 << 20, |mb| mb << 20)
    })
}

/// Card memory a list copies a texture's upload from, until the list is
/// reset.
pub(crate) struct Region {
    buffer: ID3D12Resource,
    /// The region's place in `buffer`, its host GPU address, its bytes.
    pub(crate) offset: u64,
    gpu: u64,
    len: u64,
    /// Which slab it is of (`usize::MAX`: a buffer of its own).
    slab: usize,
}

// SAFETY: a device-local buffer.
unsafe impl Send for Region {}
// SAFETY: as above.
unsafe impl Sync for Region {}

impl Region {
    pub(crate) fn buffer(&self) -> &ID3D12Resource {
        &self.buffer
    }

    pub(crate) fn gpu(&self) -> u64 {
        self.gpu
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        REGIONS[0].fetch_sub(self.len, Relaxed);
        if self.slab == usize::MAX {
            return;
        }
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let Some(slab) = pool.slabs.get_mut(self.slab) else { return };
        let free = &mut slab.free;
        let at = free.partition_point(|&(s, _)| s < self.offset);
        free.insert(at, (self.offset, self.len));
        // Joined with its neighbours.
        if at + 1 < free.len() && free[at].0 + free[at].1 == free[at + 1].0 {
            free[at].1 += free[at + 1].1;
            free.remove(at + 1);
        }
        if at > 0 && free[at - 1].0 + free[at - 1].1 == free[at].0 {
            free[at - 1].1 += free[at].1;
            free.remove(at);
        }
    }
}

fn note_use(len: u64) {
    let now = REGIONS[0].fetch_add(len, Relaxed) + len;
    REGIONS[1].fetch_max(now, Relaxed);
    REGIONS[2].fetch_max(now, Relaxed);
}

/// A region of `bytes` for one list's upload of a texture: from the pool,
/// else a buffer of its own; `None` without card memory.
pub(crate) fn region(bytes: u64) -> Option<Region> {
    let len = bytes.max(1).next_multiple_of(GRAIN);
    if len <= SLAB {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let mut found = None;
        for (i, slab) in pool.slabs.iter_mut().enumerate() {
            if let Some(at) = slab.free.iter().position(|&(_, l)| l >= len) {
                let (start, l) = slab.free[at];
                if l == len {
                    slab.free.remove(at);
                } else {
                    slab.free[at] = (start + len, l - len);
                }
                found = Some((i, start));
                break;
            }
        }
        if found.is_none() && (pool.slabs.len() as u64 + 1) * SLAB <= budget().max(SLAB)
            && let Some(host) = crate::host::host()
            && let Some(buffer) = host.uav_buffer(SLAB)
        {
            crate::host::name(&buffer, || "texture detile pool slab".into());
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let free = if len < SLAB { vec![(len, SLAB - len)] } else { Vec::new() };
            pool.slabs.push(Slab { buffer, gpu, free });
            found = Some((pool.slabs.len() - 1, 0));
        }
        if let Some((slab, offset)) = found {
            let s = &pool.slabs[slab];
            note_use(len);
            return Some(Region { buffer: s.buffer.clone(), offset, gpu: s.gpu + offset, len, slab });
        }
    }
    // A buffer of its own (zeroed by the system, but every region is
    // written whole before a copy reads it anyway).
    let host = crate::host::host()?;
    let Some(buffer) = host.uav_buffer(len) else {
        REGIONS[4].fetch_add(1, Relaxed);
        return None;
    };
    crate::host::name_frame(&buffer, || format!("texture detile region of its own {len:#x}"));
    REGIONS[3].fetch_add(1, Relaxed);
    note_use(len);
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    Some(Region { buffer, offset: 0, gpu, len, slab: usize::MAX })
}

/// An upload that could not get card memory was put off (counted).
pub(crate) fn put_off() {
    REGIONS[4].fetch_add(1, Relaxed);
}

// ---------------------------------------------------------------------
// A texture's card state.

/// Where a texture's latest content is.
enum Latest {
    /// Nothing detiled yet: the region is zeros.
    Nothing,
    /// The tiled bytes in this upload buffer, the levels in `held` the
    /// title's (the others zeros).
    Tiled(Arc<Upload>, u32),
    /// Detiled on the CPU into the staging buffer (`VICTORY_GPU_DETILE=alt`
    /// off, or no upload buffer could be had).
    Linear,
}

/// The card side of a texture's staging.
pub(crate) struct Gpu {
    plan: Arc<Plan>,
    latest: Mutex<Latest>,
}

/// Textures left on the CPU path for their layout since the start.
static FALLBACK: AtomicU64 = AtomicU64::new(0);

impl Gpu {
    /// The card side for a texture (`None`: the CPU path, said once and
    /// counted).
    pub(crate) fn new(
        layout: &Layout,
        footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
        rows: &[u32],
        mips: u32,
        total: u64,
    ) -> Option<Gpu> {
        if !wanted() {
            return None;
        }
        match Plan::new(layout, footprints, rows, mips, total) {
            Ok(plan) => Some(Gpu { plan: Arc::new(plan), latest: Mutex::new(Latest::Nothing) }),
            Err(why) => {
                FALLBACK.fetch_add(1, Relaxed);
                once!(desc = ?layout.desc, why, "a texture left to the CPU detile (VICTORY_GPU_DETILE): {why}");
                None
            }
        }
    }

    /// The refresh found the memory changed: `tiled` (the texture's memory,
    /// the layout's bytes) is copied for the card, the levels in `held`
    /// the title's. False if no upload buffer could be had (the caller
    /// detiles on the CPU).
    pub(crate) fn take_tiled(&self, tiled: &[u8], held: u32) -> bool {
        let mut latest = self.latest.lock().unwrap_or_else(|e| e.into_inner());
        let len = (self.plan.tiled as usize).min(tiled.len());
        // The buffer of the last time, if nothing else holds it (no job
        // that may not have run): else another, and the old one goes back
        // to the spare ones when its last job is done.
        let reuse = match &*latest {
            Latest::Tiled(upload, _) if Arc::strong_count(upload) > 1 => {
                reclaim();
                Arc::strong_count(upload) == 1
            }
            Latest::Tiled(..) => true,
            _ => false,
        };
        let upload = if reuse {
            let Latest::Tiled(upload, _) = std::mem::replace(&mut *latest, Latest::Nothing) else {
                unreachable!()
            };
            upload
        } else {
            match take_upload(self.plan.tiled) {
                Some(u) => Arc::new(u),
                None => return false,
            }
        };
        // SAFETY: the buffer's mapping holds the layout's bytes, and nothing
        // reads it now: no job names it (the count above), and the card is
        // past every job that did.
        unsafe { std::ptr::copy_nonoverlapping(tiled.as_ptr(), upload.data, len) };
        *latest = Latest::Tiled(upload, held);
        true
    }

    /// The refresh detiled on the CPU into the staging buffer.
    pub(crate) fn set_linear(&self) {
        *self.latest.lock().unwrap_or_else(|e| e.into_inner()) = Latest::Linear;
    }

    /// At a submission of a list that copies from `region`, after the
    /// texture's refresh: queues what fills the region with the latest
    /// content (`staging`: the staging buffer, for content detiled on the
    /// CPU).
    pub(crate) fn queue_fill(&self, region: &Region, staging: Option<&ID3D12Resource>) {
        let latest = self.latest.lock().unwrap_or_else(|e| e.into_inner());
        let job = match &*latest {
            Latest::Nothing => Job::Zeros { out: region.buffer.clone(), gpu: region.gpu, len: region.len },
            Latest::Tiled(upload, held) => Job::Detile {
                plan: self.plan.clone(),
                held: *held,
                tiled: upload.clone(),
                out: region.buffer.clone(),
                gpu: region.gpu,
            },
            Latest::Linear => match staging {
                Some(from) => Job::Copy {
                    from: from.clone(),
                    out: region.buffer.clone(),
                    offset: region.offset,
                    len: self.plan.total,
                },
                None => Job::Zeros { out: region.buffer.clone(), gpu: region.gpu, len: region.len },
            },
        };
        PENDING.lock().unwrap_or_else(|e| e.into_inner()).push(job);
    }
}

// ---------------------------------------------------------------------
// Jobs, and our list ahead of a submission's.

enum Job {
    Zeros { out: ID3D12Resource, gpu: u64, len: u64 },
    Detile { plan: Arc<Plan>, held: u32, tiled: Arc<Upload>, out: ID3D12Resource, gpu: u64 },
    Copy { from: ID3D12Resource, out: ID3D12Resource, offset: u64, len: u64 },
}

// SAFETY: D3D12 buffers.
unsafe impl Send for Job {}

static PENDING: Mutex<Vec<Job>> = Mutex::new(Vec::new());

struct Lists {
    /// Allocator and list, and the fence value after which they may be
    /// reset.
    free: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList, u64)>,
    /// Jobs sent, held until the fence passes their value.
    flying: Vec<(u64, Vec<Job>)>,
    fence: Option<ID3D12Fence>,
    /// A small buffer the zeros' pass names as its source.
    any: Option<ID3D12Resource>,
}

// SAFETY: D3D12 objects.
unsafe impl Send for Lists {}

static LISTS: Mutex<Lists> = Mutex::new(Lists { free: Vec::new(), flying: Vec::new(), fence: None, any: None });

/// Of our lists, since the start: lists sent, jobs in them, and the
/// microseconds recording and sending them took.
static SENT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Lets go of jobs whose submission the card has finished.
fn reclaim() {
    let done = {
        let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(fence) = lists.fence.clone() else { return };
        // SAFETY: a plain query.
        let completed = unsafe { fence.GetCompletedValue() };
        let (done, flying): (Vec<_>, Vec<_>) = std::mem::take(&mut lists.flying).into_iter().partition(|(v, _)| *v <= completed);
        lists.flying = flying;
        done
    };
    // Dropped outside the lock (upload buffers go back to the spare ones).
    drop(done);
}

/// Before a submission's lists go to `queue`: the jobs its lists queued
/// are recorded into a list of ours and executed ahead of them. `value`:
/// the fence value the submission signals when its lists have run (its
/// first signal; later ones are larger).
pub(crate) fn before_lists(queue: &ID3D12CommandQueue, fence: &ID3D12Fence, value: u64) {
    if mode() == 0 {
        return;
    }
    let jobs = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    {
        let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
        if lists.fence.is_none() {
            lists.fence = Some(fence.clone());
        }
    }
    reclaim();
    if jobs.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let Some(pipe) = pipeline() else {
        crate::host::fail(format_args!("texture detile jobs without the pass"));
    };
    let Some(host) = crate::host::host() else { return };
    let mut lists = LISTS.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a plain query.
    let completed = unsafe { fence.GetCompletedValue() };
    let at = lists.free.iter().position(|(_, _, v)| *v <= completed);
    let made = match at {
        Some(at) => Ok(lists.free.swap_remove(at)),
        // SAFETY: plain creation.
        None => unsafe {
            host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).and_then(|a: ID3D12CommandAllocator| {
                let l: ID3D12GraphicsCommandList = host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None)?;
                l.Close()?;
                crate::host::name(&l, || "texture detile list".into());
                Ok((a, l, 0))
            })
        },
    };
    let (allocator, list, _) = match made {
        Ok(made) => made,
        Err(e) => crate::host::fail(format_args!("no list for texture detile jobs ({e})")),
    };
    if lists.any.is_none() {
        lists.any = host.uav_buffer(256);
    }
    let Some(any) = lists.any.clone() else {
        crate::host::fail(format_args!("no buffer for texture detile jobs"));
    };
    // SAFETY: the allocator's last list has run (its value passed); live
    // buffers, the addresses lie in them.
    let recorded = unsafe {
        (|| -> windows::core::Result<()> {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            let distinct = |want: &dyn Fn(&Job) -> Option<&ID3D12Resource>| {
                let mut out: Vec<ID3D12Resource> = Vec::new();
                for j in &jobs {
                    if let Some(r) = want(j)
                        && !out.iter().any(|o| o.as_raw() == r.as_raw())
                    {
                        out.push(r.clone());
                    }
                }
                out
            };
            let copied = distinct(&|j| match j {
                Job::Copy { out, .. } => Some(out),
                _ => None,
            });
            let written = distinct(&|j| match j {
                Job::Zeros { out, .. } | Job::Detile { out, .. } => Some(out),
                _ => None,
            });
            let moves = |set: &[ID3D12Resource], from, to| -> Vec<D3D12_RESOURCE_BARRIER> {
                set.iter().map(|r| crate::host::transition(r, from, to)).collect()
            };
            if !copied.is_empty() {
                list.ResourceBarrier(&moves(&copied, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST));
                for j in &jobs {
                    if let Job::Copy { from, out, offset, len } = j {
                        list.CopyBufferRegion(out, *offset, from, 0, *len);
                    }
                }
                list.ResourceBarrier(&moves(&copied, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON));
            }
            if !written.is_empty() {
                list.ResourceBarrier(&moves(&written, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
                let any_gpu = any.GetGPUVirtualAddress();
                for j in &jobs {
                    match j {
                        Job::Zeros { gpu, len, .. } => record_zeros(&list, pipe, any_gpu, *gpu, *len),
                        Job::Detile { plan, held, tiled, gpu, .. } => record(&list, pipe, plan, *held, tiled.gpu, *gpu),
                        Job::Copy { .. } => {}
                    }
                }
                list.ResourceBarrier(&moves(&written, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON));
            }
            list.Close()?;
            queue.ExecuteCommandLists(&[Some(list.cast()?)]);
            Ok(())
        })()
    };
    if let Err(e) = recorded {
        crate::host::fail(format_args!("texture detile jobs could not be sent ({e})"));
    }
    SENT[0].fetch_add(1, Relaxed);
    SENT[1].fetch_add(jobs.len() as u64, Relaxed);
    lists.free.push((allocator, list, value));
    lists.flying.push((value, jobs));
    SENT[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
}

/// Every five seconds with the switch set: the pool and the jobs.
pub(crate) fn log() {
    if mode() == 0 {
        return;
    }
    let mb = |b: u64| b as f64 / (1u64 << 20) as f64;
    let slabs = POOL.lock().unwrap_or_else(|e| e.into_inner()).slabs.len() as u64;
    let peak = REGIONS[1].swap(REGIONS[0].load(Relaxed), Relaxed);
    tracing::info!(
        "texture detile on the card (VICTORY_GPU_DETILE): card regions {:.0} MB in use, peak {:.0} MB since the last line \
         ({:.0} since the start), pool {:.0} of {:.0} MB made; since the start {} regions of their own (pool full or \
         larger than a slab), {} uploads put off without card memory, {} textures left to the CPU for their layout; \
         {} lists of ours sent with {} jobs ({:.1} ms to record and send); tiled upload buffers {:.0} MB",
        mb(REGIONS[0].load(Relaxed)),
        mb(peak),
        mb(REGIONS[2].load(Relaxed)),
        mb(slabs * SLAB),
        mb(budget()),
        REGIONS[3].load(Relaxed),
        REGIONS[4].load(Relaxed),
        FALLBACK.load(Relaxed),
        SENT[0].load(Relaxed),
        SENT[1].load(Relaxed),
        SENT[2].load(Relaxed) as f64 / 1e3,
        mb(UPLOAD_BYTES.load(Relaxed)),
    );
}
