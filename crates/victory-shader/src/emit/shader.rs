//! A DXIL vertex, pixel, compute or mesh shader under construction: the module,
//! the `dx.op` declarations it calls, its signatures and bound resources,
//! and the metadata and container that wrap it up (SM 6.6, validator 1.6).

use super::bitcode::{Attr, Bin, Cast, Md, Module, Ty, V};
use super::container::{self, Binding, Psv, SigElement};

/// Where the shader reads its user data: `b0` in this space.
pub const USER_DATA_SPACE: u32 = 0;
/// Where immediate samplers are bound: `s<n>` in this space.
pub const SAMPLER_SPACE: u32 = 0;
/// Where root CBVs and SRVs are bound: `b<root parameter index>` and
/// `t<root parameter index>` in this space.
pub const ROOT_CBV_SPACE: u32 = 1;

/// The root SRV `t<SLOT_STRIDES>` in [`ROOT_CBV_SPACE`] of a shader that
/// reads buffers through descriptors it picks from the view heap at an
/// index it computes (SM 6.6 dynamic resources): a raw buffer of
/// [`SLOT_STRIDES_COUNT`] dwords the host keeps, dword `i` the bytes per
/// element of the buffer view at heap slot `i` (0: none or not a buffer).
/// The console's descriptor holds that stride and its hardware multiplies
/// the element index by it; the program's code does not say it, and the
/// used-resource list only names what tables hold. A shader that has this
/// root SRV also tells the host that the buffer views of the whole heap
/// must be ready when it runs, not only those its tables name.
pub const SLOT_STRIDES: u8 = 255;
/// Dwords of the [`SLOT_STRIDES`] buffer: heap indices are clamped to it.
pub const SLOT_STRIDES_COUNT: u32 = 1 << 21;
/// Where the [`SLOT_STRIDES`] buffer goes on, in dwords: two per heap slot,
/// the low and then the high dword of the title address of the buffer view
/// at that slot (0: none). For a program that takes a buffer view out of the
/// heap at an index it computes and reads the address out of the descriptor
/// to walk the memory behind it (the shadow-ray trace programs take their
/// ray-tracing structure this way).
pub const SLOT_ADDRESSES: u32 = SLOT_STRIDES_COUNT;
/// Dwords of the whole [`SLOT_STRIDES`] buffer: the strides, then the
/// addresses ([`SLOT_ADDRESSES`]).
pub const SLOT_STRIDES_DWORDS: u32 = 3 * SLOT_STRIDES_COUNT;

/// Title addresses of the buffer views of the view heap by slot, for
/// console code that reads a descriptor's address out of a table at an
/// index it computes (ray-tracing structures and the programs that walk
/// them) and finds no root view to ask the host about. Two dwords per
/// slot: the title address of the view's first byte. Slots without a
/// buffer view read zero. Like [`SLOT_STRIDES`], the host readies the
/// buffer views of the whole heap when a shader has this root SRV.
pub const SLOT_VIEWS: u8 = 252;
/// Slots of the [`SLOT_VIEWS`] buffer (heap indices are clamped to it).
pub const SLOT_VIEWS_COUNT: u32 = SLOT_STRIDES_COUNT;

/// Title memory by address (`resource::MEMORY`): console code reads and
/// writes through raw 40-bit addresses it takes from root constants,
/// descriptors and other memory (the ray-tracing structure builder and the
/// programs that walk its structures). The host keeps copies of the title
/// memory shaders use; a shader with such an access finds the copy by the
/// address:
///
/// - the root SRV `t<SLOT_MEMORY>` in [`ROOT_CBV_SPACE`] is a directory of
///   [`MEMORY_PAGES`] entries of two dwords, one per 64 KiB of title
///   address space: the number of the host copy holding that page (0: none
///   yet) and the page's byte offset in that copy;
/// - copy `n` is the raw buffer view at view-heap slot `base + n`, `base`
///   the user-data dword of kind [`USER_MEMORY_BASE`]; slot `base` itself
///   reads zeros and drops writes;
/// - the root UAV `u<SLOT_FAULTS>` takes the page numbers of accesses that
///   found no copy ([`MEMORY_FAULTS`] dwords, dword 0 a sink), which the
///   host reads afterwards to make those copies for the frames to come.
pub const SLOT_MEMORY: u8 = 254;
/// See [`SLOT_MEMORY`].
pub const SLOT_FAULTS: u8 = 253;
/// Entries of the [`SLOT_MEMORY`] directory: title addresses below 128 GiB.
pub const MEMORY_PAGES: u32 = 1 << 21;
/// Dwords of the [`SLOT_FAULTS`] buffer.
pub const MEMORY_FAULTS: u32 = 4096;

/// The root UAV `u<SLOT_MARKS>` in [`ROOT_CBV_SPACE`]: which views of the
/// view heap shaders read. Every handle a shader makes from the view heap
/// for reading (not a sampler, not an unordered access view) stores 1 at
/// dword `slot + 1` of this raw buffer (dword 0 is a sink): from the first
/// lane of the wave for an index that is the same in every lane, from
/// every lane for one that is not. The host clears it every frame and reads
/// it back, so a texture's host copy is made, kept and uploaded for the
/// shaders that read it, not for every view the title's 720,000-slot heap
/// names (`victory-gpu` `marks.rs`). `VICTORY_VIEW_MARKS=0`: none.
pub const SLOT_MARKS: u8 = 251;

/// The root SRV `t<SLOT_CBV_ROWS>` in [`ROOT_CBV_SPACE`]: a raw buffer the
/// host keeps, dword `i` the rows (16 bytes) by which the constant buffer
/// view at view-heap slot `i` begins past the host's view of it. The
/// console wants constant buffers 16-byte aligned, the host 256: the host
/// makes the view at the 256-byte boundary at or below the title's address,
/// in the title's own slot, and a shader that reads a constant buffer it
/// picks from the heap adds these rows to every row it reads
/// ([`Shader::cbuffer_row`]). Tables holding constant buffers then need no
/// copy made afresh for every draw (the dynamic slots, 262,144 on NVIDIA,
/// ran out: 1,500 tables a frame and their draws dropped, run mk1).
/// Off unless `VICTORY_CBV_IN_PLACE=1`: the title writes most of its
/// constant buffer views anew every frame (29,000 of a frame's views were
/// still copied with it on), tables without slots fell only from 1,580 to
/// 679 a frame, and both runs with it on logged a driver error (cbv1 lost
/// the card with every view in place; cbv2, stable views only, did not).
pub const SLOT_CBV_ROWS: u8 = 250;

/// Whether constant buffer views picked from the heap take their rows from
/// the host's table ([`SLOT_CBV_ROWS`]).
pub fn cbv_in_place() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CBV_IN_PLACE").is_some_and(|v| v == "1"))
}

/// Whether shaders mark the views they read ([`SLOT_MARKS`]).
pub fn view_marks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_MARKS").is_none_or(|v| v != "0"))
}
/// `RootType::Other` code of a user-data dword the host computes: the view
/// heap slot of the memory copies' views ([`SLOT_MEMORY`]).
pub const USER_MEMORY_BASE: u32 = 0x100;
/// `RootType::Other` code of a user-data dword the host computes for the
/// descriptor table of root parameter `slot`: the byte offset (32 bytes a
/// slot) of a table of its own that holds only the table's constant buffer
/// views, in the order of their offsets in the table (the used-resource
/// list's entries of kind CBV), copied for the draw. The shader reads every
/// constant buffer of that table from there and the table's other views
/// from the title's own slots, so a draw copies its constant buffers alone
/// rather than every view of every table that holds one (190,000 slots a
/// frame, the dynamic slots ran out: run mk1). `VICTORY_CBV_SPLIT=0`: off.
pub const USER_CBV_TABLE: u32 = 0x104;

/// Whether constant buffers read through tables come from tables of their
/// own ([`USER_CBV_TABLE`]); a program that picks one by an index it
/// computes is translated without (see `lower::CBV_SPLIT_REFUSED`).
pub fn cbv_split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CBV_SPLIT").is_none_or(|v| v != "0"))
}
/// `RootType::Other` code of a user-data dword the host computes: the low
/// 32 bits of the title address of the buffer view at dword `dword` of the
/// descriptor table of root parameter `slot`; [`USER_VIEW_ADDRESS`] + 1:
/// its bits from 32 up. (Console code reads the address out of a buffer
/// descriptor and uses it as a pointer.)
pub const USER_VIEW_ADDRESS: u32 = 0x101;
/// `RootType::Other` code of a user-data dword the host computes: the
/// bytes (0 to 15) by which the raw view the host binds for the root view
/// of root parameter `slot` begins before the title's address. A raw view
/// begins on 16 bytes of its host buffer; the console's root views begin
/// anywhere (the structure builder's counters sit at addresses ending in
/// 4 and c). The shader adds the dword to every byte offset through that
/// view. Asked for every root UAV, and for the root SRVs of programs with
/// a spilled user-data table.
pub const USER_VIEW_BIAS: u32 = 0x103;

/// `ShaderFlags` bits (metadata) and their `SFI0` feature bits.
pub(crate) mod flags {
    pub const RAW_BUFFERS: u64 = 1 << 4;
    /// `[earlydepthstencil]`.
    pub const EARLY_DEPTH: u64 = 1 << 3;
    /// Unordered access views outside pixel and compute shaders.
    pub const UAVS_EVERY_STAGE: u64 = 1 << 16;
    pub const DOUBLES: u64 = 1 << 2;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TYPED_UAV_LOADS: u64 = 1 << 13;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const BARYCENTRICS: u64 = 1 << 22;
    pub const INT64: u64 = 1 << 20;
    pub const ATOMIC_TYPED64: u64 = 1 << 27;
    pub const ATOMIC_HEAP64: u64 = 1 << 32;
    pub const RESOURCE_HEAP: u64 = 1 << 30;
    pub const SAMPLER_HEAP: u64 = 1 << 31;
    pub const ROVS: u64 = 1 << 18;
}

pub(crate) struct Types {
    pub void: Ty,
    pub i1: Ty,
    pub i32: Ty,
    pub f32: Ty,
    pub handle: Ty,
    pub props: Ty,
    pub bind: Ty,
    pub cbuf_ret: Ty,
    pub res_ret: Ty,
}

/// What a mesh shader declares beyond its thread-group size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mesh {
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
}

/// What a geometry shader declares: triangles in, triangle strips out on
/// stream 0, one instance.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Geometry {
    pub max_vertices: u32,
}

/// The null slot every host view heap keeps at its end (0: no limit). Shaders translated while
/// it is set read that slot for any view index at or past it. A read past a heap's end is
/// undefined: a host that caps its heap (NVIDIA at 1 000 000 views; a retail game's heap is
/// larger) hung its GPU (GPU-based validation: "descriptor heap index out of bounds"), and AMD
/// RDNA3 took a page fault and removed the device.
static VIEW_LIMIT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// See [`VIEW_LIMIT`]: `null_slot` is the last slot of the host's view heap, kept null.
pub fn set_view_heap_limit(null_slot: u32) {
    VIEW_LIMIT.store(null_slot, std::sync::atomic::Ordering::Relaxed);
}

/// See [`VIEW_LIMIT`]: the null slot shaders translated now clamp view indices to (0: none).
pub fn view_heap_limit() -> u32 {
    VIEW_LIMIT.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) struct Shader {
    pub m: Module,
    pub t: Types,
    pub pixel: bool,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// A mesh shader's per-primitive outputs.
    pub prims: Vec<SigElement>,
    /// A mesh shader (`threads` is its thread-group size).
    pub mesh: Option<Mesh>,
    /// A geometry shader (the rectangle stage, `rect.rs`); `pixel` is false.
    pub geometry: Option<Geometry>,
    /// Dwords of user data the shader reads (0: no user-data buffer).
    pub user_data: u32,
    /// Immediate samplers, bound as `s0..` in [`SAMPLER_SPACE`].
    pub samplers: u32,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    /// Root CBVs read, by root parameter index, with their handles.
    pub root_cbvs: Vec<(u8, V)>,
    /// Root SRVs (raw buffers) read, likewise.
    pub root_srvs: Vec<(u8, V)>,
    /// Root UAVs (raw buffers) read or written, likewise.
    pub root_uavs: Vec<(u8, V)>,
    pub flags: u64,
    /// Every UAV the shader opens is rasterizer ordered ([`Self::annotate`]).
    pub rov: bool,
    /// Heap handles made with a non-uniform index.
    pub nonuniform_handles: u32,
    /// View marks stored ([`SLOT_MARKS`]), and the (block, index) pairs
    /// already marked.
    pub marks: u32,
    marked: std::collections::HashSet<(usize, V)>,
    /// Constant buffer handles picked from the heap, with their slot
    /// ([`SLOT_CBV_ROWS`]), and the rows read per (block, slot).
    cbv_slots: std::collections::HashMap<V, V>,
    cbv_rows: std::collections::HashMap<(usize, V), V>,
    /// A compute shader with this thread-group size.
    pub threads: Option<[u32; 3]>,
    /// The wave size a compute shader requires (`WaveSize`).
    pub wave_size: Option<u32>,
    /// Thread-group shared memory, in dwords.
    pub lds_dwords: u32,
    lds: Option<V>,
    main: V,
    pub entry: usize,
}

impl Shader {
    pub fn new(pixel: bool) -> Self {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i8 = m.int(8);
        let i32 = m.int(32);
        let f32 = m.float();
        let i8p = m.ptr(i8);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let props = m.named_struct("dx.types.ResourceProperties", vec![i32, i32]);
        let bind = m.named_struct("dx.types.ResBind", vec![i32, i32, i32, i8]);
        let cbuf_ret = m.named_struct("dx.types.CBufRet.i32", vec![i32, i32, i32, i32]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32, f32, f32, f32, i32]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        let main = m.function("main", main_ty, None, true);
        let entry = m.new_block();
        Shader {
            m,
            t: Types {
                void,
                i1,
                i32,
                f32,
                handle,
                props,
                bind,
                cbuf_ret,
                res_ret,
            },
            pixel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh: None,
            geometry: None,
            user_data: 0,
            samplers: 0,
            cmp_samplers: 0,
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            flags: 0,
            rov: false,
            nonuniform_handles: 0,
            marks: 0,
            marked: std::collections::HashSet::new(),
            cbv_slots: std::collections::HashMap::new(),
            cbv_rows: std::collections::HashMap::new(),
            threads: None,
            wave_size: None,
            lds_dwords: 0,
            lds: None,
            main,
            entry,
        }
    }

    /// The thread-group shared memory: `[lds_dwords x i32]` in address
    /// space 3.
    pub fn lds(&mut self) -> V {
        if let Some(g) = self.lds {
            return g;
        }
        let arr = self.m.array(self.lds_dwords, self.t.i32);
        // A geometry shader has no group: its "LDS" is the thread's own array
        // (the records it writes before it emits them).
        let g = if self.geometry.is_some() {
            self.m.private_global("gsout", arr)
        } else {
            self.m.global("lds", arr, 3)
        };
        self.lds = Some(g);
        g
    }

    /// Calls `dx.op.<name>` (class and overload) with `opcode` and `args`.
    pub fn op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        let mut params = vec![self.t.i32];
        params.extend(args.iter().map(|&a| self.m.type_of(a)));
        let fty = self.m.func_ty(ret, params);
        self.m.ptr(fty);
        let f = self
            .m
            .function(&format!("dx.op.{name}"), fty, Some(attr), false);
        let mut all = vec![self.m.i32(opcode)];
        all.extend(args);
        self.m.call(block, f, all)
    }

    /// The handle of the user-data constant buffer, made in the entry block.
    pub fn user_data_handle(&mut self) -> V {
        let bind = [
            self.m.i32(0),
            self.m.i32(0),
            self.m.i32(USER_DATA_SPACE),
            self.m.i8(2), // CBV
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let (i, f) = (self.m.i32(0), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        // The size is known in `finish`, which fixes this placeholder.
        self.annotate(block, h, 13, u32::MAX)
    }

    /// The handle of the root CBV of root parameter `slot`, made in the
    /// entry block.
    pub fn root_cbv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_cbvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 13, 65536);
        self.root_cbvs.push((slot, h));
        h
    }

    /// The handle of the root SRV (a raw buffer) of root parameter `slot`.
    pub fn root_srv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_srvs.push((slot, h));
        h
    }

    /// The handle of the root UAV (a raw buffer) of root parameter `slot`:
    /// `u<slot>` in [`ROOT_CBV_SPACE`].
    pub fn root_uav_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_uavs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11 | 0x1000, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_uavs.push((slot, h));
        h
    }

    /// The handle of immediate sampler `n` (a comparison sampler when its
    /// bit in `cmp_samplers` is set).
    /// A handle to the static sampler at `n`.
    pub fn sampler_handle(&mut self, block: usize, n: u32) -> V {
        let bind = [
            self.m.i32(n),
            self.m.i32(n),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let (i, f) = (self.m.i32(n), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        let cmp = if self.cmp_samplers >> n & 1 != 0 {
            0x8000
        } else {
            0
        };
        self.annotate(block, h, 14 | cmp, 0)
    }

    /// A handle to the static sampler at `lo..=hi` picked by `index`: the
    /// console compiler chooses between two immediate samplers on a branch,
    /// and D3D12's static sampler registers can be indexed within one range.
    pub fn sampler_range_handle(&mut self, block: usize, lo: u32, hi: u32, index: V) -> V {
        let bind = [
            self.m.i32(lo),
            self.m.i32(hi),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        // The index comes from a per-lane branch condition.
        let f = self.m.bool(true);
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, index, f],
        );
        let mut cmp = 0;
        for n in lo..=hi {
            if self.cmp_samplers >> n & 1 != 0 {
                cmp = 0x8000;
            }
        }
        self.annotate(block, h, 14 | cmp, 0)
    }

    /// Annotates a handle with its resource properties. Bit 12 of `props0` is
    /// "UAV"; with `rov` set a UAV also gets bit 13, rasterizer ordered.
    pub fn annotate(&mut self, block: usize, h: V, props0: u32, props1: u32) -> V {
        let props0 = if self.rov && props0 & 0x1000 != 0 {
            self.flags |= flags::ROVS;
            props0 | 0x2000
        } else {
            props0
        };
        let p = [self.m.i32(props0), self.m.i32(props1)];
        let p = self.m.aggregate(self.t.props, &p);
        self.op(
            block,
            "annotateHandle",
            Attr::ReadNone,
            self.t.handle,
            216,
            &[h, p],
        )
    }

    /// A descriptor heap handle, annotated with `props`; `nonuniform` when
    /// the index may differ between lanes (`NonUniformResourceIndex`).
    pub fn heap_handle(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        // a view index past the host heap (`set_view_heap_limit`) reads its null slot instead
        let limit = VIEW_LIMIT.load(std::sync::atomic::Ordering::Relaxed);
        // (as `dx.op.binary` UMin: a compare and select emitted here made the bitcode invalid)
        let index = if !sampler && limit != 0 {
            let null = self.m.i32(limit);
            let i32t = self.t.i32;
            let v = self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[index, null]);
            self.m.tag(v, super::bitcode::Tag::Clamp);
            v
        } else if sampler {
            // A sampler heap has at most 2,048 samplers on any host: an
            // index past that (a number read from memory the host has not
            // got yet) stays inside the largest heap there is.
            // shortcut: not the bound heap's own count, which a title
            // heap smaller than 2,048 would need.
            let last = self.m.i32(2047);
            let i32t = self.t.i32;
            let v = self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[index, last]);
            self.m.tag(v, super::bitcode::Tag::Clamp);
            v
        } else {
            index
        };
        self.flags |= if sampler {
            flags::SAMPLER_HEAP
        } else {
            flags::RESOURCE_HEAP
        };
        // (Only with a host heap's size: the marks buffer has a dword per
        // slot up to it, and an index is clamped to it above.)
        if !sampler && props.0 & 0x1000 == 0 && limit != 0 && view_marks() {
            self.mark_view(block, index, nonuniform);
        }
        self.nonuniform_handles += u32::from(nonuniform);
        let (s, n) = (self.m.bool(sampler), self.m.bool(nonuniform));
        let h = self.op(
            block,
            "createHandleFromHeap",
            Attr::ReadNone,
            self.t.handle,
            218,
            &[index, s, n],
        );
        let h = self.annotate(block, h, props.0, props.1);
        // A constant buffer view (kind 13): its rows begin where the host's
        // table says ([`SLOT_CBV_ROWS`]).
        if !sampler && props.0 == 13 && limit != 0 && cbv_in_place() {
            self.cbv_slots.insert(h, index);
        }
        h
    }

    /// Row `row` of the constant buffer `h` as the host holds it: for a
    /// view picked from the heap, `row` plus the rows its host view begins
    /// early by ([`SLOT_CBV_ROWS`]); any other row as it is.
    pub fn cbuffer_row(&mut self, block: usize, h: V, row: V) -> V {
        let Some(&slot) = self.cbv_slots.get(&h) else {
            return row;
        };
        let rows = match self.cbv_rows.get(&(block, slot)) {
            Some(&r) => r,
            None => {
                let table = self.root_srv_handle(SLOT_CBV_ROWS);
                let two = self.m.i32(2);
                let byte = self.m.bin(block, Bin::Shl, slot, two);
                let i32t = self.t.i32;
                let undef = self.m.undef(i32t);
                let (mask, align) = (self.m.i8(1), self.m.i32(4));
                let res = self.t.res_ret;
                let r = self.op(
                    block,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    res,
                    139,
                    &[table, byte, undef, mask, align],
                );
                let f = self.m.extract(block, r, 0);
                let r = self.m.cast(block, Cast::Bitcast, f, i32t);
                self.cbv_rows.insert((block, slot), r);
                r
            }
        };
        self.m.bin(block, Bin::Add, row, rows)
    }

    /// Stores the mark of the view at heap slot `index` ([`SLOT_MARKS`]),
    /// once per block and index: from the wave's first lane when every lane
    /// has the same index (the others store to the sink, dword 0), from
    /// each lane when not.
    fn mark_view(&mut self, block: usize, index: V, nonuniform: bool) {
        if !self.marked.insert((block, index)) {
            return;
        }
        // Never rasterizer ordered ([`Self::annotate`]): the marks need no
        // order, and an ordered view would make every pixel wait.
        let rov = std::mem::replace(&mut self.rov, false);
        let marks = self.root_uav_handle(SLOT_MARKS);
        self.rov = rov;
        let one = self.m.i32(1);
        let two = self.m.i32(2);
        let slot = self.m.bin(block, Bin::Add, index, one);
        let mut at = self.m.bin(block, Bin::Shl, slot, two);
        if !nonuniform {
            let i1 = self.t.i1;
            let first = self.op(block, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]);
            self.flags |= flags::WAVE_OPS;
            let sink = self.m.i32(0);
            at = self.m.select(block, first, at, sink);
        }
        let i32t = self.t.i32;
        let undef = self.m.undef(i32t);
        let (void, mask, align) = (self.t.void, self.m.i8(1), self.m.i32(4));
        self.op(
            block,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[marks, at, undef, one, undef, undef, undef, mask, align],
        );
        self.marks += 1;
    }

    /// The finished container.
    pub fn finish(mut self) -> Vec<u8> {
        let ud_size = (self.user_data * 4).next_multiple_of(16);
        self.fix_user_data_size(ud_size);
        // The clean-up, before anything is declared about the code: a
        // feature whose last use it removed must not be declared (the
        // validator compares the flags with what the code uses).
        let passes = super::shader_opt() & super::bitcode::opt::MODULE;
        if passes != 0 {
            self.m.optimise(passes);
            LAST_SELF_PHIS.with(|c| c.set(self.m.self_phis()));
            let used = self.m.usage();
            for (flag, used) in [
                (flags::WAVE_OPS, used.wave_ops),
                (flags::ROVS, used.rov),
                (flags::INT64, used.int64),
                (flags::TYPED_UAV_LOADS, used.typed_uav_loads),
                (flags::RESOURCE_HEAP, used.heap_views),
                (flags::SAMPLER_HEAP, used.heap_samplers),
                (flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64, used.atomic64),
            ] {
                if !used {
                    self.flags &= !flag;
                }
            }
        }
        let mut resources = None;
        let mut bindings = Vec::new();
        let i32c = |s: &mut Self, v: u32| {
            let c = s.m.i32(v);
            s.m.md_value(c)
        };
        if self.user_data > 0
            || self.samplers > 0
            || !self.root_cbvs.is_empty()
            || !self.root_srvs.is_empty()
            || !self.root_uavs.is_empty()
        {
            let empty = self.m.md_str("");
            let mut cbvs = None;
            let mut cbv_list = Vec::new();
            if self.user_data > 0 {
                let arr = self.m.array(self.user_data.next_multiple_of(4), self.t.i32);
                let ty = self.m.named_struct("UserData", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, 0)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, USER_DATA_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, ud_size)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: USER_DATA_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            let blob = self.m.array(16384, self.t.i32);
            let ty = self.m.named_struct("RootCbv", vec![blob]);
            let p = self.m.ptr(ty);
            let u = self.m.undef(p);
            for (slot, _) in self.root_cbvs.clone() {
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 65536)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: ROOT_CBV_SPACE,
                    register: u32::from(slot),
                    dxil_kind: 13,
                });
            }
            if !cbv_list.is_empty() {
                cbvs = Some(self.m.md_node(cbv_list));
            }
            let mut srvs = None;
            if !self.root_srvs.is_empty() {
                let ty = self
                    .m
                    .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for (n, (slot, _)) in self.root_srvs.clone().into_iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, ROOT_CBV_SPACE)),
                        Some(i32c(&mut self, u32::from(slot))),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(i32c(&mut self, 0)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 4,
                        space: ROOT_CBV_SPACE,
                        register: u32::from(slot),
                        dxil_kind: 11,
                    });
                }
                srvs = Some(self.m.md_node(list));
            }
            let mut uavs = None;
            if !self.root_uavs.is_empty() {
                let ty = self
                    .m
                    .named_struct("struct.RWByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let no = self.m.bool(false);
                let no = self.m.md_value(no);
                let mut list = Vec::new();
                for (n, (slot, _)) in self.root_uavs.clone().into_iter().enumerate() {
                    // Id, variable, name, space, lower bound, range size,
                    // shape (11: raw buffer), globally coherent, counter,
                    // rasterizer ordered, no extra properties.
                    let ordered = self.m.bool(self.rov);
                    let ordered = self.m.md_value(ordered);
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, ROOT_CBV_SPACE)),
                        Some(i32c(&mut self, u32::from(slot))),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(no),
                        Some(no),
                        Some(ordered),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 7,
                        space: ROOT_CBV_SPACE,
                        register: u32::from(slot),
                        dxil_kind: 11,
                    });
                }
                uavs = Some(self.m.md_node(list));
            }
            let mut samplers = None;
            if self.samplers > 0 {
                let ty = self.m.named_struct("struct.SamplerState", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for n in 0..self.samplers {
                    let kind = (self.cmp_samplers >> n & 1) as u32;
                    let ops = vec![
                        Some(i32c(&mut self, n)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, SAMPLER_SPACE)),
                        Some(i32c(&mut self, n)),
                        Some(i32c(&mut self, 1)),
                        // Sampler kind: 1 comparison.
                        Some(i32c(&mut self, kind)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 1,
                        space: SAMPLER_SPACE,
                        register: n,
                        dxil_kind: 14,
                    });
                }
                samplers = Some(self.m.md_node(list));
            }
            resources = Some(self.m.md_node(vec![srvs, uavs, cbvs, samplers]));
            // PSV0 lists them as the module does: CBVs, samplers, SRVs, UAVs.
            bindings.sort_by_key(|b| match b.kind {
                2 => 0,
                1 => 1,
                7 => 3,
                _ => 2,
            });
        }
        if let Some(r) = resources {
            self.m.named_md("dx.resources", vec![r]);
        }

        let graphics = self.threads.is_none() || self.mesh.is_some();
        // DXIL requires every signature element packed into the same row to
        // share one interpolation mode. Integer / SV_IsFrontFace elements must
        // stay constant (1); otherwise a read element's mode wins over unread
        // siblings in the same row.
        if self.pixel {
            reconcile_row_interp(&mut self.inputs);
        } else if graphics {
            reconcile_row_interp(&mut self.outputs);
        }
        if graphics {
            // shortcut: every output is declared to depend on every input; the
            // exact sets need a dependence pass. Drivers read this state.
            let scalars = |els: &[SigElement]| {
                els.iter()
                    .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                    .max()
                    .unwrap_or(0)
            };
            let (ni, no) = (scalars(&self.inputs), scalars(&self.outputs));
            let mut view = vec![ni, no];
            // Mesh shaders have no inputs: then the primitive outputs' count.
            if self.mesh.is_some() {
                view.push(scalars(&self.prims));
            }
            for _ in 0..ni {
                for k in 0..no.div_ceil(32) {
                    let bits = (no - 32 * k).min(32);
                    view.push(if bits == 32 {
                        u32::MAX
                    } else {
                        (1 << bits) - 1
                    });
                }
            }
            // A geometry shader lists four streams: the other three have
            // no outputs.
            if self.geometry.is_some() {
                view.extend([0, 0, 0]);
            }
            let arr = self.m.array(view.len() as u32, self.t.i32);
            let elems: Vec<V> = view.iter().map(|&v| self.m.i32(v)).collect();
            let arr = self.m.aggregate(arr, &elems);
            let arr = self.m.md_value(arr);
            let view_node = self.m.md_node(vec![Some(arr)]);
            self.m.named_md("dx.viewIdState", vec![view_node]);
        }

        let sigs = if graphics {
            let ins = self.signature_md(self.inputs.clone());
            let outs = self.signature_md(self.outputs.clone());
            let prims = self.signature_md(self.prims.clone());
            Some(self.m.md_node(vec![ins, outs, prims]))
        } else {
            None
        };
        // View marks ([`SLOT_MARKS`]). A pixel shader that writes through an
        // unordered access view runs its depth test after the shader on the
        // host unless it says otherwise: one whose only such writes are the
        // marks, that discards nothing and writes no depth, coverage or
        // stencil, gives the same image with the test before it (as before
        // the marks), and says so. A vertex, geometry or mesh shader with
        // marks declares unordered access views outside pixel and compute.
        if self.marks > 0 {
            if self.pixel {
                let (discard, others) = self.m.writes(i64::from(SLOT_MARKS), i64::from(ROOT_CBV_SPACE));
                let exports = self.outputs.iter().any(|o| matches!(o.kind, 14 | 17..=20));
                if !discard && others == 0 && !exports {
                    self.flags |= flags::EARLY_DEPTH;
                }
            } else if self.threads.is_none() || self.mesh.is_some() {
                // (Vertex, geometry and mesh shaders alike: the validator
                // wants it of every stage but pixel and compute.)
                self.flags |= flags::UAVS_EVERY_STAGE;
            }
        }
        // A pixel shader with an SV_StencilRef output (as DXC collects it).
        if self.pixel && self.outputs.iter().any(|o| o.kind == 20) {
            self.flags |= flags::STENCIL_REF;
        }
        // Entry properties: shader flags (tag 0), geometry state (1),
        // thread-group size (4), wave size (11).
        let mut props = Vec::new();
        if self.flags != 0 {
            props.push(i32c(&mut self, 0));
            let i64t = self.m.int(64);
            let f = self.m.int_const(i64t, self.flags as i64);
            props.push(self.m.md_value(f));
        }
        if let Some(g) = self.geometry {
            // Input primitive (3: triangles), maximum vertices, stream
            // mask, output topology (5: triangle strips), instances.
            props.push(i32c(&mut self, 1));
            let state = [3, g.max_vertices, 1, 5, 1].map(|n| Some(i32c(&mut self, n)));
            props.push(self.m.md_node(state.to_vec()));
        }
        if let Some(threads) = self.threads {
            let t = threads.map(|n| Some(i32c(&mut self, n)));
            let t = self.m.md_node(t.to_vec());
            match self.mesh {
                // Mesh state (tag 9): threads, maximum vertices and
                // primitives, topology, payload bytes.
                Some(mesh) => {
                    props.push(i32c(&mut self, 9));
                    let state = [
                        mesh.max_vertices,
                        mesh.max_primitives,
                        mesh.topology.into(),
                        0,
                    ]
                    .map(|n| Some(i32c(&mut self, n)));
                    let mut node = vec![Some(t)];
                    node.extend(state);
                    props.push(self.m.md_node(node));
                }
                None => {
                    props.push(i32c(&mut self, 4));
                    props.push(t);
                }
            }
        }
        if let Some(n) = self.wave_size {
            props.push(i32c(&mut self, 11));
            let n = Some(i32c(&mut self, n));
            props.push(self.m.md_node(vec![n]));
        }
        let props =
            (!props.is_empty()).then(|| self.m.md_node(props.into_iter().map(Some).collect()));
        let main = self.m.md_value(self.main);
        let name = self.m.md_str("main");
        let entry = self
            .m
            .md_node(vec![Some(main), Some(name), sigs, resources, props]);

        let ident = self.m.md_str("victory");
        let ident = self.m.md_node(vec![Some(ident)]);
        self.m.named_md("llvm.ident", vec![ident]);
        let one = i32c(&mut self, 1);
        let six = i32c(&mut self, 6);
        let version = self.m.md_node(vec![Some(one), Some(six)]);
        self.m.named_md("dx.version", vec![version]);
        self.m.named_md("dx.valver", vec![version]);
        // Program kinds: 0 pixel, 1 vertex, 2 geometry, 5 compute, 13 mesh.
        let (name, kind) = match (self.threads, self.pixel) {
            _ if self.mesh.is_some() => ("ms", 13u32),
            _ if self.geometry.is_some() => ("gs", 2),
            (Some(_), _) => ("cs", 5u32),
            (None, true) => ("ps", 0),
            (None, false) => ("vs", 1),
        };
        let name = self.m.md_str(name);
        let sm = self.m.md_node(vec![Some(name), Some(six), Some(six)]);
        self.m.named_md("dx.shaderModel", vec![sm]);
        self.m.named_md("dx.entryPoints", vec![entry]);

        let bitcode = self.m.write();
        super::bitcode::census::note(&self.m, &bitcode);
        let mut dxil = Vec::with_capacity(24 + bitcode.len());
        dxil.extend((kind << 16 | 0x66).to_le_bytes());
        dxil.extend((((24 + bitcode.len()) / 4) as u32).to_le_bytes());
        dxil.extend(b"DXIL");
        dxil.extend(0x106u32.to_le_bytes());
        dxil.extend(16u32.to_le_bytes());
        dxil.extend((bitcode.len() as u32).to_le_bytes());
        dxil.extend(&bitcode);

        let mut features = 0u64;
        if self.flags & flags::DOUBLES != 0 {
            features |= 0x1;
        }
        if self.flags & flags::UAVS_EVERY_STAGE != 0 {
            features |= 0x4;
        }
        if self.flags & flags::STENCIL_REF != 0 {
            features |= 0x200;
        }
        if self.flags & flags::TYPED_UAV_LOADS != 0 {
            features |= 0x800;
        }
        if self.flags & flags::ROVS != 0 {
            features |= 0x1000;
        }
        if self.flags & flags::WAVE_OPS != 0 {
            features |= 0x4000;
        }
        if self.flags & flags::INT64 != 0 {
            features |= 0x8000;
        }
        if self.flags & flags::BARYCENTRICS != 0 {
            features |= 0x2_0000;
        }
        if self.flags & flags::RESOURCE_HEAP != 0 {
            features |= 0x200_0000;
        }
        if self.flags & flags::SAMPLER_HEAP != 0 {
            features |= 0x400_0000;
        }
        if self.flags & flags::ATOMIC_TYPED64 != 0 {
            features |= 0x40_0000;
        }
        if self.flags & flags::ATOMIC_HEAP64 != 0 {
            features |= 0x1000_0000;
        }
        let sfi0 = features.to_le_bytes();
        let isg1 = container::signature(&self.inputs, true);
        let osg1 = container::signature(&self.outputs, false);
        let flag = if self.pixel {
            self.outputs.iter().any(|o| (17..=19).contains(&o.kind))
        } else {
            self.outputs.iter().any(|o| o.kind == 3)
        };
        let psv0 = container::psv(&Psv {
            stage: kind as u8,
            flag: flag && self.threads.is_none(),
            inputs: &self.inputs,
            outputs: &self.outputs,
            prims: &self.prims,
            mesh: self.mesh.map(|m| (m, self.lds_dwords * 4)),
            geometry: self.geometry.map(|g| g.max_vertices),
            bindings: &bindings,
            threads: self.threads.unwrap_or_default(),
            wave_size: self.wave_size,
        });
        let hash = [0u8; 20];
        let psg1 = container::signature(&self.prims, false);
        let mut parts: Vec<(&[u8; 4], &[u8])> =
            vec![(b"SFI0", &sfi0), (b"ISG1", &isg1), (b"OSG1", &osg1)];
        if self.mesh.is_some() {
            parts.push((b"PSG1", &psg1));
        }
        parts.extend([(b"PSV0", &psv0[..]), (b"HASH", &hash), (b"DXIL", &dxil)]);
        container::build(&parts)
    }

    /// The user-data buffer's annotation carries its size, known only now.
    fn fix_user_data_size(&mut self, size: u32) {
        let old = [self.m.i32(13), self.m.i32(u32::MAX)];
        let old = self.m.aggregate(self.t.props, &old);
        let new = [self.m.i32(13), self.m.i32(size)];
        let new = self.m.aggregate(self.t.props, &new);
        self.m.replace_const(old, new);
    }

    fn signature_md(&mut self, elements: Vec<SigElement>) -> Option<Md> {
        if elements.is_empty() {
            return None;
        }
        let mut list = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let c32 = |s: &mut Self, v: u32| {
                let c = s.m.i32(v);
                Some(s.m.md_value(c))
            };
            let c8 = |s: &mut Self, v: u8| {
                let c = s.m.i8(v);
                Some(s.m.md_value(c))
            };
            let idx = c32(self, e.index);
            let indices = Some(self.m.md_node(vec![idx]));
            let usage = if e.usage != 0 {
                let (tag, mask) = (c32(self, 3), c32(self, u32::from(e.usage)));
                Some(self.m.md_node(vec![tag, mask]))
            } else {
                None
            };
            let ops = vec![
                c32(self, id as u32),
                Some(self.m.md_str(&e.name)),
                c8(self, e.comp),
                c8(self, e.kind),
                indices,
                c8(self, e.interp),
                c32(self, 1),
                c8(self, e.cols),
                c32(self, e.row.unwrap_or(u32::MAX)),
                c8(self, if e.row.is_some() { e.col } else { u8::MAX }),
                usage,
            ];
            list.push(Some(self.m.md_node(ops)));
        }
        Some(self.m.md_node(list))
    }
}

/// Ensures elements sharing a signature row declare one DXIL interpolation
/// mode. Any integer or `SV_IsFrontFace` element forces the row to `constant`
/// (`1`); otherwise, a used element's mode overrides unread siblings.
fn reconcile_row_interp(elems: &mut [SigElement]) {
    let mut row_mode: std::collections::HashMap<u32, (bool, Option<u8>)> =
        std::collections::HashMap::new();
    for e in elems.iter() {
        let Some(row) = e.row else { continue };
        if e.interp == 0 {
            continue;
        }
        let entry = row_mode.entry(row).or_insert((false, None));
        if e.kind == 13 || e.comp != 9 {
            entry.0 = true;
        } else if e.usage != 0 && entry.1.is_none() {
            entry.1 = Some(e.interp);
        }
    }
    for e in elems.iter_mut() {
        let Some(row) = e.row else { continue };
        if e.interp == 0 {
            continue;
        }
        if let Some(&(force_const, used_mode)) = row_mode.get(&row) {
            if force_const {
                e.interp = 1;
            } else if e.usage == 0
                && let Some(m) = used_mode
            {
                e.interp = m;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn elem(name: &str, kind: u8, comp: u8, interp: u8, row: u32, col: u8, usage: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp,
            row: Some(row),
            col,
            cols: 1,
            usage,
            half: false,
        }
    }

    #[test]
    fn reconcile_row_interp_matches_constant_and_used_siblings() {
        let mut elems = vec![
            elem("TEXCOORD", 0, 9, 2, 7, 0, 0),
            elem("SV_IsFrontFace", 13, 5, 1, 7, 1, 1),
            elem("TEXCOORD", 0, 9, 2, 8, 0, 0),
            elem("TEXCOORD", 0, 9, 3, 8, 1, 1),
        ];
        reconcile_row_interp(&mut elems);
        assert_eq!(elems[0].interp, 1);
        assert_eq!(elems[1].interp, 1);
        assert_eq!(elems[2].interp, 3);
        assert_eq!(elems[3].interp, 3);
    }
}

thread_local! {
    static LAST_SELF_PHIS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Diagnostics: the self-referring phis of the last shader this thread
/// finished (see `bitcode::Module::self_phis`).
pub fn last_self_phis() -> usize {
    LAST_SELF_PHIS.with(std::cell::Cell::get)
}
