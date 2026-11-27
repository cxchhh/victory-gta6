//! The run's numbers on the picture, in every build (`VICTORY_PANEL=1`; off
//! by default; F3 or the settings menu shows and hides it while our window
//! has the focus). `VICTORY_PANEL=fps`: only the frame rate, frame time and
//! their graph, in a strip; F4 switches between that and the whole panel
//! while it is shown. Rendered FPS, output FPS (with the frames frame
//! generation adds, as it reports them) and the render frame time lead the
//! panel; the frame-time graph of the last frames marks the 30 FPS budget;
//! GPU, VRAM, CPU and RAM figures follow it (ported from Vice-Tay's panel).
//!
//! Painted again up to four times a second; a frame itself only copies the
//! box out of a buffer in the card's memory onto the swap chain's buffer, in
//! the upper left corner, after the title's picture (and its interface) is
//! in it: saved frames and the frame recorder see the title's own. With
//! DLSS Frame Generation the box is part of the interface's alpha
//! (`crate::streamline::interface_alpha`), and user interface recomposition
//! puts it back over generated frames unprocessed; 2x at 4K.

use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_F3, VK_F4};
use windows_sys::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::font::{colours, glyph};
use super::{Host, HostList, transition};

mod gpu_load;

/// How often the panel is painted again.
const EVERY: Duration = Duration::from_millis(250);

/// A character's dots.
const GLYPH: (u32, u32) = (5, 7);

/// The whole panel's logical size (1x up to 1440p, 2x at 4K).
const BASE_W: u32 = 560;
const BASE_H: u32 = 348;

/// How many recent frames the frame-time graph keeps.
const HISTORY: usize = 172;

/// Mapped buffers the pictures are put into in turn; one is written again
/// once the card has read it.
const UPLOADS: usize = 4;

// Palette (sRGB 0..255).
const BG: [u8; 3] = [24, 20, 36];
const HEADER_BG: [u8; 3] = [37, 28, 48];
const BORDER: [u8; 3] = [82, 69, 96];
const GRAPH_BG: [u8; 3] = [18, 15, 29];
const GRAPH_TARGET: [u8; 3] = [165, 145, 194];
const GRAPH_FILL: [u8; 3] = [73, 57, 87];
const PINK: [u8; 3] = [250, 165, 212];
const LILAC: [u8; 3] = [209, 186, 248];
const BLUE: [u8; 3] = [151, 211, 255];
// Mint is reserved for the measured output FPS digits.
const FG_MINT: [u8; 3] = [134, 255, 170];
const AMBER: [u8; 3] = [255, 194, 106];
const RED: [u8; 3] = [255, 130, 140];
const WHITE: [u8; 3] = [255, 248, 245];
const MUTED: [u8; 3] = [205, 193, 215];
const FRAME_BUDGET_MS: f32 = 1000.0 / 30.0;

/// Pink, lilac and blue mark the frame history from oldest to newest.
fn accent(position: u32, last: u32) -> [u8; 3] {
    let last = last.max(1);
    let t = 2 * position.min(last);
    let (from, to, t) = if t <= last { (PINK, LILAC, t) } else { (LILAC, BLUE, t - last) };
    std::array::from_fn(|i| ((u32::from(from[i]) * (last - t) + u32::from(to[i]) * t) / last) as u8)
}

fn asked() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PANEL"))
}

/// Whether it is shown now.
pub(super) fn shown() -> &'static AtomicBool {
    static SHOWN: OnceLock<AtomicBool> = OnceLock::new();
    SHOWN.get_or_init(|| AtomicBool::new(asked()))
}

/// The settings menu set `VICTORY_PANEL` to `value`: `0` hides the panel,
/// `fps` shows the strip, anything else the whole panel.
pub(super) fn set(value: &str) {
    let value = value.trim();
    shown().store(!value.is_empty() && value != "0", Ordering::Relaxed);
    if !value.is_empty() && value != "0" {
        strip_state().store(value.eq_ignore_ascii_case("fps"), Ordering::Relaxed);
    }
}

/// The key was pressed since the last call (`was` keeps its state), with one
/// of our windows in front: F3 shows and hides the panel, F4 switches it
/// between the strip and the whole.
fn pressed(key: u16, was: &AtomicBool) -> bool {
    // SAFETY: a plain query.
    let down = unsafe { GetAsyncKeyState(i32::from(key)) } < 0;
    if was.swap(down, Ordering::Relaxed) || !down {
        return false;
    }
    let mut process = 0;
    // SAFETY: plain queries; room for the answer.
    unsafe { GetWindowThreadProcessId(GetForegroundWindow(), &mut process) };
    process == std::process::id()
}

/// The strip of `VICTORY_PANEL=fps`: rendered FPS, frame time and their graph.
const STRIP_W: u32 = 382;
const STRIP_H: u32 = 160;

/// Whether the panel is the strip: at first with `VICTORY_PANEL=fps`, then
/// as F4 switches it.
fn strip_state() -> &'static AtomicBool {
    static ON: OnceLock<AtomicBool> = OnceLock::new();
    ON.get_or_init(|| AtomicBool::new(std::env::var("VICTORY_PANEL").is_ok_and(|v| v.trim().eq_ignore_ascii_case("fps"))))
}

fn strip() -> bool {
    strip_state().load(Ordering::Relaxed)
}

/// The box's logical size.
fn base() -> (u32, u32) {
    if strip() { (STRIP_W, STRIP_H) } else { (BASE_W, BASE_H) }
}

/// The box's size in pixels.
fn size(scale: u32) -> (u32, u32) {
    let (w, h) = base();
    (w * scale, h * scale)
}

/// Integer scale for the swap-chain height: 1x up to 1440p, 2x at 4K.
fn scale(height: u32) -> u32 {
    if height >= 1800 { 2 } else { 1 }
}

/// Bytes per pixel for a supported swap-chain format.
fn bpp(format: DXGI_FORMAT) -> Option<usize> {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R10G10B10A2_UNORM => Some(4),
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(8),
        _ => None,
    }
}

/// IEEE 754 binary16 encoding for a non-negative float in `[0, 1]`.
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mant = bits & 0x7f_ffff;
    if exp < -14 {
        return 0;
    }
    let h_exp = ((exp + 15).clamp(0, 31) as u16) << 10;
    let h_mant = (mant >> 13) as u16;
    h_exp | h_mant
}

/// Encodes an RGB colour into `dst` (length == `bpp(format)`).
fn encode(format: DXGI_FORMAT, [r, g, b]: [u8; 3], dst: &mut [u8]) {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => dst[..4].copy_from_slice(&[r, g, b, 0xFF]),
        DXGI_FORMAT_B8G8R8A8_UNORM => dst[..4].copy_from_slice(&[b, g, r, 0xFF]),
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let r10 = (u32::from(r) * 1023 / 255) & 0x3FF;
            let g10 = (u32::from(g) * 1023 / 255) & 0x3FF;
            let b10 = (u32::from(b) * 1023 / 255) & 0x3FF;
            let word = r10 | (g10 << 10) | (b10 << 20) | (0x3 << 30);
            dst[..4].copy_from_slice(&word.to_le_bytes());
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT => {
            let to_h = |c: u8| {
                let s = f32::from(c) / 255.0;
                // Linearize roughly for HDR/scRGB swap chains.
                f32_to_f16(s * s)
            };
            dst[0..2].copy_from_slice(&to_h(r).to_le_bytes());
            dst[2..4].copy_from_slice(&to_h(g).to_le_bytes());
            dst[4..6].copy_from_slice(&to_h(b).to_le_bytes());
            dst[6..8].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
        _ => {}
    }
}

/// Simple RGB software canvas in logical coordinates (`BASE_W x BASE_H`).
struct Canvas {
    w: usize,
    h: usize,
    rgb: Vec<[u8; 3]>,
}

impl Canvas {
    fn new(w: u32, h: u32, fill: [u8; 3]) -> Canvas {
        Canvas { w: w as usize, h: h as usize, rgb: vec![fill; (w * h) as usize] }
    }

    fn rect(&mut self, x: u32, y: u32, w: u32, h: u32, color: [u8; 3]) {
        let x1 = (x + w).min(self.w as u32) as usize;
        let y1 = (y + h).min(self.h as u32) as usize;
        for yy in (y as usize).min(y1)..y1 {
            self.rgb[yy * self.w + (x as usize).min(x1)..yy * self.w + x1].fill(color);
        }
    }

    fn border(&mut self, x: u32, y: u32, w: u32, h: u32, color: [u8; 3]) {
        if w == 0 || h == 0 {
            return;
        }
        self.rect(x, y, w, 1, color);
        self.rect(x, y + h - 1, w, 1, color);
        self.rect(x, y, 1, h, color);
        self.rect(x + w - 1, y, 1, h, color);
    }

    /// Body text uses square pixels; headings are taller, tighter and slightly slanted.
    fn lettering(&mut self, mut x: u32, y: u32, (sx, sy): (u32, u32), color: [u8; 3], text: &str, display: bool) -> u32 {
        let gap = if display { (sx / 2).max(1) } else { sx };
        for ch in text.bytes() {
            for (row, mask) in glyph(ch.to_ascii_uppercase()).into_iter().enumerate() {
                let row = row as u32;
                let lean = if display { (GLYPH.1 - 1 - row) * sx / GLYPH.1 } else { 0 };
                for col in 0..GLYPH.0 {
                    if mask & (1 << (GLYPH.0 - 1 - col)) != 0 {
                        self.rect(x + lean + col * sx, y + row * sy, sx, sy, color);
                    }
                }
            }
            x += GLYPH.0 * sx + gap;
        }
        x
    }

    fn text(&mut self, x: u32, y: u32, px: u32, color: [u8; 3], text: &str) -> u32 {
        self.lettering(x, y, (px, px), color, text, false)
    }

    fn heading(&mut self, x: u32, y: u32, color: [u8; 3], text: &str) {
        self.lettering(x, y, (2, 3), color, text, true);
    }

    fn text_right(&mut self, right: u32, y: u32, px: u32, color: [u8; 3], text: &str) -> u32 {
        let left = right.saturating_sub(text.len() as u32 * (GLYPH.0 + 1) * px);
        self.text(left, y, px, color, text);
        left
    }

    /// Packs the logical canvas scaled by `scale` into swap-chain bytes.
    fn pack(&self, scale: u32, format: DXGI_FORMAT, pitch: usize) -> Vec<u8> {
        let bytes = bpp(format).unwrap_or(4);
        let s = scale as usize;
        let out_h = self.h * s;
        let mut out = vec![0u8; pitch * out_h];
        let mut enc = [0u8; 8];
        for y in 0..self.h {
            let top = (y * s) * pitch;
            for x in 0..self.w {
                encode(format, self.rgb[y * self.w + x], &mut enc[..bytes]);
                for i in 0..s {
                    let at = top + (x * s + i) * bytes;
                    out[at..at + bytes].copy_from_slice(&enc[..bytes]);
                }
            }
            for i in 1..s {
                out.copy_within(top..top + pitch, top + i * pitch);
            }
        }
        out
    }
}

/// Private committed bytes of the current process.
fn process_private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// CPU brand name from CPUID (`0x80000002..=0x80000004`), trimmed.
fn cpu_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            let mut raw = [0u8; 48];
            for (i, leaf) in (0x8000_0002u32..=0x8000_0004).enumerate() {
                let r = std::arch::x86_64::__cpuid(leaf);
                raw[i * 16..i * 16 + 4].copy_from_slice(&r.eax.to_le_bytes());
                raw[i * 16 + 4..i * 16 + 8].copy_from_slice(&r.ebx.to_le_bytes());
                raw[i * 16 + 8..i * 16 + 12].copy_from_slice(&r.ecx.to_le_bytes());
                raw[i * 16 + 12..i * 16 + 16].copy_from_slice(&r.edx.to_le_bytes());
            }
            let s = String::from_utf8_lossy(&raw);
            let clean = s
                .trim_matches(|c: char| c == '\0' || c.is_whitespace())
                .replace("8-Core Processor", "")
                .replace("16-Core Processor", "")
                .replace("12-Core Processor", "")
                .replace("6-Core Processor", "")
                .replace("Processor", "");
            clean.split_whitespace().collect::<Vec<_>>().join(" ")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            "CPU".to_owned()
        }
    })
}

/// Adapter name and dedicated VRAM size from DXGI, queried once.
fn gpu_info(host: &Host) -> &'static (String, u64) {
    static INFO: OnceLock<(String, u64)> = OnceLock::new();
    INFO.get_or_init(|| {
        let Some(adapter) = host.adapter.as_ref() else {
            return ("GPU".to_owned(), 0);
        };
        // SAFETY: plain DXGI descriptor query.
        let Ok(desc) = (unsafe { adapter.GetDesc1() }) else {
            return ("GPU".to_owned(), 0);
        };
        let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        let name = String::from_utf16_lossy(&desc.Description[..end]).trim().to_owned();
        (if name.is_empty() { "GPU".to_owned() } else { name }, desc.DedicatedVideoMemory as u64)
    })
}

/// System RAM: `(used_phys, total_phys, used_commit, total_commit)` in bytes.
fn system_memory() -> (u64, u64, u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain sized query.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return (0, 0, 0, 0);
    }
    (
        m.ullTotalPhys.saturating_sub(m.ullAvailPhys),
        m.ullTotalPhys,
        m.ullTotalPageFile.saturating_sub(m.ullAvailPageFile),
        m.ullTotalPageFile,
    )
}

/// Per-logical-CPU utilization tracker via `NtQuerySystemInformation(8)`.
#[derive(Default)]
struct CpuMeter {
    prev: Vec<(i64, i64)>,
    total_pct: f32,
    cores_pct: Vec<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SysProcPerfInfo {
    idle: i64,
    kernel: i64,
    user: i64,
    dpc: i64,
    interrupt: i64,
    interrupt_count: u32,
    _pad: u32,
}

impl CpuMeter {
    fn sample(&mut self) {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        type NtQuery = unsafe extern "system" fn(u32, *mut u8, u32, *mut u32) -> i32;
        static FN: OnceLock<Option<NtQuery>> = OnceLock::new();
        let query = *FN.get_or_init(|| unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            if ntdll.is_null() {
                return None;
            }
            let p = GetProcAddress(ntdll, c"NtQuerySystemInformation".as_ptr().cast())?;
            Some(std::mem::transmute::<unsafe extern "system" fn() -> isize, NtQuery>(p))
        });
        let Some(query) = query else { return };
        let n = std::thread::available_parallelism().map_or(8, |n| n.get()).min(64);
        let mut buf = vec![SysProcPerfInfo::default(); n];
        let mut ret = 0u32;
        // SystemProcessorPerformanceInformation = 8.
        let status = unsafe {
            query(8, buf.as_mut_ptr().cast(), (buf.len() * size_of::<SysProcPerfInfo>()) as u32, &mut ret)
        };
        if status < 0 {
            return;
        }
        let count = (ret as usize / size_of::<SysProcPerfInfo>()).min(buf.len());
        if count == 0 {
            return;
        }
        let now: Vec<(i64, i64)> = buf[..count].iter().map(|p| (p.idle, p.kernel + p.user)).collect();
        if self.prev.len() == count {
            self.cores_pct.clear();
            let mut sum = 0.0f32;
            for (&(pi, pt), &(ni, nt)) in self.prev.iter().zip(&now) {
                let dt = (nt - pt).max(1);
                let di = (ni - pi).clamp(0, dt);
                let busy = ((1.0 - di as f32 / dt as f32) * 100.0).clamp(0.0, 100.0);
                self.cores_pct.push(busy);
                sum += busy;
            }
            self.total_pct = sum / count as f32;
        }
        self.prev = now;
    }
}

/// The counters read for it. They grow, and go back to 0 with the frame-rate
/// line (right behind a look of ours).
#[derive(Clone, Copy, Default)]
struct Counters([u64; COUNTERS]);

const COUNTERS: usize = 11;

impl Counters {
    fn now() -> Counters {
        use super::{CARD_CLOCK_US, COPY_BACK, PREPARING};
        let get = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        let record_ns = crate::copystat::peek(crate::copystat::C::RecordNs);
        #[cfg(not(feature = "diagnostics"))]
        let record_ns = 0;
        Counters([
            super::worked(),
            get(&CARD_CLOCK_US[0]),
            get(&CARD_CLOCK_US[1]),
            get(&CARD_CLOCK_US[2]),
            get(&PREPARING[3]),
            get(&COPY_BACK[0]),
            get(&COPY_BACK[1]),
            get(&crate::shadow::CARD_STATS[0]),
            crate::draw::dropped_now(),
            record_ns,
            crate::draw::drawn(),
        ])
    }

    /// What came since `before`, per frame.
    fn since(self, before: Counters, frames: f64) -> [f64; COUNTERS] {
        let mut out = [0.0; COUNTERS];
        for (i, out) in out.iter_mut().enumerate() {
            let (was, is) = (before.0[i], self.0[i]);
            *out = if is >= was { is - was } else { is } as f64 / frames;
        }
        out
    }
}

/// What the panel shows, taken at a repaint.
struct Snapshot {
    fps: f64,
    /// Rendered and output frames a second, both measured over the same
    /// interval, while frame generation showed more frames than were rendered.
    frame_rates: Option<(f32, f32)>,
    frame_ms: f64,
    gpu_ms: f64,
    gpu_pct: u32,
    gpu_measured: bool,
    submit_ms: f64,
    low1_fps: f64,
    max_ms: f64,
    draws: u64,
    dropped: u64,
    width: u32,
    height: u32,
    gpu_name: String,
    vram_used_gb: f64,
    vram_budget_gb: f64,
    vram_shared_gb: f64,
    vram_rt_gb: f64,
    vram_tex_gb: f64,
    vram_tiles_gb: f64,
    mb_up: f64,
    cpu_name: String,
    cpu_pct: f32,
    cores_pct: Vec<f32>,
    proc_ram_gb: f64,
    sys_ram_used_gb: f64,
    sys_ram_total_gb: f64,
    sys_commit_gb: f64,
    history_ms: Vec<f32>,
}

/// Fits the value within one headline column even during loading-screen FPS spikes.
fn headline(c: &mut Canvas, x: u32, value: f64, color: [u8; 3]) {
    let digits = if value < 100.0 { format!("{value:.1}") } else { format!("{value:.0}") };
    let px = (174 / (digits.len() as u32 * (GLYPH.0 + 1))).clamp(2, 5);
    c.lettering(x, 68 - GLYPH.1 * px, (px.saturating_sub(1).max(2), px), color, &digits, true);
}

fn paint_history(c: &mut Canvas, history: &[f32]) {
    let (x, y, w, h) = (14, 100, 320, 52);
    let panel_bottom = y + h - 1;
    c.rect(x, y, w, h, GRAPH_BG);
    let peak = history.iter().copied().fold(0.0f32, f32::max);
    let ceiling = if peak <= FRAME_BUDGET_MS * 2.0 {
        FRAME_BUDGET_MS * 2.0
    } else if peak <= FRAME_BUDGET_MS * 4.0 {
        FRAME_BUDGET_MS * 4.0
    } else {
        peak.min(1000.0)
    };
    let to_height = |ms: f32| ((ms / ceiling).clamp(0.0, 1.0) * (h - 2) as f32).round() as u32;

    // Fixed time order: the newest frame is at the right edge, even at startup.
    let n = history.len().min(HISTORY);
    for (i, &ms) in history[history.len() - n..].iter().enumerate() {
        let slot = (HISTORY - n + i) as u32;
        let left = x + slot * w / HISTORY as u32;
        let right = x + (slot + 1) * w / HISTORY as u32;
        let height = to_height(ms).max(1);
        let top = panel_bottom - height;
        c.rect(left, top, right - left, height, GRAPH_FILL);
        let color = if ms <= FRAME_BUDGET_MS {
            accent(slot, HISTORY as u32 - 1)
        } else if ms <= FRAME_BUDGET_MS * 2.0 {
            AMBER
        } else {
            RED
        };
        c.rect(left, top, right - left, height.min(2), color);
    }
    let target = panel_bottom - to_height(FRAME_BUDGET_MS);
    for dx in (0..w).step_by(8) {
        c.rect(x + dx, target, 4, 1, GRAPH_TARGET);
    }
}

fn device_row(c: &mut Canvas, y: u32, label: &str, name: &str, load: &str, color: [u8; 3]) {
    c.heading(14, y.saturating_sub(2), color, label);
    let value_x = c.text_right(BASE_W - 12, y, 2, color, load);
    let chars = (value_x.saturating_sub(76) / 12) as usize;
    // The font is ASCII. Skip trademark symbols and shorten model names on character boundaries.
    let name: String = name.chars().filter(char::is_ascii).take(chars).collect();
    c.text(66, y, 2, WHITE, &name);
}

/// A stepped outline fits the rectangular copy without covering the readouts.
fn panel_frame(c: &mut Canvas) {
    const CUT: u32 = 8;
    let (w, h) = (c.w as u32, c.h as u32);
    c.rect(CUT, 0, w - 2 * CUT, 2, BORDER);
    c.rect(CUT, h - 2, w - 2 * CUT, 2, BORDER);
    c.rect(0, CUT, 2, h - 2 * CUT, BORDER);
    c.rect(w - 2, CUT, 2, h - 2 * CUT, BORDER);
    for y in (0..CUT).step_by(2) {
        let x = CUT - y - 2;
        for left in [x, w - x - 4] {
            c.rect(left, y, 4, 2, BORDER);
            c.rect(left, h - y - 2, 4, 2, BORDER);
        }
    }
}

/// `VICTORY_PANEL=fps`: the FPS shown (with frame generation the output,
/// else the rendered) with the 1% low, and the frame time with its longest,
/// over the frame-time graph; with frame generation the rendered FPS right
/// of the graph.
fn paint_strip(s: &Snapshot, framegen: bool) -> Canvas {
    let mut c = Canvas::new(STRIP_W, STRIP_H, BG);
    c.rect(8, 2, STRIP_W - 16, 90, HEADER_BG);
    c.rect(2, 8, STRIP_W - 4, 84, HEADER_BG);
    for x in (12..STRIP_W - 12).step_by(4) {
        let w = (STRIP_W - 12 - x).min(4);
        c.rect(x, 3, w, 2, accent(x - 12, STRIP_W - 25));
    }
    panel_frame(&mut c);
    let rates = s.frame_rates.filter(|_| framegen);
    // With frame generation the big number is what is shown (the output),
    // the rendered rate small right of the graph.
    let (fps, label) = match rates {
        Some((_, output)) => (f64::from(output), "FG FPS"),
        None => (s.fps, "RENDER FPS"),
    };
    c.heading(14, 8, WHITE, label);
    c.heading(198, 8, WHITE, "FRAME MS");
    let pace = if fps >= 30.0 { PINK } else if fps >= 15.0 { AMBER } else { RED };
    headline(&mut c, 14, fps, if rates.is_some() { FG_MINT } else { pace });
    headline(&mut c, 198, s.frame_ms, WHITE);
    c.text(14, 72, 2, MUTED, &format!("1% LOW {:.0}", s.low1_fps));
    c.text(198, 72, 2, MUTED, &format!("MAX {:.0} MS", s.max_ms));
    paint_history(&mut c, &s.history_ms);
    if let Some((rendered, _)) = rates {
        c.text(342, 104, 1, MUTED, "REN");
        c.text(342, 116, 1, WHITE, &format!("{rendered:.0}"));
        c.text(342, 128, 1, MUTED, "FPS");
    }
    c
}

/// The raised pair of outlines marks observed generation; a single outline is the idle state.
fn output_card(c: &mut Canvas, measured: bool) {
    c.rect(188, 6, 178, 76, BG);
    c.rect(194, 12, 178, 76, BG);
    c.rect(188, 4, 178, 2, BORDER);
    c.rect(366, 6, 3, 3, BORDER);
    c.rect(369, 9, 3, 3, BORDER);
    c.rect(372, 12, 2, 76, BORDER);
    c.rect(194, 88, 180, 2, BORDER);
    c.rect(191, 85, 3, 3, BORDER);
    c.rect(188, 82, 3, 3, BORDER);
    c.rect(186, 6, 2, 76, BORDER);

    c.border(347, 13, 12, 9, MUTED);
    if measured {
        c.rect(352, 17, 13, 10, BG);
        c.border(352, 17, 13, 10, MUTED);
    }
}

/// `framegen`: the presents go through a frame generation swap chain.
fn paint_hud(s: &Snapshot, framegen: bool) -> Canvas {
    if strip() {
        return paint_strip(s, framegen);
    }
    let mut c = Canvas::new(BASE_W, BASE_H, BG);
    let right = BASE_W - 12;
    c.rect(8, 2, BASE_W - 16, 90, HEADER_BG);
    c.rect(2, 8, BASE_W - 4, 84, HEADER_BG);
    for x in (12..BASE_W - 12).step_by(4) {
        let w = (BASE_W - 12 - x).min(4);
        c.rect(x, 3, w, 2, accent(x - 12, BASE_W - 25));
    }
    panel_frame(&mut c);

    let rates = s.frame_rates.filter(|_| framegen);
    output_card(&mut c, rates.is_some());
    let fps = rates.map_or(s.fps, |(rendered, _)| f64::from(rendered));
    c.heading(14, 8, WHITE, "RENDER FPS");
    if rates.is_some() {
        c.heading(198, 8, WHITE, "FG OUTPUT FPS");
    } else {
        c.heading(198, 8, MUTED, "OUTPUT FPS");
    }
    c.heading(382, 8, WHITE, "FRAME MS");
    let pace = if fps >= 30.0 { PINK } else if fps >= 15.0 { AMBER } else { RED };
    headline(&mut c, 14, fps, pace);
    if let Some((_, output)) = rates {
        headline(&mut c, 198, f64::from(output), FG_MINT);
    } else {
        c.text(198, 40, 2, MUTED, if framegen { "FG NOT ON" } else { "RENDER ONLY" });
    }
    headline(&mut c, 382, s.frame_ms, WHITE);
    c.text(14, 72, 2, MUTED, &format!("1% LOW {:.0}", s.low1_fps));
    let size_end = c.text(198, 72, 2, WHITE, &format!("{}X{}", s.width, s.height));
    // Output divided by rendered FPS, both measured over the same interval.
    if let Some((rendered, output)) = rates
        && rendered > 0.0
    {
        let ratio = format!("{:.1}X", output / rendered);
        if size_end + 6 + ratio.len() as u32 * 12 <= 366 {
            c.text_right(366, 72, 2, MUTED, &ratio);
        }
    }
    c.text(382, 72, 2, MUTED, &format!("MAX {:.0} MS", s.max_ms));

    paint_history(&mut c, &s.history_ms);
    c.text(350, 102, 2, MUTED, "SUBMIT");
    c.text_right(right, 102, 2, WHITE, &format!("{:.1} MS", s.submit_ms));
    c.text(350, 132, 2, MUTED, "GPU EST");
    c.text_right(right, 132, 2, WHITE, &format!("{:.1} MS", s.gpu_ms));
    c.text(14, 161, 2, MUTED, "33.3 MS TARGET");
    c.text_right(right, 161, 2, MUTED, &format!("{} DRAWS / {} DROPPED", s.draws, s.dropped));
    c.rect(14, 183, BASE_W - 28, 1, BORDER);

    let gpu_source = if s.gpu_measured { "APP" } else { "EST" };
    device_row(&mut c, 192, "GPU", &s.gpu_name, &format!("{gpu_source} {}%", s.gpu_pct), BLUE);
    c.heading(14, 212, BLUE, "VRAM");
    let local = if s.vram_budget_gb > 0.0 {
        format!("{:.1}/{:.1} GIB", s.vram_used_gb, s.vram_budget_gb)
    } else {
        "N/A".to_owned()
    };
    let vram_color = if s.vram_budget_gb > 0.0 && s.vram_used_gb >= 0.92 * s.vram_budget_gb { PINK } else { WHITE };
    c.text(78, 214, 2, vram_color, &local);
    c.text_right(right, 214, 2, MUTED, &format!("SHARED {:.1} GIB", s.vram_shared_gb));
    // Host textures: render targets (made by us) and textures the title
    // fills, tiles given to tiled textures; MB uploaded a frame.
    for (x, text) in [
        (14, format!("RT {:.1}", s.vram_rt_gb)),
        (120, format!("TEX {:.1}", s.vram_tex_gb)),
        (238, format!("TILES {:.1}", s.vram_tiles_gb)),
        (380, format!("UP {:.0} MB", s.mb_up)),
    ] {
        c.text(x, 236, 2, MUTED, &text);
    }

    let cpu_load = if s.cores_pct.is_empty() { "NO SAMPLE".to_owned() } else { format!("SYSTEM {:.0}%", s.cpu_pct) };
    device_row(&mut c, 263, "CPU", &s.cpu_name, &cpu_load, LILAC);
    let width = BASE_W - 28;
    for (i, &pct) in s.cores_pct.iter().enumerate() {
        let left = 14 + i as u32 * width / s.cores_pct.len() as u32;
        let right = 14 + (i + 1) as u32 * width / s.cores_pct.len() as u32;
        let w = (right - left).saturating_sub(2);
        let fill = (pct.clamp(0.0, 100.0) * 0.14).round() as u32;
        c.rect(left, 282, w, 14, GRAPH_FILL);
        c.rect(left, 296 - fill, w, fill, accent(i as u32, s.cores_pct.len().saturating_sub(1) as u32));
    }

    c.heading(14, 303, LILAC, "RAM");
    c.text(66, 305, 2, WHITE, &format!("PRIVATE {:.1} GIB", s.proc_ram_gb));
    c.text_right(right, 305, 2, MUTED, &format!("COMMIT {:.1} GIB", s.sys_commit_gb));
    c.text(66, 327, 2, MUTED, &format!("SYSTEM {:.1}/{:.1} GIB", s.sys_ram_used_gb, s.sys_ram_total_gb));
    let mem_width = 158;
    c.rect(right - mem_width, 332, mem_width, 4, GRAPH_FILL);
    if s.sys_ram_total_gb > 0.0 {
        let used = (s.sys_ram_used_gb / s.sys_ram_total_gb).clamp(0.0, 1.0);
        let fill = (used * f64::from(mem_width)).round() as u32;
        c.rect(right - mem_width, 332, fill, 4, if used >= 0.92 { PINK } else { LILAC });
    }
    c
}

/// A card buffer of the panel's picture, of that format and scale, and the
/// mapped uploads it is filled from.
struct HudBuffers {
    format: DXGI_FORMAT,
    scale: u32,
    pitch: u32,
    uploads: Vec<(ID3D12Resource, *mut u8)>,
    card: ID3D12Resource,
    /// The upload written last.
    turn: usize,
    /// The host fence's value covering the last read of each upload
    /// (`u64::MAX`: read by a list not sent yet when it was written).
    upload_fences: [u64; UPLOADS],
    /// That upload: its value is the one the host signalled last, at the
    /// next present.
    pending: Option<usize>,
}

// SAFETY: free-threaded resources, and memory only written under the presenter's lock.
unsafe impl Send for HudBuffers {}

impl HudBuffers {
    fn new(host: &Host, format: DXGI_FORMAT, scale: u32) -> Option<HudBuffers> {
        // The whole panel's: the strip (F4) is painted and copied in its
        // top left corner.
        let (width, height) = (BASE_W * scale, BASE_H * scale);
        let pitch = (width * bpp(format)? as u32).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(height);
        let uploads = (0..UPLOADS).map(|_| host.upload_buffer(bytes)).collect::<Option<Vec<_>>>()?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut card)
                .ok()?;
        }
        Some(HudBuffers {
            format,
            scale,
            pitch,
            uploads,
            card: card?,
            turn: 0,
            upload_fences: [0; UPLOADS],
            pending: None,
        })
    }

    /// The list that read the upload written last went with the last
    /// present: the value the host signalled last covers it. (Not waited
    /// for: with a submission going on, at a later present.)
    fn settle(&mut self, host: &Host) {
        let Some(slot) = self.pending else { return };
        let last = match host.submitted.try_lock() {
            Ok(last) => *last,
            Err(std::sync::TryLockError::Poisoned(e)) => *e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        };
        self.upload_fences[slot] = last;
        self.pending = None;
    }

    /// Paints the picture into a free upload, if one is; its slot.
    fn write(&mut self, snap: &Snapshot, completed: u64, framegen: bool) -> Option<usize> {
        let slot = free_upload(&self.upload_fences, self.turn, completed)?;
        let picture = paint_hud(snap, framegen).pack(self.scale, self.format, self.pitch as usize);
        self.turn = slot;
        // SAFETY: the mapping holds the picture and its previous read by the card has finished.
        unsafe { std::ptr::copy_nonoverlapping(picture.as_ptr(), self.uploads[slot].1, picture.len()) };
        self.upload_fences[slot] = u64::MAX;
        self.pending = Some(slot);
        Some(slot)
    }
}

fn free_upload(fences: &[u64; UPLOADS], turn: usize, completed: u64) -> Option<usize> {
    (1..=UPLOADS).map(|offset| (turn + offset) % UPLOADS).find(|&slot| fences[slot] <= completed)
}

/// The panel of a presenter.
pub(super) struct Panel {
    since: Instant,
    frames: u32,
    /// Frames shown since `since`, with those frame generation added.
    shown: u64,
    last: Instant,
    worst: f64,
    before: Counters,
    history: VecDeque<f32>,
    cpu: CpuMeter,
    last_size: (u32, u32),
    snap: Option<Snapshot>,
    changed: bool,
    buffers: Option<HudBuffers>,
    /// The card buffer holds a picture.
    uploaded: bool,
    failed: bool,
}

impl Panel {
    pub(super) fn new() -> Panel {
        let now = Instant::now();
        Panel {
            since: now,
            frames: 0,
            shown: 0,
            last: now,
            worst: 0.0,
            before: Counters::default(),
            history: VecDeque::with_capacity(HISTORY),
            cpu: CpuMeter::default(),
            last_size: (0, 0),
            snap: None,
            changed: false,
            buffers: None,
            uploaded: false,
            failed: false,
        }
    }

    /// A frame was presented; `line`: the counters go back to 0 right behind
    /// this call. (Not shown it asks for the key's state and no more.)
    pub(super) fn frame(&mut self, line: bool) {
        static F3: AtomicBool = AtomicBool::new(false);
        static F4: AtomicBool = AtomicBool::new(false);
        if pressed(VK_F3, &F3) {
            shown().fetch_xor(true, Ordering::Relaxed);
        }
        if !shown().load(Ordering::Relaxed) {
            self.frames = 0;
            return;
        }
        if pressed(VK_F4, &F4) {
            strip_state().fetch_xor(true, Ordering::Relaxed);
            self.changed = true;
        }
        let now = Instant::now();
        let dt_ms = (now - self.last).as_secs_f64() * 1e3;
        if self.frames == 0 && now - self.last > 2 * EVERY {
            // (Shown from here on.)
            (self.since, self.last, self.worst, self.before, self.shown) = (now, now, 0.0, Counters::now(), 0);
            self.cpu.sample();
        } else {
            if self.history.len() == HISTORY {
                self.history.pop_front();
            }
            self.history.push_back(dt_ms as f32);
            self.worst = self.worst.max(dt_ms);
            self.last = now;
        }
        self.frames += 1;
        // What frame generation reported shown for this present (1 without).
        self.shown += u64::from(crate::streamline::shown_per_frame());
        let elapsed = now - self.since;
        if elapsed < EVERY && !(line && elapsed * 2 >= EVERY) {
            return;
        }
        let Some(host) = super::host() else { return };
        self.cpu.sample();
        let frames = f64::from(self.frames);
        let counters = Counters::now();
        let c = counters.since(self.before, frames);
        self.before = counters;

        let ms = elapsed.as_secs_f64() * 1e3 / frames;
        let fps = 1e3 / ms.max(0.001);
        let output = self.shown as f64 / elapsed.as_secs_f64().max(1e-6);
        let frame_rates = (self.shown > u64::from(self.frames)).then_some((fps as f32, output as f32));
        let idle = (c[1] / 1e3).min(ms);
        let gpu_ms = (ms - idle).max(0.0);
        let measured = gpu_load::latest(&host.device);
        let gpu_pct = measured.unwrap_or_else(|| ((gpu_ms / ms.max(0.001)) * 100.0).clamp(0.0, 100.0).round() as u32);

        // 1% low FPS from recent history.
        let low1_fps = if self.history.is_empty() {
            fps
        } else {
            let mut sorted: Vec<f32> = self.history.iter().copied().collect();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let idx = (sorted.len() * 99 / 100).min(sorted.len() - 1);
            1e3 / f64::from(sorted[idx]).max(0.001)
        };

        let gb = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        let private = process_private_bytes();
        let [budget, used, _, shared] = host.video_memory().unwrap_or_default();
        let all_tex = crate::resource::HOST_TEXTURES[1].load(Ordering::Relaxed);
        let filled_tex = crate::resource::KINDS[1].load(Ordering::Relaxed);
        let (sys_used, sys_total, sys_commit, _) = system_memory();
        let (gpu_name, _) = gpu_info(host);

        self.snap = Some(Snapshot {
            fps,
            frame_rates,
            frame_ms: ms,
            gpu_ms,
            gpu_pct,
            gpu_measured: measured.is_some(),
            submit_ms: c[0] / 1e3,
            low1_fps,
            max_ms: self.worst,
            draws: c[10].round() as u64,
            dropped: c[8].round() as u64,
            width: self.last_size.0,
            height: self.last_size.1,
            gpu_name: gpu_name.clone(),
            vram_used_gb: gb(used),
            vram_budget_gb: gb(budget),
            vram_shared_gb: gb(shared),
            vram_rt_gb: gb(all_tex.saturating_sub(filled_tex)),
            vram_tex_gb: gb(filled_tex),
            vram_tiles_gb: gb(crate::tiles::given_total()),
            mb_up: c[7] / 1e6,
            cpu_name: cpu_name().to_owned(),
            cpu_pct: self.cpu.total_pct,
            cores_pct: self.cpu.cores_pct.clone(),
            proc_ram_gb: gb(private),
            sys_ram_used_gb: gb(sys_used),
            sys_ram_total_gb: gb(sys_total),
            sys_commit_gb: gb(sys_commit),
            history_ms: self.history.iter().copied().collect(),
        });
        self.changed = true;
        (self.since, self.frames, self.shown, self.worst) = (now, 0, 0, 0.0);
    }

    /// Records the copy of the box onto `back`, a buffer of the swap chain of
    /// that size and format in the state to be copied into; where the box
    /// lies, if it is copied (DLSS Frame Generation takes it for interface:
    /// `crate::streamline::interface_alpha`).
    pub(super) fn draw(
        &mut self,
        host: &Host,
        list: &HostList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
    ) -> Option<RECT> {
        self.last_size = (width, height);
        if !shown().load(Ordering::Relaxed) || self.failed {
            return None;
        }
        let snap = self.snap.as_ref()?;
        // Not the title's: the frame recorder leaves it out.
        #[cfg(feature = "diagnostics")]
        let _ours = crate::replay::record::unrecorded();
        let scale = scale(height);
        if self.buffers.as_ref().is_none_or(|b| (b.format, b.scale) != (format, scale)) {
            // (What the card may still read stays: a picture's size changes once or twice in a run.)
            std::mem::forget(self.buffers.take());
            self.buffers = HudBuffers::new(host, format, scale);
            self.failed = self.buffers.is_none();
            self.uploaded = false;
            self.changed = true;
        }
        let buffers = self.buffers.as_mut()?;
        buffers.settle(host);
        let upload = if self.changed {
            // SAFETY: a plain query of the queue's fence.
            let completed = unsafe { host.fence.GetCompletedValue() };
            if completed == u64::MAX {
                // The device is gone: nothing of it is used again.
                self.failed = true;
                return None;
            }
            let framegen = crate::streamline::framegen_kind().is_some();
            buffers.write(snap, completed, framegen)
        } else {
            None
        };
        if upload.is_some() {
            self.changed = false;
        }
        if upload.is_none() && !self.uploaded {
            return None;
        }
        let (w, h) = size(scale);
        let list = &list.list;
        // SAFETY: an open list and live resources; the box lies within both.
        unsafe {
            if let Some(slot) = upload {
                // (A buffer is in the common state at a list's start: the
                // copy promotes it.)
                list.CopyResource(&buffers.card, &buffers.uploads[slot].0);
                list.ResourceBarrier(&[transition(&buffers.card, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                self.uploaded = true;
            }
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffers.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: buffers.format,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: buffers.pitch,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let part = D3D12_BOX { left: 0, top: 0, front: 0, right: w.min(width), bottom: h.min(height), back: 1 };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&part));
        }
        Some(RECT { left: 0, top: 0, right: w.min(width) as i32, bottom: h.min(height) as i32 })
    }
}

/// A character's cell, with the space beside and below it, and the rim of
/// a text box, in dots (the start's notice).
const CELL: (u32, u32) = (6, 9);
const RIM: u32 = 3;

/// A text box's characters in a line, and lines.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Shape {
    columns: u32,
    lines: u32,
}

/// A text box's size in pixels, a dot being `scale` of them each way.
const fn text_size(shape: Shape, scale: u32) -> (u32, u32) {
    ((shape.columns * CELL.0 + 2 * RIM) * scale, (shape.lines * CELL.1 + 2 * RIM) * scale)
}

/// Pixels to a dot of a text box for a picture that high: a 720-line
/// screenshot of it shows one and a third.
fn text_scale(height: u32) -> u32 {
    (height / 540).max(1)
}

/// Whether a dot of the picture of `lines` is white.
fn lit(lines: &[String], shape: Shape, x: u32, y: u32) -> bool {
    let (Some(x), Some(y)) = (x.checked_sub(RIM), y.checked_sub(RIM)) else {
        return false;
    };
    if x % CELL.0 >= GLYPH.0 || y % CELL.1 >= GLYPH.1 || x / CELL.0 >= shape.columns {
        return false;
    }
    let Some(&c) = lines.get((y / CELL.1) as usize).and_then(|l| l.as_bytes().get((x / CELL.0) as usize)) else {
        return false;
    };
    (glyph(c.to_ascii_uppercase())[(y % CELL.1) as usize] >> (GLYPH.0 - 1 - x % CELL.0)) & 1 != 0
}

/// The text box as rows of `pitch` bytes: white text on black.
fn painted(lines: &[String], shape: Shape, scale: u32, colours: [&[u8]; 2], pitch: usize) -> Vec<u8> {
    let (width, height) = text_size(shape, scale);
    let bytes = colours[0].len();
    let mut out = vec![0; pitch * height as usize];
    for y in 0..height / scale {
        let top = pitch * (y * scale) as usize;
        for x in 0..width / scale {
            let colour = colours[usize::from(lit(lines, shape, x, y))];
            for i in 0..scale {
                let at = top + (x * scale + i) as usize * bytes;
                out[at..at + bytes].copy_from_slice(colour);
            }
        }
        for i in 1..scale as usize {
            out.copy_within(top..top + pitch, top + i * pitch);
        }
    }
    out
}

/// The buffers of a text box that size and format (written once).
struct TextBuffers {
    shape: Shape,
    format: DXGI_FORMAT,
    scale: u32,
    pitch: u32,
    upload: (ID3D12Resource, *mut u8),
    card: ID3D12Resource,
    /// It is not in the card's buffer yet.
    fresh: bool,
}

// SAFETY: free-threaded resources, and memory only written under the presenter's lock.
unsafe impl Send for TextBuffers {}

impl TextBuffers {
    fn new(host: &Host, shape: Shape, format: DXGI_FORMAT, scale: u32) -> Option<TextBuffers> {
        let (width, height) = text_size(shape, scale);
        let pitch = (width * colours(format)?[0].len() as u32).next_multiple_of(256);
        let bytes = u64::from(pitch) * u64::from(height);
        let upload = host.upload_buffer(bytes)?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut card: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer.
        unsafe {
            host.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut card)
                .ok()?;
        }
        Some(TextBuffers { shape, format, scale, pitch, upload, card: card?, fresh: false })
    }

    /// Puts the text into the upload (once: nothing reads it yet).
    fn write(&mut self, lines: &[String]) {
        let Some(colours) = colours(self.format) else { return };
        let picture = painted(lines, self.shape, self.scale, colours, self.pitch as usize);
        // SAFETY: a mapped upload buffer of the picture's size, not read yet.
        unsafe { std::ptr::copy_nonoverlapping(picture.as_ptr(), self.upload.1, picture.len()) };
        self.fresh = true;
    }

    /// Records the copy of the box onto `back` (of `width` x `height`, in the
    /// state to be copied into), its upper left corner at `at`; where it lies.
    fn copy(&mut self, list: &ID3D12GraphicsCommandList, back: &ID3D12Resource, at: (u32, u32), (width, height): (u32, u32)) -> Option<RECT> {
        let (w, h) = text_size(self.shape, self.scale);
        if at.0 >= width || at.1 >= height {
            return None;
        }
        // SAFETY: an open list and live resources; the box lies within both.
        unsafe {
            if std::mem::take(&mut self.fresh) {
                list.CopyResource(&self.card, &self.upload.0);
                list.ResourceBarrier(&[transition(&self.card, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            }
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&self.card),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: self.format,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: self.pitch,
                        },
                    },
                },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let part = D3D12_BOX { left: 0, top: 0, front: 0, right: w.min(width - at.0), bottom: h.min(height - at.1), back: 1 };
            list.CopyTextureRegion(&to, at.0, at.1, 0, &from, Some(&part));
            Some(RECT { left: at.0 as i32, top: at.1 as i32, right: (at.0 + part.right) as i32, bottom: (at.1 + part.bottom) as i32 })
        }
    }
}

/// How long the start's notice stays.
const NOTICE_FOR: Duration = Duration::from_secs(30);
/// The longest notice line, in characters.
const NOTICE_COLUMNS: u32 = 52;

/// What this card does for frame generation, said at the top of the picture
/// for the first [`NOTICE_FOR`] of a run (`crate::streamline::notice`); in
/// every build, nothing without a notice.
pub(super) struct Notice {
    lines: Vec<String>,
    shape: Shape,
    since: Option<Instant>,
    buffers: Option<TextBuffers>,
    written: bool,
    done: bool,
}

impl Notice {
    pub(super) fn new() -> Notice {
        let lines: Vec<String> = crate::streamline::notice().into_iter().map(|l| l.to_ascii_uppercase()).collect();
        let columns = lines.iter().map(|l| l.len() as u32).max().unwrap_or(0).min(NOTICE_COLUMNS);
        let shape = Shape { columns, lines: lines.len() as u32 };
        Notice { done: lines.is_empty(), lines, shape, since: None, buffers: None, written: false }
    }

    /// Records the copy of the notice onto `back` (as [`Panel::draw`]), at
    /// the top in the middle, while it is due; where it lies.
    pub(super) fn draw(&mut self, host: &Host, list: &HostList, back: &ID3D12Resource, (width, height): (u32, u32), format: DXGI_FORMAT) -> Option<RECT> {
        if self.done {
            return None;
        }
        let since = *self.since.get_or_insert_with(Instant::now);
        if since.elapsed() > NOTICE_FOR {
            // (Its buffers stay: the card may still read them.)
            std::mem::forget(self.buffers.take());
            self.done = true;
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let _ours = crate::replay::record::unrecorded();
        let scale = text_scale(height);
        if self.buffers.as_ref().is_none_or(|b| (b.format, b.scale) != (format, scale)) {
            std::mem::forget(self.buffers.take());
            self.buffers = TextBuffers::new(host, self.shape, format, scale);
            self.written = false;
            if self.buffers.is_none() {
                self.done = true;
                return None;
            }
        }
        let buffers = self.buffers.as_mut()?;
        if !std::mem::replace(&mut self.written, true) {
            buffers.write(&self.lines);
        }
        let (w, _) = text_size(self.shape, scale);
        let at = (width.saturating_sub(w) / 2, 24 * scale);
        buffers.copy(&list.list, back, at, (width, height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uploads_are_reused_only_after_their_fence_completes() {
        let fences = [40, 41, 38, 39];
        assert_eq!(free_upload(&fences, 1, 37), None);
        assert_eq!(free_upload(&fences, 1, 38), Some(2));
        assert_eq!(free_upload(&fences, 3, 40), Some(0));
        assert_eq!(free_upload(&[0; UPLOADS], 0, 0), Some(1));
        // Read by a list not sent yet: never free.
        assert_eq!(free_upload(&[u64::MAX, 5, 5, 5], 3, u64::MAX - 1), Some(1));
    }

    /// Every character the panel uses has its own picture, and no two look alike.
    #[test]
    fn characters_differ() {
        let used: Vec<u8> = (b'0'..=b'9').chain(b'A'..=b'Z').chain(*b"./-|%: ").collect();
        for (i, a) in used.iter().enumerate() {
            assert_ne!(glyph(*a), glyph(b'#'), "{} has a picture", *a as char);
            assert!(glyph(*a).iter().all(|row| row >> GLYPH.0 == 0));
            for b in &used[i + 1..] {
                assert_ne!(glyph(*a), glyph(*b), "{} {}", *a as char, *b as char);
            }
        }
    }

    fn snapshot() -> Snapshot {
        Snapshot {
            fps: 32.0,
            frame_rates: None,
            frame_ms: 31.2,
            gpu_ms: 28.7,
            gpu_pct: 92,
            gpu_measured: true,
            submit_ms: 19.8,
            low1_fps: 14.0,
            max_ms: 95.0,
            draws: 8903,
            dropped: 0,
            width: 3840,
            height: 2160,
            gpu_name: "NVIDIA GEFORCE RTX 4070".into(),
            vram_used_gb: 11.4,
            vram_budget_gb: 11.2,
            vram_shared_gb: 4.2,
            vram_rt_gb: 2.8,
            vram_tex_gb: 3.5,
            vram_tiles_gb: 1.2,
            mb_up: 40.0,
            cpu_name: "AMD RYZEN 7 9800X3D".into(),
            cpu_pct: 34.0,
            cores_pct: vec![60.0, 40.0, 90.0, 20.0],
            proc_ram_gb: 32.4,
            sys_ram_used_gb: 41.2,
            sys_ram_total_gb: 61.4,
            sys_commit_gb: 48.0,
            history_ms: vec![28.0, 31.0, 85.0, 32.0],
        }
    }

    /// The whole panel and the strip (the strip state is switched here and
    /// back: one test does both, so that no other sees it changed).
    #[test]
    fn a_box_is_painted() {
        let mut snap = snapshot();
        strip_state().store(false, Ordering::Relaxed);
        let canvas = paint_hud(&snap, false);
        assert_eq!((canvas.w, canvas.h), (BASE_W as usize, BASE_H as usize));
        let header_has = |c: &Canvas, x: usize, color: [u8; 3]| {
            (25..63).any(|y| c.rgb[y * c.w + x..y * c.w + x + 174].contains(&color))
        };
        assert_eq!((accent(0, 2), accent(1, 2), accent(2, 2)), (PINK, LILAC, BLUE));
        assert!(header_has(&canvas, 14, PINK));
        assert!(!canvas.rgb.contains(&FG_MINT));
        assert!(!paint_hud(&snap, true).rgb.contains(&FG_MINT));
        // The graph: a frame over twice the budget is red.
        assert!(canvas.rgb.contains(&RED));
        assert!(canvas.rgb.contains(&GRAPH_TARGET));
        snap.frame_rates = Some((21.0, 42.0));
        let generated = paint_hud(&snap, true);
        assert!(header_has(&generated, 14, AMBER));
        assert!(header_has(&generated, 198, FG_MINT));
        assert_eq!(generated.rgb[17 * generated.w + 352], MUTED);
        assert!(generated.rgb.iter().enumerate().all(|(i, &p)| {
            p != FG_MINT || ((198..372).contains(&(i % generated.w)) && (33..68).contains(&(i / generated.w)))
        }));
        for scale in [1u32, 2] {
            let (width, height) = size(scale);
            let pitch = (width * 4).next_multiple_of(256) as usize;
            let packed = canvas.pack(scale, DXGI_FORMAT_R10G10B10A2_UNORM, pitch);
            assert_eq!(packed.len(), pitch * height as usize);
            let mut border = [0u8; 4];
            encode(DXGI_FORMAT_R10G10B10A2_UNORM, BORDER, &mut border);
            let top = (width / 2 * 4) as usize;
            assert_eq!(&packed[top..top + 4], &border);
            let mut corner = [0u8; 4];
            encode(DXGI_FORMAT_R10G10B10A2_UNORM, BG, &mut corner);
            assert_eq!(&packed[0..4], &corner);
        }

        strip_state().store(true, Ordering::Relaxed);
        let strip = paint_hud(&snap, true);
        assert_eq!((strip.w, strip.h), (STRIP_W as usize, STRIP_H as usize));
        assert_eq!(size(2), (STRIP_W * 2, STRIP_H * 2));
        assert!(strip.rgb.contains(&RED) && strip.rgb.contains(&FG_MINT));
        snap.frame_rates = None;
        assert!(!paint_hud(&snap, true).rgb.contains(&FG_MINT));
        strip_state().store(false, Ordering::Relaxed);
    }

    /// 1x up to 1440p, 2x at 4K.
    #[test]
    fn scaled_at_4k() {
        assert_eq!((scale(1080), scale(1440), scale(2160)), (1, 1, 2));
    }

    #[test]
    fn a_text_box_is_painted() {
        let shape = Shape { columns: 40, lines: 10 };
        let lines = vec!["I".to_owned(), " i".to_owned()];
        for scale in [1, 2, 4] {
            let (width, height) = text_size(shape, scale);
            let pitch = (width * 4).next_multiple_of(256) as usize;
            let colours = colours(DXGI_FORMAT_R10G10B10A2_UNORM).unwrap();
            let picture = painted(&lines, shape, scale, colours, pitch);
            assert_eq!(picture.len(), pitch * height as usize);
            let at = |x: u32, y: u32| &picture[y as usize * pitch + x as usize * 4..][..4];
            // The rim is black, the first letter's top row lit in its middle.
            assert_eq!(at(0, 0), colours[0]);
            assert_eq!(at((RIM + 2) * scale, RIM * scale), colours[1]);
            assert_eq!(at((RIM + 2) * scale + scale - 1, RIM * scale + scale - 1), colours[1]);
            assert_eq!(at(RIM * scale, RIM * scale), colours[0]);
            // Small letters are shown as capitals, a line and a cell on.
            assert_eq!(at((RIM + CELL.0 + 2) * scale, (RIM + CELL.1) * scale), colours[1]);
            // Nothing past a line's end.
            assert_eq!(at((RIM + CELL.0 * 2 + 2) * scale, (RIM + CELL.1) * scale), colours[0]);
        }
    }

    #[test]
    fn counters_that_went_back_to_nothing_count_from_there() {
        let (mut a, mut b) = (Counters::default(), Counters::default());
        (a.0[0], b.0[0]) = (10, 30);
        (a.0[1], b.0[1]) = (50, 20);
        let c = b.since(a, 10.0);
        assert_eq!((c[0], c[1]), (2.0, 2.0));
    }
}
