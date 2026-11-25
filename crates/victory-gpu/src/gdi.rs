//! `VICTORY_GDI_PRESENT=1`: every presented frame is read back and drawn
//! into the title's window with GDI, and the swap chain is not presented.
//!
//! For hosts whose Vulkan driver cannot present to a window (under Wine on
//! a GPU whose display output is disabled, such as data-centre cards: the
//! driver answers `VK_ERROR_UNKNOWN` for every surface query), while the
//! window itself, on an X server, can be seen and recorded.
//! `VICTORY_GDI_PRESENT=<n>` draws every n-th frame only;
//! `VICTORY_GDI_PRESENT=headless`: no window and nothing drawn, frames are
//! only counted (the frame rate is logged).
//!
//! `VICTORY_SHOT_FILE=<path>`: whenever that file appears, the next frame
//! is saved as a PNG beside it (named by the file's text, else by the
//! frame number) and the file is removed: screenshots on request.

use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Direct3D12::{D3D12_PLACED_SUBRESOURCE_FOOTPRINT, ID3D12Resource};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, COLORONCOLOR, DIB_RGB_COLORS, GetDC, ReleaseDC, SRCCOPY,
    SetStretchBltMode, StretchDIBits,
};
use windows::Win32::UI::WindowsAndMessaging::{GetClientRect, GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN};

/// Every how many frames one is drawn; 0: off; `u64::MAX`: headless.
fn every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| match std::env::var("VICTORY_GDI_PRESENT") {
        Ok(v) if v.trim() == "0" || v.trim().is_empty() => 0,
        Ok(v) if v.trim().eq_ignore_ascii_case("headless") => u64::MAX,
        Ok(v) => v.trim().parse().unwrap_or(1).max(1),
        Err(_) => 0,
    })
}

/// Whether frames go to the window through GDI instead of the swap chain.
pub(crate) fn on() -> bool {
    every() != 0
}

/// Whether there is no window at all (`VICTORY_GDI_PRESENT=headless`).
pub(crate) fn headless() -> bool {
    every() == u64::MAX
}

/// With `VICTORY_SHOT_FILE`: the path to save this frame to, when the
/// request file is there (looked for at most four times a second).
pub(crate) fn shot(frame: u64) -> Option<std::path::PathBuf> {
    static FILE: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let file = FILE.get_or_init(|| std::env::var_os("VICTORY_SHOT_FILE").map(Into::into)).as_ref()?;
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if last.is_some_and(|t| now.duration_since(t) < Duration::from_millis(250)) {
            return None;
        }
        *last = Some(now);
    }
    let text = std::fs::read_to_string(file).ok()?;
    let _ = std::fs::remove_file(file);
    let name: String = text.trim().chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect();
    let name = if name.is_empty() { format!("shot-{frame}") } else { name };
    Some(file.with_file_name(format!("{name}.png")))
}

/// Saves a frame read back by [`crate::capture::read_back`] (the copy has
/// finished) as a PNG at `path`.
pub(crate) fn save(path: &std::path::Path, buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    let f = footprint.Footprint;
    let (width, height, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * height);
        // Full size BGRA, then half size on writing.
        if let Some(bgra) = to_bgra(f.Format.0 as u32, [width, height, pitch], [width, height], rows) {
            crate::capture::write_png(path, width as u32, height as u32, width * 4, true, &bgra);
        }
        buffer.Unmap(0, None);
    }
}

/// Whether frame `frame` is to be drawn: every n-th, at most 15 a second.
pub(crate) fn wanted(frame: u64) -> bool {
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let n = every();
    if n == 0 || !frame.is_multiple_of(n) {
        return false;
    }
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let now = Instant::now();
    if last.is_some_and(|t| now.duration_since(t) < Duration::from_millis(66)) {
        return false;
    }
    *last = Some(now);
    true
}

fn half(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let man = f32::from(h & 0x3ff);
    sign * match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Linear light to an 8-bit sRGB value.
fn srgb8(v: f32) -> u8 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    let s = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// The read-back frame (`width` x `height`, rows `pitch` bytes apart) as
/// 32-bit BGRA of `dw` x `dh` pixels (nearest sample).
fn to_bgra(format: u32, [width, height, pitch]: [usize; 3], [dw, dh]: [usize; 2], rows: &[u8]) -> Option<Vec<u8>> {
    let size = match format {
        27..=32 | 87 | 90 | 91 | 23..=25 => 4,
        9..=14 => 8,
        _ => return None,
    };
    let xs: Vec<usize> = (0..dw).map(|x| (x * width / dw).min(width - 1) * size).collect();
    let mut out = vec![0u8; dw * dh * 4];
    for y in 0..dh {
        let row = &rows[(y * height / dh).min(height - 1) * pitch..];
        let dst = &mut out[y * dw * 4..][..dw * 4];
        for (d, &x) in dst.chunks_exact_mut(4).zip(&xs) {
            let s = &row[x..x + size];
            let px = match format {
                27..=32 => [s[2], s[1], s[0], 255],
                87 | 90 | 91 => [s[0], s[1], s[2], 255],
                23..=25 => {
                    let v = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                    let c = |shift: u32| (((v >> shift) & 0x3ff) >> 2) as u8;
                    [c(20), c(10), c(0), 255]
                }
                _ => {
                    let c = |i: usize| srgb8(half(u16::from_le_bytes([s[i], s[i + 1]])));
                    [c(4), c(2), c(0), 255]
                }
            };
            d.copy_from_slice(&px);
        }
    }
    Some(out)
}

/// A read-back frame waiting for the drawing thread: the window, the
/// format, `[width, height, pitch]` and the rows.
struct Pending {
    hwnd: usize,
    format: u32,
    dims: [usize; 3],
    rows: Vec<u8>,
}

static PENDING: Mutex<Option<Pending>> = Mutex::new(None);
static READY: Condvar = Condvar::new();

/// The thread that draws the newest pending frame (older ones are dropped).
fn drawer() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("victory-gdi".into()).spawn(|| loop {
            let frame = {
                let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if let Some(frame) = pending.take() {
                        break frame;
                    }
                    pending = READY.wait(pending).unwrap_or_else(|e| e.into_inner());
                }
            };
            draw(&frame);
        });
    });
}

fn draw(frame: &Pending) {
    let hwnd = HWND(frame.hwnd as *mut _);
    let [width, height, _] = frame.dims;
    let mut rect = RECT::default();
    // SAFETY: plain queries on our window.
    let (cw, ch) = unsafe {
        if GetClientRect(hwnd, &mut rect).is_err() {
            return;
        }
        let (sw, sh) = (GetSystemMetrics(SM_CXSCREEN).max(1), GetSystemMetrics(SM_CYSCREEN).max(1));
        ((rect.right - rect.left).clamp(1, sw) as usize, (rect.bottom - rect.top).clamp(1, sh) as usize)
    };
    // Keep the aspect ratio inside the visible client area.
    let (dw, dh) = if cw * height > ch * width { (ch * width / height, ch) } else { (cw, cw * height / width) };
    let (dw, dh) = (dw.max(1), dh.max(1));
    let Some(bgra) = to_bgra(frame.format, frame.dims, [dw, dh], &frame.rows) else {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(format = frame.format, "GDI present: this format is not drawn"));
        return;
    };
    let info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: dw as i32,
            biHeight: -(dh as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };
    // SAFETY: our window; a DC released below; `dw * dh` 32-bit pixels as
    // `info` says.
    unsafe {
        let dc = GetDC(Some(hwnd));
        if dc.is_invalid() {
            return;
        }
        SetStretchBltMode(dc, COLORONCOLOR);
        StretchDIBits(
            dc,
            0,
            0,
            dw as i32,
            dh as i32,
            0,
            0,
            dw as i32,
            dh as i32,
            Some(bgra.as_ptr().cast()),
            &info,
            DIB_RGB_COLORS,
            SRCCOPY,
        );
        ReleaseDC(Some(hwnd), dc);
    }
}

/// Hands a frame read back by [`crate::capture::read_back`] (the copy has
/// finished) to the drawing thread, which draws it into `hwnd`.
pub(crate) fn blit(hwnd: HWND, buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    let f = footprint.Footprint;
    let (width, height, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    let rows = unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * height).to_vec();
        buffer.Unmap(0, None);
        rows
    };
    drawer();
    *PENDING.lock().unwrap_or_else(|e| e.into_inner()) =
        Some(Pending { hwnd: hwnd.0 as usize, format: f.Format.0 as u32, dims: [width, height, pitch], rows });
    READY.notify_one();
}
