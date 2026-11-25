//! `VICTORY_GDI_PRESENT=1`: every presented frame is read back and drawn
//! into the title's window with GDI, and the swap chain is not presented.
//!
//! For hosts whose Vulkan driver cannot present to a window (under Wine on
//! a GPU whose display output is disabled, such as data-centre cards: the
//! driver answers `VK_ERROR_UNKNOWN` for every surface query), while the
//! window itself, on an X server, can be seen and recorded.
//! `VICTORY_GDI_PRESENT=<n>` draws every n-th frame only.

use std::sync::OnceLock;

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Direct3D12::{D3D12_PLACED_SUBRESOURCE_FOOTPRINT, ID3D12Resource};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, COLORONCOLOR, DIB_RGB_COLORS, GetDC, ReleaseDC, SRCCOPY,
    SetStretchBltMode, StretchDIBits,
};
use windows::Win32::UI::WindowsAndMessaging::GetClientRect;

/// Every how many frames one is drawn; 0: off.
fn every() -> u64 {
    static EVERY: OnceLock<u64> = OnceLock::new();
    *EVERY.get_or_init(|| match std::env::var("VICTORY_GDI_PRESENT") {
        Ok(v) if v.trim() == "0" || v.trim().is_empty() => 0,
        Ok(v) => v.trim().parse().unwrap_or(1).max(1),
        Err(_) => 0,
    })
}

/// Whether frames go to the window through GDI instead of the swap chain.
pub(crate) fn on() -> bool {
    every() != 0
}

/// Whether frame `frame` is to be drawn.
pub(crate) fn wanted(frame: u64) -> bool {
    let n = every();
    n != 0 && frame.is_multiple_of(n)
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

/// Rows of the read-back frame as 32-bit BGRA, `width * 4` bytes apart.
fn to_bgra(format: u32, width: usize, height: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let mut out = vec![0u8; width * height * 4];
    for y in 0..height {
        let row = &rows[y * pitch..];
        let dst = &mut out[y * width * 4..][..width * 4];
        match format {
            27..=32 => {
                for (d, s) in dst.chunks_exact_mut(4).zip(row.chunks_exact(4)) {
                    d.copy_from_slice(&[s[2], s[1], s[0], 255]);
                }
            }
            87 | 90 | 91 => {
                for (d, s) in dst.chunks_exact_mut(4).zip(row.chunks_exact(4)) {
                    d.copy_from_slice(&[s[0], s[1], s[2], 255]);
                }
            }
            23..=25 => {
                for (d, s) in dst.chunks_exact_mut(4).zip(row.chunks_exact(4)) {
                    let v = u32::from_le_bytes([s[0], s[1], s[2], s[3]]);
                    let c = |shift: u32| (((v >> shift) & 0x3ff) >> 2) as u8;
                    d.copy_from_slice(&[c(20), c(10), c(0), 255]);
                }
            }
            9..=14 => {
                for (d, s) in dst.chunks_exact_mut(4).zip(row.chunks_exact(8)) {
                    let c = |i: usize| srgb8(half(u16::from_le_bytes([s[i], s[i + 1]])));
                    d.copy_from_slice(&[c(4), c(2), c(0), 255]);
                }
            }
            _ => return None,
        }
    }
    Some(out)
}

/// Draws a frame read back by [`crate::capture::read_back`] (the copy has
/// finished) into `hwnd`, stretched to its client area.
pub(crate) fn blit(hwnd: HWND, buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
    let f = footprint.Footprint;
    let (width, height, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    let bgra = unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * height);
        let bgra = to_bgra(f.Format.0 as u32, width, height, pitch, rows);
        buffer.Unmap(0, None);
        bgra
    };
    let Some(bgra) = bgra else {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(format = f.Format.0, "GDI present: this format is not drawn"));
        return;
    };
    let info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width as i32,
            biHeight: -(height as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut rect = RECT::default();
    // SAFETY: our window; a DC released below; rows of `width * height`
    // 32-bit pixels as `info` says.
    unsafe {
        if GetClientRect(hwnd, &mut rect).is_err() {
            return;
        }
        let (cw, ch) = ((rect.right - rect.left).max(1), (rect.bottom - rect.top).max(1));
        let dc = GetDC(Some(hwnd));
        if dc.is_invalid() {
            return;
        }
        SetStretchBltMode(dc, COLORONCOLOR);
        StretchDIBits(
            dc,
            0,
            0,
            cw,
            ch,
            0,
            0,
            width as i32,
            height as i32,
            Some(bgra.as_ptr().cast()),
            &info,
            DIB_RGB_COLORS,
            SRCCOPY,
        );
        ReleaseDC(Some(hwnd), dc);
    }
}
