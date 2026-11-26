//! The console's resolution ladder, and the self-check that guards it, as
//! the title stores them (after soldsawn/victory-nvidia-8gb 037ef58933,
//! `VICTORY_RESOLUTION`, by DKTTT).
//!
//! Everything here is a byte pattern and a replacement, never a stored
//! offset, so nothing from the executable lives in the tree
//! (docs/asset-policy.md).

/// The console's resolution ladder as the title stores it: each entry is
/// `width, height, 60, 1` (sixteen bytes), from 720p to the Series X's 4K.
const LADDER: [(u32, u32); 5] = [(1280, 720), (1664, 936), (1920, 1080), (2560, 1440), (3840, 2160)];

/// The ladder's bytes.
const LADDER_BYTES: usize = LADDER.len() * 16;

/// The lowest frame the title's upscaler takes: below it the title stops
/// with `FSR_API_DEBUG_ERROR: maxUpscaleSize must be 1920x1080`.
pub const LOWEST: (u32, u32) = (1920, 1080);

fn ladder_pattern() -> Vec<u8> {
    let mut pattern = Vec::with_capacity(LADDER_BYTES);
    for (w, h) in LADDER {
        for word in [w, h, 60, 1] {
            pattern.extend_from_slice(&word.to_le_bytes());
        }
    }
    pattern
}

/// Where the ladder starts in `image`, if this build has it.
pub fn ladder_at(image: &[u8]) -> Option<usize> {
    let pattern = ladder_pattern();
    image.windows(pattern.len()).position(|s| s == pattern)
}

/// The ladder's sixteen-byte entries with every one above `w`x`h` capped at
/// that size, and how many changed. The title picks its window, swap chain
/// and render target sizes from this table: lowering its ceiling is what
/// makes it render below 4K (the display mode it is told about is not the
/// lever).
pub fn capped_ladder(w: u32, h: u32) -> ([u8; LADDER_BYTES], usize) {
    let mut out = [0u8; LADDER_BYTES];
    out.copy_from_slice(&ladder_pattern());
    let target = u64::from(w) * u64::from(h);
    let mut changed = 0;
    for (i, &(lw, lh)) in LADDER.iter().enumerate() {
        if u64::from(lw) * u64::from(lh) > target {
            out[16 * i..16 * i + 4].copy_from_slice(&w.to_le_bytes());
            out[16 * i + 4..16 * i + 8].copy_from_slice(&h.to_le_bytes());
            changed += 1;
        }
    }
    (out, changed)
}

/// The offset of the `int3` of the title's pool-offset self-check in
/// `image`, if this build has it.
///
/// The check is `mov rax,[rsp+0x78]; sub rax,[rsp+0x70]; add rax,[rip+disp];
/// cmp rax,0x1fffff; jbe +1; int3`. It compares a ring-buffer offset, not a
/// size; with a capped ladder that offset grows past 2 MiB and the check
/// fires on a value unrelated to the cap, so the one `int3` becomes a `nop`.
pub fn assert_site(image: &[u8]) -> Option<usize> {
    const PREFIX: [u8; 10] = [0x48, 0x8b, 0x44, 0x24, 0x78, 0x48, 0x2b, 0x44, 0x24, 0x70];
    const ADD: [u8; 3] = [0x48, 0x03, 0x05];
    const CMP: [u8; 6] = [0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00];
    const JBE: u8 = 0x76;
    const LEN: usize = 26;
    (0..image.len().saturating_sub(LEN - 1))
        .find(|&at| {
            image[at..at + 10] == PREFIX
                && image[at + 10..at + 13] == ADD
                && image[at + 17..at + 23] == CMP
                && image[at + 23] == JBE
                && image[at + 24] == 0x01
                && image[at + 25] == 0xcc
        })
        .map(|at| at + LEN - 1)
}

/// `VICTORY_RESOLUTION=<w>x<h>`, raised to [`LOWEST`] if below it.
pub fn requested() -> Option<(u32, u32)> {
    let text = std::env::var("VICTORY_RESOLUTION").ok()?;
    let (w, h) = text.trim().split_once(['x', 'X'])?;
    let (w, h): (u32, u32) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    if u64::from(w) * u64::from(h) < u64::from(LOWEST.0) * u64::from(LOWEST.1) {
        return Some(LOWEST);
    }
    Some((w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_with_ladder(at: usize) -> Vec<u8> {
        let mut image = vec![0x90u8; at + LADDER_BYTES + 16];
        image[at..at + LADDER_BYTES].copy_from_slice(&ladder_pattern());
        image
    }

    #[test]
    fn finds_the_ladder_and_caps_the_entries_above_the_target() {
        let image = image_with_ladder(64);
        assert_eq!(ladder_at(&image), Some(64));
        let (bytes, changed) = capped_ladder(1920, 1080);
        assert_eq!(changed, 2, "1440p and 2160p");
        let entry = |i: usize| {
            let e = &bytes[16 * i..16 * i + 8];
            (u32::from_le_bytes(e[0..4].try_into().unwrap()), u32::from_le_bytes(e[4..8].try_into().unwrap()))
        };
        assert_eq!(entry(0), (1280, 720));
        assert_eq!(entry(2), (1920, 1080));
        assert_eq!(entry(3), (1920, 1080));
        assert_eq!(entry(4), (1920, 1080));
        assert_eq!(&bytes[56..64], &[60, 0, 0, 0, 1, 0, 0, 0], "refresh and flag kept");
    }

    #[test]
    fn a_build_without_the_ladder_is_left_alone() {
        assert_eq!(ladder_at(&[0x90u8; 256]), None);
    }

    #[test]
    fn finds_the_self_check_and_not_a_near_miss() {
        let site = [
            0x48, 0x8b, 0x44, 0x24, 0x78, 0x48, 0x2b, 0x44, 0x24, 0x70, 0x48, 0x03, 0x05, 0x11, 0x22, 0x33, 0x44, 0x48,
            0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01, 0xcc,
        ];
        let mut image = [0x90u8; 128];
        image[40..66].copy_from_slice(&site);
        assert_eq!(assert_site(&image), Some(65));
        image[64] = 0x02;
        assert_eq!(assert_site(&image), None, "a jbe over more than the int3");
    }

    #[test]
    fn a_request_below_1080p_is_raised() {
        // SAFETY (test): single-threaded use of the variable.
        unsafe { std::env::set_var("VICTORY_RESOLUTION", "1280x720") };
        assert_eq!(requested(), Some(LOWEST));
        unsafe { std::env::set_var("VICTORY_RESOLUTION", "2560x1440") };
        assert_eq!(requested(), Some((2560, 1440)));
        unsafe { std::env::remove_var("VICTORY_RESOLUTION") };
    }
}
