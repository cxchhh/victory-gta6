//! One mirror of title memory in the card's memory (`VICTORY_FLAT`; off by
//! default), instead of a host copy per title buffer ([`crate::shadow`]).
//!
//! Every title buffer had a host copy of its own, made on first use and
//! brought up to date range by range before each submission: some 4,400
//! copies and 4.2 GB, and some 40,000 refreshes a frame (120-150 MB,
//! 50-60 ms) of ranges that overlap from one copy to the next. Here the
//! title's GPU address space is covered by reserved buffers of 4 GiB, one
//! per 4 GiB of addresses used ("windows"): a title address is the same
//! offset in its window, whichever buffer it belongs to.
//!
//! `VICTORY_FLAT=census` changes nothing: it counts, with the frame line,
//! what such a mirror would hold and send (the distinct 64 KiB pages the
//! refreshes and the declared shader writes name, the 4 GiB slots they lie
//! in, the pages the CPU and the GPU both write in one submission, the
//! views on the copies of their own bytes).

use std::collections::HashSet;
use std::sync::Mutex;

/// The page counted and mirrored: the card's tile and the console's page.
pub(crate) const PAGE: u64 = 64 << 10;

/// The bits of a title address below the window: 4 GiB, the most a raw
/// view's 32-bit offsets reach.
pub(crate) const WINDOW_BITS: u32 = 32;

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
    address >> WINDOW_BITS != (end - 1) >> WINDOW_BITS
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
        for slot in (r.0 * PAGE) >> WINDOW_BITS..=((r.1 * PAGE - 1) >> WINDOW_BITS) {
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
        let line = 1u64 << WINDOW_BITS;
        assert!(crosses(line - 4, line + 4));
        assert!(!crosses(line - 4, line));
        assert!(!crosses(line, line + 1));
    }
}
