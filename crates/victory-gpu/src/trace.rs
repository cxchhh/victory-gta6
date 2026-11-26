//! A trace of the memory layer, recorded in a run and replayed without the
//! title (`victory trace-replay <folder>`): the measure and the check of a
//! change to how title memory reaches the card, in seconds and on the
//! same input every time, instead of a launch of the title per try.
//!
//! What a trace holds (`VICTORY_TRACE=<folder>`, from `VICTORY_TRACE_AFTER`
//! seconds after the first submission (150), for `VICTORY_TRACE_FOR`
//! seconds (10)):
//!
//! - the title buffers known when it starts and those made and let go
//!   while it runs ([`crate::shadow::register`]);
//! - every range a list asks to have brought up to date for its shaders
//!   ([`crate::shadow::Shadow::refresh`]) and every range its shaders may
//!   write ([`crate::shadow::Shadow::gpu_writes`]), with the list and the
//!   recording thread;
//! - every submission, its lists in order, and, ahead of it, the 64 KiB
//!   pages of title memory its ranges cover that the trace has not held
//!   yet, whole (pages of zeros are named, not stored);
//! - the bytes the layer copied from title memory into its copies while
//!   the submission was prepared ([`copied`]): what the title changed, as
//!   the write watch and the guards found it, at no cost of looking for
//!   it again. Bytes go to `pages.bin`.
//!
//! What it leaves out (the replay's limits): the title's draws and
//! dispatches (nothing is drawn: what shaders write is never written, so
//! copy-backs find nothing), its other submission-time steps (buffer
//! copies, fills of upload memory, textures), and view memory's guards
//! (every page is replayed as write-watched memory).
//!
//! The replay maps the traced pages at their own addresses as
//! write-watched memory, makes the title buffers known, and walks the
//! events: ranges asked on fresh host lists, pages written before the
//! submission that read them (the write watch sees the writes, as it saw
//! the title's), and the submissions sent through [`crate::host::Host`].
//! After each submission (`--verify`) every range a list asked for is
//! compared with title memory on the copy's CPU side: the bytes the card
//! reads must be the title's.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The unit pages are traced in (the write watch's and the console's).
const PAGE: u64 = 64 << 10;

/// Kinds of the 40-byte records of `events.bin`: `[kind, a, b, c, d]`.
const REGISTER: u64 = 1; // resource, start, len
const UNREGISTER: u64 = 2; // resource, start
const REFRESH: u64 = 3; // thread, list, address, len
const GPU_WRITES: u64 = 4; // thread, list, address, len
const PAGE_BYTES: u64 = 5; // address, offset in pages.bin (u64::MAX: zeros)
const SUBMIT: u64 = 6; // number, lists that follow, late
const LIST: u64 = 7; // list
const END: u64 = 8;
const RUN: u64 = 9; // address, len, offset in pages.bin, submission

struct Recorder {
    events: BufWriter<File>,
    pages: BufWriter<File>,
    /// Bytes written to `pages.bin`.
    at: u64,
    /// The pages the trace holds.
    held: std::collections::HashSet<u64>,
    started: std::time::Instant,
    length: std::time::Duration,
    submissions: u64,
    page_bytes: u64,
}

static RECORDER: Mutex<Option<Recorder>> = Mutex::new(None);
/// Whether a trace is being recorded now (read without the lock).
static ON: AtomicBool = AtomicBool::new(false);
/// 0: not asked yet; 1: waiting for the start; 2: recording or done.
static STATE: AtomicU64 = AtomicU64::new(0);

/// Whether submissions should be shown to [`submitting`]: a trace is
/// asked for and not finished.
pub(crate) fn wanted() -> bool {
    folder().is_some() && (STATE.load(Ordering::Relaxed) == 0 || ON.load(Ordering::Relaxed))
}

fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER.get_or_init(|| std::env::var_os("VICTORY_TRACE").map(PathBuf::from)).as_ref()
}

fn seconds(name: &str, usual: f64) -> std::time::Duration {
    std::time::Duration::from_secs_f64(std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(usual))
}

fn thread() -> u64 {
    // SAFETY: no arguments.
    u64::from(unsafe { windows::Win32::System::Threading::GetCurrentThreadId() })
}

fn record(r: &mut Recorder, e: [u64; 5]) {
    let mut bytes = [0u8; 40];
    for (i, v) in e.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    let _ = r.events.write_all(&bytes);
}

fn with(f: impl FnOnce(&mut Recorder)) {
    if !ON.load(Ordering::Relaxed) {
        return;
    }
    let mut r = RECORDER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = r.as_mut() {
        f(r);
    }
}

/// A host list's name in the trace: its command list, which stays put
/// while the host list itself moves (a list detached for a late
/// submission is moved between its recording and its submission).
pub(crate) fn list_id(list: &crate::host::HostList) -> usize {
    windows::core::Interface::as_raw(&list.list) as usize
}

/// A title buffer is known (from [`crate::shadow::register`]).
pub(crate) fn registered(resource: usize, start: u64, len: u64) {
    with(|r| record(r, [REGISTER, resource as u64, start, len, 0]));
}

/// A title buffer is let go.
pub(crate) fn unregistered(resource: usize, start: u64) {
    with(|r| record(r, [UNREGISTER, resource as u64, start, 0, 0]));
}

/// A list asks for a range of title memory for its shaders to read.
pub(crate) fn refresh(list: usize, address: u64, len: u64) {
    with(|r| record(r, [REFRESH, thread(), list as u64, address, len]));
}

/// A list's shaders may write a range of title memory.
pub(crate) fn gpu_writes(list: usize, address: u64, len: u64) {
    with(|r| record(r, [GPU_WRITES, thread(), list as u64, address, len]));
}

/// The layer copied `len` bytes of title memory at `address` into a copy
/// (`bytes`, the copy's side, now equal to title memory): the change the
/// replay writes before the submission being prepared.
pub(crate) fn copied(address: usize, bytes: *const u8, len: usize) {
    with(|r| {
        // SAFETY: the caller's mapping of `len` bytes.
        let _ = r.pages.write_all(unsafe { std::slice::from_raw_parts(bytes, len) });
        r.at += len as u64;
        r.page_bytes += len as u64;
        let (at, n) = (r.at - len as u64, r.submissions);
        record(r, [RUN, address as u64, len as u64, at, n]);
    });
}

/// A submission of `lists`, which bring `ranges` of title memory up to
/// date: the changed pages of the ranges, then the submission. Starts and
/// ends the trace by the clock.
pub(crate) fn submitting(lists: &[usize], ranges: &[(usize, usize)], late: bool) {
    let Some(folder) = folder() else { return };
    match STATE.load(Ordering::Relaxed) {
        0 => {
            static FIRST: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let first = *FIRST.get_or_init(std::time::Instant::now);
            if first.elapsed() >= seconds("VICTORY_TRACE_AFTER", 150.0) {
                start(folder);
            }
            return;
        }
        2 if !ON.load(Ordering::Relaxed) => return,
        _ => {}
    }
    let mut guard = RECORDER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(r) = guard.as_mut() else { return };
    let mut pages: Vec<u64> = ranges
        .iter()
        .flat_map(|&(a, n)| (a as u64 / PAGE)..(a as u64 + n as u64).div_ceil(PAGE))
        .collect();
    pages.sort_unstable();
    pages.dedup();
    let mut buf = vec![0u8; PAGE as usize];
    for page in pages {
        let address = page * PAGE;
        if !r.held.insert(address) {
            continue;
        }
        // Through the kernel's read: unmapped pages read as zeros, guards
        // are not touched.
        crate::host::copy_mapped(buf.as_mut_ptr(), address as *const u8, PAGE as usize, false);
        let at = if buf.iter().all(|&b| b == 0) {
            u64::MAX
        } else {
            let _ = r.pages.write_all(&buf);
            r.at += PAGE;
            r.page_bytes += PAGE;
            r.at - PAGE
        };
        record(r, [PAGE_BYTES, address, at, 0, 0]);
    }
    r.submissions += 1;
    let n = r.submissions;
    record(r, [SUBMIT, n, lists.len() as u64, u64::from(late), 0]);
    for &list in lists {
        record(r, [LIST, list as u64, 0, 0, 0]);
    }
    if r.started.elapsed() >= r.length {
        record(r, [END, 0, 0, 0, 0]);
        let _ = r.events.flush();
        let _ = r.pages.flush();
        ON.store(false, Ordering::Relaxed);
        tracing::info!(
            "trace of the memory layer written to {}: {} submissions, {} MB of pages (VICTORY_TRACE)",
            folder.display(),
            r.submissions,
            r.page_bytes >> 20
        );
        *guard = None;
    }
}

fn start(folder: &Path) {
    STATE.store(2, Ordering::Relaxed);
    if let Err(e) = std::fs::create_dir_all(folder) {
        tracing::warn!(%e, "no trace: its folder cannot be made");
        return;
    }
    let open = |name: &str| File::create(folder.join(name)).map(|f| BufWriter::with_capacity(8 << 20, f));
    let (Ok(events), Ok(pages)) = (open("events.bin"), open("pages.bin")) else {
        tracing::warn!("no trace: its files cannot be made");
        return;
    };
    let mut r = Recorder {
        events,
        pages,
        at: 0,
        held: std::collections::HashSet::new(),
        started: std::time::Instant::now(),
        length: seconds("VICTORY_TRACE_FOR", 10.0),
        submissions: 0,
        page_bytes: 0,
    };
    // The buffers known now, as if made at the start.
    for (i, (start, len, window)) in crate::shadow::spans().into_iter().enumerate() {
        if !window {
            record(&mut r, [REGISTER, (1 << 62) | i as u64, start, len, 0]);
        }
    }
    *RECORDER.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    ON.store(true, Ordering::Relaxed);
    tracing::info!("trace of the memory layer started (VICTORY_TRACE={})", folder.display());
}

/// What a replay measured.
#[derive(Default, Debug)]
pub struct Replayed {
    pub submissions: u64,
    pub refreshes: u64,
    pub ms: f64,
    pub submit_ms: f64,
    /// Writing the traced pages into title memory (the title's part).
    pub pages_ms: f64,
    /// Finding copies and asking for ranges (the recording threads' part).
    pub asks_ms: f64,
    pub lists_ms: f64,
    pub copies_ms: f64,
    pub copies_mb: f64,
    pub checked: u64,
    pub wrong: u64,
    pub unchecked: u64,
    pub no_copy: u64,
}

fn read_events(folder: &Path) -> std::io::Result<Vec<[u64; 5]>> {
    let mut bytes = Vec::new();
    File::open(folder.join("events.bin"))?.read_to_end(&mut bytes)?;
    Ok(bytes
        .chunks_exact(40)
        .map(|c| std::array::from_fn(|i| u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap())))
        .collect())
}

/// Replays the trace in `folder` `passes` times (see the module's
/// comment); with `verify`, checks every asked range after its submission.
pub fn replay(folder: &Path, passes: u32, verify: bool) -> Result<Vec<Replayed>, String> {
    use std::os::windows::fs::FileExt;
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    let events = read_events(folder).map_err(|e| format!("events.bin: {e}"))?;
    let pages = File::open(folder.join("pages.bin")).map_err(|e| format!("pages.bin: {e}"))?;
    let host = crate::host::host().ok_or("no host device")?;
    // Title memory: every range the trace names, joined, at its own address.
    let mut spans: Vec<(u64, u64)> = Vec::new();
    for e in &events {
        let (a, n) = match e[0] {
            REGISTER => (e[2], e[3]),
            REFRESH | GPU_WRITES => (e[3], e[4].max(1)),
            PAGE_BYTES => (e[1], PAGE),
            _ => continue,
        };
        spans.push((a / PAGE * PAGE, (a + n).div_ceil(PAGE) * PAGE));
    }
    spans.sort_unstable();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for (a, b) in spans {
        match runs.last_mut() {
            Some(last) if a <= last.1 => last.1 = last.1.max(b),
            _ => runs.push((a, b)),
        }
    }
    let (mut mapped, mut refused) = (0u64, 0u64);
    let mut owned: Vec<(u64, u64)> = Vec::new();
    for &(a, b) in &runs {
        // SAFETY: a fixed address the replay owns, or a refusal.
        // (Address 0 asks the kernel to choose: the first 64 KiB are never
        // the title's; any other answer than the address asked is let go.)
        let p = if a < PAGE {
            std::ptr::null_mut()
        } else {
            unsafe { VirtualAlloc(a as *const _, (b - a) as usize, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) }
        };
        if !p.is_null() && p as u64 != a {
            // SAFETY: the allocation just made, whole.
            unsafe { windows_sys::Win32::System::Memory::VirtualFree(p, 0, windows_sys::Win32::System::Memory::MEM_RELEASE) };
            refused += b - a;
        } else if p.is_null() {
            refused += b - a;
        } else {
            mapped += b - a;
            owned.push((a, b));
        }
    }
    // Writes into title memory as the title makes them: plain stores, which
    // the write watch sees; none where the replay could not map.
    let inside = |address: u64, len: u64| -> bool {
        let i = owned.partition_point(|&(a, _)| a <= address);
        i > 0 && owned[i - 1].1 >= address + len
    };
    let write_title = |address: u64, bytes: &[u8]| -> bool {
        if !inside(address, bytes.len() as u64) {
            return false;
        }
        // SAFETY: inside a run the replay mapped read-write.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), address as *mut u8, bytes.len()) };
        true
    };
    tracing::info!(
        "trace replay: {} events, title memory in {} runs, {} MB mapped, {} MB refused (taken in this process)",
        events.len(),
        runs.len(),
        mapped >> 20,
        refused >> 20
    );
    // The bytes copied while each submission was prepared.
    let mut runs_of: HashMap<u64, Vec<(u64, u64, u64)>> = HashMap::new();
    for e in &events {
        if e[0] == RUN {
            runs_of.entry(e[4]).or_default().push((e[1], e[2], e[3]));
        }
    }
    let mut results = Vec::new();
    let mut buf = vec![0u8; PAGE as usize];
    let mut run_buf = Vec::new();
    for _ in 0..passes {
        let mut out = Replayed::default();
        let started = std::time::Instant::now();
        let p0 = crate::host::PREPARING[0].load(Ordering::Relaxed);
        let p1 = crate::host::PREPARING[1].load(Ordering::Relaxed);
        let mut lists: HashMap<u64, (crate::host::HostList, Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>)> = HashMap::new();
        let mut i = 0;
        while i < events.len() {
            let e = events[i];
            i += 1;
            if std::env::var_os("VICTORY_TRACE_STEPS").is_some() && (i % 50_000 == 0 || (e[0] != REGISTER && i < 400_000)) {
                eprintln!("replay event {i}: {e:x?}");
            }
            match e[0] {
                REGISTER => crate::shadow::register(e[1] as usize, e[2], e[3]),
                UNREGISTER => crate::shadow::unregister(e[1] as usize, e[2]),
                REFRESH | GPU_WRITES => {
                    // Asks of memory the replay could not map (null views,
                    // memory of the title's not traced) are left out.
                    if !inside(e[3], e[4].max(1)) {
                        out.no_copy += 1;
                        continue;
                    }
                    let asked = std::time::Instant::now();
                    let entry = match lists.entry(e[2]) {
                        std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
                        std::collections::hash_map::Entry::Vacant(v) => {
                            v.insert((host.new_list().map_err(|e| format!("a host list: {e}"))?, Vec::new()))
                        }
                    };
                    // A read goes where a read goes (with `VICTORY_FLAT=1`
                    // the mirror, `crate::flat`), a write to the copies.
                    let found = if e[0] == REFRESH { crate::shadow::find_read(e[3], e[4]) } else { crate::shadow::find(e[3], e[4]) };
                    let Some(shadow) = found else {
                        out.no_copy += 1;
                        continue;
                    };
                    if e[0] == REFRESH {
                        shadow.refresh(&entry.0, e[3], e[4]);
                        entry.1.push((shadow.clone(), e[3], e[4]));
                        out.refreshes += 1;
                    } else {
                        shadow.gpu_writes(&entry.0, e[3], e[4]);
                    }
                    out.asks_ms += asked.elapsed().as_secs_f64() * 1e3;
                }
                PAGE_BYTES => {
                    let written_at = std::time::Instant::now();
                    let dst = e[1];
                    if e[2] == u64::MAX {
                        buf.fill(0);
                    } else {
                        pages.seek_read(&mut buf, e[2]).map_err(|err| format!("pages.bin at {}: {err}", e[2]))?;
                    }
                    if !write_title(dst as u64, &buf) {
                        out.unchecked += 1;
                    }
                    out.pages_ms += written_at.elapsed().as_secs_f64() * 1e3;
                }
                SUBMIT => {
                    // What the title changed for this submission.
                    let written_at = std::time::Instant::now();
                    for &(address, len, at) in runs_of.get(&e[1]).map_or(&[][..], |v| &v[..]) {
                        run_buf.resize(len as usize, 0);
                        pages.seek_read(&mut run_buf, at).map_err(|err| format!("pages.bin at {at}: {err}"))?;
                        if !write_title(address, &run_buf) {
                            out.unchecked += 1;
                        }
                    }
                    out.pages_ms += written_at.elapsed().as_secs_f64() * 1e3;
                    let n = e[2] as usize;
                    let ids: Vec<u64> = events[i..i + n].iter().map(|l| l[1]).collect();
                    i += n;
                    let taken: Vec<(crate::host::HostList, Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>)> =
                        ids.iter().filter_map(|id| lists.remove(id)).collect();
                    let closing = std::time::Instant::now();
                    for (list, _) in &taken {
                        list.close();
                    }
                    out.lists_ms += closing.elapsed().as_secs_f64() * 1e3;
                    let runs: Vec<&crate::host::HostList> = taken.iter().map(|(l, _)| l).collect();
                    let sent = std::time::Instant::now();
                    host.execute(&runs);
                    out.submit_ms += sent.elapsed().as_secs_f64() * 1e3;
                    out.submissions += 1;
                    if out.submissions % 50 == 0 {
                        eprintln!(
                            "replay: {} submissions in {:.0} ms (asks {:.0}, pages {:.0}, submit {:.0}), {} asks left out",
                            out.submissions,
                            started.elapsed().as_secs_f64() * 1e3,
                            out.asks_ms,
                            out.pages_ms,
                            out.submit_ms,
                            out.no_copy
                        );
                    }
                    // One submission in `VICTORY_TRACE_VERIFY_EVERY` (25): the
                    // asked ranges joined copy by copy, each compared once.
                    let every = std::env::var("VICTORY_TRACE_VERIFY_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(25u64).max(1);
                    if verify && out.submissions % every == 0 {
                        let mut by_copy: HashMap<u64, (std::sync::Arc<crate::shadow::Shadow>, Vec<(u64, u64)>)> = HashMap::new();
                        for (_, ranges) in &taken {
                            // The copy (or window of the mirror) the ask used.
                            for (shadow, a, n) in ranges {
                                let (shadow, a, n) = (shadow, *a, *n);
                                let (start, len) = shadow.span();
                                let (from, to) = (a.max(start), (a + n.max(1)).min(start + len));
                                if from < to {
                                    by_copy.entry(shadow.id()).or_insert_with(|| (shadow.clone(), Vec::new())).1.push((from, to));
                                }
                            }
                        }
                        for (shadow, mut spans) in by_copy.into_values() {
                            spans.sort_unstable();
                            let mut joined: Vec<(u64, u64)> = Vec::new();
                            for (a, b) in spans {
                                match joined.last_mut() {
                                    Some(last) if a <= last.1 => last.1 = last.1.max(b),
                                    _ => joined.push((a, b)),
                                }
                            }
                            for (a, b) in joined {
                                // A window of the mirror has no CPU side: what
                                // the card holds is read back.
                                let shown = if shadow.is_flat() && inside(a, b - a) { crate::flat::shown(a, b - a) } else { None };
                                let side = match &shown {
                                    Some(bytes) => Some(&bytes[..]),
                                    None => shadow.cpu_side(a, b - a),
                                };
                                match side.filter(|side| inside(a, side.len() as u64)) {
                                    Some(side) => {
                                        out.checked += 1;
                                        // SAFETY: mapped title memory of the replay.
                                        let title = unsafe { std::slice::from_raw_parts(a as *const u8, side.len()) };
                                        if title != side {
                                            out.wrong += 1;
                                            if out.wrong <= 5 {
                                                let at = title.iter().zip(side).position(|(x, y)| x != y).unwrap_or(0);
                                                eprintln!(
                                                    "replay: WRONG after submission {}: copy {} [{a:#x}, {b:#x}) first differs at {:#x}",
                                                    out.submissions,
                                                    shadow.id(),
                                                    a + at as u64
                                                );
                                            }
                                        }
                                    }
                                    None => out.unchecked += 1,
                                }
                            }
                        }
                    }
                }
                END => break,
                _ => {}
            }
        }
        out.ms = started.elapsed().as_secs_f64() * 1e3;
        out.copies_ms = (crate::host::PREPARING[0].load(Ordering::Relaxed) - p0) as f64 / 1e3;
        out.copies_mb = (crate::host::PREPARING[1].load(Ordering::Relaxed) - p1) as f64 / f64::from(1 << 20);
        // The mirror's counts for the pass, per submission (`crate::flat`).
        crate::flat::line(out.submissions as f64);
        results.push(out);
    }
    Ok(results)
}

