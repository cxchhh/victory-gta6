//! `VICTORY_WATCH_WRITES=<hash>,...` (diagnostics): what the named shaders
//! write to title buffers, as the title sees it.
//!
//! While a pipeline holding one of the named shaders is bound, every range
//! its shaders may write (`Shadow::gpu_writes`) is noted. A thread then logs
//! the first bytes of each noted range in title memory every two seconds
//! (as `u32` and as `f32`), so a value the title reads back from the card
//! (a time step, a water height looked up for the processor) shows whether
//! it arrived.

use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};

use crate::object::This;

/// Ranges noted: (shader hash, title address, length).
static NOTED: Mutex<Vec<(String, usize, usize)>> = Mutex::new(Vec::new());
const MOST: usize = 48;
const SHOWN: usize = 64;

thread_local! {
    /// The watched shader of the pipeline this thread is binding, if any.
    static BINDING: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn hashes() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        std::env::var("VICTORY_WATCH_WRITES")
            .unwrap_or_default()
            .split(',')
            .map(|h| h.trim().to_lowercase())
            .filter(|h| !h.is_empty())
            .collect()
    })
}

/// At the start of a bind: whether `pipeline` holds a watched shader.
pub(crate) fn binding(pipeline: This) {
    if hashes().is_empty() {
        return;
    }
    let hash = crate::pipeline::shader_hash_in(pipeline, hashes());
    BINDING.with(|b| *b.borrow_mut() = hash);
}

/// A range shaders of the pipeline being bound may write.
pub(crate) fn writes(address: usize, len: usize) {
    if hashes().is_empty() {
        return;
    }
    let Some(hash) = BINDING.with(|b| b.borrow().clone()) else {
        return;
    };
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    if noted.iter().any(|n| n.0 == hash && n.1 == address) || noted.len() >= MOST {
        return;
    }
    tracing::info!(hash, address = format_args!("{address:#x}"), len, "watched shader writes this title range");
    noted.push((hash, address, len));
    start();
}

/// A range shaders of the pipeline being bound read (a buffer view's range,
/// refreshed from title memory at submission): what the processor gave them,
/// logged as it is at the bind, at most once in two seconds per range.
pub(crate) fn reads(address: usize, len: usize) {
    if hashes().is_empty() {
        return;
    }
    let Some(hash) = BINDING.with(|b| b.borrow().clone()) else {
        return;
    };
    static LAST: Mutex<Vec<(usize, std::time::Instant)>> = Mutex::new(Vec::new());
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let now = std::time::Instant::now();
        if let Some(l) = last.iter_mut().find(|l| l.0 == address) {
            if now.duration_since(l.1) < std::time::Duration::from_secs(2) {
                return;
            }
            l.1 = now;
        } else if last.len() >= 64 {
            return;
        } else {
            last.push((address, now));
        }
    }
    dump(&hash, "reads", address, len, READ_SHOWN);
}

const READ_SHOWN: usize = 256;

/// Logs the first `most` bytes of `[address, address + len)` of title memory.
fn dump(hash: &str, what: &str, address: usize, len: usize, most: usize) {
    if !crate::trycopy::on() {
        return;
    }
    let n = len.min(most) & !3;
    let mut bytes = vec![0u8; n];
    // SAFETY: title memory, which may be unmapped: the fault-tolerant copy
    // says so instead of faulting.
    if !unsafe { crate::trycopy::try_copy(bytes.as_mut_ptr(), address as *const u8, n) } {
        tracing::info!(hash, what, address = format_args!("{address:#x}"), "watched range: not readable");
        return;
    }
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let hex: Vec<String> = words.iter().map(|w| format!("{w:08x}")).collect();
    let floats: Vec<String> = words.iter().map(|w| format!("{:.3}", f32::from_bits(*w))).collect();
    tracing::info!(
        hash,
        what,
        address = format_args!("{address:#x}"),
        len,
        "watched range now: [{}] as f32 [{}]",
        hex.join(" "),
        floats.join(" ")
    );
}

fn start() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let _ = std::thread::Builder::new().name("victory-writewatch".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(2));
            let noted = NOTED.lock().unwrap_or_else(|e| e.into_inner()).clone();
            for (hash, address, len) in noted {
                let n = len.min(SHOWN) & !3;
                let mut bytes = vec![0u8; n];
                // SAFETY: title memory, which may be unmapped by now: the
                // fault-tolerant copy says so instead of faulting.
                if !crate::trycopy::on() {
                    continue;
                }
                let ok = unsafe { crate::trycopy::try_copy(bytes.as_mut_ptr(), address as *const u8, n) };
                if !ok {
                    tracing::info!(hash, address = format_args!("{address:#x}"), "watched range: not readable");
                    continue;
                }
                let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let hex: Vec<String> = words.iter().map(|w| format!("{w:08x}")).collect();
                let floats: Vec<String> = words.iter().map(|w| format!("{:.4}", f32::from_bits(*w))).collect();
                tracing::info!(
                    hash,
                    address = format_args!("{address:#x}"),
                    len,
                    "watched range now: [{}] as f32 [{}]",
                    hex.join(" "),
                    floats.join(" ")
                );
            }
        });
    });
}
