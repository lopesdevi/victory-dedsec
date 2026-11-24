//! `victory precompile`: every shader container found (a shader pack such
//! as `shaders.rpf`, containers stored as is) goes through the translation a
//! title's pipelines give it, into the shader cache `run` uses. A run then
//! finds its shaders translated instead of translating them at first use.
//!
//! The cache is keyed by this executable, so the same `victory.exe` must
//! precompile and run. Containers are cut to the size their header gives:
//! the title hands the pipeline exactly that many bytes.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;

use crate::shader::{MAX_FILE, sources};

/// The container's total size from its DXBC header (offset 24).
fn container_len(bytes: &[u8]) -> Option<usize> {
    let size = u32::from_le_bytes(bytes.get(24..28)?.try_into().ok()?) as usize;
    (size >= 32 && size <= bytes.len()).then_some(size)
}

#[cfg(windows)]
pub fn run(paths: Vec<PathBuf>, jobs: Option<usize>) -> Result<()> {
    let cfg = victory_core::Config::load()?;
    let dir = cfg.cache_subdir("shaders")?;
    println!("shader cache {}", dir.display());
    victory_hle::modules::d3d12x::use_shader_cache(dir);
    let paths = crate::default_paths(paths)?;
    let threads =
        jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let start = std::time::Instant::now();
    let seen = Mutex::new(HashSet::new());
    let (found, done, failed, duplicates) = (
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
        AtomicUsize::new(0),
    );
    let reasons: Mutex<std::collections::BTreeMap<String, usize>> = Mutex::default();
    for root in &paths {
        for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            if jobs.is_empty() {
                continue;
            }
            println!("{}: {} containers, {threads} threads", entry.path().display(), jobs.len());
            found.fetch_add(jobs.len(), Ordering::Relaxed);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    s.spawn(|| {
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let Some(bytes) = job.bytes() else {
                                failed.fetch_add(1, Ordering::Relaxed);
                                continue;
                            };
                            let Some(len) = container_len(&bytes) else {
                                failed.fetch_add(1, Ordering::Relaxed);
                                continue;
                            };
                            let container = &bytes[..len];
                            if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(blake3::hash(container)) {
                                duplicates.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            if let Err(e) = victory_gpu::shader_cache::precompile(container) {
                                failed.fetch_add(1, Ordering::Relaxed);
                                let reason: String = e.lines().next().unwrap_or("").chars().take(120).collect();
                                *reasons.lock().unwrap_or_else(|e| e.into_inner()).entry(reason).or_default() += 1;
                            }
                            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                            if n % 1000 == 0 {
                                let (hits, misses) = victory_gpu::shader_cache::precompile_counts();
                                println!(
                                    "  {n}/{} ({hits} from the cache, {misses} translated, {} failed, {:.0} s)",
                                    jobs.len(),
                                    failed.load(Ordering::Relaxed),
                                    start.elapsed().as_secs_f64()
                                );
                            }
                        }
                    });
                }
            });
        }
    }
    let (hits, misses) = victory_gpu::shader_cache::precompile_counts();
    println!(
        "containers {} ({} duplicates); {hits} already in the cache, {misses} translated now, {} do not translate; {:.0} s",
        found.load(Ordering::Relaxed),
        duplicates.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
    let reasons = reasons.into_inner().unwrap_or_else(|e| e.into_inner());
    let mut reasons: Vec<_> = reasons.into_iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(&a.1));
    for (reason, n) in reasons.iter().take(25) {
        println!("  {n:6}  {reason}");
    }
    Ok(())
}

#[cfg(not(windows))]
pub fn run(_paths: Vec<PathBuf>, _jobs: Option<usize>) -> Result<()> {
    anyhow::bail!("precompile needs the Windows host")
}
