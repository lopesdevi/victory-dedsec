//! The translator's fingerprint for the shader cache key: the BLAKE3 of
//! victory-shader's sources and manifest and of the workspace lock file.
//! Translations depend on those alone (and on the run-time settings the key
//! adds), so a rebuild that leaves the translator alone keeps the cache.

use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            files(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let shader = manifest.join("../victory-shader");
    let lock = manifest.join("../../Cargo.lock");
    let mut all = Vec::new();
    files(&shader.join("src"), &mut all);
    all.push(shader.join("Cargo.toml"));
    all.push(lock.clone());
    all.sort();
    let mut h = blake3::Hasher::new();
    for f in &all {
        let rel = f.strip_prefix(&manifest).unwrap_or(f);
        h.update(rel.to_string_lossy().replace('\\', "/").as_bytes());
        h.update(&std::fs::read(f).unwrap_or_default());
    }
    println!("cargo:rustc-env=VICTORY_TRANSLATOR_FINGERPRINT={}", h.finalize().to_hex());
    println!("cargo:rerun-if-changed={}", shader.display());
    println!("cargo:rerun-if-changed={}", lock.display());
}
