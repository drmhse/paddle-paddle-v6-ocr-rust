//! Remote model fetch + local cache + SHA-256 verification.
//!
//! Models are no longer embedded in the binary. On first run each required
//! model is downloaded from the CDN, verified against its manifest checksum,
//! and cached under `$PPOCR_CACHE_DIR` (default `~/.cache/ppocr-server/models`).
//! Subsequent runs load straight from the cache. Downloads are written through
//! a temp file + atomic rename, so a present cache file is always complete and
//! verified — restarts skip re-hashing.

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::PathBuf;

/// (model id, CDN url, sha256-hex). `id` matches the ids used by the engine /
/// understanding stage; the cache file is `<id>.bin`.
pub const MANIFEST: &[(&str, &str, &str)] = &[
    ("PP-OCRv6_tiny_det", "https://cdn.drmhse.com/models/ocr-ppocrv6/v1/PP-OCRv6_tiny_det.onnx", "36f918a04075fa4002d8b4c4fdee4f5e5fdd282f6579510490f3a3e9e4cb96cb"),
    ("PP-OCRv6_tiny_rec", "https://cdn.drmhse.com/models/ocr-ppocrv6/v1/PP-OCRv6_tiny_rec.onnx", "741f5f765e7321108a3df97c888e0f65f659e62f3adb1c944e04f0898a018f7f"),
    ("PP-OCRv6_small_det", "https://cdn.drmhse.com/models/ocr-ppocrv6/v1/PP-OCRv6_small_det.onnx", "c00ff519e0fe30ffdc19763732773f53aa6b5eac61432ca03248d189800d76ee"),
    ("PP-OCRv6_small_rec", "https://cdn.drmhse.com/models/ocr-ppocrv6/v1/PP-OCRv6_small_rec.onnx", "a094aa198f9ddf29aa7f5d2dd1bc724e8dabb047ab78f401288cbbfbc62f687c"),
    ("PP-OCRv6_medium_det", "https://cdn.drmhse.com/models/ocr-ppocrv6/v1/PP-OCRv6_medium_det.onnx", "125ade0d59993f53c686fc5ce3a5a38f917297bd90b7cd0c4b1030a147e544c7"),
    ("PP-OCRv6_medium_rec", "https://cdn.drmhse.com/models/ocr-ppocrv6/v1/PP-OCRv6_medium_rec.onnx", "ed577cfca4dfa8867261b483013ed02ccddfcf48b20e11f07ac44cfcc5a75d3f"),
    ("supra-kenya-id", "https://cdn.drmhse.com/models/ocr-kenya-id/v1/model.safetensors", "a6e317fee551cf3c955d3af404b809c21ce9142dedb8fb9d91b1d24242aee3f8"),
];

fn lookup(id: &str) -> Option<(&'static str, &'static str)> {
    MANIFEST.iter().find(|(i, _, _)| *i == id).map(|(_, u, s)| (*u, *s))
}

/// Cache directory: `$PPOCR_CACHE_DIR`, else `$XDG_CACHE_HOME/ppocr-server/models`,
/// else `$HOME/.cache/ppocr-server/models`.
pub fn cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("PPOCR_CACHE_DIR") {
        return PathBuf::from(d);
    }
    let base = std::env::var("XDG_CACHE_HOME")
        .ok()
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.cache")))
        .unwrap_or_else(|| ".".to_string());
    PathBuf::from(base).join("ppocr-server").join("models")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Ensure `id` is cached + verified; return its local path. Downloads on first
/// run. A present cache file is trusted (it was verified before the atomic
/// rename); pass a fresh cache dir to force re-download.
pub fn ensure(id: &str) -> Result<PathBuf> {
    let (url, sha) = lookup(id).with_context(|| format!("unknown model id '{id}' (not in manifest)"))?;
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).with_context(|| format!("creating cache dir {}", dir.display()))?;
    let path = dir.join(format!("{id}.bin"));

    if path.exists() {
        tracing::info!(model = id, path = %path.display(), "model cached");
        return Ok(path);
    }

    tracing::info!(model = id, %url, "fetching model (first run)");
    let mut bytes = Vec::new();
    ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?
        .into_reader()
        .read_to_end(&mut bytes)
        .with_context(|| format!("downloading {url}"))?;

    let got = sha256_hex(&bytes);
    if got != sha {
        bail!("checksum mismatch for '{id}': expected {sha}, got {got}");
    }
    let tmp = dir.join(format!("{id}.tmp"));
    std::fs::write(&tmp, &bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, &path).with_context(|| format!("finalizing {}", path.display()))?;
    tracing::info!(model = id, bytes = bytes.len(), "model cached + verified");
    Ok(path)
}

/// Ensure `id` is cached and return its bytes.
pub fn ensure_bytes(id: &str) -> Result<Vec<u8>> {
    let path = ensure(id)?;
    std::fs::read(&path).with_context(|| format!("reading cached model {}", path.display()))
}
