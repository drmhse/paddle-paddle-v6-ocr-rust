//! Per-model metadata, selected via cargo size features (`tiny`/`small`/`medium`).
//!
//! The big ONNX weights are no longer embedded — they're fetched on first run
//! and cached (see `remote.rs`), keyed by model `id`. Only the small text
//! configs (detection `inference.yml`, recognition `charset.txt`) stay embedded.

pub struct EmbeddedDet {
    pub id: &'static str,
    pub yml: &'static str,
}

pub struct EmbeddedRec {
    pub id: &'static str,
    /// newline-separated character dictionary (without blank/space sentinels)
    pub charset: &'static str,
}

macro_rules! det {
    ($id:literal, $dir:literal) => {
        EmbeddedDet {
            id: $id,
            yml: include_str!(concat!("../models/", $dir, "/inference.yml")),
        }
    };
}

macro_rules! rec {
    ($id:literal, $dir:literal) => {
        EmbeddedRec {
            id: $id,
            charset: include_str!(concat!("../models/", $dir, "/charset.txt")),
        }
    };
}

pub fn det_models() -> Vec<EmbeddedDet> {
    let mut v = Vec::new();
    #[cfg(feature = "tiny")]
    v.push(det!("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_det"));
    #[cfg(feature = "small")]
    v.push(det!("PP-OCRv6_small_det", "PP-OCRv6_small_det"));
    #[cfg(feature = "medium")]
    v.push(det!("PP-OCRv6_medium_det", "PP-OCRv6_medium_det"));
    v
}

pub fn rec_models() -> Vec<EmbeddedRec> {
    let mut v = Vec::new();
    #[cfg(feature = "tiny")]
    v.push(rec!("PP-OCRv6_tiny_rec", "PP-OCRv6_tiny_rec"));
    #[cfg(feature = "small")]
    v.push(rec!("PP-OCRv6_small_rec", "PP-OCRv6_small_rec"));
    #[cfg(feature = "medium")]
    v.push(rec!("PP-OCRv6_medium_rec", "PP-OCRv6_medium_rec"));
    v
}
