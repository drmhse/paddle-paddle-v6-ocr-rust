//! Compile-time embedded models. Selected via cargo features so you can build
//! a lean single binary (e.g. `--no-default-features --features tiny`).
//!
//! Each size feature embeds both the detection and recognition model of that
//! size. The whole thing is self-contained: no sidecar files at runtime.

pub struct EmbeddedDet {
    pub id: &'static str,
    pub onnx: &'static [u8],
    pub yml: &'static str,
}

pub struct EmbeddedRec {
    pub id: &'static str,
    pub onnx: &'static [u8],
    /// newline-separated character dictionary (without blank/space sentinels)
    pub charset: &'static str,
}

macro_rules! onnx_file {
    ($dir:literal) => {
        include_bytes!(concat!("../models/", $dir, "/inference.onnx"))
    };
}

macro_rules! det {
    ($id:literal, $dir:literal) => {
        EmbeddedDet {
            id: $id,
            onnx: onnx_file!($dir),
            yml: include_str!(concat!("../models/", $dir, "/inference.yml")),
        }
    };
}

macro_rules! rec {
    ($id:literal, $dir:literal) => {
        EmbeddedRec {
            id: $id,
            onnx: onnx_file!($dir),
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
