//! Per-model defaults parsed from PaddleOCR `inference.yml`.

use anyhow::{Context, Result};
use serde::Deserialize;

/// Detection parameters. Defaults come from each model's `inference.yml`
/// (`PostProcess` / `PreProcess`); any field can be overridden per request.
#[derive(Debug, Clone, Copy)]
pub struct DetParams {
    /// Binarization threshold applied to the probability map.
    pub thresh: f32,
    /// Minimum mean-probability inside a box for it to be kept.
    pub box_thresh: f32,
    /// Box expansion ratio (Vatti clipping approximation).
    pub unclip_ratio: f32,
    /// Resize so the chosen side is bounded by this length (multiple of 32).
    pub limit_side_len: u32,
    /// `Max` => longest side <= limit; `Min` => shortest side >= limit.
    pub limit_type: LimitType,
    /// Drop boxes whose min side (px, in network space) is below this.
    pub min_size: f32,
    /// Hard cap on number of candidate contours considered.
    pub max_candidates: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LimitType {
    Max,
    Min,
}

/// ImageNet-style normalization (channel order matches PaddleOCR `img_mode`).
#[derive(Debug, Clone, Copy)]
pub struct Normalize {
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub scale: f32,
    /// true => feed BGR channel order to the network (PP-OCR default).
    pub bgr: bool,
}

#[derive(Debug, Clone)]
pub struct DetConfig {
    pub params: DetParams,
    pub normalize: Normalize,
}

// ---- minimal serde view over inference.yml ----

#[derive(Deserialize)]
struct RawYml {
    #[serde(rename = "PostProcess")]
    post_process: Option<RawPost>,
    #[serde(rename = "PreProcess")]
    pre_process: Option<RawPre>,
}

#[derive(Deserialize)]
struct RawPost {
    thresh: Option<f32>,
    box_thresh: Option<f32>,
    unclip_ratio: Option<f32>,
    max_candidates: Option<usize>,
}

#[derive(Deserialize)]
struct RawPre {
    transform_ops: Option<Vec<serde_yaml::Value>>,
}

impl DetConfig {
    /// Parse detection defaults from a model's `inference.yml` contents.
    pub fn from_yml(id: &str, yml: &str) -> Result<Self> {
        // Defaults match PaddleOCR's documented det defaults; overridden by yml.
        let mut params = DetParams {
            thresh: 0.3,
            box_thresh: 0.6,
            unclip_ratio: 1.5,
            limit_side_len: 960,
            limit_type: LimitType::Max,
            min_size: 3.0,
            max_candidates: 1000,
        };
        let mut normalize = Normalize {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            scale: 1.0 / 255.0,
            bgr: true,
        };

        let raw: RawYml =
            serde_yaml::from_str(yml).with_context(|| format!("parsing yml for {id}"))?;
        if let Some(p) = raw.post_process {
            if let Some(v) = p.thresh {
                params.thresh = v;
            }
            if let Some(v) = p.box_thresh {
                params.box_thresh = v;
            }
            if let Some(v) = p.unclip_ratio {
                params.unclip_ratio = v;
            }
            if let Some(v) = p.max_candidates {
                params.max_candidates = v;
            }
        }
        if let Some(pre) = raw.pre_process {
            apply_preprocess_yml(pre, &mut normalize);
        }

        Ok(DetConfig { params, normalize })
    }
}

/// Pull NormalizeImage mean/std/scale + DecodeImage channel order out of the
/// transform_ops list. Anything missing keeps the default.
fn apply_preprocess_yml(pre: RawPre, norm: &mut Normalize) {
    let Some(ops) = pre.transform_ops else { return };
    for op in ops {
        let Some(map) = op.as_mapping() else { continue };
        for (k, v) in map {
            match k.as_str() {
                Some("NormalizeImage") => {
                    if let Some(m) = v.get("mean").and_then(parse_vec3) {
                        norm.mean = m;
                    }
                    if let Some(s) = v.get("std").and_then(parse_vec3) {
                        norm.std = s;
                    }
                    if let Some(scale) = v.get("scale") {
                        norm.scale = parse_scale(scale).unwrap_or(norm.scale);
                    }
                }
                Some("DecodeImage") => {
                    if let Some(mode) = v.get("img_mode").and_then(|m| m.as_str()) {
                        norm.bgr = mode.eq_ignore_ascii_case("BGR");
                    }
                }
                _ => {}
            }
        }
    }
}

fn parse_vec3(v: &serde_yaml::Value) -> Option<[f32; 3]> {
    let seq = v.as_sequence()?;
    if seq.len() != 3 {
        return None;
    }
    let mut out = [0f32; 3];
    for (i, e) in seq.iter().enumerate() {
        out[i] = e.as_f64()? as f32;
    }
    Some(out)
}

/// PaddleOCR writes scale either as a float or as the string "1./255.".
fn parse_scale(v: &serde_yaml::Value) -> Option<f32> {
    if let Some(f) = v.as_f64() {
        return Some(f as f32);
    }
    let s = v.as_str()?.trim();
    if let Some((num, den)) = s.split_once('/') {
        let num: f32 = num.trim().trim_end_matches('.').parse().ok()?;
        let den: f32 = den.trim().trim_end_matches('.').parse().ok()?;
        if den != 0.0 {
            return Some(num / den);
        }
    }
    s.trim_end_matches('.').parse().ok()
}

