//! Model registry + tract inference for detection and recognition.
//!
//! Models are embedded at compile time (see `embedded.rs`). PP-OCR graphs carry
//! symbolic spatial dims that tract can't unify while free, so per request we
//! fix the input to the concrete size, optimize, and cache the plan keyed by
//! (height, width). Real workloads cluster around a few sizes.

use crate::config::{DetConfig, DetParams};
use crate::embedded;
use crate::geometry::Pt;
use crate::postprocess::{db_postprocess, DetectedBox};
use crate::preprocess::preprocess;
use crate::recognize::{
    ctc_classes_ok, rec_preprocess, rectify_crop, split_word_crops, Charset, REC_HEIGHT,
};
use anyhow::{Context, Result};
use image::DynamicImage;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use tract_onnx::prelude::*;

type Runnable = RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Bounded LRU cache of compiled plans for one model, plus the parsed model.
struct Compiled {
    infer: InferenceModel,
    cache: Mutex<PlanCache>,
    label: String,
}

struct PlanCache {
    map: HashMap<(u32, u32), Arc<Runnable>>,
    order: VecDeque<(u32, u32)>,
    cap: usize,
}

impl PlanCache {
    fn new(cap: usize) -> Self {
        PlanCache {
            map: HashMap::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
        }
    }
    fn touch(&mut self, key: (u32, u32)) {
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key);
    }
    fn insert(&mut self, key: (u32, u32), plan: Arc<Runnable>) {
        self.map.insert(key, plan);
        self.touch(key);
        while self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}

impl Compiled {
    fn from_bytes(label: &str, onnx: &[u8], cap: usize) -> Result<Self> {
        let infer = tract_onnx::onnx()
            .model_for_read(&mut Cursor::new(onnx))
            .with_context(|| format!("reading onnx for {label}"))?;
        Ok(Compiled {
            infer,
            cache: Mutex::new(PlanCache::new(cap)),
            label: label.to_string(),
        })
    }

    fn plan_for(&self, h: u32, w: u32) -> Result<Arc<Runnable>> {
        let mut cache = self.cache.lock().expect("plan cache poisoned");
        if let Some(p) = cache.map.get(&(h, w)).cloned() {
            cache.touch((h, w));
            return Ok(p);
        }
        tracing::info!(model = %self.label, h, w, "compiling plan for new size");
        let plan = self
            .infer
            .clone()
            .with_input_fact(0, f32::fact([1, 3, h as usize, w as usize]).into())
            .context("setting input fact")?
            .into_optimized()
            .context("optimizing")?
            .into_runnable()
            .context("planning")?;
        let plan = Arc::new(plan);
        cache.insert((h, w), plan.clone());
        Ok(plan)
    }
}

struct DetModel {
    cfg: DetConfig,
    compiled: Compiled,
}

struct RecModel {
    id: String,
    charset: Charset,
    compiled: Compiled,
}

pub struct Engine {
    det: BTreeMap<String, Arc<DetModel>>,
    rec: BTreeMap<String, Arc<RecModel>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OcrMode {
    General,
    Document,
    KenyaId,
}

impl OcrMode {
    pub fn as_str(self) -> &'static str {
        match self {
            OcrMode::General => "general",
            OcrMode::Document => "document",
            OcrMode::KenyaId => "kenya_id",
        }
    }

    fn padded_crops(self) -> bool {
        matches!(self, OcrMode::Document | OcrMode::KenyaId)
    }

    fn split_words(self) -> bool {
        matches!(self, OcrMode::Document | OcrMode::KenyaId)
    }

    fn normalize_id_text(self) -> bool {
        matches!(self, OcrMode::KenyaId)
    }
}

/// Per-request detection overrides applied on top of yml defaults.
#[derive(Debug, Default, Clone, Copy)]
pub struct ParamOverrides {
    pub thresh: Option<f32>,
    pub box_thresh: Option<f32>,
    pub unclip_ratio: Option<f32>,
    pub limit_side_len: Option<u32>,
}

/// One recognized text region.
#[derive(Debug, Clone)]
pub struct OcrLine {
    pub points: [Pt; 4],
    pub text: String,
    pub det_score: f32,
    pub rec_score: f32,
}

impl Engine {
    pub fn load(plan_cache_cap: usize) -> Result<Self> {
        let mut det = BTreeMap::new();
        for m in embedded::det_models() {
            tracing::info!(model = m.id, "loading detection model (embedded)");
            let cfg = DetConfig::from_yml(m.id, m.yml)?;
            let compiled = Compiled::from_bytes(m.id, m.onnx, plan_cache_cap)?;
            det.insert(m.id.to_string(), Arc::new(DetModel { cfg, compiled }));
        }
        let mut rec = BTreeMap::new();
        for m in embedded::rec_models() {
            tracing::info!(model = m.id, "loading recognition model (embedded)");
            let charset = Charset::from_dict(m.charset);
            let compiled = Compiled::from_bytes(m.id, m.onnx, plan_cache_cap)?;
            rec.insert(
                m.id.to_string(),
                Arc::new(RecModel {
                    id: m.id.to_string(),
                    charset,
                    compiled,
                }),
            );
        }
        anyhow::ensure!(!det.is_empty(), "no detection models embedded");
        anyhow::ensure!(!rec.is_empty(), "no recognition models embedded");
        Ok(Engine { det, rec })
    }

    pub fn det_ids(&self) -> Vec<String> {
        self.det.keys().cloned().collect()
    }
    pub fn rec_ids(&self) -> Vec<String> {
        self.rec.keys().cloned().collect()
    }
    pub fn default_params(&self, id: &str) -> Option<DetParams> {
        self.det.get(id).map(|m| m.cfg.params)
    }
    pub fn has_det(&self, id: &str) -> bool {
        self.det.contains_key(id)
    }
    pub fn has_rec(&self, id: &str) -> bool {
        self.rec.contains_key(id)
    }

    /// Compile and cache the fixed rec-width plans for every recognition model,
    /// so the first real request doesn't pay compile latency. Runs the bucket
    /// compiles in parallel. Detection plans are size-dependent, so not warmed.
    pub fn prewarm(&self) {
        let jobs: Vec<(Arc<RecModel>, u32)> = self
            .rec
            .values()
            .flat_map(|m| {
                crate::recognize::width_buckets()
                    .iter()
                    .map(move |&w| (m.clone(), w))
            })
            .collect();
        jobs.par_iter().for_each(|(m, w)| {
            if let Err(e) = m.compiled.plan_for(REC_HEIGHT, *w) {
                tracing::warn!(model = %m.id, w, "prewarm failed: {e:#}");
            }
        });
        tracing::info!(plans = jobs.len(), "recognition plans pre-warmed");
    }

    fn resolve_params(&self, base: DetParams, ov: ParamOverrides) -> DetParams {
        let mut p = base;
        if let Some(v) = ov.thresh {
            p.thresh = v;
        }
        if let Some(v) = ov.box_thresh {
            p.box_thresh = v;
        }
        if let Some(v) = ov.unclip_ratio {
            p.unclip_ratio = v;
        }
        if let Some(v) = ov.limit_side_len {
            p.limit_side_len = v;
        }
        p
    }

    /// Detection only.
    pub fn detect(
        &self,
        det_id: &str,
        img: &DynamicImage,
        ov: ParamOverrides,
    ) -> Result<(Vec<DetectedBox>, DetParams)> {
        let m = self
            .det
            .get(det_id)
            .with_context(|| format!("unknown det model '{det_id}'"))?
            .clone();
        let params = self.resolve_params(m.cfg.params, ov);
        let pre = preprocess(img, &params, &m.cfg.normalize);
        let (net_w, net_h) = (pre.resize_w as usize, pre.resize_h as usize);

        let plan = m.compiled.plan_for(pre.resize_h, pre.resize_w)?;
        let result = plan
            .run(tvec!(pre.tensor.clone().into()))
            .context("detection inference failed")?;
        let view = result[0]
            .to_array_view::<f32>()
            .context("det output not f32")?;
        let prob: Vec<f32> = view.iter().cloned().collect();
        anyhow::ensure!(prob.len() == net_w * net_h, "unexpected det output size");
        let boxes = db_postprocess(&prob, net_w, net_h, &pre, &params);
        Ok((boxes, params))
    }

    /// Recognize a single already-cropped/upright line image.
    pub fn recognize_image(&self, rec_id: &str, line: &image::RgbImage) -> Result<(String, f32)> {
        let m = self
            .rec
            .get(rec_id)
            .with_context(|| format!("unknown rec model '{rec_id}'"))?
            .clone();
        self.recognize_strip_raw(&m, line)
    }

    fn recognize_strip_raw(&self, m: &RecModel, line: &image::RgbImage) -> Result<(String, f32)> {
        let (tensor, pad_w) = rec_preprocess(line);
        let plan = m.compiled.plan_for(REC_HEIGHT, pad_w)?;
        let result = plan
            .run(tvec!(tensor.into()))
            .context("recognition inference failed")?;
        let view = result[0]
            .to_array_view::<f32>()
            .context("rec output not f32")?;
        let shape = view.shape();
        anyhow::ensure!(shape.len() == 3, "rec output rank != 3");
        let (t, c) = (shape[1], shape[2]);
        ctc_classes_ok(&m.charset, c)
            .with_context(|| format!("charset/model class mismatch for {}", m.id))?;
        let probs: Vec<f32> = view.iter().cloned().collect();
        Ok(m.charset.ctc_decode(&probs, t, c))
    }

    fn recognize_strip(
        &self,
        m: &RecModel,
        line: &image::RgbImage,
        split_words: bool,
    ) -> Result<(String, f32)> {
        let (whole_text, whole_score) = self.recognize_strip_raw(m, line)?;
        if !split_words || whole_text.contains(' ') {
            return Ok((whole_text, whole_score));
        }

        let word_crops = split_word_crops(line);
        if word_crops.is_empty() {
            return Ok((whole_text, whole_score));
        }

        let mut words = Vec::with_capacity(word_crops.len());
        let mut weighted_score = 0.0f32;
        let mut total_chars = 0usize;
        for crop in &word_crops {
            let (word, score) = self.recognize_strip_raw(m, crop)?;
            let word = word.trim().to_string();
            if word.is_empty() {
                return Ok((whole_text, whole_score));
            }
            let chars = word.chars().count().max(1);
            weighted_score += score * chars as f32;
            total_chars += chars;
            words.push(word);
        }

        let segmented = words.join(" ");
        let segmented_compact: String = segmented.chars().filter(|c| !c.is_whitespace()).collect();
        let whole_compact: String = whole_text.chars().filter(|c| !c.is_whitespace()).collect();
        let segmented_score = weighted_score / total_chars.max(1) as f32;

        if segmented_compact.len() >= whole_compact.len().saturating_sub(2)
            && segmented_score + 0.08 >= whole_score
        {
            Ok((segmented, segmented_score))
        } else {
            Ok((whole_text, whole_score))
        }
    }

    fn normalize_line_text(text: &str) -> String {
        let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let key = collapsed
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
            .to_uppercase();

        let normalized = match key.as_str() {
            "JAMHURIYAKENYA" | "JAMHURIYAUKENYA" | "JAMHURYAUKENYA" | "JAMHURDYAUKENYA" => {
                Some("JAMHURI YA KENYA")
            }
            "REPUBLICOFKENYA" | "REPURLICOFKENYA" | "REPURLICOEKENYA" => Some("REPUBLIC OF KENYA"),
            "SERIALNUMBER" => Some("SERIAL NUMBER"),
            "SERIALNUMBER:" => Some("SERIAL NUMBER:"),
            "IDNUMBER" => Some("ID NUMBER"),
            "IDNUMBER:" => Some("ID NUMBER:"),
            "FULLNAMES" | "FULNAMES" => Some("FULL NAMES"),
            "FULLNAMES:" | "FULNAMES:" => Some("FULL NAMES:"),
            "DATEOFBIRTH" | "DATEOEBIRTH" | "DATEOERIRTH" => Some("DATE OF BIRTH"),
            "DISTRICTOFBIRTH" | "DISTRICTOEBIRTH" => Some("DISTRICT OF BIRTH"),
            "PLACEOFISSUE" | "PLACEOEISSUE" => Some("PLACE OF ISSUE"),
            "DATEOFISSUE" | "DATEOEISSUE" => Some("DATE OF ISSUE"),
            _ => None,
        };

        if let Some(value) = normalized {
            return value.to_string();
        }

        if collapsed.contains('.') {
            let digits: String = collapsed.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() == 8 {
                return format!("{}.{}.{}", &digits[0..2], &digits[2..4], &digits[4..8]);
            }
        }

        collapsed
    }

    /// Full pipeline: detect, rectify each box, recognize, in reading order.
    pub fn ocr(
        &self,
        det_id: &str,
        rec_id: &str,
        img: &DynamicImage,
        ov: ParamOverrides,
        min_rec_score: f32,
        mode: OcrMode,
    ) -> Result<(Vec<OcrLine>, DetParams)> {
        let rec = self
            .rec
            .get(rec_id)
            .with_context(|| format!("unknown rec model '{rec_id}'"))?
            .clone();

        let (mut boxes, params) = self.detect(det_id, img, ov)?;
        // Reading order: top-to-bottom, then left-to-right.
        boxes.sort_by(|a, b| {
            let ay = a.points.iter().map(|p| p.1).fold(f32::MAX, f32::min);
            let by = b.points.iter().map(|p| p.1).fold(f32::MAX, f32::min);
            let ax = a.points.iter().map(|p| p.0).fold(f32::MAX, f32::min);
            let bx = b.points.iter().map(|p| p.0).fold(f32::MAX, f32::min);
            // group rows within ~10px, then by x
            if (ay - by).abs() <= 10.0 {
                ax.partial_cmp(&bx).unwrap()
            } else {
                ay.partial_cmp(&by).unwrap()
            }
        });

        let src = img.to_rgb8();
        // Recognition is per-box and independent -> run lines in parallel.
        // Order is preserved by collecting positionally, then dropping filtered.
        let lines: Vec<OcrLine> = boxes
            .par_iter()
            .map(|b| -> Result<Option<OcrLine>> {
                let crop = rectify_crop(&src, &b.points, mode.padded_crops());
                let (text, rec_score) = self.recognize_strip(&rec, &crop, mode.split_words())?;
                let text = if mode.normalize_id_text() {
                    Self::normalize_line_text(&text)
                } else {
                    text
                };
                Ok(if text.is_empty() || rec_score < min_rec_score {
                    None
                } else {
                    Some(OcrLine {
                        points: b.points,
                        text,
                        det_score: b.score,
                        rec_score,
                    })
                })
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok((lines, params))
    }
}
