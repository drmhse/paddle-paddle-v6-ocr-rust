//! Optional structured-understanding stage (feature = "understanding").
//!
//! Turns the `kenya_id` OCR text into a structured JSON object using Supra-50M
//! (LoRA-fused for this schema) running on candle — pure Rust, CPU, no C++/ORT.
//! Weights + tokenizer are embedded (`include_bytes!`), so there is no runtime
//! sidecar — same self-contained property as the OCR models.

use anyhow::{Error, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use serde_json::{Map, Value};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

// Big weights are fetched on first run + cached (see remote.rs); the small
// tokenizer + config stay embedded.
const MODEL_ID: &str = "supra-kenya-id";
const TOKENIZER_BYTES: &[u8] = include_bytes!("../models/supra/tokenizer.json");
const CONFIG_JSON: &str = include_str!("../models/supra/config.json");

const SYSTEM: &str = "You extract fields from the OCR text of a Kenyan National ID card. The text is noisy and labels may be garbled or separated from their values (values can appear before the labels, and labels can be grouped together). Use the field meaning to assign each value: a 3-word ALL-CAPS personal name is FULL NAMES; 8-9 digit numbers are the serial/ID numbers; MALE/FEMALE is SEX; a place name is district/place of issue. Copy text values verbatim. Normalize dates to DD.MM.YYYY. Use null for a field that is genuinely absent. Output one JSON object, nothing else.";

const FIELDS: [&str; 8] = [
    "serial_number", "id_number", "full_names", "date_of_birth",
    "sex", "district_of_birth", "place_of_issue", "date_of_issue",
];
const EOS: u32 = 2; // Supra </s>
const MAX_NEW_TOKENS: usize = 256;

static MODEL: OnceLock<Understanding> = OnceLock::new();

pub struct Understanding {
    llama: Llama,
    config: Config,
    tokenizer: Tokenizer,
    device: Device,
}

/// Load once at startup. Subsequent calls are no-ops.
pub fn init() -> Result<()> {
    if MODEL.get().is_some() {
        return Ok(());
    }
    let u = Understanding::load()?;
    let _ = MODEL.set(u);
    Ok(())
}

/// Extract structured ID fields from OCR text. Returns null-filled schema on any
/// failure so the HTTP layer always gets a well-formed object.
pub fn extract(ocr_text: &str) -> Value {
    match MODEL.get() {
        Some(m) => m.extract(ocr_text).unwrap_or_else(|_| null_fields()),
        None => null_fields(),
    }
}

impl Understanding {
    fn load() -> Result<Self> {
        let device = Device::Cpu;
        let tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES).map_err(Error::msg)?;
        let cfg: LlamaConfig = serde_json::from_str(CONFIG_JSON)?;
        let config = cfg.into_config(false);
        let weights = crate::remote::ensure(MODEL_ID)?; // fetch + cache on first run
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F16, &device)? };
        let llama = Llama::load(vb, &config)?;
        Ok(Self { llama, config, tokenizer, device })
    }

    fn extract(&self, ocr_text: &str) -> Result<Value> {
        let prompt = format!(
            "{SYSTEM}\n\nOCR lines:\n{}\n\nReturn the ID fields as JSON.\n",
            ocr_text.trim()
        );
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
        let mut cache = Cache::new(true, DType::F16, &self.config, &self.device)?;
        let mut out: Vec<u32> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();
        let mut index_pos = 0usize;
        for index in 0..MAX_NEW_TOKENS {
            let (ctx_size, ctx_index) = if index > 0 { (1, index_pos) } else { (tokens.len(), 0) };
            let ctx = &tokens[tokens.len() - ctx_size..];
            let input = Tensor::new(ctx, &self.device)?.unsqueeze(0)?;
            let logits = self
                .llama
                .forward(&input, ctx_index, &mut cache)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            index_pos += ctx.len();
            let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            if next == EOS {
                break;
            }
            // confidence = softmax probability of the chosen token
            let p = candle_nn::ops::softmax(&logits, D::Minus1)?
                .get(next as usize)?
                .to_scalar::<f32>()?;
            tokens.push(next);
            out.push(next);
            probs.push(p);
        }
        self.parse_with_conf(&out, &probs)
    }

    /// Decode the output, parse the JSON object, and attach a per-field
    /// confidence = the lowest token probability across that field's value
    /// (weakest-link). Returns `{field: {value, confidence}}`.
    fn parse_with_conf(&self, out: &[u32], probs: &[f32]) -> Result<Value> {
        let full = self.tokenizer.decode(out, true).map_err(Error::msg)?;
        // byte span each token contributes to `full` (incremental decode)
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(out.len());
        let mut prev_len = 0usize;
        for i in 0..out.len() {
            let cur = self.tokenizer.decode(&out[..=i], true).map_err(Error::msg)?;
            let start = prev_len.min(cur.len());
            spans.push((start, cur.len()));
            prev_len = cur.len();
        }
        let obj = first_json_object(&full);
        let mut m = Map::new();
        for f in FIELDS {
            let value = obj.as_ref().and_then(|o| o.get(f)).cloned().unwrap_or(Value::Null);
            let confidence = match &value {
                Value::String(s) if !s.is_empty() => locate_value(&full, f, s)
                    .and_then(|sp| min_overlap_prob(sp, &spans, probs))
                    .map(|p| Value::from((p as f64 * 100.0).round() / 100.0))
                    .unwrap_or(Value::Null),
                _ => Value::Null,
            };
            let mut fm = Map::new();
            fm.insert("value".to_string(), value);
            fm.insert("confidence".to_string(), confidence);
            m.insert(f.to_string(), Value::Object(fm));
        }
        Ok(Value::Object(m))
    }
}

/// Byte span of `value` as it appears right after `"field"` in the raw output.
fn locate_value(full: &str, field: &str, value: &str) -> Option<(usize, usize)> {
    let key = format!("\"{field}\"");
    let kpos = full.find(&key)?;
    let from = kpos + key.len();
    let rel = full[from..].find(value)?;
    let start = from + rel;
    Some((start, start + value.len()))
}

/// Lowest token probability among tokens overlapping the value's byte span.
fn min_overlap_prob(span: (usize, usize), spans: &[(usize, usize)], probs: &[f32]) -> Option<f32> {
    let (vs, ve) = span;
    let mut min_p = f32::MAX;
    for (i, &(ts, te)) in spans.iter().enumerate() {
        if ts < ve && te > vs {
            if let Some(&p) = probs.get(i) {
                min_p = min_p.min(p);
            }
        }
    }
    (min_p != f32::MAX).then_some(min_p)
}

fn null_fields() -> Value {
    let mut m = Map::new();
    for f in FIELDS {
        let mut fm = Map::new();
        fm.insert("value".to_string(), Value::Null);
        fm.insert("confidence".to_string(), Value::Null);
        m.insert(f.to_string(), Value::Object(fm));
    }
    Value::Object(m)
}

/// Take the first balanced `{...}` from the model output (guards against any
/// trailing/looping output).
fn first_json_object(text: &str) -> Option<Value> {
    let b = text.as_bytes();
    let mut depth = 0i32;
    let mut start = None;
    for (i, &c) in b.iter().enumerate() {
        if c == b'{' {
            if depth == 0 {
                start = Some(i);
            }
            depth += 1;
        } else if c == b'}' {
            depth -= 1;
            if depth == 0 {
                if let Some(s) = start {
                    return serde_json::from_str(&text[s..=i]).ok();
                }
            }
        }
    }
    None
}
