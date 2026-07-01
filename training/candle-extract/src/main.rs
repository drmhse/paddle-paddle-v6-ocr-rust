//! Standalone probe: does candle run Supra-50M (fused ID LoRA) correctly on CPU?
//! Reads OCR text on stdin, prints the raw model output. Validates the path
//! before porting into ocr-rs.
use anyhow::{Error, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use std::io::Read;
use tokenizers::Tokenizer;

const SYSTEM: &str = "You extract fields from the OCR text of a Kenyan National ID card. The text is noisy and labels may be garbled or separated from their values (values can appear before the labels, and labels can be grouped together). Use the field meaning to assign each value: a 3-word ALL-CAPS personal name is FULL NAMES; 8-9 digit numbers are the serial/ID numbers; MALE/FEMALE is SEX; a place name is district/place of issue. Copy text values verbatim. Normalize dates to DD.MM.YYYY. Use null for a field that is genuinely absent. Output one JSON object, nothing else.";

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "../supra_candle".into());
    let dtype = match std::env::args().nth(2).as_deref() {
        Some("f16") => DType::F16,
        _ => DType::F32,
    };
    let device = Device::Cpu;

    let tokenizer = Tokenizer::from_file(format!("{dir}/tokenizer.json")).map_err(Error::msg)?;
    let cfg: LlamaConfig = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json"))?)?;
    let config = cfg.into_config(false);
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[format!("{dir}/model.safetensors")], dtype, &device)?
    };
    let mut cache = Cache::new(true, dtype, &config, &device)?;
    let llama = Llama::load(vb, &config)?;
    eprintln!("model loaded");

    let mut ocr = String::new();
    std::io::stdin().read_to_string(&mut ocr)?;
    let prompt = format!(
        "{SYSTEM}\n\nOCR lines:\n{}\n\nReturn the ID fields as JSON.\n",
        ocr.trim()
    );

    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(Error::msg)?
        .get_ids()
        .to_vec();
    let eos: u32 = 2; // Supra </s>
    let start = std::time::Instant::now();
    let mut out: Vec<u32> = vec![];
    let mut index_pos = 0usize;
    for index in 0..220 {
        let (ctx_size, ctx_index) = if index > 0 { (1, index_pos) } else { (tokens.len(), 0) };
        let ctx = &tokens[tokens.len() - ctx_size..];
        let input = Tensor::new(ctx, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, ctx_index, &mut cache)?.squeeze(0)?;
        index_pos += ctx.len();
        let next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        if next == eos {
            break;
        }
        tokens.push(next);
        out.push(next);
    }
    let text = tokenizer.decode(&out, true).map_err(Error::msg)?;
    eprintln!("[{} tok in {:.2}s]", out.len(), start.elapsed().as_secs_f32());
    println!("{text}");
    Ok(())
}
