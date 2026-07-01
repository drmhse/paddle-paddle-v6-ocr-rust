# training — understanding-stage LoRA pipeline

How the `kenya_id` structured-extraction model is produced. The `ocr-rs` server
only *consumes* the fused weights (fetched from the CDN at runtime, see
[`../src/remote.rs`](../src/remote.rs)); this directory is the source that
produces them, kept for reproducibility.

**Model:** [`SupraLabs/Supra-1.5-50M-Instruct-exp`](https://huggingface.co/SupraLabs/Supra-1.5-50M-Instruct-exp)
— a 50M-param Llama — LoRA-fine-tuned on **synthetic** OCR→JSON pairs, then
fused to f16 and run on `candle` in the Rust server.

## Why synthetic data

No real logbooks/IDs are used. `gen_id.py` mints `(OCR line-list → ground-truth
JSON)` pairs that model the *actual* noise `ppocr-server` produces in `kenya_id`
mode — grounded in a real OCR dump:

- values appear **before** their labels; labels cluster together
- headers/labels garbled (`REPURLICOFKENYA`, `DATEOEAIRTH`), spaces dropped
- noisy dates (`25051993`) whose target is **normalized** to `DD.MM.YYYY`
- stray junk lines (`G`, `HOLDER'S SIGN.`, …)
- names/numbers copied **verbatim** in the target (a 50M model can't safely
  un-garble proper nouns — leave that to a downstream fuzzy-match)

Modeling that layout noise is what took the real card from 3/8 to 8/8 fields.

## Prerequisites

```bash
python3 -m venv .venv
./.venv/bin/pip install mlx-lm huggingface_hub boto3 fastapi "uvicorn[standard]" python-multipart requests
```

## Pipeline

```bash
# 1. Patch the base model dir (Supra ships tokenizer_class="TokenizersBackend",
#    which transformers/mlx_lm can't load). Snapshot Supra from the HF cache into
#    ./supra_local and rewrite tokenizer_config.json's tokenizer_class to
#    "PreTrainedTokenizerFast" (it already ships a standard tokenizer.json):
#      from huggingface_hub import snapshot_download; import json, shutil, os
#      src = snapshot_download("SupraLabs/Supra-1.5-50M-Instruct-exp")
#      os.makedirs("supra_local", exist_ok=True)
#      for f in ("model.safetensors","config.json","generation_config.json","tokenizer.json"):
#          shutil.copy(f"{src}/{f}", f"supra_local/{f}")
#      tc = json.load(open(f"{src}/tokenizer_config.json"))
#      tc["tokenizer_class"] = "PreTrainedTokenizerFast"; tc.pop("backend", None)
#      json.dump(tc, open("supra_local/tokenizer_config.json","w"), indent=2)

# 2. Generate synthetic data (1600 train + 160 valid)
./.venv/bin/python gen_id.py            # -> data_id/{train,valid}.jsonl

# 3. Supra has no chat template -> train on raw `text` format
#    ({system}\n\n{user}\n{completion}</s>) written to data_id_text/
#    (conv_pc.py has the prompt/completion variant; the text variant is inline)

# 4. LoRA fine-tune (MLX, ~5 min on an M-series Mac)
./.venv/bin/python -m mlx_lm lora --model ./supra_local --train --data ./data_id_text \
  --iters 700 --batch-size 8 --num-layers 8 \
  --adapter-path ./adapters_supra --steps-per-eval 350
#    -> 0.6M trainable params (1.16%), 2.4 MB adapter, val loss ~0.50

# 5. Evaluate on a held-out (unseen-seed) synthetic set
./.venv/bin/python eval_any.py gen_id ./supra_local ./adapters_supra 60
#    -> ~92% field accuracy, 60/60 valid JSON

# 6. Fuse LoRA into the base, convert to f16 safetensors for candle
./.venv/bin/python -m mlx_lm fuse --model ./supra_local --adapter-path ./adapters_supra \
  --save-path ./supra_id_fused
#    then f16 + materialize tied lm_head -> the safetensors uploaded to the CDN
```

## Files

| file | role |
|------|------|
| `gen_id.py`   | synthetic National-ID data generator (the artifact you iterate) |
| `eval_any.py` | held-out scorer: `eval_any.py <gen_module> <model_dir> <adapter\|none> [N]` |
| `conv_pc.py`  | chat → prompt/completion converter (for template-less models) |
| `sidecar.py`  | FastAPI dev harness: browser → ppocr-server → MLX LoRA (prototyping only) |
| `candle-extract/` | standalone Rust probe that validated candle inference before porting into `ocr-rs` |

Generated artifacts (`.venv/`, `data_*/`, `adapters*/`, `supra_*/`, `*.log`,
`candle-extract/target/`) are gitignored — regenerate them with the steps above.

## Distribution

The fused f16 weights live at
`https://cdn.drmhse.com/models/ocr-kenya-id/v1/model.safetensors` (Cloudflare R2
`drmhse` bucket). The 2.4 MB LoRA adapter is archived alongside as
`lora-adapter.safetensors`. To publish a new version, bump the version prefix
and update the manifest checksum in [`../src/remote.rs`](../src/remote.rs).
