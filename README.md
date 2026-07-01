# ppocr-server

A tiny, cross-platform OCR server for **PP-OCRv6** — text **detection +
recognition**. Inference runs on [`tract`](https://github.com/sonos/tract), a
pure-Rust ONNX engine (no ONNX Runtime, no C++). Model weights are **fetched
from a CDN on first run and cached locally**, so the binary is ~19 MB and starts
in ~0.5 s once warm. An optional [understanding stage](#understanding-stage)
turns OCR text into structured JSON.

## Quickstart

```bash
cargo build --release        # all model sizes
./target/release/ppocr-server            # 0.0.0.0:8080  (first run fetches weights)
OCR_PORT=9000 ./ppocr-server             # override port
```

Flags / env: `--host`/`OCR_HOST`, `--port`/`OCR_PORT`, `--plan-cache`/`OCR_PLAN_CACHE`
(distinct input sizes kept compiled per model, LRU; default 16), `--no-prewarm`.

Built-in pages (served from the binary, no sidecar):
- `GET /` — interactive demo: upload an image, pick models/params, see boxes + text.
- `GET /docs` — Swagger UI · `GET /openapi.json` — OpenAPI 3.0 spec.

**Size features** select which detection **and** recognition models are
available (and thus fetched on first run). The binary size is the same either
way — only the small text configs are embedded, weights are fetched:

```bash
cargo build --release --no-default-features --features tiny            # ~6 MB fetched
cargo build --release --no-default-features --features "tiny,small"
cargo build --release                                                  # all (~168 MB fetched)
```

| size feature | det | rec | charset |
|------|-----|-----|---------|
| `tiny`   | PP-OCRv6_tiny_det (1.7 MB)  | PP-OCRv6_tiny_rec (4.3 MB)  | 6,904 chars |
| `small`  | PP-OCRv6_small_det (9.5 MB) | PP-OCRv6_small_rec (20 MB)  | 18,708 chars |
| `medium` | PP-OCRv6_medium_det (59 MB) | PP-OCRv6_medium_rec (73 MB) | 18,708 chars |

## Models & cache

On startup the server ensures each required model is cached, downloading from
`cdn.drmhse.com` and **verifying SHA-256** if not. The manifest (ids, URLs,
checksums) lives in [`src/remote.rs`](src/remote.rs).

- **Cache dir**: `$PPOCR_CACHE_DIR`, else `$XDG_CACHE_HOME/ppocr-server/models`,
  else `~/.cache/ppocr-server/models`. Downloads use a temp file + atomic rename,
  so a present cache file is always complete + verified — restarts skip
  re-hashing.
- **First run needs network**; subsequent runs are fully offline. For air-gapped
  deploys, pre-seed the cache dir with `<id>.bin` files (ids from the manifest).
- **Updating a model**: bump its manifest entry (URL + checksum) and ship a new
  binary — the new checksum invalidates the old cache file automatically.

## API

### `GET /health` → `ok`

### `GET /v1/models`
Lists detection models (with default params) and recognition models.

### `POST /v1/ocr` — full text extraction (detect → recognize)
Body = raw image bytes. Query params (all optional except model selection when
more than one size is built):

| param | meaning | default |
|-------|---------|---------|
| `det_model` | detection model id (or `model`) | required if >1 |
| `rec_model` | recognition model id | required if >1 |
| `thresh`, `box_thresh`, `unclip_ratio`, `limit_side_len` | detection tuning | per-model yml |
| `min_rec_score` | drop lines below this recognition confidence | 0.0 |
| `mode` | `general` / `document` / `kenya_id` / `kenya_logbook` post-processing | `general` |
| `understand` | run the [understanding stage](#understanding-stage) (needs the feature + `mode=kenya_id`) | `false` |

```bash
curl -s -X POST \
 "http://localhost:8080/v1/ocr?det_model=PP-OCRv6_medium_det&rec_model=PP-OCRv6_medium_rec" \
 --data-binary @page.jpg | jq
```
```json
{
  "det_model": "PP-OCRv6_medium_det",
  "rec_model": "PP-OCRv6_medium_rec",
  "image": { "width": 640, "height": 200 },
  "num_lines": 2,
  "elapsed_ms": 380,
  "text": "Hello OCR World\nPP-OCRv6 detection test",
  "lines": [
    { "points": [[x,y],...4], "text": "Hello OCR World", "det_score": 0.97, "rec_score": 0.99 }
  ]
}
```
`text` is all lines joined in reading order (top→bottom, left→right); `points`
are quad corners (tl,tr,br,bl) in original-image pixels.

### `POST /v1/detect` — detection only
Boxes + scores, no text. Same detection params; model via `model`/`det_model`.

### `POST /v1/recognize` — recognition only
Body = a single **already-cropped, upright** line image → `{ text, score }`.
Model via `model`/`rec_model`.

## Understanding stage

Optional (`--features understanding`). Turns `kenya_id` OCR text into structured
JSON by running a LoRA-fused **Supra-50M** model on
[`candle`](https://github.com/huggingface/candle) (pure-Rust CPU inference). The
weights fetch + cache on first run like the OCR models; off by default so the
plain OCR build stays lean.

```bash
just build-understanding       # native
just linux-x64-understanding   # cross-compile for the server

curl -X POST "http://127.0.0.1:8080/v1/ocr?mode=kenya_id&understand=true\
&det_model=PP-OCRv6_medium_det&rec_model=PP-OCRv6_medium_rec" \
  --data-binary @id.jpg -H "Content-Type: application/octet-stream"
```

The response gains a `fields` object — each field is `{ value, confidence }`
(confidence = the model's lowest token probability across that value, 0–1). The
demo page at `/` has an "Understand → structured JSON" toggle.

- **No build-time weights** — the 136 MB Supra model is fetched + SHA-256-verified
  on first run; only the small tokenizer/config are embedded.
- **Not pure-Rust** — tokenizers pulls C/C++ deps (onig, esaxx); zig cross-compiles
  them (gnu Linux targets verified; musl-static untested).
- **Resources** — ~136 MB added to the cache, ~1.2 GB extra RAM, ~0.7 s/request
  (CPU, no GPU). Currently wired for `kenya_id` only.
- How the model is produced (synthetic-data LoRA pipeline): [`training/`](training/).

## Performance

Tuning for multi-line pages:

- **Parallel recognition** (`rayon`) — 28-line page: recognition ~10.2 s → ~3.0 s (3.3×).
- **Fixed rec-width buckets** — crops snap to one of 8 widths (160…2048), so only
  a handful of plans compile; near-100% plan-cache hits.
- **Startup pre-warm** — rec plans compile in the background at startup, so the
  first request pays no compile cost (`--no-prewarm` to disable).
- **Per-size plan cache (LRU)** — compiled tract plans reused across requests.

Full-OCR latency, 28-line page, warm, native M-series — detection is a single
inference (not parallelizable across lines), so pick a smaller det model if it
dominates:

| models (det+rec) | det | rec | total |
|------------------|----:|----:|------:|
| tiny | 0.37 s | 0.28 s | **0.65 s** |
| small | 0.63 s | 1.2 s | **1.85 s** |
| medium | 2.05 s | 2.9 s | **4.97 s** |
| small-det + medium-rec | | | **3.64 s** |

Binary / startup / footprint (Apple M4, all sizes built):

| build | binary | cold start¹ | warm start | peak RSS² | understanding / req |
|-------|-------:|------------:|-----------:|----------:|--------------------:|
| OCR-only | 19 MB | ~19 s | 0.5 s | 1.9 GB | — |
| + understanding | 25 MB | ~19 s | 0.5 s | 2.2 GB | +0.7 s |

¹ first run fetches ~300 MB (all sizes) from the CDN; network-dependent.
² all sizes loaded + prewarmed — build fewer size features for less.

## Build

tract is pure-Rust, so zig cross-compiles cleanly (this Mac arm64, Linux
x64/arm64):

```bash
cargo install cargo-zigbuild      # once
just add-targets
just linux-x64                    # or macos-arm64 / linux-arm64
                                  # + *-understanding variants for the feature
                                  # + fully-static musl: linux-x64-musl / linux-arm64-musl
upx --best --lzma target/x86_64-unknown-linux-gnu/release/ppocr-server   # optional, ~50% off code
```

The result is a single self-contained binary — copy it to the target host and
run it. First start needs outbound network to fetch weights (see
[Models & cache](#models--cache)); the understanding binary is glibc-dynamic and
wants ≥2 GB RAM. For an example systemd + nginx setup, see
[`deploy/README.md`](deploy/README.md).

## Notes & license

**Precision — f32 only.** Both quantized precisions were benchmarked and dropped;
with tract, f32 wins on every axis except disk size (which no longer matters —
weights are fetched, not embedded):

| precision | quality | speed in tract | verdict |
|-----------|---------|----------------|---------|
| **f32** | reference | fastest (AMX/AVX kernels) | **shipped** |
| fp16 | lossless in ONNXRuntime | *slower*; tract mis-executes f16 detection at some sizes (silent 0-box) | dropped |
| int8 (dynamic) | degrades CTC recognition (no calibration) | *slower* | dropped |

**Fidelity.** Pipeline mirrors PaddleOCR: DetResizeForTest + ImageNet-BGR
normalize + `DBPostProcess` for detection; `get_rotate_crop_image` perspective
rectify + resize-to-h48, `(x/255−0.5)/0.5` normalize + greedy CTC decode
(`blank + dict + space`) for recognition. Detection **unclip** approximates
`pyclipper` by growing the min-area rectangle (equivalent for DB's rectangular
boxes). No angle classifier — vertical lines are rotated 90° by aspect ratio.

**License.** Source code is Apache-2.0 ([`LICENSE`](LICENSE)). Model files are
fetched at runtime and are **not** covered by the source license:
- **PP-OCRv6 ONNX weights** — third-party PaddleOCR/PaddlePaddle artifacts; the
  CDN URLs in [`src/remote.rs`](src/remote.rs) are a convenience mirror with
  SHA-256 verification. Check upstream license/model terms before redistributing
  or using commercially.
- **`supra-kenya-id`** — a project-specific LoRA-fused model served as a runtime
  artifact; not covered by this repo's Apache-2.0 license unless separate model
  terms say so.
- The small `inference.yml`, `charset.txt`, tokenizer, and config files are
  embedded model metadata — preserve upstream/model notices when redistributing.

For different model hosting, edit the manifest in
[`src/remote.rs`](src/remote.rs), update the checksums, and rebuild.
