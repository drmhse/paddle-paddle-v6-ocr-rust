# ppocr-server

A tiny, **cross-platform**, **lean-binary** OCR server for **PP-OCRv6** — text
**detection + recognition** (full text extraction). Inference runs on
[`tract`](https://github.com/sonos/tract), a pure-Rust ONNX engine (no ONNX
Runtime). Models are **fetched from a CDN on first run and cached locally** —
the binary is ~25 MB; weights download once and are reused across restarts. See
[Model cache](#model-cache-first-run).

## Models (fetched on first run, feature-gated by size)

The size feature selects which detection **and** recognition models are
available (and therefore fetched on first run):

| size feature | det | rec | charset |
|------|-----|-----|---------|
| `tiny`   | PP-OCRv6_tiny_det (1.7 MB)  | PP-OCRv6_tiny_rec (4.3 MB)  | 6,904 chars |
| `small`  | PP-OCRv6_small_det (9.5 MB) | PP-OCRv6_small_rec (20 MB)  | 18,708 chars |
| `medium` | PP-OCRv6_medium_det (59 MB) | PP-OCRv6_medium_rec (73 MB) | 18,708 chars |

`default = ["tiny","small","medium"]` (all). The binary size is the same
regardless — only the *small* text configs (`inference.yml`, `charset.txt`) are
embedded; the ONNX weights are fetched. Build with fewer sizes to fetch less on
first run:

```bash
cargo build --release --no-default-features --features tiny           # fetches ~6 MB on first run
cargo build --release --no-default-features --features "tiny,small"
cargo build --release                                                 # all sizes (~168 MB fetched once)
```

## Model cache (first run)

On startup the server ensures each required model is present in its cache,
downloading from `cdn.drmhse.com` and **verifying SHA-256** if not. Manifest
(ids, URLs, checksums) lives in [`src/remote.rs`](src/remote.rs).

- **Cache dir**: `$PPOCR_CACHE_DIR`, else `$XDG_CACHE_HOME/ppocr-server/models`,
  else `~/.cache/ppocr-server/models`. Downloads use a temp file + atomic
  rename, so a present cache file is always complete + verified — restarts skip
  re-hashing (cold start ~1 s once cached; ~70 s first run for all sizes).
- **First run needs network**; subsequent runs are fully offline. For air-gapped
  deploys, pre-seed the cache dir with `<id>.bin` files (ids from the manifest).
- To update a model, bump its manifest entry (URL + checksum) and ship a new
  binary; the new checksum invalidates the old cache file automatically.

### Precision: why f32 only

Both quantized precisions were implemented and benchmarked, then **dropped** —
with the `tract` engine, f32 wins on every axis except disk size:

| precision | quality | speed in tract | verdict |
|-----------|---------|----------------|---------|
| **f32** | reference | fastest (AMX/AVX kernels) | **shipped** |
| fp16 | lossless in ONNXRuntime | *slower*, and tract mis-executes f16 detection at some input sizes (silent 0-box) | dropped |
| int8 (dynamic) | degrades CTC recognition (no calibration) | *slower* | dropped |

tract runtime-dispatches the best f32 SIMD kernels per CPU (it logs e.g. `AMX
optimisation activated` on Apple Silicon), so f32 is genuinely the fast path.
The binary is lean regardless of precision (weights are fetched, not embedded);
fewer size features just means less to download on first run.

## Performance

The pipeline is tuned for multi-line pages:

- **Parallel recognition** — detected lines are recognized concurrently across
  all cores (`rayon`). On a 28-line page this cut recognition from ~10.2 s to
  ~3.0 s (3.3×) on an M-series CPU.
- **Fixed rec-width buckets** — line crops snap to one of 8 widths
  (160…2048) so only a handful of plans are ever compiled, instead of one per
  32 px. Near-100% plan-cache hits.
- **Startup pre-warm** — those rec plans are compiled in the background at
  startup (parallel), so the first request pays no recognition compile cost.
  Disable with `--no-prewarm`.
- **Per-size plan cache (LRU)** — compiled tract plans are reused across
  requests; `--plan-cache N` controls how many distinct input sizes to keep.

Indicative full-OCR latency, 28-line page, warm, native M-series:

| models (det+rec) | det | rec | total |
|------------------|----:|----:|------:|
| tiny | 0.37 s | 0.28 s | **0.65 s** |
| small | 0.63 s | 1.2 s | **1.85 s** |
| medium | 2.05 s | 2.9 s | **4.97 s** |
| small-det + medium-rec | | | **3.64 s** |

Detection is a single inference (not parallelizable across lines); pick a
smaller det model if detection latency dominates.

## Web UI & docs (embedded, no sidecar)

- `GET /` — interactive **demo page**: upload an image, pick det/rec models and
  params, see boxes drawn + extracted text.
- `GET /docs` — **Swagger UI** (vendored, served from the binary).
- `GET /openapi.json` — OpenAPI 3.0 spec (model ids filled in as enums).

## Run

```bash
./ppocr-server                 # 0.0.0.0:8080
OCR_PORT=9000 ./ppocr-server
```
Env/flags: `--host`/`OCR_HOST`, `--port`/`OCR_PORT`, `--plan-cache`/`OCR_PLAN_CACHE`
(distinct input sizes kept compiled per model, LRU; default 16).

## API

### `GET /health` → `ok`

### `GET /v1/models`
Lists embedded detection models (with default params) and recognition models.

### `POST /v1/ocr` — full text extraction (detect → recognize)
Body = raw image bytes. Query params (all optional except model selection when
more than one is embedded):

| param | meaning | default |
|-------|---------|---------|
| `det_model` | detection model id (or `model`) | required if >1 |
| `rec_model` | recognition model id | required if >1 |
| `thresh`, `box_thresh`, `unclip_ratio`, `limit_side_len` | detection tuning | per-model yml |
| `min_rec_score` | drop lines below this recognition confidence | 0.0 |

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
`text` is all lines joined in reading order (top→bottom, left→right). `points`
are quad corners (tl,tr,br,bl) in original-image pixels.

### `POST /v1/detect` — detection only
Returns boxes + scores (no text). Same detection params as above; model via
`model` or `det_model`.

### `POST /v1/recognize` — recognition only
Body = a single **already-cropped, upright** line image → `{ text, score }`.
Model via `model` or `rec_model`.

## Cross-compile (cargo-zigbuild) + single-binary distribution

tract is pure-Rust, so zig cross-compiles cleanly. Targets used here: this Mac
(arm64), Linux x64, Linux arm64.

```bash
cargo install cargo-zigbuild      # once
just add-targets

just macos-arm64        # native (this Mac)
just linux-x64          # x86_64-unknown-linux-gnu
just linux-arm64        # aarch64-unknown-linux-gnu
# or fully-static musl variants: just linux-x64-musl / linux-arm64-musl
```

### Compress with UPX
```bash
upx --best --lzma target/release/ppocr-server
upx --best --lzma target/x86_64-unknown-linux-gnu/release/ppocr-server
```
The binary is ~25 MB (no embedded weights); UPX still cuts the code by ~50%.
Model weights are fetched on first run, not embedded — see [Model cache](#model-cache-first-run).

## Understanding stage (optional, `--features understanding`)

An optional stage that turns OCR text into structured JSON. It runs a LoRA-fused
**Supra-50M** model on [`candle`](https://github.com/huggingface/candle)
(pure-Rust CPU inference); the weights are fetched + cached on first run like the
OCR models. Off by default, so the plain OCR build is even leaner.

```bash
just build-understanding       # native, with the feature (no build-time weights)
just linux-x64-understanding   # cross-compile for the server
```

Then add `understand=true` on `/v1/ocr` with `mode=kenya_id`:

```bash
curl -X POST "http://127.0.0.1:8080/v1/ocr?mode=kenya_id&understand=true\
&det_model=PP-OCRv6_medium_det&rec_model=PP-OCRv6_medium_rec" \
  --data-binary @id.jpg -H "Content-Type: application/octet-stream"
```

The response gains a `fields` object — each field is `{ value, confidence }`
(confidence = the model's lowest token probability across that value, 0–1). The
web demo at `/` exposes an "Understand → structured JSON" toggle.

Notes:
- **No build-time weights.** The 136 MB Supra model is fetched on first run and
  cached (SHA-256 verified) like the OCR models — see [Model cache](#model-cache-first-run).
  Only the small tokenizer/config are embedded.
- **Not pure-Rust.** tokenizers pulls C/C++ deps (onig, esaxx); zig cross-compiles
  them, and the gnu Linux targets are verified. musl-static is untested.
- **Resources.** No binary growth (weights fetched); ~136 MB added to the model
  cache; ~1.2 GB RAM for the loaded model; CPU inference adds ~1 s/request. No GPU.
- Currently wired for `kenya_id` only.

## Deploy (systemd + nginx)

Cross-compile → copy the binary to the server → run under systemd behind nginx.
Reference units are in [`deploy/`](deploy/).

```bash
just linux-x64-understanding                 # or `just linux-x64` for OCR-only
upx --best --lzma target/x86_64-unknown-linux-gnu/release/ppocr-server   # optional
scp target/x86_64-unknown-linux-gnu/release/ppocr-server SERVER:/opt/ocr-servos/

# on the server (first time):
sudo cp deploy/ppocr-server.service /etc/systemd/system/
sudo cp deploy/ocr-servos.conf /etc/nginx/sites-available/ocr-servos
sudo ln -sf /etc/nginx/sites-available/ocr-servos /etc/nginx/sites-enabled/
sudo systemctl daemon-reload && sudo systemctl enable --now ppocr-server
sudo nginx -t && sudo systemctl reload nginx

# updates:
sudo systemctl restart ppocr-server
```

The service binds `127.0.0.1:3088`; nginx proxies `ocr.servos.dev` with a 50 MB
body limit and 300 s timeouts. Add TLS (e.g. certbot) for production. The
understanding binary is dynamically linked to glibc — fine on a normal Linux
host; ensure the box has ≥2 GB RAM.

**First start fetches models.** The unit sets `PPOCR_CACHE_DIR=/opt/ocr-servos/models-cache`;
the first start downloads the models there (needs outbound network) and caches
them, so the service may take ~1 min to become ready the first time and starts
instantly thereafter. Pre-seed that dir to skip the first-run download.

## Fidelity notes

Pipeline mirrors PaddleOCR: DetResizeForTest + ImageNet-BGR normalize +
`DBPostProcess` for detection; `get_rotate_crop_image` perspective rectify +
resize-to-h48, `(x/255−0.5)/0.5` normalize + greedy CTC decode
(`blank + dict + space`) for recognition. The detection **unclip** approximates
`pyclipper` by growing the min-area rectangle (equivalent for DB's rectangular
boxes). No angle classifier — vertical lines are rotated 90° by aspect ratio.
