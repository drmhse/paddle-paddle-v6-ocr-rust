# Example deployment (systemd + nginx)

One way to run `ppocr-server` in production — a reverse proxy in front of the
binary managed by systemd. These files are **examples**; adapt paths, user, and
hostname to your environment. `ppocr-server` is a plain HTTP server, so any
process manager / proxy works.

Files here:
- `ppocr-server.service` — systemd unit (binds `127.0.0.1:3088`, sets a model cache dir).
- `ocr-servos.conf` — nginx vhost (proxies a hostname → `127.0.0.1:3088`, 50 MB body limit, 300 s timeouts).

## Steps

```bash
# copy the cross-compiled binary to the host
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

## Adapt before using

- **systemd `User`/`Group`** — create a dedicated service account (or use one
  that owns `/opt/ocr-servos` and the cache dir).
- **install path** — the unit assumes `/opt/ocr-servos/ppocr-server`.
- **nginx `server_name`** — set your own hostname; add TLS (e.g. certbot).
- **model cache** — the unit sets `PPOCR_CACHE_DIR=/opt/ocr-servos/models-cache`.
  The **first start fetches models** there (needs outbound network, ~1 min) and
  starts instantly after. Pre-seed that dir to skip the first-run download. See
  [Models & cache](../README.md#models--cache).
- **RAM** — the understanding build wants ≥2 GB; OCR-only is lighter.
