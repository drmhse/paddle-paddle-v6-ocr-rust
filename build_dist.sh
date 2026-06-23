#!/usr/bin/env bash
# Build distribution binaries for mac-arm64, linux-x64, linux-arm64 (f32 — the
# only precision tract runs reliably + fast). UPX-compress the Linux ones.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p dist

emit() { # target_triple  build_cmd...
  local triple=$1; shift
  echo ">>> building $triple"
  if "$@"; then
    local src=target/$triple/release/ppocr-server
    [ "$triple" = native ] && src=target/release/ppocr-server
    cp "$src" "dist/ppocr-server-$triple"
    echo "OK $triple -> $(ls -lh dist/ppocr-server-$triple | awk '{print $5}')"
  else
    echo "FAIL $triple"
  fi
}

emit native                        cargo build   --release
emit x86_64-unknown-linux-gnu      cargo zigbuild --release --target x86_64-unknown-linux-gnu
emit aarch64-unknown-linux-gnu     cargo zigbuild --release --target aarch64-unknown-linux-gnu

for b in dist/ppocr-server-x86_64-unknown-linux-gnu dist/ppocr-server-aarch64-unknown-linux-gnu; do
  [ -f "$b" ] || continue
  cp "$b" "$b-upx"
  upx --best --lzma "$b-upx" >/dev/null 2>&1 && echo "UPX $b-upx -> $(ls -lh $b-upx | awk '{print $5}')"
done

echo "=== DIST ==="; ls -lh dist/ | awk 'NR>1{print $5, $9}'
echo "DIST_DONE"
