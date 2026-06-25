# Cross-platform builds via cargo-zigbuild (zig as the C/C++ cross linker).
# Requires: rustup, zig, `cargo install cargo-zigbuild`, and the rust target
# (`rustup target add <triple>`). tract is pure-Rust + models are embedded, so
# every target produces one self-contained binary — no native ONNX lib, no
# sidecar files.

bin := "ppocr-server"

default:
    @just --list

build:
    cargo build --release

run *ARGS:
    cargo run --release -- {{ARGS}}

# ---- the three requested targets ----

macos-arm64:
    cargo zigbuild --release --target aarch64-apple-darwin

linux-x64:
    cargo zigbuild --release --target x86_64-unknown-linux-gnu

linux-arm64:
    cargo zigbuild --release --target aarch64-unknown-linux-gnu

# fully-static musl variants (portable single binary)
linux-x64-musl:
    cargo zigbuild --release --target x86_64-unknown-linux-musl

linux-arm64-musl:
    cargo zigbuild --release --target aarch64-unknown-linux-musl

# build all three requested targets
all: macos-arm64 linux-x64 linux-arm64

# ---- UPX compression (single-binary distribution) ----

upx-native:
    upx --best --lzma target/release/{{bin}}

upx target:
    upx --best --lzma target/{{target}}/release/{{bin}}

# add the rust targets used above
add-targets:
    rustup target add \
      aarch64-apple-darwin \
      x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu \
      x86_64-unknown-linux-musl aarch64-unknown-linux-musl

# lean builds (only the requested OCR model sizes are fetched at runtime)
build-tiny:
    cargo build --release --no-default-features --features tiny

# ---- understanding stage (Supra-50M, candle) ----
# Adds the kenya_id structured-extraction endpoint. NO build-time weights: all
# models (OCR + Supra) are fetched on first run and cached (see remote.rs).
# Pulls C/C++ tokenizer deps, but zig cross-compiles them (gnu targets verified).

build-understanding:
    cargo build --release --features understanding

run-understanding *ARGS:
    cargo run --release --features understanding -- {{ARGS}}

linux-x64-understanding:
    cargo zigbuild --release --features understanding --target x86_64-unknown-linux-gnu

linux-arm64-understanding:
    cargo zigbuild --release --features understanding --target aarch64-unknown-linux-gnu

macos-arm64-understanding:
    cargo zigbuild --release --features understanding --target aarch64-apple-darwin
