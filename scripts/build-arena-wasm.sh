#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer user-local rustup/cargo and wasm-bindgen-cli when present
if [ -d "$HOME/.cargo/bin" ]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi
if [ -d "$HOME/.local/bin" ]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v wasm-bindgen >/dev/null 2>&1; then
  echo "Error: wasm-bindgen CLI not found in PATH." >&2
  exit 1
fi

OUT_DIR="${1:-${ARENA_WASM_OUT_DIR:-$ROOT_DIR/dist/arena-wasm}}"

echo "Building riskbot arena-wasm (wasm32-unknown-unknown, release)..."
cargo build \
  --release \
  --lib \
  --no-default-features \
  --features arena-wasm \
  --target wasm32-unknown-unknown

mkdir -p "$OUT_DIR"

echo "Running wasm-bindgen (--target web) -> $OUT_DIR..."
wasm-bindgen \
  --target web \
  --out-dir "$OUT_DIR" \
  "$ROOT_DIR/target/wasm32-unknown-unknown/release/riskbot.wasm"

WASM_FILE="$OUT_DIR/riskbot_bg.wasm"
ls -lh "$WASM_FILE"
