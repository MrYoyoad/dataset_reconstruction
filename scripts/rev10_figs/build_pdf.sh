#!/usr/bin/env bash
# Typeset notes/exact_channel_rev10.tex standalone -> notes/exact_channel_rev10.pdf
# Uses a statically linked (musl) tectonic, which runs on WEXAC's old glibc; downloads it once to ~/.local/bin.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TT="$HOME/.local/bin/tectonic-musl"
if [ ! -x "$TT" ]; then
  mkdir -p "$HOME/.local/bin"; tmp="$(mktemp -d)"
  curl -sSL -o "$tmp/tt.tgz" "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz"
  tar xzf "$tmp/tt.tgz" -C "$tmp" && mv "$tmp/tectonic" "$TT"
fi
BUILD="$(mktemp -d)"
cp "$ROOT/notes/exact_channel_rev10.tex" "$ROOT/notes/exact_channel_rev10_main.tex" "$ROOT"/figures/rev10/*.png "$BUILD/"
( cd "$BUILD" && "$TT" --keep-logs exact_channel_rev10_main.tex >/dev/null && grep -c Overfull exact_channel_rev10_main.log || true )
cp "$BUILD/exact_channel_rev10_main.pdf" "$ROOT/notes/exact_channel_rev10.pdf"
echo "wrote $ROOT/notes/exact_channel_rev10.pdf"
