#!/bin/sh
set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_directory/../.." && pwd)
distribution="$repository_root/web/dist"

cd "$repository_root"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
    echo "wasm-bindgen is required: cargo install wasm-bindgen-cli --version 0.2.127 --locked" >&2
    exit 1
fi

if [ "$distribution" != "$repository_root/web/dist" ]; then
    echo "Refusing to clean an unexpected distribution directory: $distribution" >&2
    exit 1
fi

rm -rf -- "$distribution"
mkdir -p "$distribution"
cp -R "$repository_root/web/static/"* "$distribution/"
find "$distribution" -name .DS_Store -delete

cargo run --release --bin export-web-model -- \
    "$repository_root/config/training.toml" \
    "$distribution/model"

cargo build -p yokai-web \
    --target wasm32-unknown-unknown \
    --release \
    --no-default-features \
    --features flex
wasm-bindgen \
    "$repository_root/target/wasm32-unknown-unknown/release/yokai_web.wasm" \
    --out-dir "$distribution/pkg-flex" \
    --target web \
    --no-typescript

cargo build -p yokai-web \
    --target wasm32-unknown-unknown \
    --release \
    --no-default-features \
    --features webgpu
wasm-bindgen \
    "$repository_root/target/wasm32-unknown-unknown/release/yokai_web.wasm" \
    --out-dir "$distribution/pkg-webgpu" \
    --target web \
    --no-typescript

if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Oz \
        "$distribution/pkg-flex/yokai_web_bg.wasm" \
        -o "$distribution/pkg-flex/yokai_web_bg.wasm"
    wasm-opt -Oz \
        "$distribution/pkg-webgpu/yokai_web_bg.wasm" \
        -o "$distribution/pkg-webgpu/yokai_web_bg.wasm"
fi

echo "Static web build ready in $distribution"
