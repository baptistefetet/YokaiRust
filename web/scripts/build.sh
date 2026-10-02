#!/bin/sh
set -eu

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_directory/../.." && pwd)
distribution="$repository_root/web/dist"

cd "$repository_root"

required_wasm_bindgen="0.2.127"

if ! command -v wasm-bindgen >/dev/null 2>&1; then
    echo "wasm-bindgen is required: cargo install wasm-bindgen-cli --version $required_wasm_bindgen --locked" >&2
    exit 1
fi

# The CLI must match the `wasm-bindgen` crate pinned in web/crate/Cargo.toml
# exactly, or the generated bindings fail at load time with confusing
# WASM-level errors.
installed_wasm_bindgen=$(wasm-bindgen --version | awk '{print $2}')
if [ "$installed_wasm_bindgen" != "$required_wasm_bindgen" ]; then
    echo "wasm-bindgen $installed_wasm_bindgen is installed but $required_wasm_bindgen is required:" >&2
    echo "cargo install wasm-bindgen-cli --version $required_wasm_bindgen --locked --force" >&2
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
else
    echo "Warning: wasm-opt is not installed; WebAssembly files were built without size optimization." >&2
fi

echo "Static web build ready in $distribution"
