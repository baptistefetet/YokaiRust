#!/bin/sh
set -eu

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [release-version]" >&2
    echo "Example: $0 v1.0.0" >&2
    exit 1
fi

# The page shows this label next to the source link. A build made outside a
# release is labelled "dev".
release_version=${1:-dev}
case "$release_version" in
    ''|*[!A-Za-z0-9._-]*)
        echo "Release version may contain only letters, numbers, dots, underscores and hyphens." >&2
        exit 1
        ;;
esac

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

# A browser keeps the files of a site in its cache and may combine a new page
# with an old stylesheet or an old model. The page therefore requests every
# file with this identifier in its address: a release uses its version, and a
# development build the time of the build, so no two builds share an address.
if [ "$release_version" = dev ]; then
    build_id="dev-$(date +%Y%m%d%H%M%S)"
else
    build_id=$release_version
fi

# The accepted characters cannot break the sed expressions, the HTML or a URL.
sed -e "s/@RELEASE_VERSION@/$release_version/g" \
    -e "s/@BUILD_ID@/$build_id/g" \
    "$repository_root/web/static/index.html" > "$distribution/index.html"
if grep -q '@[A-Z_]*@' "$distribution/index.html"; then
    echo "A build placeholder was not replaced in index.html." >&2
    exit 1
fi

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
