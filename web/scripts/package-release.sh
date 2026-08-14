#!/bin/sh
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <release-version> [output-directory]" >&2
    echo "Example: $0 v1.0.0" >&2
    exit 1
fi

release_version=$1
case "$release_version" in
    ''|*[!A-Za-z0-9._-]*)
        echo "Release version may contain only letters, numbers, dots, underscores and hyphens." >&2
        exit 1
        ;;
esac

script_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_directory/../.." && pwd)
release_directory=${2:-"$repository_root/web/release"}
configuration="$repository_root/config/training.toml"

model_directory=$(sed -n 's/^[[:space:]]*models[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "$configuration" | tail -n 1)
if [ -z "$model_directory" ]; then
    echo "Could not read paths.models from $configuration" >&2
    exit 1
fi

case "$model_directory" in
    /*|../*|*/../*|*/..)
        echo "The configured model directory must stay inside the repository: $model_directory" >&2
        exit 1
        ;;
esac

latest_file="$repository_root/$model_directory/latest"
if [ ! -f "$latest_file" ]; then
    echo "Accepted-generation pointer not found: $latest_file" >&2
    exit 1
fi

generation=$(tr -d '[:space:]' < "$latest_file")
case "$generation" in
    ''|*[!0-9]*)
        echo "Invalid accepted generation in $latest_file: $generation" >&2
        exit 1
        ;;
esac

generation_padded=$(printf '%06d' "$generation")
generation_directory="$model_directory/generation-$generation_padded"
metadata_file="$repository_root/$generation_directory/metadata.json"
weights_file="$repository_root/$generation_directory/model.safetensors"

for required_file in "$metadata_file" "$weights_file"; do
    if [ ! -f "$required_file" ]; then
        echo "Required champion file not found: $required_file" >&2
        exit 1
    fi
done

"$script_directory/build.sh"

distribution="$repository_root/web/dist"
for required_file in \
    "$distribution/index.html" \
    "$distribution/model/champion.bin" \
    "$distribution/pkg-flex/yokai_web_bg.wasm" \
    "$distribution/pkg-webgpu/yokai_web_bg.wasm"; do
    if [ ! -f "$required_file" ]; then
        echo "Required web build file not found: $required_file" >&2
        exit 1
    fi
done

mkdir -p "$release_directory"

model_line=${model_directory##*-}
web_archive="$release_directory/yokai-web-$release_version-g$generation.tar.gz"
champion_archive="$release_directory/yokai-champion-$model_line-g$generation.tar.gz"
checksum_file="$release_directory/SHA256SUMS"

tar -czf "$web_archive" -C "$distribution" .
tar -czf "$champion_archive" -C "$repository_root" \
    "$model_directory/latest" \
    "$generation_directory/metadata.json" \
    "$generation_directory/model.safetensors"

(
    cd "$release_directory"
    web_archive_name=$(basename "$web_archive")
    champion_archive_name=$(basename "$champion_archive")
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$web_archive_name" "$champion_archive_name"
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$web_archive_name" "$champion_archive_name"
    else
        echo "Neither shasum nor sha256sum is available." >&2
        exit 1
    fi
) > "$checksum_file"

echo "Release files ready in $release_directory"
echo "  $(basename "$web_archive")"
echo "  $(basename "$champion_archive")"
echo "  $(basename "$checksum_file")"
