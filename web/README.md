# Static web mode

The web application is a presentation layer around the Rust engine. The rules,
game history, neural encoder, champion inference and MCTS all run in WebAssembly
inside a Web Worker. JavaScript only renders Rust snapshots with Phaser and
submits actions chosen by the player.

The generated site has no API and no server-side runtime. It tries WebGPU first
and automatically loads the Burn Flex CPU build when WebGPU is unavailable.

## Build

Install the one-time WASM prerequisites:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.127 --locked
```

Then build the accepted champion and both browser backends:

```bash
./web/scripts/build.sh
```

The script reads the active model path from `config/training.toml` and writes a
self-contained deployable directory to `web/dist/`. Generated model and WASM
files are ignored by Git.

To build the site and prepare versioned release archives with checksums, run:

```bash
./web/scripts/package-release.sh v1.0.0
```

This writes ignored artifacts under `web/release/`: a ready-to-deploy web
archive, a minimal archive containing the accepted native checkpoint, and
`SHA256SUMS`. The model archive preserves its `models/...` path and can be
extracted at the repository root. It intentionally excludes optimizer state,
training snapshots and self-play data.

## Run and deploy

Serve the contents of `web/dist/`, rather than opening `index.html` through a
`file://` URL. For example, a local smoke test can use:

```bash
python3 -m http.server 8080 --directory web/dist
```

WebGPU requires a secure context in production, so deploy the same files on an
HTTPS static host. No route, database or backend process is required. The host
must serve `.wasm` files with the `application/wasm` MIME type; current static
web servers normally do this by default.

If Rust is not installed on the web server, download the web archive from the
[latest GitHub Release](https://github.com/baptistefetet/YokaiRust/releases/latest)
and extract it into the HTTPS document root. All generated JavaScript,
WebAssembly and champion files are included; no build step runs on the server.
