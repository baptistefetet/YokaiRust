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
