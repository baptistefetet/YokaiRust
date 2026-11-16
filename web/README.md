# Static web mode

The web application is a presentation layer around the Rust engine. The rules,
game history, neural encoder, champion inference and MCTS all run in WebAssembly
inside a Web Worker. JavaScript only renders Rust snapshots with Phaser and
submits actions chosen by the player. Board dimensions come from Rust snapshots.
The browser uses the same `NetworkEvaluator` and `ModelMetadata` as native
clients; its asynchronous search awaits tensor readback.

The generated site has no API and no server-side runtime. It tries WebGPU first
and automatically loads the Burn Flex CPU build when WebGPU is unavailable.

The site plays the 5×6 game. Its options panel, opened on the first visit and
then from the logo, chooses the number of human players: one against the
accepted champion, who plays Second at the top, or two on the same screen. The
choice is kept in the browser's local storage. The panel also has a *Rules*
button that opens the official English rulebook in another tab, so a game in
progress is kept. The bottom-right corner of the page shows the release
version and a link to the source repository.

Snapshots carry everything the presentation needs to stay free of rules: board
dimensions, the piece kinds a hand can hold, the promoted form of each kind,
the legal actions and the sides played by a person. When a move exists both
with and without promotion, Rust lists two legal actions that differ only by
their `promote` flag, and the page asks which one to play.

## Build

Install the one-time WASM prerequisites:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.127 --locked
brew install binaryen
```

Binaryen provides `wasm-opt`, a tool that rewrites a compiled WebAssembly
module into a smaller one with the same behavior. The build script runs it on
both browser modules. Without it the script still produces a working site, but
it prints a warning and leaves the modules at their compiled size: install it
before packaging a release. Where Homebrew is not available, Binaryen comes
from the system's package manager or from its
[releases](https://github.com/WebAssembly/binaryen/releases).

The build also needs the accepted champion checkpoint on disk: the script
exports the generation referenced by `latest` under the model path configured
in `config/training.toml` (ignored by Git). Train one first, or download the
checkpoint archive from the
[latest release](https://github.com/baptistefetet/YokaiRust/releases/latest)
and extract it at the repository root.

Then build the accepted champion and both browser backends:

```bash
./web/scripts/build.sh
```

The script reads the active model path from `config/training.toml` and writes a
self-contained deployable directory to `web/dist/`. Generated model and WASM
files are ignored by Git. The page labels such a build `dev`; pass a version
(`./web/scripts/build.sh v1.0.0`) to show it instead, as the release script
does.

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

## Source and verification

Edit `web/crate/src/lib.rs` for the browser adapter and `web/static/` for
presentation. `web/generated/` and `web/dist/` contain generated artifacts;
editing generated `yokai_web.js` bindings will not change their Rust source.

Before release, compile both WASM backends (the CI does this separately), run
`./web/scripts/build.sh`, then serve `web/dist/` locally. The build must not
print the `wasm-opt` warning: a release carries the size-optimized modules.
Verify initial loading, a human move and AI reply, captures and drops, a move
offered with and without promotion, a hand holding two pieces of one kind,
game completion and reset.
Play both a one-player and a two-player game, and check that the chosen option
survives a reload. Inspect a mobile viewport as well as desktop. Exercise the
Flex module explicitly even when WebGPU is available; successful fallback alone
does not validate WebGPU. Check the browser console and failed requests, and
ensure the release contains both WASM modules, model bytes, metadata and static
assets.

An automated browser often has no graphics adapter and then only exercises the
Flex module. Check which `pkg-…` directory the page requested before concluding
that WebGPU works.
