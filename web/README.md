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
progress is kept. The bottom-right corner of the board shows the release
version and the GitHub mark; together they form one link to the source
repository, large enough for a finger on a phone. The link is an HTML element
that the page keeps over that corner of the canvas at any window size.

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
before packaging a release. The packaging command requires it. Where Homebrew is not available, Binaryen comes
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
extracted at the repository root. Both archives use the accepted generation read
at the start of packaging, even if training promotes a new champion during the build.
The checkpoint archive excludes optimizer state,
training snapshots and self-play data.

## Files kept by the browser

A browser keeps the files of a site in its cache and does not always ask the
server whether they changed. After a new deployment it could then show the new
page with the previous stylesheet, or run the new engine with the previous
champion. To prevent this, the build writes a build identifier into
`index.html`: the release version, or the time of the build for a development
build. The page, its script and the worker add it to the address of every file
they request (`app.css?v=v1.0.0`), so the files of two builds never share an
address.

`index.html` itself has a fixed address. Where the host allows it, serve that
file with the `Cache-Control: no-cache` header, which makes the browser check
for a new version at each visit.

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

## Source

Edit `web/crate/src/lib.rs` for the browser adapter and `web/static/` for
presentation. `web/generated/` and `web/dist/` contain generated artifacts;
editing generated `yokai_web.js` bindings will not change their Rust source.

## Browser tests

`web/tests/` holds automated tests that open the built site in a browser and
play it. They use [Playwright](https://playwright.dev), a library that drives a
browser from a script, with the Chromium that Playwright downloads for that
purpose. The browser runs without a window, and the browsers installed on the
machine are not used.

Install both once; Node.js 20 or later is required:

```bash
cd web/tests
npm ci
npx playwright install --no-shell chromium
```

`npm ci` installs the Playwright version recorded in `package-lock.json` into
`web/tests/node_modules`, ignored by Git. The second command downloads
Chromium into Playwright's cache, shared by every project on the machine
(`~/Library/Caches/ms-playwright` on macOS).

After a build, run the tests from the repository root:

```bash
npm test --prefix web/tests
```

The tests serve `web/dist/` on a local port and play each scenario twice: in a
desktop window driven with a mouse, where the page must load the WebGPU
module, and on a phone-sized screen driven by touch, where the browser has no
graphics adapter and the page must fall back to the Flex module. The first
case needs a machine with a graphics adapter.

| Scenario | What it checks |
| --- | --- |
| Loading | The page requests the expected `pkg-…` directory only, and every file except the page and its icon with the build identifier. |
| One player | A person's move is answered by the champion; the game is played to its end and a new one starts from the result panel. |
| Two players | Captures, a drop, a move played with and without promotion, a hand showing two pieces of one kind, the end of the game, the options opened from the logo, and the number of players kept after a reload. |

A scenario also fails when the page reports an error or when one of its
requests fails. To test another copy of the site, such as an extracted release
archive, give its path:

```bash
YOKAI_WEB_ROOT=/path/to/site npm test --prefix web/tests
```

The board is drawn on a canvas, a single picture in which the browser cannot
find a piece or a button. `web/tests/game-page.js` therefore reads their
positions from the page's Phaser objects, then presses the screen there as a
person would. It is the only test file that depends on the names used in
`main.js`.

## Before a release

Compile both WASM backends (the CI does this separately), run
`./web/scripts/build.sh`, then the browser tests. The build must not print the
`wasm-opt` warning: a release carries the size-optimized modules.

The tests do not judge what the page looks like. Serve `web/dist/` locally and
look at the board in a desktop window and in a mobile viewport; check there
that a touch opens the rulebook and the source repository. Ensure the release
contains both WASM modules, model bytes, metadata and static assets.
