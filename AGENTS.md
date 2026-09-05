# Agent instructions

## Workflow

- Develop, commit and push directly on `main`. Do not create feature
  branches or pull requests unless explicitly requested.
- Keep changes isolated: one focused change per commit, with a message that
  explains the reasoning.

## Required checks before pushing

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo doc --no-deps
```

- Public API additions must carry documentation that satisfies `cargo doc`.
- After touching the web interface or the accepted champion, rebuild with
  `./web/scripts/build.sh` and verify `web/dist/` stays deployable.
- Build the static site with `./web/scripts/build.sh` and publish a release
  with `./web/scripts/package-release.sh vX.Y.Z`; published releases carry the
  deployable website and the minimal accepted checkpoint separately.

## Learning experiments

- The accepted checkpoint selected by `latest` is the published player and
  the source for training. A candidate replaces it only after a paired arena
  shows a statistically credible strength improvement and the noisy
  productivity probe passes.
- Record a fixed reference checkpoint for each experiment before running it,
  and retain that checkpoint for independent regression comparisons. Keep
  its identifier in the experiment report rather than hard-coding a generation
  into general project instructions or player interfaces.
- Training runs are atomic at generation boundaries; `latest` always points
  to the accepted champion. Rejected candidates may remain on disk for
  diagnostics but never become self-play sources; do not delete or rewrite
  checkpoint history.
- Prefer isolated, comparable changes: tune loss weighting or decisive
  endgame sampling before considering another architecture change.

## Documentation

- Write for someone learning Rust and neural networks. Explain concepts
  directly without assuming knowledge of another systems programming language.
- Describe the current implementation in the guides. Keep measurement
  provenance and checkpoint identifiers in the training results and reports.

## Repository conventions

- Follow the absolute board orientation (First at the bottom, moves toward
  rank 4; Second at the top, moves toward rank 1).
- Move notation: `b2-b3`; drop notation: `kodama@a4`.
- The Metal preflight test is a diagnostic: run it to confirm the training
  state round-trip, but do not treat its output as a regression gate.
