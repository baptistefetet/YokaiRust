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

- Generation 16 is the publication and regression reference. A candidate
  replaces it only after a paired arena shows a statistically credible
  strength improvement and the noisy productivity probe still passes.
- Training runs are atomic at generation boundaries; `latest` always points
  to the accepted champion. Rejected candidates may remain on disk for
  diagnostics but never become self-play sources; do not delete or rewrite
  checkpoint history.
- Prefer isolated, comparable changes: tune loss weighting or decisive
  endgame sampling before considering another architecture change.
- Prefer `--headless` for non-interactive training runs.

## Repository conventions

- Follow the absolute board orientation (First at the bottom, moves toward
  rank 4; Second at the top, moves toward rank 1).
- Move notation: `b2-b3`; drop notation: `kodama@a4`.
- The Metal preflight test is a diagnostic: run it to confirm the training
  state round-trip, but do not treat its output as a regression gate.
