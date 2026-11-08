# Validation records — 8 November 2026

This directory contains the evidence for the current 3×4 baseline:
[training results](../../training-results.md) and
[performance results](../../performance-results.md). The campaign trained a
network from random initialization through 30 attempts. Its accepted champion
is generation 30, independently evaluated against a fixed reference checkpoint.

## Records

| Files | Contents |
| --- | --- |
| [`protocol.json`](protocol.json) | Initialization, frozen reference hashes, selection rules and recorded protocol amendments. |
| [`training-observations.json`](training-observations.json) | Every attempt's source, outcomes, promotion gates and phase timings. |
| `confirmation-{26,30}.plan.json` | Candidate hashes and arena settings recorded before evaluation. |
| `confirmation-{26,30}.json` and `.decision.json` | Both independent arena results and confirmation decisions. |
| [`performance-protocol.json`](performance-protocol.json) | Source revisions, common fixture, measurement order and budgets. |
| [`performance-observations.json`](performance-observations.json) | Per-run outputs, work counts, timing summaries and memory use. |
| [`data-audit.json`](data-audit.json) | Replay identities, completeness and restart ancestry. |
| [`verification.json`](verification.json) | Rust checks, Metal diagnostic and browser verification. |
| [`archive-verification.json`](archive-verification.json) | Verified release members and SHA-256 hashes. |

The JSON files preserve the recorded inputs and observations. They are the
source of truth for the summaries, including rejected candidates and the failed
independent comparison.

## Training and evaluation

Run training from the repository root on a Mac with Metal. The configuration
snapshot is [`config/retrain-20261108.toml`](../../../config/retrain-20261108.toml).
It uses seed `20261108` and isolated model/data paths. For a separate repetition,
copy it to a new configuration, choose empty output directories and record the
reference and evaluation plan before training. The CLI runs a requested number
of complete attempts:

```bash
cargo run --release -- train --config path/to/new-experiment.toml --generations 30
```

Existing model/data directories resume their accepted champion and persisted
work. They do not start a fresh run. Preserve the recorded campaign when
creating another experiment.

The campaign's supporting scripts are scoped to its frozen protocol and local
process records:

- `continue_until_promotion.py <source-generation>` advances one complete
  attempt at a time until the accepted champion changes.
- `confirm.py <attempt-boundary>` checks the accepted chain, configuration and
  reference hashes, records the candidate identity, then runs an independent
  arena. It refuses to overwrite existing observations.

Independent arenas use 800 games at 400 simulations per move. Each receives a
disjoint range of 400 pair seeds, with candidate colors swapped within a pair.
Confirmation requires 55% of points and p ≤ 0.05/3 for up to three comparisons.
The observed candidates are generations 26 and 30; their outcomes appear in the
training report.

## Fixed-workload performance comparison

The fixed checkpoint includes Adam state. The replay fixture contains the
256 bootstrap trajectories from the campaign's first generation. Its hash and
the two source revisions are recorded in `performance-protocol.json`.

```bash
python3 docs/experiments/2026-11-08/measure.py build
python3 docs/experiments/2026-11-08/measure.py run
```

The build step extracts both Git revisions into temporary directories, compiles
`measurement.rs` and each revision's CPU benchmark, then retains the executables
under the ignored experiment data directory. Refreshing source timestamps
prevents the shared Cargo cache from reusing the other revision's crate.
Dependency builds are reused.

The runner interleaves three repetitions per revision without concurrent
training or another GPU workload. CPU measurements use the engine benchmark's
five inner samples. Metal measurements use the same checkpoint for inference
and 64 self-play games, then restore the same Adam state for 400 updates on the
fixed corpus. Results retain game replays, inference work, optimizer metrics and
post-update predictions. `/usr/bin/time -l` records maximum resident memory.
The raw labels `before` and `after` identify the reference and measured revisions.

## Local artifacts

Checkpoints, replays, logs, binaries and the fixed corpus live under the ignored
`models/alpha-zero-from-scratch-20261108/` and
`data/alpha-zero-from-scratch-20261108/` directories. The fixed reference is
stored separately under `models/references/pre-cleanup-20261108/`.
These files are required to rerun the measurement tools; the minimal release
checkpoint contains the accepted player, not the complete training state.
