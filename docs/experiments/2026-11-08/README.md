# Fresh training and cleanup verification — 8 November 2026

`protocol.json` was recorded before training. It fixes the old champion's
identity and hashes, initialization, selection boundaries and independent
confirmation criteria. `performance-protocol.json` fixes the source revisions,
measurement order and common training fixture. Results are summarized in
[training results](../../training-results.md) and
[performance results](../../performance-results.md).

## Reproduction

Run commands from the repository root on a Mac with Metal. The configuration
snapshot is `config/retrain-20261108.toml`. Its only changes from the starting
configuration are seed `20261108` and separate empty model/data directories.
The initial training command was:

```bash
cargo run --release -- train --config config/retrain-20261108.toml --generations 20
```

The checkpoint and replay directories must be absent for a fresh run. If they
already exist, this command resumes them. Preserve the original experiment;
use new paths and a new protocol for a repetition.

Before any independent arena, the user requested extending the first block to
22 attempts. The dated amendment is preserved in `protocol.json`; after the
initial 20-attempt command, run two more attempts with `--generations 2`.
A second user amendment extends this block until the next promotion after
champion 20, even beyond attempt 22. Call that attempt N. Before the first
reference arena, resolve the boundaries to N, N+10 and N+20 from the internal
promotion reports. Independent seeds and thresholds remain unchanged. The next promotion was
attempt **26**, so the resolved boundaries are **26, 36 and 46**.

At a planned boundary, the confirmation runner verifies the accepted champion
chain, configuration and reference hashes before comparing networks:

```bash
python3 docs/experiments/2026-11-08/confirm.py 26
```

It compares only the final accepted champion, writes its identity before the
arena starts, and refuses to overwrite existing observations. After the failed comparison at 26, continue until one further promotion with
`python3 docs/experiments/2026-11-08/continue_until_promotion.py 26`.
The second arena uses that successor and the second independent seed.
If needed, the third comparison follows ten more attempts, with unchanged
training settings. Every arena
uses a disjoint range of 400 pair seeds and the stricter significance
threshold in the protocol. The unused base seeds were separated by 1000
after detecting that adjacent bases would reuse almost every pair; the first
arena was unaffected and remains unchanged.

## Fixed-workload performance comparison

The retained reference includes optimizer state. The fixed replay fixture is
the 256 bootstrap trajectories from this experiment's first generation. Its
hash is recorded before measurement. No performance sample may run alongside
training or another GPU workload.

```bash
python3 docs/experiments/2026-11-08/measure.py build
python3 docs/experiments/2026-11-08/measure.py run
```

The build step extracts the two recorded Git revisions into temporary
directories, compiles the common `measurement.rs` harness and each revision's
CPU benchmark, then retains their executables under the ignored experiment
data directory. It refreshes source timestamps to prevent a shared Cargo cache
from reusing the other revision's crate. Dependency builds are reused.

The run step alternates before/after order over three repetitions. CPU tests
use the existing five-sample engine benchmark. Metal measurements use the same
reference network for inference and 64 self-play games, then restore the same
Adam state for 400 updates on the fixed corpus. Results include game replays,
inference work counts, optimizer metrics and a prediction after updating. The
process log also records maximum resident memory through `/usr/bin/time -l`.
These comparisons are independent of the newly trained model's strength.

Raw checkpoints, replays, logs and the fixed corpus live in the ignored
`models/` and `data/` directories. The old reference is retained separately
under `models/references/pre-cleanup-20261108/`; removing obsolete experiments
must not remove it or the new experiment's history.
