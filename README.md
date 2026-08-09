# YokaiRust

YokaiRust is a fast, testable implementation of the official 3×4 rules of
**Yōkaï no Mori**. It now includes the first complete AlphaZero training loop;
the next major milestone is the animated terminal interface. The project is
intentionally built in small, verified milestones so it can also serve as a
first serious Rust codebase.

## Current milestone

The repository currently contains the rules engine, deterministic MCTS and the
first trainable AlphaZero pipeline:

- compact typed board, pieces, hands, actions, outcomes and transitions;
- official movement, capture, promotion, parachuting and victory rules;
- threefold-repetition detection;
- canonical 132-action policy encoding for both players;
- deterministic, versioned JSON replays;
- batch-oriented policy/value evaluators with a bounded prediction cache;
- PUCT search in a contiguous node arena, including safe subtree reuse;
- self-play-only Dirichlet noise and a configurable temperature schedule;
- legal-policy masking plus per-action prior, visits, policy and Q diagnostics;
- optional MCTS analyses embedded in validated replay files;
- a versioned 17-plane canonical neural encoder with horizontal augmentation;
- a configurable Burn residual network with 132 policy logits and a value head;
- CPU inference for deterministic tests and WGPU/Metal inference on Apple Silicon;
- a batching inference service shared by concurrent self-play games;
- generation-based SafeTensors checkpoints with validated metadata and an
  atomically published `champion` pointer;
- reproducible parallel self-play, a rolling replay buffer and whole-game
  training/validation splits;
- explicit policy/value metrics, including entropy, calibration, illegal policy
  mass and top-1 accuracy;
- a paired, color-alternating promotion arena without noise and with a 55%
  promotion threshold;
- unit and property-based tests.

Ratatui is deliberately deferred to the next milestone.

## Board coordinates

The engine keeps one absolute orientation. First starts at the bottom and moves
toward rank 4; Second starts at the top and moves toward rank 1.

```text
      a4 b4 c4   Second
      a3 b3 c3
      a2 b2 c2
      a1 b1 c1   First
```

Moves use `b2-b3`. Drops use a full piece name such as `kodama@a4`.

## Development

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

Release builds use Cargo's portable defaults. Runtime data, self-play datasets
and trained models are ignored by Git.

Cargo downloads crate sources into `~/.cargo/registry` and compiles this
project into its local `target/` directory. Rust toolchains live in `~/.rustup`.
No package is installed globally by the build. The generated `models/` and
`data/self-play/` directories remain inside this project.

## Text analysis and replays

Until the TUI milestone, the binary exposes deliberately small diagnostic
commands without pulling in a CLI framework:

```bash
# Pure MCTS from the official initial position. Defaults: 200 simulations, seed 42.
cargo run -- analyze [simulations] [seed]

# Validate every action and print a versioned Rust replay.
cargo run -- replay path/to/game.json
```

`Mcts::search` never adds exploration noise and is suitable for arenas or human
play. `Mcts::search_self_play` is the explicit self-play entry point and is the
only one that adds Dirichlet noise. A fixed seed reproduces both the tree search
and temperature-based action sampling.

## AlphaZero training

The checked-in configuration targets Metal and is intentionally substantial:
256 self-play games, 400 simulations per move, 10 training epochs and a
200-game paired arena. Self-play runs 16 concurrent games and selects 8 distinct
MCTS leaves per game before each inference, which feeds Metal efficiently without
requiring hundreds of blocked game threads. The promotion arena instead runs 128
games concurrently with sequential PUCT (`search_batch_size = 1`), so virtual
losses cannot influence its decision. Workers are concurrent games, not a promise
to keep the same number of CPU cores busy. Adjust `config/training.toml` for short
experiments.

```bash
# Run one generation with the checked-in Metal configuration.
cargo run --release -- train --config config/training.toml --headless

# Resume automatically from the latest champion and replay buffer.
cargo run --release -- train --resume latest --headless

# Run five successive generations, still resuming from the latest champion.
cargo run --release -- train --resume latest --generations 5 --headless
```

`--headless` disables the future TUI, not textual diagnostics. Progress is
written immediately to standard error: roughly twenty updates during self-play,
one metrics line per epoch, roughly twenty arena updates, and every checkpoint
or promotion decision. Arena updates include the running candidate/champion/draw
counts and provisional score. All lines include elapsed wall-clock time. Phase
summaries also report average/maximum inference batch, backend throughput and
client wait. One generation is run by default; use `--generations N` to request
an explicit sequence.

On the development M4 Max, a measured generation took about 3 minutes 30 seconds
in release mode: roughly 1 minute 55 for self-play, 15 seconds for training and
1 minute 20 for the arena. This excludes a one-time 30-second incremental release
relink. The original one-leaf implementation projected roughly 16 minutes for
self-play alone. These figures are measurements, not runtime guarantees; game
length and model behavior change between generations. Generation 5 took 5 minutes
19 seconds because its games were more than twice as long as generation 1 games.

Set `backend = "cpu"` in the TOML file for deterministic debugging without
Metal. The command bootstraps generation zero when no model exists, then:

1. generates self-play games from the current champion;
2. updates the rolling replay buffer and trains a candidate;
3. evaluates candidate and champion with paired seeds and alternating colors;
4. publishes the candidate as champion only if its arena score reaches 55%.

Training keeps the epoch with the lowest combined validation loss. It stops
early after `early_stopping_patience` consecutive epochs without improvement,
then sends that restored best epoch—not merely the last epoch—to the arena.

Each invocation completes the requested number of generations. Checkpoints are
written under `models/generation-N/`; `models/champion` points to the published
champion. The replay buffer is stored in `data/self-play/buffer.json`. Writes use
temporary paths followed by atomic renames so an interrupted write cannot replace
the last complete generation. After `Ctrl+C`, rerunning the command starts again
from the last published champion.

### Experimental observations through generation 5

The first local training sequence produced the following self-play trend. An
example corresponds to one played position, so examples per game also measure
average game length.

| Generation | Games | Examples | Average plies | Draws |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 7,499 | 29.3 | 10 (3.9%) |
| 2 | 256 | 9,460 | 37.0 | 20 (7.8%) |
| 3 | 256 | 13,916 | 54.4 | 44 (17.2%) |
| 4 | 256 | 15,826 | 61.8 | 80 (31.3%) |
| 5 | 256 | 16,113 | 62.9 | 93 (36.3%) |

The steadily increasing draw rate reproduces the same failure mode previously
observed in the C++ and Node.js prototypes. Its cause is not established yet: it
may represent convergence toward drawing play, or an undesirable learned cycle.
It must therefore be treated as an open training-quality problem rather than as
evidence that later generations are unconditionally stronger.

At the same time, the observed promotion arenas were decisive, including 200-0
scores for generations 4 and 5. A diagnostic with the generation 3 and 4 models
in both argument orders produced 20-0 for generation 4 and 0-20 after swapping
them, ruling out an inverted candidate/champion counter. The discrepancy between
decisive zero-temperature arenas and increasingly draw-heavy exploratory
self-play remains to be investigated through replay analysis and fixed baselines.

Generation 4 also exposed validation overfitting: validation loss was best at
epoch 1 and degraded through epoch 10 while training loss kept improving. The
pipeline now restores the lowest-validation-loss epoch and stops after the
configured patience. Generation 5 validated this behavior by stopping after
epoch 3 and sending the restored epoch 1 model to the arena.

## Rules source

The engine follows the official 3×4 rulebook rather than preserving behavioral
differences from the older JavaScript and C++ implementations:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
