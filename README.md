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
- a persistent tactical curriculum that expands from terminal positions to the
  full dataset only after validated promotions;
- explicit policy/value metrics, including entropy, calibration, illegal policy
  mass and top-1 accuracy;
- a paired, color-alternating promotion arena without noise and with a 55%
  promotion threshold;
- noise-free mirror and noisy self-play gates that reject repetition-prone models;
- unit and property-based tests.

Ratatui is deliberately deferred to the next milestone.

## Learning and code-reading guides

- [Reading YokaiRust as a C++ developer learning Rust](docs/reading-guide.md)
- [AlphaZero in YokaiRust](docs/alphazero-guide.md)

The first guide maps the Rust constructs used here to familiar C++ concepts and
suggests a module-by-module reading order. The second explains policy/value
targets, the terminal curriculum, promotion gates and why continued self-play is
not a proof of perfect play.

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
256 self-play games, 10 training epochs, a 200-game paired arena, a 64-game
candidate mirror diagnostic and a 64-game exploratory candidate probe. Self-play
runs 16 concurrent games and selects 8
distinct MCTS leaves per game before each inference, which feeds Metal efficiently
without requiring hundreds of blocked game threads. The promotion arenas instead
use sequential PUCT (`search_batch_size = 1`), so virtual losses cannot influence
their decisions. Workers are concurrent games, not a promise to keep the same
number of CPU cores busy.

Self-play keeps Dirichlet noise at every root but selects the most visited move
(`exploration_temperature = 0`). The complete noisy MCTS visit distribution
remains the policy training target, so the network still learns about explored
alternatives even though the played move is selected greedily.

Training starts with an automatic terminal curriculum. Its checked-in phases use
the last 8, 16, 32 and 64 plies of decisive games before returning to the complete
dataset. Early positions with ambiguous outcomes and draw cycles are therefore
not allowed to drown the initial tactical signal. Each phase specifies its own
search budget and self-play-only repetition contempt. A rejected candidate stays
in the same phase; only a successful promotion advances the persistent state in
`data/self-play/curriculum-state.json`. The final phase repeats indefinitely, so
one command can train from generation zero without per-generation editing.

Repetition contempt changes search exploration only: official outcomes, stored
value targets and promotion games still score a draw as zero. A candidate must
reach the 55% paired-arena score, keep its noise-free mirror draw rate at or below
35%, and keep its noisy self-play draw rate at or below 20%. Both gates use 64
games. The exploratory probe catches models whose cycles appear only after root
Dirichlet noise changes the trajectory.

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
one metrics line per epoch, roughly twenty updates per arena, and every checkpoint,
curriculum or promotion decision. Arena updates include the running
candidate/champion/draw counts and provisional score. All lines include elapsed
wall-clock time. Phase summaries also report average/maximum inference batch,
backend throughput and client wait. One generation is run by default; use
`--generations N` to request an explicit sequence.

On the development M4 Max, the terminal-8 experiment took about 2 minutes 40
seconds. A measured full-depth terminal-32 generation including the new mirror
gate took about 6 minutes 45 seconds: 3 minutes 20 for self-play, 17 seconds for
training, 2 minutes 15 for the paired arena and 53 seconds for the mirror gate.
This excludes a one-time 30-second incremental release relink. The original
one-leaf implementation projected roughly 16 minutes for self-play alone. These
figures are measurements, not guarantees; game length and model behavior change
between generations.

Set `backend = "cpu"` in the TOML file for deterministic debugging without
Metal. The command bootstraps generation zero when no model exists, then:

1. generates self-play games from the current champion;
2. selects the active curriculum window, updates the rolling replay buffer and
   trains a candidate;
3. evaluates candidate and champion with paired seeds and alternating colors;
4. runs the candidate against an identical copy with official draw values;
5. previews candidate self-play with the active exploration settings;
6. publishes the candidate only if the arena score and both draw gates pass,
   then advances the curriculum when its phase has enough promotions.

Training keeps the epoch with the lowest combined validation loss. It stops
early after `early_stopping_patience` consecutive epochs without improvement,
then sends that restored best epoch—not merely the last epoch—to the arena.

Each invocation completes the requested number of generations. Checkpoints are
written under `models/generation-N/`; `models/champion` points to the published
champion. The replay buffer is stored in `data/self-play/buffer.json`. Writes use
temporary paths followed by atomic renames so an interrupted write cannot replace
the last complete generation. After `Ctrl+C`, rerunning the command starts again
from the last published champion.

### Experimental observations and draw-cycle diagnosis

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

The steadily increasing draw rate reproduced the same failure mode previously
observed in the C++ and Node.js prototypes. Diagnostics established that it was
an undesirable learned cycle: champion 5 drew all 32 noise-free mirror games,
and increasing its search from 200 to 400 simulations made draws substantially
more frequent. Because an official repetition is worth zero, a deeper search
rationally preferred it whenever the inaccurate value head rated alternatives
as losses.

The arena counters were checked in both argument orders, ruling out an inverted
candidate/champion result. The real weakness was the promotion protocol: a model
could decisively exploit its predecessor while developing a different mirror
cycle. Generation 13 demonstrated this by passing its old arena but drawing
42/64 neutral mirror games; its checkpoint was retained but the champion pointer
was returned to generation 11.

The terminal curriculum proposed during debugging, combined with repetition
contempt during full-depth self-play, reversed the trend:

| Candidate data | Terminal window | Simulations | Contempt | Draws |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 16 | 400 | 0.50 | 38/256 (14.8%) |
| 12 | 16 | 400 | 0.50 | 21/256 (8.2%) |
| 13 | 32 | 400 | 0.50 | 18/256 (7.0%) |
| 14 | 32 | 400 | 0.50 | 16/256 (6.25%) |

Candidate 14 then produced 0/64 mirror draws but was correctly rejected because
its paired score against champion 11 was only 50%. These measurements show that
the runaway draw trend is controlled without weakening the official rules or
promoting a merely non-cycling but otherwise equal candidate.

Candidate 15 exposed a second failure mode after initially passing 200-0 and
0/64 mirror draws: its following noisy self-play batch produced 76/256 draws
(29.7%). Its checkpoint was retained, champion was returned to generation 11,
and the exploratory candidate gate was added at 20% so the same regression is
now rejected before publication.

Candidate 16 confirmed the deterministic gate itself: it also beat champion 11
by 200-0, then drew all 64 mirror games and was rejected. The exploratory probe
was skipped because the candidate was already ineligible. Champion 11 therefore
remains published while its full-depth self-play batches stay in the measured
5.9-8.2% draw range.

Generation 4 also exposed validation overfitting: validation loss was best at
epoch 1 and degraded through epoch 10 while training loss kept improving. The
pipeline now restores the lowest-validation-loss epoch and stops after the
configured patience. Generation 5 validated this behavior by stopping after
epoch 3 and sending the restored epoch 1 model to the arena.

## Rules source

The engine follows the official 3×4 rulebook rather than preserving behavioral
differences from the older JavaScript and C++ implementations:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
