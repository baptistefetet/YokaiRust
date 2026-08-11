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
- a versioned 129-plane canonical neural encoder covering eight successive
  states, with horizontal augmentation;
- a configurable Burn residual network with 132 policy logits and a value head;
- CPU inference for deterministic tests and WGPU/Metal inference on Apple Silicon;
- a batching inference service shared by concurrent self-play games;
- generation-based SafeTensors checkpoints with validated metadata and an
  atomically published pointer to the latest network;
- reproducible parallel self-play, a rolling replay buffer and whole-game
  training/validation splits;
- the complete AlphaZero dataset by default, with a terminal-window mode kept
  only for focused diagnostic experiments;
- explicit policy/value metrics, including entropy, calibration, illegal policy
  mass and top-1 accuracy;
- a paired, color-alternating arena against the previous network;
- noise-free mirror and noisy self-play diagnostics for repetition cycles;
- unit and property-based tests.

Ratatui is deliberately deferred to the next milestone.

## Learning and code-reading guides

- [Reading YokaiRust as a C++ developer learning Rust](docs/reading-guide.md)
- [AlphaZero in YokaiRust](docs/alphazero-guide.md)

The first guide maps the Rust constructs used here to familiar C++ concepts and
suggests a module-by-module reading order. The second explains policy/value
targets, continuous latest-network updates, draw diagnostics and why continued
self-play is not a proof of perfect play.

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
256 self-play games, 400 optimizer steps, a 200-game paired arena, a 64-game
mirror diagnostic and a 64-game exploratory probe. Self-play runs 16 concurrent
games and selects 8 distinct MCTS leaves per game before each inference, which
feeds Metal efficiently without requiring hundreds of blocked game threads. The
arena uses sequential PUCT (`search_batch_size = 1`), so virtual losses cannot
influence its measurement. Workers are concurrent games, not a promise to keep
the same number of CPU cores busy.

The default loop follows AlphaZero's latest-network behavior. Every trained
generation is published atomically and produces the next self-play batch,
regardless of its arena score. Self-play adds Dirichlet noise at every root,
samples from MCTS visits for the first 12 plies, then becomes greedy. All played
non-terminal positions enter the rolling buffer and official draws always have
value zero.

The 55% paired score and the 35%/20% draw limits are behavior indicators only.
They are logged but cannot reject a generation. A finite
`terminal_window_plies` and non-zero `repetition_contempt` remain available for
controlled experiments, but neither is enabled by the standard configuration.

```bash
# Run one generation with the checked-in Metal configuration.
cargo run --release -- train --config config/training.toml --headless

# Resume automatically from the latest network and replay buffer.
cargo run --release -- train --resume latest --headless

# Run five successive generations, still resuming from the latest network.
cargo run --release -- train --resume latest --generations 5 --headless
```

`--headless` disables the future TUI, not textual diagnostics. Progress is
written immediately to standard error: roughly twenty updates during self-play,
one metrics line per 100 optimizer steps, roughly twenty updates per arena, and
every checkpoint publication. Arena updates include the running
candidate/previous/draw counts and provisional score. All lines include elapsed
wall-clock time. Phase summaries also report average/maximum inference batch,
backend throughput and client wait. One generation is run by default; use
`--generations N` to request an explicit sequence.

Arena progress is completion-ordered because games run concurrently. Fast wins
can therefore appear as a block before slower losses even though paired game
indices alternate the candidate between `First` and `Second`. The final report
separates both assignments; intermediate counters must not be read as a
chronological winning streak.

This was verified by replaying the generation-8 arena exactly. The first 100
completed games were candidate wins and the final 100 were losses, but the
seat-aware result was candidate `First`: 60 wins/40 losses, candidate `Second`:
40 wins/60 losses. Completion time, not game index or player assignment, caused
the apparent two-block result.

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

1. generates self-play games from the latest network;
2. updates the rolling replay buffer and trains the next network;
3. saves and publishes that network for subsequent self-play;
4. evaluates it against the previous generation with paired seeds and colors;
5. measures its noise-free mirror and noisy self-play draw rates.

Training performs a fixed `steps_per_generation` number of uniformly sampled
mini-batch updates. Its work therefore does not silently grow with the replay
buffer. Adam moments are stored with every generation and restored before the
next one; the log reports `Adam resumed=true` when continuity was recovered.
Validation is diagnostic and runs every `validation_interval_steps` updates.

An optional terminal-window schedule can emphasize decisive tactics without
manual generation-by-generation edits. The complete replay buffer always stays
in the training set; terminal positions are additionally sampled until they
reach `decisive_fraction`. Their window starts at `initial_plies`, is multiplied
by `growth_factor` each generation, and the extra sampling ends at
`full_dataset_generation`. Reports store the effective window and number of
extra examples, so a resumed run follows the same deterministic curriculum.

Each invocation completes the requested number of generations. Checkpoints are
written under `models/generation-N/`; `models/latest` points to the latest
network. Each new checkpoint also contains `training-model.bin` and
`optimizer.bin`; these make optimizer resume exact but use additional disk space.
Existing `models/champion` pointers are still accepted during migration.
The replay buffer is stored in
`data/self-play/buffer.json`, and structured generation summaries are stored in
`data/self-play/reports/`. Writes use
temporary paths followed by atomic renames so an interrupted write cannot replace
the last complete generation. After `Ctrl+C`, rerunning the command starts again
from the last published network.

### Continuous AlphaZero experiment (August 2026)

A clean run from random generation 0 was stopped after generation 15 once the
same draw regime had repeated for several generations. It used the standard
configuration: all positions, opening temperature 1, official draw value 0 and
no repetition contempt or promotion gate. This run is the baseline from before
the fixed-step correction: Adam was reset at every generation and full-buffer
epochs made the number of optimizer updates grow with the buffer.

| Generation | Self-play draws | Previous-generation arena | Mirror draws | Exploratory draws | Validation policy/value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4/256 | 200/0/0 | 0/64 | 1/64 | 2.619 / 1.159 |
| 3 | 10/256 | 100/100/0 | 0/64 | 12/64 | 1.819 / 0.802 |
| 4 | 49/256 | 0/100/100 | 64/64 | 7/64 | 1.643 / 0.701 |
| 7 | 22/256 | 200/0/0 | 0/64 | 15/64 | 1.421 / 0.748 |
| 10 | 54/256 | 100/0/100 | 64/64 | 19/64 | 1.172 / 0.570 |
| 11 | 97/256 | 0/0/200 | 64/64 | 41/64 | 1.159 / 0.604 |
| 12 | 144/256 | 0/0/200 | 64/64 | 24/64 | 1.136 / 0.589 |
| 15 | 130/256 | 100/0/100 | 64/64 | 27/64 | 1.099 / 0.552 |

The run proves that the continuous-update plumbing works, including generations
that score below 55%. It also reproduces the symptom that originally looked like
a failure: deterministic play becomes repetition-dominated from generation 10
onward and does not recover by generation 15. Lower losses do not resolve that
ambiguity. Policy loss measures agreement with the current MCTS target, so it can
improve while MCTS learns either a strong strategy or a bad cycle; value loss also
becomes easier when more targets are the draw value zero. Top-1 still rose from 9%
during the first epoch to about 70%, while illegal policy mass fell from 91% to
4%, confirming that optimization itself was active. The frozen-baseline results
below are needed to interpret the symptom as strength progress rather than
stagnation.

The isolated artifacts are under ignored paths
`models/alpha-zero-from-zero/` and `data/alpha-zero-from-zero/`. Generation 15 is
the last complete checkpoint. Generation 16 self-play completed and was saved,
but training was interrupted before its checkpoint was created.

### Fixed-step verification (August 2026)

After preserving Adam across checkpoints and replacing growing full-buffer
epochs with 400 sampled mini-batch updates, a second clean run was stopped after
generation 12. Its artifacts are under `models/alpha-zero-fixed-step-v2/` and
`data/alpha-zero-fixed-step-v2/`.

| Generation | Self-play draws | Previous-generation arena C/R/D | Mirror draws | Exploratory draws | Validation policy/value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4/256 | 200/0/0 | 0/64 | 2/64 | 2.676 / 1.185 |
| 4 | 43/256 | 200/0/0 | 64/64 | 6/64 | 1.547 / 0.772 |
| 8 | 69/256 | 100/0/100 | 64/64 | 29/64 | 1.295 / 0.621 |
| 10 | 84/256 | 100/0/100 | 64/64 | 31/64 | 1.193 / 0.497 |
| 11 | 115/256 | 0/0/200 | 64/64 | 37/64 | 1.219 / 0.533 |
| 12 | 141/256 | 100/0/100 | 0/64 | 24/64 | 1.131 / 0.461 |

The draw curve therefore survived the optimizer corrections and closely matched
the earlier run: generation 12 had 141 draws instead of 144. It is not, however,
evidence of stagnant playing strength. In paired 40-game arenas, generation 12
beat generations 1, 4 and 8 by 40-0. Against generation 11 it scored 20 wins,
20 draws and no losses. Every color split was favorable or undefeated.

The correct interpretation is that deterministic copies can converge on a
repetition while the newer network still dominates historical opponents. Draw
rate remains useful behavioral information, especially under exploratory
self-play, but an arbitrary low-draw threshold is not a strength oracle. Frozen
baselines—and eventually an exact solver—are required before changing the
official zero value of a repetition.

This does not make a draw-dominated replay buffer harmless. In the generation-12
batch, drawn games already supplied 4,385 of 8,500 examples (51.6%), and the
starter/non-starter/draw result was 57/58/141. The model was stronger than its
history but was not yet moving toward the known theoretical result: the side
that moves second has a forced win. Extending stochastic move selection from 12
to 48 plies reduced a paired probe from 18 to 8 draws, but changed the
starter/non-starter win split from 14/32 to 29/27. It replaced repetitions with
random mistakes rather than improving the signal, so that setting was not
adopted as a quick fix.

### Historical guarded-pipeline observations

The following measurements predate the switch to continuous AlphaZero updates.
They are retained because they motivated the draw diagnostics, not because they
describe the current publication policy.

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
