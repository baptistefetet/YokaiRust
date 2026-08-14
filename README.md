# YokaiRust

YokaiRust is a fast, testable Rust implementation of the official 3×4 rules of
**Yōkaï no Mori**, with a complete AlphaZero-style training loop. A 5×6 variant
is planned, so learning depends only on game rules, self-play and the project's
own checkpoints.

## Current state

The repository contains:

- typed board, pieces, hands, actions, outcomes and official transitions;
- capture, promotion, parachuting, victory and threefold-repetition rules;
- a canonical 132-action policy encoding and versioned JSON replays;
- deterministic PUCT/MCTS with subtree reuse and batched inference;
- an 80-plane history encoder, 50 non-spatial features and action-aligned
  repetition context;
- a Burn residual network with policy and Win/Draw/Loss (WDL) heads;
- CPU tests and WGPU/Metal training on Apple Silicon;
- parallel self-play, a rolling replay buffer and stable whole-game validation;
- draw-aware search, cycle-adjacent restarts and guarded model promotion;
- atomic model/optimizer checkpoints and structured JSON diagnostics.

The current training line is **v25**. It is the only retained runtime dataset:

- models: `models/alpha-zero-draw-aware-v25`;
- self-play and reports: `data/alpha-zero-draw-aware-v25`;
- accepted champion: generation 13;
- last evaluated candidate: generation 15, rejected on strength.

The next product milestone is the Ratatui interface. It can advance independently
using the stable `Game`, `Action`, `Replay` and analysis contracts.

## Guides

- [Reading YokaiRust as a C++ developer learning Rust](docs/reading-guide.md)
- [AlphaZero in YokaiRust](docs/alphazero-guide.md)

The earlier browser project remains a useful visual reference for the future
interface: [`baptistefetet/yokai`](https://github.com/baptistefetet/yokai).

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

Changes are developed, committed and pushed directly on `main`. Do not create
feature branches or pull requests unless explicitly requested.

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check

# Explicit Apple Metal training/checkpoint preflight.
cargo test --test neural metal_training_state_round_trip_runs_one_real_update -- --ignored --exact
```

Release builds use Cargo's portable defaults. Runtime datasets and models are
ignored by Git and stay inside `data/` and `models/`.

## Commands

```bash
# Pure MCTS from the official initial position.
cargo run -- analyze [simulations] [seed]

# Validate every action and print a versioned replay.
cargo run -- replay path/to/game.json

# Start or continue AlphaZero training in the configured v25 paths.
cargo run --release -- train --config config/training.toml --generations 15 --headless
cargo run --release -- train --resume latest --generations 5 --headless

# Evaluate every v25 checkpoint on the final stable validation split.
cargo run --release -- diagnose-endgames --config config/training.toml
```

`--headless` disables the future TUI, not textual progress. Generation-boundary
writes are atomic. `latest` points to the accepted champion; rejected candidates
remain available for diagnostics but never become self-play sources.

The ignored Metal arena probe can compare two retained generations:

```bash
YOKAI_CANDIDATE_MODELS=models/alpha-zero-draw-aware-v25 \
YOKAI_CANDIDATE_GENERATION=15 \
YOKAI_REFERENCE_MODELS=models/alpha-zero-draw-aware-v25 \
YOKAI_REFERENCE_GENERATION=13 \
YOKAI_ARENA_GAMES=200 \
cargo test --release --test performance \
  compare_saved_generations_from_environment -- --ignored --nocapture
```

## Current AlphaZero pipeline

The checked-in Metal configuration uses a 64-channel, four-block residual
network with 64-unit shared and value layers. Each generation runs:

1. 256 self-play games at 200 MCTS simulations per regular move;
2. 25% cycle-adjacent restarts at 800 simulations per move;
3. 400 Adam updates with batch size 256;
4. a 200-game paired strength arena at 400 simulations per move;
5. four deterministic mirror games for diagnosis;
6. a 64-game noisy self-play productivity probe.

The champion is the single source of self-play, weights and optimizer state.
The next attempt after a rejection starts from that same accepted checkpoint,
but sees a larger replay buffer and a new deterministic seed.

### Promotion

A candidate is promoted only when it:

1. scores at least 55% against the champion in paired games with shared random
   0–4 ply openings and swapped colors;
2. produces at most 20% draws in the noisy 64-game self-play probe.

Deterministic candidate-versus-itself draws remain recorded but do not veto a
candidate. Identical strong policies can settle into one stable line, while
Dirichlet noise and temperature still produce diverse, decisive training games.
The productivity probe measures that actual training behavior directly.

This rule agrees with the relevant algorithms: AlphaGo Zero gated only against
the current best player, AlphaZero later trained continuously without a
checkpoint gate, and KataGo makes gatekeeping optional. A drawn game is not an
empty sample: its scalar value is neutral, but every retained position still
teaches the MCTS visit policy.

- [AlphaGo Zero](https://dsbrown1331.github.io/advanced-ai/readings/alphaGoZero.pdf)
- [AlphaZero](https://www.davidsilver.uk/wp-content/uploads/2020/03/alphazero-science_compressed.pdf)
- [KataGo self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)

### Draw-aware learning

Official draws always remain draws. The stored WDL target is never rewritten.
The pipeline handles repetition feedback at four points:

- WDL predicts a certain draw separately from uncertain win/loss;
- self-play values a draw at `+0.75` for the starter and `-0.75` for the
  non-starter, while official arenas use neutral `P(win) - P(loss)`;
- one quarter of trajectories restart one to eight plies before observed
  repetition cycles with a larger search budget;
- the starter's drawing defence remains a policy target, while unresolved
  non-starter policy targets are omitted. When the rules identify the exact
  action causing a third repetition, only that action is removed and the other
  MCTS visits are renormalized.

Every official WDL target remains fully weighted. The auxiliary scalar loss has
weight `0.25` and compares `P(win) - P(loss)` with `+1/0/-1`; it complements,
but never replaces, categorical WDL learning.

### Validation

Whole games are assigned to validation by a stable hash of generation and seed.
Growing the replay buffer therefore does not reshuffle old games or leak one
game's positions across training and validation.

Reports record policy/WDL/scalar losses, top-1 metrics, illegal policy mass,
calibration, draw error, policy weighting, repetition behavior, self-play
outcomes and arena results by seat. Loss alone is not a strength measurement:
the paired arena and noisy draw probe remain the publication criteria.

## Latest training result

The completed 15-generation v25 run took 1 h 17 min 14 s. The accepted sequence
was:

```text
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 11 → 13
```

Champion 13 was promoted with:

| Measurement | Result |
| --- | ---: |
| self-play | 97 First wins / 93 Second wins / 66 draws |
| validation policy loss / top-1 | 1.564 / 57.0% |
| validation WDL loss / top-1 | 0.808 / 65.4% |
| validation scalar MSE | 0.800 |
| strength arena | 131 wins / 26 losses / 43 draws, score 76.2% |
| deterministic mirror | 4/4 draws, diagnostic only |
| noisy productivity probe | 7/64 draws |

Candidates 14 and 15 were rejected only because their arena scores were 53.3%
and 54.0%. Their noisy probes remained healthy at 4/64 and 7/64 draws. The
champion therefore stopped for measured strength, not because of the mirror
diagnostic.

Before obsolete checkpoints were removed, two frozen cross-run arenas measured
champion 13 at 65.9% over the former accepted baseline across 400 games and
66.5% over the strongest rejected baseline across 200 games. Both seat splits
were positive. The promotion change therefore produced a real strength gain,
not merely more accepted generation numbers.

Neural self-play for generations 2–15 evaluated 21,894,924 positions in about
592.3 backend seconds: approximately **36,966 positions/s**. Keep this as the
current speed reference; the architecture did not change during the promotion
experiment, so small differences should be treated as runtime variation.

### Remaining weakness

The frozen v25 endgame split contains 400 games, including 88 draws, and 13,786
positions. Champion 13 reaches 64.7% drawn policy top-1 at distance `9–16` and
61.6% at `17+`, but drawn WDL top-1 falls to 5.0% and 1.4%. Playing strength has
improved significantly; long-horizon draw classification has not been solved.

## Recommended next research

Do not change the network or bootstrap next. The current bootstrap promoted its
first model immediately, and the 64×4 architecture retains adequate throughput.
The next learning experiment should target the remaining long-horizon weakness
with one isolated change:

1. broaden cycle-only restarts into a deeper state archive, in the spirit of
   [Go-Exploit](https://arxiv.org/abs/2302.12359), to revisit strategically
   useful late-game positions rather than only terminal repetition failures;
2. if that is insufficient, strengthen the decisive endgame curriculum while
   keeping all positions and official outcomes;
3. reserve an adaptive rollout/network blend for a future bootstrap problem.
   [Warm-Start AlphaZero](https://arxiv.org/abs/2004.12357) and its
   [adaptive follow-up](https://arxiv.org/abs/2105.06136) support that technique,
   but v25 provides no evidence that bootstrap is the current bottleneck.

Any new experiment should change one variable, retain champion 13 as its frozen
reference, record self-play throughput, and require a positive paired result
before becoming the new baseline.

## Rules source

The engine follows the official 3×4 rulebook:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
