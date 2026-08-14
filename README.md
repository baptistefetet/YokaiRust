# YokaiRust

YokaiRust is a fast, testable Rust implementation of the 3×4 rules of
**Yōkaï no Mori**—the French yōkai-themed edition of *Dōbutsu shōgi* (“animal
shogi”), a beginner-friendly mini-shogi variant. It includes a complete
AlphaZero-style training loop. A 5×6 variant is planned. Training depends only
on game rules, self-play and the project's own checkpoints.

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
- draw-aware search, visited-state archive restarts and guarded model promotion;
- a Ratatui interface for local play, champion play and replay analysis;
- a static one-player web interface running the same Rust rules, MCTS and
  champion through WebAssembly, with WebGPU and CPU fallback;
- atomic model/optimizer checkpoints and structured JSON diagnostics.

The stable training line is **v26**:

- models: `models/alpha-zero-visited-restarts-v26`;
- self-play, reports and diagnostics: `data/alpha-zero-visited-restarts-v26`;
- accepted champion: generation 16.

The learning baseline is now frozen at v26/g16. The Ratatui interface supports
local human play, human-versus-champion play and replay viewing using the stable
`Game`, `Action`, `Replay` and analysis contracts.

## Development roadmap

The Ratatui interface is complete for its current scope. The next planned
feature is the 5×6 game variant. Learning will resume only as an isolated
experiment against the frozen v26/g16 baseline, targeting long-horizon draw WDL
through loss weighting or decisive endgame sampling before considering another
architecture change.

Generation 16 remains the publication and regression reference. A learning
experiment will replace it only after a paired arena shows a statistically
credible strength improvement and the noisy productivity gate still passes.

## Guides

- [Reading YokaiRust as a C++ developer learning Rust](docs/reading-guide.md)
- [AlphaZero in YokaiRust](docs/alphazero-guide.md)

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

# Build the local API documentation; public additions must be documented.
cargo doc --no-deps

# Build the static browser version and exported champion under web/dist/.
./web/scripts/build.sh

# Explicit Apple Metal training/checkpoint preflight.
cargo test --test neural metal_training_state_round_trip_runs_one_real_update -- --ignored --exact
```

Release builds use Cargo's portable defaults. Runtime datasets and models are
ignored by Git and stay inside `data/` and `models/`.

## Commands

```bash
# Play a local two-human match in the Ratatui interface.
cargo run -- play

# Play as First at the bottom against the latest accepted champion.
cargo run --release -- play human-vs-cpu

# Open a validated replay and step through it with the arrow keys.
cargo run -- watch path/to/game.json

# Pure MCTS from the official initial position.
cargo run -- analyze [simulations] [seed]

# Validate every action and print a versioned replay.
cargo run -- replay path/to/game.json

# Start or continue AlphaZero training in the configured active paths.
cargo run --release -- train --config config/training.toml --generations 15 --headless
cargo run --release -- train --resume latest --generations 5 --headless

# Evaluate every checkpoint on the active run's final stable validation split.
cargo run --release -- diagnose-endgames --config config/training.toml
```

### Static web interface

The browser mode is a one-player game against the accepted champion. Rust owns
the complete `Game`, legal actions, history-aware encoder, MCTS and Burn neural
network inside a Web Worker; JavaScript only renders snapshots and animations.
It therefore does not duplicate the rules or search implementation.

The output under `web/dist/` is a collection of static HTML, JavaScript, image,
model and WebAssembly files. It needs no application backend. WebGPU is selected
when available, with an automatically loaded Burn Flex CPU fallback. Deploy the
directory on HTTPS (or serve it from localhost while developing); see
[the web build guide](web/README.md) for prerequisites and commands.

### Ratatui interface

First is always displayed at the bottom and Second at the top. Use the arrow
keys or WASD to move on the board, Enter to select and play, Tab to move between
the board and the current player's hand, Escape to cancel, `N` to restart and
`Q` to quit. Number keys 1–3 select Tanuki, Kitsune and Kodama from the hand.

`human-vs-cpu` loads the accepted generation referenced by `latest` under the
model path in `config/training.toml`. Model loading and deterministic MCTS run
on a background worker, leaving rendering and input responsive. The human is
First at the bottom. The same champion analyzes every human and CPU turn, so the
interface exposes the current side's root value, priors, visits, policy and Q
values. Human predictions never play a move; CPU predictions wait briefly before
the chosen move is applied and highlighted on the board.

The same right-hand panels show move history and stored replay analyses. Local
human-versus-human play does not run inference, so its prediction values remain
empty.

Training reports progress as text; `--headless` is accepted for explicit
non-interactive runs. Generation-boundary writes are atomic. `latest` points to
the accepted champion; rejected candidates remain available for diagnostics but
never become self-play sources.

## Current AlphaZero pipeline

The checked-in Metal configuration uses a 64-channel, four-block residual
network with 64-unit shared and value layers. Each generation runs:

1. 256 self-play games at 200 MCTS simulations per regular move;
2. 25% restarts from recent visited states at 800 simulations per move;
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
- [AlphaZero](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf)
- [KataGo self-play training](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)

### Draw-aware learning

Official draws always remain draws. The stored WDL target is never rewritten.
The pipeline handles repetition feedback at four points:

- WDL predicts a certain draw separately from uncertain win/loss;
- self-play values a draw at `+0.75` for the starter and `-0.75` for the
  non-starter, while official arenas use neutral `P(win) - P(loss)`;
- one quarter of trajectories restart from uniformly sampled nonterminal states
  visited in the recent replay buffer, with a larger search budget;
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

## Latest training result: v26

The v26 training run took 1 h 20 min 03 s on an Apple M4 Max. Its 25%
deeper-search restarts sample all recently visited nonterminal states. The
accepted champion is generation 16, with these final measurements:

| Measurement | Result |
| --- | ---: |
| self-play | 92 First wins / 118 Second wins / 46 draws |
| validation policy loss / top-1 | 1.508 / 57.2% |
| validation WDL loss / top-1 | 0.770 / 64.9% |
| validation scalar MSE | 0.734 |
| paired strength arena | 79 wins / 31 losses / 90 draws, score 62.0% |
| deterministic mirror | 0/4 draws |
| noisy productivity probe | 6/64 draws |

An independent 400-game paired arena at 400 simulations per move confirmed the
promotion result against its reference:

| Candidate / reference / draws | Score | Candidate as First | Candidate as Second |
| ---: | ---: | ---: | ---: |
| 170 / 66 / 164 | **63.0%** | 62.5% | 63.5% |

The approximate 95% interval is 59.5–66.5%. Both seats are positive, and the
interval excludes 50%.

### Restart behavior and speed

Across v26, 960 games restarted at depths 1–232, with a mean depth of 33.3
plies. Restarted games drew 60/960 times (6.2%), versus 197/3,136 (6.3%) for
games starting from the initial position. The broader archive did not make its
own trajectories less productive.

Neural self-play evaluated 40,217,309 positions in about 1,188.4 backend
seconds: **33,842 positions/s** on the Apple M4 Max. The accepted champion's run
reached 29,961 positions/s.

### Endgame diagnosis

The current frozen v26 split contains 438 games, including 24 draws, and 18,271
positions. At distance `17+`, drawn policy top-1 is 61.8%; at `9–16`, it is
55.6%. Drawn WDL top-1 is 7.0% and 1.3% in those two buckets.

The strength gain therefore does not solve long-horizon draw classification.
It comes from better play as measured directly, while the WDL draw head remains
the clearest learning weakness.

## Research conclusion

Visited-state restarts are now the stable baseline. They follow Go-Exploit's
simple “Visited States” variant: decisive and drawn games both feed the archive,
and duplicate states naturally weight frequently visited regions. Network
shape, optimizer, replay retention, promotion and all other budgets remained
unchanged. [Go-Exploit](https://arxiv.org/abs/2302.12359)

The next learning change should not be another architecture rewrite. The
endgame diagnosis isolates the remaining problem: long-horizon draw WDL. If
optimization resumes, target that head or strengthen decisive endgame sampling
with one isolated change.

## Rules source

The engine follows the [official 3×4
rulebook](https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf).
