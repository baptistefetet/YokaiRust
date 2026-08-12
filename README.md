# YokaiRust

YokaiRust is a fast, testable implementation of the official 3×4 rules of
**Yōkaï no Mori**, with a complete AlphaZero-style training loop written in
Rust. A 5×6 variant is planned, so the learning pipeline deliberately depends
only on game rules, self-play and its own checkpoints.

## Current milestone

The repository contains:

- typed board, pieces, hands, actions, outcomes and official transitions;
- capture, promotion, parachuting, victory and threefold-repetition rules;
- a canonical 132-action policy encoding and versioned JSON replays;
- deterministic PUCT/MCTS with subtree reuse and batched inference;
- a 130-plane history-and-role encoder plus action-aligned repetition context;
- a Burn residual network with policy and Win/Draw/Loss (WDL) heads;
- CPU tests and WGPU/Metal training on Apple Silicon;
- parallel self-play, a rolling replay buffer and stable whole-game validation;
- draw-aware search, cycle-adjacent restarts and guarded model promotion;
- atomic checkpoints for the accepted champion and its optimizer state;
- dataset, optimization, arena and draw diagnostics in JSON reports.

The next product milestone is the Ratatui interface. Training can continue
independently because the UI only needs the stable `Game`, `Action`, `Replay`
and analysis contracts.

## Guides

- [Reading YokaiRust as a C++ developer learning Rust](docs/reading-guide.md)
- [AlphaZero in YokaiRust](docs/alphazero-guide.md)

The AlphaZero guide starts with a glossary—WDL, policy, value, logits, MCTS,
PUCT, loss, batches and checkpoints—before describing the architecture.

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

Release builds use Cargo's portable defaults. Runtime datasets and models are
ignored by Git and stay inside `data/` and `models/`.

## Text analysis and replays

```bash
# Pure MCTS from the official initial position.
cargo run -- analyze [simulations] [seed]

# Validate every action and print a versioned replay.
cargo run -- replay path/to/game.json
```

`Mcts::search` is deterministic and noise-free. `Mcts::search_self_play` is the
only entry point that injects Dirichlet noise. A fixed seed reproduces search
and temperature-based action selection.

## AlphaZero training

The checked-in Metal configuration runs 256 self-play trajectories, 400 Adam
updates, a 200-game paired strength arena, four deterministic mirror games and
a 64-game exploratory draw probe per candidate.

```bash
# Start from random generation zero in the configured paths.
cargo run --release -- train --config config/training.toml --generations 15 --headless

# Resume from the accepted champion, optimizer state and replay buffer.
cargo run --release -- train --resume latest --generations 5 --headless
```

`--headless` disables the future TUI, not textual progress. Checkpoints are
stored under the configured model path and `latest` points to the accepted
champion. Self-play data, replays and structured generation reports are stored
under the configured data path. Generation-boundary writes are atomic.

### One generation

1. Let the champion generate 75% of trajectories from fresh initial
   states and restart 25% two to eight plies before observed repetition cycles.
2. Add every resulting position and its official W/D/L result to the rolling
   replay buffer.
3. Resume the champion's weights and Adam moments for a fixed 400 updates.
4. Compare the candidate with the champion on paired random 0–4 ply openings.
5. Measure deterministic initial-position cycles and noisy self-play draws.
6. Promote only when strength and both draw gates pass.

The champion is the sole source of weights, optimizer state and self-play. A
candidate becomes the new champion only after all three checks pass. A rejected
candidate is kept as an experiment report but never generates training data.
The next attempt starts from the same safe checkpoint with a larger replay
buffer and a different deterministic seed.

### Draw-aware learning

Official draws always remain draws. The stored WDL target is never rewritten.
The pipeline attacks the feedback loop at four different points:

- the WDL head distinguishes a certain draw from an uncertain win/loss mixture;
- self-play values a draw at `+0.75` for the starter and `-0.75` for the
  non-starter, while official play still uses neutral `P(win) - P(loss)`;
- cycle-adjacent restarts shorten the horizon where conversion failed;
- policy loss does not imitate the non-starter's moves from a game it ultimately
  failed to convert. The starter's drawing defence and every official WDL target
  remain fully weighted.

This last rule follows the observed failure directly: the starter is allowed to
learn a drawing defence, while the non-starter should not receive a sharp
policy-imitation reward for failing to convert. It uses only self-play targets
and the generic repetition rule, so it applies unchanged to the planned 5×6
game.

The early decisive-tail curriculum is temporary. It oversamples tactical ends
during bootstrap, widens geometrically, and stops at candidate 11. The complete
buffer is always retained.

### Validation and metrics

Whole games are assigned to validation by a stable hash of their generation and
seed. Adding a new generation therefore does not reshuffle old games between
training and validation. This makes longitudinal loss curves interpretable
while preventing positions from one game leaking across both sets.

Reports include:

- policy and WDL cross-entropy;
- policy entropy, top-1 agreement and illegal probability mass;
- WDL top-1, value calibration and draw-probability error;
- the mean policy weight after omitting drawn non-starter policy targets;
- target entropy, action coverage, immediate-draw mass and repetition mass;
- separate draw buckets for starter and non-starter positions;
- initial/restarted self-play outcomes, arena results by seat and both gates.

Loss alone is not a strength proof: each network changes the MCTS targets used
by the next one. Read its trend beside top-1, illegal mass, arena score and draw
behavior. Arena progress is completion-ordered, so only the final paired result
is meaningful.

## Latest completed research run: v16 through generation 15

This deterministic run started from random weights. Its only selection change
from v15 was conservative rollback: a rejected candidate can no longer become
the source of later training or self-play. `source` is the accepted champion;
`arena` is candidate/champion/draw over 200 games; `probe` is exploratory draws
out of 64. Promotion requires arena ≥55%, mirror 0/4 and probe ≤12/64.

| Gen | Source | Self-play draws | Train policy | Valid policy/WDL | Valid top-1 | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0 | 7/256 | 2.232 | 2.594 / 1.036 | 32.6% | 172/27/1 | 0/4 | 0/64 | promoted |
| 2 | 1 | 15/256 | 1.865 | 2.213 / 1.164 | 39.3% | 163/37/0 | 0/4 | 5/64 | promoted |
| 3 | 2 | 32/256 | 1.801 | 1.998 / 1.111 | 43.5% | 105/77/18 | 0/4 | 4/64 | promoted |
| 4 | 3 | 45/256 | 1.730 | 1.864 / 1.105 | 47.8% | 137/51/12 | 0/4 | 4/64 | promoted |
| 5 | 4 | 39/256 | 1.683 | 1.821 / 0.971 | 47.5% | 129/58/13 | 0/4 | 9/64 | promoted |
| 6 | 5 | 61/256 | 1.631 | 1.760 / 1.100 | 50.5% | 148/34/18 | 0/4 | 11/64 | promoted |
| 7 | 6 | 55/256 | 1.605 | 1.713 / 1.001 | 52.4% | 104/28/68 | 4/4 | 7/64 | draw rejection |
| 8 | 6 | 62/256 | 1.592 | 1.687 / 0.984 | 52.2% | 67/29/104 | 4/4 | 20/64 | draw rejection |
| 9 | 6 | 75/256 | 1.576 | 1.650 / 0.950 | 54.3% | 101/18/81 | 4/4 | 14/64 | draw rejection |
| 10 | 6 | 58/256 | 1.581 | 1.635 / 0.950 | 55.7% | 117/21/62 | 4/4 | 12/64 | draw rejection |
| 11 | 6 | 69/256 | 1.556 | 1.613 / 0.941 | 55.4% | 148/20/32 | 4/4 | 21/64 | draw rejection |
| 12 | 6 | 77/256 | 1.550 | 1.602 / 0.909 | 56.6% | 102/14/84 | 4/4 | 11/64 | draw rejection |
| 13 | 6 | 52/256 | 1.535 | 1.574 / 0.899 | 58.0% | 147/16/37 | 4/4 | 20/64 | draw rejection |
| 14 | 6 | 59/256 | 1.534 | 1.551 / 0.890 | 58.5% | 81/50/69 | 4/4 | 15/64 | draw rejection |
| 15 | 6 | 52/256 | 1.532 | 1.541 / 0.860 | 59.6% | 54/63/83 | 4/4 | 22/64 | strength + draw rejection |

Across 3,840 games and 115,204 positions, validation policy loss fell 40.6%,
top-1 rose from 32.6% to 59.6%, and illegal probability mass fell from 45.9%
to 9.2%. Those are genuine learning signals, and six successive candidates were
promoted. They are not proof of continued playing progress: generations 7–14
all learned deterministic cycles, and generation 15 also regressed below 50%
against champion 6 despite having the best loss.

The rollback successfully prevents rejected weights from poisoning later data,
but retries from champion 6 do not reliably discover a conversion. Omitting the
failed non-starter policy avoids teaching the known failure; it supplies no
better action. The next experiment therefore gives cycle-adjacent restart games
a larger MCTS budget. This remains pure AlphaZero-style internal search: no
solver, oracle, external label or board-size-specific knowledge is involved.

## Rules source

The engine follows the official 3×4 rulebook:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
