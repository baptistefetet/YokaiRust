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
- cycle-adjacent restarts shorten the horizon where conversion failed and use
  800 MCTS simulations per move there instead of the regular 200;
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

## Latest completed research run: v17 through generation 15

This deterministic run started from random weights and kept the conservative
rollback from v16. Its one experimental change was to raise MCTS from 200 to
800 simulations only in the 25% cycle-adjacent restart trajectories. `source`
is the accepted champion; `arena` is candidate/champion/draw over 200 games;
`probe` is exploratory draws out of 64. Promotion requires arena ≥55%, mirror
0/4 and probe ≤12/64.

| Gen | Source | Self-play draws | Train policy | Valid policy/WDL | Valid top-1 | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0 | 7/256 | 2.232 | 2.594 / 1.036 | 32.6% | 172/27/1 | 0/4 | 0/64 | promoted |
| 2 | 1 | 27/256 | 1.861 | 2.233 / 1.244 | 36.9% | 118/78/4 | 0/4 | 1/64 | promoted |
| 3 | 2 | 36/256 | 1.799 | 2.071 / 1.106 | 41.1% | 152/47/1 | 0/4 | 4/64 | promoted |
| 4 | 3 | 34/256 | 1.751 | 2.004 / 1.112 | 41.1% | 103/91/6 | 0/4 | 2/64 | strength rejection |
| 5 | 3 | 45/256 | 1.766 | 1.924 / 0.951 | 44.9% | 161/34/5 | 4/4 | 4/64 | draw rejection |
| 6 | 3 | 52/256 | 1.764 | 1.888 / 0.846 | 46.6% | 170/24/6 | 0/4 | 4/64 | promoted |
| 7 | 6 | 37/256 | 1.692 | 1.821 / 0.813 | 49.5% | 130/51/19 | 0/4 | 5/64 | promoted |
| 8 | 7 | 43/256 | 1.663 | 1.762 / 0.804 | 50.4% | 60/128/12 | 0/4 | 6/64 | strength rejection |
| 9 | 7 | 58/256 | 1.653 | 1.748 / 0.773 | 52.4% | 109/65/26 | 4/4 | 9/64 | draw rejection |
| 10 | 7 | 53/256 | 1.652 | 1.747 / 0.738 | 51.6% | 126/32/42 | 4/4 | 4/64 | draw rejection |
| 11 | 7 | 53/256 | 1.646 | 1.719 / 0.718 | 52.3% | 121/61/18 | 0/4 | 10/64 | promoted |
| 12 | 11 | 63/256 | 1.618 | 1.695 / 0.732 | 53.3% | 65/63/72 | 4/4 | 11/64 | strength + draw rejection |
| 13 | 11 | 66/256 | 1.619 | 1.687 / 0.715 | 53.5% | 75/69/56 | 0/4 | 8/64 | strength rejection |
| 14 | 11 | 70/256 | 1.616 | 1.663 / 0.729 | 54.6% | 82/71/47 | 4/4 | 9/64 | strength + draw rejection |
| 15 | 11 | 65/256 | 1.597 | 1.655 / 0.734 | 55.0% | 119/23/58 | 4/4 | 9/64 | draw rejection |

Across 3,840 games and 114,563 positions, validation policy loss fell 36.2%,
top-1 rose from 32.6% to 55.0%, and illegal probability mass fell from 45.9%
to 10.2%. Six candidates were promoted. Unlike v16, learning crossed the old
generation-7 cycle and reached champion 11: focused search therefore moved the
playing plateau, not just the losses.

It did not eliminate the failure. Generation 15 beat champion 11 by 119–23 with
58 draws, yet its greedy initial-position mirror repeated in all four games.
This is a useful distinction: the candidate improved broadly across random
openings while one exact opening policy collapsed into a cycle. The repeated
strength regressions immediately after promotions also suggest that 400 Adam
updates at the fixed `0.001` rate perturb a mature policy too abruptly.

The next run keeps deeper restarts and reduces the learning rate to `0.00025`
once the accepted source reaches generation 7. The schedule follows the champion
rather than the attempt number, so rejected candidates cannot advance it. It
remains pure from-scratch learning with no solver, oracle, external label or
board-size-specific knowledge.

## Rules source

The engine follows the official 3×4 rulebook:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
