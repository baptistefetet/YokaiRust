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
- a Burn residual network with policy and Win/Draw/Loss heads;
- CPU tests and WGPU/Metal training on Apple Silicon;
- parallel self-play, a rolling replay buffer and stable whole-game validation;
- draw-aware search, cycle-adjacent restarts and guarded model promotion;
- atomic checkpoints for an accepted champion and a private learner lineage;
- dataset, optimization, arena and draw diagnostics in JSON reports.

The next product milestone is the Ratatui interface. Training can continue
independently because the UI only needs the stable `Game`, `Action`, `Replay`
and analysis contracts.

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

# Resume from the accepted champion, learner state and replay buffer.
cargo run --release -- train --resume latest --generations 5 --headless
```

`--headless` disables the future TUI, not textual progress. Checkpoints are
stored under the configured model path. `latest` points to the accepted
champion; `learner` may point to a later unpublished checkpoint. Self-play data,
replays and structured generation reports are stored under the configured data
path. Generation-boundary writes are atomic.

### One generation

1. Generate 75% of trajectories from fresh initial states and restart 25% two
   to eight plies before previously observed repetition cycles.
2. Add every resulting position and its official W/D/L result to the rolling
   replay buffer.
3. Resume the learner's weights and Adam moments for a fixed 400 updates.
4. Compare the candidate with the champion on paired random 0–4 ply openings.
5. Measure deterministic initial-position cycles and noisy self-play draws.
6. Promote only when strength and both draw gates pass.

The champion is always the self-play source. A candidate that is strong but
cycles remains a private learner so later updates can cross that plateau. A
candidate that loses the strength arena rolls the learner back to the champion.

### Draw-aware learning

Official draws always remain draws. The stored WDL target is never rewritten.
The pipeline attacks the feedback loop at four different points:

- the WDL head distinguishes a certain draw from an uncertain win/loss mixture;
- self-play values a draw at `+0.25` for the starter and `-0.25` for the
  non-starter, while official play still uses neutral `P(win) - P(loss)`;
- cycle-adjacent restarts shorten the horizon where conversion failed;
- policy loss discounts repetition imitation only in drawn positions where the
  player to move is the non-starter. With the checked-in discount of `0.9`, a
  target carrying 100% repetition mass retains 10% of its policy weight. Its
  WDL loss remains fully weighted.

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
- the mean policy weight after non-starter draw discounting;
- target entropy, action coverage, immediate-draw mass and repetition mass;
- separate draw buckets for starter and non-starter positions;
- initial/restarted self-play outcomes, arena results by seat and both gates.

Loss alone is not a strength proof: each network changes the MCTS targets used
by the next one. Read its trend beside top-1, illegal mass, arena score and draw
behavior. Arena progress is completion-ordered, so only the final paired result
is meaningful.

## Draw-aware v11: 15-generation result

This clean run started from random weights with the WDL head, role-aware draw
utility, targeted restarts and separate learner lineage. It predates the stable
validation split and repetition-weighted policy loss described above; its
validation values are therefore useful trend indicators but not a perfectly
fixed longitudinal holdout.

`arena` is candidate/reference/draw over 200 games. `probe` is exploratory draws
out of 64. A candidate is promoted only when arena ≥55%, mirror is 0/4 and probe
is at most 12/64 (20%).

| Gen | Source champion/learner | Self-play draws | Train policy | Valid policy/WDL | Valid top-1 | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0/0 | 3/256 | 2.190 | 2.463 / 1.011 | 30.8% | 166/34/0 | 0/4 | 2/64 | promoted |
| 2 | 1/1 | 36/256 | 1.801 | 2.185 / 1.086 | 37.3% | 152/45/3 | 0/4 | 5/64 | promoted |
| 3 | 2/2 | 56/256 | 1.736 | 1.934 / 0.693 | 46.6% | 107/45/48 | 0/4 | 8/64 | promoted |
| 4 | 3/3 | 71/256 | 1.683 | 1.788 / 0.665 | 49.8% | 113/47/40 | 4/4 | 15/64 | draw rejection |
| 5 | 3/4 | 74/256 | 1.608 | 1.700 / 0.753 | 54.5% | 63/109/28 | 0/4 | 11/64 | strength rollback |
| 6 | 3/3 | 78/256 | 1.673 | 1.764 / 0.743 | 51.0% | 129/29/42 | 4/4 | 18/64 | draw rejection |
| 7 | 3/6 | 60/256 | 1.605 | 1.664 / 0.696 | 55.2% | 101/31/68 | 4/4 | 13/64 | draw rejection |
| 8 | 3/7 | 61/256 | 1.564 | 1.656 / 0.751 | 55.3% | 148/19/33 | 0/4 | 12/64 | promoted |
| 9 | 8/8 | 75/256 | 1.554 | 1.609 / 0.696 | 56.4% | 109/62/29 | 0/4 | 10/64 | promoted |
| 10 | 9/9 | 86/256 | 1.540 | 1.546 / 0.737 | 58.1% | 88/39/73 | 4/4 | 11/64 | draw rejection |
| 11 | 9/10 | 72/256 | 1.515 | 1.559 / 0.616 | 57.7% | 106/17/77 | 4/4 | 14/64 | draw rejection |
| 12 | 9/11 | 60/256 | 1.515 | 1.569 / 0.656 | 57.4% | 89/28/83 | 4/4 | 18/64 | draw rejection |
| 13 | 9/12 | 83/256 | 1.484 | 1.509 / 0.750 | 59.4% | 113/35/52 | 4/4 | 14/64 | draw rejection |
| 14 | 9/13 | 77/256 | 1.475 | 1.486 / 0.675 | 61.3% | 123/25/52 | 4/4 | 15/64 | draw rejection |
| 15 | 9/14 | 74/256 | 1.467 | 1.486 / 0.679 | 60.4% | 118/29/53 | 4/4 | 26/64 | draw rejection |

The run produced 3,840 games and 108,077 positions. Initial-state trajectories
drew 470/2,880 (16.3%); deliberately difficult restarts drew 496/960 (51.7%).
Drawn games supplied 12,731 positions, 11.8% of the final buffer. Promotions
were generations 1, 2, 3, 8 and 9.

Learning clearly continued: from generation 1 to 15, validation policy loss
fell 39.7%, WDL loss fell 32.9%, policy top-1 rose from 30.8% to 60.4%, and
illegal mass fell from 44.4% to 6.3%. Behavior also improved through champion
9. The remaining failure is narrower: candidates 10–15 all passed the strength
arena, often decisively, but all entered the same deterministic mirror cycle.

An internal audit of the generation-6 buffer found that 1,002/2,084 drawn
non-starter positions assigned policy mass to a repetition. When present, that
mass averaged 76.4%; decisive non-starter positions averaged only 2.5%
repetition mass overall. That measurement motivates the new policy weighting.
The next clean run must determine whether it shortens the six-candidate plateau
without weakening arena performance. No external evaluator or generated label
is part of that experiment.

## Rules source

The engine follows the official 3×4 rulebook:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
