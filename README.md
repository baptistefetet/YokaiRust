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
- atomic checkpoints for an accepted champion and a private learner lineage;
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

# Resume from the accepted champion, learner state and replay buffer.
cargo run --release -- train --resume latest --generations 5 --headless
```

`--headless` disables the future TUI, not textual progress. Checkpoints are
stored under the configured model path. `latest` points to the accepted
champion; `learner` may point to a later unpublished checkpoint. Self-play data,
replays and structured generation reports are stored under the configured data
path. Generation-boundary writes are atomic.

### One generation

1. Let the private learner generate 75% of trajectories from fresh initial
   states and restart 25% two to eight plies before observed repetition cycles.
2. Add every resulting position and its official W/D/L result to the rolling
   replay buffer.
3. Resume the learner's weights and Adam moments for a fixed 400 updates.
4. Compare the candidate with the champion on paired random 0–4 ply openings.
5. Measure deterministic initial-position cycles and noisy self-play draws.
6. Promote only when strength and both draw gates pass.

The two checkpoint pointers have distinct jobs. `champion` is the conservative,
published reference. `learner` is the model that trains **and generates the next
self-play batch**. A strong candidate that cycles remains private but advances
the learner, so later generations can discover a way out of that plateau. A
candidate that loses the strength arena rolls the learner back to the champion.

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

## Latest completed research run: v13 through generation 15

This clean run started from random weights with the stable validation split,
WDL head, role-aware draw utility and targeted restarts. It was the first run
where the private learner, rather than the last accepted champion, generated
the next self-play batch after a draw-only rejection.

`arena` is candidate/reference/draw over 200 games. `probe` is exploratory draws
out of 64. A candidate is promoted only when arena ≥55%, mirror is 0/4 and probe
is at most 12/64 (20%).

| Gen | Champion/learner source | Self-play draws | Train policy | Valid policy/WDL | Valid top-1 | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0/0 | 3/256 | 2.203 | 2.582 / 1.858 | 33.8% | 154/46/0 | 0/4 | 3/64 | promoted |
| 2 | 1/1 | 33/256 | 1.821 | 2.124 / 1.263 | 39.3% | 160/38/2 | 0/4 | 4/64 | promoted |
| 3 | 2/2 | 44/256 | 1.757 | 2.031 / 1.006 | 41.9% | 108/38/54 | 4/4 | 4/64 | draw rejection |
| 4 | 2/3 | 52/256 | 1.706 | 1.905 / 0.930 | 45.8% | 150/49/1 | 0/4 | 15/64 | draw rejection |
| 5 | 2/4 | 81/256 | 1.663 | 1.858 / 0.996 | 47.1% | 141/48/11 | 4/4 | 11/64 | draw rejection |
| 6 | 2/5 | 75/256 | 1.631 | 1.819 / 0.949 | 47.7% | 167/21/12 | 4/4 | 14/64 | draw rejection |
| 7 | 2/6 | 80/256 | 1.592 | 1.774 / 0.936 | 48.9% | 185/13/2 | 4/4 | 12/64 | draw rejection |
| 8 | 2/7 | 78/256 | 1.570 | 1.726 / 0.956 | 51.2% | 182/15/3 | 0/4 | 19/64 | draw rejection |
| 9 | 2/8 | 88/256 | 1.553 | 1.714 / 0.933 | 52.0% | 182/16/2 | 4/4 | 22/64 | draw rejection |
| 10 | 2/9 | 105/256 | 1.537 | 1.679 / 0.934 | 53.4% | 186/13/1 | 4/4 | 25/64 | draw rejection |
| 11 | 2/10 | 130/256 | 1.513 | 1.670 / 0.943 | 53.9% | 189/8/3 | 4/4 | 25/64 | draw rejection |
| 12 | 2/11 | 97/256 | 1.503 | 1.649 / 0.921 | 54.1% | 192/6/2 | 4/4 | 40/64 | draw rejection |
| 13 | 2/12 | 145/256 | 1.481 | 1.627 / 0.902 | 55.8% | 185/13/2 | 0/4 | 21/64 | draw rejection |
| 14 | 2/13 | 106/256 | 1.459 | 1.609 / 0.904 | 56.6% | 180/18/2 | 0/4 | 27/64 | draw rejection |
| 15 | 2/14 | 117/256 | 1.455 | 1.597 / 0.949 | 56.7% | 188/11/1 | 4/4 | 35/64 | draw rejection |

The run produced 3,840 games and 99,293 positions. From generation 1 to 15,
validation policy loss fell 38.1%, policy top-1 rose from 33.8% to 56.7%, and
illegal mass fell from 45.1% to 6.6%. The private lineage also became vastly
stronger than champion 2. These are real improvements, but not a solved training
loop: self-play draws rose to 117/256 at generation 15 and drawn games supplied
14,758 positions (14.9%) of the final buffer.

The experiment found why the previous weighting was insufficient. It only
discounted policy in non-starter draw positions in proportion to visit mass on
an already repeated state. Even at generation 15 the mean policy weight was
0.977, so 97.7% of the original imitation signal remained. The checked-in next
experiment uses the simpler trajectory-level rule described above: a drawn
non-starter position trains WDL but has policy weight zero. No external
evaluator or generated label is involved.

## Rules source

The engine follows the official 3×4 rulebook:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
