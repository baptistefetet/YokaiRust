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
- an 80-plane board-history encoder plus 50 non-spatial features and
  action-aligned repetition context;
- a Burn residual network with policy and Win/Draw/Loss (WDL) heads;
- CPU tests and WGPU/Metal training on Apple Silicon;
- parallel self-play, a rolling replay buffer and stable whole-game validation;
- draw-aware search, cycle-adjacent restarts and guarded model promotion;
- atomic checkpoints for the accepted champion and its optimizer state;
- dataset, optimization, arena and draw diagnostics in JSON reports.

The next product milestone is the Ratatui interface. Training can continue
independently because the UI only needs the stable `Game`, `Action`, `Replay`
and analysis contracts.

## JavaScript predecessor

This Rust project follows an earlier browser implementation:
[`baptistefetet/yokai`](https://github.com/baptistefetet/yokai). That repository
remains the visual reference for the future interface. Its Phaser front end
contains the 3×4 and 5×6 boards and piece sprites, animated moves, captures,
drops and promotions, plus overlays that expose the network's policy and value
predictions.

It is also an architectural reference for the learning work. Its TensorFlow.js
network encodes the position from the player-to-move perspective and processes
the board with two small convolutional layers. Captured pieces have neither an
order nor a position on the board, so each hand is represented separately by
one scalar count per piece type. The spatial features and these global scalars
then join a shared dense trunk, which branches into a scalar value head and a
policy head. Early experiments with datasets restricted to endgame positions
gave particularly convincing predictions: both the preferred moves and the
position evaluations were often close to what a human player would expect.

The JavaScript bootstrap also differed in an important way. Until the first
network was promoted, self-play did not query the randomly initialized model:
MCTS started with uniform action priors and evaluated leaves with random
rollouts governed only by the game rules. Neural policy and value predictions
were enabled only after that initial dataset had trained a first accepted
model.

That result is an important baseline, but it answers an easier and more local
question than the current project. Endgame-only data teaches the network on a
short horizon with a dense, reliable terminal signal. YokaiRust is trying to
learn the complete game from random weights and self-play, including the long
credit-assignment problem that precedes those endgames. The old project is
therefore a source of UI assets, implementation ideas and experimental
comparisons.

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

Project changes are developed, committed, and pushed directly on `main`. Do not
create feature branches or pull requests unless explicitly requested.

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check

# Explicit Apple Metal training/checkpoint preflight.
cargo test --test neural metal_training_state_round_trip_runs_one_real_update -- --ignored --exact
```

Release builds use Cargo's portable defaults. Runtime datasets and models are
ignored by Git and stay inside `data/` and `models/`.

## Text analysis and replays

```bash
# Pure MCTS from the official initial position.
cargo run -- analyze [simulations] [seed]

# Validate every action and print a versioned replay.
cargo run -- replay path/to/game.json

# Evaluate every saved checkpoint by distance from its validation result.
cargo run --release -- diagnose-endgames --config config/training.toml
```

`Mcts::search` is deterministic and noise-free. `Mcts::search_self_play` is the
only entry point that injects Dirichlet noise. A fixed seed reproduces search
and temperature-based action selection.

The endgame diagnostic is read-only. It evaluates every saved checkpoint on
the final buffer's stable whole-game validation split and writes versioned JSON
under `data/.../diagnostics/endgame-distance/`.

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
   states and restart 25% one to eight plies before observed repetition cycles.
2. Add every resulting position and its official W/D/L result to the rolling
   replay buffer.
3. Resume the champion's weights and Adam moments for a fixed 400 updates.
4. Compare the candidate with the champion on paired random 0–4 ply openings.
5. Measure deterministic initial-position cycles and noisy self-play draws.
6. Promote only when strength and the noisy self-play draw gate pass; keep the
   deterministic mirror result as a diagnostic.

The champion is the sole source of weights, optimizer state and self-play. A
candidate becomes the new champion only after both promotion checks pass. A
rejected candidate is kept as an experiment report but never generates
training data.
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
- policy loss normally omits the non-starter's moves from a game it failed to
  convert. At an immediate third repetition, it removes only that known drawing
  action and renormalizes MCTS's remaining alternatives. The starter's defence
  and every official WDL target remain fully weighted.

This last rule follows the observed failure directly: the starter is allowed to
learn a drawing defence, while the non-starter should not receive a sharp
policy-imitation reward for failing to convert. Once the rules expose the exact
action that ends in a draw, the other search visits finally provide a positive
training target. This uses only self-play and the generic repetition rule, so it
applies unchanged to the planned 5×6 game.

The early decisive-tail curriculum is temporary. It oversamples tactical ends
during bootstrap, widens geometrically, and stops at candidate 11. The complete
buffer is always retained.

### Validation and metrics

Whole games are assigned to validation by a stable hash of their generation and
seed. Adding a new generation therefore does not reshuffle old games between
training and validation. This makes longitudinal loss curves interpretable
while preventing positions from one game leaking across both sets.

Reports include:

- policy and WDL cross-entropy plus scalar value MSE;
- policy entropy, top-1 agreement and illegal probability mass;
- WDL top-1, value calibration and draw-probability error;
- the mean policy weight after omitting unresolved drawn non-starter targets;
- target entropy, action coverage, immediate-draw mass and repetition mass;
- separate draw buckets for starter and non-starter positions;
- initial/restarted self-play outcomes, arena results by seat and both gates.

Loss alone is not a strength proof: each network changes the MCTS targets used
by the next one. Read its trend beside top-1, illegal mass, arena score and draw
behavior. Arena progress is completion-ordered, so only the final paired result
is meaningful.

## Previous completed research run: v21 through generation 15

This deterministic run started from random weights. Compared with v20, its only
change was the generation-zero bootstrap: until candidate 1 was accepted,
self-play MCTS used uniform priors and seeded random rollouts rather than the
random neural policy and WDL heads. Candidate training and every promotion
check remained neural. After that first promotion, generation 2 and all later
self-play used the ordinary neural evaluator.

`source` is the last accepted champion; `arena` is candidate/champion/draw over
200 games; `probe` is exploratory draws out of 64. `P/V` means policy/value
(WDL). Promotion requires arena ≥55%, mirror 0/4 and probe ≤12/64.

| Gen | Source | Self-play draws | Train loss P/V | Valid loss P/V | Valid top-1 P/V | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0 | 2/256 | 2.449 / 0.256 | 2.901 / 1.419 | 23.3% / 49.6% | 157/43/0 | 0/4 | 1/64 | promoted (rollout) |
| 2 | 1 | 27/256 | 1.910 / 0.303 | 2.332 / 1.431 | 35.3% / 52.5% | 107/64/29 | 0/4 | 2/64 | promoted |
| 3 | 2 | 47/256 | 1.776 / 0.337 | 2.145 / 1.150 | 38.7% / 58.3% | 128/22/50 | 0/4 | 0/64 | promoted |
| 4 | 3 | 52/256 | 1.715 / 0.363 | 2.024 / 1.128 | 42.3% / 59.9% | 96/67/37 | 4/4 | 0/64 | draw rejection |
| 5 | 3 | 50/256 | 1.706 / 0.439 | 1.940 / 0.973 | 45.5% / 62.8% | 116/31/53 | 4/4 | 4/64 | draw rejection |
| 6 | 3 | 48/256 | 1.707 / 0.480 | 1.907 / 0.768 | 47.0% / 68.0% | 121/29/50 | 4/4 | 2/64 | draw rejection |
| 7 | 3 | 47/256 | 1.708 / 0.513 | 1.855 / 0.686 | 48.8% / 67.6% | 121/34/45 | 0/4 | 1/64 | promoted |
| 8 | 7 | 57/256 | 1.620 / 0.500 | 1.772 / 0.756 | 52.3% / 66.2% | 142/29/29 | 0/4 | 2/64 | promoted |
| 9 | 8 | 51/256 | 1.602 / 0.502 | 1.757 / 0.776 | 51.8% / 66.2% | 94/92/14 | 0/4 | 3/64 | strength rejection |
| 10 | 8 | 53/256 | 1.602 / 0.541 | 1.748 / 0.740 | 52.3% / 66.8% | 158/32/10 | 4/4 | 3/64 | draw rejection |
| 11 | 8 | 51/256 | 1.614 / 0.559 | 1.741 / 0.724 | 53.0% / 66.6% | 160/25/15 | 0/4 | 1/64 | promoted |
| 12 | 11 | 54/256 | 1.587 / 0.560 | 1.712 / 0.729 | 53.1% / 67.1% | 159/23/18 | 0/4 | 2/64 | promoted |
| 13 | 12 | 63/256 | 1.576 / 0.559 | 1.689 / 0.762 | 53.5% / 66.4% | 133/44/23 | 0/4 | 7/64 | promoted |
| 14 | 13 | 54/256 | 1.557 / 0.565 | 1.674 / 0.759 | 53.9% / 66.3% | 132/46/22 | 0/4 | 6/64 | promoted |
| 15 | 14 | 59/256 | 1.552 / 0.574 | 1.659 / 0.743 | 54.7% / 66.9% | 37/57/106 | 4/4 | 6/64 | strength + draw rejection |

Across 3,840 games and 120,165 positions, validation policy loss fell 42.8%
from candidate 1 to 15 and policy top-1 rose from 23.3% to 54.7%. Validation
WDL loss fell 47.7% and WDL top-1 rose from 49.6% to 66.9%; illegal policy mass
fell from 52.7% to 10.2%. Training WDL loss moved in the opposite direction,
from 0.256 to 0.574, because the rollout generation contained only two draws
and was much easier than the progressively draw-rich neural buffer. The
validation trend and frozen diagnostic are the meaningful generalization
signals, not that raw training comparison by itself.

Nine candidates were promoted. The accepted sequence is `0 -> 1 -> 2 -> 3 ->
7 -> 8 -> 11 -> 12 -> 13 -> 14`, versus v20's `0 -> 1 -> 2 -> 3 -> 5 -> 6 ->
8 -> 12 -> 13`. The rollout bootstrap therefore produced a decisive first
promotion with only 2/256 self-play draws and ultimately one additional
promotion and a generation-14 champion. It did not make every early target
better: v21 remained stuck on champion 3 for candidates 4 through 6, whereas
v20 promoted candidates 5 and 6.

The late cyclic failure also remains. Candidate 10 beat champion 8 by
158 wins to 32 with only 10 arena draws, yet all four mirrors cycled. Candidate
15 achieved the run's best policy validation loss and top-1, but lost to
champion 14 by 37 wins to 57 with 106 draws and also cycled in all four mirrors.
Losses are therefore necessary learning indicators, but not substitutes for
paired strength and behavioral gates: the network can imitate its increasingly
cyclic search targets more accurately while becoming a worse game-playing
teacher.

## Handoff: next JavaScript-informed research program

This section is the authoritative starting point for the next development
conversation. Preserve one changed variable per run: the objective is to learn
which JavaScript idea helps, not merely to produce one incomparable mixture.

Keep this README current throughout the research, not only at the end of the
whole program. After every completed 15-generation run, and before starting the
next one:

- replace the latest-results table with the new generation 1–15 measurements;
- record the single changed variable, accepted champion sequence, losses,
  top-1 metrics, arena results and draw-gate results;
- state what the experiment demonstrated, including negative or inconclusive
  outcomes;
- update the next experiment and its unchanged baseline explicitly;
- record the runtime data/model paths that must be retained for comparison;
- remove obsolete plans and stale provisional conclusions rather than growing
  an ambiguous chronological log.

Commit and push each implementation separately from its completed experimental
report. A new conversation must be able to recover the current state and next
action from this README plus the structured JSON reports alone.

### 0. Frozen comparison artifacts

The v20, v21 and v22 runs are complete through candidate 15 and no trainer is
writing their directories. Their accepted champions are generations 13, 14 and
13 respectively. In all three runs `reports/generation-000015.json` records the
final rejected candidate. Retain all six ignored runtime directories for
subsequent comparisons:

- `data/alpha-zero-draw-aware-v20` and `models/alpha-zero-draw-aware-v20`;
- `data/alpha-zero-draw-aware-v21` and `models/alpha-zero-draw-aware-v21`;
- `data/alpha-zero-draw-aware-v22` and `models/alpha-zero-draw-aware-v22`.

The structured JSON reports are authoritative if prose and runtime files ever
disagree.

The early-stopped v23 capacity ablation is not a complete 15-candidate run.
Its reports stop at candidate 7 and its accepted champion is generation 7.
Retain `data/alpha-zero-draw-aware-v23` and
`models/alpha-zero-draw-aware-v23` as explicitly partial comparison artifacts.

### 1. Frozen v21 endgame-distance diagnostic

The read-only diagnostic is complete for every saved v21 checkpoint from 0
through 15. Its versioned JSON reports are in
`data/alpha-zero-draw-aware-v21/diagnostics/endgame-distance/`. Every checkpoint
was evaluated against the same final-buffer split: 410 complete validation
games, comprising 84 draws and 326 decisive results, with 12,631 positions.
The split is the stable seed-42 whole-game split, so no game crosses between
training and validation. The diagnostic changes neither sampling nor gradients.

The table below is final candidate 15. Policy loss uses the configured
draw-policy weighting, including omission of unresolved drawn non-starter
targets; example counts still include every official position.

| Distance | Outcome | Examples | Policy loss | Policy top-1 | WDL loss | WDL top-1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | all | 410 | 1.692 | 57.6% | 0.168 | 94.4% |
| 1 | draw | 84 | 2.221 | 10.7% | 0.270 | 91.7% |
| 1 | decisive | 326 | 1.556 | 69.6% | 0.142 | 95.1% |
| 2–4 | all | 1,151 | 1.669 | 58.8% | 0.412 | 84.4% |
| 2–4 | draw | 189 | 0.175 | 95.2% | 0.372 | 93.1% |
| 2–4 | decisive | 962 | 1.854 | 51.7% | 0.419 | 82.7% |
| 5–8 | all | 1,335 | 1.779 | 54.6% | 0.528 | 80.0% |
| 5–8 | draw | 118 | 0.477 | 86.4% | 0.996 | 78.8% |
| 5–8 | decisive | 1,217 | 1.834 | 51.5% | 0.483 | 80.1% |
| 9–16 | all | 2,210 | 1.782 | 50.3% | 0.673 | 72.2% |
| 9–16 | draw | 57 | 1.706 | 50.9% | 3.016 | 12.3% |
| 9–16 | decisive | 2,153 | 1.783 | 50.3% | 0.611 | 73.8% |
| 17+ | all | 7,525 | 1.599 | 55.3% | 0.883 | 58.9% |
| 17+ | draw | 245 | 1.482 | 60.8% | 3.434 | 2.0% |
| 17+ | decisive | 7,280 | 1.603 | 55.1% | 0.797 | 60.8% |

On this frozen split, candidate 1 to 15 policy loss improves in every distance
bucket: by 42.9% at one ply, 43.6% at `2–4`, 40.6% at `5–8`, 39.6% at `9–16`
and 41.8% at `17+`. WDL loss also improves everywhere, but the gain shrinks
with horizon: 82.7%, 56.8%, 47.6%, 43.6% and 36.1% respectively. Candidate
15's WDL top-1 correspondingly falls from 94.4% one ply before the result to
58.9% at `17+`.

The remaining failure is specifically long-horizon draws. Candidate 15 reaches
91.7% drawn WDL top-1 at one ply and 93.1% at `2–4`, but only 12.3% at `9–16`
and 2.0% at `17+`. Champion 14 is similar at long range: 4.5% top-1 with 3.322
WDL loss on drawn `17+` positions. v21 improves all-position long-range WDL
over v20 candidate 15 (0.883 / 58.9% versus 0.961 / 54.3%), but worsens the
draw-only result (3.434 / 2.0% versus 2.389 / 5.3%). The splits differ because
the runs generated different games, so this cross-run comparison is directional
rather than paired.

### 2. v21 conclusion: rollout bootstrap helps initialization, not cycles

The rollout implementation and 15-generation experiment are complete. Reports
persist the evaluator metadata, and generation 1 records
`random_rollout { max_plies: 512 }`; generation 2 records `neural`, proving the
switch happened immediately after the first promotion. Candidate training,
arena, mirror and probe remained neural throughout.

The bootstrap gives a cleaner first dataset and a stronger final accepted
sequence than v20, but it does not solve the feedback loop. It neither prevents
mid-run mirror rejections nor teaches the WDL head to recognize distant draws.
The result supports keeping rollout bootstrap as the baseline while changing a
different variable next; it does not justify relaxing either draw gate.

### 3. Completed experiment: v22 separates global features

The v22 implementation and 15-generation run are complete in the
`alpha-zero-draw-aware-v22` model/data paths. The experiment kept v21's rollout
bootstrap, seed 42, game budgets, replay sampling, optimizer schedule, restart
logic, promotion threshold and draw gates unchanged. Its sole ablation was the
encoder/network boundary:

- a hand is an unordered count per droppable piece type and player;
- keep every board-history feature spatial;
- move the existing normalized hand counts, repetition count and starter flag
  out of constant board planes;
- preserve those global features for every existing history frame;
- concatenate the scalar branch after the convolutions and feed the resulting
  shared representation to both policy and WDL heads, as in the JavaScript
  network.

Encoder version 4 supplies 80 spatial planes and 50 global scalars. Model
format version 3 adds a 64-unit dense shared representation after the residual
tower. CPU tests cover tensor shapes, historical scalar preservation, both-head
connectivity and exact checkpoint round trips. The strict Metal preflight also
passes: it performs one real autodiff update, saves the model and Adam state,
reloads both and checks the policy logits exactly.

`source` is the last accepted champion; `arena` is candidate/champion/draw over
200 games; `probe` is exploratory draws out of 64. `P/V` means policy/value
(WDL). Promotion requires arena ≥55%, mirror 0/4 and probe ≤12/64.

| Gen | Source | Self-play draws | Train loss P/V | Valid loss P/V | Valid top-1 P/V | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0 | 2/256 | 1.800 / 0.281 | 2.633 / 1.176 | 26.0% / 54.6% | 180/20/0 | 0/4 | 2/64 | promoted (rollout) |
| 2 | 1 | 31/256 | 1.682 / 0.339 | 2.154 / 1.118 | 38.6% / 59.1% | 117/23/60 | 4/4 | 5/64 | draw rejection |
| 3 | 1 | 40/256 | 1.758 / 0.469 | 2.073 / 0.762 | 43.0% / 63.2% | 171/11/18 | 0/4 | 3/64 | promoted |
| 4 | 3 | 59/256 | 1.708 / 0.541 | 1.926 / 0.762 | 44.7% / 62.7% | 129/67/4 | 0/4 | 2/64 | promoted |
| 5 | 4 | 47/256 | 1.649 / 0.531 | 1.860 / 0.830 | 47.6% / 62.8% | 137/39/24 | 0/4 | 4/64 | promoted |
| 6 | 5 | 68/256 | 1.629 / 0.548 | 1.806 / 0.814 | 48.3% / 62.9% | 115/44/41 | 0/4 | 3/64 | promoted |
| 7 | 6 | 55/256 | 1.584 / 0.567 | 1.752 / 0.818 | 48.3% / 62.3% | 115/26/59 | 4/4 | 6/64 | draw rejection |
| 8 | 6 | 61/256 | 1.575 / 0.597 | 1.708 / 0.759 | 51.6% / 63.6% | 93/46/61 | 4/4 | 3/64 | draw rejection |
| 9 | 6 | 56/256 | 1.558 / 0.644 | 1.682 / 0.724 | 51.6% / 65.3% | 94/19/87 | 4/4 | 5/64 | draw rejection |
| 10 | 6 | 55/256 | 1.557 / 0.650 | 1.630 / 0.713 | 54.3% / 65.1% | 106/18/76 | 4/4 | 3/64 | draw rejection |
| 11 | 6 | 48/256 | 1.547 / 0.660 | 1.614 / 0.715 | 55.4% / 65.9% | 157/14/29 | 0/4 | 12/64 | promoted |
| 12 | 11 | 69/256 | 1.446 / 0.642 | 1.531 / 0.722 | 58.8% / 65.6% | 123/55/22 | 0/4 | 6/64 | promoted |
| 13 | 12 | 68/256 | 1.441 / 0.640 | 1.515 / 0.768 | 59.1% / 63.6% | 83/23/94 | 0/4 | 9/64 | promoted |
| 14 | 13 | 75/256 | 1.424 / 0.658 | 1.489 / 0.780 | 60.0% / 64.0% | 76/74/50 | 4/4 | 13/64 | strength + both draw gates |
| 15 | 13 | 65/256 | 1.417 / 0.678 | 1.478 / 0.749 | 60.8% / 64.9% | 138/19/43 | 4/4 | 11/64 | draw rejection |

Across 3,840 games and 147,819 positions, validation policy loss fell 43.9%
from candidate 1 to 15, policy top-1 rose from 26.0% to 60.8%, and illegal
policy mass fell from 24.0% to 4.8%. Validation WDL loss fell 36.3% and WDL
top-1 rose from 54.6% to 64.9%, but the best WDL top-1 was already 65.9% at
candidate 11. The accepted sequence is `0 -> 1 -> 3 -> 4 -> 5 -> 6 -> 11 ->
12 -> 13`: eight candidates were promoted and generation 13 is the final
champion.

The scalar branch materially helped policy learning but did not remove the
cyclic feedback loop. Directionally, candidate 15 improves over v21 candidate
15 from 1.659 to 1.478 validation policy loss and from 54.7% to 60.8% policy
top-1. Its WDL result is slightly worse: 0.749 versus 0.743 loss and 64.9%
versus 66.9% top-1. These are different self-generated validation sets, so the
cross-run comparison is not paired. Behaviorally, candidates 7 through 10 all
beat champion 6 in the arena but cycled in every deterministic mirror;
candidate 11 finally escaped and was promoted. Candidate 15 later beat
champion 13 by 138/19/43 but again cycled in all four mirrors.

#### v22 frozen endgame-distance diagnostic

Every v22 checkpoint from 0 through 15 was evaluated against the same final
buffer split. The 405 complete validation games contain 89 draws and 316
decisive results, with 15,395 positions. Reports are in
`data/alpha-zero-draw-aware-v22/diagnostics/endgame-distance/`. The final
candidate 15 result is:

| Distance | Outcome | Examples | Policy loss | Policy top-1 | WDL loss | WDL top-1 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | all | 405 | 1.364 | 79.3% | 0.180 | 95.1% |
| 1 | draw | 89 | 1.837 | 86.5% | 0.493 | 85.4% |
| 1 | decisive | 316 | 1.231 | 77.2% | 0.092 | 97.8% |
| 2–4 | all | 1,156 | 1.412 | 65.9% | 0.325 | 88.3% |
| 2–4 | draw | 222 | 0.310 | 90.5% | 0.553 | 83.3% |
| 2–4 | decisive | 934 | 1.568 | 60.1% | 0.271 | 89.5% |
| 5–8 | all | 1,360 | 1.631 | 56.3% | 0.424 | 85.1% |
| 5–8 | draw | 176 | 0.644 | 89.8% | 0.990 | 64.8% |
| 5–8 | decisive | 1,184 | 1.704 | 51.4% | 0.340 | 88.1% |
| 9–16 | all | 2,324 | 1.603 | 54.9% | 0.608 | 75.9% |
| 9–16 | draw | 130 | 1.416 | 56.9% | 2.780 | 0.8% |
| 9–16 | decisive | 2,194 | 1.610 | 54.8% | 0.479 | 80.3% |
| 17+ | all | 10,150 | 1.444 | 61.4% | 0.896 | 55.8% |
| 17+ | draw | 517 | 1.145 | 68.1% | 2.741 | 0.2% |
| 17+ | decisive | 9,633 | 1.455 | 61.0% | 0.797 | 58.8% |

From candidate 1 to 15 on this frozen split, policy loss improves by 41.1%,
43.9%, 37.5%, 37.1% and 41.0% across the five distance buckets. WDL loss
improves by 82.9%, 73.0%, 59.7%, 45.0% and only 29.7% as the horizon grows.
The intended long-range draw improvement did not occur: draw WDL top-1 falls
to 0.8% at 9–16 plies and 0.2% at 17+. Compared directionally with v21
candidate 15, the 17+ draw loss is lower (2.741 versus 3.434), but top-1 is
also lower (0.2% versus 2.0%); the frozen splits contain different games and
cannot support a paired claim.

#### v22 speed baseline before training

There is no equivalent preserved v21 measurement, so these numbers are a v22
baseline rather than evidence of a speedup or regression. They were measured
on 2026-08-12 on an Apple M4 Max (40-core GPU, 128 GB) with macOS 26.5.2,
Rust 1.97.1, Burn 0.21.0/Metal and an optimized release build. The table reports
the median of three consecutive runs; each run performs an untimed warmup for
every batch size before measuring 4,096 positions (at least 16 batches).

| Batch | Median ms/batch | Median positions/s | Observed positions/s |
| ---: | ---: | ---: | ---: |
| 1 | 2.258 | 443 | 274–465 |
| 8 | 2.168 | 3,690 | 3,684–3,690 |
| 16 | 2.171 | 7,371 | 7,281–7,371 |
| 32 | 2.217 | 14,436 | 14,408–14,457 |
| 64 | 2.227 | 28,744 | 27,750–29,412 |
| 128 | 2.514 | 50,925 | 46,458–53,533 |

Reproduce the microbenchmark with:

```bash
cargo test --release --test performance benchmark_metal_inference_batches -- --ignored --exact --nocapture
```

Generation 1 used rollout search and therefore performed no neural inference.
The neural self-play stages processed 26,350,089 positions at a
backend-time-weighted 36,458 positions/s. These end-to-end figures include the
batching behavior created by concurrent search; unlike the isolated benchmark,
they are affected by game length and worker availability.

| Gen | Total elapsed | Inference positions | Average batch | Backend positions/s |
| ---: | ---: | ---: | ---: | ---: |
| 2 | not preserved | 1,769,053 | 91.3 | 38,349 |
| 3 | not preserved | 1,393,378 | 84.5 | 34,885 |
| 4 | not preserved | 1,458,090 | 91.7 | 36,885 |
| 5 | not preserved | 1,566,943 | 61.9 | 26,820 |
| 6 | 3:17 (resumed) | 1,674,344 | 97.6 | 39,755 |
| 7 | 8:02 | 2,074,804 | 92.5 | 38,013 |
| 8 | 5:59 | 1,846,660 | 96.3 | 39,329 |
| 9 | 4:08 | 1,984,307 | 97.2 | 39,461 |
| 10 | 4:24 | 1,970,321 | 79.2 | 33,978 |
| 11 | 6:33 | 2,098,732 | 71.1 | 30,484 |
| 12 | 6:21 | 2,098,025 | 73.6 | 31,427 |
| 13 | 5:35 | 2,044,663 | 99.1 | 39,807 |
| 14 | 5:18 | 2,387,947 | 108.5 | 44,558 |
| 15 | 5:32 | 1,982,822 | 104.2 | 42,881 |

The resumed candidate 6–15 segment took 55:10 in the pipeline. Candidate 6
reused its persisted self-play, so its elapsed time is not a full-generation
measurement. Wall times for candidates 2–5 were not preserved. Arena inference
measurements were preserved for the resumed segment; each cell is
candidate/champion:

| Gen | Average batch | Backend positions/s |
| ---: | ---: | ---: |
| 6 | 24.6 / 24.7 | 9,107 / 9,156 |
| 7 | 17.8 / 17.8 | 6,849 / 6,840 |
| 8 | 23.1 / 23.1 | 8,857 / 8,937 |
| 9 | 27.4 / 28.7 | 10,086 / 10,597 |
| 10 | 22.8 / 23.2 | 8,653 / 8,798 |
| 11 | 33.1 / 33.2 | 12,086 / 12,288 |
| 12 | 33.6 / 34.1 | 12,192 / 12,351 |
| 13 | 23.3 / 23.5 | 8,905 / 8,969 |
| 14 | 36.2 / 36.8 | 13,530 / 13,589 |
| 15 | 39.2 / 40.3 | 13,962 / 14,249 |

There is still no preserved v21 runtime baseline, so none of these figures
establishes a cross-version speedup or regression. Within v22, the large
self-play batches reach 26.8k–44.6k positions/s while smaller arena batches
reach 6.8k–14.2k positions/s per evaluator, consistent with the isolated
batch-size curve above.

### 4. Early-stopped experiment: v23 uses a smaller network

The `alpha-zero-draw-aware-v23` run changed only model capacity: 32
feature channels and two residual blocks replace v22's 64 channels and four
blocks. The 64-unit shared dense representation, encoder, objectives, seed,
game budgets, optimizer schedule, restart logic and promotion gates remain
unchanged. Model format version 4 prevents checkpoints with the old shape from
being mixed into the run.

This reduced the model from roughly 442,000 to 136,000 parameters. The run was
stopped after seven complete candidates because the expected performance gain
did not materialize and promotion progress lagged v22. Generation 8 was
interrupted during self-play and was not persisted. The accepted sequence is
`0 -> 1 -> 4 -> 7`.

| Gen | Source | Self-play draws | Valid loss P/V | Valid top-1 P/V | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0 | 2/256 | 2.519 / 0.944 | 26.7% / 56.5% | 187/13/0 | 0/4 | 1/64 | promoted (rollout) |
| 2 | 1 | 33/256 | 2.129 / 1.013 | 38.6% / 59.5% | 153/18/29 | 4/4 | 0/64 | draw rejection |
| 3 | 1 | 39/256 | 2.025 / 0.726 | 41.9% / 63.2% | 161/10/29 | 4/4 | 2/64 | draw rejection |
| 4 | 1 | 40/256 | 1.958 / 0.691 | 43.5% / 63.5% | 182/8/10 | 0/4 | 5/64 | promoted |
| 5 | 4 | 46/256 | 1.847 / 0.679 | 47.2% / 64.7% | 87/87/26 | 0/4 | 1/64 | strength rejection |
| 6 | 4 | 42/256 | 1.799 / 0.695 | 50.0% / 64.4% | 130/31/39 | 4/4 | 5/64 | draw rejection |
| 7 | 4 | 50/256 | 1.775 / 0.648 | 50.7% / 64.9% | 134/22/44 | 0/4 | 2/64 | promoted |

The isolated pre-training benchmark was repeated three times on 2026-08-14 on
the same Apple M4 Max, macOS 26.5.2 and Rust 1.97.1 setup as v22. Reducing the
parameter count does not produce a uniform inference speedup on the 3×4 board;
fixed Metal dispatch costs dominate most batch sizes:

| Batch | v22 median positions/s | v23 median positions/s | Change |
| ---: | ---: | ---: | ---: |
| 1 | 443 | 451 | +1.9% |
| 8 | 3,690 | 3,697 | +0.2% |
| 16 | 7,371 | 7,299 | −1.0% |
| 32 | 14,436 | 14,181 | −1.8% |
| 64 | 28,744 | 27,134 | −5.6% |
| 128 | 50,925 | 54,838 | +7.7% |

The full pipeline remains the meaningful performance test because self-play
creates dynamic batches while optimization exercises backward passes that this
inference-only benchmark does not measure.

That full-pipeline result is negative. Neural self-play for candidates 2–7
processed 8,876,042 positions at a backend-time-weighted 33,152 positions/s;
the same v22 candidate range achieved 35,396 positions/s. The smaller model is
therefore 6.3% slower in this sample despite having 69% fewer parameters.
Dynamic batch sizes and fixed Metal dispatch costs dominate the saved parameter
count. v23 also promoted three candidates through attempt 7 versus five for
v22, although its candidate-7 global WDL metrics were better.

The partial frozen diagnostic uses 155 validation games and 5,210 positions.
Only 20 validation games are draws, all of whose retained examples lie within
eight plies of the result, so it cannot answer the long-range draw question.
The smaller network is rejected as the next baseline: lower memory use alone
is not a significant project improvement when neither runtime nor learning
progress improves. The next run restores v22 capacity and targets the WDL
objective directly with an auxiliary scalar value loss.

### 5. Completed experiment: v24 adds an ordered value signal

The fresh `alpha-zero-draw-aware-v24` run restores v22's 64 feature channels
and four residual blocks. It changes one learning objective: the original WDL
cross-entropy is retained at full weight, while a `0.25`-weighted MSE term
compares `P(win) - P(loss)` with the scalar result `+1/0/-1`. This adds an
ordered value signal without making a certain draw equivalent to a balanced
win/loss prediction. Model format version 5 and fresh model/data paths isolate
the optimizer state and reports from earlier runs.

The 15-generation run completed in 1 h 18 min 51 s. `P/V/S` below means policy
loss, WDL loss and unweighted scalar MSE. Promotion still used all three v24
checks: arena score at least 55%, mirror `0/4`, and exploratory probe at most
`12/64` draws.

| Gen | Source | Self-play draws | Valid loss P/V/S | Valid top-1 P/V | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 2/256 | 2.588 / 1.117 / 1.240 | 25.1% / 57.8% | 175/25/0 | 0/4 | 1/64 | promoted |
| 2 | 1 | 30/256 | 2.133 / 1.053 / 1.058 | 38.4% / 61.1% | 176/19/5 | 0/4 | 0/64 | promoted |
| 3 | 2 | 44/256 | 2.005 / 1.096 / 1.136 | 42.5% / 57.8% | 127/71/2 | 0/4 | 1/64 | promoted |
| 4 | 3 | 48/256 | 1.898 / 0.966 / 0.982 | 45.2% / 60.3% | 148/48/4 | 0/4 | 4/64 | promoted |
| 5 | 4 | 56/256 | 1.854 / 1.016 / 1.013 | 46.7% / 59.7% | 119/79/2 | 0/4 | 2/64 | promoted |
| 6 | 5 | 53/256 | 1.781 / 0.899 / 0.961 | 48.8% / 61.0% | 94/63/43 | 4/4 | 0/64 | mirror rejection |
| 7 | 5 | 45/256 | 1.769 / 0.830 / 0.893 | 49.3% / 61.3% | 80/46/74 | 0/4 | 2/64 | promoted |
| 8 | 7 | 53/256 | 1.672 / 0.857 / 0.867 | 52.3% / 63.2% | 157/27/16 | 0/4 | 3/64 | promoted |
| 9 | 8 | 54/256 | 1.647 / 0.877 / 0.848 | 52.8% / 63.0% | 90/42/68 | 4/4 | 5/64 | mirror rejection |
| 10 | 8 | 53/256 | 1.622 / 0.832 / 0.826 | 53.9% / 64.2% | 110/32/58 | 4/4 | 7/64 | mirror rejection |
| 11 | 8 | 50/256 | 1.609 / 0.813 / 0.795 | 53.8% / 65.0% | 44/88/68 | 0/4 | 7/64 | strength rejection |
| 12 | 8 | 53/256 | 1.577 / 0.773 / 0.776 | 55.0% / 66.1% | 119/20/61 | 4/4 | 6/64 | mirror rejection |
| 13 | 8 | 58/256 | 1.555 / 0.766 / 0.745 | 56.6% / 66.6% | 65/22/113 | 4/4 | 4/64 | mirror rejection |
| 14 | 8 | 48/256 | 1.536 / 0.758 / 0.753 | 57.6% / 66.6% | 149/26/25 | 4/4 | 0/64 | mirror rejection |
| 15 | 8 | 61/256 | 1.527 / 0.765 / 0.752 | 58.0% / 66.7% | 118/22/60 | 4/4 | 10/64 | mirror rejection |

The accepted sequence was `0 → 1 → 2 → 3 → 4 → 5 → 7 → 8`. The auxiliary
objective accelerated early promotions and raised final validation WDL top-1
to 66.7%, versus 64.9% in v22. It did not produce a better final champion:
attempts 9–15 remained anchored to champion 8, six solely because of the
deterministic mirror check. Those six candidates all passed the strength arena
and produced only 0–10 draws in 64 noisy games. Candidate 14, for example,
beat champion 8 by `149/26/25` and then produced `0/64` exploratory draws, but
was rejected because identical deterministic copies drew four times.

The frozen endgame diagnostic also rules out a significant long-range value
improvement. Candidate 15 raises drawn top-1 at distance `9–16` from v22's
0.8% to 12.8%, but its draw loss worsens from 2.780 to 3.544; at `17+`, both
runs remain at 0.2% top-1 while loss worsens from 2.741 to 3.289. Accepted
champion 8 is weaker still on those distant draws. The scalar term improves
some global value metrics without solving the mature cyclic policy.

The architecture retained v22 inference performance. Across neural self-play
for generations 2–15, v24 evaluated 21,736,183 positions in 616.9 backend
seconds, or 35,235 positions/s weighted by positions. That is within 0.5% of
v22's 35,396 positions/s and confirms that the auxiliary term is effectively
training-only.

### 6. Active experiment: v25 makes mirror play diagnostic

The promotion rule had become more conservative than the algorithms it was
modelled after:

- [AlphaGo Zero](https://dsbrown1331.github.io/advanced-ai/readings/alphaGoZero.pdf)
  compared a checkpoint only against the current best player and promoted it
  above a 55% winning margin; it did not add candidate-versus-itself vetoes.
- [AlphaZero](https://www.davidsilver.uk/wp-content/uploads/2020/03/alphazero-science_compressed.pdf)
  removed checkpoint gating altogether, always generated self-play with the
  latest parameters, and trained drawn outcomes as value `0` while retaining
  the MCTS policy target.
- [KataGo's training loop](https://github.com/lightvector/KataGo/blob/master/SelfplayTraining.md)
  makes its gatekeeper optional and states that accepting every model is faster
  and works normally; gating is mainly useful early for debugging.

A draw is therefore not an empty sample: its value target is neutral, but every
position still teaches the searched policy. Repeated identical trajectories
are the real concern. AlphaZero's supplementary experiments report a chess
match with more than 90% draws and low diversity; sampling among near-equal
moves increased its win rate from 5.8% to 14%.

v25 changes only the promotion decision. The four deterministic mirror games
and their zero-draw diagnostic limit remain recorded, but no longer veto a
candidate. Promotion requires the diversified paired arena and at most 20%
draws in 64 games generated with the actual noisy self-play settings. Fresh
`alpha-zero-draw-aware-v25` paths isolate the new trajectory; network format,
v24's scalar loss, data budgets and all search settings remain unchanged.

### 7. Later ablations, in this order

Test these ideas one at a time:

1. **Adaptive rollout warm-start.** The current bootstrap switches abruptly
   from uniform-prior random-rollout MCTS to the full network after the first
   promotion. [Warm-Start AlphaZero](https://arxiv.org/abs/2004.12357) instead
   blends leaf values as `v = (1-w) v_network + w v_rollout`, reducing `w`
   linearly over the first iterations; its three small-board experiments found
   moderate but game-dependent gains. A [follow-up](https://arxiv.org/abs/2105.06136)
   switches adaptively when neural MCTS beats the rollout-assisted player. Test
   this only if v25 still needs a better bootstrap, because v24's observed
   bottleneck starts after its rapid early promotions.
2. **Deeper-state restarts.** [Go-Exploit](https://arxiv.org/abs/2302.12359)
   samples self-play starting states from an archive and improved sample
   efficiency in Connect Four and 9×9 Go. Generalize the current cycle-only
   restart archive if v25 needs more late-game diversity.
3. **Explicit recent moves.** Encode the origin and destination of the last two
   moves directly, then test whether some full historical board frames can be
   removed. Do not discard repetition context: the game remains responsible
   for exact occurrence counts.
4. **Stronger endgame curriculum.** Increase the early fraction of decisive
   terminal tails and expand from `1` to `2`, `4`, `8`, `16`, then all plies.
   Advance this schedule from the accepted source champion, not rejected
   candidate attempt numbers. The JavaScript experiments sometimes trained
   entirely on the last one or ten positions; the current Rust curriculum only
   guarantees a 25% tail fraction.

Any encoder or network shape change requires a checkpoint metadata/version
bump and round-trip tests. WDL targets remain official, drawn games remain in
the replay buffer, captured pieces remain unordered non-spatial counts, and the
same design must stay applicable to the planned 5×6 board.

## Rules source

The engine follows the official 3×4 rulebook:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
