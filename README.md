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
   states and restart 25% one to eight plies before observed repetition cycles.
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

- policy and WDL cross-entropy;
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

## Latest completed research run: v19 through generation 15

This deterministic run started from random weights. Compared with v18, its only
change was how a drawn non-starter example is supervised. Normally its policy
loss has weight zero: copying every decision from a failed conversion would
teach the failure. When the rules prove that one available action immediately
causes the third repetition, v19 instead removes that action from the recorded
MCTS distribution, renormalizes the alternatives and trains on the corrected
target. The relative preference among the remaining moves still comes from
MCTS.

`source` is the last accepted champion; `arena` is candidate/champion/draw over
200 games; `probe` is exploratory draws out of 64. Promotion requires arena
≥55%, mirror 0/4 and probe ≤12/64.

| Gen | Source | Self-play draws | Train policy | Valid policy/WDL | Valid top-1 | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0 | 7/256 | 2.253 | 2.565 / 1.065 | 31.9% | 168/32/0 | 0/4 | 1/64 | promoted |
| 2 | 1 | 25/256 | 1.870 | 2.277 / 1.241 | 37.4% | 152/48/0 | 0/4 | 3/64 | promoted |
| 3 | 2 | 39/256 | 1.795 | 2.089 / 0.970 | 41.9% | 75/113/12 | 0/4 | 2/64 | strength rejection |
| 4 | 2 | 47/256 | 1.826 | 2.023 / 0.891 | 44.9% | 136/62/2 | 0/4 | 4/64 | promoted |
| 5 | 4 | 40/256 | 1.753 | 1.932 / 0.847 | 48.6% | 147/49/4 | 0/4 | 3/64 | promoted |
| 6 | 5 | 53/256 | 1.685 | 1.864 / 0.880 | 48.6% | 120/42/38 | 4/4 | 9/64 | draw rejection |
| 7 | 5 | 51/256 | 1.688 | 1.837 / 0.768 | 49.5% | 160/25/15 | 0/4 | 2/64 | promoted |
| 8 | 7 | 56/256 | 1.609 | 1.746 / 0.809 | 53.2% | 82/69/49 | 4/4 | 8/64 | strength + draw rejection |
| 9 | 7 | 54/256 | 1.621 | 1.733 / 0.765 | 53.1% | 65/42/93 | 0/4 | 4/64 | promoted |
| 10 | 9 | 63/256 | 1.588 | 1.710 / 0.774 | 54.3% | 56/60/84 | 4/4 | 6/64 | strength + draw rejection |
| 11 | 9 | 66/256 | 1.581 | 1.696 / 0.755 | 55.2% | 92/17/91 | 0/4 | 9/64 | promoted |
| 12 | 11 | 56/256 | 1.568 | 1.670 / 0.760 | 56.1% | 132/32/36 | 4/4 | 17/64 | draw rejection |
| 13 | 11 | 68/256 | 1.573 | 1.661 / 0.752 | 55.9% | 66/40/94 | 4/4 | 13/64 | draw rejection |
| 14 | 11 | 80/256 | 1.569 | 1.647 / 0.753 | 56.5% | 61/72/67 | 0/4 | 7/64 | strength rejection |
| 15 | 11 | 65/256 | 1.573 | 1.637 / 0.746 | 57.5% | 90/23/87 | 4/4 | 9/64 | draw rejection |

Across 3,840 games and 125,670 positions, validation policy loss fell 36.2%,
top-1 rose from 31.9% to 57.5%, and illegal probability mass fell from 46.9%
to 10.3%. Seven candidates were promoted, reaching champion 11. That is one
more promotion than v18, and the raw immediate-draw mass of drawn non-starter
positions fell from v18's 63.8% to 53.8% in the final buffer. The correction
therefore has the intended local effect.

It does not solve the complete problem. Generation 15 predicts the accumulated
dataset better than every earlier candidate and beats champion 11 by 90 wins to
23, yet 87 arena games are draws and all four deterministic mirrors cycle. This
is a concrete counterexample to "the loss goes down, therefore the player gets
better": the network can increasingly imitate a replay buffer whose difficult
positions still lack successful conversion trajectories. The mirror gate is
what prevents this superficially strong but cyclic network from becoming the
new self-play teacher.

The active v20 experiment lets targeted restarts begin one ply before a known
draw, instead of two to eight plies before it. At that exact decision, the
deeper 800-simulation search can explore the alternatives that the corrected
target now preserves. This remains ordinary self-play from an observed game
prefix. Complete this run through generation 15 before changing another
variable.

## Handoff: next JavaScript-informed research program

This section is the authoritative starting point for the next development
conversation. Preserve one changed variable per run: the objective is to learn
which JavaScript idea helps, not merely to produce one incomparable mixture.

### 0. Finish and freeze the v20 baseline

The active v20 run is the one-ply-restart experiment described above. Its
runtime artifacts are in `data/alpha-zero-draw-aware-v20` and
`models/alpha-zero-draw-aware-v20`; they are intentionally ignored by Git.
Before implementing v21:

1. ensure that `reports/generation-000015.json` exists for v20 and that no
   trainer is still writing those directories;
2. if interrupted, resume the existing paths instead of starting a second v20;
3. replace the v19 table above with the complete v20 generation 1–15 table and
   conclusions;
4. retain the v20 data and checkpoints until every v21 comparison is complete.

The structured JSON reports are authoritative if prose and runtime files ever
disagree.

### 1. Add an endgame-distance diagnostic

Before changing training behavior, report validation policy and WDL metrics by
distance from the terminal result: `1`, `2–4`, `5–8`, `9–16`, and `17+` plies.
Keep whole games on one side of the existing stable train/validation split.
This diagnostic must not alter sampling or gradients.

It answers whether the network still predicts endgames well but loses accuracy
with a longer horizon. Run it on saved v20 checkpoints so v21 has a frozen
comparison. Report at least example count, policy loss/top-1 and WDL loss/top-1
for each bucket; keep drawn and decisive positions distinguishable.

### 2. v21 changes only the generation-zero bootstrap

The first new training run must reproduce the useful bootstrap behavior of the
JavaScript project. Until the first candidate is accepted, self-play MCTS uses
uniform legal-action priors and random rollouts instead of generation zero's
random neural policy and WDL. A rejection leaves champion generation zero in
place, so the following attempt must also use rollouts. Immediately after the
first promotion, all self-play returns to the ordinary neural evaluator.

Implement this as an explicit, validated configuration option, not as a hidden
generation-number special case. The activation condition is the accepted
source champion being generation zero, rather than the candidate attempt being
generation one. Give v21 fresh paths such as
`models/alpha-zero-draw-aware-v21` and `data/alpha-zero-draw-aware-v21`; never
reuse the v20 buffer or weights.

The intended configuration shape is explicit enough to survive resume:

```toml
[self_play.bootstrap]
mode = "random_rollout_until_first_promotion"
rollout_max_plies = 512
```

The rollout contract is:

- start from the complete leaf `Game`, including its repetition history;
- choose uniformly among legal actions with the search's seeded RNG;
- stop on an official terminal outcome or the configured safety limit;
- return `+1`, `0`, or `-1` from the leaf player-to-move's perspective, with a
  limit reached while ongoing treated as zero;
- expand the leaf with uniform priors and do not query the inference service;
- omit root Dirichlet noise during this bootstrap, matching the JavaScript
  behavior; random rollouts and the opening temperature provide exploration;
- store the usual normalized MCTS visits as policy targets and the final
  self-play result as the official WDL target;
- keep candidate training, arena, mirror gate and exploratory probe neural.

`UniformEvaluator` is not an implementation of this algorithm: it always
returns value zero and has no complete `Game` from which to play. Put the
rollout path where MCTS still owns the reconstructed leaf game, or refactor the
leaf-evaluation boundary without sending full games through the GPU inference
service. Preserve deterministic seed ordering and official repetition rules.

Required tests cover deterministic rollouts for a fixed seed, win/loss sign
from the leaf perspective, safety-limit draws, rollout use after a rejected
generation-zero candidate, and the switch to neural evaluation after the first
promotion. Progress and reports must state which bootstrap evaluator generated
self-play, so a saved dataset is not ambiguous.

Everything else remains identical to v20: seed 42, network and encoder,
self-play and restart budgets, replay sampling, optimizer and learning-rate
schedule, promotion threshold and both draw gates. Run 15 candidate generations
from scratch. Compare the complete champion sequence, policy/WDL losses and
top-1 metrics, endgame-distance buckets, illegal mass, initial/restarted draws,
arena W/D/L, deterministic mirror draws and exploratory draws. A lower loss
alone is not success; the main question is whether the first targets are more
useful and the late cyclic plateau is delayed or avoided.

Before the full run, execute `cargo fmt --check`, `cargo test`, and strict
Clippy. Then build once with `cargo build --release` and launch the checked-in
configuration with:

```bash
target/release/yokai train --config config/training.toml \
  --generations 15 --headless
```

### 3. Later ablations, in this order

Only after v21 is documented should subsequent fresh runs test these ideas one
at a time:

1. **Scalar hand branch.** A hand is an unordered count per droppable piece
   type and player. Keep board history spatial, but move the existing normalized
   hand counts, repetition count and starter flag out of constant board planes.
   Concatenate these global scalars after the convolutions, as in the JavaScript
   network, and feed the resulting shared representation to both heads. Preserve
   the information from all existing history frames for this first comparison.
2. **Smaller network.** Compare the current 64 filters and four residual blocks
   with 32 filters and two blocks. The JavaScript 3×4 model had roughly 77,000
   parameters versus roughly 410,000 currently; a smaller Rust model may learn
   more reliably from the available number of correlated positions.
3. **Explicit recent moves.** Encode the origin and destination of the last two
   moves directly, then test whether some full historical board frames can be
   removed. Do not discard repetition context: the game remains responsible
   for exact occurrence counts.
4. **Auxiliary scalar value loss.** Retain the three-class WDL head, but add a
   small MSE term on `P(win) - P(loss)` against `+1/0/-1`. This imports the
   ordered scalar signal that worked well in JavaScript without making a
   certain draw indistinguishable from a 50/50 win/loss prediction.
5. **Stronger endgame curriculum.** Increase the early fraction of decisive
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
