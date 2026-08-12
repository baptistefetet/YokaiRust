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

## Latest completed research run: v20 through generation 15

This deterministic run started from random weights. Compared with v19, its only
change was the targeted-restart distance: a restart may begin one ply before an
observed repetition draw, instead of two to eight plies before it. At that exact
decision, the deeper 800-simulation search can explore the alternatives that
the corrected non-starter target preserves. This remains ordinary self-play
from an observed game prefix.

`source` is the last accepted champion; `arena` is candidate/champion/draw over
200 games; `probe` is exploratory draws out of 64. Promotion requires arena
≥55%, mirror 0/4 and probe ≤12/64.

| Gen | Source | Self-play draws | Train policy | Valid policy/WDL | Valid top-1 | Arena | Mirror | Probe | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 0 | 7/256 | 2.253 | 2.565 / 1.065 | 31.9% | 168/32/0 | 0/4 | 1/64 | promoted |
| 2 | 1 | 33/256 | 1.858 | 2.238 / 0.973 | 39.1% | 177/23/0 | 0/4 | 1/64 | promoted |
| 3 | 2 | 34/256 | 1.803 | 2.063 / 0.965 | 44.0% | 148/50/2 | 0/4 | 1/64 | promoted |
| 4 | 3 | 35/256 | 1.730 | 1.950 / 0.969 | 45.3% | 106/56/38 | 4/4 | 3/64 | draw rejection |
| 5 | 3 | 49/256 | 1.735 | 1.908 / 0.866 | 47.7% | 161/38/1 | 0/4 | 7/64 | promoted |
| 6 | 5 | 57/256 | 1.671 | 1.839 / 0.806 | 50.3% | 144/40/16 | 0/4 | 5/64 | promoted |
| 7 | 6 | 63/256 | 1.646 | 1.771 / 0.816 | 51.0% | 72/46/82 | 4/4 | 15/64 | draw rejection |
| 8 | 6 | 64/256 | 1.635 | 1.744 / 0.875 | 51.7% | 96/22/82 | 0/4 | 4/64 | promoted |
| 9 | 8 | 46/256 | 1.560 | 1.676 / 0.859 | 53.7% | 94/38/68 | 4/4 | 7/64 | draw rejection |
| 10 | 8 | 53/256 | 1.569 | 1.664 / 0.866 | 54.2% | 64/25/111 | 4/4 | 7/64 | draw rejection |
| 11 | 8 | 50/256 | 1.557 | 1.647 / 0.845 | 54.9% | 107/15/78 | 4/4 | 13/64 | draw rejection |
| 12 | 8 | 63/256 | 1.554 | 1.638 / 0.827 | 55.1% | 112/22/66 | 0/4 | 9/64 | promoted |
| 13 | 12 | 57/256 | 1.524 | 1.603 / 0.810 | 56.7% | 84/46/70 | 0/4 | 12/64 | promoted |
| 14 | 13 | 69/256 | 1.500 | 1.588 / 0.842 | 57.4% | 45/85/70 | 4/4 | 11/64 | strength + draw rejection |
| 15 | 13 | 63/256 | 1.504 | 1.586 / 0.815 | 57.2% | 117/43/40 | 4/4 | 1/64 | draw rejection |

Across 3,840 games and 127,717 positions, validation policy loss fell 38.2%,
policy top-1 rose from 31.9% to 57.2%, WDL top-1 rose from 58.1% to 63.3%,
and illegal probability mass fell from 46.9% to 9.2%. Eight candidates were
promoted. The accepted sequence is `0 -> 1 -> 2 -> 3 -> 5 -> 6 -> 8 -> 12 ->
13`. Compared with v19, the one-ply restart therefore produced one more
promotion and moved the final champion from 11 to 13. In particular, candidates
12 and 13 both passed every gate after three consecutive rejections from
champion 8.

The change delays the cyclic plateau but does not remove it. Candidate 14 is
weaker than champion 13 and cycles in all four mirrors. Candidate 15 lowers
validation policy loss again and beats champion 13 by 117 wins to 43, with
only 40 arena draws and one exploratory draw, yet all four deterministic
mirrors still cycle. The final buffer's drawn non-starter immediate-draw mass
is 55.2%, versus 53.8% in v19, so the exact-decision restarts do not produce a
consistent reduction in that local metric. The mirror gate again prevents a
superficially strong but cyclic network from becoming the self-play teacher.

The next action is the endgame-distance diagnostic below. Run it on the frozen
v20 checkpoints before implementing or starting v21; training behavior must
remain unchanged until that comparison exists.

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

### 0. Frozen v20 baseline

The one-ply-restart run is complete through candidate 15 and no trainer is
writing its directories. Its accepted champion is generation 13, and
`reports/generation-000015.json` records the final rejected candidate. Runtime
artifacts remain in `data/alpha-zero-draw-aware-v20` and
`models/alpha-zero-draw-aware-v20`; they are intentionally ignored by Git and
must be retained until every v21 comparison is complete.

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
