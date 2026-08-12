# AlphaZero in YokaiRust

This guide explains what the network learns, how one generation works and why
repetition draws need special handling. The implementation is self-contained:
it learns from game rules, MCTS and self-play, and the same design is intended
for the future 5×6 game.

## Vocabulary used in this project

These terms occur in logs, configuration and source names:

| Term | Plain meaning |
| --- | --- |
| **WDL** | **Win / Draw / Loss**: victory, draw or defeat. |
| **policy** | The network's preference over actions: “what should I try?” |
| **value** | Its estimate of the eventual result: “how good is this position?” |
| **logit** | An unrestricted raw network output. `softmax` turns several logits into probabilities summing to one. |
| **softmax** | Conversion from arbitrary logits to positive probabilities whose sum is one. |
| **target / label** | The answer used for training: MCTS visits for policy, final WDL result for value. |
| **loss** | A numeric error to minimize. Lower means closer to targets, not automatically stronger play. |
| **cross-entropy** | The loss comparing a predicted probability distribution with a target distribution. |
| **MCTS** | Monte Carlo Tree Search: explore a tree, evaluate leaves and propagate values upward. |
| **PUCT** | The MCTS selection formula combining value, visit count and policy prior. |
| **prior** | Policy probability attached to an action before deep search. |
| **ply** | One action by one player. |
| **self-play** | Games where both sides use the private learner and MCTS. |
| **checkpoint** | Saved network parameters, metadata and Adam optimizer state. |
| **champion** | Accepted checkpoint used as the published player and arena reference. |
| **learner** | Private checkpoint being optimized and used for self-play; it may temporarily cycle. |
| **arena** | Noise-free match measuring a candidate against the champion. |
| **epoch** | One complete dataset pass. YokaiRust uses a fixed count of sampled mini-batches instead. |
| **batch** | Several positions processed together to use the GPU efficiently. |
| **encoder** | Deterministic conversion from a `Game` into numeric planes understood by the network. |
| **convolution** | A small learned filter reused over every board location to detect local patterns. |
| **residual block** | Two convolutions plus a shortcut adding the input back, which helps optimization. |
| **Adam** | The optimizer updating parameters from gradients while remembering moving averages of past gradients. |
| **Dirichlet noise** | Random probability mass mixed into root priors during self-play so different actions are explored. |
| **training set** | Examples used to update parameters. |
| **validation set** | Held-out examples used only to measure generalization, never to update parameters. |

In formulas and field names, `W`, `D` and `L` are the WDL probabilities. The
neutral value `W - L` approaches `+1` for a win, `-1` for a loss and `0` for a
draw.

## Network targets

For every non-terminal position, the network predicts:

- **policy**: 132 action logits;
- **WDL**: win, draw and loss logits from the current player's perspective.

The input contains the current state and seven preceding states. Each state
contributes 16 piece/hand planes; two final planes encode current repetition
count and whether the player to move started the game. The policy branch also
receives, for each action slot, the occurrence count of the resulting position
and an immediate-threefold flag. MCTS computes this context from the complete
`Game` history.

The policy target is the normalized MCTS visit distribution, not merely the
action eventually played. Temperature affects action sampling but never
sharpens the stored target.

The WDL target is the official final result:

- win if the recorded player to move eventually won;
- loss if that player lost;
- draw after an official repetition.

Official search converts WDL to `P(win) - P(loss)`. A certain draw is therefore
different from an uncertain 50/50 win/loss mixture. Terminal states have no
policy example because they have no legal move distribution.

## Why the network has one trunk and two heads

[`src/neural/model.rs`](../src/neural/model.rs) has three conceptual parts:

```text
encoded game [batch, 130, 4, 3]
                 |
       shared residual trunk
          /              \
 policy head          WDL head
 132 logits            3 logits
```

The shared convolutional trunk learns board patterns useful to both questions.
The policy head maps those patterns plus per-action repetition context to 132
action slots. The WDL head maps the same patterns to three outcome logits.
Separate heads keep “which move?” and “who wins?” distinct while sharing most
of the computation.

The default model is deliberately small: 64 feature channels and four residual
blocks. A 3×4 board does not justify a large image model, and small generations
remain affordable on one workstation. These sizes are configuration, not game
rules, so the future 5×6 version can scale them.

### Canonical orientation

The engine stores one absolute board orientation, but the encoder rotates the
view so the current player's forward direction is always the same. The network
learns one concept of “my piece moves forward” instead of duplicating it for
First and Second. `Position` stays absolute so rules, replay and UI do not
inherit neural conventions.

### History and repetition context

The board alone cannot tell how often it occurred. Eight frames provide recent
motion; the current repetition plane and per-action occurrence counts expose
the immediate rule consequence. `Game` still owns the complete history and is
the sole authority for declaring a draw. Neural input guides search; it never
replaces the rules engine.

### Fixed 132-action output

A fixed output shape lets one dense layer and checkpoint format cover every
position. Illegal actions are masked and legal probabilities renormalized
before MCTS uses them. This batches more simply than a variable-length vector,
while `PolicyIndex` keeps the action mapping typed and tested.

## One generation

```text
private learner --noisy self-play + cycle restarts
       |
       v
rolling replay buffer --stable game split--> validation
       |
       v
resume private learner + Adam, then optimize
       |
       v
candidate --versus champion + draw gates--> publish champion
       |
       +-- strong but cycling: keep private learner
       +-- strength regression: roll learner back to champion
```

The learner is both the optimization lineage and the self-play source. This is
essential: if self-play stayed attached to an old champion, later candidates
would only get better at imitating old searches. A strong but cycling candidate
therefore generates the next batch without becoming the published player. The
draw-aware targets and cycle restarts are responsible for helping it escape. A
genuine strength regression abandons that branch and resets the learner to the
champion.

Each generation uses a fixed optimizer-step budget. Buffer growth therefore
does not silently increase training work. Checkpoints preserve both model
parameters and Adam moments.

Whole games, rather than individual positions, are assigned to validation. A
stable hash of `(generation, seed)` keeps old games on the same side when the
buffer grows. This prevents leakage and makes losses more comparable over time.

### Follow one generation in the source

| Step | Main code | Read first |
| --- | --- | --- |
| Encode a position | [`neural.rs`](../src/neural.rs) | `encode_game`, then `encode_position_with_history` |
| Run the network | [`neural/model.rs`](../src/neural/model.rs) | `AlphaZeroNetworkConfig::init`, then `forward` |
| Get predictions | [`neural/evaluator.rs`](../src/neural/evaluator.rs) | `Evaluator for NetworkEvaluator` |
| Batch requests | [`neural/service.rs`](../src/neural/service.rs) | `InferenceClient`, then `InferenceService` |
| Search a move | [`search.rs`](../src/search.rs) | `Mcts::search_internal` and `Node` |
| Record targets | [`training/data.rs`](../src/training/data.rs) | `SelfPlayRecorder::record`, then `finish_from_game` |
| Compute losses | [`training/trainer.rs`](../src/training/trainer.rs) | `BatchTensors::new`, `forward_losses`, `policy_loss_weight` |
| Orchestrate | [`training/pipeline.rs`](../src/training/pipeline.rs) | `run_generation_with_progress` |

Burn's `Tensor<B, 4>` means a rank-four tensor on backend `B`, close to a C++
template such as `Tensor<Backend, 4>`. `B: Backend` is a compile-time capability
constraint. `Autodiff<B>` adds gradient tracking for training; its inner backend
performs cheaper inference during self-play and arenas.

## Architecture decisions at a glance

| Choice | Why it exists | Cost or limitation |
| --- | --- | --- |
| Burn backend generic | The same model code runs on CPU tests and Metal training. | Generic tensor errors are initially more verbose than concrete C++ types. |
| One GPU inference service | Many games share large batches instead of each owning a model. | Workers wait briefly for batching and communicate through channels. |
| Contiguous MCTS node `Vec` | Indices are stable, allocation is cheap and traversal is cache-friendly. | Removing arbitrary nodes is less convenient than pointer-owned trees. |
| Rolling replay buffer | Mixes recent generations so one noisy batch does not replace all knowledge. | Targets are generated by networks of different ages. |
| Fixed optimizer steps | Per-generation training cost stays constant as the buffer grows. | Not every example is visited in every generation. |
| Separate champion and learner | Learning and self-play can advance across a temporary cycle without publishing that model. | The private data generator may be imperfect, so draw-aware training is necessary. |
| Whole-game validation | Positions from one outcome never leak across train and validation. | Game lengths make the exact position fraction vary slightly. |
| Structured progress enum | CLI today and Ratatui later consume the same typed events. | Adding an event requires updating every exhaustive `match`. |
| Atomic files at boundaries | Interruption leaves the previous complete state resumable. | Temporary files briefly require additional disk space. |

These choices favor explicit ownership and testable boundaries over framework
magic. `pipeline.rs` is intentionally orchestration code: neural math belongs in
`trainer.rs`, game generation in `self_play.rs`, comparison in `arena.rs`, and
persistence close to the data it protects.

## Why repetitions form a feedback loop

With limited search, a known short draw can look safer than a long, uncertain
winning line. MCTS then gives the repetition a sharp visit target. Cross-entropy
teaches that target back to the network, making the same cycle easier to select
in the next generation.

Deleting drawn games is not a solution. Their positions contain the strongest
defence found by the starter and the exact conversion failures the non-starter
must learn to overcome. Relabelling them would also corrupt the official value
target.

YokaiRust instead separates four responsibilities.

### 1. WDL keeps the outcome observable

The value head learns draw probability explicitly. Stored targets and official
arenas remain neutral.

### 2. Self-play gives the two roles different incentives

During self-play only, the starter evaluates a draw at `+0.25` and the
non-starter at `-0.25`:

```text
starter:     win - loss + 0.25 * draw
non-starter: win - loss - 0.25 * draw
```

Both still rank win above draw above loss. The starter learns its best defence;
the non-starter is pushed to search for a conversion.

### 3. Restarts spend search near the failure

One quarter of trajectories restart two to eight plies before a historical
threefold repetition. The full prefix is replayed, so exact repetition state,
starter role and encoder history remain valid. The local temperature schedule
restarts at zero to explore alternatives near the cycle.

### 4. Policy weighting stops rewarding failed conversion

Every example still trains WDL at full weight. The starter's moves in a draw
also remain normal policy examples because they describe a successful defence.
For the non-starter, however, the final draw proves that the complete conversion
attempt failed. Its policy weight is therefore zero for every position in that
drawn trajectory, including the earlier choices that led toward the cycle.

This does not invent a result or a better action. It separates two statements
already present in self-play: “this game ended in a draw” remains a full WDL
target, while “imitate the failed converter's search distribution” is omitted.
Policy for the non-starter is still learned from every decisive game.

## Promotion measurements

A candidate must pass three independent checks:

1. score at least 55% against the champion in 200 paired games;
2. produce no draw in four deterministic initial-position mirror games;
3. stay at or below 20% draws in 64 noisy self-play games.

Each arena pair shares a random legal 0–4 ply opening and swaps candidate color.
This avoids counting one deterministic trajectory hundreds of times. Mirror
games stay on the official initial positions because they answer a different
question: does the candidate deterministically enter a repetition cycle?

The gates protect publication; they are not training labels.

## Reading the metrics

| Metric | What it measures | Warning sign |
| --- | --- | --- |
| policy loss | Weighted cross-entropy against MCTS visits | Training falls while stable validation rises. |
| policy weight | Mean policy-loss multiplier after drawn non-starter examples are omitted | A drop means draws occupy more of the buffer. |
| WDL loss | Cross-entropy against final W/D/L | Persistent high validation loss means weak result prediction. |
| policy top-1 | Agreement with the largest visit target | Useful learning signal, not a strength score. |
| WDL top-1 | Agreement with observed result class | Can hide poor draw calibration if read alone. |
| draw error | Error in predicted draw probability | Rising values expose draw miscalibration. |
| illegal mass | Raw probability wasted on illegal actions | Should fall as representation learning improves. |
| entropy | Spread of predicted actions | Sudden collapse can indicate premature certainty. |
| repetition mass | Visit mass on already seen positions | Compare draw starter and non-starter buckets. |
| W/L/D and arena | Actual behavior | Primary check that lower loss becomes better play. |

Losses are moving-target measurements because every new network changes the
MCTS policy used to supervise the next one. A healthy run should show a downward
long-term validation trend, rising top-1, falling illegal mass, repeated arena
promotions and bounded draw probes. Any one of these alone is insufficient.

Arena progress is completion-ordered: short games can finish in a visible block
before long games. Only the final paired result and its per-seat split should be
interpreted.

## Does continued self-play guarantee perfect play?

No. Self-play can plateau, forget, cycle or overfit its recent opponents. More
compute creates more samples from the current learning process; it does not turn
that process into a proof.

A convincing strong model should repeatedly show:

- improving stable validation metrics;
- positive paired results against current and frozen historical champions;
- controlled deterministic and exploratory draw rates;
- balanced results across absolute colors and starter roles;
- reproducibility across several random seeds.

The project goal is a genuinely from-scratch AlphaZero learner. Its success is
measured by internal learning and playing strength, with no external component
in data generation, targets, search, promotion or runtime play.

## Can Ratatui development start now?

Yes. UI code should consume engine-level state and actions, not Burn tensors.
Use a fixed checkpoint or `UniformEvaluator` for reproducible interface tests;
training can advance independently.
