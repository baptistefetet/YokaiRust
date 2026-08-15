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
| **self-play** | Games where both sides use the current champion and MCTS. |
| **checkpoint** | Saved network parameters, metadata and Adam optimizer state. |
| **champion** | Accepted checkpoint used as the published player and arena reference. |
| **arena** | Noise-free match measuring a candidate against the champion. |
| **epoch** | One complete dataset pass. YokaiRust uses a fixed count of sampled mini-batches instead. |
| **batch** | Several positions processed together to use the GPU efficiently. |
| **encoder** | Deterministic conversion from a `Game` into numeric planes understood by the network. |
| **convolution** | A small learned filter reused over every board location to detect local patterns. |
| **residual block** | Two convolutions plus a shortcut adding the input back, which helps optimization. |
| **BatchNorm** | Normalizes each channel's activations for stable training; its scale and shift replace convolution biases. |
| **Adam** | The optimizer updating parameters from gradients while remembering moving averages of past gradients. |
| **learning rate** | Size of each optimizer update. Too large can damage a mature policy; too small slows learning. |
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
contributes ten spatial piece planes and six scalar hand counts. Two final
scalars encode current repetition count and whether the player to move started
the game. The policy branch also receives, for each action slot, the occurrence
count of the resulting position and an immediate-threefold flag. MCTS computes
this context from the complete `Game` history.

The policy target is the normalized MCTS visit distribution, not merely the
action eventually played. Temperature affects action sampling but never
sharpens the stored target.

The WDL target is the official final result:

- win if the recorded player to move eventually won;
- loss if that player lost;
- draw after an official repetition.

Training retains categorical WDL cross-entropy and adds a small auxiliary loss:
`0.25 × (P(win) - P(loss) - result)²`, where the result is `+1`, `0` or `-1`.
The auxiliary term supplies an ordered value signal; WDL cross-entropy still
forces a certain draw to differ from balanced win/loss uncertainty.

Official search converts WDL to `P(win) - P(loss)`. A certain draw is therefore
different from an uncertain 50/50 win/loss mixture. Terminal states have no
policy example because they have no legal move distribution.

### How generation zero is bootstrapped

The current Rust pipeline saves a randomly initialized network as champion
generation zero. Generation one's MCTS therefore already uses its random policy
priors and random WDL estimates. Training can overcome those initial biases,
but the first policy targets partly reflect them.

The earlier JavaScript implementation used a different bootstrap. It disabled
the network until the first promotion: legal actions began with uniform priors,
and a random rollout from each leaf supplied the search value. After this
rules-only search produced the first dataset and an accepted network, later
generations switched to neural policy and value evaluation. If the first model
was rejected, rollout self-play remained active until one was accepted.

These are two distinct from-scratch strategies. The Rust approach is closer to
[AlphaGo Zero](https://deepmind.google/blog/alphago-zero-starting-from-scratch/),
whose search used a randomly initialized network and deliberately omitted
rollouts. The JavaScript approach provides a less arbitrary but noisier
bootstrap target. Comparing them while holding every later generation constant
is a useful experiment.

## Where the input numbers come from

Every constant in the network signature derives from a game rule rather than
from an arbitrary choice:

| Constant | Value | Derivation |
| --- | --- | --- |
| Board squares | 12 | 3 columns × 4 rows. |
| Piece planes per frame | 10 | 5 piece types × 2 owners: one binary sheet per (owner, type) pair. |
| History frames | 8 | The current position plus the seven preceding positions. |
| Spatial input planes | 80 | 10 planes × 8 frames. |
| Hand features per frame | 6 | Normalized count of each of the three droppable piece types, for both players. |
| Global features | 50 | 6 per frame × 8 frames, plus the repetition count and the starter-role flag. |
| Move slots | 96 | Any of the 12 squares can move in any of the 8 directions. |
| Drop slots | 36 | Any of the 12 squares can receive any of the 3 hand piece types. |
| Policy slots | 132 | 96 moves + 36 drops. |
| Policy context features | 264 | Per action slot: occurrence count of the resulting position and an immediate threefold-repetition flag. |

The spatial input is a stack of 80 binary sheets over the 4×3 board. Each
history frame contributes ten sheets, one per owner and piece type; a sheet
holds `1` on every square occupied by that piece. Summing the ten sheets of
frame zero recovers the current placement, which makes a single frame easy to
verify by hand.

The 50 global values carry everything without a board coordinate: the hand
composition of both players for every frame (48 values), the current repetition
count capped at three, and whether the player to move started the game.

The 264 policy context values are action-specific, which is why they bypass the
shared trunk and enter only the policy head: “how often has this move's
resulting position occurred” and “does this move complete the third repetition”
are properties of a candidate action, not of the position alone. The occurrence
count is capped at three and normalized to `[0, 1]`; the flag is `1` exactly
when the move would end the game by repetition.

## Why the network has one trunk and two heads

[`src/neural/model.rs`](../src/neural/model.rs) has three conceptual parts:

```text
board history [batch, 80, 4, 3]    globals [batch, 50]
                |                         |
       shared residual tower              |
                |                         |
                +------ concatenate ------+
                            |
                 shared dense representation
                     /                 \
            policy head              WDL head
             132 logits                3 logits
```

The shared convolutional tower learns board patterns useful to both questions.
After the convolutions, its flattened output is concatenated with the global
features and mixed by a shared dense layer. The policy head maps that
representation plus per-action repetition context to 132 action slots. The WDL
head maps the same shared representation to three outcome logits. Separate
heads keep “which move?” and “who wins?” distinct while sharing the state
representation.

### Layer by layer

1. **Input convolution.** A 3×3 kernel slides over the 80 input planes and
   produces 64 channels of unchanged spatial size (padding `Same`, no bias).
   Each channel is a learned local detector; after training, individual channels
   respond to motifs such as “my piece adjacent to an enemy piece” or “empty
   corner”. Batch normalization and ReLU follow.
2. **Residual tower.** Four blocks are stacked. Each block applies two 3×3
   convolutions, with normalization and ReLU between them, then adds the block
   input back to its own output before a final ReLU: `output = input + f(input)`.
   The shortcut carries the original features through all four blocks, which is
   what keeps deep stacks trainable (the ResNet idea).
3. **Flatten and fuse.** The tower output is 64 channels × 12 squares = 768
   values. Concatenating the 50 global features gives an 818-value vector, which
   a dense layer compresses to the shared 64-unit representation.
4. **Policy head.** The shared representation is concatenated with the 264
   per-action context values (328 total) and mapped to 132 action logits.
5. **WDL head.** The shared representation alone passes through a 64-unit hidden
   layer and is mapped to three outcome logits.

All convolutions are bias-free; the affine scaling lives in the BatchNorm
layers, which is the standard ResNet convention. The heads emit raw logits;
callers apply softmax before treating them as probabilities, and search converts
WDL logits to `P(win) - P(loss)` (see [Network targets](#network-targets)).

With the default configuration, roughly two thirds of the 442,000 parameters sit
in the residual tower. The size is deliberately small because MCTS queries the
network thousands of times per move; every width remains a checkpointed setting
in `AlphaZeroNetworkConfig`, so the future 5×6 game can scale it.

### Canonical orientation

The engine stores one absolute board orientation, but the encoder rotates the
view so the current player's forward direction is always the same. The network
learns one concept of “my piece moves forward” instead of duplicating it for
First and Second. `Position` stays absolute so rules, replay and UI do not
inherit neural conventions.

### Hands are unordered counts

A captured piece in hand has no board coordinate, and the order in which pieces
were captured carries no information. The encoder therefore keeps only one
normalized count per droppable piece type for each player and history frame.
These counts stay in a 48-value scalar vector, outside the convolutions. Current
repetition and starter role add two more global features. The complete scalar
vector is concatenated with the spatial board features after the residual
tower, matching the separation used by the earlier JavaScript network.

### History and repetition context

The board alone cannot tell how often it occurred. Eight frames provide recent
motion; the current repetition scalar and per-action occurrence counts expose
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
accepted champion --noisy self-play + visited-state restarts
       |
       v
rolling replay buffer --stable game split--> validation
       |
       v
resume champion + Adam, then optimize
       |
       v
candidate --versus champion + exploratory draw gate--> publish champion
       |
       +-- failed strength/productivity check: keep champion as data source
```

One accepted checkpoint is deliberately used for all three roles: self-play
source, optimization source and arena reference. The next attempt still differs
after a rejection because the replay buffer grew and its random seed changed.
Promotion protects playing strength and actual noisy self-play productivity;
deterministic candidate-versus-itself cycles are retained as a diagnostic, not
as a veto.

Each generation uses a fixed optimizer-step budget. Buffer growth therefore
does not silently increase training work. Checkpoints preserve both model
parameters and Adam moments. The learning rate starts high enough for bootstrap
and can fall at accepted-champion milestones. Rejected attempt numbers do not
advance that schedule: training maturity is measured by published progress.

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
| Compute losses | [`training/trainer.rs`](../src/training/trainer.rs) | `BatchTensors::new`, `forward_losses`, `scalar_value_loss_weight` |
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
| Fixed optimizer steps + staged learning rate | Cost stays constant while later updates can become gentler. | Not every example is visited in every generation. |
| One accepted champion | Rejected candidates cannot generate the next dataset. | Escaping a plateau may take several attempts from the same weights. |
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

During self-play only, the starter evaluates a draw at `+0.75` and the
non-starter at `-0.75`:

```text
starter:     win - loss + 0.75 * draw
non-starter: win - loss - 0.75 * draw
```

Both still rank win above draw above loss. The starter learns its best defence;
the non-starter is pushed to search for a conversion.

### 3. Restarts explore recent visited states

One quarter of trajectories restart from a uniformly sampled non-initial,
nonterminal state visited in the recent replay buffer. Both decisive and drawn
games contribute states, and repeated occurrences remain in the archive so
frequently visited regions receive proportionally more restarts. The complete
prefix is replayed, preserving exact repetition counts, starter role and encoder
history.

These trajectories use 800 MCTS simulations per move instead of the regular
200. Their local temperature schedule restarts at ply zero, adding exploration
throughout the game tree and producing shorter, more independent value targets.
The current network, rules and PUCT search still produce every target.

### 4. Policy supervision rejects a known repeating action

Every example still trains WDL at full weight. The starter's moves in a draw
also remain normal policy examples because they describe a successful defence.
For the non-starter, however, the final draw proves that the complete conversion
attempt failed. Its ordinary policy weight is therefore zero for positions in
that drawn trajectory. There is one useful exception: if the generic rules say
that an available action would immediately create the third repetition, that
action is removed from the recorded MCTS distribution and the remaining visit
probabilities are renormalized to sum to one.

This does not claim that one remaining action wins. It makes only the observation
provided by the rules and the completed game: this exact action draws, which is
an unwanted result for the converter. MCTS still decides the relative preference
among every alternative. If it visited none, policy supervision stays omitted.
The official draw remains a full WDL target in all cases, and policy for the
non-starter is still learned normally from every decisive game.

## Promotion measurements

A candidate must pass two independent checks:

1. score at least 55% against the champion in 200 paired games;
2. stay at or below 20% draws in 64 noisy self-play games.

Each arena pair shares a random legal 0–4 ply opening and swaps candidate color.
This avoids counting one deterministic trajectory hundreds of times. Mirror
games stay on the official initial positions because they answer a different
question: does the candidate deterministically enter a repetition cycle? Four
mirror games are recorded, but they do not veto a candidate that is strong and
still produces decisive exploratory games.

The strength and exploratory checks protect publication; none of these
measurements is a training label.

## Reading the metrics

| Metric | What it measures | Warning sign |
| --- | --- | --- |
| policy loss | Weighted cross-entropy against MCTS visits | Training falls while stable validation rises. |
| policy weight | Mean multiplier after unresolved drawn non-starter examples are omitted | A drop means unsolved draws occupy more of the buffer. |
| WDL loss | Cross-entropy against final W/D/L | Persistent high validation loss means weak result prediction. |
| scalar value MSE | Squared error of `P(win) - P(loss)` against `+1/0/-1` | Measures the ordered auxiliary objective; it must improve without hiding poor draw classification. |
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
- positive paired results against the champion and a frozen previous baseline;
- recorded deterministic draw behavior and controlled exploratory draw rates;
- balanced results across absolute colors and starter roles;
- reproducibility across several random seeds.

The project goal is a genuinely from-scratch AlphaZero learner. Its success is
measured by internal learning and playing strength, with no external component
in data generation, targets, search, promotion or runtime play.