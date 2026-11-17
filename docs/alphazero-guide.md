# AlphaZero in YokaiRust

This guide explains what the network learns, how one generation works and why
repetition draws need special handling. The implementation learns from the
5×6 game rules, MCTS and self-play. The measurements and the published
champion are documented in [Training results](training-results.md).

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
| **temperature** | Sharpness dial when sampling a move from visit counts: `T = 1` samples proportionally, `T → 0` plays the most visited action. An action that wins at once is played at any temperature. |
| **subtree reuse** | Keeping the search tree below the action actually played, so the next search starts warm instead of from scratch. |
| **virtual loss** | Temporary penalty on a path whose leaf is still waiting for its batched evaluation, so parallel simulations spread out instead of piling onto one line. |
| **contempt** | A search-only bias against (or toward) draws. It changes move choice during search, never the official game result. |
| **training set** | Examples used to update parameters. |
| **validation set** | Held-out examples used only to measure generalization, never to update parameters. |

In formulas and field names, `W`, `D` and `L` are the WDL probabilities. The
neutral value `W - L` approaches `+1` for a win, `-1` for a loss and `0` for a
draw.

## Network targets

For every non-terminal position, the network predicts:

- **policy**: 570 action logits;
- **WDL**: win, draw and loss logits from the current player's perspective.

The input describes the current state, recent history, hands and repetition
context. [The input layout](#where-the-input-numbers-come-from) explains each
feature and its dimensions. The engine retains the full history for rules.

The policy target is the normalized MCTS visit distribution, not merely the
action eventually played. Temperature affects action sampling but never
sharpens the stored target.

The WDL target is the official final result:

- win if the recorded player to move eventually won;
- loss if that player lost;
- draw after an official repetition.

Training combines categorical WDL cross-entropy with a small auxiliary loss:
`0.25 × (P(win) - P(loss) - result)²`, where the result is `+1`, `0` or `-1`.
The auxiliary term supplies an ordered value signal; WDL cross-entropy
forces a certain draw to differ from balanced win/loss uncertainty. This
auxiliary loss is a project addition, not standard AlphaZero; its weight is
the `scalar_value_loss_weight` key in `config/training.toml`.

Two further dataset-shaping options exist in the configuration:

- `mirror_augmentation` — each sampled example may be reflected left-to-right
  (the board and the movement rules are symmetric in their columns), doubling
  effective data diversity at no self-play cost. A classic technique from the
  AlphaGo Zero lineage.
- `[optimization.terminal_window_schedule]` — a bootstrap-only schedule that
  oversamples a growing tail of *decisive* endgames alongside the complete
  training set, then stops oversampling at a configured generation. Validation
  keeps the complete held-out games throughout. The separate fixed
  `terminal_window_plies` option instead restricts both sets to decisive tails.
  These are project experiments aimed at teaching endgame conversions.

The WDL head distinguishes a certain draw from a 50/50 win/loss mixture.
Official search converts both distributions to the same scalar value:
`P(win) - P(loss) = 0`. This conversion discards that distinction. Terminal
states have no policy example because they have no legal move distribution.

### How generation zero is bootstrapped

An empty model directory starts with a randomly initialized champion at
generation zero. What is configurable — via `[self_play.bootstrap]` in
`config/training.toml` — is the evaluator self-play uses until the first
candidate is accepted:

- `mode = "neural"` searches with the random network immediately. Generation
  one's MCTS then uses its random policy priors and random WDL estimates;
  training can overcome those initial biases, but the first policy targets
  partly reflect them. This mirrors
  [AlphaGo Zero](https://deepmind.google/blog/alphago-zero-starting-from-scratch/),
  whose search used a randomly initialized network and deliberately omitted
  rollouts.
- `mode = "random_rollout_until_first_promotion"` — the checked-in default —
  disables the network until the first promotion: legal
  actions begin with uniform priors, and a random rollout from each leaf
  supplies the search value. After this rules-only search produces the first
  dataset and an accepted network, later generations switch to neural policy
  and value evaluation. If the first model is rejected, rollout self-play
  remains active until one is accepted.

The rollout mode derives its initial targets from simulated games instead of
random network predictions. These rollout targets can be noisy. The two modes
allow bootstrap experiments with the same subsequent training process.

## Where the input numbers come from

Board and action counts follow the game rules. The eight-frame history and
feature layout are modeling choices; the rules do not prescribe them:

| Constant | Value | Derivation |
| --- | --- | --- |
| Board squares | 30 | 5 columns × 6 rows. |
| Piece planes per frame | 12 | 6 piece types × 2 owners: one binary sheet per (owner, type) pair. A promoted piece is a type of its own. |
| History frames | 8 | Modeling choice: the current position plus seven preceding positions. |
| Spatial input planes | 96 | 12 planes × 8 frames. |
| Hand features per frame | 6 | Normalized count of each of the three droppable piece types, for both players. |
| Global features | 50 | 6 per frame × 8 frames, plus the repetition count and the starter-role flag. |
| Move slots | 480 | Any of the 30 squares can move in any of the 8 directions, with or without promotion. |
| Drop slots | 90 | Any of the 30 squares can receive any of the 3 hand piece types. |
| Policy slots | 570 | 480 moves + 90 drops. |
| Policy context features | 1,140 | Per action slot: occurrence count of the resulting position and an immediate threefold-repetition flag. |

The spatial input is a stack of 96 binary sheets over the 6×5 board. Each
history frame contributes twelve sheets, one per owner and piece type; a sheet
holds `1` on every square occupied by that piece. Summing the twelve sheets of
frame zero recovers the current placement, which makes a single frame easy to
verify by hand.

A hand count is divided by the number of copies of that piece in the game
(four Kirin, four Oni, six Kodama), so a value of `1` means that the player
holds every copy.

The 50 global values carry everything without a board coordinate: the hand
composition of both players for every frame (48 values), the current repetition
count capped at three, and whether the player to move started the game.

The 1,140 policy context values are action-specific, which is why they bypass the
shared trunk and enter only the policy head: “how often has this move's
resulting position occurred” and “does this move complete the third repetition”
are properties of a candidate action, not of the position alone. The occurrence
count is capped at three and normalized to `[0, 1]`; the flag is `1` exactly
when the move would end the game by repetition.

## Why the network has one trunk and two heads

[`src/neural/model.rs`](../src/neural/model.rs) has three conceptual parts:

```text
board history [batch, 96, 6, 5]    globals [batch, 50]
                |                         |
       shared residual tower              |
                |                         |
                +------ concatenate ------+
                            |
                 shared dense representation
                     /                 \
            policy head              WDL head
             570 logits                3 logits
```

The shared convolutional tower learns board patterns useful to both questions.
After the convolutions, its flattened output is concatenated with the global
features and mixed by a shared dense layer. The policy head maps that
representation plus per-action repetition context to 570 action slots. The WDL
head maps the same shared representation to three outcome logits. Separate
heads keep “which move?” and “who wins?” distinct while sharing the state
representation.

### Layer by layer

1. **Input convolution.** A 3×3 kernel slides over the 96 input planes and
   produces 64 channels of unchanged spatial size (padding `Same`, no bias).
   Each channel is a learned local detector; after training, individual channels
   respond to motifs such as “my piece adjacent to an enemy piece” or “empty
   corner”. Batch normalization and ReLU follow.
2. **Residual tower.** Four blocks are stacked. Each block applies two 3×3
   convolutions, with normalization and ReLU between them, then adds the block
   input back to its own output before a final ReLU: `output = input + f(input)`.
   The shortcut carries the original features through all four blocks, which is
   what keeps deep stacks trainable (the ResNet idea).
3. **Flatten and fuse.** The tower output is 64 channels × 30 squares = 1,920
   values. Concatenating the 50 global features gives a 1,970-value vector,
   which a dense layer compresses to the shared 64-unit representation.
4. **Policy head.** The shared representation is concatenated with the 1,140
   per-action context values (1,204 total) and mapped to 570 action logits.
5. **WDL head.** The shared representation alone passes through a 64-unit hidden
   layer and is mapped to three outcome logits.

All convolutions are bias-free; the affine scaling lives in the BatchNorm
layers, which is the standard ResNet convention. The heads emit raw logits;
callers apply softmax before treating them as probabilities, and search converts
WDL logits to `P(win) - P(loss)` (see [Network targets](#network-targets)).

With the default configuration the network has about 1.17 million parameters.
The policy head's single dense layer holds 59% of them (1,204 inputs × 570
outputs), the residual tower 25% and the fusion layer 11%. Each layer width is
a checkpointed setting in `AlphaZeroNetworkConfig`.
[Next steps](next-steps.md) names the layer to reconsider.

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
tower.

### History and repetition context

The board alone cannot tell how often it occurred. Eight frames provide recent
motion; the current repetition scalar and per-action occurrence counts expose
the immediate rule consequence. `Game` still owns the complete history and is
the sole authority for declaring a draw. Neural input guides search; it never
replaces the rules engine.

### Fixed 570-action output

A fixed output shape lets one dense layer and checkpoint format cover every
position. A board move occupies one slot per origin square, direction and
promotion choice; a drop occupies one slot per target square and piece type. Illegal actions are masked and legal probabilities renormalized
before MCTS uses them. This batches more simply than a variable-length vector,
while `PolicyIndex` keeps the action mapping typed and tested.

## One generation

```text
accepted champion --noisy self-play + visited-state restarts
       |
       v
rolling replay buffer --stable family split--> validation
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
Promotion checks both playing strength and the draw rate in noisy self-play.

A `pending-generation.json` journal reserves the attempt and snapshots its
configuration. Loading a user TOML rejects unknown keys, including nested ones,
and identifies invalid settings by field name. Journal JSON keeps its existing
compatibility rules so historical fields can still be read.
Self-play is reused after an interruption; once candidate
weights exist, optimization metrics and weights are reused too. Arena and
productivity checks may run again. The completed decision report is written
before `latest`: if publishing the pointer fails, restarting finishes that
decision without retraining. An incomplete attempt requires its original
configuration; use new experiment paths to start with different settings.
The last journal remains on disk and is replaced when the next attempt starts.
These boundaries protect process-interruption recovery, not concurrent training
processes writing to the same directories.

Each generation uses a fixed optimizer-step budget. Buffer growth therefore
does not silently increase training work. Checkpoints preserve both model
parameters and Adam moments. The learning rate starts high enough for bootstrap
and can fall at accepted-champion milestones. Rejected attempt numbers do not
advance that schedule: training maturity is measured by published progress.

Whole trajectory families are assigned to validation. A stable hash of the
original `(source_generation, seed)` determines the bucket; every restart
inherits that identity, including restarts of restarts and after the parent
is evicted. Tiny corpora may have an empty bucket: moving an example to fill
it would break this guarantee. Keep the split seed and fraction fixed within
one experiment.

The family guarantee requires stored ancestry. A loaded game without it uses
its own `(source_generation, seed)` as the split identity, which cannot establish
separation from related trajectories. Use a fresh self-play directory when a
clean split is required.

The [reading guide](reading-guide.md) maps these phases to the Rust modules
and explains ownership, errors and the shared native/browser inference path.

Burn's `Tensor<B, 4>` is a tensor with four dimensions, such as
`[batch, channels, rows, columns]`. `B` is a type parameter selecting the
computation backend. The constraint `B: Backend` tells the Rust compiler that
this type must implement Burn's `Backend` trait. `Autodiff<B>` adds gradient
tracking for training; its inner backend performs inference without that
tracking during self-play and arenas.

## Architecture decisions at a glance

| Choice | Why it exists | Cost or limitation |
| --- | --- | --- |
| Burn backend generic | The same model code runs on CPU, Metal and browser backends. | Compiler errors can involve long type names combining tensors, backends and gradient tracking. |
| One GPU inference service | Many games share large batches instead of each owning a model. | Workers wait briefly for batching and communicate through channels. |
| Contiguous MCTS node `Vec` | Indices are stable, allocation is cheap and traversal is cache-friendly. | Removing arbitrary nodes is less convenient than pointer-owned trees. |
| Rolling replay buffer | Mixes recent generations so one noisy batch does not replace all knowledge. | Targets are generated by networks of different ages. |
| Fixed optimizer steps + staged learning rate | Cost stays constant while later updates can become gentler. | Not every example is visited in every generation. |
| One accepted champion | Rejected candidates cannot generate the next dataset. | Escaping a plateau may take several attempts from the same weights. |
| Family-based validation | A trajectory and its recorded descendants stay in the same bucket. | Requires stored ancestry; independent games may still revisit the same state. |
| Structured progress enum | The pipeline emits typed events; the CLI decides how to print them. | Adding an event requires updating every exhaustive `match`. |
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

> **Project-specific experiments.** Mechanisms 2 and 4 below, the auxiliary
> scalar loss, and `repetition_contempt` adapt training to repetition draws.
> They are not standard AlphaZero components. Each subsection names its
> configuration keys. Mechanism 3 follows the published Go-Exploit "Visited
> States" variant ([paper](https://arxiv.org/abs/2302.12359)).
>
> Mechanisms 2 and 4 assume that the starter is the defending side. Nothing
> establishes such an asymmetry on this board, so the checked-in configuration
> turns both off: a draw is worth zero to either player and both players'
> moves are imitated.

### 1. WDL keeps the outcome observable

The value head learns draw probability explicitly. Stored targets and official
arenas remain neutral.

### 2. Self-play gives the two roles different incentives

During self-play only, the starter can evaluate a draw at `+v` and the
non-starter at `-v`, where `v` is `starter_draw_value`:

```text
starter:     win - loss + v * draw
non-starter: win - loss - v * draw
```

With `v` below one, both still rank win above draw above loss. The starter
learns its best defence; the non-starter is pushed to search for a conversion.

Configuration key: `starter_draw_value` (`0.0` in the checked-in configuration,
which disables the mechanism).
The code also implements a mutually exclusive alternative, `repetition_contempt`,
which penalizes only the exact action causing a repetition; it is disabled
(`0.0`) in the checked-in configuration.

### 3. Restarts explore recent visited states

One quarter of trajectories restart from a uniformly sampled non-initial,
nonterminal state visited in the recent replay buffer. Both decisive and drawn
games contribute states, and repeated occurrences remain in the archive so
frequently visited regions receive proportionally more restarts. The complete
prefix is replayed, preserving exact repetition counts, starter role and encoder
history.

These trajectories use 800 MCTS simulations per move instead of the regular
200. Their local temperature schedule restarts at ply zero, adding exploration
at later positions in the game tree. The network, rules and PUCT search produce
the targets, and the trajectory remains related to its source family.

Configuration keys: `restart_fraction` (0.25) and `restart_simulations` (800).

### 4. Policy targets in drawn games

The configured `non_starter_draw_policy_weight` of 1.0 gives both players the
full MCTS visit distribution, including visits to moves that repeat a position.
Every example also trains the value head on the official outcome.

A weight below 1.0 enables an experimental preference for avoiding draws for
the player who moved second. If an available move causes a third repetition
and other moves received visits, training removes the repeating moves and
normalizes the remaining visits to sum to one. This adjusted target receives
full weight. Otherwise the original target receives the configured weight.

## Promotion measurements

A candidate must pass two independent checks:

1. score at least 55% against the champion in 200 paired games, with a
   one-sided paired sign-flip p-value at most 0.05;
2. stay at or below 20% draws in 64 noisy self-play games.

Each pair contributes a candidate score of 0, 0.5, 1, 1.5 or 2 points. The
report retains counts of these five outcomes. The statistical test compares
the summed advantage against all independent sign flips of the paired
advantages; integer half-points allow exact dynamic programming. This is the
paired-sample permutation test described in the
[SciPy reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
(`samples`, one sample, `greater`). The two colors are never treated as
independent observations. The 5% threshold applies to one fixed-size arena;
it does not control false promotions over an unlimited sequence of attempts.
Confirm a research gain with an independent, preplanned comparison. An arena
result without paired statistics cannot establish a promotion under this gate.

Each arena pair shares a random legal 0–4 ply opening and swaps candidate color.
Sampling openings exposes the models to different trajectories. The separate
productivity probe uses the noisy self-play settings that generate training
data. Deterministic play between identical models can settle into one stable
line, so it cannot substitute for that probe.

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
- positive paired results against the champion and a fixed reference model;
- controlled exploratory draw rates;
- balanced results across absolute colors and starter roles;
- reproducibility across several random seeds.

The project goal is an AlphaZero learner trained from scratch. Its success is
measured by internal learning and playing strength, with no external component
in data generation, targets, search, promotion or runtime play.

Winning sooner is not a separate learning target. Terminal wins all have value
`+1` from the winner's perspective, and visit counts follow the policy priors
whenever several actions look almost equally good. The most visited action can
therefore be a promising continuation while another action ends the game.

The search settles that case with the rules alone. When a legal action wins at
once, by capturing the opposing Koropokkuru or by leaving the opponent without
a legal action, that action is played whatever the visit counts and the
temperature say. The policy target still stores the raw visit distribution.
Nothing favors a shorter win beyond that single ply: the search can still
choose a longer line over a forced win in a few moves.
