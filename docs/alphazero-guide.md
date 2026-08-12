# AlphaZero in YokaiRust

This guide explains what the network learns, how one generation works and why
repetition draws need special handling. The implementation is self-contained:
it learns from game rules, MCTS and self-play, and the same design is intended
for the future 5×6 game.

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

## One generation

```text
accepted champion
       |
       v
noisy self-play + cycle restarts
       |
       v
rolling replay buffer --stable game split--> validation
       |
       v
resume private learner + Adam, then optimize
       |
       v
candidate --arena + draw gates--> publish champion
       |
       +-- strong but cycling: keep private learner
       +-- strength regression: roll learner back to champion
```

The champion remains the safe self-play source. The learner is an optimization
lineage: a strong candidate may receive more updates after a draw-only rejection
without becoming the published player. A genuine strength regression abandons
that branch.

Each generation uses a fixed optimizer-step budget. Buffer growth therefore
does not silently increase training work. Checkpoints preserve both model
parameters and Adam moments.

Whole games, rather than individual positions, are assigned to validation. A
stable hash of `(generation, seed)` keeps old games on the same side when the
buffer grows. This prevents leakage and makes losses more comparable over time.

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

Every example still trains WDL at full weight. For a drawn position where the
current player is the non-starter, only policy cross-entropy is discounted:

```text
policy_weight = 1 - discount * repetition_visit_mass
```

The checked-in `discount = 0.9` leaves weight 1 for a non-repeating target and
weight 0.1 for a fully cyclic target. Starter draw policies, decisive games and
non-repetition actions are untouched.

This is not a hidden value label. It says only that a limited MCTS search which
failed to convert should be imitated less confidently when it mostly revisits
known states.

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
| policy weight | Mean multiplier after repetition discount | A sharp drop means many non-starter draw targets are cyclic. |
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
