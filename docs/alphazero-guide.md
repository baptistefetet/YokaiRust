# AlphaZero in YokaiRust

This document explains what the training loop learns, why the draw problem
appeared, and what continued training can and cannot guarantee.

## The two predictions

For every non-terminal position, the network produces:

- **policy**: 132 logits, one per encoded action;
- **value**: one number in `[-1, 1]` from the current player's perspective.

The policy is not trained directly on the action that happened. Its target is
the normalized MCTS visit distribution after illegal actions are masked. This
retains information about alternatives explored by the search.

The value target is the final game result:

- `+1` if the player to move in the recorded position eventually won;
- `-1` if that player lost;
- `0` for an official draw.

There is deliberately no policy example for a terminal position: it has no legal
move distribution to learn.

## Why midgame examples are noisy but not invalid

A midgame self-play position has a valid final result, but that result is a
high-variance estimate of the position's true game-theoretic value. Weak players
can win good positions or lose winning positions. Early networks therefore feed
uncertain targets back into themselves.

The terminal curriculum reduces that variance while bootstrapping:

1. train on the last 8 plies of decisive games;
2. expand to 16, 32 and 64 plies after validated promotions;
3. finally return to all buffered positions.

Draw games are excluded while a finite terminal window is active, because their
last action is the action that completed a repetition, exactly the policy we do
not want to imitate during tactical bootstrapping.

This curriculum is a training strategy, not a change to Yokai rules.

## One generation

```text
champion
   |
   +--> noisy self-play --> replay buffer --> train/validation split
   |                                          |
   |                                          v
   +--------------------------------------> candidate
                                              |
                    +-------------------------+-----------------------+
                    |                         |                       |
              vs champion               mirror probe           noisy probe
              score >= 55%             draws <= 35%           draws <= 20%
                    |                         |                       |
                    +-------------------------+-----------------------+
                                              |
                                      publish only if all pass
```

The candidate starts from champion weights rather than random weights. A rejected
candidate checkpoint is retained for debugging, but the champion pointer and
curriculum phase do not advance.

## Why draws increased

An official repetition is worth zero. If an inaccurate value head predicts that
all non-repeating alternatives lose, deeper MCTS rationally selects the safe
draw. Self-play then trains the policy to reproduce that branch and trains the
value to predict zero, creating a feedback loop.

Three mechanisms address different parts of the loop:

- **terminal curriculum** supplies clearer decisive targets;
- **repetition contempt** makes the player causing a repetition dislike that
  branch during self-play search;
- **promotion gates** stop deterministic or noise-sensitive cycles from becoming
  champion.

Contempt is search-only. Replay outcomes and arena scores preserve the official
draw value of zero, so the published model must still prove itself under the real
rules.

## Reading the metrics

| Metric | Useful interpretation | Common warning sign |
| --- | --- | --- |
| policy loss | Cross-entropy against MCTS visits | Training falls while validation rises: policy overfit. |
| value loss | Squared error against final result | Persistently high: weak result prediction or noisy labels. |
| entropy | Spread of predicted legal moves | Sudden collapse can indicate premature policy certainty. |
| top-1 | Agreement with the largest MCTS target | Useful for tactical curriculum, not a complete strength score. |
| calibration | Difference between value prediction and outcome | Low validation error means values better match observed results. |
| illegal mass | Raw probability assigned to illegal actions | High values waste network capacity even though MCTS masks them. |
| W/L/D | Actual behavior | The primary health signal; always separate draws from wins. |

Loss improvements do not prove playing-strength improvements. Promotion arenas
and fixed tactical/baseline tests remain necessary.

## Will continued self-play produce a perfect model?

There is no such guarantee. Neural self-play can plateau, forget useful patterns,
cycle between strategies or exploit weaknesses specific to the current champion.
More compute only produces more samples from the current process; it does not
turn that process into a proof of optimal play.

A convincing **strong** model should satisfy all of these repeatedly:

- immediate and multi-ply tactical suites;
- stable low draw rates with and without exploration noise;
- positive results against frozen historical baselines;
- promotion across multiple seeds, not one lucky arena;
- calibrated value predictions on held-out games.

A **perfect** model requires an independent oracle. For a finite 3x4 game that
normally means a complete solver/tablebase, then checking every reachable state
or at least measuring the network/MCTS decisions against the oracle. AlphaZero
can be an excellent player without being a proof-producing solver.

## Can Ratatui development start now?

Yes. The TUI depends on stable engine-side contracts (`Game`, `Action`, `Replay`,
`ActionAnalysis`) rather than on convergence of the latest model. Training can
continue independently while UI work proceeds.

For reproducible UI tests, use a fixed checkpoint or `UniformEvaluator` instead
of whatever `models/champion` happens to reference. The one-player mode can load
the current champion at runtime, while replay and two-player modes require no
neural model at all.

The next UI milestone should avoid importing Burn types into widgets. A small
application/controller layer can expose board state, legal actions and analysis
rows; Ratatui should only render that state and emit user intents.
