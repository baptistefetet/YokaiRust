# AlphaZero in YokaiRust

This document explains what the training loop learns, why the draw problem
appeared, and what continued training can and cannot guarantee.

## The two predictions

For every non-terminal position, the network produces:

- **policy**: 132 logits, one per encoded action;
- **value**: one number in `[-1, 1]` from the current player's perspective.

Its input contains the current state plus the seven preceding states. Each
state contributes 16 planes for pieces and hands; a final plane contains the
current repetition count, for 129 planes in total. This history is necessary
because repetition makes the board alone non-Markovian: the same visible
position can have a different value depending on which earlier state the next
move would repeat.

The policy is not trained directly on the action that happened. Its target is
the normalized MCTS visit distribution after illegal actions are masked. This
retains information about alternatives explored by the search.
Move-selection temperature is applied only when choosing the played action. It
never sharpens the stored policy target: even after self-play becomes greedy,
the target remains the complete normalized visit distribution.

The value target is the final game result:

- `+1` if the player to move in the recorded position eventually won;
- `-1` if that player lost;
- `0` for an official draw.

There is deliberately no policy example for a terminal position: it has no legal
move distribution to learn.

## Why every played position is retained

A midgame self-play position has a valid final result, but that result is a
high-variance estimate of the position's true game-theoretic value. Weak players
can win good positions or lose winning positions. Early networks therefore feed
uncertain targets back into themselves.

Standard AlphaZero nevertheless retains every non-terminal position. Mixing
many games in a rolling buffer makes those noisy samples useful in aggregate and
lets opening and midgame knowledge improve alongside tactics.

The optional `terminal_window_plies` setting keeps only the tail of decisive
games. It is useful for the diagnostic experiment suggested during development:
if a network cannot learn nearly forced final moves, the policy/value pipeline
probably contains a bug. It is not enabled in normal training.

## One generation

```text
latest network
      |
      +--> noisy self-play --> replay buffer --> train/validation split
                                                   |
                                                   v
                                             next network
                                                   |
                                      save and publish unconditionally
                                                   |
                    +------------------------------+------------------+
                    |                              |                  |
              vs previous                    mirror probe        noisy probe
             score diagnostic               draw warning       draw warning
```

The next network starts from the latest weights. Once training completes, it
becomes the source of the following self-play batch. Arena and draw results are
measurements: they never choose which network is allowed to continue learning.

Optimization is continuous too. Each generation performs a fixed number of
uniformly sampled mini-batch updates instead of full passes over an ever-growing
buffer. The checkpoint stores both the trainable model and Adam's first/second
moments, so restarting the command does not silently reset the optimizer.
Validation checkpoints are measurements; they no longer roll the model back to
an earlier point with optimizer moments belonging to a different model.

This is the specific distinction made by the original publications. AlphaGo
Zero evaluated each candidate and promoted it only above 55%; AlphaZero used the
latest parameters continuously and omitted that selection step. Many community
projects still call the gated variant “AlphaZero”, which explains the common
terminology overlap. YokaiRust keeps the 55% number as a readable strength
indicator, not as a publication condition.

- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero paper](https://www.nature.com/articles/nature24270)

## Why draws can increase while strength improves

An official repetition is worth zero. If the value head predicts that
non-repeating alternatives lose, deeper MCTS rationally selects the safe draw.
Self-play then trains the policy to reproduce that branch and trains the value
to predict zero, creating a feedback loop. This can reflect an inaccurate value
estimate, but it can also be the correct decision against an equally strong
opponent.

The standard configuration deliberately leaves this feedback visible rather
than changing its objective: all positions remain in the dataset, repetition
contempt is zero and every generation continues. Three separate measurements
make this behavior visible:

- self-play W/L/D shows behavior with exploration;
- the mirror probe exposes deterministic repetition cycles;
- the paired arena tells whether the new network improved against its predecessor.

The fixed-step generation-12 run demonstrates why no single draw threshold can
diagnose strength: it drew 141/256 self-play games, yet beat generations 1, 4
and 8 by 40-0 each, then scored 20 wins and 20 draws against generation 11.
Terminal windows and repetition contempt remain explicit research experiments,
not hidden corrections to the default algorithm.

For a controlled bootstrap, `terminal_window_schedule` automates the diagnostic
suggested during development: begin with only the final decisive positions,
then widen the tail geometrically and finally restore every position and draw.
The schedule depends only on the checkpoint generation, so interruption and
resume cannot silently change the selected dataset.

For this particular game, draws cannot be treated as the expected endpoint of
perfect play. Dōbutsu Shōgi, whose 3×4 rules and initial position correspond to
this implementation, has been strongly solved: the player moving second has a
forced win in 78 plies. Self-play reports therefore distinguish absolute piece
owner (`First`/`Second`) from move order (`starter`/`non-starter`). Progress
toward the known result should eventually favor `non-starter`, not repetition.

A longer exploration window is not automatically an improvement. In a paired
generation-12 diagnostic using identical seeds and 200 simulations per move,
the 12-ply schedule produced starter/non-starter/draw counts of 14/32/18. A
48-ply schedule reduced draws to 8/64 but changed the split to 29/27/8: it mostly
replaced repetitions with random mistakes and erased the known second-mover
signal. The default therefore remains 12 plies while the training-data remedy is
evaluated separately.

## Reading the metrics

| Metric | Useful interpretation | Common warning sign |
| --- | --- | --- |
| policy loss | Cross-entropy against MCTS visits | Training falls while validation rises: policy overfit. |
| value loss | Squared error against final result | Persistently high: weak result prediction or noisy labels. |
| entropy | Spread of predicted legal moves | Sudden collapse can indicate premature policy certainty. |
| top-1 | Agreement with the largest MCTS target | Useful for tactical tests, not a complete strength score. |
| calibration | Difference between value prediction and outcome | Low validation error means values better match observed results. |
| illegal mass | Raw probability assigned to illegal actions | High values waste network capacity even though MCTS masks them. |
| W/L/D | Actual behavior | The primary health signal; always separate draws from wins. |

Loss improvements do not prove playing-strength improvements. Diagnostic arenas
and fixed tactical/baseline tests remain necessary.

Policy loss is a moving-target metric: each network changes the MCTS visit
distribution used to train the next network. It therefore has no fixed zero-loss
reference across generations. Value loss has a second trap in this game: as the
draw fraction grows, more targets equal exactly zero, which can lower mean squared
error without making the player stronger. Always read both losses beside top-1,
illegal mass and W/L/D.

Arena progress is emitted when concurrent games finish, not by game index. Short
wins can all appear before long losses. Only the final paired result is meaningful;
YokaiRust also reports candidate results separately as absolute `First` and
`Second` to expose side-dependent outcomes.

## Will continued self-play produce a perfect model?

There is no such guarantee. Neural self-play can plateau, forget useful patterns,
cycle between strategies or exploit weaknesses specific to recent opponents.
More compute only produces more samples from the current process; it does not
turn that process into a proof of optimal play.

A convincing **strong** model should satisfy all of these repeatedly:

- immediate and multi-ply tactical suites;
- stable, explainable W/L/D behavior with and without exploration noise;
- positive results against frozen historical baselines;
- progress across multiple seeds and historical checkpoints;
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
of whatever `models/latest` happens to reference. The one-player mode can load
the latest network at runtime, while replay and two-player modes require no
neural model at all.

The next UI milestone should avoid importing Burn types into widgets. A small
application/controller layer can expose board state, legal actions and analysis
rows; Ratatui should only render that state and emit user intents.
