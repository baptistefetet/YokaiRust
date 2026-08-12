# AlphaZero in YokaiRust

This document explains what the training loop learns, why the draw problem
appeared, and what continued training can and cannot guarantee.

## The two predictions

For every non-terminal position, the network produces:

- **policy**: 132 logits, one per encoded action;
- **WDL**: three logits for win, draw and loss from the current player's perspective.

Its input contains the current state plus the seven preceding states. Each
state contributes 16 planes for pieces and hands; two final planes contain the
current repetition count and whether the player to move was the starter, for
130 planes in total. The policy branch also receives two action-aligned features
for each of its 132 slots: the occurrence count of the resulting position and
an immediate-threefold flag. Search derives these exactly from the complete
game history. A finite board-history stack alone does not make repetition fully
Markovian.

The policy is not trained directly on the action that happened. Its target is
the normalized MCTS visit distribution after illegal actions are masked. This
retains information about alternatives explored by the search.
Move-selection temperature is applied only when choosing the played action. It
never sharpens the stored policy target: even after self-play becomes greedy,
the target remains the complete normalized visit distribution.

The WDL target is the one-hot final game result:

- win if the player to move in the recorded position eventually won;
- loss if that player lost;
- draw for an official repetition.

Official search derives its neutral scalar as `P(win) - P(loss)`. Unlike the old
scalar target, a certain draw is therefore distinct from an uncertain 50/50
mixture of wins and losses.

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
champion --> noisy self-play --> replay buffer --> train learner candidate
   ^                                                    |          |
   +-- publish only if arena + both draw gates ----------+          |
                                                        +-- keep if arena passed
```

The candidate starts from a separate private learner lineage. The champion
remains the source of every following self-play batch until a candidate passes
the official paired arena and both anti-cycle gates. When a candidate passes
the strength arena but fails a draw gate, it stays private but its weights and
Adam state initialize the next optimization attempt. Failing the strength arena
rolls that lineage back to the champion. This preserves learning across
draw-only rejections without publishing a cycling player.

Once the replay buffer contains a drawn game, it also exposes validated ongoing
prefixes from two to eight plies before the terminal repetition. The checked-in
`cycle_restart_fraction = 0.25` deterministically replaces one quarter of the
next trajectories with these prefixes. The complete action history, starter and
seven-frame encoder context are preserved, while the exploration-temperature
schedule restarts at local ply zero. Generation zero has no archive and therefore
uses only official initial positions.

Optimization is continuous too. Each generation performs a fixed number of
uniformly sampled mini-batch updates instead of full passes over an ever-growing
buffer. The checkpoint stores both the trainable model and Adam's first/second
moments, so restarting the command does not silently reset the optimizer.
Validation checkpoints are measurements; they no longer roll the model back to
an earlier point with optimizer moments belonging to a different model.

This deliberately follows the guarded AlphaGo Zero-style update used in the
project's original plan. AlphaGo Zero evaluated each candidate and promoted it
only above 55%; AlphaZero used the latest parameters continuously and omitted
that selection step. The continuous variant was tested here first, but one
anti-repetition experiment published a candidate that then lost 0-200 to its
source. YokaiRust therefore requires the 55% result and both draw gates before
publication.

- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero paper](https://www.nature.com/articles/nature24270)

## Why draws can increase while strength improves

An official repetition is worth zero. If the WDL head predicts that
non-repeating alternatives lose, deeper MCTS rationally selects the safe draw.
Self-play then trains the policy to reproduce that branch. The old scalar head
also learned zero; the WDL head now learns an explicit draw probability, which
makes the loop observable and shapeable. This can reflect an inaccurate result
estimate, but it can also be the correct decision against an equally strong
opponent.

The standard bootstrap keeps every position and every official result, but uses
the explicit WDL draw probability inside self-play search. With the configured
`starter_draw_value = 0.25`, a current starter evaluates a draw as `+0.25` and a
current non-starter as `-0.25`; both still rank win > draw > loss. The stored WDL
target remains official and every arena uses neutral `P(win) - P(loss)`.
The older edge-local `repetition_contempt` remains available for reproducing
historical experiments but is mutually exclusive with the role-aware utility.
Three separate measurements both expose the behavior and protect the next
self-play source:

- self-play W/L/D shows behavior with exploration;
- the mirror probe exposes deterministic repetition cycles from the official
  initial position;
- the paired arena measures improvement on shared random legal 0-4 ply openings.

The two games of each arena pair start from the same opening and exchange the
candidate's absolute color. This cancels much of the opening and move-order
bias. Using different seeds without different openings is not sufficient in a
noise-free, temperature-zero search: it merely repeats the same deterministic
trajectory and makes a nominal 200-game score look more informative than it is.
The separate mirror remains on the official initial position. Four games cover
both absolute starting orientations and their color-swapped pairs; more
temperature-zero copies would repeat the same trajectories. Since the game is
known to be decisive, any mirrored repetition now rejects the candidate.

The fixed-step generation-12 run demonstrates why no single draw threshold can
diagnose strength: it drew 141/256 self-play games, yet beat generations 1, 4
and 8 by 40-0 each, then scored 20 wins and 20 draws against generation 11.
Setting `starter_draw_value = 0.0` restores the neutral experiment. The shaped
default is explicit because neutral runs repeatedly poisoned later candidates.

The clean diversified-arena run through candidate 15 demonstrates the intended
interaction. Candidate 4 developed a 64/64 initial mirror cycle and was rejected
despite a 74% arena score. Attempts 5–9 remained anchored to champion 3; attempt
10 eventually removed the cycle and passed all three gates. Champions then
advanced through 11, 12 and 14. Candidate 15 again cycled and was rejected, so
the buffer stayed sourced from champion 14. Across all attempts only 7.1% of
games and 7.8% of positions in the buffer came from draws.

This does not mean the raw network is draw-proof. A 64-game champion-14 probe
gave 25 neutral noisy draws, 11 shaped noisy draws, and 4 neutral
temperature-zero draws. Repetition contempt is therefore still an exploration
aid. It changes leaf preferences inside self-play MCTS, not official outcomes
or replay values. The arena strength score and deterministic mirror are always
measured with unshaped search.

The August v7 diagnostic showed why a search-only contempt is not sufficient:
it reduced the first two self-play batches to 1/256 and 6/256 draws, but the
second candidate then lost 0-200 to its predecessor under official arena rules.
Any future contempt experiment must therefore sit behind a champion promotion
gate; a low draw count alone must never publish a network.

For a controlled bootstrap, `terminal_window_schedule` automates the diagnostic
suggested during development without replacing the normal replay buffer. Every
position and draw remains present; final decisive positions are oversampled to
`decisive_fraction`, their tail widens geometrically, and the additional
sampling eventually stops. The schedule depends only on the checkpoint
generation, so interruption and resume cannot silently change the selected
dataset.

For this particular game, draws cannot be treated as the expected endpoint of
perfect play. Dōbutsu Shōgi, whose 3×4 rules and initial position correspond to
this implementation, has been strongly solved: the player moving second has a
forced win in 78 plies. Self-play reports therefore distinguish absolute piece
owner (`First`/`Second`) from move order (`starter`/`non-starter`). Progress
toward the known result should eventually favor `non-starter`, not repetition.
See the [original 2009 retrograde-analysis report](https://ipsj.ixsq.nii.ac.jp/records/62415)
and the independently verified [interactive tablebase](https://dobutsu.brianhliou.com/).

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
| value loss | WDL cross-entropy against the final result | Persistently high: weak result prediction or noisy labels. |
| entropy | Spread of predicted legal moves | Sudden collapse can indicate premature policy certainty. |
| top-1 | Agreement with the largest MCTS target | Useful for tactical tests, not a complete strength score. |
| calibration | Difference between value prediction and outcome | Low validation error means values better match observed results. |
| WDL top-1 | Agreement with the observed win/draw/loss class | A high aggregate score can hide poor draw recognition; read with draw error. |
| draw error | Absolute error of predicted draw probability | Rising values expose WDL miscalibration before scalar `W-L` necessarily changes. |
| illegal mass | Raw probability assigned to illegal actions | High values waste network capacity even though MCTS masks them. |
| W/L/D | Actual behavior | The primary health signal; always separate draws from wins. |

Each generation report also derives target-only diagnostics from the self-play
data, split between drawn and decisive games: visit-policy entropy, largest
action probability, legal-action coverage, mass on already-seen positions, and
mass on actions that immediately close a threefold draw. These require neither
an oracle nor assumptions specific to the 3×4 board and are the first evidence
to inspect when the learner stalls.

Loss improvements do not prove playing-strength improvements. Diagnostic arenas
and fixed tactical/baseline tests remain necessary.

Policy loss is a moving-target metric: each network changes the MCTS visit
distribution used to train the next network. It therefore has no fixed zero-loss
reference across generations. The old scalar MSE had a second trap in this game:
a certain draw and an uncertain 50/50 win/loss mixture both targeted zero. WDL
removes that aliasing, but a lower cross-entropy still does not prove stronger
play. Always read both losses beside top-1, illegal mass and W/L/D.

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

A **perfectness claim** requires an independent proof or oracle, but that oracle
is not part of AlphaZero. For 3×4 it may be used manually to audit a finished
checkpoint; it must never influence examples, search, promotion or gameplay.
The normal pipeline intentionally relies only on self-play and fixed learned or
search baselines, so the same design remains applicable to the planned 5×6 game
where no tablebase is expected. AlphaZero can be an excellent player without
being a proof-producing solver.

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
