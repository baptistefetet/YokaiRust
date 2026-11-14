# First 5×6 training run — 14 November 2026

This directory records the first training run on the 5×6 board. The run is
exploratory: it asks whether the pipeline validated on the 3×4 game learns the
larger game with the same network, before any change of architecture.

## Protocol

The protocol was recorded before the run; [`protocol.json`](protocol.json)
holds the exact identifiers.

- **Configuration:** [`config/training.toml`](../../../config/training.toml)
  at the recorded revision. Metal backend, seed `20261108`.
- **Initialization:** empty `models/expert-5x6` and `data/expert-5x6`
  directories, random weights, no imported optimizer state or replay.
- **Network:** 64 filters, four residual blocks, 64-unit shared and value
  layers; model format 5, encoder version 5, rules version 3.
- **Fixed reference:** generation 0 of this run, the random initialization.
  The pipeline creates it when the run starts, so its file hashes are added to
  the protocol at the first generation boundary. It stays in the model
  directory, whose history is never rewritten.
- **Promotion:** the pipeline's gates, unchanged: 55% of 200 paired arena games
  at 400 simulations per move, a one-sided paired p-value of at most 0.05, and
  at most 20% draws among 64 noisy probe games.

Three settings differ from the 3×4 configuration, each for a reason stated in
the configuration file: draws are worth zero to both players in self-play
search and both players' moves are imitated in drawn games; the learning rate
stays constant; arena searches evaluate eight leaves per inference.

### What the run has to show

These criteria were written before the first generation:

1. Every generation completes, and its duration is recorded.
2. Candidates keep being promoted: at least half of the first ten attempts.
3. Validation policy accuracy rises and the share of policy mass on illegal
   actions falls from one accepted generation to the next.
4. Self-play draws stay below 20% and mean game length stays well under the
   256-ply limit.
5. The latest champion beats the fixed reference and an earlier champion of
   the same run in separate arenas.

## Measurements before the run

One generation was timed from random initialization in throwaway directories,
on an Apple M4 Max, to choose the arena settings. Each row is a single run.

| Arena search | Inference batch | Arena | Whole generation | Arena result |
| --- | --- | --- | --- | --- |
| 1 leaf per inference | 128 | 6 min 27 s | 7 min 30 s | 180–18, 2 draws |
| 8 leaves per inference | 128 | 1 min 43 s | 2 min 36 s | 184–14, 2 draws |
| 8 leaves per inference | 512 | 1 min 16 s | 2 min 14 s | 184–14, 2 draws |

With one leaf per inference, the last ten games of the arena took 3 min 14 s,
as long as the first 190. The third row is the recorded configuration.

Before the 256-ply draw rule existed, the same generation did not complete:
with a 512-ply safety limit one bootstrap game exceeded it, and with a
4,096-ply limit the last ten arena games were still running after six minutes.

## Observations: the first ten attempts

The ten attempts ran in two invocations of five generations and took
67 min 43 s in total. Eight candidates were promoted; the accepted champion
after the tenth attempt is **generation 10**, reached through generations
1, 2, 3, 4, 6, 7 and 9.

| Attempt | Source | Arena W–L–D | Score | Paired p | Probe draws | Verdict | Duration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 188–12–0 | 94.00% | 7.4 × 10⁻²⁶ | 4/64 | promoted | 2 min 01 s |
| 2 | 1 | 152–26–22 | 81.50% | 1.0 × 10⁻¹⁸ | 8/64 | promoted | 6 min 50 s |
| 3 | 2 | 130–46–24 | 71.00% | 5.5 × 10⁻¹⁰ | 8/64 | promoted | 7 min 51 s |
| 4 | 3 | 121–37–42 | 71.00% | 2.6 × 10⁻¹¹ | 5/64 | promoted | 6 min 51 s |
| 5 | 4 | 93–81–26 | 53.00% | 0.23 | 8/64 | rejected | 7 min 52 s |
| 6 | 4 | 141–31–28 | 77.50% | 3.7 × 10⁻¹⁴ | 5/64 | promoted | 7 min 18 s |
| 7 | 6 | 112–41–47 | 67.75% | 2.1 × 10⁻⁹ | 7/64 | promoted | 7 min 47 s |
| 8 | 7 | 59–120–21 | 34.75% | 1.0 | 8/64 | rejected | 7 min 07 s |
| 9 | 7 | 102–80–18 | 55.50% | 0.038 | 6/64 | promoted | 6 min 47 s |
| 10 | 9 | 135–38–27 | 74.25% | 2.0 × 10⁻¹² | 9/64 | promoted | 7 min 19 s |

The score counts a draw as half a point. The first attempt is short because
its self-play uses random rollouts instead of the network. From the second
attempt on, a generation takes 7 min 18 s on average: 4 min 18 s of self-play,
8 s of optimization, 1 min 58 s of arena and 55 s of probe.

### Self-play games

Each attempt plays 256 games. “Wins without capture” counts the games lost by
a player who had no legal action.

| Attempt | Mean plies | Median | 90th percentile | Move-limit draws | Repetition draws | Wins without capture | Positions from drawn games |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 37.8 | 33 | 63 | 2 | 0 | 1 | 5% |
| 2 | 63.0 | 50 | 125 | 12 | 1 | 7 | 22% |
| 3 | 79.9 | 56 | 204 | 22 | 5 | 11 | 29% |
| 4 | 82.9 | 54 | 256 | 27 | 11 | 10 | 39% |
| 5 | 86.6 | 55 | 256 | 30 | 7 | 13 | 39% |
| 6 | 85.4 | 59 | 256 | 30 | 5 | 8 | 35% |
| 7 | 87.2 | 58 | 237 | 22 | 16 | 5 | 33% |
| 8 | 76.2 | 58 | 189 | 12 | 13 | 7 | 24% |
| 9 | 70.5 | 53 | 143 | 9 | 13 | 11 | 19% |
| 10 | 86.4 | 64 | 201 | 14 | 15 | 22 | 23% |

Draws peaked at 38 games out of 256 (14.8%) in attempts 4 and 7. Games that
reach the 256-ply limit became rarer after attempt 6, while repetition draws
became more frequent. A drawn game is long, so it supplies more training
positions than its share of games suggests.

### Independent arenas

Generation 10 played three separate arenas of 200 games at 400 simulations per
move with seed `20261114`, a seed the training run does not use. The seed and
the three references were chosen before the arenas ran.

| Reference | W–L–D | Score | Paired p |
| --- | --- | --- | --- |
| Generation 0 (fixed reference) | 198–0–2 | 99.50% | 7.9 × 10⁻³¹ |
| Generation 2 | 191–1–8 | 97.50% | 1.6 × 10⁻³⁰ |
| Generation 4 | 166–6–28 | 90.00% | 5.0 × 10⁻²⁹ |

### Validation on a fixed set of games

The pipeline measures validation metrics at the end of each attempt, on the
validation games that the buffer holds at that moment. That set grows with
each attempt. For the accepted generations, top-1 policy accuracy rose at
every step (0.066, 0.123, 0.218, 0.230, 0.293, 0.296, 0.303, 0.312). The share
of policy mass on illegal actions fell from 0.401 to 0.193, with one exception:
0.219 for generation 4 and 0.223 for generation 6.

`yokai diagnose-endgames` gives a comparison on one fixed set: after the tenth
attempt it evaluated every checkpoint on the same 240 validation games
(15,023 positions, 25 of the games drawn). A game is assigned to validation
from its seed, so no checkpoint was ever trained on these games. *WDL* is the
value head's win/draw/loss prediction; a network that always answers one third
for each outcome has a WDL loss of 1.10.

| Checkpoint | Policy top-1 | Policy loss | WDL loss | WDL top-1 | WDL loss, 17 plies or more before the end | WDL top-1, last 16 plies |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.002 | 6.35 | 1.10 | 0.346 | 1.10 | 0.430 |
| 1 | 0.089 | 4.27 | 2.22 | 0.529 | 2.45 | 0.637 |
| 2 | 0.168 | 3.59 | 2.29 | 0.467 | 2.51 | 0.609 |
| 3 | 0.237 | 3.35 | 1.63 | 0.489 | 1.76 | 0.596 |
| 4 | 0.276 | 3.20 | 1.84 | 0.493 | 2.08 | 0.668 |
| 5 | 0.283 | 3.16 | 1.51 | 0.525 | 1.64 | 0.656 |
| 6 | 0.286 | 3.12 | 1.17 | 0.547 | 1.30 | 0.697 |
| 7 | 0.302 | 3.06 | 1.45 | 0.525 | 1.56 | 0.638 |
| 8 | 0.301 | 3.05 | 1.31 | 0.526 | 1.44 | 0.664 |
| 9 | 0.305 | 3.03 | 1.15 | 0.548 | 1.23 | 0.659 |
| 10 | 0.312 | 3.01 | 1.16 | 0.566 | 1.24 | 0.680 |

The policy head improves steadily. The value head does not generalize yet:
its WDL loss on unseen games is still above the 1.10 of a network that knows
nothing, although it names the right outcome more often (0.57 against 0.35).
It is confident and wrong too often. The same loss measured on training
positions is 0.48 at the end of attempt 10, so the network has memorized the
outcome of the games it trained on. The error is concentrated far from the end
of a game, which is 76% of the positions; one ply before the end, generation 10
names the outcome in 90% of the positions. The gap narrows as the buffer grows
(2.45 for generation 1, 1.24 for generation 10, far from the end).

The 3×4 run reached a validation WDL loss of 0.75 by its fifth attempt, with
games about half as long. On this board a game yields more positions that
all share one outcome, and the buffer held 2,560 games after ten attempts.

### Criteria

| Criterion | Result |
| --- | --- |
| 1. Every generation completes, duration recorded | Met: ten attempts, 67 min 43 s. |
| 2. At least half of the first ten attempts promoted | Met: eight of ten. |
| 3. Policy accuracy rises, illegal mass falls across accepted generations | Accuracy: met at every step. Illegal mass: fell overall, with one increase of 0.004. |
| 4. Self-play draws below 20%, mean length well under 256 plies | Met: at most 14.8% draws, mean length between 38 and 87 plies. |
| 5. Latest champion beats the fixed reference and an earlier champion | Met: 99.5% against generation 0, 97.5% against generation 2, 90.0% against generation 4. |

The run continues from generation 10 with the same configuration. Two
measurements decide what follows: the promotion rate, and the WDL loss far
from the end of the game on a fixed validation set. While that loss stays
above 1.10, the value head's confidence cannot be trusted in the positions
that make up most of a game.

## Records

| Files | Contents |
| --- | --- |
| [`protocol.json`](protocol.json) | Configuration and reference identifiers recorded before the run, and the criteria. |
| [`training-observations.json`](training-observations.json) | Every attempt's source, phase durations, self-play outcomes and lengths, final training and validation metrics, arena and probe. |
| [`collect_observations.py`](collect_observations.py) | Builds the file above from the pipeline's reports, the stored replays and the training logs. |
| `arena-10-vs-{0,2,4}.json` | The three independent arenas of generation 10. |
| [`endgame-distance/`](endgame-distance/) | The fixed-set validation of generations 0 to 10, by number of plies before the end of the game. |

Generation 10 is identified by the SHA-256 hashes of its files:
`metadata.json` `195cd169f56815dbaabdf2d4c154268ad86b9ba3bef02226d1b31b1a8bfbb6b9`
and `model.safetensors`
`be44dbbb35050d239abb99fa5c9c308defabc26435daa99fc7ee412af96acb72`.

The independent arenas ran with `compare_saved_generations` at revision
`720f393`, with an inference batch of at most 512 positions as in the
pipeline's arena. That revision wrote a fixed `128` in the report's
`inference_maximum_batch` field; the three reports keep the field as written.

```bash
YOKAI_CANDIDATE_MODELS=models/expert-5x6 YOKAI_CANDIDATE_GENERATION=10 \
YOKAI_REFERENCE_MODELS=models/expert-5x6 YOKAI_REFERENCE_GENERATION=0 \
YOKAI_ARENA_SEED=20261114 YOKAI_ARENA_REPORT=arena-10-vs-0.json \
cargo bench --bench performance -- compare_saved_generations

cargo run --release -- diagnose-endgames --config config/training.toml
```
