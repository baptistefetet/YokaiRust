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

The run continued from generation 10 with the same configuration.

## Observations: attempts 11 to 30

Twenty more attempts took 2 h 29 min, 7 min 27 s each on average. Ten
candidates were promoted; the accepted champion after the thirtieth attempt is
**generation 29**. The whole run has used 3 h 36 min 49 s.

| Attempt | Source | Arena W–L–D | Score | Paired p | Probe draws | Verdict | Mean plies | Move-limit draws | Repetition draws |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | 10 | 90–86–24 | 51.00% | 0.41 | 3/64 | rejected | 77.9 | 6 | 16 |
| 12 | 10 | 107–66–27 | 60.25% | 2.2 × 10⁻³ | 16/64 | rejected (draw gate) | 88.5 | 17 | 22 |
| 13 | 10 | 114–62–24 | 63.00% | 1.1 × 10⁻⁵ | 11/64 | promoted | 82.4 | 15 | 26 |
| 14 | 13 | 82–96–22 | 46.50% | 0.86 | 8/64 | rejected | 83.6 | 16 | 17 |
| 15 | 13 | 79–105–16 | 43.50% | 0.99 | 7/64 | rejected | 85.6 | 8 | 26 |
| 16 | 13 | 113–55–32 | 64.50% | 3.0 × 10⁻⁷ | 11/64 | promoted | 80.2 | 12 | 20 |
| 17 | 16 | 97–65–38 | 58.00% | 7.7 × 10⁻³ | 5/64 | promoted | 90.1 | 13 | 41 |
| 18 | 17 | 97–61–42 | 59.00% | 8.4 × 10⁻⁴ | 13/64 | rejected (draw gate) | 76.6 | 8 | 27 |
| 19 | 17 | 54–98–48 | 39.00% | 1.0 | 3/64 | rejected | 85.6 | 15 | 24 |
| 20 | 17 | 107–64–29 | 60.75% | 1.0 × 10⁻⁴ | 5/64 | promoted | 87.3 | 13 | 18 |
| 21 | 20 | 125–49–26 | 69.00% | 1.2 × 10⁻⁷ | 8/64 | promoted | 78.8 | 10 | 19 |
| 22 | 21 | 62–118–20 | 36.00% | 1.0 | 4/64 | rejected | 83.3 | 12 | 20 |
| 23 | 21 | 119–73–8 | 61.50% | 4.9 × 10⁻⁵ | 5/64 | promoted | 78.6 | 7 | 22 |
| 24 | 23 | 56–117–27 | 34.75% | 1.0 | 10/64 | rejected | 78.6 | 6 | 28 |
| 25 | 23 | 98–83–19 | 53.75% | 0.086 | 8/64 | rejected | 77.6 | 4 | 24 |
| 26 | 23 | 105–67–28 | 59.50% | 8.0 × 10⁻⁴ | 8/64 | promoted | 84.8 | 13 | 21 |
| 27 | 26 | 111–70–19 | 60.25% | 5.4 × 10⁻⁴ | 10/64 | promoted | 87.0 | 13 | 24 |
| 28 | 27 | 108–74–18 | 58.50% | 4.5 × 10⁻⁴ | 3/64 | promoted | 85.9 | 12 | 24 |
| 29 | 28 | 130–57–13 | 68.25% | 9.1 × 10⁻⁷ | 6/64 | promoted | 80.7 | 3 | 23 |
| 30 | 29 | 83–84–33 | 49.75% | 0.57 | 7/64 | rejected | 84.8 | 17 | 25 |

Three things differ from the first ten attempts:

- **Half of the candidates are promoted**, against eight in ten before, and
  the accepted ones score between 58% and 69%.
- **Some candidates are much weaker than their source.** Attempts 19, 22 and
  24 scored between 35% and 39%. Each next attempt started again from the
  accepted champion and two of the three following candidates were promoted.
- **The draw gate rejected two candidates that had won their arena**
  (attempts 12 and 18, with 25.0% and 20.3% of probe games drawn). Self-play
  draws stayed between 22 and 42 games out of 256, except 54 games (21.1%) in
  attempt 17. Repetition is now the usual cause; games that reach the 256-ply
  limit number 3 to 17 per attempt. Mean game length stays between 77 and 90
  plies.

### Independent arenas of generation 29

Three arenas of 200 games at 400 simulations per move, with seed `20261115`
and references chosen before they ran.

| Reference | W–L–D | Score | Paired p |
| --- | --- | --- | --- |
| Generation 0 (fixed reference) | 200–0–0 | 100.00% | 7.9 × 10⁻³¹ |
| Generation 10 | 197–2–1 | 98.75% | 3.2 × 10⁻³⁰ |
| Generation 20 | 185–5–10 | 95.00% | 2.5 × 10⁻²⁹ |

Generation 20 itself is ten attempts and three promotions older than
generation 29, so the run is still gaining strength at this point.

### Validation on a fixed set of games

After the thirtieth attempt the buffer holds the 5,120 games of the last
twenty attempts. `yokai diagnose-endgames` evaluated the accepted generations
on its 525 validation games (36,937 positions, 72 of the games drawn). These
are not the games of the table for the first ten attempts, so the two tables
cannot be compared line by line.

| Checkpoint | Policy top-1 | Policy loss | WDL loss | WDL top-1 | WDL loss, 17 plies or more before the end | WDL top-1, last 16 plies |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.002 | 6.35 | 1.10 | 0.363 | 1.10 | 0.422 |
| 10 | 0.328 | 2.98 | 1.24 | 0.530 | 1.35 | 0.697 |
| 13 | 0.344 | 2.91 | 1.10 | 0.538 | 1.21 | 0.726 |
| 16 | 0.358 | 2.87 | 1.04 | 0.551 | 1.16 | 0.744 |
| 17 | 0.363 | 2.84 | 1.06 | 0.537 | 1.16 | 0.723 |
| 20 | 0.369 | 2.81 | 1.07 | 0.553 | 1.17 | 0.720 |
| 21 | 0.384 | 2.78 | 1.06 | 0.551 | 1.19 | 0.748 |
| 23 | 0.392 | 2.75 | 1.02 | 0.562 | 1.13 | 0.759 |
| 26 | 0.399 | 2.74 | 1.01 | 0.569 | 1.14 | 0.771 |
| 27 | 0.400 | 2.72 | 1.03 | 0.569 | 1.15 | 0.780 |
| 28 | 0.406 | 2.71 | 1.04 | 0.570 | 1.14 | 0.750 |
| 29 | 0.409 | 2.70 | 1.07 | 0.559 | 1.19 | 0.759 |

The policy head still improves at every accepted generation. The value head's
WDL loss over all positions has fallen below the 1.10 of an uninformed
prediction, and it names the outcome of 76% of the positions in the last 16
plies of a game. Far from the end, 78% of the positions, its loss is between
1.13 and 1.19: it has stopped falling since generation 16 and remains above
1.10. The loss on training positions rose from 0.48 to 0.66 over the same
attempts, so the network memorizes less than it did. In a middle-game
position of an unseen game, though, its probabilities are still less reliable
than giving each outcome one chance in three.

The run continues from generation 29 with the same configuration. If the
value head stays where it is, the next change to try is the amount of
self-play data per attempt rather than the network.

## Attempts 31 to 35

The run continued from generation 29 with the same configuration. Its next
five attempts promoted generations 32, 34 and 35; release `web-v2.0.0`
published generation 35.

| Attempt | Source | Arena W–L–D | Score | Paired p | Probe draws | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 31 | 29 | 74–81–45 | 48.25% | 0.74 | 7.8% | rejected |
| 32 | 29 | 94–60–46 | 58.5% | 2.9 × 10⁻³ | 12.5% | promoted |
| 33 | 32 | 81–86–33 | 48.75% | 0.66 | 6.2% | rejected |
| 34 | 32 | 124–50–26 | 68.5% | 5.0 × 10⁻⁸ | 7.8% | promoted |
| 35 | 34 | 123–60–17 | 65.75% | 7.2 × 10⁻⁶ | 10.9% | promoted |

These figures come from the training log; the detailed observations of the
attempts after the thirtieth are not collected yet.

In an independent arena of 200 games at 400 simulations with seed `20261116`,
generation 35 scored **134 wins, 45 losses and 21 draws against generation
29: 72.25% of points**, with a one-sided paired p-value of 6.8 × 10⁻¹⁰. Both
color assignments favor it: 63–27–10 as First and 71–18–11 as Second. The
arena ran while the training run was using the same machine.

## The published champion: generation 46

The run continued from generation 35 with the same configuration. Its next
twelve attempts promoted generations 38, 41, 45 and 46; release `web-v2.1.0`
publishes **generation 46**.

| Attempt | Source | Arena W–L–D | Score | Paired p | Probe draws | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 36 | 35 | 97–84–19 | 53.25% | 0.13 | 3.1% | rejected |
| 37 | 35 | 68–90–42 | 44.5% | 0.97 | 12.5% | rejected |
| 38 | 35 | 136–43–21 | 73.25% | 1.7 × 10⁻¹⁰ | 10.9% | promoted |
| 39 | 38 | 92–84–24 | 52.0% | 0.30 | 12.5% | rejected |
| 40 | 38 | 53–94–53 | 39.75% | 1.00 | 9.4% | rejected |
| 41 | 38 | 104–79–17 | 56.25% | 0.021 | 7.8% | promoted |
| 42 | 41 | 63–95–42 | 42.0% | 1.00 | 7.8% | rejected |
| 43 | 41 | 97–78–25 | 54.75% | 0.062 | 7.8% | rejected |
| 44 | 41 | 90–90–20 | 50.0% | 0.54 | 17.2% | rejected |
| 45 | 41 | 116–68–16 | 62.0% | 2.0 × 10⁻⁵ | 7.8% | promoted |
| 46 | 45 | 91–57–52 | 58.5% | 2.7 × 10⁻³ | 9.4% | promoted |
| 47 | 46 | 85–71–44 | 53.5% | 0.14 | 9.4% | rejected |

These figures come from the pipeline's generation reports. Attempts 38 to 47
ran in one command and took 1 h 10 min 45 s.

The reference of the independent arena was chosen before it ran: generation
35, the champion published until then. In 200 games at 400 simulations with
seed `20261300`, generation 46 scored **158 wins, 25 losses and 17 draws:
83.25% of points**, with a one-sided paired p-value of 1.3 × 10⁻¹⁷. Both color
assignments favor it: 77–16–7 as First and 81–9–10 as Second.

This arena ran at revision `31b9500`, whose search always plays an action
that wins at once. The pipeline's arenas above and the earlier independent
arenas ran before that rule, when the search played the most visited action
even if another one won the game.

## Records

| Files | Contents |
| --- | --- |
| [`protocol.json`](protocol.json) | Configuration and reference identifiers recorded before the run, and the criteria. |
| [`training-observations.json`](training-observations.json), [`training-observations-30.json`](training-observations-30.json) | Every attempt's source, phase durations, self-play outcomes and lengths, final training and validation metrics, arena and probe: the first ten attempts, then all thirty. |
| [`collect_observations.py`](collect_observations.py) | Builds the file above from the pipeline's reports, the stored replays and the training logs. |
| `arena-10-vs-{0,2,4}.json`, `arena-29-vs-{0,10,20}.json`, [`arena-35-vs-29.json`](arena-35-vs-29.json), [`arena-46-vs-35.json`](arena-46-vs-35.json) | The independent arenas of generations 10, 29, 35 and 46. |
| [`endgame-distance/`](endgame-distance/) | The fixed-set validation of generations 0 to 10 after the tenth attempt, by number of plies before the end of the game. |
| [`endgame-distance-30/`](endgame-distance-30/) | The same validation of generations 0 to 30 after the thirtieth attempt. |

The champions are identified by the SHA-256 hashes of their files.

| Generation | `metadata.json` | `model.safetensors` |
| --- | --- | --- |
| 10 | `195cd169f56815dbaabdf2d4c154268ad86b9ba3bef02226d1b31b1a8bfbb6b9` | `be44dbbb35050d239abb99fa5c9c308defabc26435daa99fc7ee412af96acb72` |
| 29 | `763cf4de87f3d246b218633e9c675ca77c6d72a5f3d1e5797fdde0ea7acf9ab4` | `f8d3b5d726f21feb8853db3ffaedcb207ec76baed78d98b96a1f88d13c51ea06` |
| 35 | `b3b40b147a59213fdc4d06ddc444af95c432ae2e7a780392649b0dcdb8d6f4c4` | `4c8a03a996c426041a7ab8bc3b01ebf8ace6d907aab742eb5a7511d33d70769f` |
| 46 | `8663f79eb95b81df0b546030623600cd4b1923ae6650400358ddbb1ec5b1c032` | `af18f5d3c236e105bc7ab8a69db50fb5c2fcfc11b761d03fbb1e7d29910130ca` |

The independent arenas ran with `compare_saved_generations`, with an inference
batch of at most 512 positions as in the pipeline's arena. Revision `720f393`,
used for generation 10, wrote a fixed `128` in the report's
`inference_maximum_batch` field; those three reports keep the field as written.

```bash
YOKAI_CANDIDATE_MODELS=models/expert-5x6 YOKAI_CANDIDATE_GENERATION=10 \
YOKAI_REFERENCE_MODELS=models/expert-5x6 YOKAI_REFERENCE_GENERATION=0 \
YOKAI_ARENA_SEED=20261114 YOKAI_ARENA_REPORT=arena-10-vs-0.json \
cargo bench --bench performance -- compare_saved_generations

cargo run --release -- diagnose-endgames --config config/training.toml
```
