# Training baseline — 15 November 2026

The published champion is **generation 46**, trained from random
initialization with seed `20261108`. In an independent 200-game arena it
scored **158 wins, 25 losses and 17 draws against generation 35: 83.25% of
points**, with a one-sided paired p-value of **1.3 × 10⁻¹⁷**. Generation 35 had
scored 72.25% against generation 29 in its own independent arena, and
generation 29 had won all 200 games against the fixed reference, the random
initialization of the run.

[Release v2.1.0](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.1.0)
carries this checkpoint and the web site that plays it. Training continues in
the same lineage: `latest`, under the model path of `config/training.toml`,
selects the newest accepted generation for play, training and browser export.
The [AlphaZero guide](alphazero-guide.md) explains the learning method; the
[experiment records](experiments/2026-11-14/README.md) contain the protocol,
full measurements and reproduction commands.

## Training configuration

Training starts with empty model and data directories, random weights and no
imported optimizer state or replay. The configuration is
[`config/training.toml`](../config/training.toml); the
[protocol](experiments/2026-11-14/protocol.json) records its hash and the
source revision.

The network has 64 channels, four residual blocks and 64-unit shared/value
layers, about 1.17 million parameters. It uses model format 5, encoder
version 5 and rules version 3.

| Phase per attempt | Budget |
| --- | --- |
| Self-play | 256 games, 200 simulations per regular move |
| Visited-state restarts | 25% when an archive exists, 800 simulations per move |
| Optimization | 400 Adam updates, batch size 256 |
| Promotion arena | 200 games, 400 simulations per move |
| Noisy productivity probe | 64 games, at most 20% draws |

Promotion requires at least 55% of points, a one-sided paired p-value at most
0.05 and a passing productivity probe. A draw earns half a point; score is not
win rate. Rejected candidates never supply subsequent weights, Adam state or
self-play. Their attempt's games still enrich the replay buffer.

The learning rate stays at 0.001. Decisive-tail oversampling applies to the
first ten attempts. Mirrored positions augment training examples; restarts
inherit their root trajectory's train/validation family. A draw is worth zero
to both players in search, and both players' moves are imitated in drawn
games.

## Observed training results

Up to the published generation, the run made **46 attempts**, including
**25 promotions**, and produced **11,776 self-play games** and **18,400
optimizer updates**. Generation 46 passed its promotion arena against source
generation 45 with 91 wins, 57 losses and 52 draws: **58.5%**,
p = 2.7 × 10⁻³. Its productivity probe drew **6/64 games**, within the 20%
limit. A forty-seventh attempt was rejected.

In the first thirty attempts, self-play games last 77 to 90 plies on average
from the eleventh attempt on, far from the 256-ply limit. Their draws stay
below 20% in every attempt but one (54 of 256 games in attempt 17).

The [generation results](experiments/2026-11-14/training-observations-30.json)
include every accepted and rejected candidate of those thirty attempts, with
outcomes and phase timings. The experiment report gives the pipeline verdicts
of attempts 31 to 47.

## Independent strength evaluation

The fixed reference is generation 0 of the run, identified by its file hashes
in the [protocol](experiments/2026-11-14/protocol.json). Each arena plays
200 games at 400 simulations per move, with eight leaves per inference, no
root noise and random legal openings of 0–4 plies. Each pair shares an opening
and swaps colors. The seeds are not used by the training run, and the
references were chosen before each arena ran.

| Candidate | Reference | Wins / losses / draws | Score | Paired p |
| --- | --- | --- | ---: | ---: |
| Generation 10 | Generation 0 (fixed reference) | 198 / 0 / 2 | 99.50% | 7.9 × 10⁻³¹ |
| Generation 10 | Generation 2 | 191 / 1 / 8 | 97.50% | 1.6 × 10⁻³⁰ |
| Generation 10 | Generation 4 | 166 / 6 / 28 | 90.00% | 5.0 × 10⁻²⁹ |
| Generation 29 | Generation 0 (fixed reference) | 200 / 0 / 0 | 100.00% | 7.9 × 10⁻³¹ |
| Generation 29 | Generation 10 | 197 / 2 / 1 | 98.75% | 3.2 × 10⁻³⁰ |
| Generation 29 | Generation 20 | 185 / 5 / 10 | 95.00% | 2.5 × 10⁻²⁹ |
| Generation 35 | Generation 29 | 134 / 45 / 21 | 72.25% | 6.8 × 10⁻¹⁰ |
| **Generation 46** | **Generation 35** | **158 / 25 / 17** | **83.25%** | **1.3 × 10⁻¹⁷** |

The base seeds are `20261114`, `20261115`, `20261116` and `20261300` for
generations 10, 29, 35 and 46.

| Generation 46's seat | Wins / losses / draws | Score |
| --- | --- | ---: |
| First | 77 / 16 / 7 | 80.5% |
| Second | 81 / 9 / 10 | 86.0% |

## What the network predicts

Both heads are measured on games the network never trained on, for the
generations up to the thirtieth attempt. The policy head improves at every
accepted generation: it names the move preferred by the search in 41% of the
positions at generation 29, against 33% at generation 10.

The value head is the weak part. In the last 16 plies of a game it names the
outcome of 76% of the positions. Farther from the end, which is 78% of the
positions, its loss stays between 1.13 and 1.19, above the 1.10 of a
prediction that gives each outcome one chance in three. The champion's
strength therefore rests on its policy and on search more than on its
evaluation of middle-game positions.

## Execution time

The first thirty attempts took **3 h 36 min 49 s** of pipeline time, 7 min
14 s each on average, excluding compilation, pauses between commands and
independent arenas. Timings use whole-second CLI timestamps on an Apple M4 Max
with the default release profile.

| Phase | Total |
| --- | ---: |
| Self-play, including restarts | 2 h 05 min 32 s |
| Optimization and validation | 5 min 42 s |
| Promotion arenas | 58 min 46 s |
| Noisy productivity probes | 26 min 44 s |
| Other transitions | 5 s |
| **Total** | **3 h 36 min 49 s** |

Attempts 31 to 35 took 37 min 10 s more, and attempts 38 to 47 took 1 h
10 min 45 s. Some phases overlapped compilation, code checks and independent
arenas, so these durations are observations of this run, not a benchmark.

## Verification and artifacts

The required checks passed at the published revision: 99 tests, Clippy with
warnings denied, formatting and documentation generation.

Browser verification ran in Firefox on the site extracted from the published
web archive, with one test-only script added to its page to read the game
state. Through pointer events it played a one-player game against the champion
to its end on each module, WebGPU in a window and Flex headless, and 24 plies
of a two-player game on Flex, with captures, without page errors or failed
requests. The page showed the release version and the source link. Drops,
promotion choices, a reload and a mobile viewport were not exercised for this
release. Both archives pass SHA-256 verification.

- `models/expert-5x6/`: the initial network and every candidate, with Adam
  states for trained candidates; `latest` selects the accepted generation.
- `data/expert-5x6/`: self-play games, replay buffer and generation reports.
- [Release v2.1.0](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.1.0):
  deployable website and minimal checkpoint of generation 46 as separate
  archives.

## Limits of the measurements

Every opponent is an earlier generation of the same run: the arenas establish
progress at the stated search budget, not strength against people or another
program. Generation 46 was measured against generation 35 only, in one arena.

That arena ran with a search that always plays an action winning at once. The
earlier independent arenas and every promotion arena of the run chose the
most visited action even then, so their figures are not strictly comparable
with it.

The base seeds of generations 10, 29 and 35 are consecutive and pair i uses
base seed + i, so the seed ranges of those three comparisons overlap instead
of being disjoint. The seed range of generation 46 is separate from them.

One training seed does not establish learning efficiency across seeds or
equal-compute runs. No fixed-workload cost measurement exists for this board.
