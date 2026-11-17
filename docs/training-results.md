# Training baseline — 17 November 2026

The published champion is **generation 64**. It beat the previous web champion,
generation 53, with **227 wins, 150 losses and 23 draws** in a preplanned
400-game match. Its score was **238.5 points (59.625%)**, with a one-sided
paired p-value of **4.99 × 10⁻⁶**. It scored 62% as First and 57.25% as Second.
[Independent match and protocol](experiments/2026-11-17/README.md)

The run started from random weights with seed `20261108`. Generation 64
descends from generations 53, 59 and 62. The web release includes the static
site and a separate native checkpoint. [Release v2.3.1](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.3.1)

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
to both players in search. Through attempt 57, the policy target for the player
who moved second in a drawn game discarded immediate third repetitions when
other moves had received visits, even with a policy weight of 1.0. From attempt 58 on,
that weight preserves the full visit distribution for both players.

## Observed training results

Through attempt 64, the run completed **64 attempts**, **30 promotions**,
**16,384 self-play games** and **25,600 optimizer updates**. The seven attempts
after attempt 57 promoted generations 59, 62 and 64. Generation 64 earned
68.25% against its source, generation 62, with 129 wins, 56 losses and 15
draws. Its productivity probe drew 7 of 64 games. [All seven attempt reports](experiments/2026-11-17/README.md#training-outcomes)

In the first thirty attempts, self-play games last 77 to 90 plies on average
from the eleventh attempt on, far from the 256-ply limit. Their draws stay
below 20% in every attempt but one (54 of 256 games in attempt 17).

The [generation results](experiments/2026-11-14/training-observations-30.json)
include every accepted and rejected candidate of those thirty attempts, with
outcomes and phase timings. The experiment report gives the pipeline verdicts
of attempts 31 to 57.

## Independent strength evaluation

The fixed reference is generation 0 of the run, identified by its file hashes
in the [protocol](experiments/2026-11-14/protocol.json). The arenas through generation 53 used
200 games. Generation 64 played 400 games against the published reference.
All these matches use 400 simulations per move, with eight leaves per inference, no
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
| Generation 46 | Generation 35 | 158 / 25 / 17 | 83.25% | 1.3 × 10⁻¹⁷ |
| Generation 53 | Generation 46 | 93 / 74 / 33 | 54.75% | 0.097 |
| **Generation 64** | **Generation 53** | **227 / 150 / 23** | **59.625%** | **4.99 × 10⁻⁶** |

The base seeds were `20261114`, `20261115`, `20261116`, `20261300`,
`20261500` and `20261750` for generations 10, 29, 35, 46, 53 and 64.

Generation 53 scored above half against generation 46, with p = 0.097.
That comparison remains inconclusive at the 0.05 level. [Generation 53 report](experiments/2026-11-14/README.md#the-published-champion-generation-53)

## What the network predicts

Both heads are measured on games the network never trained on, for the
generations up to the thirtieth attempt. The policy head improves at every
accepted generation: it names the move preferred by the search in 41% of the
positions at generation 29, against 33% at generation 10.

The value head is the weak part. In the last 16 plies of a game it names the
outcome of 76% of the positions. Farther from the end, which is 78% of the
positions, its loss stays between 1.13 and 1.19, above the 1.10 of a
prediction that gives each outcome one chance in three. These measurements cover checkpoints through attempt 30.

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

Attempts 31 to 35 took 37 min 10 s more, attempts 38 to 47 took 1 h
10 min 45 s and attempts 48 to 57 took 1 h 11 min 53 s. Some phases
overlapped compilation, code checks, browser tests and independent arenas, so
these durations are observations of this run, not a benchmark.

## Verification and artifacts

For v2.3.0 the four required Rust checks passed, including 99 tests. The Metal diagnostic
restored the training state and ran an optimizer update. Binaryen optimized both
WebAssembly modules. All six Playwright scenarios passed on the site extracted
from the release archive using Chromium with desktop WebGPU and mobile Flex.
They exercised loading,
complete games against the champion, two-player games, promotion choices,
captures, drops and restarting from the result panel.

- `models/expert-5x6/` holds every checkpoint and its Adam state for trained
  candidates. The `latest` pointer selects the accepted champion.
- `data/expert-5x6/` holds self-play games, the replay buffer and generation reports.
- The release provides both archives and their SHA-256 checksums.
  [Release v2.3.0](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.3.0)

## Limits of the measurements

Every opponent is an earlier generation of the same run: the arenas establish
progress at the stated search budget, not strength against people or another
program. The continuation combines seven attempts with the draw-target
correction. The match measures the resulting champion. Isolating the effect of
the correction would require a comparable continuation using the earlier target.

The independent arenas of generations 46, 53 and 64 ran with a search that always
plays an action winning at once, as did the self-play and the promotion
arenas from attempt 48 on. The earlier independent arenas and the promotion
arenas up to attempt 47 chose the most visited action even then, so their
figures are not strictly comparable with the later ones.

The base seeds of generations 10, 29 and 35 are consecutive and pair i uses
base seed + i, so the seed ranges of those three comparisons overlap instead
of being disjoint. The seed ranges of generations 46, 53 and 64 are separate from
them and from each other.

One training seed does not establish learning efficiency across seeds or
equal-compute runs. No fixed-workload cost measurement exists for this board.
