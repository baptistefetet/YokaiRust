# Training baseline — 16 November 2026

The published champion is **generation 53**, trained from random
initialization with seed `20261108`. In an independent 200-game arena it
scored **93 wins, 74 losses and 33 draws against generation 46: 54.75% of
points**, with a one-sided paired p-value of **0.097**. That score is above
one half, but it does not establish at the 0.05 level that generation 53 is
stronger than generation 46, the champion published before it. Generation 46
had scored 83.25% against generation 35 in its own independent arena,
generation 35 had scored 72.25% against generation 29, and generation 29 had
won all 200 games against the fixed reference, the random initialization of
the run.

[Release v2.2.0](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.2.0)
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

Up to the published generation, the run made **53 attempts**, including
**27 promotions**, and produced **13,568 self-play games** and **21,200
optimizer updates**. Generation 53 passed its promotion arena against source
generation 49 with 97 wins, 68 losses and 35 draws: **57.25%**,
p = 9.7 × 10⁻³. Its productivity probe drew **5/64 games**, within the 20%
limit. Attempts 54 to 57 were rejected.

In the first thirty attempts, self-play games last 77 to 90 plies on average
from the eleventh attempt on, far from the 256-ply limit. Their draws stay
below 20% in every attempt but one (54 of 256 games in attempt 17).

The [generation results](experiments/2026-11-14/training-observations-30.json)
include every accepted and rejected candidate of those thirty attempts, with
outcomes and phase timings. The experiment report gives the pipeline verdicts
of attempts 31 to 57.

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
| Generation 46 | Generation 35 | 158 / 25 / 17 | 83.25% | 1.3 × 10⁻¹⁷ |
| **Generation 53** | **Generation 46** | **93 / 74 / 33** | **54.75%** | **0.097** |

The base seeds are `20261114`, `20261115`, `20261116`, `20261300` and
`20261500` for generations 10, 29, 35, 46 and 53.

| Generation 53's seat | Wins / losses / draws | Score |
| --- | --- | ---: |
| First | 45 / 44 / 11 | 50.5% |
| Second | 48 / 30 / 22 | 59.0% |

The last arena is the first of the run that does not separate a champion
from its reference. The promotion arenas on the way from generation 46 to
generation 53 gave 62.75% and 57.25%; the
[experiment report](experiments/2026-11-14/README.md#the-published-champion-generation-53)
explains why a promotion arena tends to overstate a gain.

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

Attempts 31 to 35 took 37 min 10 s more, attempts 38 to 47 took 1 h
10 min 45 s and attempts 48 to 57 took 1 h 11 min 53 s. Some phases
overlapped compilation, code checks, browser tests and independent arenas, so
these durations are observations of this run, not a benchmark.

## Verification and artifacts

The required checks passed at the published revision: 99 tests, Clippy with
warnings denied, formatting and documentation generation.

Browser verification ran in Firefox 160 on the site extracted from the
published web archive. The served files were not modified: a test-only hook,
installed through the browser's automation interface, read the game state,
and every action was a touch event. On each module, WebGPU in a window and
Flex headless, it played a one-player game against the champion to its end,
started a new game from the result panel and reloaded the page, which kept
the chosen number of players. On Flex it also played 30 plies of a two-player
game opened from the logo. The games included captures, drops, hands holding
several pieces of one kind and moves chosen with and without promotion; the
Flex games ran in a 390×844 mobile viewport. In that viewport a touch on the
Rules button opened the rulebook and a touch on the version opened the source
repository; the layout was inspected at four sizes from 320×568 to 1280×800.
The site produced no page error and no failed request. Both archives pass
SHA-256 verification.

- `models/expert-5x6/`: the initial network and every candidate, with Adam
  states for trained candidates; `latest` selects the accepted generation.
- `data/expert-5x6/`: self-play games, replay buffer and generation reports.
- [Release v2.2.0](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.2.0):
  deployable website and minimal checkpoint of generation 53 as separate
  archives.

## Limits of the measurements

Every opponent is an earlier generation of the same run: the arenas establish
progress at the stated search budget, not strength against people or another
program. Generation 53 was measured against generation 46 only, in one arena
of 200 games, which cannot tell a gain of a few points of score from no gain.

The independent arenas of generations 46 and 53 ran with a search that always
plays an action winning at once, as did the self-play and the promotion
arenas from attempt 48 on. The earlier independent arenas and the promotion
arenas up to attempt 47 chose the most visited action even then, so their
figures are not strictly comparable with the later ones.

The base seeds of generations 10, 29 and 35 are consecutive and pair i uses
base seed + i, so the seed ranges of those three comparisons overlap instead
of being disjoint. The seed ranges of generations 46 and 53 are separate from
them and from each other.

One training seed does not establish learning efficiency across seeds or
equal-compute runs. No fixed-workload cost measurement exists for this board.
