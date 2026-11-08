# Training baseline — 8 November 2026

The accepted champion is **generation 30**, trained from random initialization
with seed `20261108`. In an independent 800-game arena against the fixed
reference, it scored **313 wins, 162 losses and 325 draws: 59.4375% of points**.
The one-sided paired p-value is **3.54 × 10⁻¹⁵**. Both color assignments favor
the champion.

`config/training.toml` selects this checkpoint for play, training and browser
export. The [AlphaZero guide](alphazero-guide.md) explains the learning method;
the [experiment records](experiments/2026-11-08/README.md) contain the frozen
configuration, full measurements and reproduction tools.

## Training configuration

Training starts with empty model/data directories, random weights and no
imported optimizer state or replay. The measured training/search revision is
`f3ce72638d8e72bd645f525baa132413315969e5`; the configuration is frozen in
[`config/retrain-20261108.toml`](../config/retrain-20261108.toml).

The network has 64 channels, four residual blocks and 64-unit shared/value
layers. It uses model format 5, encoder version 4 and rules version 1.

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

The learning rate is 0.001 until the accepted source reaches generation 7,
then 0.00025. Decisive-tail oversampling applies to the first ten attempts.
Mirrored positions augment training examples; restarts inherit their root
trajectory's train/validation family.

## Observed training results

The campaign completed **30 attempts**, including **14 promotions**, and
produced **7,680 self-play games** and **12,000 optimizer updates**. Generation
30 passed its promotion arena against source generation 26 with 63 wins,
40 losses and 97 draws: **55.75%**, p = 0.0002098. Its productivity probe drew
**7/64 games**, within the 20% limit.

The [complete generation results](experiments/2026-11-08/training-observations.json)
include every accepted and rejected candidate, outcomes and phase timings.
The [data audit](experiments/2026-11-08/data-audit.json) confirms that all replay
records exist, generation/seed identities are unique, and all 1,856 restarted
games have their root ancestry in the retained trajectories.

## Independent strength evaluation

The fixed reference is generation 21 in the separate reference lineage,
identified by its file hashes in the [protocol](experiments/2026-11-08/protocol.json).
Generation numbers identify attempts within a lineage, not comparable strength
levels. Reference weights and Adam state are retained independently of training.

Each arena uses 800 games, 400 simulations per move, exact sequential PUCT
(one leaf per request), 128 game workers, no root noise and random legal
openings of 0–4 plies. Each pair shares an opening and swaps candidate colors.
Inference services collect batches of 64–128 requests with a 1 ms wait.

Confirmation requires at least 55% of points and p ≤ 0.05/3, accounting for
up to three comparisons. Candidates are selected by internal promotion before
their independent arena. The campaign contains two independent measurements:

| Candidate vs reference | Wins / losses / draws | Score | Paired p | Result |
| --- | --- | ---: | ---: | --- |
| Generation 26 | 232 / 386 / 182 | 40.375% | ≈ 1 | Failed |
| **Generation 30** | **313 / 162 / 325** | **59.4375%** | **3.54 × 10⁻¹⁵** | **Confirmed** |

Their base seeds are `2026110801` and `2026111801`. Pair i uses base seed + i,
so the two 400-seed ranges are disjoint. The arenas contain 169 and 166 distinct
openings and took 392.00 and 430.21 seconds respectively.

| Champion's seat | Wins / losses / draws | Score |
| --- | --- | ---: |
| First | 156 / 86 / 158 | 58.750% |
| Second | 157 / 76 / 167 | 60.125% |

The champion's paired score counts for 0, 0.5, 1, 1.5 and 2 points are
`[9, 34, 219, 73, 65]`. Full results, checkpoint identities and run plans are
available for [generation 26](experiments/2026-11-08/confirmation-26.json) and
[generation 30](experiments/2026-11-08/confirmation-30.json).

## Execution time

The pipeline took **2 h 25 min 47 s**, excluding compilation, pauses between
commands, independent arenas and performance tests. Timings use whole-second
CLI timestamps on an Apple M4 Max with Rust 1.97.1 and the default release profile.

| Phase | Total |
| --- | ---: |
| Self-play, including restarts | 55 min 56 s |
| Optimization and validation | 2 min 29 s |
| Promotion arenas | 1 h 16 min 12 s |
| Noisy productivity probes | 10 min 39 s |
| Other transitions | 31 s |
| **Total** | **2 h 25 min 47 s** |

Some training phases overlapped compilation and code checks. These durations
are observations of this run; the separate
[performance measurements](performance-results.md) use interleaved runs without
concurrent training.

## Verification and artifacts

The required checks passed: 86 tests, Clippy with warnings denied, formatting
and documentation generation. Two Metal tests are normally ignored; the
training-state round-trip diagnostic was also run and passed.

Browser verification completed three WebGPU games and two Flex games, including
28 captures, 20 drops and resets. A human move with an AI reply was checked
through the interface, and desktop/mobile layouts were inspected. Both release
archives pass SHA-256 and byte-for-byte verification. See the
[verification record](experiments/2026-11-08/verification.json).

- `models/alpha-zero-from-scratch-20261108/`: the initial network and all 30
  candidates, with Adam states for trained candidates; `latest` selects 30.
- `data/alpha-zero-from-scratch-20261108/`: self-play, replay buffer, reports,
  process logs, independent arenas and performance fixtures.
- `models/references/pre-cleanup-20261108/`: fixed reference and optimizer state.
- [Release v1.0.11](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v1.0.11):
  deployable website and minimal accepted checkpoint as separate archives.

## Limits of the measurements

This establishes playing strength against the fixed opponent at the stated
search budget. One training seed does not establish learning efficiency across
seeds or equal-compute runs. Execution costs and their measurement limits are
reported in [performance results](performance-results.md).
