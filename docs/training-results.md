# Training results — 8 November 2026

The accepted champion is **generation 30 of the new, freshly initialized
lineage**. It beats the October reference, generation 21, in an independent
800-game arena: **313 wins, 162 losses and 325 draws**, or **59.4375% of points**.
The one-sided paired p-value is **3.54 × 10⁻¹⁵**, below the conservative
three-comparison threshold of 0.05/3. Both color assignments favor the new model.

`config/training.toml` now selects this lineage for play, training and browser
export. Generation numbers identify attempts within a lineage; they are not
strength ratings and cannot be compared across lineages.

The [AlphaZero guide](alphazero-guide.md) explains the algorithm. This page
contains only this experiment's observations; its
[protocol and amendments](experiments/2026-11-08/protocol.json),
[training observations](experiments/2026-11-08/training-observations.json) and
[reproduction instructions](experiments/2026-11-08/README.md) retain provenance.

## Initialization and controls

The run began with empty model/data directories, random weights seeded with
`20261108`, and no imported optimizer state or replay. The training/search code
is `f3ce72638d8e72bd645f525baa132413315969e5`. No rules, architecture, losses,
search budgets or optimizer settings were changed during this run; only the
seed and storage paths differ from the starting configuration.

The network has 64 channels, four residual blocks and 64-unit shared/value
layers. Model format 5, encoder version 4 and rules version 1 remain unchanged.
The fixed reference was copied before training to
`models/references/pre-cleanup-20261108/generation-000021/`, including its Adam
state. Its weights SHA-256 is
`9a0e348b484e09ba73cbc3f10d0f3441da50eae93b7dce3cd0da6595d546bdee`;
all reference hashes are recorded in the protocol and rechecked before arenas.

| Phase per attempt | Budget |
| --- | --- |
| Self-play | 256 games, 200 simulations per regular move |
| Visited-state restarts | 25% when an archive exists, 800 simulations per move |
| Optimization | 400 Adam updates, batch size 256 |
| Promotion arena | 200 games, 400 simulations per move |
| Noisy productivity probe | 64 games, at most 20% draws |

Promotion requires at least 55% of points **and** a one-sided paired p-value
at most 0.05 **and** a passing productivity probe. A draw earns half a point;
a score is not a win rate. Rejected candidates never supply subsequent weights,
Adam state or self-play. Their attempt's games still enrich the replay buffer.

The learning rate is 0.001 until the accepted source reaches generation 7,
then 0.00025. Early decisive-tail oversampling stops at attempt 11. Mirrored
positions augment training examples; restarts inherit their root trajectory's
train/validation family. The [data audit](experiments/2026-11-08/data-audit.json)
found all 7,680 replay records, unique generation/seed identities, and complete
root ancestry for all 1,856 restarted games.

## Results by attempt

Wins, losses and draws below are from the candidate's perspective against its
accepted source. Every decision, including rejected improvements, is retained.

| Candidate | Source | Wins / losses / draws | Score | Paired p | Probe draws | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | 0 | 179 / 21 / 0 | 89.50% | 1.65e-24 | 2/64 | Accepted |
| 2 | 1 | 175 / 23 / 2 | 88.00% | 6.62e-24 | 1/64 | Accepted |
| 3 | 2 | 127 / 72 / 1 | 63.75% | 2.13e-06 | 2/64 | Accepted |
| 4 | 3 | 90 / 84 / 26 | 51.50% | 0.366 | 4/64 | Rejected: arena |
| 5 | 3 | 161 / 37 / 2 | 81.00% | 1.59e-17 | 1/64 | Accepted |
| 6 | 5 | 134 / 55 / 11 | 69.75% | 6.4e-08 | 3/64 | Accepted |
| 7 | 6 | 143 / 49 / 8 | 73.50% | 2e-15 | 4/64 | Accepted |
| 8 | 7 | 155 / 26 / 19 | 82.25% | 5.29e-22 | 4/64 | Accepted |
| 9 | 8 | 80 / 76 / 44 | 51.00% | 0.406 | 5/64 | Rejected: arena |
| 10 | 8 | 87 / 56 / 57 | 57.75% | 0.00164 | 0/64 | Accepted |
| 11 | 10 | 66 / 57 / 77 | 52.25% | 0.15 | 5/64 | Rejected: arena |
| 12 | 10 | 101 / 30 / 69 | 67.75% | 4.73e-12 | 4/64 | Accepted |
| 13 | 12 | 90 / 66 / 44 | 56.00% | 0.0282 | 6/64 | Accepted |
| 14 | 13 | 45 / 84 / 71 | 40.25% | 1 | 11/64 | Rejected: arena |
| 15 | 13 | 111 / 32 / 57 | 69.75% | 1.68e-17 | 9/64 | Accepted |
| 16 | 15 | 43 / 27 / 130 | 54.00% | 0.00369 | 6/64 | Rejected: arena |
| 17 | 15 | 40 / 40 / 120 | 50.00% | 0.547 | 17/64 | Rejected: both gates |
| 18 | 15 | 39 / 29 / 132 | 52.50% | 0.0312 | 7/64 | Rejected: arena |
| 19 | 15 | 35 / 53 / 112 | 45.50% | 0.987 | 4/64 | Rejected: arena |
| 20 | 15 | 71 / 19 / 110 | 63.00% | 2.84e-14 | 7/64 | Accepted |
| 21 | 20 | 37 / 54 / 109 | 45.75% | 0.998 | 14/64 | Rejected: both gates |
| 22 | 20 | 65 / 24 / 111 | 60.25% | 3.22e-10 | 14/64 | Rejected: probe |
| 23 | 20 | 43 / 16 / 141 | 56.75% | 3.67e-05 | 13/64 | Rejected: probe |
| 24 | 20 | 41 / 30 / 129 | 52.75% | 0.0449 | 12/64 | Rejected: arena |
| 25 | 20 | 49 / 31 / 120 | 54.50% | 0.000498 | 12/64 | Rejected: arena |
| 26 | 20 | 65 / 21 / 114 | 61.00% | 7.92e-09 | 7/64 | Accepted |
| 27 | 26 | 57 / 57 / 86 | 50.00% | 0.547 | 8/64 | Rejected: arena |
| 28 | 26 | 82 / 28 / 90 | 63.50% | 7.11e-15 | 20/64 | Rejected: probe |
| 29 | 26 | 39 / 26 / 135 | 53.25% | 0.011 | 5/64 | Rejected: arena |
| 30 | 26 | 63 / 40 / 97 | 55.75% | 0.00021 | 7/64 | Accepted |

Generation 22 passed its strength arena but drew 14/64 noisy games; generation
23 drew 13/64. Both exceeded the 20% limit and were rejected. Generation 28
likewise won its arena but drew 20/64. The final champion, generation 30,
passed both checks: 55.75% against generation 26, p = 0.0002098, and 7/64 probe
draws. Its promotion was followed by the independent comparison below.

## Independent comparisons with the October champion

The user extended the initial 20-attempt plan to at least 22, then to the next
promotion, which occurred at 26. After its failed independent comparison, the
user requested one further promotion before another duel; that successor was
30. These selection changes were recorded before the corresponding independent
arena. Neither candidate was selected by testing several checkpoints against
the reference and keeping the best result.

Both arenas used 800 games, 400 simulations per move, exact sequential PUCT
(one leaf per request), 128 game workers, no root noise and paired random legal
openings of 0–4 plies. Each pair swaps candidate colors. The inference services
used minimum/maximum batches of 64/128 with a 1 ms collection wait.

| New candidate vs old generation 21 | Wins / losses / draws | Score | Paired p | Result |
| --- | --- | ---: | ---: | --- |
| 26 | 232 / 386 / 182 | 40.375% | ≈ 1 | Failed |
| **30** | **313 / 162 / 325** | **59.4375%** | **3.54 × 10⁻¹⁵** | **Confirmed** |

The first arena used base seed `2026110801` and took 392.00 seconds. The second
used `2026111801` and took 430.21 seconds. Pair i uses base seed + i: before the
second arena, the unused seed bases were corrected to be 1,000 apart, avoiding
the overlap that initially adjacent bases would have caused. Thus the completed
arenas used disjoint 400-seed ranges. Their opening histories contained 169 and
166 distinct values respectively; repeated openings within a sampled arena are
expected. The first arena's data was not changed or rerun.

| Generation 30's seat | Wins / losses / draws | Score |
| --- | --- | ---: |
| First | 156 / 86 / 158 | 58.750% |
| Second | 157 / 76 / 167 | 60.125% |

The successful comparison's paired score counts for 0, 0.5, 1, 1.5 and 2 points
are `[9, 34, 219, 73, 65]`. The threshold remained 55% with p ≤ 0.05/3 throughout;
the reserved third comparison was unnecessary. Full results and checkpoint
identities are retained for [generation 26](experiments/2026-11-08/confirmation-26.json)
and [generation 30](experiments/2026-11-08/confirmation-30.json), with their
adjacent run-plan files.

## Execution time

The 30 complete attempts generated **7,680 self-play games** and applied
**12,000 optimizer updates**. The pipeline took **2 h 25 min 47 s**, excluding
compilation, pauses between commands, independent arenas and performance tests.
These timings come from whole-second CLI timestamps on an Apple M4 Max with
Rust 1.97.1 and the default release profile.

| Phase | Total |
| --- | ---: |
| Self-play, including restarts | 55 min 56 s |
| Optimization and validation | 2 min 29 s |
| Promotion arenas | 1 h 16 min 12 s |
| Noisy productivity probes | 10 min 39 s |
| Other transitions | 31 s |
| **Total** | **2 h 25 min 47 s** |

Early attempts overlapped compilation of the measurement executables and the
required checks. These phase durations are observations, not an estimate of a
speedup. The separate [fixed-workload performance comparison](performance-results.md)
uses interleaved runs without concurrent training.

## Verification

The required checks passed: 86 tests (two Metal tests normally ignored),
Clippy with warnings denied, formatting and documentation generation. The
Metal training-state round-trip diagnostic also passed before training.
Browser verification completed three WebGPU games and two Flex games, including
28 captures, 20 drops and resets; a human move with an AI reply was checked
through the interface. Desktop and mobile layouts were inspected. Both release
archives match their local files byte for byte and pass SHA-256 verification.
See the [verification record](experiments/2026-11-08/verification.json).

The old local training/model directories were removed after preserving and
rechecking the immutable October reference. All 31 new checkpoints, including
initialization and rejected candidates, remain intact. The
[cleanup record](experiments/2026-11-08/cleanup.json) lists the removed paths.

## Artifacts and limits

- `models/alpha-zero-from-scratch-20261108/`: all new checkpoints and Adam states;
  `latest` selects generation 30.
- `data/alpha-zero-from-scratch-20261108/`: self-play, replay buffer, reports,
  process logs, independent arenas and performance fixtures.
- `models/references/pre-cleanup-20261108/`: immutable October reference retained
  for future comparisons.
- [Release v1.0.11](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v1.0.11):
  deployable website and minimal accepted checkpoint as separate archives.

This establishes a stronger player against this fixed opponent at the stated
search budget. It does not prove perfect play or absence of every regression.
One fresh seed and a historical checkpoint do not establish unchanged learning
efficiency across seeds or equal-compute runs. The measured implementation
compatibility and execution costs are reported separately; the small CPU
preparation overhead is retained there rather than hidden by the strength gain.
