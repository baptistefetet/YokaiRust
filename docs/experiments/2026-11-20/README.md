# Five promotions from generation 64

## Protocol recorded before attempt 65

Generation 64 was the accepted champion when this experiment began. Its
weights match the hashes in the [previous experiment](../2026-11-17/README.md).
Keep this checkpoint for the final match. Continue with its saved Adam
optimizer state and the existing replay buffer at training revision
`f55a6c4eb185ec14bf93dcd658853a045f831083`. The SHA-256 of
`config/training.toml` is
`a86bdc52b5f92a94f5a0d9e76803618a65ac0ff5b31626311248f2a9fadec20c`.

| Generation 64 file | SHA-256 |
| --- | --- |
| model.safetensors | `7aeefdc9e680f6cd593a21f7fd109192c04e291ab32191c6f93528e0914c5c81` |
| metadata.json | `77e5e3b315aa34db9077d6b55ea68b798d9549592e0db060ba073d02eef10e7a` |
| training-model.bin | `7e2558bf25d0d45f70adc8c5b280d6d8d4ab7fdf7df9c8b8948c4cd3cb0f256a` |
| optimizer.bin | `659f702c90ad1ebc05b50c883eced46a06866510a7f94bc278202b5fb9016365` |

The initial stopping rule was ten promotions after generation 64. Each
candidate plays 200 paired games against its source champion. Promotion
requires a score of at least 55%, a one-sided paired p-value at most 0.05
and at most 20% draws in a separate 64-game noisy self-play probe.

Play a single independent match between the champion at the stopping point and
generation 64. Use 400 games in 200 pairs, with both models playing the same opening from
opposite seats. Reserve opening seeds 20262000–20262199 before training.
Use 0–4 random legal opening plies, 400 search simulations per move, eight
leaves per inference and no search noise. A win earns one point and a draw
earns half a point. Report wins, losses, draws, scores by seat and the
one-sided paired p-value. A score above 50% with p ≤ 0.05 is evidence that
the new champion is stronger under these match conditions.

## Stopping rule amended during attempt 80

With four promotions recorded (66, 72, 73 and 76), the requested training
budget changed to **five promotions**. Stop at the fifth, then play the
previously specified match against generation 64. Publish a new web release
if the fifth champion wins the match by scoring more than 200 points.
Report the paired test alongside the score.

## Independent match

Generation 80 scored **314/400 points (78.5%)** against generation 64,
with 281 wins, 53 losses and 66 draws. The one-sided paired p-value was 1.2e-35.
The 200 opening pairs contained 147 distinct opening histories.
The match supports a strength gain over generation 64 at this search budget. [Full arena record](arena-80-vs-64.json)

| New champion’s seat | Wins | Losses | Draws | Score |
| --- | ---: | ---: | ---: | ---: |
| First | 140 | 26 | 34 | 78.5% |
| Second | 141 | 27 | 32 | 78.5% |

## Training outcomes

The run promoted five candidates in 16 attempts using 4,096 self-play games
and 6,400 optimizer updates. The promoted generations were 66, 72, 73, 76
and 80. Training took 2 h 02 min 40 s from the start of attempt 65 to the
fifth promotion. The independent match took another 4 min 52 s.

| Attempt | Source | Wins / losses / draws | Score | Paired p | Probe draws | Decision |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| [65](reports/generation-000065.json) | 64 | 99 / 90 / 11 | 52.25% | 0.287 | 8/64 | Rejected |
| [66](reports/generation-000066.json) | 64 | 108 / 51 / 41 | 64.25% | 6.375e-07 | 7/64 | Promoted |
| [67](reports/generation-000067.json) | 66 | 91 / 92 / 17 | 49.75% | 0.5643 | 4/64 | Rejected |
| [68](reports/generation-000068.json) | 66 | 75 / 102 / 23 | 43.25% | 0.9778 | 8/64 | Rejected |
| [69](reports/generation-000069.json) | 66 | 94 / 93 / 13 | 50.25% | 0.5 | 5/64 | Rejected |
| [70](reports/generation-000070.json) | 66 | 65 / 119 / 16 | 36.5% | 0.9998 | 6/64 | Rejected |
| [71](reports/generation-000071.json) | 66 | 67 / 99 / 34 | 42% | 0.9951 | 4/64 | Rejected |
| [72](reports/generation-000072.json) | 66 | 106 / 82 / 12 | 56% | 0.02646 | 5/64 | Promoted |
| [73](reports/generation-000073.json) | 72 | 106 / 82 / 12 | 56% | 0.01436 | 6/64 | Promoted |
| [74](reports/generation-000074.json) | 73 | 90 / 97 / 13 | 48.25% | 0.7458 | 4/64 | Rejected |
| [75](reports/generation-000075.json) | 73 | 60 / 98 / 42 | 40.5% | 0.9988 | 1/64 | Rejected |
| [76](reports/generation-000076.json) | 73 | 104 / 83 / 13 | 55.25% | 0.02842 | 9/64 | Promoted |
| [77](reports/generation-000077.json) | 76 | 90 / 90 / 20 | 50% | 0.5337 | 9/64 | Rejected |
| [78](reports/generation-000078.json) | 76 | 87 / 102 / 11 | 46.25% | 0.9077 | 8/64 | Rejected |
| [79](reports/generation-000079.json) | 76 | 82 / 106 / 12 | 44% | 0.9904 | 9/64 | Rejected |
| [80](reports/generation-000080.json) | 76 | 113 / 69 / 18 | 61% | 0.001648 | 11/64 | Promoted |

## Generation 80 checkpoint

| File | SHA-256 |
| --- | --- |
| model.safetensors | `bf47137df8270076f8c9c722ceca60f246983d3fe88be7c569ee1746a98a7379` |
| metadata.json | `90360127482133b6d42edbdbcce3ab9f1369ff37042f79b512e233f9aee47247` |

## Reproducing the independent match

From the repository root, with both checkpoints available, choose a new output path.

```bash
YOKAI_CANDIDATE_MODELS=models/expert-5x6 YOKAI_CANDIDATE_GENERATION=80 \
YOKAI_REFERENCE_MODELS=models/expert-5x6 YOKAI_REFERENCE_GENERATION=64 \
YOKAI_ARENA_GAMES=400 YOKAI_ARENA_SEED=20262000 YOKAI_ARENA_REPORT=arena-rerun.json \
cargo bench --bench performance -- compare_saved_generations
```
