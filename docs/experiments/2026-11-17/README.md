# Three promotions from the published 5×6 champion

## Protocol recorded before attempt 58

The reference is generation 53 of `models/expert-5x6`, distributed with
[web-v2.2.1](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.2.1).
The local weights match the weights in the release archive. Retain generation
53 throughout training and evaluation.

| Reference file | SHA-256 |
| --- | --- |
| Release champion archive | `574777b4cdcf3ed3ca24ca4313e478a25d2b8bf6a613254ae3c41b55f9f15f8f` |
| model.safetensors | `7cc61a3ce24f423516b9fe85bab97efe247117381863e806be4daf15bc3ae4c2` |
| metadata.json | `651656c66836d8e4d35b4f91a0d508eba5f6dafb3272dcb6660170ecf8e4b65e` |

Resume after completed attempt 57, with the accepted champion’s Adam state
and the existing replay buffer. The training revision is `fc2b219`.
The SHA-256 of `config/training.toml` is
`a86bdc52b5f92a94f5a0d9e76803618a65ac0ff5b31626311248f2a9fadec20c`.
The only change to the learning objective is that neutral draw supervision
preserves all original visits for both players. Previously a position in a
drawn game could discard a third-repetition move for the player who moved
second. All search budgets, optimizer settings and promotion gates carry over.

Stop at the third new promotion. Each candidate must score at least 55% in
its 200-game paired arena, attain one-sided paired p ≤ 0.05 and draw at most
20% of its 64 noisy probe games. Play the resulting champion once against
generation 53 over 400 games, using 200 color-swapped opening pairs, 400
simulations per move, eight leaves per inference, no search noise, 0–4 random
opening plies and base seed `20261750`. A win earns one point and a draw half.
Publish a new web release if the new champion scores more than 200 points.
Report the paired p-value separately to quantify the evidence of a strength gain.

## Training outcomes

Each row covers 256 self-play games, 400 optimizer updates, a 200-game arena
and a 64-game productivity probe. The full generation reports contain the
training and validation losses, paired scores, seat breakdowns and replay statistics.

| Attempt | Source | Wins / losses / draws | Score | Paired p | Probe draws | Decision |
| ---: | ---: | --- | ---: | ---: | ---: | --- |
| [58](reports/generation-000058.json) | 53 | 78 / 87 / 35 | 47.75% | 0.7719 | 12/64 | Rejected |
| [59](reports/generation-000059.json) | 53 | 123 / 71 / 6 | 63.00% | 2.56e-05 | 6/64 | Promoted |
| [60](reports/generation-000060.json) | 59 | 70 / 95 / 35 | 43.75% | 0.9838 | 1/64 | Rejected |
| [61](reports/generation-000061.json) | 59 | 81 / 83 / 36 | 49.50% | 0.5939 | 9/64 | Rejected |
| [62](reports/generation-000062.json) | 59 | 107 / 75 / 18 | 58.00% | 0.01892 | 5/64 | Promoted |
| [63](reports/generation-000063.json) | 62 | 74 / 75 / 51 | 49.75% | 0.5671 | 7/64 | Rejected |
| [64](reports/generation-000064.json) | 62 | 129 / 56 / 15 | 68.25% | 1.088e-07 | 7/64 | Promoted |

## Independent match

Generation 64 scored **238.5/400 points (59.625%)** against
generation 53 with 227 wins, 150 losses and 23 draws. The paired
one-sided p-value was 4.991e-06 across 200 pairs
and 143 distinct opening histories. The candidate
won the scheduled match. [Full arena record](arena-64-vs-53.json)

| Candidate seat | Wins | Losses | Draws | Score |
| --- | ---: | ---: | ---: | ---: |
| First | 118 | 70 | 12 | 62.00% |
| Second | 109 | 80 | 11 | 57.25% |

Generation 64 checkpoint hashes

| File | SHA-256 |
| --- | --- |
| metadata.json | `77e5e3b315aa34db9077d6b55ea68b798d9549592e0db060ba073d02eef10e7a` |
| model.safetensors | `7aeefdc9e680f6cd593a21f7fd109192c04e291ab32191c6f93528e0914c5c81` |

## Reproducing the independent match

Run from the repository root with the two recorded checkpoints available.
The report destination must be a new file.

```bash
YOKAI_CANDIDATE_MODELS=models/expert-5x6 YOKAI_CANDIDATE_GENERATION=64 \
YOKAI_REFERENCE_MODELS=models/expert-5x6 YOKAI_REFERENCE_GENERATION=53 \
YOKAI_ARENA_GAMES=400 YOKAI_ARENA_SEED=20261750 YOKAI_ARENA_REPORT=arena-rerun.json \
cargo bench --bench performance -- compare_saved_generations
```
