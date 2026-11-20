# Training results

**Generation 80** is the published 5×6 champion. It beat generation 64, the
previous web champion, with **281 wins, 53 losses and 66 draws** in 400 games.
The release includes both the playable site and the native checkpoint.
[Web v2.4.0](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.4.0)

## How to read the two matches

| What the match measures | Opponent | Games | Generation 80’s score |
| --- | --- | ---: | ---: |
| Whether to promote the candidate | Generation 76, its training source | 200 | 61% |
| Progress since the previous release | Generation 64, chosen before training | 400 | 78.5% |

Promotion requires at least 55% of the points and a one-sided paired
p-value at most 0.05. The candidate also plays 64 games against itself
with extra randomness to check that it still produces decisive games.
Generation 80 drew 11 of those 64 games, below the 20% limit.
[Promotion report](https://github.com/baptistefetet/YokaiRust/blob/web-v2.4.0/docs/experiments/2026-11-20/README.md#training-outcomes)

The independent match tested the gain across five promotions. Each model
played both sides of the same opening, with 400 search simulations per move.
A win earns one point and a draw earns half, so generation 80 scored
281 + 66/2 = **314 points out of 400**. It earned 157 points from each seat.
The one-sided paired p-value was **1.20 × 10⁻³⁵**.
[Match protocol and results](https://github.com/baptistefetet/YokaiRust/blob/web-v2.4.0/docs/experiments/2026-11-20/README.md#independent-match)

## Start a run from scratch

From the repository root, copy the training configuration for your run.
It uses Metal on Apple Silicon. For CPU training, set `backend = "cpu"`
in the copy.

```sh
cp config/training.toml config/my-run.toml
```

In the copy, set both paths to new directories. The trainer creates them,
initializes generation zero with random weights and generates its own games.

```toml
[paths]
models = "models/my-run"
self_play = "data/my-run"
```

Run one training attempt. Repeat the command to continue from this run’s
accepted champion and saved games. Each attempt evaluates one candidate,
so several attempts can pass before the champion changes.

```sh
cargo run --release -- train --config config/my-run.toml --generations 1
```

The original run used seed `20261108`. Its protocols, match records and
per-generation measurements are available for comparison.
[Experiment archive at web-v2.4.0](https://github.com/baptistefetet/YokaiRust/tree/web-v2.4.0/docs/experiments)
