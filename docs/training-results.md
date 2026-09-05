# Training results

The accepted champion is **generation 19**. It is the model selected by
`latest` for local play, browser export and training. Generation **16** is
the fixed reference for the comparison recorded on this page. All checkpoints,
including rejected candidates, remain available in the local model directory.

The [AlphaZero guide](alphazero-guide.md) explains the learning algorithm and
metrics; this page records the evaluation protocol and measured results.

## Training and evaluation protocol

The settings live in [`config/training.toml`](../config/training.toml). The
network has 64 channels, four residual blocks, and 64-unit shared and value
layers. Training runs on Metal with the following budget per attempt:

| Phase | Budget |
| --- | --- |
| Self-play | 256 games, 200 MCTS simulations per regular move |
| Visited-state restarts | 25% of games when an archive is available, 800 simulations per move |
| Optimization | 400 Adam updates, batch size 256 |
| Strength arena | 200 games against the source champion, 400 simulations per move |
| Productivity probe | 64 noisy self-play games |

The learning rate starts at 0.001 and becomes 0.00025 when the source champion
reaches generation 7. Mirrored examples augment the training set. Decisive
endgame tails are oversampled during bootstrap; from attempt 11, the complete
buffer is used without this extra sampling.

Each arena pair shares a randomly sampled legal opening of 0–4 plies, then
exchanges candidate colors. Search uses no root noise. Promotion requires at
least 55% of points, a one-sided paired p-value at most 0.05, and at most 20%
draws in the productivity probe. See
[Promotion measurements](alphazero-guide.md#promotion-measurements) for the test
and its assumptions.

The accepted champion supplies self-play, trainable weights and Adam state.
A rejected candidate never becomes the next training source. Games generated
for its attempt remain in the rolling buffer, so the next attempt receives more
examples while starting from the accepted weights.

## Results by generation

Wins and losses are from the candidate's perspective. A win earns one point
and a draw half a point: a score of 60% is not a 60% win rate. Generation zero
is the randomly initialized network.

| Candidate | Source champion | Wins / losses / draws | Score | Paired p-value | Probe draws | Recorded decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | 0 | 175 / 25 / 0 | 87.50% | — | 1/64 | Accepted |
| 2 | 1 | 131 / 68 / 1 | 65.75% | — | 2/64 | Accepted |
| 3 | 2 | 128 / 61 / 11 | 66.75% | — | 2/64 | Accepted |
| 4 | 3 | 106 / 32 / 62 | 68.50% | — | 1/64 | Accepted |
| 5 | 4 | 126 / 26 / 48 | 75.00% | — | 1/64 | Accepted |
| 6 | 5 | 63 / 84 / 53 | 44.75% | — | 2/64 | Rejected |
| 7 | 5 | 88 / 65 / 47 | 55.75% | — | 0/64 | Accepted |
| 8 | 7 | 123 / 39 / 38 | 71.00% | — | 4/64 | Accepted |
| 9 | 8 | 95 / 23 / 82 | 68.00% | — | 2/64 | Accepted |
| 10 | 9 | 51 / 44 / 105 | 51.75% | — | 4/64 | Rejected |
| 11 | 9 | 84 / 29 / 87 | 63.75% | — | 3/64 | Accepted |
| 12 | 11 | 29 / 40 / 131 | 47.25% | — | 7/64 | Rejected |
| 13 | 11 | 106 / 38 / 56 | 67.00% | — | 8/64 | Accepted |
| 14 | 13 | 73 / 27 / 100 | 61.50% | — | 8/64 | Accepted |
| 15 | 14 | 23 / 21 / 156 | 50.50% | — | 5/64 | Rejected |
| 16 | 14 | 79 / 31 / 90 | 62.00% | — | 6/64 | Accepted |
| 17 | 16 | 101 / 32 / 67 | 67.25% | 7.82e-13 | 5/64 | Accepted |
| 18 | 17 | 94 / 88 / 18 | 51.50% | 0.1768 | 6/64 | Rejected |
| 19 | 17 | 141 / 38 / 21 | 75.75% | 1.09e-15 | 5/64 | Accepted |

Decisions are reported as recorded in the checkpoints' generation reports.
A dash means paired counts were not saved, so the p-value cannot be
reconstructed. Those accepted rows do not establish that the statistical
requirement stated above was met.

Generation 19 scored 74.5% as First and 77.0% as Second against its source
champion. Its productivity probe drew 5/64 games (7.81%). Generation 18 is a
useful example of rejection: its draw rate was acceptable, but its strength
improvement was not established. A larger generation number alone does not
mean a stronger model.

## Champion versus the regression reference

An independent arena compared generation 19 directly with generation 16 on
400 games, with 400 MCTS simulations per move. The opening seed, `20260905`,
was fixed before the comparison. The 200 pairs included 111 distinct opening
histories.

| Generation 19's seat | Wins / losses / draws | Score |
| --- | --- | ---: |
| First | 81 / 65 / 54 | 54.00% |
| Second | 108 / 55 / 37 | 63.25% |
| Both | **189 / 120 / 91** | **58.625%** |

The one-sided paired p-value is **2.08e-11**, confirming an advantage under
this protocol. Pair counts for 0, 0.5, 1, 1.5 and 2 candidate points are
`[3, 6, 122, 57, 12]`. The comparison report records its full configuration
and source commit, `c05515a7ebf00c603d9bdd5113eaf79101721ab3`.

## Limits of the measurements

These arenas measure strength against particular opponents at a fixed search
budget. They do not prove perfect play or a training-efficiency gain. Comparing
learning efficiency requires complete runs at equal compute budgets, with
multiple random seeds.

The retained replay buffer contains trajectories without complete ancestry.
Related training and validation examples can therefore overlap, limiting what
validation losses say about generalization. The family split in the code cannot
reconstruct missing ancestry; an experiment requiring a clean validation split
needs a fresh self-play directory. The strength arenas generate their own
openings independently of that split.

## Checkpoints and reports

Paths are configured in `config/training.toml`:

- `models/alpha-zero-visited-restarts-v26/`: checkpoints, optimizer states and
  the `latest` champion pointer;
- `data/alpha-zero-visited-restarts-v26/reports/`: generation reports, including
  training metrics, arena outcomes and promotion decisions;
- `data/alpha-zero-visited-restarts-v26/confirmation-arenas/`: independent
  comparison results;
- `data/alpha-zero-visited-restarts-v26/logs/`: execution logs and diagnostic
  driver sources.

These generated files are ignored by Git. The deployable website and minimal
accepted checkpoint are available separately in the
[latest release](https://github.com/baptistefetet/YokaiRust/releases/latest).
