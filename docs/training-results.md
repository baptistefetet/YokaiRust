# Training results

The accepted champion is **generation 21**. It is the model selected by
`latest` for local play, browser export and training. Generation **19** is
the fixed reference for the October 2 continuation; the earlier comparison
used generation **16**. All checkpoints, including rejected candidates,
remain available in the local model directory.

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
| 20 | 19 | 34 / 102 / 64 | 33.00% | 1.0000 | 6/64 | Rejected |
| 21 | 19 | 97 / 70 / 33 | 56.75% | 1.58e-5 | 5/64 | Accepted |
| 22 | 21 | 56 / 49 / 95 | 51.75% | 0.2219 | 4/64 | Rejected |

Decisions are reported as recorded in the checkpoints' generation reports.
A dash means paired counts were not saved, so the p-value cannot be
reconstructed. Those accepted rows do not establish that the statistical
requirement stated above was met.

Generation 21 scored 53.0% as First and 60.5% as Second against its source
champion, generation 19. Its productivity probe drew 5/64 games (7.81%).
Generation 22 is a useful example of rejection: its draw rate was acceptable,
but its strength improvement was not established. A larger generation number
alone does not mean a stronger model.

## Continuation on October 2, 2026

Attempts 20–22 used the existing configuration and replay buffer, without
changing the architecture, losses or search budgets. Model format 5,
encoder version 4 and rules version 1 remained compatible. Every attempt
resumed the accepted champion's weights and Adam state: attempts 20 and 21
started from 19, then attempt 22 started from the newly accepted 21.

The three-attempt budget, fixed reference 19, its file hashes and the
independent confirmation protocol were recorded before training in
`experiments/continuation-20-22-20261002.json` under the self-play directory.
The run used source commit `f46135d7f9d58d3dac3b72654e1d8a884deffbc4` and
configuration SHA-256
`3b360f401ff1232b2c0f1279f08c51458e5544f3f1fbabdce775ed01f576388e`.
The reference weights have SHA-256
`655ddd1f7d0dcfe7539210508f1a7a5a2bccc471f7ae602d8515c1b502cb9a9b`;
its weights, metadata and training state were checked unchanged after the run.

Elapsed times below come from whole-second timestamps in
`logs/continuation-20-22-20261002.log`. Self-play includes the 64 archive restarts
per attempt; optimization includes validation metrics.

| Attempt | Self-play | Optimization | Arena | Probe | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 | 1:48 | 0:07 | 2:38 | 0:23 | 4:57 |
| 21 | 1:50 | 0:05 | 3:10 | 0:31 | 5:37 |
| 22 | 3:18 | 0:05 | 2:48 | 0:27 | 6:40 |
| **Block** | **6:56** | **0:17** | **8:36** | **1:21** | **17:14** |

Times are minutes:seconds. Totals include four seconds of transitions outside
the listed phases; compilation took another 29.24 seconds. The independent
confirmation below is separate. The previous block, attempts 17–19, took
17:09, but used different weights, trajectories and buffer contents. These
observations do not measure a speedup from the engine changes. In this block,
generating and evaluating games took almost all the time; optimizer updates
and validation took 17 seconds in total.

## Independent regression comparisons

### September 5: generation 19 versus generation 16

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

### October 2: generation 21 versus generation 19

The final accepted champion was compared with the fixed starting reference
after all three attempts. The protocol selected this comparison before
training, with no additional attempts or repeat arenas based on its outcome.
It used 400 games, 400 MCTS simulations per move, one leaf per inference,
128 game workers, no root noise and paired legal openings of 0–4 plies.
The independent seed was `20261002`; the 200 pairs included 109 distinct
opening histories. The arena took 303.21 seconds.

| Generation 21's seat | Wins / losses / draws | Score |
| --- | --- | ---: |
| First | 87 / 75 / 38 | 53.00% |
| Second | 91 / 71 / 38 | 55.00% |
| Both | **178 / 146 / 76** | **54.00%** |

The one-sided paired p-value is **0.0007079**, supporting an advantage against
generation 19 under this protocol. However, the score is below the preplanned
55% minimum, so the full independent confirmation criterion **was not met**.
Generation 21 remains the accepted champion because it passed the training
pipeline's selection arena and productivity gate; this separate comparison
does not alter `latest`.
Pair counts for 0, 0.5, 1, 1.5 and 2 candidate points are
`[5, 9, 144, 33, 9]`.

The full protocol is in `confirmation-arenas/protocol-20261002.json`; the
result is in
`confirmation-arenas/generation-000021-versus-000019-seed-20261002.json`.
Both use the source commit and reference hashes recorded for this continuation.

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
- `data/alpha-zero-visited-restarts-v26/experiments/`: continuation protocols,
  checkpoint hashes and completion records;
- `data/alpha-zero-visited-restarts-v26/logs/`: execution logs and diagnostic
  driver sources.

These generated files are ignored by Git. The deployable website and minimal
accepted checkpoint are available separately in the
[latest release](https://github.com/baptistefetet/YokaiRust/releases/latest).
