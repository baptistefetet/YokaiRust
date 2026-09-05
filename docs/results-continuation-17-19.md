# Continuation: generations 17–19

On September 5, 2026, three further attempts resumed the accepted generation
16 checkpoint, its Adam state and the existing v26 replay buffer. Generation
19 became the accepted champion; generation 16 remains the frozen reference.

## Protocol

The run used commit `c05515a7ebf00c603d9bdd5113eaf79101721ab3` and the unchanged
hyperparameters in `config/training.toml`, following the correctness fixes.
There was no architecture or search-budget change during these attempts.

Each attempt generated 256 games, including 64 visited-state restarts, then
applied 400 Adam updates at learning rate 0.00025 and batch size 256. Promotion
required all three conditions:

- at least 55% of points in a 200-game paired arena against the source champion;
- a one-sided paired sign-flip p-value of at most 0.05;
- at most 20% draws in a separate 64-game noisy self-play probe.

Arenas used 400 simulations per move, no root noise, and shared random
0–4-ply openings with exchanged colors. Self-play retained its 200 regular
simulations and 800 simulations on restarted trajectories.

The three attempts completed in **17 min 09 s** on Metal, excluding compilation
and the independent confirmation arena. They added 768 self-play games and
performed 1,200 optimizer updates. After a rejection, the next attempt reused
the accepted champion's weights and the enriched buffer.

## Promotion results

Wins and losses below are from the candidate's perspective. A draw earns half
a point; the score is a points percentage, not a win percentage.

| Candidate | Reference | Wins / losses / draws | Score | Paired p-value | Probe draws | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 17 | 16 | 101 / 32 / 67 | 67.25% | 7.82e-13 | 5/64 (7.81%) | Accepted |
| 18 | 17 | 94 / 88 / 18 | 51.50% | 0.1768 | 6/64 (9.38%) | Rejected |
| 19 | 17 | 141 / 38 / 21 | 75.75% | 1.09e-15 | 5/64 (7.81%) | Accepted |

Generation 19 scored 74.5% as First and 77.0% as Second against generation 17.
Generation 18 demonstrates why attempt number alone does not measure strength:
it passed the productivity probe but did not establish an improvement.

## Independent confirmation against generation 16

Generation 19 was checked on 400 additional paired games with seed 20260905,
selected before this arena ran, at the same 400-simulation budget. The result
confirmed its advantage over the frozen generation 16 reference:

| Candidate / reference | Wins / losses / draws | Score | Paired p-value |
| --- | --- | ---: | ---: |
| 19 / 16 | 189 / 120 / 91 | **58.625%** | **2.08e-11** |

The candidate scored 54.0% as First (81/65/54) and 63.25% as Second
(108/55/37). The 200 pairs contained 111 distinct opening histories. Paired
score counts for 0, 0.5, 1, 1.5 and 2 candidate points were respectively
`[3, 6, 122, 57, 12]`.

The rebuilt WebGPU site loaded generation 19 and completed a human move and
an AI reply without engine errors. Both WASM backends and the minimal accepted
checkpoint are distributed in release `web-v1.0.2`.

## Interpretation and artifacts

These measurements compare playing strength under the stated search budget.
They do not establish that the fixes reduce the training time needed to reach
a given strength: that would require a controlled comparison of complete runs.
The old buffer was retained, so its legacy restart leakage still limits the
interpretation of validation losses. The paired arenas use separately sampled
openings and do not select moves from the training/validation split.

Local artifacts remain under `data/alpha-zero-visited-restarts-v26/`:

- `reports/generation-000017.json` through `generation-000019.json`;
- `logs/continuation-17-19-20260905.log`;
- `confirmation-arenas/` for the independent comparison and `logs/` for its
  output and one-off Rust driver source.

All checkpoint history is retained under `models/alpha-zero-visited-restarts-v26/`.
The `latest` pointer selects generation 19. Browser and terminal play keep the
same MCTS behavior: a final win has no preference for fewer moves, and final
selection uses visit counts rather than an explicit immediate-win override.
