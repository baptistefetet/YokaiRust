# Performance baseline — 8 November 2026

On an Apple M4 Max, the fixed Metal workload takes a median **25.293 s for
64 self-play games** and **4.288 s for 400 optimizer updates with validation**.
These measurements use a frozen checkpoint and corpus; the current champion's
playing strength is measured separately in [training results](training-results.md).

A paired code-revision comparison produced **identical game replays, inference
work counts, optimizer metrics and post-update predictions in all six runs**.
The measured revision spends **3–3.5% more time preparing CPU evaluation
requests** than the reference revision. Metal timing ranges overlap and do not
establish a speedup or a material end-to-end slowdown.

## Protocol

- Measured revision: `f3ce726`; comparison reference: `3247ee7`.
- Same Apple M4 Max, Rust 1.97.1, default Cargo release profile; CPU not pinned.
- Order: reference, measured, measured, reference, reference, measured.
  No concurrent training or other deliberately launched GPU workload.
- CPU: `benches/engine.rs`, its default iterations and five inner samples per
  case; fixed 22-ply middle position, 200-simulation MCTS, batch 8, seed 42,
  uniform evaluation and a fresh tree.
- Metal: fixed reference checkpoint, generation 21, seed `20261108`;
  inference batches 8 and 128, each with 20 warm-up and 200 timed batches;
  64 self-play games, 16 workers, 200 simulations and search batches of eight.
- Optimization: restore the same Adam state, then perform 400 updates at
  learning rate 0.00025 and batch size 256, validating every 100 steps on the
  same first-generation corpus. This does not modify any stored checkpoint.

The [protocol](experiments/2026-11-08/performance-protocol.json),
[common harness and runner](experiments/2026-11-08/README.md#fixed-workload-performance-comparison)
and [raw measurement summary](experiments/2026-11-08/performance-observations.json)
record inputs, code revisions and reproduction steps. The build refreshes
archived source timestamps to prevent Cargo's shared target cache from reusing
the other revision's crate; both executables are retained separately. The raw
records label the reference and measured revisions `before` and `after`.

## Measured costs

Values are medians over three runs, with the minimum–maximum range in
parentheses. CPU values are medians of the existing five-sample run medians.
A positive change means more time or memory; the ranges are not confidence
intervals. Loading checkpoints/data and compilation are outside phase timers.

| Workload | Reference revision | Measured revision | Median change |
| --- | ---: | ---: | ---: |
| Request, initial position | 0.614 µs (0.608–0.620) | 0.632 µs (0.630–0.635) | +3.0% |
| Request, middle game | 2.294 µs (2.282–2.298) | 2.375 µs (2.375–2.394) | +3.5% |
| Perft, depth 6 | 6.868 ms (6.794–6.960) | 6.967 ms (6.932–7.005) | +1.4% |
| MCTS, initial position | 0.376 ms (0.374–0.380) | 0.382 ms (0.382–0.382) | +1.5% |
| MCTS, middle game | 0.467 ms (0.463–0.468) | 0.465 ms (0.465–0.466) | -0.4% |
| Metal inference, batch 8 | 2.233 ms (2.056–2.233) | 2.136 ms (2.010–2.198) | -4.3% |
| Metal inference, batch 128 | 2.277 ms (2.175–2.309) | 2.267 ms (2.164–2.316) | -0.4% |
| Metal self-play, 64 games | 27.812 s (25.045–29.641) | 25.293 s (24.621–30.003) | -9.1% |
| 400 updates + validation | 4.140 s (4.115–4.541) | 4.288 s (4.212–4.303) | +3.6% |
| Maximum resident memory | 218.344 MiB (217.609–219.078) | 221.500 MiB (221.469–225.047) | +1.4% |

Request construction shows a small consistent increase, and initial-position
MCTS/perft are around 1.5% slower. Middle-game MCTS is essentially unchanged.
Metal self-play varies substantially between repetitions; its lower median
is not evidence of a general speedup. Optimization is 3.6% higher by the median,
within overlapping observed ranges. Maximum resident memory increased by
about 3.2 MiB. The observed CPU cost and memory differences are measurable;
the Metal ranges do not establish a training-throughput regression.

## Observable behavior

All six runs generated the same 64 complete games, containing **3,474 recorded
positions** and **597,409 neural evaluations**. Their replay digest matches
across code versions and repetitions. The frozen training split contained
14,364 augmented training examples and 570 validation examples.

Every optimizer metric at steps 100, 200, 300 and 400 matched exactly. The
initial-position policy after 400 updates matched in its f32 bits, and the
post-update value was identical. Perft counted **21,323** six-ply paths.
The summary stores digests and per-run work counts.

This is a bounded compatibility check on real self-play and optimization,
complementing the rules/search/training tests. It is not an exhaustive proof
for every state, and it does not measure the future 5×6 architecture. The
training baseline measures the accepted champion against a fixed opponent
under its own independent evaluation protocol.
