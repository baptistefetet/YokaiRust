# First 5×6 training run — 14 November 2026

This directory records the first training run on the 5×6 board. The run is
exploratory: it asks whether the pipeline validated on the 3×4 game learns the
larger game with the same network, before any change of architecture.

## Protocol

The protocol was recorded before the run; [`protocol.json`](protocol.json)
holds the exact identifiers.

- **Configuration:** [`config/training.toml`](../../../config/training.toml)
  at the recorded revision. Metal backend, seed `20261108`.
- **Initialization:** empty `models/expert-5x6` and `data/expert-5x6`
  directories, random weights, no imported optimizer state or replay.
- **Network:** 64 filters, four residual blocks, 64-unit shared and value
  layers; model format 5, encoder version 5, rules version 3.
- **Fixed reference:** generation 0 of this run, the random initialization.
  The pipeline creates it when the run starts, so its file hashes are added to
  the protocol at the first generation boundary. It stays in the model
  directory, whose history is never rewritten.
- **Promotion:** the pipeline's gates, unchanged: 55% of 200 paired arena games
  at 400 simulations per move, a one-sided paired p-value of at most 0.05, and
  at most 20% draws among 64 noisy probe games.

Three settings differ from the 3×4 configuration, each for a reason stated in
the configuration file: draws are worth zero to both players in self-play
search and both players' moves are imitated in drawn games; the learning rate
stays constant; arena searches evaluate eight leaves per inference.

### What the run has to show

These criteria were written before the first generation:

1. Every generation completes, and its duration is recorded.
2. Candidates keep being promoted: at least half of the first ten attempts.
3. Validation policy accuracy rises and the share of policy mass on illegal
   actions falls from one accepted generation to the next.
4. Self-play draws stay below 20% and mean game length stays well under the
   256-ply limit.
5. The latest champion beats the fixed reference and an earlier champion of
   the same run in separate arenas.

## Measurements before the run

One generation was timed from random initialization in throwaway directories,
on an Apple M4 Max, to choose the arena settings. Each row is a single run.

| Arena search | Inference batch | Arena | Whole generation | Arena result |
| --- | --- | --- | --- | --- |
| 1 leaf per inference | 128 | 6 min 27 s | 7 min 30 s | 180–18, 2 draws |
| 8 leaves per inference | 128 | 1 min 43 s | 2 min 36 s | 184–14, 2 draws |
| 8 leaves per inference | 512 | 1 min 16 s | 2 min 14 s | 184–14, 2 draws |

With one leaf per inference, the last ten games of the arena took 3 min 14 s,
as long as the first 190. The third row is the recorded configuration.

Before the 256-ply draw rule existed, the same generation did not complete:
with a 512-ply safety limit one bootstrap game exceeded it, and with a
4,096-ply limit the last ten arena games were still running after six minutes.
