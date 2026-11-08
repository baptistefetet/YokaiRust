# Next steps

The 3×4 baseline has now been trained again from random initialization and
compared with the retained October champion. The new accepted model wins the
independent comparison. [Training results](training-results.md) record the full
run, including rejected candidates and the failed first confirmation;
[performance results](performance-results.md) separate execution costs from
playing strength. The [reading guide](reading-guide.md) describes the current
implementation.

The web interface remains dedicated to playing. Diagnostics belong in the TUI.
The 5×6 expert game is a separate implementation milestone.

## 1. Preserve this baseline while improving durability

Keep the new checkpoint history, replay ancestry, fixed reference and experiment
reports available for regression comparisons. Future candidates must still pass
both the paired strength arena and noisy productivity probe.

Temporary files followed by rename protect against interrupted writes, but the
complete publication sequence does not explicitly synchronize files and parent
directories against power loss. Handle checkpoint files, journal and report
before the accepted `latest` pointer, and verify recovery at each boundary.

**Done when:** interruption and power-loss guarantees are documented and tested
without weakening promotion or rewriting checkpoint history.

## 2. Profile the costs that matter for larger training runs

The fixed-workload comparison preserves exact observed behavior across cleanup.
It also records a small CPU request-construction overhead; do not describe the
cleanup as cost-free. Metal timings overlap, so their lower or higher medians
alone do not justify a speedup or slowdown claim.

In the fresh run, game generation, promotion arenas and productivity probes
occupied almost all pipeline time. Measure inference batch collection, partial
requests and generation tails before changing batching or optimizer readbacks.
Use identical checkpoints, inputs and interleaved timings without concurrent
training. Preserve outputs as well as timing measurements.

Measure replay-buffer memory and I/O before changing its representation.
Splits and augmentation clone examples; generation files and the buffer retain
overlapping data. Any replacement must preserve MCTS targets, restart prefixes,
family identities and per-game retention. Add long-history search cases before
optimizing game cloning for the larger board.

**Done when:** a measured bottleneck is reduced on the complete workload while
behavior and artifact compatibility remain demonstrably stable.

## 3. Add useful diagnostics for learning and play

A tactical diagnostic set should cover verified captures, defence, promotion,
drops and repetitions from both colors. Keep it outside training. A small
worked example can follow one position through encoding, predictions, legal
masking, MCTS targets and a CPU optimizer update.

Investigate suspicious moves through the TUI and the existing replay format.
Retain full history, checkpoint identity and search settings. A missed capture
alone does not establish whether rules, search or the learned policy are wrong.

**Done when:** a reproducible case explains a weakness and suggests a specific
experiment instead of relying on loss curves or isolated game impressions.

## 4. Keep future learning comparisons independent

One successful fresh seed establishes trainability and a stronger player under
the recorded protocol. Comparing learning efficiency requires multiple fresh
seeds with equal compute budgets. Record hypotheses, references, candidate
selection and independent evaluation rules before observing results.

Arena pair i uses `base_seed + i`; reserve non-overlapping ranges for separate
comparisons. Apply the planned multiple-comparison correction and retain failed
arenas. Also version seed derivation before very long runs: the current probe's
`generation << 56` repeats every 256 attempts.

Prefer an isolated loss-weight or decisive-endgame sampling experiment before
another architecture change. Continue from accepted checkpoints; rejected
weights never become self-play sources.

**Done when:** independent comparisons and repeated equal-budget runs support
any claimed learning-efficiency improvement.

## Deferred: the 5×6 expert game

The expert game needs its own verified rules, action representation, network
and artifact identity. Shared dimensions alone do not implement it: the current
compact policy index and tensors intentionally describe 3×4.

Design both variants together when this work begins, with rule tests and fixed
performance fixtures for each. Preserve the working 3×4 player and avoid
speculative generic abstractions that have not yet been justified by the
second game's actual requirements.
