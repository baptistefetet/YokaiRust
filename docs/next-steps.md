# Next steps

The 5×6 rules, terminal interface, neural encoder and policy mapping are
implemented and verified; [Rules of the 5×6 game](rules.md) states them. The
learning pipeline has a validated baseline on the classic 3×4 game:
[training results](training-results.md) describe that champion and
[performance results](performance-results.md) its execution costs. The
[reading guide](reading-guide.md) describes the implementation.

No champion exists for the 5×6 board yet. Until one is accepted, the terminal
offers two-player games and uniform-prior analysis, and the published web site
remains the 3×4 release. The first three steps below lead to a 5×6 player; the
following ones apply to any training run.

The web interface remains dedicated to playing. Diagnostics belong in the TUI.

## 1. Choose and profile a network for the 5×6 board

The default network was sized for 12 squares and 132 actions. On this board it
has about 1.17 million parameters, 59% of them in the single dense layer that
maps the 64-unit shared representation and 1,140 context values to 570 action
logits. That layer ignores which square an action starts from, while most of
its inputs are per-action values that concern one output each.

Compare a small number of candidates before spending training time: a wider
shared representation, and a policy head that keeps the board structure
(per-square outputs for the eight directions, two promotion choices and three
drops) with the repetition context applied per action. A change of layer
structure needs a new model format version.

Measure the pipeline as well. Record the time of each phase, game lengths and
draw rates over several generations, then set `simulations`,
`exploration_plies`, the arena budget and the restart budget from those
measurements.

**Done when:** a frozen configuration and written protocol exist for the first
5×6 run, with a measured time per generation.

## 2. Train the first 5×6 champion

`config/training.toml` points to empty `models/expert-5x6` and
`data/expert-5x6` directories, so the 3×4 lineage is never read or modified.
Record the fixed reference checkpoint before the run, keep the paired arena
and noisy productivity probe as promotion gates, and store the experiment
report under `docs/experiments/`.

**Done when:** an accepted champion beats the fixed reference in an independent
arena and `yokai play human-vs-cpu` plays it.

## 3. Port the web interface to the 5×6 board

The browser adapter already receives board dimensions from Rust snapshots. The
presentation needs the 5×6 board and piece sprites, the choice between a plain
and a promoting move, and a mode selection: one or two human players, and the
classic or expert game. The classic mode can load the frozen WebAssembly bundle
and model of release `web-v1.0.11` instead of keeping two rule sets in the
current engine.

**Done when:** `./web/scripts/build.sh` produces a deployable site that plays
the expert game against the 5×6 champion, and both modes pass the smoke test of
the [web build guide](../web/README.md).

## 4. Preserve the baseline while improving durability

Keep checkpoint history, replay ancestry, fixed reference and experiment
reports available for regression comparisons. Future candidates must still pass
both the paired strength arena and noisy productivity probe.

Temporary files followed by rename protect against interrupted writes, but the
complete publication sequence does not explicitly synchronize files and parent
directories against power loss. Handle checkpoint files, journal and report
before the accepted `latest` pointer, and verify recovery at each boundary.

**Done when:** interruption and power-loss guarantees are documented and tested
without weakening promotion or rewriting checkpoint history.

## 5. Profile the costs that matter for larger training runs

Game generation, promotion arenas and productivity probes occupy almost all
measured pipeline time. Measure inference batch collection, partial
requests and generation tails before changing batching or optimizer readbacks.
Use identical checkpoints, inputs and interleaved timings without concurrent
training. Preserve outputs as well as timing measurements. The performance
baseline also identifies CPU request preparation as a small measurable cost;
Metal measurements need enough repetitions to distinguish changes from noise.

Measure replay-buffer memory and I/O before changing its representation.
Splits and augmentation clone examples; generation files and the buffer retain
overlapping data. Any replacement must preserve MCTS targets, restart prefixes,
family identities and per-game retention. Add long-history search cases before
optimizing game cloning for the larger board.

**Done when:** a measured bottleneck is reduced on the complete workload while
behavior and artifact compatibility remain demonstrably stable.

## 6. Add useful diagnostics for learning and play

A tactical diagnostic set should cover verified captures, defence, promotion,
drops and repetitions from both colors. Keep it outside training. A small
worked example can follow one position through encoding, predictions, legal
masking, MCTS targets and a CPU optimizer update.

Investigate suspicious moves through the TUI and the existing replay format.
Retain full history, checkpoint identity and search settings. A missed capture
alone does not establish whether rules, search or the learned policy are wrong.

**Done when:** a reproducible case explains a weakness and suggests a specific
experiment instead of relying on loss curves or isolated game impressions.

## 7. Keep future learning comparisons independent

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
