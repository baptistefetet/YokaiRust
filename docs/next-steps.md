# Next steps

The 5×6 rules, terminal and web interfaces, neural encoder and policy mapping
are implemented and verified; [Rules of the 5×6 game](rules.md) states them.
A champion has been trained from random initialization and is published with
the web site: [training results](training-results.md) describe it, and the
[records of its training run](experiments/2026-11-14/README.md) hold the
protocol, the fixed reference and the raw measurements. The
[reading guide](reading-guide.md) describes the implementation.

The web interface remains dedicated to playing. Diagnostics belong in the TUI.

## 1. Strengthen the 5×6 champion

Champions beat their predecessors clearly in independent arenas until the
latest one, whose arena shows a gain too small to be credible. The policy
head kept improving over the attempts measured. The value head is the weak
part: on games it has never trained on, it reads the last plies of a game
well but predicts the outcome of a middle-game position no better than giving
each result one chance in three. The training report gives the measurements.

Continue the run under its recorded protocol, and repeat the fixed-set
validation by distance to the end of the game and the independent arenas at
regular intervals. If the value head stops improving, change one thing at a
time, in this order: the amount of self-play data per attempt, then loss
weighting or decisive-endgame sampling.

The network comes last. Its policy head ends in a single dense layer that
holds 59% of the 1.17 million parameters and ignores which square an action
starts from. A head that keeps the board structure (per-square outputs for
the eight directions, two promotion choices and three drops) is the candidate
to compare; a change of layer structure needs a new model format version and
a fresh run.

**Done when:** the report shows either a value head that beats an uninformed
prediction far from the end of the game, or independent arenas in which
further attempts no longer bring a credible gain, and names the next isolated
change.

## 2. Remove the settings that assume a defending starter

`starter_draw_value`, `repetition_contempt` and
`non_starter_draw_policy_weight` adapt search and training to a game in which
the starter defends and the other player must convert. The checked-in
configuration sets all three to their neutral value. They still run through search, self-play,
the trainer, the diagnostics and their tests.

Remove them between two training runs, not during one: the configuration file
is part of a run's recorded protocol. Stored generation reports must remain
readable afterwards.

**Done when:** the configuration, the code and the
[AlphaZero guide](alphazero-guide.md) no longer carry these settings, and
search and training produce the same results as before on fixed seeds.

## 3. Preserve the baseline while improving durability

Keep checkpoint history, replay ancestry, fixed reference and experiment
reports available for regression comparisons. Future candidates must still pass
both the paired strength arena and noisy productivity probe.

Temporary files followed by rename protect against interrupted writes, but the
complete publication sequence does not explicitly synchronize files and parent
directories against power loss. Handle checkpoint files, journal and report
before the accepted `latest` pointer, and verify recovery at each boundary.

**Done when:** interruption and power-loss guarantees are documented and tested
without weakening promotion or rewriting checkpoint history.

## 4. Profile the costs that matter for larger training runs

Game generation, promotion arenas and productivity probes occupy almost all
measured pipeline time. Measure inference batch collection, partial
requests and generation tails before changing batching or optimizer readbacks.
Use identical checkpoints, inputs and interleaved timings without concurrent
training. Preserve outputs as well as timing measurements. No fixed-workload
cost baseline exists for this board yet: record one first, with enough
repetitions for Metal timings to distinguish changes from noise.

Measure replay-buffer memory and I/O before changing its representation.
Splits and augmentation clone examples; generation files and the buffer retain
overlapping data. Any replacement must preserve MCTS targets, restart prefixes,
family identities and per-game retention. Add long-history search cases before
optimizing game cloning for the larger board.

**Done when:** a measured bottleneck is reduced on the complete workload while
behavior and artifact compatibility remain demonstrably stable.

## 5. Add useful diagnostics for learning and play

A tactical diagnostic set should cover verified captures, defence, promotion,
drops and repetitions from both colors. Keep it outside training. A small
worked example can follow one position through encoding, predictions, legal
masking, MCTS targets and a CPU optimizer update.

Investigate suspicious moves through the TUI and the existing replay format.
Retain full history, checkpoint identity and search settings. A missed capture
alone does not establish whether rules, search or the learned policy are wrong.

**Done when:** a reproducible case explains a weakness and suggests a specific
experiment instead of relying on loss curves or isolated game impressions.

## 6. Keep future learning comparisons independent

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
