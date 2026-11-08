# Next steps

The current milestone keeps the 3×4 game and accepted model compatible while
simplifying shared code. The [reading guide](reading-guide.md) describes the
implementation; [training results](training-results.md) retain measurement
provenance and checkpoint identities. The expert game is a later project.

## Completed cleanup

- MCTS checks the complete parent history before reusing a subtree. Different
  positions or histories reset the tree instead of retaining unrelated nodes.
- Native and browser search share neural inference and model metadata
  validation. Asynchronous evaluators no longer need a dummy synchronous method.
- Training and inference share repetition context and history extraction.
  Tests compare live requests with reloaded examples, including restarts.
- Board and policy dimensions derive from their definitions. Rust snapshots
  supply the logical board size to the browser; visual asset dimensions remain
  presentation settings.
- Board-move enumeration and the early-exit legality check use one traversal;
  attack detection and move validation share movement geometry.
- CI checks native code and both mutually exclusive WASM backends separately.

These refactorings do not change the network architecture, targets or artifact
versions. A fresh training run is not required to use them. Existing checkpoints
and experiment reports remain useful and are retained.

## 1. Make observed moves reproducible

A player reported the AI passing up an apparent immediate king capture. Without
the full game history, this does not establish a rules, search or network bug.

Use the existing `Replay` format to export games and inspect a chosen ply.
Technical diagnostics should include the checkpoint identity, search settings,
seed and backend, alongside legal actions, network predictions and MCTS visits.
Preserve history because repetition affects both rules and neural input.

**Done when:** an exported game can reproduce the suspicious turn and explain
whether the capture was legal and why another move received more visits.
Winning sooner is not currently a separate search or training objective.

## 2. Continue cleanup where it removes a concrete burden

The generation pipeline still combines phase orchestration with dataset
preparation, candidate recovery and persistence. Extract small phase helpers
when they clarify these responsibilities, preserving the recovery tests.
Browser transport and scene rendering can likewise be separated when their
lifecycle or error handling needs work.

Measure replay-buffer memory and I/O before changing its representation. Dataset
splits and augmentation currently clone examples; generation files and the
buffer duplicate stored targets. Any replacement must preserve MCTS targets,
restart prefixes, family identities and per-game retention.

**Done when:** a specific responsibility is easier to follow or a measured cost
is reduced, without adding a parallel implementation or changing play.

## 3. Strengthen publication durability

Temporary files followed by rename protect against interrupted writes, but the
complete publication sequence does not explicitly synchronize files and parent
directories against power loss. Handle checkpoint files, journal and report
before the accepted `latest` pointer, then verify recovery at each boundary.

**Done when:** interruption and durability guarantees are documented and tested.
Checkpoint history must remain intact; storage migration is a separate change.

## 4. Establish diagnostics before another learning experiment

A small worked example should follow one position through encoding, predictions,
legal masking, MCTS targets and one CPU optimizer update. A separate tactical
set should cover verified captures, defence, promotion, drops and repetitions,
from both colors. Keep evaluation cases out of training.

Measure inference batching on a fixed Metal workload, including partial leaf
requests and generation tails. Collection time and the reason a batch starts
are more informative than mean client latency alone.

**Done when:** diagnostics explain whether a weakness comes from rules,
representation, predictions or search, and measurements identify a useful
experiment rather than suggesting one from loss curves alone.

## 5. Choose a controlled training experiment

Define the hypothesis, compute budgets, seeds, promotion gates and fixed
reference before starting. Keep the reference identifier in the report, and
retain the checkpoint for independent comparisons. Start from `latest`; rejected
candidates never become self-play sources.

A fresh run is appropriate when testing a changed representation or when clean
trajectory ancestry is required. Use separate model/data directories and keep
prior reports as historical evidence. Otherwise, try one targeted loss-weight
or decisive-endgame sampling change before changing architecture.

**Done when:** repeated comparable runs and an independent, preplanned arena
support any claimed learning improvement. Promotion still requires the paired
strength test and noisy productivity probe.

## Deferred: expert mode

Supporting a 5×6 game requires its own rules, action representation, model and
artifact identity. Shared dimensions alone do not implement that mode; the
current compact policy index and fixed tensors deliberately describe 3×4.
Design the two variants together only after this baseline is stable. Do not add
speculative generic engines or rewrite checkpoint history in preparation.
