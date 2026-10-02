# Next steps

The remaining priorities serve the project's two goals: understanding Rust and
neural networks, and improving the accepted player's strength with evidence.
The completed corrections below establish the baseline for the proposed work.

The [reading guide](reading-guide.md) describes the current code, and the
[training results](training-results.md) record learning experiments. The
[engine performance report](performance-results.md) records the CPU benchmark
protocol and measurements. Keep checkpoint identifiers in experiment reports;
use the accepted champion selected by `latest` as the starting point for future
work.

## Completed baseline

- Position construction and deserialization enforce material limits across the
  board and both hands; Kodama and promoted Kodama share one physical limit.
- Played moves and repetition previews share a private position transition,
  removing temporary game allocations from previews. Regression tests cover
  repetition counts and fixed-depth move counts.
- Reproducible CPU benchmarks cover evaluation-request construction, move-tree
  counting and MCTS with uniform evaluation.
- Native and browser loaders share the model-format version. The web viewport
  allows zoom, and builds warn when optional `wasm-opt` optimization is skipped.

The performance report describes the scope of the measured CPU gains. The
training results record a subsequent short continuation, its strength
evaluations and phase durations. A controlled Metal workload is still needed
to measure the effect of the engine changes on self-play throughput.

## 1. Make surprising moves reproducible

**Why:** a player observed the AI passing up an apparent immediate king capture
before eventually winning. Without the game history, this observation cannot
establish a rules, search or network bug. All terminal wins currently have the
same value, and search selects by visits, without a preference for shorter wins.

**Proposed work:** let a player export a game using the existing
[`Replay`](../src/replay.rs) format, then extend analysis to inspect a chosen
ply of that replay. Preserve the complete history because repetition affects
the result. Record the checkpoint identity, search settings, seed and backend
in technical diagnostics, without displaying a generation number to the player.
Show legal actions, network predictions and MCTS visits and values together.

**Done when:** an exported game can be validated and opened at the suspicious
turn, with enough information to check whether the capture was legal and
understand why another move was selected.

## 2. Walk through one learning example from beginning to end

**Why:** the guides explain the components, but a small worked example would
make the connection between Rust code and the learning algorithm easier to see.

**Proposed work:** follow one position through encoding, network outputs, legal
action masking, MCTS and the recorded training example. Then use a completed
game to explain the win/draw/loss target, compute the losses and perform one
optimizer update on CPU. Annotate tensor dimensions, player perspective,
ownership and the difference between inference and gradient tracking. Build
the example on the existing implementation so it stays consistent with it.

**Done when:** a beginner can run the example, locate each operation in the
source and explain what the network predicts, what search contributes and what
the optimizer changes. One update illustrates the mechanism; it does not
demonstrate a strength improvement.

## 3. Add a small, stable tactical evaluation set

**Why:** an arena score measures overall results against an opponent, but does
not explain weaknesses in captures, defence or repetition handling. Existing
rule and search tests already cover some of these cases.

**Proposed work:** extend that coverage with a few verified positions or replay
prefixes: immediate victories, avoiding an immediate loss, promotion, drops and
repetitions, from both absolute colors. Compare the network alone with MCTS at
several fixed search budgets. Keep deterministic engine tests separate from
model evaluation, which requires a checkpoint. Keep evaluation cases out of
training and reserve fresh games for independent confirmation.

**Done when:** a report identifies which cases fail, whether more search helps
and whether a candidate regresses. Any expected winning or drawing outcome
must be justified by the rules or a verified continuation.

An explicit preference for immediate wins can be considered after this
diagnostic. It would change search behavior and would need its own evaluation;
it would not show that the network had learned a better policy.

## 4. Prepare comparable training experiments

**Why:** the retained buffer contains games with incomplete ancestry, which
limits validation claims. Fewer generations also need not mean less computation.

**Proposed work:** define the hypothesis, budgets, random seeds, promotion gates
and fixed reference before running an experiment. Use isolated model and data
directories; when clean validation is required, generate fresh self-play with
complete family tracking and keep related examples in the same partition.
Preserve all existing checkpoints and reports.

**Done when:** a written protocol compares candidates from the same starting
checkpoint and data conditions, recording self-play, optimization, evaluation
and elapsed time. A claim about learning efficiency requires repeated runs at
comparable compute budgets; an independent, preplanned arena confirms strength.
Repeatedly trying candidates until one passes is not that confirmation.

## 5. Try one targeted learning change, then reassess

**Why:** the current evidence supports a stronger champion, but does not yet
identify which change would produce one faster.

**Proposed work:** use the diagnostics to choose one experiment: adjust the
scalar value loss weight, or revisit decisive endgame sampling, whose extra
sampling currently ends after bootstrap. Keep the architecture and other
settings fixed. Compare against an unchanged control under the protocol above.
Only then decide whether another short block of training is useful.

**Done when:** the report states whether the change improves playing strength
at comparable cost, passes the paired arena and noisy productivity gates, and
preserves tactical performance. An inconclusive or negative result is worth
recording. Rejected candidates never replace the accepted training source.

## 6. Measure and tune inference batching

**Why:** the self-play inference threshold can require every worker to submit
a full batch of leaves, although terminal nodes and initial root expansion
produce smaller requests. The service can then wait for its timeout while no
worker can submit more work. The frequency and cost on Metal remain unmeasured.

**Proposed work:** record collection time, batch sizes and why each batch starts
(threshold reached or timeout). Measure a fixed self-play workload, including
the end of a generation when fewer games remain. Compare thresholds and worker
counts one change at a time. If coordination by request count is warranted,
track producers that can actually submit work, rather than all live client
handles. Keep search settings and the reference checkpoint fixed.

**Done when:** a reproducible report compares elapsed self-play time and batch
utilization before and after the change. Tests cover partial requests, worker
completion and the maximum backend batch size. Mean client latency alone does
not establish how much time was spent collecting a batch.

## 7. Simplify the remaining search and inference boundaries

**Why:** this should remain a project someone can understand and modify.
The shared position transition removes one concrete source of overhead and
duplication. Evaluation still has boundaries worth examining before adding
more abstractions.

**Proposed work:** use the worked example and benchmarks to choose one focused
change: avoid preparing neural context for evaluators that do not use it,
remove the browser evaluator's unused synchronous implementation, or share
the common native/browser inference transformations. Keep historical context
in cache keys wherever it is part of the network input. Extract small pipeline
helpers when they clarify a phase; retain the existing recovery tests.

**Done when:** the same behavior is easier to follow and the repository checks
pass. Any claimed performance gain is measured with the relevant CPU or neural
workload. A long file alone is not a reason to introduce more modules.

## 8. Strengthen persistence, then assess storage costs

**Why:** temporary files followed by rename protect against interrupted writes,
but the current code does not explicitly make files and directory updates
durable against a power failure. The replay buffer also duplicates examples
stored in generation files and is rewritten in full. Its full read occurs at
training startup, before the generation loop.

**Proposed work:** first handle durability across the complete publication
sequence: checkpoint files, journal and report, then the accepted `latest`
pointer, with the appropriate file and directory synchronization. Verify
recovery at each boundary. Separately measure buffer size, peak memory and
read/write time before selecting a different storage representation.

**Done when:** publication ordering and recovery guarantees are documented and
tested. Any later storage migration preserves checkpoint history, provenance,
restart prefixes, trajectory-family validation splits and per-game retention.
Reconstructing examples from replays must retain their MCTS policy targets.

## Deferred larger changes

Revisit these only when the measurements or reading work identify a concrete
need:

- Separate Metal/CPU features before considering further crate boundaries in
  the existing workspace.
- Consider a replay-buffer manifest or compact binary format after measuring
  storage costs; keep migration separate from durability fixes.
- Consider a fully separated MCTS driver, distributed encoding or a persisted
  phase state machine only if smaller refactorings leave a clear obstacle.
- Reprofile CPU search before introducing packed position keys, incremental
  history fingerprints, smaller tree nodes or tree compaction. The new material
  invariant makes compact keys feasible, but their benefit still needs measuring.
- Treat replacing Phaser as a separate web project, supported by loading or
  interaction measurements.
