# Next steps

This is a backlog of proposed work, not a list of features already implemented
or experiments already run. The priorities serve the project's two goals:
understanding Rust and neural networks, and improving the accepted player's
strength with evidence. None of these tasks has started.

The [reading guide](reading-guide.md) describes the current code, and the
[training results](training-results.md) record the measurements. Keep checkpoint
identifiers in experiment reports; use the accepted champion selected by
`latest` as the starting point for future work.

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

## 6. Simplify code where reading reveals a concrete obstacle

**Why:** this should remain a project someone can understand and modify.
Additional abstractions or a larger network would also add learning overhead.

**Proposed work:** while working through the example, record specific confusing
boundaries or duplicated logic. Refactor one at a time, using existing behavior
tests. Start with the affected function or type; a long file alone is not a
reason to introduce more modules.

**Done when:** the same behavior is easier to follow and the repository checks
pass. Consider architectural or performance changes only when a measured
limitation justifies their added complexity.
