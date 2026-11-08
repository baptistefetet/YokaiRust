# Reading the Rust code

This guide is a suggested route through the repository for someone learning
Rust and neural networks. It explains the language features that this project
uses through their role in the code. No knowledge of another systems
programming language is required.

## Recommended reading order

1. [`tests/engine.rs`](../tests/engine.rs): start here — the integration tests
   are the shortest executable specification of the official rules, and each
   test name states the rule it pins down.
2. [`notation.rs`](../src/notation.rs): a compact, complete
   `Display`/`FromStr` round trip with typed errors — a comfortable first
   contact with idiomatic Rust.
3. [`game.rs`](../src/game.rs): domain types, board representation and rules.
4. [`policy.rs`](../src/policy.rs): the bijection between legal actions and the
   132 neural-network outputs.
5. [`replay.rs`](../src/replay.rs): a small example of validated serialization.
6. [`search/mod.rs`](../src/search/mod.rs): PUCT and the contiguous node arena; its
   executable specification is [`tests/search.rs`](../tests/search.rs).
7. [`neural.rs`](../src/neural.rs) and the `neural/` directory: canonical input,
   residual network, shared inference and model metadata; native checkpoints
   add filesystem storage around that metadata.
8. [`training/data.rs`](../src/training/data.rs): supervised examples and the
   rolling replay buffer.
9. [`training/trainer.rs`](../src/training/trainer.rs): losses and optimization.
10. [`training/pipeline.rs`](../src/training/pipeline.rs): orchestration only;
    read it after understanding the components it calls, alongside
    [`tests/training.rs`](../tests/training.rs).
11. [`main.rs`](../src/main.rs): CLI parsing and presentation of progress
    events.
12. [`ui.rs`](../src/ui.rs) and [`ui/ai.rs`](../src/ui/ai.rs): the Ratatui
    interface — a place to make a first visible change. `ui/ai.rs` is an example
    of the channel-plus-worker-thread pattern.
13. [`web/crate/src/lib.rs`](../web/crate/src/lib.rs): the WebAssembly
    boundary — the same engine driven asynchronously from the browser.

For machine-learning vocabulary, read the glossary at the start of
[`alphazero-guide.md`](alphazero-guide.md); this code-reading guide assumes
those terms but does not assume prior neural-network experience.

Public Rust items use `///` API documentation, and each source module
starts with a `//!` overview. `cargo doc --no-deps --open` builds a browsable
local reference with links between those types. Comments inside functions are
reserved for invariants and design choices that the code alone cannot explain.

## Rust concepts used in the project

| Rust construct | Meaning | Example in this project |
| --- | --- | --- |
| `enum` | A value with one of a defined set of variants; each variant can carry its own data. | `Player` is `First` or `Second`; `Action` carries the details of a move or drop. |
| `struct` | A value grouping named fields. | `Position` keeps its fields private so constructors can validate the board. |
| `Option<T>` | Either `Some(value)` of type `T`, or `None`. | A board square contains `Option<Piece>`: a piece or an empty square. |
| `Result<T, E>` | Either `Ok(value)` of type `T`, or `Err(error)` of type `E`. | Applying a move returns a transition or a `MoveError`. |
| `&T`, `&mut T` | Borrow a value without taking ownership; `&mut T` gives exclusive access for mutation. | Reading a `Game` uses a shared borrow; applying an action requires a mutable borrow. |
| `Box<T>` | Own a value stored in a separate heap allocation. | Large search results can travel in events without making every event equally large. |
| `Arc<T>` | Share ownership through a thread-safe reference count. Cloning the handle shares the same value. | Inference clients share service state across game workers. |
| `trait` | A set of operations that a type promises to implement. | `Evaluator` lets search obtain predictions from different backends. |
| `match` | Choose a branch by a value's variant and access the data it carries. Every possible case must be covered. | The CLI handles each kind of `TrainingProgress` event. |
| iterator chains | Describe how to process a sequence; operations run when the iterator is consumed. | `map` transforms actions into probabilities, and `collect` builds the resulting vector. |

## Ownership choices used here

`Position` is a compact, `Copy` value. Copying it is deliberate: MCTS creates
many temporary game states and a 3x4 board is cheaper and clearer as a value than
behind shared ownership.

`Game` is not `Copy`. It owns action and position histories because repetition
is path-dependent. A simulation clones a `Game`, applies actions to the clone,
and leaves the caller unchanged.

The private `Position::after_legal_action` transition copies just the board and
hands, then computes captures, promotion and decisive wins. Both `Game::apply`
and the repetition preview call it after checking legality. Only `Game` consults
the earlier positions to detect a draw. Previewing a move therefore needs no
temporary history or repetition map, and the movement rules live in one place.
A lazy board-move traversal serves both full enumeration and the early-exit
check for whether any move exists.

Large neural models are moved into an `InferenceService`. Worker games only
clone an `InferenceClient`, a small handle used to send prediction requests to
the service. Cloning a client does not clone the network.

When reading a signature, use this checklist:

- `T`: passed by value, moving ownership or copying the value when `T` is `Copy`;
- `&T`: shared read-only borrow;
- `&mut T`: exclusive mutable borrow;
- returned `T`: ownership leaves the function;
- `T: Trait`: the compiler requires type `T` to implement the named trait.

## Error handling

Library code returns typed errors with `Result`. For example:

```rust
pub fn apply(&mut self, action: Action) -> Result<Transition, MoveError>
```

The caller must handle success or failure. Inside a function returning a
compatible `Result`, this:

```rust
game.apply(action)?;
```

means: apply the action, extract the `Transition` on success, otherwise return
the converted error immediately. It is not an exception and performs no stack
unwinding.

`expect` is mostly confined to tests or invariants that cannot be violated by a
valid Yokai position. Recoverable runtime failures use `Result` instead.

## Domain invariants

The following boundaries connect the rules, network and interfaces:

- `Position` stores absolute board orientation.
- Policy mapping and neural encoding share the player-to-move orientation.
- `Game` owns repetition history and the official outcome.
- `Action` is the UI, replay and engine move type.
- `PolicyIndex` is only the neural representation of an action.
- MCTS selects legal policy probabilities and renormalizes them before expansion.
- Repetition contempt changes self-play search, never official outcomes.

These boundaries let the Ratatui layer (`src/ui.rs`) render a `Game`, submit an
`Action`, display `ActionAnalysis`, and navigate a `Replay` while confining
Burn and champion loading to its background worker (`src/ui/ai.rs`).

## Reading the MCTS arena

`Mcts` stores nodes in one `Vec<Node>`, a growable array.
Each node records `first_child` and `child_count`; its children are a contiguous
range of indices into that array. A child is reached by indexing the vector,
so each node does not need its own allocation or pointers to other nodes.

Values are stored from the node's player-to-move perspective. Moving from a
child back to its parent changes player, so backpropagation negates the value at
each level. Consequently PUCT reads a child's value as `-child.mean_value()`.
The sign convention is tested explicitly in `tests/search.rs`.

The root retains the complete position history. Reuse requires either that
exact history or its extension by the action just played. A different path
resets the tree, even when it reaches the same board: repetition and the neural
history would otherwise make old evaluations invalid.

## Following the native and browser paths

`NetworkEvaluator` in `neural/evaluator.rs` prepares tensors, runs the network
and decodes policy/WDL probabilities. Native search uses its synchronous
`Evaluator` implementation. Browser search uses `AsyncEvaluator` to await the
GPU readback. Both call the same prediction and decoding code; an evaluator
only needs the trait required by its search entry point.

`EvaluationRequest::from_game` collects recent positions and action-specific
repetition counts. The recorder and encoder use the same helpers, so training
and inference agree about historical context. The cache key includes that
context: identical boards can still need different predictions.

`ModelMetadata` in `neural/metadata.rs` is shared by native and browser loaders.
It rejects incompatible formats, encoders, rules, policy widths and empty
network layers before loading weights. `checkpoint.rs` adds native file I/O and
optimizer persistence; the browser adapter loads exported bytes.

The rules define board dimensions. Tensors and terminal layout derive their
sizes from those constants; browser snapshots carry rows and columns for
JavaScript. Policy width derives from the board, movement directions and hand
piece list. These are still the fixed 3×4 rules and model representation.

For the complete learning sequence and persistence boundaries, follow
[One generation](alphazero-guide.md#one-generation) alongside
`training/pipeline.rs`. It emits `TrainingProgress` data; `main.rs` formats it.
Start at `run_generation_with_progress`: it reserves or resumes the attempt,
then calls four private phase helpers in order: `load_or_generate_self_play`,
`prepare_dataset`, `load_or_train_candidate` and `evaluate_candidate`.
`complete_generation` saves the decision before publishing it. `PreparedDataset`
groups the selected examples and their sampling statistics so they stay together
through optimization and reporting; it does not introduce a new file format.

## Why the training code is split into modules

The module boundaries are dependency boundaries, not arbitrary file sizes:

- `data.rs` knows examples, replays and buffers, but no GPU or optimizer;
- `trainer.rs` knows tensors, losses and Adam, but no self-play or promotion;
- `self_play.rs` knows how to produce games with an `Evaluator`, but not whether
  that evaluator is a neural network, a test double or another backend;
- `arena.rs` compares two `Evaluator` implementations without training either;
- `pipeline.rs` owns sequencing, persistence and promotion, not math details;
- `main.rs` owns command-line input and textual output, not training policy.

`Evaluator` is usually used through a generic type parameter. For example,
`Mcts<E>::search` requires `Evaluator`, while `search_async` requires
`AsyncEvaluator`; construction and tree management need neither trait. The
compiler produces code for the concrete evaluator types used by the program; this is
called monomorphization. The search algorithm can therefore be reused without
choosing an evaluator implementation at each call during execution.

The batching service uses one owning `InferenceService` and many cloneable
`InferenceClient` handles. The service owns the background worker thread;
clients send requests through channels, which transfer messages between threads.
When the service is dropped, Rust calls its `Drop` implementation. That code
requests shutdown and waits for the worker to finish, tying thread cleanup to
the lifetime of the owning value.

## Safe modification workflow

Tests focus on observable behavior: legal play, search decisions, learning
objectives, dataset partitioning and recovery after interrupted training.
Use complete save/load scenarios for persistence contracts. Avoid separate
tests that merely repeat defaults, error wording or presentation labels.
Keep fixtures small and share board construction through `tests/common`.

After changing rules, policy encoding or repetition behavior, run:

```bash
cargo test --test engine
cargo test --test properties
cargo test --test search
```

Before pushing changes, run the complete checks:

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
cargo doc --no-deps
```

Local Metal benchmarks live in `benches/performance.rs` and run only when
named explicitly (`cargo bench --bench performance -- <name>|all`). They are
not part of the correctness suite because they load saved checkpoints, can
take minutes, and measure speed rather than verify behavior.

For CPU rules and search timings without a GPU or saved model, run:

```bash
cargo bench --no-default-features --bench engine -- all
```

Replace `all` with `request`, `perft`, or `mcts` to select cases. The protocol
and measurements live in [Engine performance](performance-results.md).

The CI also checks both browser backends separately. `flex` and `webgpu` are
mutually exclusive, so a workspace-wide `--all-features` is not suitable:

```bash
cargo clippy --locked -p yokai-web --target wasm32-unknown-unknown --no-default-features --features flex -- -D warnings
cargo clippy --locked -p yokai-web --target wasm32-unknown-unknown --no-default-features --features webgpu -- -D warnings
```

After web changes, follow the [web build and smoke-test workflow](../web/README.md).
The Metal training-state round-trip test is a separate diagnostic, run with
`cargo test --test neural metal_training_state_round_trip_runs_one_real_update -- --ignored --exact`.
It needs access to a real GPU and is not a regression gate.
