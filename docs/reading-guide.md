# Reading YokaiRust as a C++ developer learning Rust

This guide is a suggested route through the repository. It assumes solid C++
experience, but no prior Rust fluency. The goal is not to teach all of Rust; it
is to explain the language features that this project actually uses.

## Recommended reading order

1. [`game.rs`](../src/game.rs): domain types, board representation and rules.
2. [`policy.rs`](../src/policy.rs): the bijection between legal actions and the
   132 neural-network outputs.
3. [`replay.rs`](../src/replay.rs): a small example of validated serialization.
4. [`search.rs`](../src/search.rs): PUCT and the contiguous node arena.
5. [`neural.rs`](../src/neural.rs) and the `neural/` directory: canonical input,
   residual network, checkpoints and batched inference.
6. [`training/data.rs`](../src/training/data.rs): supervised examples and the
   rolling replay buffer.
7. [`training/trainer.rs`](../src/training/trainer.rs): losses and optimization.
8. [`training/pipeline.rs`](../src/training/pipeline.rs): orchestration only;
   read it after understanding the components it calls.
9. [`main.rs`](../src/main.rs): CLI parsing and presentation of progress events.

The integration tests are often the shortest executable specification. Start
with [`tests/engine.rs`](../tests/engine.rs), then `search.rs` and `training.rs`.
For machine-learning vocabulary, read the glossary at the start of
[`alphazero-guide.md`](alphazero-guide.md); the code-reading guide assumes those
terms but does not assume prior neural-network experience.

## A C++ to Rust translation table

| Rust in this project | Approximate C++ mental model | Important difference |
| --- | --- | --- |
| `enum Player` | `enum class Player` | Rust enums can also carry typed payloads, as `Action` and `Outcome` do. |
| `struct Position` | Small value-type class | Its fields are private and invariants are established by constructors. |
| `Option<T>` | `std::optional<T>` | Pattern matching makes the empty case explicit. |
| `Result<T, E>` | `std::expected<T, E>` | `?` returns the error to the caller after an automatic conversion. |
| `&T`, `&mut T` | `T const&`, `T&` | The borrow checker proves aliasing rules at compile time. |
| `Box<T>` | `std::unique_ptr<T>` | Ownership is still unique, but moving is the default operation. |
| `Arc<T>` | `std::shared_ptr<T>` | Atomic shared ownership; mutability still requires synchronization. |
| `trait Evaluator` | Interface/concept | Generic call sites use static dispatch unless `dyn Trait` is requested. |
| `match` | Exhaustive `switch` plus destructuring | Adding an enum variant forces every relevant match to be revisited. |
| iterator chains | `<algorithm>` and ranges | Iterators are lazy and normally compile to ordinary loops. |

## Ownership choices used here

`Position` is a compact, `Copy` value. Copying it is deliberate: MCTS creates
many temporary game states and a 3x4 board is cheaper and clearer as a value than
behind shared ownership.

`Game` is not `Copy`. It owns action and position histories because repetition
is path-dependent. A simulation clones a `Game`, applies actions to the clone,
and leaves the caller unchanged.

Large neural models are moved into an `InferenceService`. Worker games only
clone an `InferenceClient`, which is a small channel handle comparable to a
thread-safe façade around one GPU worker.

When reading a signature, use this checklist:

- `T`: ownership enters the function;
- `&T`: shared read-only borrow;
- `&mut T`: exclusive mutable borrow;
- returned `T`: ownership leaves the function;
- `T: Trait`: compile-time capability requirement, similar to a C++ concept.

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

The following boundaries are intentional and should remain stable for the TUI:

- `Position` stores absolute board orientation.
- Neural encoding alone canonicalizes the player-to-move perspective.
- `Game` owns repetition history and the official outcome.
- `Action` is the UI, replay and engine move type.
- `PolicyIndex` is only the neural representation of an action.
- Illegal policy logits are masked before probabilities reach MCTS.
- Repetition contempt changes self-play search, never official outcomes.

These boundaries mean the future Ratatui layer can render a `Game`, submit an
`Action`, display `ActionAnalysis`, and navigate a `Replay` without depending on
Burn or the training pipeline.

## Reading the MCTS arena

`Mcts` stores nodes in one `Vec<Node>` rather than allocating a polymorphic tree.
Each node records `first_child` and `child_count`; its children are a contiguous
slice of indices. This resembles a cache-friendly C++ object pool.

Values are stored from the node's player-to-move perspective. Moving from a
child back to its parent changes player, so backpropagation negates the value at
each level. Consequently PUCT reads a child's value as `-child.mean_value()`.
The sign convention is tested explicitly in `tests/search.rs`.

## Reading one training generation

The top-level function in `training/pipeline.rs` should now read as these named
steps:

1. load the accepted champion;
2. generate self-play and persist replays;
3. select whole-game train/validation splits;
4. restore the trainable weights and Adam moments, then apply a fixed number of
   sampled mini-batch updates;
5. save the candidate without changing the champion;
6. `run_official_arena` against the champion;
7. `run_candidate_diagnostics` with mirror and exploratory draw gates;
8. publish the candidate only when all three gates pass.

Progress is represented as the `TrainingProgress` enum. The pipeline emits data;
`main.rs` decides how to print it. Ratatui can later consume the same events.

## Why the training code is split into modules

The module boundaries are dependency boundaries, not arbitrary file sizes:

- `data.rs` knows examples, replays and buffers, but no GPU or optimizer;
- `trainer.rs` knows tensors, losses and Adam, but no self-play or promotion;
- `self_play.rs` knows how to produce games with an `Evaluator`, but not whether
  that evaluator is a neural network, a test double or another backend;
- `arena.rs` compares two `Evaluator` implementations without training either;
- `pipeline.rs` owns sequencing, persistence and promotion, not math details;
- `main.rs` owns command-line input and textual output, not training policy.

This resembles C++ code where domain objects, algorithms, persistence and UI
are separate libraries behind small interfaces. The important Rust difference
is that `Evaluator` is usually a generic type parameter. Calls are statically
dispatched and monomorphized, like templates constrained by a concept; there is
no virtual-call requirement.

The batching service uses one owning `InferenceService` and many cloneable
`InferenceClient` handles. Think of the service as an RAII object containing a
worker `std::jthread`, and the clients as small producers holding channel
senders. Dropping the service sends shutdown and joins the worker, so thread
lifetime is expressed by ownership rather than a separate global manager.

`TrainingProgress` is an enum carrying different payloads. It plays the role of
a closed `std::variant` event protocol. `pipeline.rs` produces events and
`main.rs` exhaustively matches them; adding a new variant makes the compiler
identify every consumer that must handle it.

## Safe modification workflow

After changing rules, policy encoding or repetition behavior, run:

```bash
cargo test --test engine
cargo test --test properties
cargo test --test search
```

After changing training code, also run:

```bash
cargo test --test training
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The ignored tests in `tests/performance.rs` are local Metal benchmarks. They are
not part of the fast correctness suite because they load saved checkpoints and
can take minutes.

## A practical way to learn from the code

For each module, first read its public structs and function signatures, then the
corresponding tests, and only then the implementation. Try small changes such as
adding a diagnostic field or a test position before editing generic Burn code.
Rust compiler errors are often the most precise explanation of an ownership or
type relationship; solve the first error first, because later errors may be
cascades.
