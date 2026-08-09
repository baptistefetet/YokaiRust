# YokaiRust

YokaiRust is a fast, testable implementation of the official 3×4 rules of
**Yōkaï no Mori**. The long-term goal is an AlphaZero-powered opponent and an
animated terminal interface. The project is intentionally built in small,
verified milestones so it can also serve as a first serious Rust codebase.

## Current milestone

The repository currently contains the rules engine and deterministic MCTS:

- compact typed board, pieces, hands, actions, outcomes and transitions;
- official movement, capture, promotion, parachuting and victory rules;
- threefold-repetition detection;
- canonical 132-action policy encoding for both players;
- deterministic, versioned JSON replays;
- batch-oriented policy/value evaluators with a bounded prediction cache;
- PUCT search in a contiguous node arena, including safe subtree reuse;
- self-play-only Dirichlet noise and a configurable temperature schedule;
- legal-policy masking plus per-action prior, visits, policy and Q diagnostics;
- optional MCTS analyses embedded in validated replay files;
- unit and property-based tests.

Burn and Ratatui are deliberately not dependencies of this milestone.

## Board coordinates

The engine keeps one absolute orientation. First starts at the bottom and moves
toward rank 4; Second starts at the top and moves toward rank 1.

```text
      a4 b4 c4   Second
      a3 b3 c3
      a2 b2 c2
      a1 b1 c1   First
```

Moves use `b2-b3`. Drops use a full piece name such as `kodama@a4`.

## Development

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --check
```

Release builds use thin LTO and a single code-generation unit. Runtime data,
self-play datasets and trained models are ignored by Git.

## Text analysis and replays

Until the TUI milestone, the binary exposes two deliberately small diagnostic
commands without pulling in a CLI framework:

```bash
# Pure MCTS from the official initial position. Defaults: 200 simulations, seed 42.
cargo run -- analyze [simulations] [seed]

# Validate every action and print a versioned Rust replay.
cargo run -- replay path/to/game.json
```

`Mcts::search` never adds exploration noise and is suitable for arenas or human
play. `Mcts::search_self_play` is the explicit self-play entry point and is the
only one that adds Dirichlet noise. A fixed seed reproduces both the tree search
and temperature-based action sampling.

## Rules source

The engine follows the official 3×4 rulebook rather than preserving behavioral
differences from the older JavaScript and C++ implementations:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
