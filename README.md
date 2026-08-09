# YokaiRust

YokaiRust is a fast, testable implementation of the official 3×4 rules of
**Yōkaï no Mori**. The long-term goal is an AlphaZero-powered opponent and an
animated terminal interface. The project is intentionally built in small,
verified milestones so it can also serve as a first serious Rust codebase.

## Current milestone

The repository currently contains the rules engine:

- compact typed board, pieces, hands, actions, outcomes and transitions;
- official movement, capture, promotion, parachuting and victory rules;
- threefold-repetition detection;
- canonical 132-action policy encoding for both players;
- deterministic, versioned JSON replays;
- unit and property-based tests.

MCTS, Burn and Ratatui are deliberately not dependencies of this milestone.

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

## Rules source

The engine follows the official 3×4 rulebook rather than preserving behavioral
differences from the older JavaScript and C++ implementations:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
