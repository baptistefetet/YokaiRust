# YokaiRust

YokaiRust is a fast, testable Rust implementation of the 5×6 expert rules of
**Yōkaï no Mori**—the French yōkai-themed edition of *Goro Goro shōgi*, a small
shogi variant with drops and promotions. It includes a complete
AlphaZero-style training loop, a Ratatui terminal interface and a static
WebAssembly web mode. Training depends only on game rules, self-play and the
project's own checkpoints.

The code doubles as a learning project: no `unsafe`, exhaustive rule tests,
and every public item documented.

## Documentation map

- [Rules of the 5×6 game](docs/rules.md) — the rules as the engine applies
  them, and the choices made where the rulebook leaves room.
- [Reading the Rust code](docs/reading-guide.md) — suggested reading order,
  ownership, error handling and module boundaries, explained for a Rust beginner.
- [AlphaZero in YokaiRust](docs/alphazero-guide.md) — every machine-learning
  concept the code relies on, from the glossary to the promotion gates.
- [Training results](docs/training-results.md) — the validated champion of
  the classic 3×4 game, its training baseline and independent evaluation.
- [Engine performance](docs/performance-results.md) — CPU and Metal costs
  measured on the classic 3×4 game, compatibility checks and benchmark
  protocol.
- [Next steps](docs/next-steps.md) — proposed priorities, their rationale and
  completion criteria for future work.
- [Web build guide](web/README.md) — building and deploying the browser mode.

## Current state

- Complete 5×6 rules — capture, optional and mandatory promotion, drops with
  their Kodama restrictions, victory by capture and threefold repetition —
  with typed positions, actions and versioned replays.
- Deterministic PUCT/MCTS with batched inference, and an AlphaZero pipeline
  (parallel self-play, replay buffer, paired arena, guarded promotion) built
  on Burn with WGPU/Metal acceleration on Apple Silicon.
- A Ratatui interface for local two-player games, playing the champion and
  replay analysis, plus the same engine compiled to WebAssembly for the
  browser.
- The `latest` checkpoint pointer selects the accepted champion for play and
  training. No 5×6 champion has been trained yet, so only two-player games and
  uniform-prior analysis are available on this board.
- The classic 3×4 game is preserved at the Git tag `classic-3x4`, with its
  validated champion and web build published as release `web-v1.0.11`. Its
  [training results](docs/training-results.md) and
  [performance measurements](docs/performance-results.md) remain the reference
  for the learning pipeline.

## Board coordinates

The engine keeps one absolute orientation. First starts at the bottom and moves
toward rank 6; Second starts at the top and moves toward rank 1.

```text
      a6 b6 c6 d6 e6   Second
      a5 b5 c5 d5 e5
      a4 b4 c4 d4 e4
      a3 b3 c3 d3 e3
      a2 b2 c2 d2 e2
      a1 b1 c1 d1 e1   First
```

A move is written `b3-b4`, a promoting move `c4-c5+` and a drop `kodama@a4`.

## Commands

Playing against or analyzing with the champion (`play human-vs-cpu`, `analyze`,
the web build) needs a trained model under the path configured in
`config/training.toml` (`models/…`, ignored by Git). A checkpoint trained on
another board is rejected when it is loaded: the published release archives
contain the classic 3×4 champion and only work with the `classic-3x4` tag.

```bash
# Play a local two-human match in the Ratatui interface.
cargo run -- play

# Play as First at the bottom against the latest accepted champion.
cargo run --release -- play human-vs-cpu

# Open a validated replay and step through it with the arrow keys.
cargo run -- watch path/to/game.json

# Analyze the initial position with the champion (uniform priors if none).
cargo run -- analyze [simulations] [seed]

# Validate every action and print a versioned replay.
cargo run -- replay path/to/game.json

# Start or continue AlphaZero training in the configured active paths.
# Training always resumes from the accepted champion and any persisted
# self-play games of the attempt in progress.
cargo run --release -- train --config config/training.toml --generations 15

# Evaluate every checkpoint on the buffer's validation split.
cargo run --release -- diagnose-endgames --config config/training.toml
```

### Ratatui interface

First is always displayed at the bottom and Second at the top. Use the arrow
keys or WASD to move on the board, Enter to select and play, Tab to move between
the board and the current player's hand, Escape to cancel, `N` to restart and
`Q` to quit. Number keys 1–3 select Kirin, Oni and Kodama from the hand.
When a move may or may not promote, the interface asks: `Y` promotes, `N`
keeps the piece unpromoted and Escape cancels the move.

`human-vs-cpu` loads the accepted generation referenced by `latest` under the
model path in `config/training.toml`. Model loading and deterministic MCTS run
on a background worker, leaving rendering and input responsive. The human is
First at the bottom. The same champion analyzes every human and CPU turn, so the
interface exposes the current side's root value, priors, visits, policy and Q
values. Human predictions never play a move; CPU predictions wait briefly before
the chosen move is applied and highlighted on the board.

The same right-hand panels show move history and stored replay analyses. Local
human-versus-human play does not run inference, so its prediction values remain
empty.

### Static web interface

The browser mode is a one-player game against the accepted champion. Rust owns
the complete `Game`, legal actions, encoder, MCTS and Burn network inside a Web
Worker; JavaScript only renders snapshots, so no rules or search logic is
duplicated. The output under `web/dist/` is fully static (WebGPU with an
automatic CPU fallback) — see the [web build guide](web/README.md). The
published site plays the classic 3×4 game: its prebuilt archive in the
[latest release](https://github.com/baptistefetet/YokaiRust/releases/latest)
can be extracted into any HTTPS document root. The browser presentation of the
5×6 board is not written yet.

## Rules source

The engine follows the expert section of the [official
rulebook](https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf).
[Rules of the 5×6 game](docs/rules.md) lists the points where the engine had
to interpret it.
