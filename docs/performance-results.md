# Engine performance measurements

## Shared position transition — 2 October 2026

`Game::repetition_count_after` now previews a checked action through the same
private position transition as `Game::apply`. It no longer allocates a temporary
game, builds its history and repetition map, or checks the action twice.
Repetition history still belongs to `Game`; decisive wins still take precedence
over a repetition draw. Position hashing and the inference batching policy are
unchanged.

The baseline uses the rules at `f82dbe1`, after the material validation fix. The
benchmark harness is committed in `ee139df`. Both measurements used the same
Mac Studio with an Apple M4 Max (12 performance and 4 efficiency cores), Rust
1.97.1, and the default Cargo release profile. The benchmark runs on one thread;
it was not pinned to a particular core. No GPU or checkpoint was used.

```bash
cargo bench --no-default-features --bench engine -- all
```

Each case warms up for one tenth of its sample iteration count, then reports
five samples. The table gives the median, with the minimum–maximum range in
parentheses. Timing excludes compilation and fixture construction. The fixed
middle-game replay has 22 plies and 24 legal actions. MCTS uses 200 simulations,
batches of eight, seed 42, uniform evaluation, no root noise, and a fresh tree
for every search.

| Case | Iterations per sample | Before | After | Median speedup |
| --- | ---: | ---: | ---: | ---: |
| Evaluation request, initial | 100,000 | 1.733 µs (1.716–1.741) | 0.614 µs (0.607–0.615) | 2.82× |
| Evaluation request, middle | 50,000 | 8.034 µs (8.032–8.120) | 2.299 µs (2.288–2.311) | 3.49× |
| Perft, initial, depth 6 | 100 | 6.940 ms (6.897–7.159) | 6.806 ms (6.775–6.818) | 1.02× |
| MCTS, initial | 500 | 0.726 ms (0.722–0.774) | 0.377 ms (0.377–0.379) | 1.92× |
| MCTS, middle | 500 | 1.055 ms (1.031–1.092) | 0.465 ms (0.465–0.466) | 2.27× |

These are local CPU measurements, not a statistical performance guarantee. The
small perft difference is not evidence of a useful move-generation speedup.
The MCTS gains apply to uniform leaf evaluation; neural inference adds work
that this benchmark does not measure. They establish neither a Metal self-play
speedup nor a playing-strength improvement.

Correctness checks include the existing rules and search tests, randomized
comparison of repetition previews with a direct history count, and a frozen
perft result of 21,323 six-ply continuations from either starting player.
An additional comparison of the pre-change and post-change executables used
1,024 deterministic random games (29,362 visited positions, 203,769 legal-action
previews and transitions, and 55 fresh MCTS searches). The digest of their
observable results matched exactly. This was a one-off comparison; the history
count property and perft reference remain in the test suite.

The stronger material validation was also checked against the retained active
buffer: 204,348 example positions and 7,482 non-empty restart-history positions
in 4,864 games satisfy the two-piece limit for each physical kind. Kodama and
promoted Kodama count together. No stored data or checkpoint was rewritten.
