# YokaiRust

YokaiRust is a fast, testable implementation of the official 3×4 rules of
**Yōkaï no Mori**. It now includes the first complete AlphaZero training loop;
the next major milestone is the animated terminal interface. The project is
intentionally built in small, verified milestones so it can also serve as a
first serious Rust codebase.

## Current milestone

The repository currently contains the rules engine, deterministic MCTS and the
first trainable AlphaZero pipeline:

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
- a versioned 130-plane canonical neural encoder covering eight successive
  states plus the starter role, with horizontal augmentation;
- action-aligned repetition context for all 132 policy actions;
- a configurable Burn residual network with 132 policy logits and a WDL head;
- CPU inference for deterministic tests and WGPU/Metal inference on Apple Silicon;
- a batching inference service shared by concurrent self-play games;
- generation-based SafeTensors checkpoints with validated metadata and separate
  atomic pointers for the accepted champion and private learner lineage;
- reproducible parallel self-play, a rolling replay buffer and whole-game
  training/validation splits;
- deterministic cycle-adjacent restarts for a configurable share of self-play;
- the complete AlphaZero dataset, with temporary decisive-tail oversampling
  during bootstrap;
- explicit policy/WDL metrics, including entropy, calibration, illegal policy
  mass and top-1 accuracy;
- a paired, color-alternating promotion arena against the champion, using
  reproducible short openings to avoid replaying one deterministic game;
- noise-free mirror and noisy self-play gates against repetition cycles;
- unit and property-based tests.

Ratatui is deliberately deferred to the next milestone.

## Learning and code-reading guides

- [Reading YokaiRust as a C++ developer learning Rust](docs/reading-guide.md)
- [AlphaZero in YokaiRust](docs/alphazero-guide.md)

The first guide maps the Rust constructs used here to familiar C++ concepts and
suggests a module-by-module reading order. The second explains policy/value
targets, guarded champion updates, draw diagnostics and why continued self-play
is not a proof of perfect play.

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

Release builds use Cargo's portable defaults. Runtime data, self-play datasets
and trained models are ignored by Git.

Cargo downloads crate sources into `~/.cargo/registry` and compiles this
project into its local `target/` directory. Rust toolchains live in `~/.rustup`.
No package is installed globally by the build. The generated `models/` and
`data/self-play/` directories remain inside this project.

## Text analysis and replays

Until the TUI milestone, the binary exposes deliberately small diagnostic
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

## AlphaZero training

The checked-in configuration targets Metal and is intentionally substantial:
256 self-play games, 400 optimizer steps, a 200-game paired arena, a 64-game
mirror diagnostic and a 64-game exploratory probe. Self-play runs 16 concurrent
games and selects 8 distinct MCTS leaves per game before each inference, which
feeds Metal efficiently without requiring hundreds of blocked game threads. The
arena uses sequential PUCT (`search_batch_size = 1`), so virtual losses cannot
influence its measurement. Workers are concurrent games, not a promise to keep
the same number of CPU cores busy.

The default loop now protects self-play with an accepted champion. A trained
candidate is promoted only when it scores at least 55% in the paired arena,
has no deterministic mirror draw from either absolute starting orientation and
stays below 20% draws in its noisy self-play probe. Rejected checkpoints remain
available for diagnosis, but never generate later training data. Self-play adds
Dirichlet noise at every
root, samples from MCTS visits for the first 12 plies, then becomes greedy. All
non-terminal positions produced by the champion enter the rolling buffer and
official draws always have value zero. The checked-in bootstrap additionally
discourages the player that causes a repetition inside self-play MCTS and
temporarily oversamples decisive tails; neither changes stored game outcomes.

```bash
# Run one generation with the checked-in Metal configuration.
cargo run --release -- train --config config/training.toml --headless

# Resume automatically from the champion and replay buffer.
cargo run --release -- train --resume latest --headless

# Run five successive candidate generations from the current champion.
cargo run --release -- train --resume latest --generations 5 --headless
```

`--headless` disables the future TUI, not textual diagnostics. Progress is
written immediately to standard error: roughly twenty updates during self-play,
one metrics line per 100 optimizer steps, roughly twenty updates per arena, and
every checkpoint publication. Arena updates include the running
candidate/previous/draw counts and provisional score. All lines include elapsed
wall-clock time. Phase summaries also report average/maximum inference batch,
backend throughput and client wait. One generation is run by default; use
`--generations N` to request an explicit sequence.

Arena progress is completion-ordered because games run concurrently. Fast wins
can therefore appear as a block before slower losses even though paired game
indices alternate the candidate between `First` and `Second`. The final report
separates both assignments; intermediate counters must not be read as a
chronological winning streak.

This was first noticed by replaying the old generation-8 arena exactly. The
first 100 completed games were candidate wins and the final 100 were losses.
Completion order explained the visible blocks, but a later audit found the more
important issue: without randomized openings, those 200 games represented only
a few deterministic trajectories. The current log reports `distinct_openings`
so this cannot remain hidden.

On the development M4 Max, the terminal-8 experiment took about 2 minutes 40
seconds. A measured full-depth terminal-32 generation including the new mirror
gate took about 6 minutes 45 seconds: 3 minutes 20 for self-play, 17 seconds for
training, 2 minutes 15 for the paired arena and 53 seconds for the mirror gate.
This excludes a one-time 30-second incremental release relink. The mirror probe
now covers both absolute starters in four games because running 64 deterministic
copies was redundant. The original one-leaf implementation projected roughly
16 minutes for self-play alone. These
figures are measurements, not guarantees; game length and model behavior change
between generations.

Set `backend = "cpu"` in the TOML file for deterministic debugging without
Metal. The command bootstraps generation zero when no model exists, then:

1. generates self-play games from the champion;
2. updates the rolling replay buffer and trains the next network;
3. saves the candidate without changing the champion;
4. evaluates it against the champion with paired seeds and colors;
5. measures its noise-free mirror and noisy self-play draw rates;
6. publishes it only if the strength arena and both draw gates pass.

Training performs a fixed `steps_per_generation` number of uniformly sampled
mini-batch updates. Its work therefore does not silently grow with the replay
buffer. Adam moments are stored with every generation and restored before the
next one; the log reports `Adam resumed=true` when continuity was recovered.
Validation is diagnostic and runs every `validation_interval_steps` updates.

The enabled terminal-window schedule emphasizes decisive tactics without manual
generation-by-generation edits. The complete replay buffer always stays in the
training set; terminal positions are additionally sampled until they reach
`decisive_fraction`. Their window starts at `initial_plies`, is multiplied by
`growth_factor` each candidate attempt, and the extra sampling ends at
`full_dataset_generation`. Reports store the effective window and number of
extra examples, so a resumed run follows the same deterministic curriculum.

Each invocation completes the requested number of candidate generations.
Checkpoints are written under `models/generation-N/`; for compatibility the
`models/latest` pointer now identifies the accepted champion, not merely the
newest saved candidate. Each new checkpoint also contains `training-model.bin` and
`optimizer.bin`; these make optimizer resume exact but use additional disk space.
Existing `models/champion` pointers are still accepted during migration.
The replay buffer is stored in
`data/self-play/buffer.json`, and structured generation summaries are stored in
`data/self-play/reports/`. Writes use
temporary paths followed by atomic renames so an interrupted write cannot replace
the last complete generation. After `Ctrl+C`, rerunning the command starts again
from the last accepted champion.

### Continuous AlphaZero experiment (August 2026)

A clean run from random generation 0 was stopped after generation 15 once the
same draw regime had repeated for several generations. It used the standard
configuration: all positions, opening temperature 1, official draw value 0 and
no repetition contempt or promotion gate. This run is the baseline from before
the fixed-step correction: Adam was reset at every generation and full-buffer
epochs made the number of optimizer updates grow with the buffer.

| Generation | Self-play draws | Previous-generation arena | Mirror draws | Exploratory draws | Validation policy/value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4/256 | 200/0/0 | 0/64 | 1/64 | 2.619 / 1.159 |
| 3 | 10/256 | 100/100/0 | 0/64 | 12/64 | 1.819 / 0.802 |
| 4 | 49/256 | 0/100/100 | 64/64 | 7/64 | 1.643 / 0.701 |
| 7 | 22/256 | 200/0/0 | 0/64 | 15/64 | 1.421 / 0.748 |
| 10 | 54/256 | 100/0/100 | 64/64 | 19/64 | 1.172 / 0.570 |
| 11 | 97/256 | 0/0/200 | 64/64 | 41/64 | 1.159 / 0.604 |
| 12 | 144/256 | 0/0/200 | 64/64 | 24/64 | 1.136 / 0.589 |
| 15 | 130/256 | 100/0/100 | 64/64 | 27/64 | 1.099 / 0.552 |

The run proves that the continuous-update plumbing works, including generations
that score below 55%. It also reproduces the symptom that originally looked like
a failure: deterministic play becomes repetition-dominated from generation 10
onward and does not recover by generation 15. Lower losses do not resolve that
ambiguity. Policy loss measures agreement with the current MCTS target, so it can
improve while MCTS learns either a strong strategy or a bad cycle; value loss also
becomes easier when more targets are the draw value zero. Top-1 still rose from 9%
during the first epoch to about 70%, while illegal policy mass fell from 91% to
4%, confirming that optimization itself was active. The frozen-baseline results
below are needed to interpret the symptom as strength progress rather than
stagnation.

The isolated artifacts are under ignored paths
`models/alpha-zero-from-zero/` and `data/alpha-zero-from-zero/`. Generation 15 is
the last complete checkpoint. Generation 16 self-play completed and was saved,
but training was interrupted before its checkpoint was created.

### Fixed-step verification (August 2026)

After preserving Adam across checkpoints and replacing growing full-buffer
epochs with 400 sampled mini-batch updates, a second clean run was stopped after
generation 12. Its artifacts are under `models/alpha-zero-fixed-step-v2/` and
`data/alpha-zero-fixed-step-v2/`.

| Generation | Self-play draws | Previous-generation arena C/R/D | Mirror draws | Exploratory draws | Validation policy/value |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4/256 | 200/0/0 | 0/64 | 2/64 | 2.676 / 1.185 |
| 4 | 43/256 | 200/0/0 | 64/64 | 6/64 | 1.547 / 0.772 |
| 8 | 69/256 | 100/0/100 | 64/64 | 29/64 | 1.295 / 0.621 |
| 10 | 84/256 | 100/0/100 | 64/64 | 31/64 | 1.193 / 0.497 |
| 11 | 115/256 | 0/0/200 | 64/64 | 37/64 | 1.219 / 0.533 |
| 12 | 141/256 | 100/0/100 | 0/64 | 24/64 | 1.131 / 0.461 |

The draw curve therefore survived the optimizer corrections and closely matched
the earlier run: generation 12 had 141 draws instead of 144. It is not, however,
evidence of stagnant playing strength. In paired 40-game arenas, generation 12
beat generations 1, 4 and 8 by 40-0. Against generation 11 it scored 20 wins,
20 draws and no losses. Every color split was favorable or undefeated.

The correct interpretation is that deterministic copies can converge on a
repetition while the newer network still dominates historical opponents. Draw
rate remains useful behavioral information, especially under exploratory
self-play, but an arbitrary low-draw threshold is not a strength oracle. Frozen
baselines—and eventually an exact solver—are required before changing the
official zero value of a repetition.

This does not make a draw-dominated replay buffer harmless. In the generation-12
batch, drawn games already supplied 4,385 of 8,500 examples (51.6%), and the
starter/non-starter/draw result was 57/58/141. The model was stronger than its
history but was not yet moving toward the known theoretical result: the side
that moves second has a forced win. Extending stochastic move selection from 12
to 48 plies reduced a paired probe from 18 to 8 draws, but changed the
starter/non-starter win split from 14/32 to 29/27. It replaced repetitions with
random mistakes rather than improving the signal, so that setting was not
adopted as a quick fix.

### Draw-signal experiments (August 2026)

The generation-12 result triggered three isolated follow-up runs. Each was
stopped as soon as the old failure was already unambiguous; continuing to 12
would only have spent more compute confirming it.

| Self-play source generation | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-step v2 | 4 | 9 | 13 | 43 | 39 |
| Untempered policy target v3 | 4 | 8 | 17 | 17 | 57 |
| Eight-state history v4 | 5 | 24 | 74 | — | — |
| Mixed decisive-tail curriculum v6 | 5 | 18 | 28 | 19 | 73 |
| Repetition contempt 0.5 v7 | 1 | 6 | — | — | — |

Separating the played-move temperature from the stored MCTS visit distribution
prevents late policies from becoming artificially one-hot, but did not remove
the draw attractor. Adding seven previous states reduces repetition ambiguity
and matches AlphaZero's finite chess/shogi history, but does not encode the full
threefold context and made short safe cycles easier to recognize. Replacing the
buffer with only the final won position was also rejected: its first candidate
lost the first 90 arena games against the random generation zero. Keeping the
full buffer and oversampling decisive tails avoided that catastrophic forgetting
and produced temporary 200-0 improvements, but draws returned at generation 5.

A final self-play-only repetition penalty nearly eliminated draws in v7, while
all recorded outcomes and arenas retained the official zero value. It exposed a
different failure immediately: generation 2 lost its official paired arena
against generation 1 by 0-200. Continuous latest-network publication had
already made this regressed candidate the next self-play source. Avoiding draws
therefore cannot substitute for a strength gate.

The practical conclusion was stricter than the earlier frozen-baseline result:
relative improvement and lower losses were not sufficient. It motivated the
current guarded loop, where self-play remains anchored to a champion and every
candidate must pass the paired 55% arena plus both draw gates. Rejected
candidates never become the next data generator.

### Guarded 15-generation validation (August 2026)

Two clean 15-candidate campaigns validate both the protection and the bootstrap
remedy. The first (`guarded-v8`) used neutral self-play and no terminal
oversampling. It promoted generations 3, 4 and 6, then correctly kept champion
6 while candidates 8–15 repeatedly converged on deterministic cycles. Several
would have passed the strength arena with 100 wins and 100 draws, but produced
64/64 mirror draws and were rejected.

The second (`guarded-curriculum-v9`) combined the same gates with self-play
repetition contempt 0.5 and temporary decisive-tail oversampling. It promoted
generations 1, 4, 5, 7 and 13. Extra tail sampling was already zero from
generation 4 and disabled structurally at generation 11; the later promotion
of generation 13 therefore did not depend on a permanently restricted dataset.

These two campaigns also exposed a flaw in their strength measurement: without
search noise or opening diversification, an arena seed changed only the random
first player. Its 200 games therefore repeated very few deterministic paths;
the frequent 100-game result blocks were not 200 independent observations.
The draw gates still prevented contaminated checkpoints from becoming
champions, but the historical arena scores below must not be overinterpreted.
The current arena instead gives each color-swapped pair the same random legal
0-4 ply opening and reports the number of distinct openings it actually used.

| Measurement after candidate 15 | Guarded v8 | Guarded curriculum v9 |
| --- | ---: | ---: |
| Accepted champion | 6 | 13 |
| Promotions | 3 | 5 |
| Drawn self-play games in full buffer | 393/3840 (10.2%) | 231/3840 (6.0%) |
| Draw positions in full buffer | 12,766/129,242 (9.9%) | 9,608/142,397 (6.7%) |
| Final candidate exploratory draws | 28/64 (43.8%) | 9/64 (14.1%) |

The v9 self-play batches ranged from 1 to 31 draws out of 256, rather than
entering the earlier majority-draw regime. A separate generation-13 probe made
the remaining limitation explicit: neutral noisy self-play drew 19/64 games,
whereas shaped self-play drew 4/64. The accepted network is therefore not proved
optimal and the search shaping still matters. Its official noise-free 400-search
mirror was nevertheless 0/64 draws, which is directly relevant to future
one-player play from the initial position. Move-order results remain mixed, so
the known forced win for the second mover has not yet been learned reliably.

### Diversified-arena 15-generation validation (August 2026)

The third clean campaign (`diverse-arena-v10`) reran the shaped bootstrap from
random weights with the corrected arena. Its 100 paired openings yielded 55 to
73 distinct legal histories per candidate, and intermediate scores no longer
appeared as artificial 100-game blocks. Candidates 1, 2, 3, 10, 11, 12 and 14
were promoted. Candidates 4–9 were rejected for an initial-position cycle,
candidate 13 for scoring only 52.7%, and candidate 15 for both mirror and
exploratory cycles. The published pointer consequently remained on champion 14.

Self-play draw counts across the 15 attempts were `1, 8, 7, 18, 14, 16, 12,
23, 11, 12, 24, 20, 35, 37, 36` out of 256. The full buffer contained 274/3,840
drawn games (7.1%) and 11,208/143,383 positions from draws (7.8%). There is a
late plateau near 14%, but not the former runaway to a majority-draw dataset.
Most importantly, candidate 15 could not become the next data source despite
winning its arena 129/35/36, because it drew 64/64 mirror games and 18/64 noisy
probe games.

The learning metrics also moved materially rather than remaining flat. From
candidate 1 to 15, validation policy loss fell from 2.785 to 1.550, top-1 rose
from 32.0% to 58.3%, illegal policy mass fell from 51.0% to 8.4%, and value loss
fell from 1.326 to 0.674. Value learning still plateaus around 0.67 and the
starter/non-starter split remains mixed, so this is a stable bootstrap rather
than evidence of perfect play.

A controlled champion-14 probe separates network behavior from search shaping:
neutral noisy self-play drew 25/64, shaped noisy self-play 11/64, and neutral
temperature-zero play 4/64 at 200 simulations. Its official 400-simulation
mirror was 0/64. The v10 conclusion was therefore precise: training-data
poisoning is contained and competitive play is mostly decisive, but exploration
still relies on repetition contempt and the network has not solved the game.

### Research findings and next experiments

The v10 data narrows the failure beyond "too many zero values." Draws account
for only 274/3,840 games and 11,208/143,383 positions, so they no longer dominate
the buffer. They nevertheless carry unusually confident imitation targets: the
mean visit-policy entropy is 0.862 on draw positions versus 0.977 on decisive
positions, and the mean largest action probability is 70.6% versus 66.1%.
Cross-entropy therefore teaches a comparatively sharp version of the cycle even
though the accompanying scalar value target is zero.

The finite history is a second source of label aliasing. Of the 274 v10 draws,
259 close a cycle of period 4, 11 of period 8, three of period 12 and one of
period 16. The eight-frame encoder cannot distinguish the complete repetition
context for the 15 games whose period exceeds seven preceding positions. The
old generation-12 run has the same problem in 7/144 draws, with periods as long
as 26. This is a minority rather than the dominant failure, but it proves that
seven preceding positions reduce ambiguity without making a threefold game
fully Markovian. Search still owns the exact `Game` history and detects terminal
repetitions correctly; the network input and its training targets do not.

Finally, the tactical horizon is not small merely because the board is small.
The independently reproduced tablebase contains 246,803,167 reachable
positions, the initial forced win takes 78 plies, and some winning positions
take 173 plies. A 200-simulation search guided by a still inaccurate value head
can discover a short safe cycle much more easily than a long refutation. See
[Solving Dōbutsu Shōgi](https://brianhliou.com/posts/dobutsu-shogi/) and the
[MIT-licensed Rust tablebase](https://github.com/brianhliou/dobutsu-shogi).

These observations rule out simply deleting draws or extending random move
sampling. Deletion removes exactly the defensive lines the winning side must
learn to refute, while the 48-ply experiment already showed that more random
moves replace cycles with mistakes. The proposed recovery instead makes every
draw informative and spends additional search near the states where conversion
failed.

#### Recovery implementation: draw-aware targeted self-play

The representation, WDL loss, role-aware utility, targeted restarts and separate
learner lineage described below are now implemented. They change the checkpoint
and encoder formats, so the next controlled campaign must start at generation 0.
Tablebase evaluation remains the next implementation experiment; no result from
the new pipeline is claimed yet. The checked-in configuration isolates it
under `models/alpha-zero-draw-aware-v11/` and
`data/alpha-zero-draw-aware-v11/` so legacy checkpoints and examples cannot be
loaded accidentally.

1. **Expose the missing game context to the evaluator (implemented).** A constant plane
   indicating whether the current player was the starter. For the policy head,
   also provide an action-aligned feature containing the occurrence count of
   the position reached by each legal action and a flag for an immediate
   threefold draw. Include these features in `EvaluationRequest` and its cache
   key. Search already computes the exact result from the full `Game`; this
   change lets the learned prior distinguish two otherwise identical boards
   where one action repeats and the other does not. It avoids an unbounded stack
   of board-history planes without claiming that the value input contains the
   complete repetition multiset.
2. **Replace the scalar value with a WDL head (implemented).** Train three logits against the
   one-hot Win/Draw/Loss result from the current player's perspective. Official
   arenas and play continue to use the neutral value
   `Q_official = P(win) - P(loss)`. Unlike scalar MSE, WDL cross-entropy gives a
   certain draw a different target from an uncertain 50/50 win/loss state. Lc0
   made the same representation change and uses the explicit draw probability
   to adjust search utility; see [its WDL and contempt
   notes](https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/).
3. **Use role-aware draw utility only during self-play (implemented).** Dōbutsu Shōgi is a
   forced win for the non-starter. A draw is therefore a successful defence for
   the starter and a failed conversion for the non-starter. With WDL probabilities
   from the current player's perspective, use
   `Q_self_play = W - L + cD` when that player is the starter and
   `Q_self_play = W - L - cD` otherwise, initially with `c = 0.25`. Because
   `0 < c < 1`, both roles still order outcomes as win > draw > loss, but the
   draw now supplies an adversarial margin. This is preferable to penalizing the
   player that happens to close the loop: the starter should learn the strongest
   drawing defence while the non-starter is forced to find its refutation.
   Stored outcomes remain official W/D/L and all promotion searches use `c = 0`.
4. **Restart a minority of games immediately before known cycles (implemented).** Preserve a
   complete `Game` snapshot, including its full prefix and initial player, for
   positions two to eight plies before a repetition. Start 25% of the next
   self-play batch from this archive and 75% from the official initial position.
   The implementation keeps the configured trajectory count fixed and reports
   actual inference positions so ablations can also be normalized by compute.
   This revisits the exact failure states, produces shorter value targets, and
   gives the non-starter repeated chances to find a conversion. Go-Exploit
   reports better value accuracy and sample efficiency from this general restart
   strategy in Connect Four and 9x9 Go; see [Targeted Search Control in
   AlphaZero](https://arxiv.org/abs/2302.12359). YokaiRust should target the
   archive at cycles rather than sample every historical state uniformly.
5. **Separate learner progress from champion publication (implemented).** The previous pipeline
   reloaded the accepted champion's model and optimizer after every rejection.
   Consequently, rejected attempts did not form a learning lineage: champion 6
   produced candidates 8–15, but each attempt restarted from champion 6. The
   accepted champion remains the safe self-play and publication source, while a
   separate `learner` pointer retains optimization progress. A rejected learner
   now continues when it passed the strength arena but failed a draw gate; it is
   rolled back to the champion when it regresses in the official arena. This lets
   a strong-but-cycling candidate receive subsequent counterexamples without
   publishing it.

The repetition features, role-aware WDL utility and targeted restarts are
complementary. The features make the immediate choice observable, WDL says
which role failed when a trajectory draws, and the restart archive determines
where to spend the next search budget. The separate learner pointer then lets
that new evidence accumulate across rejected publication attempts. None of
these changes requires a drawn trajectory to be relabelled as an official win
or loss.

#### Oracle and evaluation protocol

The solved game makes a stronger evaluation possible than a historical ladder
alone. First validate a tablebase adapter against YokaiRust's exact Try and
repetition semantics, then keep it outside training and report:

- WDL accuracy and calibration on fixed fresh positions;
- probability mass assigned to tablebase-optimal actions;
- the same measurements bucketed by distance to result;
- conversion rate as non-starter from the two official orientations;
- cycle period, draw-policy entropy and draw-origin buffer mass.

The tablebase can later provide a separate solver-assisted experiment by
reanalysing fresh, cycle-adjacent positions, but that must be labelled as
supervised oracle assistance rather than learning from zero. Its first purpose
is to reveal absolute progress and distinguish "fewer draws" from "closer to
perfect play." OpenSpiel similarly keeps fixed MCTS-plus-solver evaluators
outside its AlphaZero learner; see [OpenSpiel
AlphaZero](https://openspiel.readthedocs.io/en/stable/alpha_zero.html).

Run the recovery as controlled ablations from generation 0: scalar baseline,
context-aware scalar, context-aware neutral WDL (`c = 0`), role-aware WDL
(`c = 0.25`), then role-aware WDL plus 25% targeted restarts. Add the separate
learner lineage as a second experimental factor, then extend the best run beyond
15 candidates. Keep the same seeds, network trunk and simulator budget; a
claimed improvement should reproduce across at least three
seeds. Success is a sustained rise in tablebase agreement and non-starter
conversion, not merely lower losses or a lower raw draw rate.

Gumbel AlphaZero remains the next search ablation if root diagnostics show poor
action coverage; it has strong published evidence under small simulation
budgets and on Animal Shogi specifically. In a 2026 comparison trained with 800
million simulator evaluations, standard AlphaZero scored `31 ± 2%` and Gumbel
AlphaZero `67 ± 5%` against the same anchored opponent under the paper's
800-rollout evaluation; see [Revisiting Regularized Policy Optimization for
Two-Player Games](https://arxiv.org/html/2602.10894v2). Gumbel does not, by
itself, correct the aliased repetition context or the zero-valued draw target.
Policy-target pruning is likewise conditional on showing that Dirichlet-only
visits carry material target mass. Historical opponents are useful only if the
fixed ladder confirms forgetting. These remain secondary to repairing the
measured draw signal.

### Historical guarded-pipeline observations

The following measurements predate the switch to continuous AlphaZero updates.
They are retained because they motivated the draw diagnostics, not because they
describe the current publication policy.

The first local training sequence produced the following self-play trend. An
example corresponds to one played position, so examples per game also measure
average game length.

| Generation | Games | Examples | Average plies | Draws |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 256 | 7,499 | 29.3 | 10 (3.9%) |
| 2 | 256 | 9,460 | 37.0 | 20 (7.8%) |
| 3 | 256 | 13,916 | 54.4 | 44 (17.2%) |
| 4 | 256 | 15,826 | 61.8 | 80 (31.3%) |
| 5 | 256 | 16,113 | 62.9 | 93 (36.3%) |

The steadily increasing draw rate reproduced the same failure mode previously
observed in the C++ and Node.js prototypes. Diagnostics established that it was
an undesirable learned cycle: champion 5 drew all 32 noise-free mirror games,
and increasing its search from 200 to 400 simulations made draws substantially
more frequent. Because an official repetition is worth zero, a deeper search
rationally preferred it whenever the inaccurate value head rated alternatives
as losses.

The arena counters were checked in both argument orders, ruling out an inverted
candidate/champion result. The real weakness was the promotion protocol: a model
could decisively exploit its predecessor while developing a different mirror
cycle. Generation 13 demonstrated this by passing its old arena but drawing
42/64 neutral mirror games; its checkpoint was retained but the champion pointer
was returned to generation 11.

The terminal curriculum proposed during debugging, combined with repetition
contempt during full-depth self-play, reversed the trend:

| Candidate data | Terminal window | Simulations | Contempt | Draws |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 16 | 400 | 0.50 | 38/256 (14.8%) |
| 12 | 16 | 400 | 0.50 | 21/256 (8.2%) |
| 13 | 32 | 400 | 0.50 | 18/256 (7.0%) |
| 14 | 32 | 400 | 0.50 | 16/256 (6.25%) |

Candidate 14 then produced 0/64 mirror draws but was correctly rejected because
its paired score against champion 11 was only 50%. These measurements show that
the runaway draw trend is controlled without weakening the official rules or
promoting a merely non-cycling but otherwise equal candidate.

Candidate 15 exposed a second failure mode after initially passing 200-0 and
0/64 mirror draws: its following noisy self-play batch produced 76/256 draws
(29.7%). Its checkpoint was retained, champion was returned to generation 11,
and the exploratory candidate gate was added at 20% so the same regression is
now rejected before publication.

Candidate 16 confirmed the deterministic gate itself: it also beat champion 11
by 200-0, then drew all 64 mirror games and was rejected. The exploratory probe
was skipped because the candidate was already ineligible. Champion 11 therefore
remains published while its full-depth self-play batches stay in the measured
5.9-8.2% draw range.

Generation 4 also exposed validation overfitting: validation loss was best at
epoch 1 and degraded through epoch 10 while training loss kept improving. The
pipeline now restores the lowest-validation-loss epoch and stops after the
configured patience. Generation 5 validated this behavior by stopping after
epoch 3 and sending the restored epoch 1 model to the arena.

## Rules source

The engine follows the official 3×4 rulebook rather than preserving behavioral
differences from the older JavaScript and C++ implementations:

<https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf>
