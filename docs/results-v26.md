# v26 training results

This page preserves the original generation 16 measurements. See
[generations 17–19](results-continuation-17-19.md) for the September 2026
continuation and the current accepted champion.

This is the frozen research log of the **v26** training line, the current
regression and publication baseline. The concepts used below (arena, gating,
WDL, restarts, …) are all defined in the
[AlphaZero guide](alphazero-guide.md).

Active paths of the run:

- models: `models/alpha-zero-visited-restarts-v26`;
- self-play, reports and diagnostics: `data/alpha-zero-visited-restarts-v26`;
- accepted champion: generation 16.

## The v26 recipe

The checked-in Metal configuration uses a 64-channel, four-block residual
network with 64-unit shared and value layers. Each generation runs:

1. 256 self-play games at 200 MCTS simulations per regular move;
2. 25% restarts from recent visited states at 800 simulations per move;
3. 400 Adam updates with batch size 256;
4. a 200-game paired strength arena at 400 simulations per move;
5. four deterministic mirror games for diagnosis;
6. a 64-game noisy self-play productivity probe.

The champion is the single source of self-play, weights and optimizer state.
The next attempt after a rejection starts from that same accepted checkpoint,
but sees a larger replay buffer and a new deterministic seed.

### Promotion criteria

A candidate is promoted only when it:

1. scores at least 55% against the champion in paired games with shared random
   0–4 ply openings and swapped colors;
2. produces at most 20% draws in the noisy 64-game self-play probe.

Deterministic candidate-versus-itself draws remain recorded but do not veto a
candidate; the [guide](alphazero-guide.md) explains why, with references to
AlphaGo Zero, AlphaZero and KataGo gating practice.

### Draw-aware settings

Official draws always remain draws and the stored WDL target is never
rewritten. v26 handles repetition feedback with these settings (rationale in
the guide):

- self-play values a draw at `+0.75` for the starter and `-0.75` for the
  non-starter, while official arenas use neutral `P(win) - P(loss)`;
- one quarter of trajectories restart from uniformly sampled nonterminal
  states visited in the recent replay buffer, with a larger search budget;
- the starter's drawing defence remains a policy target, while unresolved
  non-starter policy targets are omitted;
- the auxiliary scalar loss has weight `0.25` and compares `P(win) - P(loss)`
  with `+1/0/-1`; it complements, never replaces, categorical WDL learning.

## Final measurements

The v26 training run took 1 h 20 min 03 s on an Apple M4 Max. Its 25%
deeper-search restarts sample all recently visited nonterminal states. The
accepted champion is generation 16, with these final measurements:

| Measurement | Result |
| --- | ---: |
| self-play | 92 First wins / 118 Second wins / 46 draws |
| validation policy loss / top-1 | 1.508 / 57.2% |
| validation WDL loss / top-1 | 0.770 / 64.9% |
| validation scalar MSE | 0.734 |
| paired strength arena | 79 wins / 31 losses / 90 draws, score 62.0% |
| deterministic mirror | 0/4 draws |
| noisy productivity probe | 6/64 draws |

An independent 400-game paired arena at 400 simulations per move confirmed the
promotion result against its reference:

| Candidate / reference / draws | Score | Candidate as First | Candidate as Second |
| ---: | ---: | ---: | ---: |
| 170 / 66 / 164 | **63.0%** | 62.5% | 63.5% |

The approximate 95% interval is 59.5–66.5%. Both seats are positive, and the
interval excludes 50%.

## Restart behavior and speed

Across v26, 960 games restarted at depths 1–232, with a mean depth of 33.3
plies. Restarted games drew 60/960 times (6.2%), versus 197/3,136 (6.3%) for
games starting from the initial position. The broader archive did not make its
own trajectories less productive.

Neural self-play evaluated 40,217,309 positions in about 1,188.4 backend
seconds: **33,842 positions/s** on the Apple M4 Max. The accepted champion's
run reached 29,961 positions/s.

## Endgame diagnosis

The current frozen v26 split contains 438 games, including 24 draws, and
18,271 positions. At distance `17+`, drawn policy top-1 is 61.8%; at `9–16`,
it is 55.6%. Drawn WDL top-1 is 7.0% and 1.3% in those two buckets.

The strength gain therefore does not solve long-horizon draw classification.
It comes from better play as measured directly, while the WDL draw head
remains the clearest learning weakness.

## Research conclusion

Visited-state restarts are now the stable baseline. They follow Go-Exploit's
simple “Visited States” variant: decisive and drawn games both feed the
archive, and duplicate states naturally weight frequently visited regions.
Network shape, optimizer, replay retention, promotion and all other budgets
remained unchanged. [Go-Exploit](https://arxiv.org/abs/2302.12359)

The next learning change should not be another architecture rewrite. The
endgame diagnosis isolates the remaining problem: long-horizon draw WDL. If
optimization resumes, target that head or strengthen decisive endgame sampling
with one isolated change.
