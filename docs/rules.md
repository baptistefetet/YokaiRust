# Rules of the 5×6 game

YokaiRust implements the expert version of **Yōkaï no Mori**, played on a board
of 5 columns and 6 rows. It is the French yōkai-themed edition of *Goro Goro
shōgi*, a small shogi variant designed by the Japan Shogi Association. The
reference is the expert section of the
[official rulebook](https://cdn.1j1ju.com/medias/b8/2f/eb-yokai-no-mori-rulebook.pdf).

This page states the rules exactly as the engine applies them. Where the
rulebook is silent or ambiguous, the engine has to choose; those choices are
listed in [Interpretation choices](#interpretation-choices). The executable
specification is [`tests/engine.rs`](../tests/engine.rs): each rule below has a
test whose name states it.

## Board and starting position

Columns are named `a` to `e` from left to right and ranks `1` to `6` from
bottom to top. The engine keeps one absolute orientation: First starts at the
bottom and moves toward rank 6; Second starts at the top and moves toward
rank 1.

```text
  6   on  ki  KO  ki  on    Second
  5    .   .   .   .   .
  4    .  kd  kd  kd   .
  3    .  KD  KD  KD   .
  2    .   .   .   .   .
  1   ON  KI  KO  KI  ON    First
       a   b   c   d   e
```

Each player owns one Koropokkuru (`KO`), two Kirin (`KI`), two Oni (`ON`) and
three Kodama (`KD`): 16 pieces in total. The rulebook draws the starting
player at random, so the engine accepts either player as the starter. A player
must act on every turn; passing is not allowed.

## How the pieces move

Every piece moves exactly one square. No piece slides or jumps, so a move can
never be blocked by a piece standing in between. In the diagrams, `x` marks the
squares that the piece `P` can reach when its owner moves up the page.

```text
Koropokkuru      Kirin            Oni              Kodama
  x x x            x x x            x x x            . x .
  x P x            x P x            . P .            . P .
  x x x            . x .            x . x            . . .
```

| Piece | Movement | Promotes to |
| --- | --- | --- |
| Koropokkuru | One square in any of the eight directions. | — |
| Kirin | One square in any direction except diagonally backward. | — |
| Oni | One square straight forward or along any of the four diagonals. | Super Oni |
| Kodama | One square straight forward. | Kodama Samurai |
| Super Oni | Like a Kirin. | — |
| Kodama Samurai | Like a Kirin. | — |

A piece may move to an empty square or to a square holding an opposing piece,
which it captures. It may never move onto a piece of its own side.

## Promotion

A player's promotion zone is the two ranks farthest from their side: ranks 5
and 6 for First, ranks 1 and 2 for Second. An Oni or a Kodama may promote when
its move enters the zone, leaves it, or stays inside it. A capture does not
prevent the promotion.

Promotion is optional, with one exception: a Kodama that reaches the last rank
must promote, because an unpromoted Kodama there could never move again.

A promoted piece stays promoted for as long as it remains on the board.

## Captures and drops

A captured piece changes sides and goes to the capturing player's reserve (the
*hand*). A captured Super Oni or Kodama Samurai returns to its unpromoted side.

Instead of moving, a player may drop one piece from their hand onto any empty
square. A dropped piece is always unpromoted, even inside the promotion zone;
it can promote on a later move. Three restrictions apply to Kodama drops:

1. A Kodama cannot be dropped on the last rank, where it could not move.
2. A Kodama cannot be dropped in a column that already holds an unpromoted
   Kodama of the same player. A Kodama Samurai in the column does not count.
3. A Kodama cannot be dropped so that it checkmates the opposing Koropokkuru.
   Checkmate by any other drop, or by a Kodama move, is allowed.

## End of the game

- **Capture.** A player who captures the opposing Koropokkuru wins. Reaching
  the far side of the board with a Koropokkuru does not win in this version.
- **No legal action.** A player who has no legal action on their turn loses.
- **Repetition.** The game is drawn when the same position occurs for the
  third time. Two positions are the same when the board, both hands and the
  player to move are identical.
- **Move limit.** The game is drawn when 256 plies (128 turns of each player)
  have been played without a winner. A capture or a third repetition on the
  last ply takes precedence.

## Interpretation choices

The rulebook describes the expert game as a short list of additions to the
classic 3×4 rules. The engine needs a precise answer in a few places where the
text leaves room.

| Topic | Rulebook | Engine |
| --- | --- | --- |
| Koropokkuru safety | The checkmate example says the Koropokkuru “cannot capture … or escape by moving since it would be in position to be captured”. | A Koropokkuru may not move onto a square attacked by an opposing piece. Every other action is legal even if it leaves the Koropokkuru attacked. |
| End by checkmate | The example ends with “So Player 1 wins” as soon as the checkmate stands. | Checkmate does not end the game by itself. A checkmated player who still has a legal action must play one, and the opponent wins by then capturing the Koropokkuru. The engine does not force that capture: a player who declines it has not won yet. A checkmated player with no legal action loses at once. |
| Kodama drop checkmate | Forbidden when the Koropokkuru “cannot move without getting captured and cannot capture the parachuted kodama without being captured himself”. | Uses the shogi definition, which is wider: the drop is forbidden only when *no* opposing board move answers it. Besides a Koropokkuru move, another piece capturing the Kodama, or a capture of the dropping player's own Koropokkuru, also makes the drop legal. |
| Repetition | “If the players move their yōkaï back and forth three times in succession, the game ends in a draw.” | Draw at the third occurrence of the same position, wherever the earlier occurrences are in the game. Japanese Goro Goro shōgi uses four occurrences and penalizes perpetual check; the engine follows the rulebook's count and has no perpetual-check rule. |
| Move limit | None. | Draw after 256 plies. Two weak players can wander on this board without ever repeating a position, and self-play training and arenas need every game to end. The limit is several times the length of a typical game, so it only ends games that are going nowhere. |
| No legal action | Passing is forbidden; the case is not described. | The player to move loses. |
| Same-column Kodama | “Two kodama belonging to the same player can never be positioned in the same column.” | Enforced on drops. A move cannot break the rule, because an unpromoted Kodama never changes column. |

Because pieces move a single square, the Koropokkuru safety test is exact and
cheap: a square is attacked when an opposing piece stands one step away in a
direction it can move.

## Notation

Moves name the origin and destination squares; drops name the piece and the
target square. A trailing `+` marks a promotion.

| Action | Example |
| --- | --- |
| Move | `b3-b4` |
| Promoting move | `c4-c5+` |
| Drop | `kodama@a4`, `oni@c2`, `kirin@e5` |

When a move has both a plain and a promoting form, both are distinct legal
actions; the terminal interface asks which one to play.

## How the rules are verified

- [`tests/engine.rs`](../tests/engine.rs) pins each rule with positions built
  for that purpose. Rules that depend on the direction of play are checked
  for both players.
- The same file fixes the number of action sequences from the starting
  position (*perft*): 16, 250, 4,202 and 69,581 sequences of one to four plies.
  A change to move generation that alters any of these counts fails the test.
- [`tests/properties.rs`](../tests/properties.rs) plays seeded random games and
  checks after every ply that each generated action can be applied, that
  actions and policy indices convert back and forth without loss, and that
  repetition counts match the recorded history.
