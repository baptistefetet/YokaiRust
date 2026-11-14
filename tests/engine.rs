//! Executable specification of board setup, movement and official outcomes.
//!
//! Squares are written `square(row, column)` with row 0 at the top (rank 6)
//! and column 0 on the left (file `a`). First plays from the bottom toward
//! row 0; Second plays from the top toward row 5.

mod common;

use common::{position_with, square};

use std::{collections::BTreeSet, str::FromStr};

use yokai::{
    Action, BOARD_SQUARES, DrawReason, Game, HandPiece, MOVE_LIMIT_PLIES, Outcome, Piece,
    PieceKind, Player, Position, PositionError, Replay, ReplayError, Square, WinReason,
};

const NO_HANDS: [[u8; 3]; 2] = [[0; 3]; 2];

fn step(from: Square, to: Square) -> Action {
    Action::Move {
        from,
        to,
        promote: false,
    }
}

fn promotion(from: Square, to: Square) -> Action {
    Action::Move {
        from,
        to,
        promote: true,
    }
}

fn kodama_drop(to: Square) -> Action {
    Action::Drop {
        piece: HandPiece::Kodama,
        to,
    }
}

/// Destinations reachable from `from`, ignoring the promotion flag.
fn destinations(position: &Position, from: Square) -> BTreeSet<Square> {
    position
        .legal_actions()
        .into_iter()
        .filter_map(|action| match action {
            Action::Move {
                from: origin, to, ..
            } if origin == from => Some(to),
            Action::Move { .. } | Action::Drop { .. } => None,
        })
        .collect()
}

#[test]
fn initial_position_matches_the_official_setup() {
    let position = Position::initial(Player::First);
    assert_eq!(position.physical_piece_count(), 16);

    let back_rank = [
        PieceKind::Oni,
        PieceKind::Kirin,
        PieceKind::Koropokkuru,
        PieceKind::Kirin,
        PieceKind::Oni,
    ];
    for (column, kind) in (0..).zip(back_rank) {
        assert_eq!(
            position.piece_at(square(5, column)),
            Some(Piece::new(kind, Player::First))
        );
        assert_eq!(
            position.piece_at(square(0, column)),
            Some(Piece::new(kind, Player::Second))
        );
    }
    for column in 0..5 {
        let kodama = (1..=3).contains(&column);
        assert_eq!(
            position.piece_at(square(3, column)),
            kodama.then_some(Piece::new(PieceKind::Kodama, Player::First))
        );
        assert_eq!(
            position.piece_at(square(2, column)),
            kodama.then_some(Piece::new(PieceKind::Kodama, Player::Second))
        );
        assert_eq!(position.piece_at(square(1, column)), None);
        assert_eq!(position.piece_at(square(4, column)), None);
    }

    // Oni 2 + Kirin 3 + Koropokkuru 3 + Kirin 3 + Oni 2, and three Kodama
    // captures: the two Kodama lines start face to face.
    assert_eq!(position.legal_actions().len(), 16);
}

#[test]
fn initial_game_has_the_reference_number_of_continuations() {
    fn count_continuations(game: &Game, depth: u8) -> u64 {
        if depth == 0 {
            return 1;
        }
        game.legal_actions()
            .into_iter()
            .map(|action| {
                let mut next = game.clone();
                next.apply(action).unwrap();
                count_continuations(&next, depth - 1)
            })
            .sum()
    }

    // Cross-checked against an independent implementation of the rules. Early
    // wins contribute no continuations, while games reaching exactly four
    // plies contribute one. The setup is symmetric, so both starters agree.
    for player in [Player::First, Player::Second] {
        let game = Game::new(player);
        assert_eq!(count_continuations(&game, 1), 16);
        assert_eq!(count_continuations(&game, 2), 250);
        assert_eq!(count_continuations(&game, 3), 4_202);
        assert_eq!(count_continuations(&game, 4), 69_581);
    }
}

#[test]
fn material_limits_combine_the_board_and_both_players_hands() {
    for (kind, copies) in [
        (PieceKind::Kirin, 4),
        (PieceKind::Oni, 4),
        (PieceKind::SuperOni, 4),
        (PieceKind::KodamaSamurai, 6),
    ] {
        let hand_piece = HandPiece::from_captured(kind).unwrap();
        assert_eq!(usize::from(hand_piece.copies()), copies);
        let index = hand_piece.index();

        // Every copy on the board, alternating owners.
        let mut board = [None; BOARD_SQUARES];
        for (slot, owner) in [Player::First, Player::Second]
            .into_iter()
            .cycle()
            .take(copies)
            .enumerate()
        {
            board[slot] = Some(Piece::new(kind, owner));
        }
        assert!(Position::from_parts(board, NO_HANDS, Player::First).is_ok());

        let mut hands = NO_HANDS;
        hands[1][index] = 1;
        assert_eq!(
            Position::from_parts(board, hands, Player::First),
            Err(PositionError::TooManyPiecesOfKind(hand_piece))
        );

        // One copy moves from the board to a hand: the total is unchanged.
        board[0] = None;
        assert!(Position::from_parts(board, hands, Player::First).is_ok());

        // Every copy in one hand, then one too many across both hands.
        let board = [None; BOARD_SQUARES];
        let mut hands = NO_HANDS;
        hands[0][index] = hand_piece.copies();
        assert!(Position::from_parts(board, hands, Player::First).is_ok());
        hands[1][index] = 1;
        assert_eq!(
            Position::from_parts(board, hands, Player::First),
            Err(PositionError::TooManyPiecesOfKind(hand_piece))
        );
    }
}

#[test]
fn promoted_and_unpromoted_pieces_share_one_material_limit() {
    let position = position_with(
        &[
            (3, 0, PieceKind::Kodama, Player::First),
            (3, 1, PieceKind::Kodama, Player::First),
            (3, 2, PieceKind::Kodama, Player::First),
            (2, 0, PieceKind::KodamaSamurai, Player::Second),
            (2, 1, PieceKind::KodamaSamurai, Player::Second),
        ],
        [[0, 0, 1], [0; 3]],
        Player::First,
    );
    assert_eq!(
        Position::from_parts(*position.board(), [[0, 0, 1], [0, 0, 1]], Player::First),
        Err(PositionError::TooManyPiecesOfKind(HandPiece::Kodama))
    );

    // Deserialization applies the same checks as `from_parts`.
    let mut json = serde_json::to_value(position).unwrap();
    assert_eq!(
        serde_json::from_value::<Position>(json.clone()).unwrap(),
        position
    );
    json["hands"][1][2] = 1.into();
    assert!(serde_json::from_value::<Position>(json).is_err());
}

#[test]
fn positions_reject_kodama_placements_that_legal_play_cannot_produce() {
    let board_with = |pieces: &[(u8, u8, PieceKind, Player)]| {
        let mut board = [None; BOARD_SQUARES];
        for &(row, column, kind, owner) in pieces {
            board[square(row, column).index()] = Some(Piece::new(kind, owner));
        }
        Position::from_parts(board, NO_HANDS, Player::First)
    };

    assert_eq!(
        board_with(&[
            (3, 1, PieceKind::Kodama, Player::First),
            (1, 1, PieceKind::Kodama, Player::First),
        ]),
        Err(PositionError::KodamaShareColumn)
    );
    // A promoted Kodama or an opposing Kodama may share the column.
    assert!(
        board_with(&[
            (3, 1, PieceKind::Kodama, Player::First),
            (1, 1, PieceKind::KodamaSamurai, Player::First),
            (2, 1, PieceKind::Kodama, Player::Second),
        ])
        .is_ok()
    );

    assert_eq!(
        board_with(&[(0, 2, PieceKind::Kodama, Player::First)]),
        Err(PositionError::KodamaOnLastRow)
    );
    assert_eq!(
        board_with(&[(5, 2, PieceKind::Kodama, Player::Second)]),
        Err(PositionError::KodamaOnLastRow)
    );
    // Row 0 is Second's own back rank, not its last row.
    assert!(board_with(&[(0, 2, PieceKind::Kodama, Player::Second)]).is_ok());
}

#[test]
fn every_piece_follows_its_printed_dots_for_both_players() {
    // Offsets are (rows toward the opponent, columns to the right).
    const KIRIN: &[(i8, i8)] = &[(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, 0)];
    const ONI: &[(i8, i8)] = &[(1, -1), (1, 0), (1, 1), (-1, -1), (-1, 1)];
    const KODAMA: &[(i8, i8)] = &[(1, 0)];
    let expectations = [
        (PieceKind::Kirin, KIRIN),
        (PieceKind::Oni, ONI),
        (PieceKind::SuperOni, KIRIN),
        (PieceKind::Kodama, KODAMA),
        (PieceKind::KodamaSamurai, KIRIN),
    ];

    for (kind, offsets) in expectations {
        for (player, forward) in [(Player::First, -1), (Player::Second, 1)] {
            // The Koropokkuru stay in two corners, away from the tested piece.
            let position = position_with(
                &[
                    (5, 0, PieceKind::Koropokkuru, Player::First),
                    (0, 4, PieceKind::Koropokkuru, Player::Second),
                    (3, 2, kind, player),
                ],
                NO_HANDS,
                player,
            );
            let expected = offsets
                .iter()
                .map(|&(toward_opponent, right)| {
                    let row = 3 + toward_opponent * forward;
                    square(u8::try_from(row).unwrap(), u8::try_from(2 + right).unwrap())
                })
                .collect::<BTreeSet<_>>();
            assert_eq!(
                destinations(&position, square(3, 2)),
                expected,
                "{kind:?} of {player:?}"
            );
        }
    }
}

#[test]
fn a_captured_promoted_piece_returns_to_hand_unpromoted() {
    for (promoted, hand_piece) in [
        (PieceKind::KodamaSamurai, HandPiece::Kodama),
        (PieceKind::SuperOni, HandPiece::Oni),
    ] {
        let position = position_with(
            &[
                (5, 2, PieceKind::Koropokkuru, Player::First),
                (0, 2, PieceKind::Koropokkuru, Player::Second),
                (3, 0, PieceKind::Kirin, Player::First),
                (2, 0, promoted, Player::Second),
            ],
            NO_HANDS,
            Player::First,
        );
        let mut game = Game::from_position(position);
        let transition = game
            .apply(step(square(3, 0), square(2, 0)))
            .expect("capture must be legal");

        assert_eq!(transition.captured, Some(promoted));
        assert_eq!(game.position().hand_count(Player::First, hand_piece), 1);
    }
}

#[test]
fn promotion_is_offered_when_a_move_touches_the_promotion_zone() {
    let oni_at = |row, column, player| {
        position_with(
            &[
                (5, 4, PieceKind::Koropokkuru, Player::First),
                (0, 4, PieceKind::Koropokkuru, Player::Second),
                (row, column, PieceKind::Oni, player),
            ],
            NO_HANDS,
            player,
        )
    };
    let both = |position: &Position, from, to| {
        (
            position.is_legal_action(step(from, to)),
            position.is_legal_action(promotion(from, to)),
        )
    };

    // First promotes on rows 0 and 1: entering, moving inside and leaving.
    let entering = oni_at(2, 1, Player::First);
    assert_eq!(both(&entering, square(2, 1), square(1, 1)), (true, true));
    let inside = oni_at(1, 1, Player::First);
    assert_eq!(both(&inside, square(1, 1), square(0, 1)), (true, true));
    assert_eq!(both(&inside, square(1, 1), square(2, 0)), (true, true));
    // Outside the zone the promoting variant does not exist.
    let outside = oni_at(3, 1, Player::First);
    assert_eq!(both(&outside, square(3, 1), square(2, 1)), (true, false));

    // Second promotes on rows 4 and 5.
    let entering = oni_at(3, 1, Player::Second);
    assert_eq!(both(&entering, square(3, 1), square(4, 1)), (true, true));
    let outside = oni_at(1, 1, Player::Second);
    assert_eq!(both(&outside, square(1, 1), square(2, 1)), (true, false));

    // Both variants are generated as distinct actions.
    let actions = oni_at(2, 1, Player::First).legal_actions();
    assert!(actions.contains(&step(square(2, 1), square(1, 1))));
    assert!(actions.contains(&promotion(square(2, 1), square(1, 1))));
}

#[test]
fn only_oni_and_kodama_have_a_promoted_face() {
    for kind in [
        PieceKind::Koropokkuru,
        PieceKind::Kirin,
        PieceKind::SuperOni,
        PieceKind::KodamaSamurai,
    ] {
        let position = position_with(
            &[
                (5, 4, PieceKind::Koropokkuru, Player::Second),
                (1, 1, kind, Player::First),
            ],
            NO_HANDS,
            Player::First,
        );
        assert!(position.is_legal_action(step(square(1, 1), square(0, 1))));
        assert!(!position.is_legal_action(promotion(square(1, 1), square(0, 1))));
    }
}

#[test]
fn promoting_flips_the_piece_and_may_capture() {
    let position = position_with(
        &[
            (5, 4, PieceKind::Koropokkuru, Player::First),
            (0, 4, PieceKind::Koropokkuru, Player::Second),
            (2, 1, PieceKind::Oni, Player::First),
            (1, 0, PieceKind::Kirin, Player::Second),
        ],
        NO_HANDS,
        Player::First,
    );

    let mut promoted = Game::from_position(position);
    let transition = promoted
        .apply(promotion(square(2, 1), square(1, 0)))
        .expect("capturing while entering the zone may promote");
    assert!(transition.promoted);
    assert_eq!(transition.captured, Some(PieceKind::Kirin));
    assert_eq!(
        promoted.position().piece_at(square(1, 0)),
        Some(Piece::new(PieceKind::SuperOni, Player::First))
    );

    let mut unpromoted = Game::from_position(position);
    let transition = unpromoted
        .apply(step(square(2, 1), square(1, 0)))
        .expect("promotion is optional");
    assert!(!transition.promoted);
    assert_eq!(
        unpromoted.position().piece_at(square(1, 0)),
        Some(Piece::new(PieceKind::Oni, Player::First))
    );
}

#[test]
fn a_kodama_must_promote_on_the_last_row_only() {
    let kodama_at = |row| {
        position_with(
            &[
                (5, 4, PieceKind::Koropokkuru, Player::First),
                (0, 4, PieceKind::Koropokkuru, Player::Second),
                (row, 1, PieceKind::Kodama, Player::First),
            ],
            NO_HANDS,
            Player::First,
        )
    };

    // Entering the zone on its first row leaves the choice open.
    let entering = kodama_at(2);
    assert!(entering.is_legal_action(step(square(2, 1), square(1, 1))));
    assert!(entering.is_legal_action(promotion(square(2, 1), square(1, 1))));

    // An unpromoted Kodama on the last row could never move again.
    let arriving = kodama_at(1);
    assert!(!arriving.is_legal_action(step(square(1, 1), square(0, 1))));
    let mut game = Game::from_position(arriving);
    let transition = game
        .apply(promotion(square(1, 1), square(0, 1)))
        .expect("the promoting move must be legal");
    assert!(transition.promoted);
    assert_eq!(
        game.position().piece_at(square(0, 1)),
        Some(Piece::new(PieceKind::KodamaSamurai, Player::First))
    );

    // Second's last row is rank 1.
    let second_arriving = position_with(
        &[
            (5, 4, PieceKind::Koropokkuru, Player::First),
            (0, 4, PieceKind::Koropokkuru, Player::Second),
            (4, 1, PieceKind::Kodama, Player::Second),
        ],
        NO_HANDS,
        Player::Second,
    );
    assert!(!second_arriving.is_legal_action(step(square(4, 1), square(5, 1))));
    assert!(second_arriving.is_legal_action(promotion(square(4, 1), square(5, 1))));
}

#[test]
fn a_dropped_piece_keeps_its_unpromoted_face_inside_the_zone() {
    let position = position_with(
        &[
            (5, 4, PieceKind::Koropokkuru, Player::First),
            (0, 4, PieceKind::Koropokkuru, Player::Second),
        ],
        [[0, 1, 0], [0; 3]],
        Player::First,
    );
    let mut game = Game::from_position(position);
    let transition = game
        .apply(Action::Drop {
            piece: HandPiece::Oni,
            to: square(0, 0),
        })
        .expect("an Oni may be dropped on the last row");

    assert!(!transition.promoted);
    assert_eq!(
        game.position().piece_at(square(0, 0)),
        Some(Piece::new(PieceKind::Oni, Player::First))
    );
}

#[test]
fn a_kodama_cannot_be_dropped_on_the_last_row() {
    let position = position_with(
        &[
            (5, 4, PieceKind::Koropokkuru, Player::First),
            (0, 4, PieceKind::Koropokkuru, Player::Second),
        ],
        [[1, 0, 1], [0; 3]],
        Player::First,
    );
    assert!(!position.is_legal_action(kodama_drop(square(0, 0))));
    assert!(position.is_legal_action(kodama_drop(square(1, 0))));
    // The restriction is specific to the piece that could not move again.
    assert!(position.is_legal_action(Action::Drop {
        piece: HandPiece::Kirin,
        to: square(0, 0),
    }));

    let second = position_with(
        &[
            (5, 4, PieceKind::Koropokkuru, Player::First),
            (0, 4, PieceKind::Koropokkuru, Player::Second),
        ],
        [[0; 3], [0, 0, 1]],
        Player::Second,
    );
    assert!(!second.is_legal_action(kodama_drop(square(5, 0))));
    assert!(second.is_legal_action(kodama_drop(square(0, 0))));
}

#[test]
fn a_kodama_cannot_be_dropped_in_a_column_holding_its_owners_kodama() {
    let position = position_with(
        &[
            (5, 4, PieceKind::Koropokkuru, Player::First),
            (0, 4, PieceKind::Koropokkuru, Player::Second),
            (3, 0, PieceKind::Kodama, Player::First),
            (3, 1, PieceKind::KodamaSamurai, Player::First),
            (2, 2, PieceKind::Kodama, Player::Second),
        ],
        [[0, 0, 1], [0; 3]],
        Player::First,
    );
    assert!(!position.is_legal_action(kodama_drop(square(4, 0))));
    assert!(!position.is_legal_action(kodama_drop(square(1, 0))));
    // A Kodama Samurai or an opposing Kodama does not block the column.
    assert!(position.is_legal_action(kodama_drop(square(4, 1))));
    assert!(position.is_legal_action(kodama_drop(square(4, 2))));
}

/// Second's Koropokkuru sits in the a6 corner. First's Kirin on b4 guards a5
/// and b5, and First's Kirin on c5 guards b6: a Kodama dropped on a5 attacks
/// the Koropokkuru, which can neither flee nor capture it.
fn drop_checkmate_pieces() -> Vec<(u8, u8, PieceKind, Player)> {
    vec![
        (5, 4, PieceKind::Koropokkuru, Player::First),
        (0, 0, PieceKind::Koropokkuru, Player::Second),
        (2, 1, PieceKind::Kirin, Player::First),
        (1, 2, PieceKind::Kirin, Player::First),
    ]
}

#[test]
fn a_kodama_drop_may_not_deliver_checkmate() {
    let hands = [[1, 0, 1], [0; 3]];
    let mating = position_with(&drop_checkmate_pieces(), hands, Player::First);
    assert!(!mating.is_legal_action(kodama_drop(square(1, 0))));
    assert!(!mating.legal_actions().contains(&kodama_drop(square(1, 0))));
    // Elsewhere in the same column the drop does not attack the Koropokkuru.
    assert!(mating.is_legal_action(kodama_drop(square(2, 0))));

    // The restriction concerns Kodama drops only: another piece may be
    // dropped with checkmate, and a Kodama may deliver it by moving.
    let mut game = Game::from_position(mating);
    let transition = game
        .apply(Action::Drop {
            piece: HandPiece::Kirin,
            to: square(1, 0),
        })
        .expect("a Kirin may be dropped with checkmate");
    assert_eq!(
        transition.outcome,
        Outcome::Win {
            player: Player::First,
            reason: WinReason::OpponentHasNoLegalAction,
        }
    );

    let mut pieces = drop_checkmate_pieces();
    pieces.push((2, 0, PieceKind::Kodama, Player::First));
    let advancing = position_with(&pieces, NO_HANDS, Player::First);
    assert!(advancing.is_legal_action(step(square(2, 0), square(1, 0))));

    // The same position seen from the other side: Second drops on e2 against
    // First's Koropokkuru in the e1 corner.
    let rotated = drop_checkmate_pieces()
        .into_iter()
        .map(|(row, column, kind, owner)| (5 - row, 4 - column, kind, owner.opponent()))
        .collect::<Vec<_>>();
    let second_mating = position_with(&rotated, [[0; 3], [0, 0, 1]], Player::Second);
    assert!(!second_mating.is_legal_action(kodama_drop(square(4, 4))));
    assert!(second_mating.is_legal_action(kodama_drop(square(3, 4))));
}

#[test]
fn a_kodama_drop_with_check_is_legal_when_the_koropokkuru_can_be_saved() {
    let hands = [[0, 0, 1], [0; 3]];
    let drop = kodama_drop(square(1, 0));

    // Without the Kirin on c5, the Koropokkuru flees to b6.
    let mut pieces = drop_checkmate_pieces();
    pieces.retain(|&(row, column, ..)| (row, column) != (1, 2));
    assert!(position_with(&pieces, hands, Player::First).is_legal_action(drop));

    // Without the Kirin on b4, the Koropokkuru captures the unguarded Kodama.
    let mut pieces = drop_checkmate_pieces();
    pieces.retain(|&(row, column, ..)| (row, column) != (2, 1));
    assert!(position_with(&pieces, hands, Player::First).is_legal_action(drop));

    // Another defender may capture the Kodama: Second's Oni on b6 reaches a5.
    let mut pieces = drop_checkmate_pieces();
    pieces.push((0, 1, PieceKind::Oni, Player::Second));
    assert!(position_with(&pieces, hands, Player::First).is_legal_action(drop));

    // The opponent is not checkmated if it can capture First's Koropokkuru.
    let mut pieces = drop_checkmate_pieces();
    pieces.push((4, 4, PieceKind::Kirin, Player::Second));
    assert!(position_with(&pieces, hands, Player::First).is_legal_action(drop));
}

#[test]
fn capturing_the_opposing_koropokkuru_wins_immediately() {
    let position = position_with(
        &[
            (5, 2, PieceKind::Koropokkuru, Player::First),
            (1, 1, PieceKind::Koropokkuru, Player::Second),
            (2, 1, PieceKind::Kirin, Player::First),
        ],
        NO_HANDS,
        Player::First,
    );
    let mut game = Game::from_position(position);
    let transition = game
        .apply(step(square(2, 1), square(1, 1)))
        .expect("king capture must be legal");

    assert_eq!(
        transition.outcome,
        Outcome::Win {
            player: Player::First,
            reason: WinReason::KoropokkuruCaptured,
        }
    );
}

#[test]
fn reaching_the_last_row_with_the_koropokkuru_does_not_win() {
    let position = position_with(
        &[
            (1, 1, PieceKind::Koropokkuru, Player::First),
            (5, 4, PieceKind::Koropokkuru, Player::Second),
        ],
        NO_HANDS,
        Player::First,
    );
    let mut game = Game::from_position(position);
    let transition = game
        .apply(step(square(1, 1), square(0, 1)))
        .expect("the Koropokkuru may enter the last row");

    assert_eq!(transition.outcome, Outcome::Ongoing);
}

#[test]
fn a_koropokkuru_cannot_move_to_an_attacked_square() {
    let position = position_with(
        &[
            (1, 1, PieceKind::Koropokkuru, Player::First),
            (5, 4, PieceKind::Koropokkuru, Player::Second),
            (0, 0, PieceKind::Kirin, Player::Second),
        ],
        NO_HANDS,
        Player::First,
    );
    // Second's Kirin on a6 guards b6 (sideways) and a5 (forward).
    assert!(!position.is_legal_action(step(square(1, 1), square(0, 1))));
    assert!(!position.is_legal_action(step(square(1, 1), square(1, 0))));
    assert!(position.is_legal_action(step(square(1, 1), square(0, 2))));
}

#[test]
fn other_pieces_may_leave_the_koropokkuru_exposed_and_lose_it() {
    let position = position_with(
        &[
            (5, 2, PieceKind::Koropokkuru, Player::First),
            (0, 2, PieceKind::Koropokkuru, Player::Second),
            (4, 2, PieceKind::Kirin, Player::Second),
            (5, 0, PieceKind::Oni, Player::First),
        ],
        NO_HANDS,
        Player::First,
    );
    let mut game = Game::from_position(position);
    // First ignores the attack on its Koropokkuru: legal, and losing.
    let ignoring = game
        .apply(step(square(5, 0), square(4, 0)))
        .expect("a piece other than the Koropokkuru moves freely");
    assert_eq!(ignoring.outcome, Outcome::Ongoing);

    let capture = game
        .apply(step(square(4, 2), square(5, 2)))
        .expect("the exposed Koropokkuru can be captured");
    assert_eq!(
        capture.outcome,
        Outcome::Win {
            player: Player::Second,
            reason: WinReason::KoropokkuruCaptured,
        }
    );
}

#[test]
fn having_no_legal_action_is_a_loss() {
    let position = position_with(
        &[
            (5, 4, PieceKind::Koropokkuru, Player::First),
            (0, 0, PieceKind::Koropokkuru, Player::Second),
            (2, 1, PieceKind::Kirin, Player::First),
            (1, 3, PieceKind::Kirin, Player::First),
        ],
        NO_HANDS,
        Player::First,
    );
    let mut game = Game::from_position(position);
    // The Kirin steps to c5 and guards b6. The Koropokkuru is not attacked,
    // yet every square around it is, and Second has nothing else to play.
    let transition = game
        .apply(step(square(1, 3), square(1, 2)))
        .expect("the quiet move must be legal");

    let lost = Outcome::Win {
        player: Player::First,
        reason: WinReason::OpponentHasNoLegalAction,
    };
    assert_eq!(transition.outcome, lost);
    // A game started from that position is decided before any move.
    assert_eq!(Game::from_position(*game.position()).outcome(), lost);
}

#[test]
fn third_occurrence_of_a_position_is_a_draw() {
    let position = position_with(
        &[
            (5, 2, PieceKind::Koropokkuru, Player::First),
            (0, 2, PieceKind::Koropokkuru, Player::Second),
        ],
        NO_HANDS,
        Player::First,
    );
    let mut game = Game::from_position(position);
    let cycle = [
        step(square(5, 2), square(5, 1)),
        step(square(0, 2), square(0, 3)),
        step(square(5, 1), square(5, 2)),
        step(square(0, 3), square(0, 2)),
    ];

    for _ in 0..2 {
        for action in cycle {
            game.apply(action)
                .expect("repetition cycle must stay legal");
        }
    }

    assert_eq!(
        game.outcome(),
        Outcome::Draw {
            reason: DrawReason::ThreefoldRepetition,
        }
    );
    assert_eq!(game.current_repetition_count(), 3);
}

#[test]
fn an_undecided_game_is_drawn_at_the_move_limit() {
    // The two Koropokkuru walk closed paths of 8 and 9 distinct squares, far
    // apart. The paths only realign every 72 turns, so no position occurs
    // three times before the limit.
    let first_path = [
        (5, 0),
        (5, 1),
        (5, 2),
        (4, 2),
        (3, 2),
        (3, 1),
        (3, 0),
        (4, 0),
    ];
    let second_path = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 2),
        (1, 1),
        (1, 0),
    ];
    let at = |path: &[(u8, u8)], turn: usize| {
        let (row, column) = path[turn % path.len()];
        square(row, column)
    };
    let mut game = Game::from_position(position_with(
        &[
            (5, 0, PieceKind::Koropokkuru, Player::First),
            (0, 0, PieceKind::Koropokkuru, Player::Second),
        ],
        NO_HANDS,
        Player::First,
    ));

    for ply in 0..MOVE_LIMIT_PLIES {
        assert_eq!(game.outcome(), Outcome::Ongoing, "after {ply} plies");
        let path: &[(u8, u8)] = if ply % 2 == 0 {
            &first_path
        } else {
            &second_path
        };
        game.apply(step(at(path, ply / 2), at(path, ply / 2 + 1)))
            .expect("each step of the walk is legal");
    }

    assert_eq!(
        game.outcome(),
        Outcome::Draw {
            reason: DrawReason::MoveLimit,
        }
    );
}

#[test]
fn notation_is_unambiguous_and_round_trips() {
    let board_move = step(square(3, 2), square(2, 2));
    let promoting_move = promotion(square(2, 4), square(1, 4));
    let drop = kodama_drop(square(2, 0));
    assert_eq!(board_move.to_string(), "c3-c4");
    assert_eq!(promoting_move.to_string(), "e4-e5+");
    assert_eq!(drop.to_string(), "kodama@a4");
    assert_eq!(Action::from_str("c3-c4").unwrap(), board_move);
    assert_eq!(Action::from_str("E4-E5+").unwrap(), promoting_move);
    assert_eq!(Action::from_str("KODAMA@A4").unwrap(), drop);
    assert_eq!(
        Action::from_str("kirin@e6").unwrap().to_string(),
        "kirin@e6"
    );
    assert_eq!(Action::from_str("oni@a1").unwrap().to_string(), "oni@a1");

    // The board ends at file `e` and rank 6.
    for invalid in ["f1-e1", "a7-a6", "a0-a1", "tanuki@a4", "c3-c4++", "c3"] {
        assert!(Action::from_str(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn replay_json_round_trips_and_revalidates_actions() {
    let mut game = Game::new(Player::First);
    game.apply(Action::from_str("c3-c4").unwrap()).unwrap();
    let replay = Replay::from_game(&game, Some(42));
    let json = replay.to_json_pretty().unwrap();
    let decoded = Replay::from_json(&json).unwrap();

    assert_eq!(decoded, replay);
    assert_eq!(decoded.to_game().unwrap().position(), game.position());
}

#[test]
fn replay_rejects_truncated_json_and_unknown_versions() {
    assert!(matches!(
        Replay::from_json("{\"format_version\":"),
        Err(ReplayError::Json(_))
    ));

    let game = Game::new(Player::First);
    let mut replay = Replay::from_game(&game, None);
    replay.format_version = 99;
    assert!(matches!(
        replay.to_game(),
        Err(ReplayError::UnsupportedFormatVersion(99))
    ));
}

#[test]
fn json_cannot_bypass_validated_indices_or_material_bounds() {
    for index in 0..=u8::MAX {
        assert_eq!(
            serde_json::from_str::<Square>(&index.to_string()).ok(),
            Square::from_index(index)
        );
    }
    for index in 0..=1_000_u16 {
        assert_eq!(
            serde_json::from_str::<yokai::PolicyIndex>(&index.to_string()).ok(),
            yokai::PolicyIndex::new(index)
        );
    }

    let initial = Position::initial(Player::First);
    let mut json = serde_json::to_value(initial).unwrap();
    assert_eq!(
        serde_json::from_value::<Position>(json.clone()).unwrap(),
        initial
    );
    json["hands"][0][0] = 255.into();
    assert!(serde_json::from_value::<Position>(json).is_err());

    let mut game = Game::new(Player::First);
    game.apply("c3-c4".parse().unwrap()).unwrap();
    let mut json = serde_json::to_value(Replay::from_game(&game, None)).unwrap();
    json["actions"][0]["from"] = 255.into();
    assert!(matches!(
        Replay::from_json(&json.to_string()),
        Err(ReplayError::Json(_))
    ));
}
