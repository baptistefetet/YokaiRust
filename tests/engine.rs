//! Executable specification of board setup, movement and official outcomes.

use std::str::FromStr;

use yokai::{
    Action, BOARD_SQUARES, DrawReason, Game, HandPiece, Outcome, Piece, PieceKind, Player,
    Position, Replay, ReplayError, Square, WinReason,
};

fn square(row: u8, column: u8) -> Square {
    Square::new(row, column).expect("test square must be valid")
}

fn position_with(
    pieces: &[(u8, u8, PieceKind, Player)],
    hands: [[u8; 3]; 2],
    side_to_move: Player,
) -> Position {
    let mut board = [None; BOARD_SQUARES];
    for &(row, column, kind, owner) in pieces {
        board[square(row, column).index()] = Some(Piece::new(kind, owner));
    }
    Position::from_parts(board, hands, side_to_move).expect("test position must be valid")
}

#[test]
fn initial_position_matches_the_official_setup() {
    let position = Position::initial(Player::First);
    assert_eq!(position.physical_piece_count(), 8);
    assert_eq!(
        position.piece_at(square(3, 1)),
        Some(Piece::new(PieceKind::Koropokkuru, Player::First))
    );
    assert_eq!(
        position.piece_at(square(0, 1)),
        Some(Piece::new(PieceKind::Koropokkuru, Player::Second))
    );
    assert_eq!(position.legal_actions().len(), 4);
}

#[test]
fn both_players_move_kodama_forward_in_absolute_coordinates() {
    let first = position_with(
        &[
            (3, 0, PieceKind::Koropokkuru, Player::First),
            (0, 2, PieceKind::Koropokkuru, Player::Second),
            (2, 1, PieceKind::Kodama, Player::First),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    assert!(first.is_legal_action(Action::Move {
        from: square(2, 1),
        to: square(1, 1),
    }));

    let second = position_with(
        &[
            (3, 0, PieceKind::Koropokkuru, Player::First),
            (0, 2, PieceKind::Koropokkuru, Player::Second),
            (1, 1, PieceKind::Kodama, Player::Second),
        ],
        [[0; 3]; 2],
        Player::Second,
    );
    assert!(second.is_legal_action(Action::Move {
        from: square(1, 1),
        to: square(2, 1),
    }));
}

#[test]
fn a_captured_samurai_returns_to_hand_as_a_kodama() {
    let position = position_with(
        &[
            (3, 1, PieceKind::Koropokkuru, Player::First),
            (0, 1, PieceKind::Koropokkuru, Player::Second),
            (2, 0, PieceKind::Tanuki, Player::First),
            (1, 0, PieceKind::KodamaSamurai, Player::Second),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let mut game = Game::from_position(position);
    let transition = game
        .apply(Action::Move {
            from: square(2, 0),
            to: square(1, 0),
        })
        .expect("capture must be legal");

    assert_eq!(transition.captured, Some(PieceKind::KodamaSamurai));
    assert_eq!(
        game.position().hand_count(Player::First, HandPiece::Kodama),
        1
    );
}

#[test]
fn an_on_board_kodama_promotes_when_it_enters_the_goal_row() {
    let position = position_with(
        &[
            (3, 0, PieceKind::Koropokkuru, Player::First),
            (0, 2, PieceKind::Koropokkuru, Player::Second),
            (1, 1, PieceKind::Kodama, Player::First),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let mut game = Game::from_position(position);
    let transition = game
        .apply(Action::Move {
            from: square(1, 1),
            to: square(0, 1),
        })
        .expect("promotion move must be legal");

    assert!(transition.promoted);
    assert_eq!(
        game.position().piece_at(square(0, 1)),
        Some(Piece::new(PieceKind::KodamaSamurai, Player::First))
    );
}

#[test]
fn a_kodama_can_be_dropped_on_the_last_row_without_promotion() {
    let position = position_with(
        &[
            (3, 1, PieceKind::Koropokkuru, Player::First),
            (0, 2, PieceKind::Koropokkuru, Player::Second),
        ],
        [[0, 0, 1], [0; 3]],
        Player::First,
    );
    let action = Action::Drop {
        piece: HandPiece::Kodama,
        to: square(0, 0),
    };
    assert!(position.is_legal_action(action));

    let mut game = Game::from_position(position);
    let transition = game.apply(action).expect("official 3x4 drop must be legal");
    assert!(!transition.promoted);
    assert_eq!(
        game.position().piece_at(square(0, 0)),
        Some(Piece::new(PieceKind::Kodama, Player::First))
    );
}

#[test]
fn capturing_the_opposing_koropokkuru_wins_immediately() {
    let position = position_with(
        &[
            (3, 1, PieceKind::Koropokkuru, Player::First),
            (1, 1, PieceKind::Koropokkuru, Player::Second),
            (2, 1, PieceKind::Tanuki, Player::First),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let mut game = Game::from_position(position);
    let transition = game
        .apply(Action::Move {
            from: square(2, 1),
            to: square(1, 1),
        })
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
fn a_safe_koropokkuru_on_the_goal_row_wins() {
    let position = position_with(
        &[
            (1, 1, PieceKind::Koropokkuru, Player::First),
            (2, 2, PieceKind::Koropokkuru, Player::Second),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let mut game = Game::from_position(position);
    let transition = game
        .apply(Action::Move {
            from: square(1, 1),
            to: square(0, 1),
        })
        .expect("safe goal move must be legal");

    assert_eq!(
        transition.outcome,
        Outcome::Win {
            player: Player::First,
            reason: WinReason::KoropokkuruReachedGoal,
        }
    );
}

#[test]
fn a_koropokkuru_cannot_move_to_an_attacked_square() {
    let position = position_with(
        &[
            (1, 1, PieceKind::Koropokkuru, Player::First),
            (2, 2, PieceKind::Koropokkuru, Player::Second),
            (0, 0, PieceKind::Tanuki, Player::Second),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    assert!(!position.is_legal_action(Action::Move {
        from: square(1, 1),
        to: square(0, 1),
    }));
}

#[test]
fn having_no_legal_action_is_a_loss() {
    let position = position_with(
        &[
            (2, 1, PieceKind::Koropokkuru, Player::First),
            (0, 0, PieceKind::Koropokkuru, Player::Second),
            (1, 2, PieceKind::Tanuki, Player::First),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let mut game = Game::from_position(position);
    let transition = game
        .apply(Action::Move {
            from: square(1, 2),
            to: square(1, 1),
        })
        .expect("mating move must be legal");

    assert_eq!(
        transition.outcome,
        Outcome::Win {
            player: Player::First,
            reason: WinReason::OpponentHasNoLegalAction,
        }
    );
}

#[test]
fn third_occurrence_of_a_position_is_a_draw() {
    let position = position_with(
        &[
            (3, 1, PieceKind::Koropokkuru, Player::First),
            (0, 1, PieceKind::Koropokkuru, Player::Second),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let mut game = Game::from_position(position);
    let cycle = [
        Action::Move {
            from: square(3, 1),
            to: square(3, 0),
        },
        Action::Move {
            from: square(0, 1),
            to: square(0, 2),
        },
        Action::Move {
            from: square(3, 0),
            to: square(3, 1),
        },
        Action::Move {
            from: square(0, 2),
            to: square(0, 1),
        },
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
fn notation_is_unambiguous_and_round_trips() {
    let board_move = Action::Move {
        from: square(2, 1),
        to: square(1, 1),
    };
    let drop = Action::Drop {
        piece: HandPiece::Kodama,
        to: square(0, 0),
    };
    assert_eq!(board_move.to_string(), "b2-b3");
    assert_eq!(drop.to_string(), "kodama@a4");
    assert_eq!(Action::from_str("b2-b3").unwrap(), board_move);
    assert_eq!(Action::from_str("KODAMA@A4").unwrap(), drop);
}

#[test]
fn replay_json_round_trips_and_revalidates_actions() {
    let mut game = Game::new(Player::First);
    game.apply(Action::from_str("b2-b3").unwrap()).unwrap();
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
