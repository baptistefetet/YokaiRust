//! Board fixtures shared by the engine, search and learning tests.

use yokai::{BOARD_SQUARES, Piece, PieceKind, Player, Position, Square};

pub(crate) fn square(row: u8, column: u8) -> Square {
    Square::new(row, column).expect("test square must be valid")
}

pub(crate) fn position_with(
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
