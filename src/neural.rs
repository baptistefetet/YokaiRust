//! Versioned neural representation shared by every inference backend.

pub mod checkpoint;
pub mod evaluator;
pub mod model;
pub mod service;

use burn::prelude::{Backend, Tensor, TensorData};

use crate::{
    BOARD_HEIGHT, BOARD_SQUARES, BOARD_WIDTH, Game, HandPiece, PieceKind, Player, Position, Square,
};

pub const ENCODER_VERSION: u16 = 1;
pub const INPUT_PLANES: usize = 17;
pub const INPUT_VALUES: usize = INPUT_PLANES * BOARD_SQUARES;

const CURRENT_PIECES_START: usize = 0;
const OPPONENT_PIECES_START: usize = 5;
const CURRENT_HAND_START: usize = 10;
const OPPONENT_HAND_START: usize = 13;
const REPETITION_PLANE: usize = 16;

/// A channel-first `[17, 4, 3]` input stored contiguously.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedPosition {
    values: [f32; INPUT_VALUES],
}

impl EncodedPosition {
    #[must_use]
    pub const fn values(&self) -> &[f32; INPUT_VALUES] {
        &self.values
    }

    #[must_use]
    pub const fn get(&self, plane: usize, row: usize, column: usize) -> f32 {
        self.values[plane * BOARD_SQUARES + row * BOARD_WIDTH as usize + column]
    }
}

/// Packs encoded positions into a channel-first Burn tensor.
///
/// # Panics
///
/// Panics when `positions` is empty; inference and training batches must always
/// contain at least one state.
#[must_use]
pub fn encoded_batch_tensor<B: Backend>(
    positions: &[EncodedPosition],
    device: &B::Device,
) -> Tensor<B, 4> {
    assert!(!positions.is_empty(), "an encoded batch cannot be empty");
    let values = positions
        .iter()
        .flat_map(|position| position.values.iter().copied())
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [positions.len(), INPUT_PLANES, 4, 3]),
        device,
    )
}

/// Encodes a game from the perspective of its player to move.
#[must_use]
pub fn encode_game(game: &Game) -> EncodedPosition {
    encode_position(game.position(), game.current_repetition_count())
}

/// Encodes a position using the canonical perspective of its player to move.
///
/// The first ten planes are the five current and five opposing piece types.
/// Six constant planes hold normalized hand counts, and the last constant
/// plane stores the current occurrence count normalized by three.
#[must_use]
pub fn encode_position(position: &Position, repetition_count: u8) -> EncodedPosition {
    let current = position.side_to_move();
    let mut values = [0.0; INPUT_VALUES];

    for absolute_square in Square::ALL {
        let Some(piece) = position.piece_at(absolute_square) else {
            continue;
        };
        let canonical_square = canonical_square(absolute_square, current);
        let owner_offset = if piece.owner == current {
            CURRENT_PIECES_START
        } else {
            OPPONENT_PIECES_START
        };
        let plane = owner_offset + piece_plane(piece.kind);
        values[index(plane, canonical_square)] = 1.0;
    }

    for player in [current, current.opponent()] {
        let plane_start = if player == current {
            CURRENT_HAND_START
        } else {
            OPPONENT_HAND_START
        };
        for piece in HandPiece::ALL {
            let normalized = f32::from(position.hand_count(player, piece)) / 2.0;
            fill_plane(&mut values, plane_start + piece.index(), normalized);
        }
    }

    fill_plane(
        &mut values,
        REPETITION_PLANE,
        f32::from(repetition_count.min(3)) / 3.0,
    );
    EncodedPosition { values }
}

const fn canonical_square(square: Square, current: Player) -> Square {
    match current {
        Player::First => square,
        Player::Second => square.rotated(),
    }
}

const fn piece_plane(kind: PieceKind) -> usize {
    match kind {
        PieceKind::Koropokkuru => 0,
        PieceKind::Tanuki => 1,
        PieceKind::Kitsune => 2,
        PieceKind::Kodama => 3,
        PieceKind::KodamaSamurai => 4,
    }
}

const fn index(plane: usize, square: Square) -> usize {
    plane * BOARD_SQUARES + square.index()
}

fn fill_plane(values: &mut [f32; INPUT_VALUES], plane: usize, value: f32) {
    let start = plane * BOARD_SQUARES;
    values[start..start + BOARD_SQUARES].fill(value);
}

const _: () = assert!(BOARD_HEIGHT == 4 && BOARD_WIDTH == 3);
