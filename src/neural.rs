//! Versioned neural representation shared by every inference backend.

pub mod checkpoint;
pub mod evaluator;
pub mod model;
pub mod service;

use burn::prelude::{Backend, Tensor, TensorData};

use crate::{
    BOARD_HEIGHT, BOARD_SQUARES, BOARD_WIDTH, Game, HandPiece, POLICY_ACTIONS, PieceKind, Player,
    Position, Square,
};

pub const ENCODER_VERSION: u16 = 3;
pub const HISTORY_LENGTH: usize = 8;
pub const HISTORY_POSITIONS: usize = HISTORY_LENGTH - 1;
const POSITION_PLANES: usize = 16;
pub const INPUT_PLANES: usize = POSITION_PLANES * HISTORY_LENGTH + 2;
pub const INPUT_VALUES: usize = INPUT_PLANES * BOARD_SQUARES;
pub const POLICY_CONTEXT_FEATURES: usize = POLICY_ACTIONS * 2;

const CURRENT_PIECES_OFFSET: usize = 0;
const OPPONENT_PIECES_OFFSET: usize = 5;
const CURRENT_HAND_OFFSET: usize = 10;
const OPPONENT_HAND_OFFSET: usize = 13;
const REPETITION_PLANE: usize = INPUT_PLANES - 2;
const STARTER_PLANE: usize = INPUT_PLANES - 1;

/// A channel-first `[130, 4, 3]` input stored contiguously.
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

/// Packs per-action repetition counts and immediate-draw flags.
///
/// # Panics
///
/// Panics when `contexts` is empty.
#[must_use]
pub fn policy_context_batch_tensor<B: Backend>(
    contexts: &[[u8; POLICY_ACTIONS]],
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(
        !contexts.is_empty(),
        "a policy context batch cannot be empty"
    );
    let values = contexts
        .iter()
        .flat_map(|context| {
            context.iter().flat_map(|&count| {
                [
                    f32::from(count.min(3)) / 3.0,
                    if count >= 3 { 1.0 } else { 0.0 },
                ]
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [contexts.len(), POLICY_CONTEXT_FEATURES]),
        device,
    )
}

/// Encodes a game from the perspective of its player to move.
#[must_use]
pub fn encode_game(game: &Game) -> EncodedPosition {
    let positions = game.position_history();
    let history = std::array::from_fn(|offset| {
        positions
            .len()
            .checked_sub(offset + 2)
            .map(|index| positions[index])
    });
    encode_position_with_history(
        game.position(),
        game.current_repetition_count(),
        game.position().side_to_move() == game.initial_player(),
        &history,
    )
}

/// Encodes a position using the canonical perspective of its player to move.
///
/// This compatibility helper encodes an isolated position with zero-filled
/// history. Inference and training should provide real history whenever it is
/// available.
#[must_use]
pub fn encode_position(position: &Position, repetition_count: u8) -> EncodedPosition {
    encode_position_with_history(position, repetition_count, true, &[None; HISTORY_POSITIONS])
}

/// Encodes the current position and up to seven preceding positions.
///
/// Each time step contains five current-player piece planes, five opponent
/// piece planes and six normalized hand-count planes. Every frame is oriented
/// and owned from the current player's perspective. Missing history is zero;
/// the final plane stores the current occurrence count normalized by three.
#[must_use]
pub fn encode_position_with_history(
    position: &Position,
    repetition_count: u8,
    current_player_is_starter: bool,
    history: &[Option<Position>; HISTORY_POSITIONS],
) -> EncodedPosition {
    let current = position.side_to_move();
    let mut values = [0.0; INPUT_VALUES];
    encode_frame(&mut values, 0, position, current);
    for (offset, historical) in history.iter().enumerate() {
        if let Some(historical) = historical {
            encode_frame(&mut values, offset + 1, historical, current);
        }
    }

    fill_plane(
        &mut values,
        REPETITION_PLANE,
        f32::from(repetition_count.min(3)) / 3.0,
    );
    fill_plane(
        &mut values,
        STARTER_PLANE,
        if current_player_is_starter { 1.0 } else { 0.0 },
    );
    EncodedPosition { values }
}

fn encode_frame(
    values: &mut [f32; INPUT_VALUES],
    frame: usize,
    position: &Position,
    current: Player,
) {
    let frame_start = frame * POSITION_PLANES;

    for absolute_square in Square::ALL {
        let Some(piece) = position.piece_at(absolute_square) else {
            continue;
        };
        let canonical_square = canonical_square(absolute_square, current);
        let owner_offset = if piece.owner == current {
            CURRENT_PIECES_OFFSET
        } else {
            OPPONENT_PIECES_OFFSET
        };
        let plane = frame_start + owner_offset + piece_plane(piece.kind);
        values[index(plane, canonical_square)] = 1.0;
    }

    for player in [current, current.opponent()] {
        let plane_start = if player == current {
            CURRENT_HAND_OFFSET
        } else {
            OPPONENT_HAND_OFFSET
        };
        for piece in HandPiece::ALL {
            let normalized = f32::from(position.hand_count(player, piece)) / 2.0;
            fill_plane(
                values,
                frame_start + plane_start + piece.index(),
                normalized,
            );
        }
    }
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
