//! Versioned neural representation shared by every inference backend.

#[cfg(feature = "native")]
pub mod checkpoint;
#[cfg(feature = "native")]
pub mod evaluator;
pub mod model;
#[cfg(feature = "native")]
pub mod service;

use burn::prelude::{Backend, Tensor, TensorData};

use crate::{
    BOARD_HEIGHT, BOARD_SQUARES, BOARD_WIDTH, Game, HandPiece, POLICY_ACTIONS, PieceKind, Player,
    Position, Square,
};

/// Version stored in checkpoints so incompatible feature layouts are rejected.
pub const ENCODER_VERSION: u16 = 4;
/// Number of temporal frames: the current position plus seven predecessors.
pub const HISTORY_LENGTH: usize = 8;
/// Number of preceding positions supplied alongside the current one.
pub const HISTORY_POSITIONS: usize = HISTORY_LENGTH - 1;
const POSITION_PLANES: usize = 10;
/// Total spatial channels (`10 piece planes × 8 temporal frames`).
pub const INPUT_PLANES: usize = POSITION_PLANES * HISTORY_LENGTH;
/// Flat number of values in one `[planes, rows, columns]` board input.
pub const INPUT_VALUES: usize = INPUT_PLANES * BOARD_SQUARES;
/// Hand-count features stored for each temporal frame.
pub const GLOBAL_FEATURES_PER_FRAME: usize = 6;
/// Total non-spatial features: temporal hands, repetition and starter role.
pub const GLOBAL_INPUT_FEATURES: usize = GLOBAL_FEATURES_PER_FRAME * HISTORY_LENGTH + 2;
/// Two repetition features for each of the 132 policy actions.
pub const POLICY_CONTEXT_FEATURES: usize = POLICY_ACTIONS * 2;

const CURRENT_PIECES_OFFSET: usize = 0;
const OPPONENT_PIECES_OFFSET: usize = 5;
const CURRENT_HAND_OFFSET: usize = 0;
const OPPONENT_HAND_OFFSET: usize = 3;
const REPETITION_FEATURE: usize = GLOBAL_INPUT_FEATURES - 2;
const STARTER_FEATURE: usize = GLOBAL_INPUT_FEATURES - 1;

/// A channel-first `[80, 4, 3]` board input and 50 global features.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedPosition {
    values: [f32; INPUT_VALUES],
    global_values: [f32; GLOBAL_INPUT_FEATURES],
}

impl EncodedPosition {
    /// Borrows the flat channel-first board input.
    #[must_use]
    pub const fn values(&self) -> &[f32; INPUT_VALUES] {
        &self.values
    }

    /// Borrows hand, repetition and player-role features.
    #[must_use]
    pub const fn global_values(&self) -> &[f32; GLOBAL_INPUT_FEATURES] {
        &self.global_values
    }

    /// Reads one spatial feature using explicit plane, row and column indices.
    #[must_use]
    pub const fn get(&self, plane: usize, row: usize, column: usize) -> f32 {
        self.values[plane * BOARD_SQUARES + row * BOARD_WIDTH as usize + column]
    }

    /// Reads one non-spatial feature.
    #[must_use]
    pub const fn get_global(&self, feature: usize) -> f32 {
        self.global_values[feature]
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

/// Packs the non-spatial features associated with encoded positions.
///
/// # Panics
///
/// Panics when `positions` is empty.
#[must_use]
pub fn global_batch_tensor<B: Backend>(
    positions: &[EncodedPosition],
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(!positions.is_empty(), "an encoded batch cannot be empty");
    let values = positions
        .iter()
        .flat_map(|position| position.global_values.iter().copied())
        .collect::<Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [positions.len(), GLOBAL_INPUT_FEATURES]),
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
        // `positions` includes the current state. Offset zero therefore selects
        // `len - 2`, the state immediately before it.
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
/// piece planes and six normalized scalar hand counts. Every frame is oriented
/// and owned from the current player's perspective. Missing history is zero.
/// Two final global features store the current occurrence count and whether the
/// current player started the game.
#[must_use]
pub fn encode_position_with_history(
    position: &Position,
    repetition_count: u8,
    current_player_is_starter: bool,
    history: &[Option<Position>; HISTORY_POSITIONS],
) -> EncodedPosition {
    let current = position.side_to_move();
    let mut values = [0.0; INPUT_VALUES];
    let mut global_values = [0.0; GLOBAL_INPUT_FEATURES];
    encode_frame(&mut values, &mut global_values, 0, position, current);
    for (offset, historical) in history.iter().enumerate() {
        if let Some(historical) = historical {
            encode_frame(
                &mut values,
                &mut global_values,
                offset + 1,
                historical,
                current,
            );
        }
    }

    // Repetition is capped because the third occurrence ends an official game;
    // larger values would carry no additional rules information.
    global_values[REPETITION_FEATURE] = f32::from(repetition_count.min(3)) / 3.0;
    global_values[STARTER_FEATURE] = if current_player_is_starter { 1.0 } else { 0.0 };
    EncodedPosition {
        values,
        global_values,
    }
}

fn encode_frame(
    values: &mut [f32; INPUT_VALUES],
    global_values: &mut [f32; GLOBAL_INPUT_FEATURES],
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
        let feature_start = if player == current {
            CURRENT_HAND_OFFSET
        } else {
            OPPONENT_HAND_OFFSET
        };
        for piece in HandPiece::ALL {
            let normalized = f32::from(position.hand_count(player, piece)) / 2.0;
            let feature = frame * GLOBAL_FEATURES_PER_FRAME + feature_start + piece.index();
            global_values[feature] = normalized;
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

const _: () = assert!(BOARD_HEIGHT == 4 && BOARD_WIDTH == 3);
