//! Yokai No Mori 3x4 engine and supporting formats.
//!
//! The engine stores the board in an absolute orientation: First starts at the
//! bottom and moves toward row zero. Neural-network canonicalization belongs in
//! the policy/encoder layers, never in the rules themselves.

pub mod game;
pub mod notation;
pub mod policy;
pub mod replay;

pub use game::{
    Action, BOARD_HEIGHT, BOARD_SQUARES, BOARD_WIDTH, DrawReason, Game, HandPiece, MoveError,
    Outcome, Piece, PieceKind, Player, Position, PositionError, RULES_VERSION, Square, Transition,
    WinReason,
};
pub use policy::{POLICY_ACTIONS, PolicyIndex};
pub use replay::{REPLAY_FORMAT_VERSION, Replay, ReplayError};
