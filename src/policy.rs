//! Stable bijection between domain [`Action`] values and neural policy slots.
//!
//! The network always sees the side to move as if it were [`Player::First`].
//! Rotating Second's actions here lets one set of weights learn both seats and
//! prevents board-orientation details from leaking into MCTS or the UI.
//!
//! Layout of the 570-slot policy vector (canonical perspective):
//!
//! ```text
//! slots   0..480  board moves — index = (origin_square × 8 + direction) × 2
//!                                       + promotion flag
//!                 (30 origin squares × 8 king-move directions × 2)
//! slots 480..570  drops       — index = 480 + destination_square × 3 + piece
//!                 (30 destination squares × 3 droppable hand pieces)
//!
//! direction codes as (row delta, column delta):
//!    0:(-1,-1)  1:(-1, 0)  2:(-1,+1)
//!    3:( 0,-1)             4:( 0,+1)
//!    5:(+1,-1)  6:(+1, 0)  7:(+1,+1)
//! ```
//!
//! A step and its promoting variant are different decisions, so they occupy
//! neighbouring slots. Most slots are geometrically valid but illegal in a
//! given position; the search masks them with the position's actual legal
//! actions.

use serde::{Deserialize, Serialize};

use crate::game::{Action, BOARD_SQUARES, Game, HandPiece, Player, Square};

/// Width of the policy head: 30 origins × 8 directions × 2 promotion flags
/// + 30 squares × 3 drops.
pub const POLICY_ACTIONS: usize = BOARD_POLICY_ACTIONS as usize + DROP_POLICY_ACTIONS as usize;
const MOVE_DIRECTIONS: u16 = DIRECTIONS.len() as u16;
const PROMOTION_FLAGS: u16 = 2;
const DROP_KINDS: u16 = HandPiece::ALL.len() as u16;
const BOARD_POLICY_ACTIONS: u16 = BOARD_SQUARES as u16 * MOVE_DIRECTIONS * PROMOTION_FLAGS;
const DROP_POLICY_ACTIONS: u16 = BOARD_SQUARES as u16 * DROP_KINDS;

/// A validated index into the fixed `AlphaZero` policy vector.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PolicyIndex(u16);

impl<'de> Deserialize<'de> for PolicyIndex {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let index = u16::deserialize(deserializer)?;
        Self::new(index).ok_or_else(|| {
            serde::de::Error::custom(format!("policy index must be in 0..{POLICY_ACTIONS}"))
        })
    }
}

impl PolicyIndex {
    /// Validates and wraps a raw policy-vector index.
    #[must_use]
    pub const fn new(index: u16) -> Option<Self> {
        if (index as usize) < POLICY_ACTIONS {
            Some(Self(index))
        } else {
            None
        }
    }

    /// Returns the compact raw index.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }

    /// Returns the index in the type expected by Rust slices and tensors.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl Action {
    /// Encodes an action from the current player's canonical perspective.
    ///
    /// # Examples
    ///
    /// ```
    /// use yokai::{Action, Player, Square};
    ///
    /// let from = Square::new(2, 1).unwrap(); // square index 11
    /// let to = Square::new(1, 1).unwrap(); // one row toward the top
    /// let action = Action::Move { from, to, promote: true };
    ///
    /// let index = action.policy_index(Player::First).unwrap();
    /// // (origin × 8 + direction) × 2 + promotion flag
    /// assert_eq!(index.as_usize(), (11 * 8 + 1) * 2 + 1);
    /// assert_eq!(Action::from_policy_index(index, Player::First), Some(action));
    /// ```
    #[must_use]
    pub fn policy_index(self, player: Player) -> Option<PolicyIndex> {
        match self {
            Self::Move { from, to, promote } => {
                let canonical_from = canonical_square(from, player);
                let canonical_to = canonical_square(to, player);
                let row_delta = i16::from(canonical_to.row()) - i16::from(canonical_from.row());
                let column_delta =
                    i16::from(canonical_to.column()) - i16::from(canonical_from.column());
                let direction = direction_index(row_delta, column_delta)?;
                let step =
                    u16::try_from(canonical_from.index()).ok()? * MOVE_DIRECTIONS + direction;
                PolicyIndex::new(step * PROMOTION_FLAGS + u16::from(promote))
            }
            Self::Drop { piece, to } => {
                let canonical_to = canonical_square(to, player);
                let destination = u16::try_from(canonical_to.index()).ok()?;
                let piece_index = u16::try_from(piece.index()).ok()?;
                PolicyIndex::new(BOARD_POLICY_ACTIONS + destination * DROP_KINDS + piece_index)
            }
        }
    }

    /// Decodes geometry only. Legality still depends on the game position.
    #[must_use]
    pub fn from_policy_index(index: PolicyIndex, player: Player) -> Option<Self> {
        let raw = index.get();
        if raw < BOARD_POLICY_ACTIONS {
            let step = raw / PROMOTION_FLAGS;
            let promote = raw % PROMOTION_FLAGS == 1;
            let canonical_from = Square::from_index(u8::try_from(step / MOVE_DIRECTIONS).ok()?)?;
            let (row_delta, column_delta) = direction_delta(step % MOVE_DIRECTIONS)?;
            let canonical_to = canonical_from.offset(row_delta, column_delta)?;
            Some(Self::Move {
                from: decanonical_square(canonical_from, player),
                to: decanonical_square(canonical_to, player),
                promote,
            })
        } else {
            let drop_index = raw - BOARD_POLICY_ACTIONS;
            let piece = HandPiece::ALL[usize::from(drop_index % DROP_KINDS)];
            let canonical_to = Square::from_index(u8::try_from(drop_index / DROP_KINDS).ok()?)?;
            Some(Self::Drop {
                piece,
                to: decanonical_square(canonical_to, player),
            })
        }
    }
}

pub(crate) const fn canonical_square(square: Square, player: Player) -> Square {
    match player {
        Player::First => square,
        Player::Second => square.rotated(),
    }
}

/// The neural policy receives the same repetition context during play and training.
pub(crate) fn action_repetition_counts(game: &Game) -> [u8; POLICY_ACTIONS] {
    let player = game.position().side_to_move();
    let mut counts = [0; POLICY_ACTIONS];
    for action in game.legal_actions() {
        if let (Some(index), Some(count)) = (
            action.policy_index(player),
            game.repetition_count_after(action),
        ) {
            counts[index.as_usize()] = count;
        }
    }
    counts
}

/// Inverse of [`canonical_square`]. The 180° rotation is an involution
/// (applying it twice is the identity), so the inverse is the same function —
/// this alias only exists to make encode/decode call sites read symmetrically.
const fn decanonical_square(square: Square, player: Player) -> Square {
    canonical_square(square, player)
}

/// The eight king-move directions in canonical perspective, indexed by their
/// policy code. Both lookup functions below read this single table, so the
/// encoding and its inverse can never drift apart.
#[rustfmt::skip]
const DIRECTIONS: [(i8, i8); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
];

const fn direction_index(row_delta: i16, column_delta: i16) -> Option<u16> {
    let mut direction = 0;
    while direction < DIRECTIONS.len() {
        let (row, column) = DIRECTIONS[direction];
        if row as i16 == row_delta && column as i16 == column_delta {
            return Some(direction as u16);
        }
        direction += 1;
    }
    None
}

const fn direction_delta(direction: u16) -> Option<(i8, i8)> {
    if (direction as usize) < DIRECTIONS.len() {
        Some(DIRECTIONS[direction as usize])
    } else {
        None
    }
}
