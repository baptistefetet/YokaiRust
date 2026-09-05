//! Stable bijection between domain [`Action`] values and neural policy slots.
//!
//! The network always sees the side to move as if it were [`Player::First`].
//! Rotating Second's actions here lets one set of weights learn both seats and
//! prevents board-orientation details from leaking into MCTS or the UI.
//!
//! Layout of the 132-slot policy vector (canonical perspective):
//!
//! ```text
//! slots  0..96   board moves — index = origin_square × 8 + direction
//!                (12 origin squares × 8 king-move directions)
//! slots 96..132  drops       — index = 96 + destination_square × 3 + piece
//!                (12 destination squares × 3 droppable hand pieces)
//!
//! direction codes as (row delta, column delta):
//!    0:(-1,-1)  1:(-1, 0)  2:(-1,+1)
//!    3:( 0,-1)             4:( 0,+1)
//!    5:(+1,-1)  6:(+1, 0)  7:(+1,+1)
//! ```
//!
//! Most slots are geometrically valid but illegal in a given position; the
//! search masks them with the position's actual legal actions.

use serde::{Deserialize, Serialize};

use crate::game::{Action, BOARD_SQUARES, HandPiece, Player, Square};

/// Width of the policy head: 12 origins × 8 directions + 12 squares × 3 drops.
pub const POLICY_ACTIONS: usize = 132;
const BOARD_POLICY_ACTIONS: u8 = 96;

/// A validated index into the fixed `AlphaZero` policy vector.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct PolicyIndex(u8);

impl<'de> Deserialize<'de> for PolicyIndex {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let index = u8::deserialize(deserializer)?;
        Self::new(index).ok_or_else(|| serde::de::Error::custom("policy index must be in 0..132"))
    }
}

impl PolicyIndex {
    /// Validates and wraps a raw policy-vector index.
    #[must_use]
    pub const fn new(index: u8) -> Option<Self> {
        if (index as usize) < POLICY_ACTIONS {
            Some(Self(index))
        } else {
            None
        }
    }

    /// Returns the compact raw index.
    #[must_use]
    pub const fn get(self) -> u8 {
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
    /// let from = Square::new(2, 1).unwrap(); // square index 7
    /// let to = Square::new(1, 1).unwrap(); // one row toward the top
    /// let action = Action::Move { from, to };
    ///
    /// let index = action.policy_index(Player::First).unwrap();
    /// assert_eq!(index.as_usize(), 7 * 8 + 1); // origin × 8 + direction
    /// assert_eq!(Action::from_policy_index(index, Player::First), Some(action));
    /// ```
    #[must_use]
    pub fn policy_index(self, player: Player) -> Option<PolicyIndex> {
        match self {
            Self::Move { from, to } => {
                let canonical_from = canonical_square(from, player);
                let canonical_to = canonical_square(to, player);
                let row_delta = i16::from(canonical_to.row()) - i16::from(canonical_from.row());
                let column_delta =
                    i16::from(canonical_to.column()) - i16::from(canonical_from.column());
                let direction = direction_index(row_delta, column_delta)?;
                let index = u8::try_from(canonical_from.index()).ok()? * 8 + direction;
                PolicyIndex::new(index)
            }
            Self::Drop { piece, to } => {
                let canonical_to = canonical_square(to, player);
                let destination = u8::try_from(canonical_to.index()).ok()?;
                let piece_index = u8::try_from(piece.index()).ok()?;
                PolicyIndex::new(BOARD_POLICY_ACTIONS + destination * 3 + piece_index)
            }
        }
    }

    /// Decodes geometry only. Legality still depends on the game position.
    #[must_use]
    pub fn from_policy_index(index: PolicyIndex, player: Player) -> Option<Self> {
        let raw = index.get();
        if raw < BOARD_POLICY_ACTIONS {
            let from_index = raw / 8;
            let direction = raw % 8;
            let canonical_from = Square::from_index(from_index)?;
            let (row_delta, column_delta) = direction_delta(direction)?;
            let canonical_to = canonical_from.offset(row_delta, column_delta)?;
            Some(Self::Move {
                from: decanonical_square(canonical_from, player),
                to: decanonical_square(canonical_to, player),
            })
        } else {
            let drop_index = raw - BOARD_POLICY_ACTIONS;
            let destination = drop_index / 3;
            if usize::from(destination) >= BOARD_SQUARES {
                return None;
            }
            let piece = match drop_index % 3 {
                0 => HandPiece::Tanuki,
                1 => HandPiece::Kitsune,
                2 => HandPiece::Kodama,
                _ => return None,
            };
            let canonical_to = Square::from_index(destination)?;
            Some(Self::Drop {
                piece,
                to: decanonical_square(canonical_to, player),
            })
        }
    }
}

const fn canonical_square(square: Square, player: Player) -> Square {
    match player {
        Player::First => square,
        Player::Second => square.rotated(),
    }
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

const fn direction_index(row_delta: i16, column_delta: i16) -> Option<u8> {
    let mut direction = 0;
    while direction < DIRECTIONS.len() {
        let (row, column) = DIRECTIONS[direction];
        if row as i16 == row_delta && column as i16 == column_delta {
            return Some(direction as u8);
        }
        direction += 1;
    }
    None
}

const fn direction_delta(direction: u8) -> Option<(i8, i8)> {
    if (direction as usize) < DIRECTIONS.len() {
        Some(DIRECTIONS[direction as usize])
    } else {
        None
    }
}
