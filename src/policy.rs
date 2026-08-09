use serde::{Deserialize, Serialize};

use crate::game::{Action, BOARD_SQUARES, HandPiece, Player, Square};

pub const POLICY_ACTIONS: usize = 132;
const BOARD_POLICY_ACTIONS: u8 = 96;

/// A validated index into the fixed `AlphaZero` policy vector.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PolicyIndex(u8);

impl PolicyIndex {
    #[must_use]
    pub const fn new(index: u8) -> Option<Self> {
        if (index as usize) < POLICY_ACTIONS {
            Some(Self(index))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl Action {
    /// Encodes an action from the current player's canonical perspective.
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

const fn decanonical_square(square: Square, player: Player) -> Square {
    canonical_square(square, player)
}

const fn direction_index(row_delta: i16, column_delta: i16) -> Option<u8> {
    match (row_delta, column_delta) {
        (-1, -1) => Some(0),
        (-1, 0) => Some(1),
        (-1, 1) => Some(2),
        (0, -1) => Some(3),
        (0, 1) => Some(4),
        (1, -1) => Some(5),
        (1, 0) => Some(6),
        (1, 1) => Some(7),
        _ => None,
    }
}

const fn direction_delta(direction: u8) -> Option<(i8, i8)> {
    match direction {
        0 => Some((-1, -1)),
        1 => Some((-1, 0)),
        2 => Some((-1, 1)),
        3 => Some((0, -1)),
        4 => Some((0, 1)),
        5 => Some((1, -1)),
        6 => Some((1, 0)),
        7 => Some((1, 1)),
        _ => None,
    }
}
