//! Human-readable coordinates and actions used by the CLI and the TUI.
//!
//! Notation is deliberately kept outside the rules engine: formatting `c3-c4`
//! is a presentation concern, while [`Action`] remains the shared typed value.
//!
//! Files run from `a` to `e` and ranks from `1` (First's back rank) to `6`.
//! A trailing `+` marks a move that promotes the piece, as in shogi notation.
//!
//! # Examples
//!
//! ```
//! use yokai::Action;
//!
//! let push: Action = "c3-c4".parse().unwrap();
//! assert_eq!(push.to_string(), "c3-c4");
//!
//! let promotion: Action = "c4-c5+".parse().unwrap();
//! assert_eq!(promotion.to_string(), "c4-c5+");
//!
//! let drop: Action = "kodama@a4".parse().unwrap();
//! assert_eq!(drop.to_string(), "kodama@a4");
//! ```

use std::{fmt, str::FromStr};

use thiserror::Error;

use crate::game::{Action, BOARD_HEIGHT, BOARD_WIDTH, HandPiece, Square};

/// Parsing failures for squares, hand pieces and complete actions.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NotationError {
    /// A coordinate is not a file `a`–`e` followed by rank `1`–`6`.
    #[error("invalid square notation: {0}")]
    InvalidSquare(String),
    /// An action is neither `from-to`, `from-to+` nor `piece@to`.
    #[error("invalid action notation: {0}")]
    InvalidAction(String),
    /// A drop names a piece that cannot exist in hand.
    #[error("unknown hand piece: {0}")]
    UnknownHandPiece(String),
}

impl fmt::Display for Square {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = char::from(b'a' + self.column());
        let rank = BOARD_HEIGHT - self.row();
        write!(formatter, "{file}{rank}")
    }
}

impl FromStr for Square {
    type Err = NotationError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let bytes = input.as_bytes();
        if bytes.len() != 2 {
            return Err(NotationError::InvalidSquare(input.to_owned()));
        }

        let file = bytes[0].to_ascii_lowercase();
        let rank = bytes[1];
        if !(b'a'..b'a' + BOARD_WIDTH).contains(&file)
            || !(b'1'..b'1' + BOARD_HEIGHT).contains(&rank)
        {
            return Err(NotationError::InvalidSquare(input.to_owned()));
        }

        let column = file - b'a';
        let row = BOARD_HEIGHT - (rank - b'0');
        Square::new(row, column).ok_or_else(|| NotationError::InvalidSquare(input.to_owned()))
    }
}

impl fmt::Display for HandPiece {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Kirin => "kirin",
            Self::Oni => "oni",
            Self::Kodama => "kodama",
        };
        formatter.write_str(name)
    }
}

impl FromStr for HandPiece {
    type Err = NotationError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        if input.eq_ignore_ascii_case("kirin") {
            Ok(Self::Kirin)
        } else if input.eq_ignore_ascii_case("oni") {
            Ok(Self::Oni)
        } else if input.eq_ignore_ascii_case("kodama") {
            Ok(Self::Kodama)
        } else {
            Err(NotationError::UnknownHandPiece(input.to_owned()))
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Move { from, to, promote } => {
                let suffix = if *promote { "+" } else { "" };
                write!(formatter, "{from}-{to}{suffix}")
            }
            Self::Drop { piece, to } => write!(formatter, "{piece}@{to}"),
        }
    }
}

impl FromStr for Action {
    type Err = NotationError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        if let Some((piece, to)) = input.split_once('@') {
            return Ok(Self::Drop {
                piece: piece.parse()?,
                to: to.parse()?,
            });
        }
        if let Some((from, to)) = input.split_once('-') {
            let (to, promote) = match to.strip_suffix('+') {
                Some(to) => (to, true),
                None => (to, false),
            };
            return Ok(Self::Move {
                from: from.parse()?,
                to: to.parse()?,
                promote,
            });
        }
        Err(NotationError::InvalidAction(input.to_owned()))
    }
}
