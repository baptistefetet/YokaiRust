//! Human-readable coordinates and actions used by the CLI and future TUI.
//!
//! Notation is deliberately kept outside the rules engine: formatting `b2-b3`
//! is a presentation concern, while [`Action`] remains the shared typed value.

use std::{fmt, str::FromStr};

use thiserror::Error;

use crate::game::{Action, BOARD_HEIGHT, BOARD_WIDTH, HandPiece, Square};

/// Parsing failures for squares, hand pieces and complete actions.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NotationError {
    /// A coordinate is not a file `a`–`c` followed by rank `1`–`4`.
    #[error("invalid square notation: {0}")]
    InvalidSquare(String),
    /// An action is neither `from-to` nor `piece@to`.
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
            Self::Tanuki => "tanuki",
            Self::Kitsune => "kitsune",
            Self::Kodama => "kodama",
        };
        formatter.write_str(name)
    }
}

impl FromStr for HandPiece {
    type Err = NotationError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        if input.eq_ignore_ascii_case("tanuki") {
            Ok(Self::Tanuki)
        } else if input.eq_ignore_ascii_case("kitsune") {
            Ok(Self::Kitsune)
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
            Self::Move { from, to } => write!(formatter, "{from}-{to}"),
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
            return Ok(Self::Move {
                from: from.parse()?,
                to: to.parse()?,
            });
        }
        Err(NotationError::InvalidAction(input.to_owned()))
    }
}
