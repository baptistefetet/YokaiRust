use std::collections::HashMap;

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const BOARD_WIDTH: u8 = 3;
pub const BOARD_HEIGHT: u8 = 4;
pub const BOARD_SQUARES: usize = (BOARD_WIDTH as usize) * (BOARD_HEIGHT as usize);
pub const RULES_VERSION: u16 = 1;

/// The two absolute sides of the game board.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Player {
    First,
    Second,
}

impl Player {
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::First,
        }
    }

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
        }
    }

    #[must_use]
    pub const fn forward_delta(self) -> i8 {
        match self {
            Self::First => -1,
            Self::Second => 1,
        }
    }

    #[must_use]
    pub const fn goal_row(self) -> u8 {
        match self {
            Self::First => 0,
            Self::Second => BOARD_HEIGHT - 1,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PieceKind {
    Koropokkuru,
    Tanuki,
    Kitsune,
    Kodama,
    KodamaSamurai,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    pub kind: PieceKind,
    pub owner: Player,
}

impl Piece {
    #[must_use]
    pub const fn new(kind: PieceKind, owner: Player) -> Self {
        Self { kind, owner }
    }
}

/// Pieces that can exist in a player's hand.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandPiece {
    Tanuki,
    Kitsune,
    Kodama,
}

impl HandPiece {
    pub const ALL: [Self; 3] = [Self::Tanuki, Self::Kitsune, Self::Kodama];

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Tanuki => 0,
            Self::Kitsune => 1,
            Self::Kodama => 2,
        }
    }

    #[must_use]
    pub const fn piece_kind(self) -> PieceKind {
        match self {
            Self::Tanuki => PieceKind::Tanuki,
            Self::Kitsune => PieceKind::Kitsune,
            Self::Kodama => PieceKind::Kodama,
        }
    }

    #[must_use]
    pub const fn from_captured(kind: PieceKind) -> Option<Self> {
        match kind {
            PieceKind::Tanuki => Some(Self::Tanuki),
            PieceKind::Kitsune => Some(Self::Kitsune),
            PieceKind::Kodama | PieceKind::KodamaSamurai => Some(Self::Kodama),
            PieceKind::Koropokkuru => None,
        }
    }
}

/// A compact board coordinate, stored as a row-major index in `[0, 11]`.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Square(u8);

impl Square {
    pub const ALL: [Self; BOARD_SQUARES] = [
        Self(0),
        Self(1),
        Self(2),
        Self(3),
        Self(4),
        Self(5),
        Self(6),
        Self(7),
        Self(8),
        Self(9),
        Self(10),
        Self(11),
    ];

    #[must_use]
    pub const fn new(row: u8, column: u8) -> Option<Self> {
        if row < BOARD_HEIGHT && column < BOARD_WIDTH {
            Some(Self(row * BOARD_WIDTH + column))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn from_index(index: u8) -> Option<Self> {
        if (index as usize) < BOARD_SQUARES {
            Some(Self(index))
        } else {
            None
        }
    }

    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[must_use]
    pub const fn row(self) -> u8 {
        self.0 / BOARD_WIDTH
    }

    #[must_use]
    pub const fn column(self) -> u8 {
        self.0 % BOARD_WIDTH
    }

    /// Rotates the square by 180 degrees. This is the canonical transform for
    /// positions where Second is to move.
    #[must_use]
    pub const fn rotated(self) -> Self {
        Self((BOARD_WIDTH * BOARD_HEIGHT - 1) - self.0)
    }

    #[must_use]
    pub const fn mirrored_horizontally(self) -> Self {
        Self(self.row() * BOARD_WIDTH + (BOARD_WIDTH - 1 - self.column()))
    }

    #[must_use]
    pub(crate) fn offset(self, row_delta: i8, column_delta: i8) -> Option<Self> {
        let row = i16::from(self.row()) + i16::from(row_delta);
        let column = i16::from(self.column()) + i16::from(column_delta);
        if row < 0 || column < 0 {
            return None;
        }
        Self::new(u8::try_from(row).ok()?, u8::try_from(column).ok()?)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    Move { from: Square, to: Square },
    Drop { piece: HandPiece, to: Square },
}

impl Action {
    #[must_use]
    pub const fn destination(self) -> Square {
        match self {
            Self::Move { to, .. } | Self::Drop { to, .. } => to,
        }
    }

    #[must_use]
    pub const fn mirrored_horizontally(self) -> Self {
        match self {
            Self::Move { from, to } => Self::Move {
                from: from.mirrored_horizontally(),
                to: to.mirrored_horizontally(),
            },
            Self::Drop { piece, to } => Self::Drop {
                piece,
                to: to.mirrored_horizontally(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WinReason {
    KoropokkuruCaptured,
    KoropokkuruReachedGoal,
    OpponentHasNoLegalAction,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawReason {
    ThreefoldRepetition,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Outcome {
    Ongoing,
    Win { player: Player, reason: WinReason },
    Draw { reason: DrawReason },
}

impl Outcome {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Ongoing)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Position {
    board: [Option<Piece>; BOARD_SQUARES],
    hands: [[u8; 3]; 2],
    side_to_move: Player,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PositionError {
    #[error("a player cannot have more than two copies of a hand piece")]
    HandCountTooLarge,
    #[error("a position cannot contain more than one Koropokkuru per player")]
    TooManyKoropokkuru,
    #[error("a position cannot contain more than eight physical pieces")]
    TooManyPieces,
}

impl Position {
    #[must_use]
    pub fn initial(side_to_move: Player) -> Self {
        let board = [
            Some(Piece::new(PieceKind::Tanuki, Player::Second)),
            Some(Piece::new(PieceKind::Koropokkuru, Player::Second)),
            Some(Piece::new(PieceKind::Kitsune, Player::Second)),
            None,
            Some(Piece::new(PieceKind::Kodama, Player::Second)),
            None,
            None,
            Some(Piece::new(PieceKind::Kodama, Player::First)),
            None,
            Some(Piece::new(PieceKind::Kitsune, Player::First)),
            Some(Piece::new(PieceKind::Koropokkuru, Player::First)),
            Some(Piece::new(PieceKind::Tanuki, Player::First)),
        ];

        Self {
            board,
            hands: [[0; 3]; 2],
            side_to_move,
        }
    }

    /// Creates a position from explicit board and hand storage.
    ///
    /// # Errors
    ///
    /// Returns a [`PositionError`] when basic material bounds are violated.
    pub fn from_parts(
        board: [Option<Piece>; BOARD_SQUARES],
        hands: [[u8; 3]; 2],
        side_to_move: Player,
    ) -> Result<Self, PositionError> {
        if hands.iter().flatten().any(|&count| count > 2) {
            return Err(PositionError::HandCountTooLarge);
        }

        for player in [Player::First, Player::Second] {
            let king_count = board
                .iter()
                .flatten()
                .filter(|piece| piece.owner == player && piece.kind == PieceKind::Koropokkuru)
                .count();
            if king_count > 1 {
                return Err(PositionError::TooManyKoropokkuru);
            }
        }

        let board_count = board.iter().flatten().count();
        let hand_count: usize = hands
            .iter()
            .flatten()
            .map(|&count| usize::from(count))
            .sum();
        if board_count + hand_count > 8 {
            return Err(PositionError::TooManyPieces);
        }

        Ok(Self {
            board,
            hands,
            side_to_move,
        })
    }

    #[must_use]
    pub const fn board(&self) -> &[Option<Piece>; BOARD_SQUARES] {
        &self.board
    }

    #[must_use]
    pub const fn hands(&self) -> &[[u8; 3]; 2] {
        &self.hands
    }

    #[must_use]
    pub const fn side_to_move(&self) -> Player {
        self.side_to_move
    }

    #[must_use]
    pub const fn piece_at(&self, square: Square) -> Option<Piece> {
        self.board[square.index()]
    }

    #[must_use]
    pub const fn hand_count(&self, player: Player, piece: HandPiece) -> u8 {
        self.hands[player.index()][piece.index()]
    }

    #[must_use]
    pub fn physical_piece_count(&self) -> usize {
        self.board.iter().flatten().count()
            + self
                .hands
                .iter()
                .flatten()
                .map(|&count| usize::from(count))
                .sum::<usize>()
    }

    #[must_use]
    pub fn is_legal_action(&self, action: Action) -> bool {
        match action {
            Action::Move { from, to } => self.is_legal_board_move(from, to),
            Action::Drop { piece, to } => {
                self.hand_count(self.side_to_move, piece) > 0 && self.piece_at(to).is_none()
            }
        }
    }

    pub fn legal_actions_into(&self, actions: &mut Vec<Action>) {
        actions.clear();
        actions.reserve(32);

        for from in Square::ALL {
            let Some(piece) = self.piece_at(from) else {
                continue;
            };
            if piece.owner != self.side_to_move {
                continue;
            }

            for &(relative_row, relative_column) in movement_offsets(piece.kind) {
                let row_delta = relative_row * piece.owner.forward_delta();
                let Some(to) = from.offset(row_delta, relative_column) else {
                    continue;
                };
                let action = Action::Move { from, to };
                if self.is_legal_action(action) {
                    actions.push(action);
                }
            }
        }

        for hand_piece in HandPiece::ALL {
            if self.hand_count(self.side_to_move, hand_piece) == 0 {
                continue;
            }
            for to in Square::ALL {
                if self.piece_at(to).is_none() {
                    actions.push(Action::Drop {
                        piece: hand_piece,
                        to,
                    });
                }
            }
        }
    }

    #[must_use]
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(32);
        self.legal_actions_into(&mut actions);
        actions
    }

    #[must_use]
    pub fn has_legal_action(&self) -> bool {
        for from in Square::ALL {
            let Some(piece) = self.piece_at(from) else {
                continue;
            };
            if piece.owner != self.side_to_move {
                continue;
            }
            for &(relative_row, relative_column) in movement_offsets(piece.kind) {
                let row_delta = relative_row * piece.owner.forward_delta();
                if let Some(to) = from.offset(row_delta, relative_column)
                    && self.is_legal_board_move(from, to)
                {
                    return true;
                }
            }
        }

        HandPiece::ALL.iter().any(|&piece| {
            self.hand_count(self.side_to_move, piece) > 0 && self.board.iter().any(Option::is_none)
        })
    }

    fn is_legal_board_move(&self, from: Square, to: Square) -> bool {
        let Some(piece) = self.piece_at(from) else {
            return false;
        };
        if piece.owner != self.side_to_move {
            return false;
        }
        if self
            .piece_at(to)
            .is_some_and(|target| target.owner == self.side_to_move)
        {
            return false;
        }

        let absolute_row = i16::from(to.row()) - i16::from(from.row());
        let column = i16::from(to.column()) - i16::from(from.column());
        let relative_row = absolute_row * i16::from(piece.owner.forward_delta());
        let movement_matches = movement_offsets(piece.kind)
            .iter()
            .any(|&(row, col)| i16::from(row) == relative_row && i16::from(col) == column);
        if !movement_matches {
            return false;
        }

        if piece.kind == PieceKind::Koropokkuru {
            let mut board_after_move = self.board;
            board_after_move[from.index()] = None;
            board_after_move[to.index()] = Some(piece);
            if square_is_attacked(&board_after_move, to, piece.owner.opponent()) {
                return false;
            }
        }

        true
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Transition {
    pub action: Action,
    pub player: Player,
    pub captured: Option<PieceKind>,
    pub promoted: bool,
    pub outcome: Outcome,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MoveError {
    #[error("the game is already finished")]
    GameAlreadyFinished,
    #[error("illegal action: {0:?}")]
    IllegalAction(Action),
}

/// A game adds history-dependent rules (currently repetition) around a compact
/// `Position`. `Position` stays cheap to copy for future tree search.
#[derive(Clone, Debug)]
pub struct Game {
    position: Position,
    initial_player: Player,
    outcome: Outcome,
    actions: Vec<Action>,
    repetitions: HashMap<Position, u8>,
}

impl Game {
    #[must_use]
    pub fn new(starting_player: Player) -> Self {
        Self::from_position(Position::initial(starting_player))
    }

    #[must_use]
    pub fn new_random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let starting_player = if rng.random::<bool>() {
            Player::First
        } else {
            Player::Second
        };
        Self::new(starting_player)
    }

    #[must_use]
    pub fn from_position(position: Position) -> Self {
        let mut repetitions = HashMap::with_capacity(32);
        repetitions.insert(position, 1);
        Self {
            position,
            initial_player: position.side_to_move(),
            outcome: Outcome::Ongoing,
            actions: Vec::with_capacity(64),
            repetitions,
        }
    }

    #[must_use]
    pub const fn position(&self) -> &Position {
        &self.position
    }

    #[must_use]
    pub const fn initial_player(&self) -> Player {
        self.initial_player
    }

    #[must_use]
    pub const fn outcome(&self) -> Outcome {
        self.outcome
    }

    #[must_use]
    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    #[must_use]
    pub fn current_repetition_count(&self) -> u8 {
        self.repetitions.get(&self.position).copied().unwrap_or(0)
    }

    #[must_use]
    pub fn legal_actions(&self) -> Vec<Action> {
        if self.outcome.is_terminal() {
            Vec::new()
        } else {
            self.position.legal_actions()
        }
    }

    #[must_use]
    pub fn is_legal_action(&self, action: Action) -> bool {
        !self.outcome.is_terminal() && self.position.is_legal_action(action)
    }

    /// Applies one legal action and updates terminal and repetition state.
    ///
    /// # Errors
    ///
    /// Returns [`MoveError::GameAlreadyFinished`] after a terminal outcome, or
    /// [`MoveError::IllegalAction`] when the action is not legal in this position.
    pub fn apply(&mut self, action: Action) -> Result<Transition, MoveError> {
        if self.outcome.is_terminal() {
            return Err(MoveError::GameAlreadyFinished);
        }
        if !self.position.is_legal_action(action) {
            return Err(MoveError::IllegalAction(action));
        }

        let player = self.position.side_to_move;
        let mut captured = None;
        let mut promoted = false;
        let mut moved_kind = None;

        match action {
            Action::Move { from, to } => {
                let Some(mut piece) = self.position.board[from.index()] else {
                    return Err(MoveError::IllegalAction(action));
                };
                moved_kind = Some(piece.kind);
                if let Some(target) = self.position.board[to.index()] {
                    captured = Some(target.kind);
                    if let Some(hand_piece) = HandPiece::from_captured(target.kind) {
                        self.position.hands[player.index()][hand_piece.index()] += 1;
                    }
                }

                self.position.board[from.index()] = None;
                if piece.kind == PieceKind::Kodama && to.row() == player.goal_row() {
                    piece.kind = PieceKind::KodamaSamurai;
                    promoted = true;
                }
                self.position.board[to.index()] = Some(piece);
            }
            Action::Drop { piece, to } => {
                self.position.hands[player.index()][piece.index()] -= 1;
                self.position.board[to.index()] = Some(Piece::new(piece.piece_kind(), player));
            }
        }

        self.position.side_to_move = player.opponent();
        self.actions.push(action);

        self.outcome = if captured == Some(PieceKind::Koropokkuru) {
            Outcome::Win {
                player,
                reason: WinReason::KoropokkuruCaptured,
            }
        } else if moved_kind == Some(PieceKind::Koropokkuru)
            && action.destination().row() == player.goal_row()
            && !square_is_attacked(
                &self.position.board,
                action.destination(),
                player.opponent(),
            )
        {
            Outcome::Win {
                player,
                reason: WinReason::KoropokkuruReachedGoal,
            }
        } else if !self.position.has_legal_action() {
            Outcome::Win {
                player,
                reason: WinReason::OpponentHasNoLegalAction,
            }
        } else {
            let occurrence = self.repetitions.entry(self.position).or_insert(0);
            *occurrence += 1;
            if *occurrence >= 3 {
                Outcome::Draw {
                    reason: DrawReason::ThreefoldRepetition,
                }
            } else {
                Outcome::Ongoing
            }
        };

        Ok(Transition {
            action,
            player,
            captured,
            promoted,
            outcome: self.outcome,
        })
    }
}

const KROPOKKURU_MOVES: &[(i8, i8)] = &[
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];
const TANUKI_MOVES: &[(i8, i8)] = &[(-1, 0), (0, -1), (0, 1), (1, 0)];
const KITSUNE_MOVES: &[(i8, i8)] = &[(-1, -1), (-1, 1), (1, -1), (1, 1)];
// Relative row +1 always means "forward". `forward_delta` maps it to the
// absolute board direction of the piece owner.
const KODAMA_MOVES: &[(i8, i8)] = &[(1, 0)];
const KODAMA_SAMURAI_MOVES: &[(i8, i8)] = &[(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, 0)];

fn movement_offsets(kind: PieceKind) -> &'static [(i8, i8)] {
    match kind {
        PieceKind::Koropokkuru => KROPOKKURU_MOVES,
        PieceKind::Tanuki => TANUKI_MOVES,
        PieceKind::Kitsune => KITSUNE_MOVES,
        PieceKind::Kodama => KODAMA_MOVES,
        PieceKind::KodamaSamurai => KODAMA_SAMURAI_MOVES,
    }
}

fn square_is_attacked(
    board: &[Option<Piece>; BOARD_SQUARES],
    target: Square,
    attacker: Player,
) -> bool {
    Square::ALL.iter().copied().any(|from| {
        let Some(piece) = board[from.index()] else {
            return false;
        };
        if piece.owner != attacker {
            return false;
        }

        let absolute_row = i16::from(target.row()) - i16::from(from.row());
        let column = i16::from(target.column()) - i16::from(from.column());
        let relative_row = absolute_row * i16::from(piece.owner.forward_delta());
        movement_offsets(piece.kind)
            .iter()
            .any(|&(row, col)| i16::from(row) == relative_row && i16::from(col) == column)
    })
}
