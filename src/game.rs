//! Pure rules engine for the official 3×4 game.
//!
//! [`Position`] is the small, copyable board value used heavily by search.
//! [`Game`] wraps it with the path history required for threefold repetition.
//! Keeping those responsibilities separate makes both the rules and MCTS easier
//! to reason about: most move generation needs a board, while repetition needs
//! the exact path that led to it.

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Number of files (columns) on the official board.
pub const BOARD_WIDTH: u8 = 3;
/// Number of ranks (rows) on the official board.
pub const BOARD_HEIGHT: u8 = 4;
/// Total number of addressable board squares.
pub const BOARD_SQUARES: usize = (BOARD_WIDTH as usize) * (BOARD_HEIGHT as usize);
/// Version written into persisted artifacts that depend on these rules.
pub const RULES_VERSION: u16 = 1;

/// The two absolute sides of the game board.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Player {
    /// The player initially displayed at the bottom of the absolute board.
    First,
    /// The player initially displayed at the top of the absolute board.
    Second,
}

impl Player {
    /// Returns the other player.
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::First => Self::Second,
            Self::Second => Self::First,
        }
    }

    /// Returns a stable zero-based index for arrays keyed by player.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::First => 0,
            Self::Second => 1,
        }
    }

    /// Converts relative "forward" movement into an absolute row delta.
    #[must_use]
    pub const fn forward_delta(self) -> i8 {
        match self {
            Self::First => -1,
            Self::Second => 1,
        }
    }

    /// Returns the absolute row on which this player's king or pawn arrives.
    #[must_use]
    pub const fn goal_row(self) -> u8 {
        match self {
            Self::First => 0,
            Self::Second => BOARD_HEIGHT - 1,
        }
    }
}

/// Identity and movement pattern of a piece, independently of its owner.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PieceKind {
    /// King-like piece whose capture or safe arrival wins the game.
    Koropokkuru,
    /// Orthogonal one-step mover.
    Tanuki,
    /// Diagonal one-step mover.
    Kitsune,
    /// Forward-moving pawn that promotes on the far rank.
    Kodama,
    /// Promoted Kodama; it reverts to a Kodama when captured.
    KodamaSamurai,
}

/// A board piece combines a movement kind with an absolute owner.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    /// Movement pattern currently used by the piece.
    pub kind: PieceKind,
    /// Player for whom the piece currently fights.
    pub owner: Player,
}

impl Piece {
    /// Builds a piece from its kind and owner.
    #[must_use]
    pub const fn new(kind: PieceKind, owner: Player) -> Self {
        Self { kind, owner }
    }
}

/// Pieces that can exist in a player's hand.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandPiece {
    /// Captured Tanuki ready to be dropped.
    Tanuki,
    /// Captured Kitsune ready to be dropped.
    Kitsune,
    /// Captured Kodama, including a captured promoted Kodama.
    Kodama,
}

impl HandPiece {
    /// Every hand-piece type in stable array order.
    pub const ALL: [Self; 3] = [Self::Tanuki, Self::Kitsune, Self::Kodama];

    /// Returns the stable array index used by [`Position::hands`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Tanuki => 0,
            Self::Kitsune => 1,
            Self::Kodama => 2,
        }
    }

    /// Returns the unpromoted board kind created by a drop.
    #[must_use]
    pub const fn piece_kind(self) -> PieceKind {
        match self {
            Self::Tanuki => PieceKind::Tanuki,
            Self::Kitsune => PieceKind::Kitsune,
            Self::Kodama => PieceKind::Kodama,
        }
    }

    /// Converts a captured board piece to its hand representation.
    ///
    /// The king is absent because capturing it ends the game instead of adding
    /// it to a hand; a promoted Kodama loses its promotion when captured.
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
    /// Every board square in row-major order.
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

    /// Builds a square from zero-based absolute row and column coordinates.
    ///
    /// Returning [`Option`] makes an off-board coordinate impossible to store.
    #[must_use]
    pub const fn new(row: u8, column: u8) -> Option<Self> {
        if row < BOARD_HEIGHT && column < BOARD_WIDTH {
            Some(Self(row * BOARD_WIDTH + column))
        } else {
            None
        }
    }

    /// Builds a square from its row-major storage index.
    #[must_use]
    pub const fn from_index(index: u8) -> Option<Self> {
        if (index as usize) < BOARD_SQUARES {
            Some(Self(index))
        } else {
            None
        }
    }

    /// Returns the row-major array index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the zero-based absolute row.
    #[must_use]
    pub const fn row(self) -> u8 {
        self.0 / BOARD_WIDTH
    }

    /// Returns the zero-based absolute column.
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

    /// Mirrors a square around the vertical center line (`a` ↔ `c`).
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

/// A complete player decision: either move a board piece or drop a hand piece.
///
/// This domain type is shared by rules, replays, search and the future UI. The
/// neural network uses [`crate::PolicyIndex`] only as a derived representation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Moves the piece on `from` to `to`, possibly capturing or promoting.
    Move {
        /// Source square.
        from: Square,
        /// Destination square.
        to: Square,
    },
    /// Places one captured hand piece on an empty square.
    Drop {
        /// Kind removed from the current player's hand.
        piece: HandPiece,
        /// Empty destination square.
        to: Square,
    },
}

impl Action {
    /// Returns the square affected by either action variant.
    #[must_use]
    pub const fn destination(self) -> Square {
        match self {
            Self::Move { to, .. } | Self::Drop { to, .. } => to,
        }
    }

    /// Mirrors the action around the board's vertical axis.
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

/// Decisive terminal conditions from the official rules.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WinReason {
    /// The opponent's Koropokkuru was captured.
    KoropokkuruCaptured,
    /// A Koropokkuru safely reached its far rank.
    KoropokkuruReachedGoal,
    /// The opponent has no move or drop available.
    OpponentHasNoLegalAction,
}

/// Non-decisive terminal conditions from the official rules.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawReason {
    /// The same full position, including side to move and hands, occurred thrice.
    ThreefoldRepetition,
}

/// Current official status of a game.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Outcome {
    /// Legal play may continue.
    Ongoing,
    /// One player won for the supplied rules reason.
    Win {
        /// Winner in absolute board coordinates.
        player: Player,
        /// Rule that ended the game.
        reason: WinReason,
    },
    /// The game ended without a winner.
    Draw {
        /// Rule that ended the game.
        reason: DrawReason,
    },
}

impl Outcome {
    /// Reports whether no further action may legally be applied.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Ongoing)
    }
}

/// Board, hands and side to move without any path history.
///
/// The type is [`Copy`] on purpose: a 3×4 position is small and MCTS creates
/// many temporary states. Private fields ensure callers cannot bypass material
/// invariants after construction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct Position {
    board: [Option<Piece>; BOARD_SQUARES],
    hands: [[u8; 3]; 2],
    side_to_move: Player,
}

/// Invalid material configurations rejected by [`Position::from_parts`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PositionError {
    /// At least one hand count exceeds the physical material limit.
    #[error("a player cannot have more than two copies of a hand piece")]
    HandCountTooLarge,
    /// One player has multiple king pieces on the board.
    #[error("a position cannot contain more than one Koropokkuru per player")]
    TooManyKoropokkuru,
    /// Board and hand counts describe more pieces than the set contains.
    #[error("a position cannot contain more than eight physical pieces")]
    TooManyPieces,
}

impl Position {
    /// Creates the official initial material with the requested player to move.
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

    /// Borrows the row-major board storage without allowing mutation.
    #[must_use]
    pub const fn board(&self) -> &[Option<Piece>; BOARD_SQUARES] {
        &self.board
    }

    /// Borrows `[player][hand piece]` counts without allowing mutation.
    #[must_use]
    pub const fn hands(&self) -> &[[u8; 3]; 2] {
        &self.hands
    }

    /// Returns the player who must choose the next action.
    #[must_use]
    pub const fn side_to_move(&self) -> Player {
        self.side_to_move
    }

    /// Returns the optional piece occupying a square.
    #[must_use]
    pub const fn piece_at(&self, square: Square) -> Option<Piece> {
        self.board[square.index()]
    }

    /// Returns how many pieces of one kind a player may currently drop.
    #[must_use]
    pub const fn hand_count(&self, player: Player, piece: HandPiece) -> u8 {
        self.hands[player.index()][piece.index()]
    }

    /// Mirrors the absolute board around its vertical axis. Hands, ownership,
    /// and the side to move are unchanged.
    #[must_use]
    pub fn mirrored_horizontally(self) -> Self {
        let mut board = [None; BOARD_SQUARES];
        for square in Square::ALL {
            board[square.mirrored_horizontally().index()] = self.board[square.index()];
        }
        Self {
            board,
            hands: self.hands,
            side_to_move: self.side_to_move,
        }
    }

    /// Counts physical pieces across board and hands.
    ///
    /// Promotion changes a kind, not the number of physical game pieces.
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

    /// Checks an action against movement, ownership, occupancy and king safety.
    #[must_use]
    pub fn is_legal_action(&self, action: Action) -> bool {
        match action {
            Action::Move { from, to } => self.is_legal_board_move(from, to),
            Action::Drop { piece, to } => {
                self.hand_count(self.side_to_move, piece) > 0 && self.piece_at(to).is_none()
            }
        }
    }

    /// Reuses a caller-owned vector to enumerate every legal action.
    ///
    /// This allocation-free form matters inside MCTS, where move generation is
    /// performed thousands of times. The convenience [`Self::legal_actions`]
    /// method is better for ordinary application code.
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

    /// Allocates and returns every legal action in deterministic order.
    #[must_use]
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(32);
        self.legal_actions_into(&mut actions);
        actions
    }

    /// Checks whether at least one legal move or drop exists without allocating.
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

/// Observable effects of one successfully applied action.
///
/// A UI can use this value for messages and animations without reconstructing
/// captures or promotions by comparing two positions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Transition {
    /// Action that was applied.
    pub action: Action,
    /// Player who applied it.
    pub player: Player,
    /// Captured board kind, before a promoted Kodama reverts in hand.
    pub captured: Option<PieceKind>,
    /// Whether this action promoted a Kodama.
    pub promoted: bool,
    /// Official game status after the action.
    pub outcome: Outcome,
}

/// Reasons why [`Game::apply`] can reject an action.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MoveError {
    /// An action was submitted after a terminal outcome.
    #[error("the game is already finished")]
    GameAlreadyFinished,
    /// The action is not legal in the current position.
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
    positions: Vec<Position>,
    repetitions: HashMap<Position, u8>,
}

impl Game {
    /// Creates an official initial game with a deterministic starting player.
    #[must_use]
    pub fn new(starting_player: Player) -> Self {
        Self::from_position(Position::initial(starting_player))
    }

    /// Creates an initial game and samples which player moves first.
    #[must_use]
    pub fn new_random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let starting_player = if rng.random::<bool>() {
            Player::First
        } else {
            Player::Second
        };
        Self::new(starting_player)
    }

    /// Starts a game from an arbitrary valid position with fresh history.
    ///
    /// This is used by MCTS and visited-state restarts. Earlier occurrences are
    /// intentionally absent unless restored through the private replay helper.
    #[must_use]
    pub fn from_position(position: Position) -> Self {
        let mut repetitions = HashMap::with_capacity(32);
        repetitions.insert(position, 1);
        Self {
            position,
            initial_player: position.side_to_move(),
            outcome: Outcome::Ongoing,
            actions: Vec::with_capacity(64),
            positions: vec![position],
            repetitions,
        }
    }

    /// Borrows the current path-independent position.
    #[must_use]
    pub const fn position(&self) -> &Position {
        &self.position
    }

    /// Returns the player who moved first in this game trajectory.
    #[must_use]
    pub const fn initial_player(&self) -> Player {
        self.initial_player
    }

    /// Returns the cached official outcome.
    #[must_use]
    pub const fn outcome(&self) -> Outcome {
        self.outcome
    }

    /// Borrows the actions already played, oldest first.
    #[must_use]
    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    /// Borrows the complete position path, including the initial position.
    #[must_use]
    pub fn position_history(&self) -> &[Position] {
        &self.positions
    }

    /// Returns an in-process fingerprint of the complete path to this position.
    /// It is used to decide whether a search subtree can be reused safely.
    #[must_use]
    pub fn history_fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.positions.hash(&mut hasher);
        hasher.finish()
    }

    /// Returns the number of occurrences of the current full position.
    #[must_use]
    pub fn current_repetition_count(&self) -> u8 {
        self.repetitions.get(&self.position).copied().unwrap_or(0)
    }

    /// Returns how often the position reached by `action` would have occurred.
    ///
    /// Terminal wins return zero because repetition is not consulted after a
    /// decisive result. This deliberately includes the third occurrence for an
    /// action that ends the game by repetition.
    #[must_use]
    pub fn repetition_count_after(&self, action: Action) -> Option<u8> {
        if !self.is_legal_action(action) {
            return None;
        }
        // Applying on a fresh shell avoids cloning the potentially long action,
        // position and repetition histories for every legal policy slot.
        let mut next = Self::from_position(self.position);
        let transition = next.apply(action).ok()?;
        if matches!(transition.outcome, Outcome::Win { .. }) {
            Some(0)
        } else {
            Some(
                self.repetitions
                    .get(next.position())
                    .copied()
                    .unwrap_or(0)
                    .saturating_add(1),
            )
        }
    }

    /// Enumerates official legal actions, or an empty list after game end.
    #[must_use]
    pub fn legal_actions(&self) -> Vec<Action> {
        if self.outcome.is_terminal() {
            Vec::new()
        } else {
            self.position.legal_actions()
        }
    }

    /// Checks both game status and path-independent action legality.
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
        self.positions.push(self.position);

        // Decisive results take precedence over repetition. In particular, a
        // captured king must never be converted into a draw merely because the
        // resulting storage happens to match an earlier position.
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
