//! Pure rules engine for the 5×6 expert game.
//!
//! [`Position`] is the small, copyable board value used heavily by search.
//! [`Game`] wraps it with the path history required for threefold repetition.
//! Keeping those responsibilities separate makes both the rules and MCTS easier
//! to reason about: most move generation needs a board, while repetition needs
//! the exact path that led to it.
//!
//! Every piece moves a single step, so no piece can be pinned and no attack
//! can be blocked. That keeps the king-safety and drop-checkmate tests short.
//! `docs/rules.md` lists the rules and the few points where this engine
//! interprets the printed rulebook.

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Number of files (columns) on the official board.
pub const BOARD_WIDTH: u8 = 5;
/// Number of ranks (rows) on the official board.
pub const BOARD_HEIGHT: u8 = 6;
/// Total number of addressable board squares.
pub const BOARD_SQUARES: usize = (BOARD_WIDTH as usize) * (BOARD_HEIGHT as usize);
/// Number of far ranks in which a player's Kodama and Oni may promote.
pub const PROMOTION_ZONE_ROWS: u8 = 2;
/// Number of physical pieces in the complete set, kings included.
pub const PHYSICAL_PIECES: usize = 16;
/// Version written into persisted artifacts that depend on these rules.
pub const RULES_VERSION: u16 = 2;

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

    /// Returns the absolute row farthest from this player's starting camp.
    #[must_use]
    pub const fn last_row(self) -> u8 {
        match self {
            Self::First => 0,
            Self::Second => BOARD_HEIGHT - 1,
        }
    }

    /// Reports whether `square` lies in the zone where this player promotes:
    /// the [`PROMOTION_ZONE_ROWS`] ranks nearest the opponent's camp.
    #[must_use]
    pub const fn promotion_zone_contains(self, square: Square) -> bool {
        match self {
            Self::First => square.row() < PROMOTION_ZONE_ROWS,
            Self::Second => square.row() >= BOARD_HEIGHT - PROMOTION_ZONE_ROWS,
        }
    }
}

/// Identity and movement pattern of a piece, independently of its owner.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PieceKind {
    /// King-like piece whose capture wins the game.
    Koropokkuru,
    /// Steps in every direction except the two backward diagonals.
    Kirin,
    /// Steps to the three forward squares and the two backward diagonals.
    Oni,
    /// Promoted Oni; it moves like a Kirin and reverts to an Oni when captured.
    SuperOni,
    /// Forward-moving pawn that may promote in the promotion zone.
    Kodama,
    /// Promoted Kodama; it moves like a Kirin and reverts to a Kodama when
    /// captured.
    KodamaSamurai,
}

impl PieceKind {
    /// Returns the kind shown on the other face of the piece, when it has one.
    ///
    /// Only the Oni and the Kodama can be flipped; promotion is never undone
    /// on the board.
    #[must_use]
    pub const fn promoted(self) -> Option<Self> {
        match self {
            Self::Oni => Some(Self::SuperOni),
            Self::Kodama => Some(Self::KodamaSamurai),
            Self::Koropokkuru | Self::Kirin | Self::SuperOni | Self::KodamaSamurai => None,
        }
    }
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
    /// Captured Kirin ready to be dropped.
    Kirin,
    /// Captured Oni, including a captured Super Oni.
    Oni,
    /// Captured Kodama, including a captured Kodama Samurai.
    Kodama,
}

impl HandPiece {
    /// Every hand-piece type in stable array order.
    pub const ALL: [Self; 3] = [Self::Kirin, Self::Oni, Self::Kodama];

    /// Returns the stable array index used by [`Position::hands`].
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Kirin => 0,
            Self::Oni => 1,
            Self::Kodama => 2,
        }
    }

    /// Returns the unpromoted board kind created by a drop.
    #[must_use]
    pub const fn piece_kind(self) -> PieceKind {
        match self {
            Self::Kirin => PieceKind::Kirin,
            Self::Oni => PieceKind::Oni,
            Self::Kodama => PieceKind::Kodama,
        }
    }

    /// Returns how many physical copies of this piece the complete set holds.
    ///
    /// Both players start with half of them. Because captured pieces change
    /// sides, one hand can eventually hold every copy.
    #[must_use]
    pub const fn copies(self) -> u8 {
        match self {
            Self::Kirin | Self::Oni => 4,
            Self::Kodama => 6,
        }
    }

    /// Converts a captured board piece to its hand representation.
    ///
    /// The king is absent because capturing it ends the game instead of adding
    /// it to a hand; a promoted piece loses its promotion when captured.
    #[must_use]
    pub const fn from_captured(kind: PieceKind) -> Option<Self> {
        match kind {
            PieceKind::Kirin => Some(Self::Kirin),
            PieceKind::Oni | PieceKind::SuperOni => Some(Self::Oni),
            PieceKind::Kodama | PieceKind::KodamaSamurai => Some(Self::Kodama),
            PieceKind::Koropokkuru => None,
        }
    }
}

/// A compact board coordinate, stored as a row-major index in `[0, 29]`.
/// Row 0 is rank 6 at the top; row 5 is rank 1 at the bottom.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct Square(u8);

impl<'de> Deserialize<'de> for Square {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let index = u8::deserialize(deserializer)?;
        Self::from_index(index).ok_or_else(|| {
            serde::de::Error::custom(format!("square index must be in 0..{BOARD_SQUARES}"))
        })
    }
}

impl Square {
    /// Every board square in row-major order.
    pub const ALL: [Self; BOARD_SQUARES] = {
        let mut squares = [Self(0); BOARD_SQUARES];
        let mut index = 0;
        while index < BOARD_SQUARES {
            squares[index] = Self(index as u8);
            index += 1;
        }
        squares
    };

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

    /// Mirrors a square around the vertical center line (`a` ↔ `e`).
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
/// This domain type is shared by rules, replays, search and the interfaces. The
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
        /// Whether the piece is flipped to its promoted face by this move.
        ///
        /// Promotion is the mover's choice, so the same step can exist as two
        /// distinct actions. The rules reject the flag where no promotion is
        /// available and require it where promotion is mandatory.
        #[serde(default)]
        promote: bool,
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
            Self::Move { from, to, promote } => Self::Move {
                from: from.mirrored_horizontally(),
                to: to.mirrored_horizontally(),
                promote,
            },
            Self::Drop { piece, to } => Self::Drop {
                piece,
                to: to.mirrored_horizontally(),
            },
        }
    }
}

/// Decisive terminal conditions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WinReason {
    /// The opponent's Koropokkuru was captured.
    KoropokkuruCaptured,
    /// The opponent has no move or drop available. Players may not skip a
    /// turn, so being unable to act loses the game.
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
    /// Returns the official result from `player`'s perspective: a win is `1`,
    /// a loss is `-1`, and a draw or ongoing game is `0`.
    /// Search-specific draw preferences do not change this value.
    #[must_use]
    pub fn value_for(self, player: Player) -> f32 {
        match self {
            Self::Win { player: winner, .. } if winner == player => 1.0,
            Self::Win { .. } => -1.0,
            Self::Ongoing | Self::Draw { .. } => 0.0,
        }
    }

    /// Reports whether no further action may legally be applied.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        !matches!(self, Self::Ongoing)
    }
}

/// Board, hands and side to move without any path history.
///
/// The type is [`Copy`] on purpose: a 5×6 position is small and MCTS creates
/// many temporary states. Private fields ensure callers cannot bypass material
/// invariants after construction.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct Position {
    board: [Option<Piece>; BOARD_SQUARES],
    hands: [[u8; 3]; 2],
    side_to_move: Player,
}

impl<'de> Deserialize<'de> for Position {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Decode storage first, then use the same material checks as Rust callers.
        #[derive(Deserialize)]
        struct Parts {
            board: [Option<Piece>; BOARD_SQUARES],
            hands: [[u8; 3]; 2],
            side_to_move: Player,
        }
        let parts = Parts::deserialize(deserializer)?;
        Self::from_parts(parts.board, parts.hands, parts.side_to_move)
            .map_err(serde::de::Error::custom)
    }
}

/// Invalid configurations rejected by [`Position::from_parts`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum PositionError {
    /// One player has multiple king pieces on the board.
    #[error("a position cannot contain more than one Koropokkuru per player")]
    TooManyKoropokkuru,
    /// Board and hand counts describe more pieces than the set contains.
    #[error("a position cannot contain more than sixteen physical pieces")]
    TooManyPieces,
    /// Board and both hands contain more copies of one physical kind than the
    /// set holds. A promoted piece counts as its unpromoted kind because
    /// promotion does not add a piece.
    #[error("a position cannot contain that many physical {0:?} pieces")]
    TooManyPiecesOfKind(HandPiece),
    /// Two unpromoted Kodama of one player stand in the same column.
    #[error("two unpromoted Kodama of one player cannot share a column")]
    KodamaShareColumn,
    /// An unpromoted Kodama stands on its owner's last row, where it could
    /// never move again.
    #[error("an unpromoted Kodama cannot stand on its owner's last row")]
    KodamaOnLastRow,
}

/// Promotion flags that one geometrically valid board move may carry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PromotionChoice {
    /// The piece has no other face, or the move stays outside the zone.
    Unavailable,
    /// The mover decides whether to flip the piece.
    Optional,
    /// An unpromoted piece could never move again from the destination.
    Mandatory,
}

impl PromotionChoice {
    fn of(piece: Piece, from: Square, to: Square) -> Self {
        let owner = piece.owner;
        // Entering the zone, leaving it and moving inside it all qualify.
        let touches_zone = owner.promotion_zone_contains(from) || owner.promotion_zone_contains(to);
        if piece.kind.promoted().is_none() || !touches_zone {
            Self::Unavailable
        } else if piece.kind == PieceKind::Kodama && to.row() == owner.last_row() {
            Self::Mandatory
        } else {
            Self::Optional
        }
    }

    const fn allows(self, promote: bool) -> bool {
        match self {
            Self::Unavailable => !promote,
            Self::Optional => true,
            Self::Mandatory => promote,
        }
    }
}

impl Position {
    /// Creates the official initial material with the requested player to move.
    ///
    /// Each camp lines up Oni, Kirin, Koropokkuru, Kirin, Oni on its back
    /// rank, with three Kodama on the central files of its third rank. The two
    /// Kodama lines therefore start face to face.
    #[must_use]
    pub fn initial(side_to_move: Player) -> Self {
        const BACK_RANK: [PieceKind; BOARD_WIDTH as usize] = [
            PieceKind::Oni,
            PieceKind::Kirin,
            PieceKind::Koropokkuru,
            PieceKind::Kirin,
            PieceKind::Oni,
        ];
        let mut board = [None; BOARD_SQUARES];
        for (player, back_row, kodama_row) in [
            (Player::Second, 0, 2),
            (Player::First, BOARD_HEIGHT - 1, BOARD_HEIGHT - 3),
        ] {
            for (column, kind) in (0..BOARD_WIDTH).zip(BACK_RANK) {
                board[usize::from(back_row * BOARD_WIDTH + column)] =
                    Some(Piece::new(kind, player));
            }
            for column in 1..BOARD_WIDTH - 1 {
                board[usize::from(kodama_row * BOARD_WIDTH + column)] =
                    Some(Piece::new(PieceKind::Kodama, player));
            }
        }

        Self {
            board,
            hands: [[0; 3]; 2],
            side_to_move,
        }
    }

    /// Creates a position from explicit board and hand storage.
    ///
    /// Material may be absent for a partial position, but each physical kind
    /// is limited to the official set across the board and both players' hands
    /// (see [`HandPiece::copies`]). A promoted piece shares the limit of its
    /// unpromoted kind. Kodama placements that legal play can never produce
    /// are rejected as well.
    ///
    /// # Errors
    ///
    /// Returns a [`PositionError`] when these bounds are violated.
    pub fn from_parts(
        board: [Option<Piece>; BOARD_SQUARES],
        hands: [[u8; 3]; 2],
        side_to_move: Player,
    ) -> Result<Self, PositionError> {
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

        let mut piece_counts = [0_usize; 3];
        for piece in board.iter().flatten() {
            if let Some(hand_piece) = HandPiece::from_captured(piece.kind) {
                piece_counts[hand_piece.index()] += 1;
            }
        }
        for hand_piece in HandPiece::ALL {
            let index = hand_piece.index();
            let count =
                piece_counts[index] + usize::from(hands[0][index]) + usize::from(hands[1][index]);
            if count > usize::from(hand_piece.copies()) {
                return Err(PositionError::TooManyPiecesOfKind(hand_piece));
            }
        }

        let board_count = board.iter().flatten().count();
        let hand_count: usize = hands
            .iter()
            .flatten()
            .map(|&count| usize::from(count))
            .sum();
        if board_count + hand_count > PHYSICAL_PIECES {
            return Err(PositionError::TooManyPieces);
        }

        let mut kodama_columns = [[false; BOARD_WIDTH as usize]; 2];
        for square in Square::ALL {
            let Some(piece) = board[square.index()] else {
                continue;
            };
            if piece.kind != PieceKind::Kodama {
                continue;
            }
            if square.row() == piece.owner.last_row() {
                return Err(PositionError::KodamaOnLastRow);
            }
            let occupied = &mut kodama_columns[piece.owner.index()][usize::from(square.column())];
            if *occupied {
                return Err(PositionError::KodamaShareColumn);
            }
            *occupied = true;
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

    /// Checks an action against movement, ownership, occupancy, promotion,
    /// king safety and the Kodama drop restrictions.
    #[must_use]
    pub fn is_legal_action(&self, action: Action) -> bool {
        match action {
            Action::Move { from, to, promote } => self
                .movable_piece(from, to)
                .is_some_and(|piece| PromotionChoice::of(piece, from, to).allows(promote)),
            Action::Drop { piece, to } => self.is_legal_drop(piece, to),
        }
    }

    /// Reuses a caller-owned vector to enumerate every legal action.
    ///
    /// This allocation-free form matters inside MCTS, where move generation is
    /// performed thousands of times. The convenience [`Self::legal_actions`]
    /// method is better for ordinary application code.
    pub fn legal_actions_into(&self, actions: &mut Vec<Action>) {
        actions.clear();
        actions.reserve(64);
        actions.extend(self.legal_board_moves());
        actions.extend(self.legal_drops());
    }

    /// Allocates and returns every legal action in deterministic order.
    #[must_use]
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(64);
        self.legal_actions_into(&mut actions);
        actions
    }

    /// Checks whether at least one legal move or drop exists without allocating.
    #[must_use]
    pub fn has_legal_action(&self) -> bool {
        self.legal_steps().next().is_some() || self.legal_drops().next().is_some()
    }

    // A "step" is a board move without its promotion flag. This lazy traversal
    // supports complete enumeration, early exit and the drop-checkmate test.
    fn legal_steps(&self) -> impl Iterator<Item = (Piece, Square, Square)> + '_ {
        Square::ALL
            .into_iter()
            .filter_map(move |from| {
                let piece = self.piece_at(from)?;
                (piece.owner == self.side_to_move).then_some((from, piece))
            })
            .flat_map(move |(from, piece)| {
                movement_offsets(piece.kind)
                    .iter()
                    .filter_map(move |&(row, column)| {
                        let to = from.offset(row * piece.owner.forward_delta(), column)?;
                        self.movable_piece(from, to).map(|piece| (piece, from, to))
                    })
            })
    }

    // An optional promotion yields two actions for one step: the plain move
    // first, then the promoting one.
    fn legal_board_moves(&self) -> impl Iterator<Item = Action> + '_ {
        self.legal_steps().flat_map(|(piece, from, to)| {
            let choice = PromotionChoice::of(piece, from, to);
            [false, true]
                .into_iter()
                .filter(move |&promote| choice.allows(promote))
                .map(move |promote| Action::Move { from, to, promote })
        })
    }

    fn legal_drops(&self) -> impl Iterator<Item = Action> + '_ {
        HandPiece::ALL
            .into_iter()
            .filter(|&piece| self.hand_count(self.side_to_move, piece) > 0)
            .flat_map(move |piece| {
                Square::ALL
                    .into_iter()
                    .filter(move |&to| self.is_legal_drop(piece, to))
                    .map(move |to| Action::Drop { piece, to })
            })
    }

    /// Applies a previously checked action without allocating game history.
    /// The outcome includes decisive wins only; Game adds path-dependent draws.
    fn after_legal_action(&self, action: Action) -> (Self, Transition) {
        let mut next = *self;
        let player = next.side_to_move;
        let mut captured = None;

        match action {
            Action::Move { from, to, promote } => {
                let mut piece = next.board[from.index()].expect("legal move has a source piece");
                if let Some(target) = next.board[to.index()] {
                    captured = Some(target.kind);
                    if let Some(hand_piece) = HandPiece::from_captured(target.kind) {
                        next.hands[player.index()][hand_piece.index()] += 1;
                    }
                }

                next.board[from.index()] = None;
                if promote {
                    piece.kind = piece
                        .kind
                        .promoted()
                        .expect("legal promotion has a promoted kind");
                }
                next.board[to.index()] = Some(piece);
            }
            Action::Drop { piece, to } => {
                next.hands[player.index()][piece.index()] -= 1;
                next.board[to.index()] = Some(Piece::new(piece.piece_kind(), player));
            }
        }

        next.side_to_move = player.opponent();

        let outcome = if captured == Some(PieceKind::Koropokkuru) {
            Outcome::Win {
                player,
                reason: WinReason::KoropokkuruCaptured,
            }
        } else if !next.has_legal_action() {
            Outcome::Win {
                player,
                reason: WinReason::OpponentHasNoLegalAction,
            }
        } else {
            Outcome::Ongoing
        };
        (
            next,
            Transition {
                action,
                player,
                captured,
                promoted: matches!(action, Action::Move { promote: true, .. }),
                outcome,
            },
        )
    }

    /// Returns the piece that may legally step from `from` to `to`.
    ///
    /// A Koropokkuru may not step onto an attacked square. Other pieces are
    /// free to move even when that leaves their Koropokkuru exposed: the game
    /// is decided by the capture itself.
    fn movable_piece(&self, from: Square, to: Square) -> Option<Piece> {
        let piece = self.piece_at(from)?;
        if piece.owner != self.side_to_move {
            return None;
        }
        if self
            .piece_at(to)
            .is_some_and(|target| target.owner == self.side_to_move)
        {
            return None;
        }

        if !matches_movement(piece, from, to) {
            return None;
        }

        if piece.kind == PieceKind::Koropokkuru {
            let mut board_after_move = self.board;
            board_after_move[from.index()] = None;
            board_after_move[to.index()] = Some(piece);
            if square_is_attacked(&board_after_move, to, piece.owner.opponent()) {
                return None;
            }
        }

        Some(piece)
    }

    /// Checks the hand, the destination and the three Kodama drop
    /// restrictions: a Kodama may not land on the last row, share a column
    /// with another unpromoted Kodama of its owner, or deliver checkmate.
    fn is_legal_drop(&self, piece: HandPiece, to: Square) -> bool {
        let player = self.side_to_move;
        if self.hand_count(player, piece) == 0 || self.piece_at(to).is_some() {
            return false;
        }
        if piece != HandPiece::Kodama {
            return true;
        }

        let own_kodama = Some(Piece::new(PieceKind::Kodama, player));
        let column_is_free = Square::ALL
            .into_iter()
            .filter(|square| square.column() == to.column())
            .all(|square| self.piece_at(square) != own_kodama);
        to.row() != player.last_row() && column_is_free && !self.kodama_drop_checkmates(to)
    }

    /// Reports whether dropping a Kodama on `to` attacks the opposing
    /// Koropokkuru and leaves the opponent no way to save it.
    ///
    /// Only a board move can answer: a drop neither moves the Koropokkuru nor
    /// removes the adjacent attacker, and one-step attacks cannot be blocked.
    fn kodama_drop_checkmates(&self, to: Square) -> bool {
        let player = self.side_to_move;
        let opponent = player.opponent();
        let opposing_king = Some(Piece::new(PieceKind::Koropokkuru, opponent));
        let attacks_king = to
            .offset(player.forward_delta(), 0)
            .is_some_and(|attacked| self.piece_at(attacked) == opposing_king);
        if !attacks_king {
            return false;
        }

        let mut after_drop = *self;
        after_drop.board[to.index()] = Some(Piece::new(PieceKind::Kodama, player));
        after_drop.side_to_move = opponent;
        !after_drop
            .legal_steps()
            .any(|(_, from, destination)| after_drop.step_saves_koropokkuru(from, destination))
    }

    /// Reports whether stepping `from` → `to` leaves the mover's Koropokkuru
    /// out of capture range, or wins outright by capturing the opposing one.
    fn step_saves_koropokkuru(&self, from: Square, to: Square) -> bool {
        let mover = self.side_to_move;
        if self
            .piece_at(to)
            .is_some_and(|target| target.kind == PieceKind::Koropokkuru)
        {
            return true;
        }

        let mut board = self.board;
        board[to.index()] = board[from.index()].take();
        let own_king = Some(Piece::new(PieceKind::Koropokkuru, mover));
        Square::ALL
            .into_iter()
            .find(|square| board[square.index()] == own_king)
            .is_none_or(|king| !square_is_attacked(&board, king, mover.opponent()))
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
    /// Captured board kind, before a promoted piece reverts in hand.
    pub captured: Option<PieceKind>,
    /// Whether this action flipped the moved piece to its promoted face.
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
/// `Position`. `Position` stays cheap to copy during tree search.
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
    /// Arena diagnostics use it to count distinct opening paths.
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
        let (next, transition) = self.position.after_legal_action(action);
        if matches!(transition.outcome, Outcome::Win { .. }) {
            Some(0)
        } else {
            Some(
                self.repetitions
                    .get(&next)
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

        let (position, mut transition) = self.position.after_legal_action(action);
        self.position = position;
        self.actions.push(action);
        self.positions.push(position);

        // Decisive wins take precedence over history-dependent repetition.
        if transition.outcome == Outcome::Ongoing {
            let occurrence = self.repetitions.entry(position).or_insert(0);
            *occurrence += 1;
            if *occurrence >= 3 {
                transition.outcome = Outcome::Draw {
                    reason: DrawReason::ThreefoldRepetition,
                };
            }
        }
        self.outcome = transition.outcome;
        Ok(transition)
    }
}

const KOROPOKKURU_MOVES: &[(i8, i8)] = &[
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];
// Relative row +1 always means "forward". `forward_delta` maps it to the
// absolute board direction of the piece owner.
//
// Kirin, Super Oni and Kodama Samurai share the same six steps: every
// direction except the two backward diagonals.
const KIRIN_MOVES: &[(i8, i8)] = &[(1, -1), (1, 0), (1, 1), (0, -1), (0, 1), (-1, 0)];
const ONI_MOVES: &[(i8, i8)] = &[(1, -1), (1, 0), (1, 1), (-1, -1), (-1, 1)];
const KODAMA_MOVES: &[(i8, i8)] = &[(1, 0)];

fn movement_offsets(kind: PieceKind) -> &'static [(i8, i8)] {
    match kind {
        PieceKind::Koropokkuru => KOROPOKKURU_MOVES,
        PieceKind::Kirin | PieceKind::SuperOni | PieceKind::KodamaSamurai => KIRIN_MOVES,
        PieceKind::Oni => ONI_MOVES,
        PieceKind::Kodama => KODAMA_MOVES,
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

        matches_movement(piece, from, target)
    })
}

fn matches_movement(piece: Piece, from: Square, to: Square) -> bool {
    let absolute_row = i16::from(to.row()) - i16::from(from.row());
    let column = i16::from(to.column()) - i16::from(from.column());
    let relative_row = absolute_row * i16::from(piece.owner.forward_delta());
    movement_offsets(piece.kind)
        .iter()
        .any(|&(row, col)| i16::from(row) == relative_row && i16::from(col) == column)
}
