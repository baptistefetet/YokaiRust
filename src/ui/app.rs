//! Input handling and shared application state for match and replay modes.

use std::time::Instant;

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use yokai::{Action, ActionAnalysis, Game, HandPiece, Player, Replay, Square};

use super::ai::AiWorker;
use super::render::{player_label, transition_message};
use super::session::{Controller, Focus, MatchSession, ReplaySession, Selection, Session};

pub(super) struct App {
    pub(super) session: Session,
    pub(super) cursor: Square,
    pub(super) focus: Focus,
    pub(super) selection: Option<Selection>,
    pub(super) hand_index: usize,
    pub(super) analysis_offset: usize,
    pub(super) notice: String,
    pub(super) should_quit: bool,
}

impl App {
    pub(super) fn for_human_match() -> Self {
        Self::for_match_session(MatchSession::human_vs_human())
    }

    pub(super) fn for_cpu_match(worker: AiWorker) -> Self {
        Self::for_match_session(MatchSession::human_vs_cpu(worker))
    }

    pub(super) fn for_match_session(session: MatchSession) -> Self {
        Self {
            session: Session::Match(session),
            cursor: initial_cursor(),
            focus: Focus::Board,
            selection: None,
            hand_index: 0,
            analysis_offset: 0,
            notice: "Select a piece, then its destination".to_owned(),
            should_quit: false,
        }
    }

    pub(super) fn for_replay(replay: Replay) -> Result<Self, yokai::ReplayError> {
        Ok(Self {
            session: Session::Replay(ReplaySession::new(replay)?),
            cursor: initial_cursor(),
            focus: Focus::Board,
            selection: None,
            hand_index: 0,
            analysis_offset: 0,
            notice: "Use ← and → to step through the game".to_owned(),
            should_quit: false,
        })
    }

    pub(super) const fn game(&self) -> &Game {
        match &self.session {
            Session::Match(session) => &session.game,
            Session::Replay(session) => &session.game,
        }
    }

    pub(super) fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    pub(super) fn tick_at(&mut self, now: Instant) {
        let update = match &mut self.session {
            Session::Match(session) => session.tick(now),
            Session::Replay(_) => None,
        };
        if let Some(update) = update {
            if let Some(action) = update.action {
                self.cursor = action.destination();
                self.selection = None;
                self.focus = Focus::Board;
            }
            self.notice = update.notice;
        }
    }

    pub(super) fn handle_key(&mut self, key: KeyEvent) {
        if matches!(key.code, KeyCode::Char('q' | 'Q')) {
            self.should_quit = true;
            return;
        }
        match self.session {
            Session::Match(_) => self.handle_match_key(key.code),
            Session::Replay(_) => self.handle_replay_key(key.code),
        }
    }

    pub(super) fn handle_match_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('n' | 'N') => self.reset_match(),
            KeyCode::Esc => self.cancel_selection(),
            KeyCode::Tab | KeyCode::BackTab => self.toggle_focus(),
            KeyCode::Char('1'..='3') => {
                let KeyCode::Char(digit) = code else {
                    return;
                };
                self.select_hand_piece((digit as usize) - ('1' as usize));
            }
            KeyCode::Enter | KeyCode::Char(' ') => self.activate(),
            KeyCode::Up | KeyCode::Char('w' | 'W' | 'k' | 'K') => self.move_selection(-1, 0),
            KeyCode::Down | KeyCode::Char('s' | 'S' | 'j' | 'J') => self.move_selection(1, 0),
            KeyCode::Left | KeyCode::Char('a' | 'A' | 'h' | 'H') => self.move_selection(0, -1),
            KeyCode::Right | KeyCode::Char('d' | 'D' | 'l' | 'L') => self.move_selection(0, 1),
            _ => {}
        }
    }

    pub(super) fn handle_replay_key(&mut self, code: KeyCode) {
        if matches!(code, KeyCode::Up | KeyCode::PageUp) {
            self.analysis_offset = self.analysis_offset.saturating_sub(1);
            return;
        }
        if matches!(code, KeyCode::Down | KeyCode::PageDown) {
            let analysis_count = match &self.session {
                Session::Replay(session) => session
                    .current_analyses()
                    .map_or(0, <[ActionAnalysis]>::len),
                Session::Match(_) => 0,
            };
            self.analysis_offset = (self.analysis_offset + 1).min(analysis_count.saturating_sub(1));
            return;
        }
        let Session::Replay(session) = &self.session else {
            return;
        };
        let target = match code {
            KeyCode::Left | KeyCode::Char('h' | 'H') => session.ply.saturating_sub(1),
            KeyCode::Right | KeyCode::Char('l' | 'L' | ' ') => {
                (session.ply + 1).min(session.replay.actions.len())
            }
            KeyCode::Home | KeyCode::Char('g') => 0,
            KeyCode::End | KeyCode::Char('G') => session.replay.actions.len(),
            _ => return,
        };
        self.seek_replay(target);
    }

    pub(super) fn seek_replay(&mut self, target: usize) {
        let Session::Replay(session) = &mut self.session else {
            return;
        };
        match session.seek(target) {
            Ok(()) => {
                self.analysis_offset = 0;
                self.cursor = session
                    .replay
                    .actions
                    .get(session.ply.saturating_sub(1))
                    .map_or_else(initial_cursor, |action| action.destination());
                self.notice = if session.ply == session.replay.actions.len() {
                    "End of replay".to_owned()
                } else {
                    format!("Position before move {}", session.ply + 1)
                };
            }
            Err(error) => self.notice = format!("Invalid replay: {error}"),
        }
    }

    pub(super) fn reset_match(&mut self) {
        let Session::Match(session) = &mut self.session else {
            return;
        };
        session.reset();
        self.cursor = initial_cursor();
        self.focus = Focus::Board;
        self.selection = None;
        self.analysis_offset = 0;
        "New game — Player 1 moves first".clone_into(&mut self.notice);
    }

    pub(super) fn cancel_selection(&mut self) {
        self.selection = None;
        self.focus = Focus::Board;
        "Selection cancelled".clone_into(&mut self.notice);
    }

    pub(super) fn toggle_focus(&mut self) {
        if self.game().outcome().is_terminal() {
            return;
        }
        self.focus = match self.focus {
            Focus::Board => Focus::Hand,
            Focus::Hand => Focus::Board,
        };
        self.notice = match self.focus {
            Focus::Board => "Board focused".to_owned(),
            Focus::Hand => "Hand focused — choose with ←/→ or 1/2/3".to_owned(),
        };
    }

    pub(super) fn move_selection(&mut self, row_delta: i8, column_delta: i8) {
        if self.focus == Focus::Hand {
            let change = if row_delta < 0 || column_delta < 0 {
                -1
            } else {
                1
            };
            self.hand_index = if change < 0 {
                self.hand_index.saturating_sub(1)
            } else {
                (self.hand_index + 1).min(HandPiece::ALL.len() - 1)
            };
            return;
        }

        let row = move_axis(self.cursor.row(), row_delta, yokai::BOARD_HEIGHT);
        let column = move_axis(self.cursor.column(), column_delta, yokai::BOARD_WIDTH);
        self.cursor = Square::new(row, column).expect("clamped board cursor");
    }

    pub(super) fn select_hand_piece(&mut self, index: usize) {
        if self.game().outcome().is_terminal() || index >= HandPiece::ALL.len() {
            return;
        }
        self.hand_index = index;
        self.focus = Focus::Hand;
        self.activate();
    }

    pub(super) fn activate(&mut self) {
        if self.game().outcome().is_terminal() {
            "The game is over — press N to start a new game".clone_into(&mut self.notice);
            return;
        }
        let cpu_turn = match &self.session {
            Session::Match(session) => {
                session.controller(session.game.position().side_to_move()) == Controller::Cpu
            }
            Session::Replay(_) => false,
        };
        if cpu_turn {
            "Wait for the CPU to finish its move".clone_into(&mut self.notice);
            return;
        }
        match self.focus {
            Focus::Hand => self.activate_hand(),
            Focus::Board => self.activate_board(),
        }
    }

    pub(super) fn activate_hand(&mut self) {
        let piece = HandPiece::ALL[self.hand_index];
        let player = self.game().position().side_to_move();
        if self.game().position().hand_count(player, piece) == 0 {
            self.notice = format!("No {piece} in {}'s hand", player_label(player));
            return;
        }
        self.selection = Some(Selection::Drop(piece));
        self.focus = Focus::Board;
        self.notice = format!("{piece} selected — choose a green square");
    }

    pub(super) fn activate_board(&mut self) {
        match self.selection {
            Some(Selection::MoveFrom(from)) => {
                let action = Action::Move {
                    from,
                    to: self.cursor,
                };
                if self.game().is_legal_action(action) {
                    self.apply_action(action);
                } else if self
                    .game()
                    .position()
                    .piece_at(self.cursor)
                    .is_some_and(|piece| piece.owner == self.game().position().side_to_move())
                {
                    self.selection = Some(Selection::MoveFrom(self.cursor));
                    self.notice = format!("Source changed to {}", self.cursor);
                } else {
                    "Illegal destination — green squares are legal".clone_into(&mut self.notice);
                }
            }
            Some(Selection::Drop(piece)) => {
                let action = Action::Drop {
                    piece,
                    to: self.cursor,
                };
                if self.game().is_legal_action(action) {
                    self.apply_action(action);
                } else {
                    "This piece cannot be dropped here".clone_into(&mut self.notice);
                }
            }
            None => {
                let player = self.game().position().side_to_move();
                match self.game().position().piece_at(self.cursor) {
                    Some(piece) if piece.owner == player => {
                        self.selection = Some(Selection::MoveFrom(self.cursor));
                        self.notice = format!("{} selected — choose a green square", self.cursor);
                    }
                    Some(_) => {
                        "That piece belongs to your opponent".clone_into(&mut self.notice);
                    }
                    None => {
                        "Empty square — select a piece or press Tab to use the hand"
                            .clone_into(&mut self.notice);
                    }
                }
            }
        }
    }

    pub(super) fn apply_action(&mut self, action: Action) {
        let Session::Match(session) = &mut self.session else {
            return;
        };
        match session.game.apply(action) {
            Ok(transition) => {
                session.predictions = None;
                if let Some(ai) = &session.ai {
                    ai.worker.advance(action, &session.game);
                }
                self.selection = None;
                self.focus = Focus::Board;
                self.cursor = action.destination();
                self.notice = transition_message(transition);
                if let Err(error) = session.request_cpu_move(Instant::now()) {
                    self.notice = format!("{} · CPU error: {error}", self.notice);
                }
            }
            Err(error) => self.notice = format!("Move rejected: {error}"),
        }
    }

    pub(super) fn effective_selection(&self) -> Option<Selection> {
        let Session::Match(session) = &self.session else {
            return None;
        };
        self.selection.or_else(|| {
            let action = session.ai.as_ref()?.state.pending_action()?;
            Some(match action {
                Action::Move { from, .. } => Selection::MoveFrom(from),
                Action::Drop { piece, .. } => Selection::Drop(piece),
            })
        })
    }

    pub(super) fn legal_destination(&self, square: Square) -> bool {
        let Session::Match(session) = &self.session else {
            return false;
        };
        match self.effective_selection() {
            Some(Selection::MoveFrom(from)) => session
                .game
                .is_legal_action(Action::Move { from, to: square }),
            Some(Selection::Drop(piece)) => session
                .game
                .is_legal_action(Action::Drop { piece, to: square }),
            None => false,
        }
    }

    pub(super) fn selected_source(&self) -> Option<Square> {
        match self.effective_selection() {
            Some(Selection::MoveFrom(square)) => Some(square),
            Some(Selection::Drop(_)) | None => None,
        }
    }

    pub(super) fn focused_ai_destination(&self, now: Instant) -> Option<Square> {
        let Session::Match(session) = &self.session else {
            return None;
        };
        session.ai.as_ref()?.state.focused_destination(now)
    }

    pub(super) fn selected_hand_piece(
        &self,
        player: Player,
        piece: HandPiece,
        index: usize,
    ) -> bool {
        let Session::Match(session) = &self.session else {
            return false;
        };
        let side_to_move = session.game.position().side_to_move();
        if side_to_move != player {
            return false;
        }
        let human_focus = session.controller(player) == Controller::Human
            && self.focus == Focus::Hand
            && self.hand_index == index;
        let selected_drop = self.selection == Some(Selection::Drop(piece));
        let ai_drop = session.ai.as_ref().is_some_and(|ai| {
            matches!(
                ai.state.pending_action(),
                Some(Action::Drop { piece: selected, .. }) if selected == piece
            )
        });
        human_focus || selected_drop || ai_drop
    }

    pub(super) fn board_cursor(&self, square: Square) -> bool {
        let Session::Match(session) = &self.session else {
            return false;
        };
        session.controller(session.game.position().side_to_move()) == Controller::Human
            && self.focus == Focus::Board
            && self.cursor == square
    }

    pub(super) fn last_action(&self) -> Option<Action> {
        match &self.session {
            Session::Match(session) => session.game.actions().last().copied(),
            Session::Replay(session) => session
                .ply
                .checked_sub(1)
                .and_then(|index| session.replay.actions.get(index))
                .copied(),
        }
    }
}

pub(super) fn initial_cursor() -> Square {
    Square::new(3, 1).expect("initial cursor is on the board")
}

pub(super) fn move_axis(value: u8, delta: i8, upper_bound: u8) -> u8 {
    if delta < 0 {
        value.saturating_sub(delta.unsigned_abs())
    } else {
        value
            .saturating_add(delta.unsigned_abs())
            .min(upper_bound - 1)
    }
}
