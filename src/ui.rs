//! Ratatui application state, input handling and rendering.
//!
//! The rules remain in the library. This module only coordinates a live match
//! or a replay cursor, which keeps future CPU thinking outside the rendering
//! loop and lets both modes share the same board, history and analysis panels.

mod ai;

use std::{
    cmp::Ordering,
    error::Error,
    fmt::{self, Write as _},
    io,
    str::FromStr,
    time::{Duration, Instant},
};

use ratatui::{
    Frame,
    crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Cell, Paragraph, Row, Table, Wrap},
};
use yokai::{
    Action, ActionAnalysis, DrawReason, Game, HandPiece, Outcome, Piece, PieceKind, Player, Replay,
    SearchResult, Square, TrainingConfig, WinReason,
};

use self::ai::{AiEvent, AiWorker};

const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(100);
const AI_SOURCE_FOCUS_DURATION: Duration = Duration::from_millis(800);
const AI_MOVE_DELAY: Duration = Duration::from_millis(1_500);
const ACTIVE_TRAINING_CONFIG: &str = "config/training.toml";
const MINIMUM_WIDTH: u16 = 70;
const MINIMUM_HEIGHT: u16 = 24;

/// Match type accepted by the `play` command.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum PlayMode {
    /// Two people alternate on the same terminal.
    #[default]
    HumanVsHuman,
    /// The human is First at the bottom; the latest champion is Second.
    HumanVsCpu,
}

impl FromStr for PlayMode {
    type Err = io::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "human-vs-human" | "hvh" => Ok(Self::HumanVsHuman),
            "human-vs-cpu" | "hvc" => Ok(Self::HumanVsCpu),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown play mode `{input}`"),
            )),
        }
    }
}

/// Starts the fullscreen interface for a local match.
pub(crate) fn play(mode: PlayMode) -> Result<(), Box<dyn Error>> {
    let app = match mode {
        PlayMode::HumanVsHuman => App::for_human_match(),
        PlayMode::HumanVsCpu => {
            let config = TrainingConfig::load(ACTIVE_TRAINING_CONFIG)?;
            App::for_cpu_match(AiWorker::spawn(config)?)
        }
    };
    run(app)?;
    Ok(())
}

/// Starts the fullscreen replay viewer for an already validated replay.
pub(crate) fn watch(replay: Replay) -> io::Result<()> {
    let app = App::for_replay(replay).map_err(io::Error::other)?;
    run(app)
}

fn run(mut app: App) -> io::Result<()> {
    ratatui::run(|terminal| {
        while !app.should_quit {
            app.tick();
            terminal.draw(|frame| render(frame, &app))?;
            if event::poll(EVENT_POLL_INTERVAL)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                app.handle_key(key);
            }
        }
        Ok(())
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Controller {
    Human,
    Cpu,
}

impl Controller {
    const fn label(self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Cpu => "CPU",
        }
    }
}

#[derive(Clone, Debug)]
struct PredictionSnapshot {
    player: Player,
    root_value: Option<f32>,
    wdl: Option<[f32; 3]>,
    actions: Vec<ActionAnalysis>,
}

struct MatchSession {
    game: Game,
    controllers: [Controller; 2],
    predictions: Option<PredictionSnapshot>,
    ai: Option<AiRuntime>,
}

impl MatchSession {
    fn human_vs_human() -> Self {
        Self {
            game: Game::new(Player::First),
            controllers: [Controller::Human, Controller::Human],
            predictions: None,
            ai: None,
        }
    }

    fn human_vs_cpu(worker: AiWorker) -> Self {
        Self {
            game: Game::new(Player::First),
            // Absolute First is always drawn at the bottom, which keeps the
            // human's orientation stable in every single-player game.
            controllers: [Controller::Human, Controller::Cpu],
            predictions: None,
            ai: Some(AiRuntime {
                worker,
                state: AiState::Loading,
                generation: None,
                simulations: None,
                next_request_id: 0,
            }),
        }
    }

    fn reset(&mut self) {
        self.game = Game::new(Player::First);
        self.predictions = None;
        if let Some(ai) = &mut self.ai {
            ai.next_request_id = ai.next_request_id.wrapping_add(1);
            ai.worker.reset();
            ai.state = if ai.generation.is_some() {
                AiState::Idle
            } else {
                AiState::Loading
            };
        }
    }

    const fn controller(&self, player: Player) -> Controller {
        self.controllers[player.index()]
    }

    fn request_cpu_move(&mut self, now: Instant) -> Result<(), String> {
        if self.game.outcome().is_terminal()
            || self.controller(self.game.position().side_to_move()) != Controller::Cpu
        {
            return Ok(());
        }
        self.request_search(now, SearchPurpose::CpuMove)
    }

    fn request_human_predictions(&mut self, now: Instant) -> Result<(), String> {
        if self.game.outcome().is_terminal()
            || self.controller(self.game.position().side_to_move()) != Controller::Human
            || self.predictions.is_some()
            || !matches!(self.ai.as_ref().map(|ai| &ai.state), Some(AiState::Idle))
        {
            return Ok(());
        }
        self.request_search(now, SearchPurpose::HumanPrediction)
    }

    fn request_search(&mut self, now: Instant, purpose: SearchPurpose) -> Result<(), String> {
        let Some(ai) = &mut self.ai else {
            return Err("CPU controller has no worker".to_owned());
        };
        let request_id = ai.next_request_id;
        ai.next_request_id = ai.next_request_id.wrapping_add(1);
        if let Err(message) = ai.worker.search(request_id, &self.game) {
            ai.state = AiState::Failed(message.clone());
            return Err(message);
        }
        ai.state = AiState::Thinking {
            request_id,
            requested_at: now,
            purpose,
        };
        self.predictions = None;
        Ok(())
    }

    fn tick(&mut self, now: Instant) -> Option<MatchUpdate> {
        let mut update = self.process_ai_events(now);
        if let Some(move_update) = self.apply_pending_cpu_move(now) {
            update = Some(move_update);
        }
        if let Err(message) = self.request_human_predictions(now) {
            update = Some(MatchUpdate::notice(format!("CPU error: {message}")));
        }
        update
    }

    fn process_ai_events(&mut self, now: Instant) -> Option<MatchUpdate> {
        let mut update = None;
        loop {
            let event = self.ai.as_ref()?.worker.try_event();
            let Some(event) = event else {
                break;
            };
            if let Some(event_update) = self.process_ai_event(event, now) {
                update = Some(event_update);
            }
        }
        update
    }

    fn process_ai_event(&mut self, event: AiEvent, now: Instant) -> Option<MatchUpdate> {
        let ai = self.ai.as_mut()?;
        match event {
            AiEvent::Ready {
                generation,
                simulations,
            } => {
                ai.generation = Some(generation);
                ai.simulations = Some(simulations);
                if matches!(ai.state, AiState::Loading) {
                    ai.state = AiState::Idle;
                }
                Some(MatchUpdate::notice(format!(
                    "Champion generation {generation} loaded"
                )))
            }
            AiEvent::SearchReady {
                request_id,
                result,
                search_started,
                search_finished,
            } if ai.state.matches_request(request_id) => {
                let purpose = ai
                    .state
                    .search_purpose(request_id)
                    .expect("matching search has a purpose");
                let player = self.game.position().side_to_move();
                let search_duration = search_finished.saturating_duration_since(search_started);
                self.predictions = Some(PredictionSnapshot {
                    player,
                    root_value: Some(result.root_value),
                    wdl: None,
                    actions: result.analysis.clone(),
                });
                Some(match purpose {
                    SearchPurpose::HumanPrediction => {
                        ai.state = AiState::Idle;
                        MatchUpdate::notice(format!(
                            "Champion predictions ready for {} · {:.2}s",
                            player_label(player),
                            search_duration.as_secs_f32()
                        ))
                    }
                    SearchPurpose::CpuMove => {
                        let action = result.best_action;
                        ai.state = AiState::WaitingToPlay {
                            request_id,
                            result,
                            destination_at: now + AI_SOURCE_FOCUS_DURATION,
                            apply_at: now + AI_MOVE_DELAY,
                            search_duration,
                        };
                        let focus = match action {
                            Action::Move { from, .. } => format!("source {from}"),
                            Action::Drop { piece, .. } => format!("{piece} in hand"),
                        };
                        MatchUpdate::notice(format!("CPU focuses {focus}"))
                    }
                })
            }
            AiEvent::Failed {
                request_id,
                message,
            } if request_id.is_none_or(|id| ai.state.matches_request(id)) => {
                ai.state = AiState::Failed(message.clone());
                Some(MatchUpdate::notice(format!("CPU error: {message}")))
            }
            AiEvent::SearchReady { .. } | AiEvent::Failed { .. } => None,
        }
    }

    fn apply_pending_cpu_move(&mut self, now: Instant) -> Option<MatchUpdate> {
        let ai = self.ai.as_mut()?;
        if !matches!(ai.state, AiState::WaitingToPlay { apply_at, .. } if now >= apply_at) {
            return None;
        }
        let state = std::mem::replace(&mut ai.state, AiState::Idle);
        let AiState::WaitingToPlay {
            result,
            search_duration,
            ..
        } = state
        else {
            unreachable!("due CPU move must be waiting to play");
        };
        let action = result.best_action;
        match self.game.apply(action) {
            Ok(transition) => {
                self.predictions = None;
                ai.worker.advance(action, &self.game);
                Some(MatchUpdate {
                    notice: format!(
                        "{} · CPU search {:.2}s",
                        transition_message(transition),
                        search_duration.as_secs_f32()
                    ),
                    action: Some(action),
                })
            }
            Err(error) => {
                let message = format!("CPU produced an illegal move: {error}");
                ai.state = AiState::Failed(message.clone());
                Some(MatchUpdate::notice(message))
            }
        }
    }

    fn ai_status(&self, now: Instant) -> Option<String> {
        let ai = self.ai.as_ref()?;
        Some(match &ai.state {
            AiState::Loading => "Loading latest champion…".to_owned(),
            AiState::Idle => format!(
                "CPU ready{}",
                ai.generation
                    .map_or_else(String::new, |generation| format!(" · g{generation}"))
            ),
            AiState::Thinking { requested_at, .. } if ai.generation.is_none() => format!(
                "Loading latest champion… · move queued {:.1}s",
                now.saturating_duration_since(*requested_at).as_secs_f32()
            ),
            AiState::Thinking {
                requested_at,
                purpose: SearchPurpose::CpuMove,
                ..
            } => format!(
                "CPU thinking… {:.1}s",
                now.saturating_duration_since(*requested_at).as_secs_f32()
            ),
            AiState::Thinking {
                requested_at,
                purpose: SearchPurpose::HumanPrediction,
                ..
            } => format!(
                "Champion analyzing human turn… {:.1}s",
                now.saturating_duration_since(*requested_at).as_secs_f32()
            ),
            AiState::WaitingToPlay {
                result,
                destination_at,
                apply_at,
                ..
            } if now < *destination_at => match result.best_action {
                Action::Move { from, .. } => format!(
                    "CPU focuses {from} · target in {:.1}s",
                    destination_at.saturating_duration_since(now).as_secs_f32()
                ),
                Action::Drop { piece, .. } => format!(
                    "CPU selects {piece} in hand · target in {:.1}s",
                    destination_at.saturating_duration_since(now).as_secs_f32()
                ),
            },
            AiState::WaitingToPlay {
                result, apply_at, ..
            } => format!(
                "CPU targets {} · plays in {:.1}s",
                result.best_action.destination(),
                apply_at.saturating_duration_since(now).as_secs_f32()
            ),
            AiState::Failed(message) => format!("CPU unavailable: {message}"),
        })
    }
}

struct AiRuntime {
    worker: AiWorker,
    state: AiState,
    generation: Option<u32>,
    simulations: Option<u32>,
    next_request_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchPurpose {
    HumanPrediction,
    CpuMove,
}

enum AiState {
    Loading,
    Idle,
    Thinking {
        request_id: u64,
        requested_at: Instant,
        purpose: SearchPurpose,
    },
    WaitingToPlay {
        request_id: u64,
        result: Box<SearchResult>,
        destination_at: Instant,
        apply_at: Instant,
        search_duration: Duration,
    },
    Failed(String),
}

impl AiState {
    const fn matches_request(&self, expected: u64) -> bool {
        matches!(
            self,
            Self::Thinking { request_id, .. } | Self::WaitingToPlay { request_id, .. }
                if *request_id == expected
        )
    }

    const fn search_purpose(&self, expected: u64) -> Option<SearchPurpose> {
        match self {
            Self::Thinking {
                request_id,
                purpose,
                ..
            } if *request_id == expected => Some(*purpose),
            Self::Loading
            | Self::Idle
            | Self::Thinking { .. }
            | Self::WaitingToPlay { .. }
            | Self::Failed(_) => None,
        }
    }

    fn pending_action(&self) -> Option<Action> {
        match self {
            Self::WaitingToPlay { result, .. } => Some(result.best_action),
            Self::Loading | Self::Idle | Self::Thinking { .. } | Self::Failed(_) => None,
        }
    }

    fn focused_destination(&self, now: Instant) -> Option<Square> {
        match self {
            Self::WaitingToPlay {
                result,
                destination_at,
                ..
            } if now >= *destination_at => Some(result.best_action.destination()),
            Self::Loading
            | Self::Idle
            | Self::Thinking { .. }
            | Self::WaitingToPlay { .. }
            | Self::Failed(_) => None,
        }
    }
}

struct MatchUpdate {
    notice: String,
    action: Option<Action>,
}

impl MatchUpdate {
    fn notice(notice: String) -> Self {
        Self {
            notice,
            action: None,
        }
    }
}

#[derive(Clone, Debug)]
struct ReplaySession {
    replay: Replay,
    game: Game,
    ply: usize,
}

impl ReplaySession {
    fn new(replay: Replay) -> Result<Self, yokai::ReplayError> {
        replay.to_game()?;
        Ok(Self {
            game: Game::new(replay.initial_player),
            replay,
            ply: 0,
        })
    }

    fn seek(&mut self, target: usize) -> Result<(), yokai::MoveError> {
        let target = target.min(self.replay.actions.len());
        let mut game = Game::new(self.replay.initial_player);
        for &action in &self.replay.actions[..target] {
            game.apply(action)?;
        }
        self.game = game;
        self.ply = target;
        Ok(())
    }

    fn current_analyses(&self) -> Option<&[ActionAnalysis]> {
        self.replay
            .analyses
            .as_ref()?
            .get(self.ply)
            .map(Vec::as_slice)
    }
}

enum Session {
    Match(MatchSession),
    Replay(ReplaySession),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Focus {
    Board,
    Hand,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Selection {
    MoveFrom(Square),
    Drop(HandPiece),
}

struct App {
    session: Session,
    cursor: Square,
    focus: Focus,
    selection: Option<Selection>,
    hand_index: usize,
    analysis_offset: usize,
    notice: String,
    should_quit: bool,
}

impl App {
    fn for_human_match() -> Self {
        Self::for_match_session(MatchSession::human_vs_human())
    }

    fn for_cpu_match(worker: AiWorker) -> Self {
        Self::for_match_session(MatchSession::human_vs_cpu(worker))
    }

    fn for_match_session(session: MatchSession) -> Self {
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

    fn for_replay(replay: Replay) -> Result<Self, yokai::ReplayError> {
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

    const fn game(&self) -> &Game {
        match &self.session {
            Session::Match(session) => &session.game,
            Session::Replay(session) => &session.game,
        }
    }

    fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    fn tick_at(&mut self, now: Instant) {
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

    fn handle_key(&mut self, key: KeyEvent) {
        if matches!(key.code, KeyCode::Char('q' | 'Q')) {
            self.should_quit = true;
            return;
        }
        match self.session {
            Session::Match(_) => self.handle_match_key(key.code),
            Session::Replay(_) => self.handle_replay_key(key.code),
        }
    }

    fn handle_match_key(&mut self, code: KeyCode) {
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

    fn handle_replay_key(&mut self, code: KeyCode) {
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

    fn seek_replay(&mut self, target: usize) {
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

    fn reset_match(&mut self) {
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

    fn cancel_selection(&mut self) {
        self.selection = None;
        self.focus = Focus::Board;
        "Selection cancelled".clone_into(&mut self.notice);
    }

    fn toggle_focus(&mut self) {
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

    fn move_selection(&mut self, row_delta: i8, column_delta: i8) {
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

    fn select_hand_piece(&mut self, index: usize) {
        if self.game().outcome().is_terminal() || index >= HandPiece::ALL.len() {
            return;
        }
        self.hand_index = index;
        self.focus = Focus::Hand;
        self.activate();
    }

    fn activate(&mut self) {
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

    fn activate_hand(&mut self) {
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

    fn activate_board(&mut self) {
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

    fn apply_action(&mut self, action: Action) {
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

    fn effective_selection(&self) -> Option<Selection> {
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

    fn legal_destination(&self, square: Square) -> bool {
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

    fn selected_source(&self) -> Option<Square> {
        match self.effective_selection() {
            Some(Selection::MoveFrom(square)) => Some(square),
            Some(Selection::Drop(_)) | None => None,
        }
    }

    fn focused_ai_destination(&self, now: Instant) -> Option<Square> {
        let Session::Match(session) = &self.session else {
            return None;
        };
        session.ai.as_ref()?.state.focused_destination(now)
    }

    fn selected_hand_piece(&self, player: Player, piece: HandPiece, index: usize) -> bool {
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

    fn board_cursor(&self, square: Square) -> bool {
        let Session::Match(session) = &self.session else {
            return false;
        };
        session.controller(session.game.position().side_to_move()) == Controller::Human
            && self.focus == Focus::Board
            && self.cursor == square
    }

    fn last_action(&self) -> Option<Action> {
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

fn initial_cursor() -> Square {
    Square::new(3, 1).expect("initial cursor is on the board")
}

fn move_axis(value: u8, delta: i8, upper_bound: u8) -> u8 {
    if delta < 0 {
        value.saturating_sub(delta.unsigned_abs())
    } else {
        value
            .saturating_add(delta.unsigned_abs())
            .min(upper_bound - 1)
    }
}

fn render(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let now = Instant::now();
    if area.width < MINIMUM_WIDTH || area.height < MINIMUM_HEIGHT {
        let message = format!(
            "Terminal too small\n\nMinimum: {MINIMUM_WIDTH}×{MINIMUM_HEIGHT}\nCurrent: {}×{}\n\nPress Q to quit",
            area.width, area.height
        );
        frame.render_widget(
            Paragraph::new(message)
                .alignment(Alignment::Center)
                .block(Block::bordered().title(" YokaiRust ")),
            area,
        );
        return;
    }

    let [header, main, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(18),
        Constraint::Length(3),
    ])
    .areas(area);
    render_header(frame, app, header, now);
    render_main(frame, app, main, now);
    render_footer(frame, app, footer);
}

fn render_header(frame: &mut Frame<'_>, app: &App, area: Rect, now: Instant) {
    let mode = match &app.session {
        Session::Match(session) => {
            let mut label = format!(
                "P1 {} vs P2 {}",
                session.controllers[0].label(),
                session.controllers[1].label()
            );
            if let Some(ai) = &session.ai
                && let Some(generation) = ai.generation
            {
                let _ = write!(label, " g{generation}");
                if let Some(simulations) = ai.simulations {
                    let _ = write!(label, "/{simulations}");
                }
            }
            label
        }
        Session::Replay(session) => {
            format!("Replay {}/{}", session.ply, session.replay.actions.len())
        }
    };
    let status = match &app.session {
        Session::Replay(session) if session.ply < session.replay.actions.len() => {
            format!("before {}", session.replay.actions[session.ply])
        }
        Session::Replay(_) => outcome_text(app.game().outcome()),
        Session::Match(session) => match app.game().outcome() {
            Outcome::Ongoing
                if session.controller(app.game().position().side_to_move()) == Controller::Cpu =>
            {
                session
                    .ai_status(now)
                    .unwrap_or_else(|| "CPU turn".to_owned())
            }
            Outcome::Ongoing => format!(
                "Turn: {}",
                short_player_label(app.game().position().side_to_move())
            ),
            outcome => outcome_text(outcome),
        },
    };
    let line = Line::from(vec![
        Span::styled(mode, Style::default().fg(Color::Cyan).bold()),
        Span::raw("  │  "),
        Span::styled(status, Style::default().fg(Color::Yellow)),
        Span::raw("  │  "),
        Span::raw(app.notice.as_str()),
    ]);
    frame.render_widget(
        Paragraph::new(line).block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" YokaiRust "),
        ),
        area,
    );
}

fn render_main(frame: &mut Frame<'_>, app: &App, area: Rect, now: Instant) {
    if area.width >= 100 {
        let [board, history, analysis] = Layout::horizontal([
            Constraint::Length(34),
            Constraint::Length(24),
            Constraint::Min(42),
        ])
        .areas(area);
        render_board_column(frame, app, board, now);
        render_history(frame, app, history);
        render_analysis(frame, app, analysis);
    } else {
        let [board, sidebar] =
            Layout::horizontal([Constraint::Length(34), Constraint::Min(36)]).areas(area);
        let [history, analysis] =
            Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(sidebar);
        render_board_column(frame, app, board, now);
        render_history(frame, app, history);
        render_analysis(frame, app, analysis);
    }
}

fn render_board_column(frame: &mut Frame<'_>, app: &App, area: Rect, now: Instant) {
    let [second_hand, board, first_hand] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(12),
        Constraint::Length(3),
    ])
    .areas(area);
    render_hand(frame, app, Player::Second, second_hand);
    render_board(frame, app, board, now);
    render_hand(frame, app, Player::First, first_hand);
}

fn render_hand(frame: &mut Frame<'_>, app: &App, player: Player, area: Rect) {
    let position = app.game().position();
    let spans = HandPiece::ALL
        .iter()
        .enumerate()
        .flat_map(|(index, &piece)| {
            let selected = app.selected_hand_piece(player, piece, index);
            let style = if selected {
                Style::default().fg(Color::Black).bg(Color::Yellow).bold()
            } else if position.hand_count(player, piece) > 0 {
                Style::default().fg(player_color(player)).bold()
            } else {
                Style::default().fg(Color::DarkGray)
            };
            [
                Span::styled(
                    format!(
                        "{} {}×{}",
                        index + 1,
                        hand_piece_code(piece),
                        position.hand_count(player, piece)
                    ),
                    style,
                ),
                Span::raw("  "),
            ]
        })
        .collect::<Vec<_>>();
    let title = format!(" {} ", player_label(player));
    frame.render_widget(
        Paragraph::new(Line::from(spans)).block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(player_color(player)))
                .title(title),
        ),
        area,
    );
}

fn render_board(frame: &mut Frame<'_>, app: &App, area: Rect, now: Instant) {
    let rows = Layout::vertical([Constraint::Length(3); 4]).split(area);
    for (row, row_area) in rows.iter().enumerate() {
        let columns = Layout::horizontal([Constraint::Ratio(1, 3); 3]).split(*row_area);
        for (column, &cell_area) in columns.iter().enumerate() {
            let row = u8::try_from(row).expect("board row fits in u8");
            let column = u8::try_from(column).expect("board column fits in u8");
            let square = Square::new(row, column).expect("rendered board square");
            render_square(frame, app, square, cell_area, now);
        }
    }
}

fn render_square(frame: &mut Frame<'_>, app: &App, square: Square, area: Rect, now: Instant) {
    let piece = app.game().position().piece_at(square);
    let (border_color, border_type) = match square_highlight(app, square, now) {
        SquareHighlight::SelectedSource => (Color::Yellow, BorderType::Double),
        SquareHighlight::FocusedDestination | SquareHighlight::Cursor => {
            (Color::Cyan, BorderType::Double)
        }
        SquareHighlight::LegalDestination => (Color::Green, BorderType::Thick),
        SquareHighlight::LastDestination => (Color::Yellow, BorderType::Thick),
        SquareHighlight::LastSource => (Color::DarkGray, BorderType::Thick),
        SquareHighlight::None => (Color::Gray, BorderType::Plain),
    };
    let content = piece.map_or_else(
        || "·".to_owned(),
        |piece| format!("{} {}", owner_arrow(piece.owner), piece_code(piece)),
    );
    let piece_style = piece.map_or_else(
        || Style::default().fg(Color::DarkGray),
        |piece| Style::default().fg(player_color(piece.owner)).bold(),
    );
    frame.render_widget(
        Paragraph::new(Span::styled(content, piece_style))
            .alignment(Alignment::Center)
            .block(
                Block::bordered()
                    .border_type(border_type)
                    .border_style(Style::default().fg(border_color))
                    .title(Span::styled(
                        square.to_string(),
                        Style::default().fg(Color::DarkGray),
                    )),
            ),
        area,
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SquareHighlight {
    SelectedSource,
    FocusedDestination,
    Cursor,
    LegalDestination,
    LastDestination,
    LastSource,
    None,
}

fn square_highlight(app: &App, square: Square, now: Instant) -> SquareHighlight {
    if app.selected_source() == Some(square) {
        return SquareHighlight::SelectedSource;
    }
    if app.focused_ai_destination(now) == Some(square) {
        return SquareHighlight::FocusedDestination;
    }
    let cursor = app.board_cursor(square);
    if cursor && app.selection.is_some() {
        return SquareHighlight::Cursor;
    }
    if app.legal_destination(square) {
        return SquareHighlight::LegalDestination;
    }
    let last_action = app.last_action();
    if last_action.is_some_and(|action| action.destination() == square) {
        return SquareHighlight::LastDestination;
    }
    if last_action
        .is_some_and(|action| matches!(action, Action::Move { from, .. } if from == square))
    {
        return SquareHighlight::LastSource;
    }
    if cursor {
        return SquareHighlight::Cursor;
    }
    SquareHighlight::None
}

fn render_history(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let (actions, applied) = match &app.session {
        Session::Match(session) => (session.game.actions(), session.game.actions().len()),
        Session::Replay(session) => (session.replay.actions.as_slice(), session.ply),
    };
    let available_rows = usize::from(area.height.saturating_sub(3)).max(1);
    let (start, end) = visible_window(actions.len(), applied, available_rows);
    let rows = actions[start..end]
        .iter()
        .enumerate()
        .map(|(offset, action)| {
            let index = start + offset;
            let style = match index.cmp(&applied) {
                Ordering::Equal => Style::default().fg(Color::Yellow).bold(),
                Ordering::Less => Style::default().fg(Color::White),
                Ordering::Greater => Style::default().fg(Color::DarkGray),
            };
            Row::new([
                Cell::from(format!("{}.", index + 1)),
                Cell::from(action.to_string()),
            ])
            .style(style)
        });
    let table = Table::new(rows, [Constraint::Length(4), Constraint::Min(8)])
        .header(
            Row::new(["#", "Move"])
                .style(Style::default().fg(Color::Cyan).bold())
                .bottom_margin(0),
        )
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" Move history "),
        );
    frame.render_widget(table, area);
}

fn render_analysis(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let view = analysis_view(app);
    let title = view.perspective.map_or_else(
        || " Predictions ".to_owned(),
        |(player, controller)| {
            format!(
                " Predictions · {} · {} ",
                short_player_label(player),
                controller.label()
            )
        },
    );
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let [summary_area, table_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner);
    let summary = if let Some(wdl) = view.wdl {
        format!(
            "Root {:+.3}  W/D/L {:.1}%/{:.1}%/{:.1}%",
            view.root_value.unwrap_or(wdl[0] - wdl[2]),
            wdl[0] * 100.0,
            wdl[1] * 100.0,
            wdl[2] * 100.0
        )
    } else if let Some(value) = view.root_value {
        format!("Root {value:+.3}  W/D/L —")
    } else {
        "Root —  W/D/L —".to_owned()
    };
    frame.render_widget(
        Paragraph::new(summary).style(Style::default().fg(Color::DarkGray)),
        summary_area,
    );

    let Some(actions) = view.actions else {
        frame.render_widget(
            Paragraph::new(view.note)
                .style(Style::default().fg(Color::DarkGray).italic())
                .wrap(Wrap { trim: true }),
            table_area,
        );
        return;
    };
    let max_rows = usize::from(table_area.height.saturating_sub(1));
    let played_action = match &app.session {
        Session::Replay(session) => session.replay.actions.get(session.ply).copied(),
        Session::Match(_) => None,
    };
    let offset = app
        .analysis_offset
        .min(actions.len().saturating_sub(max_rows.max(1)));
    let rows = actions.iter().skip(offset).take(max_rows).map(|entry| {
        let style = if Some(entry.action) == played_action {
            Style::default().fg(Color::Yellow).bold()
        } else {
            Style::default()
        };
        Row::new([
            Cell::from(entry.action.to_string()),
            Cell::from(format!("{:.2}", entry.prior)),
            Cell::from(entry.visits.to_string()),
            Cell::from(format!("{:.2}", entry.visit_probability)),
            Cell::from(format!("{:+.2}", entry.q_value)),
        ])
        .style(style)
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(8),
            Constraint::Length(5),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(5),
        ],
    )
    .header(
        Row::new(["Move", "Prior", "Visits", "Policy", "Q"])
            .style(Style::default().fg(Color::Cyan).bold()),
    );
    frame.render_widget(table, table_area);
}

struct AnalysisView<'a> {
    perspective: Option<(Player, Controller)>,
    root_value: Option<f32>,
    wdl: Option<[f32; 3]>,
    actions: Option<&'a [ActionAnalysis]>,
    note: &'static str,
}

fn analysis_view(app: &App) -> AnalysisView<'_> {
    match &app.session {
        Session::Match(session) => session.predictions.as_ref().map_or(
            AnalysisView {
                perspective: Some((
                    session.game.position().side_to_move(),
                    session.controller(session.game.position().side_to_move()),
                )),
                root_value: None,
                wdl: None,
                actions: None,
                note: match session.ai.as_ref().map(|ai| &ai.state) {
                    None => "No model analysis runs in human/human mode.",
                    Some(AiState::Loading) => "Loading the latest champion…",
                    Some(AiState::Thinking {
                        purpose: SearchPurpose::HumanPrediction,
                        ..
                    }) => "Champion is analyzing the human position…",
                    Some(AiState::Thinking {
                        purpose: SearchPurpose::CpuMove,
                        ..
                    }) => "Champion is choosing the CPU move…",
                    Some(AiState::Failed(_)) => "Champion analysis is unavailable.",
                    Some(AiState::Idle | AiState::WaitingToPlay { .. }) => {
                        "Champion analysis is starting…"
                    }
                },
            },
            |predictions| AnalysisView {
                perspective: Some((predictions.player, session.controller(predictions.player))),
                root_value: predictions.root_value,
                wdl: predictions.wdl,
                actions: Some(&predictions.actions),
                note: "",
            },
        ),
        Session::Replay(session) => AnalysisView {
            perspective: None,
            root_value: None,
            wdl: None,
            actions: session.current_analyses(),
            note: if session.ply == session.replay.actions.len() {
                "End of replay."
            } else if session.replay.analyses.is_some() {
                "No analysis is stored for this position."
            } else {
                "This replay contains no prediction data."
            },
        },
    }
}

fn render_footer(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let help = match app.session {
        Session::Match(_) => "Q quit · N new · Arrows move · Enter play · Tab hand · Esc cancel",
        Session::Replay(_) => "Q quit · ←/→ moves · ↑/↓ analysis · Home/End or G/G bounds",
    };
    frame.render_widget(
        Paragraph::new(help)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray))
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title(" Controls "),
            ),
        area,
    );
}

fn visible_window(total: usize, applied: usize, capacity: usize) -> (usize, usize) {
    if total <= capacity {
        return (0, total);
    }
    let anchor = applied.min(total.saturating_sub(1));
    let start = anchor
        .saturating_sub(capacity / 2)
        .min(total.saturating_sub(capacity));
    (start, start + capacity)
}

const fn player_color(player: Player) -> Color {
    match player {
        Player::First => Color::Cyan,
        Player::Second => Color::Magenta,
    }
}

const fn player_label(player: Player) -> &'static str {
    match player {
        Player::First => "Player 1 · bottom",
        Player::Second => "Player 2 · top",
    }
}

const fn short_player_label(player: Player) -> &'static str {
    match player {
        Player::First => "P1 bottom",
        Player::Second => "P2 top",
    }
}

const fn owner_arrow(player: Player) -> &'static str {
    match player {
        Player::First => "▲",
        Player::Second => "▼",
    }
}

const fn hand_piece_code(piece: HandPiece) -> &'static str {
    match piece {
        HandPiece::Tanuki => "TA",
        HandPiece::Kitsune => "KI",
        HandPiece::Kodama => "KD",
    }
}

const fn piece_code(piece: Piece) -> &'static str {
    match piece.kind {
        PieceKind::Koropokkuru => "KO",
        PieceKind::Tanuki => "TA",
        PieceKind::Kitsune => "KI",
        PieceKind::Kodama => "KD",
        PieceKind::KodamaSamurai => "SA",
    }
}

fn outcome_text(outcome: Outcome) -> String {
    match outcome {
        Outcome::Ongoing => "Game in progress".to_owned(),
        Outcome::Win { player, reason } => {
            format!(
                "{} wins ({})",
                player_label(player),
                win_reason_text(reason)
            )
        }
        Outcome::Draw { reason } => format!("Draw ({})", draw_reason_text(reason)),
    }
}

const fn win_reason_text(reason: WinReason) -> &'static str {
    match reason {
        WinReason::KoropokkuruCaptured => "Koropokkuru captured",
        WinReason::KoropokkuruReachedGoal => "Koropokkuru reached the goal",
        WinReason::OpponentHasNoLegalAction => "opponent has no legal move",
    }
}

const fn draw_reason_text(reason: DrawReason) -> &'static str {
    match reason {
        DrawReason::ThreefoldRepetition => "threefold repetition",
    }
}

fn transition_message(transition: yokai::Transition) -> String {
    let mut message = format!(
        "{} plays {}",
        player_label(transition.player),
        transition.action
    );
    if let Some(captured) = transition.captured {
        let _ = write!(message, " · captures {}", PieceKindLabel(captured));
    }
    if transition.promoted {
        message.push_str(" · promotes to samurai");
    }
    if transition.outcome.is_terminal() {
        let _ = write!(message, " · {}", outcome_text(transition.outcome));
    }
    message
}

struct PieceKindLabel(PieceKind);

impl fmt::Display for PieceKindLabel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.0 {
            PieceKind::Koropokkuru => "Koropokkuru",
            PieceKind::Tanuki => "Tanuki",
            PieceKind::Kitsune => "Kitsune",
            PieceKind::Kodama => "Kodama",
            PieceKind::KodamaSamurai => "Kodama samurai",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ai::AiCommand;
    use super::*;
    use ratatui::{Terminal, backend::TestBackend};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, ratatui::crossterm::event::KeyModifiers::NONE)
    }

    #[test]
    fn human_match_selects_and_applies_only_a_legal_move() {
        let mut app = App::for_human_match();
        app.handle_key(key(KeyCode::Up));
        app.handle_key(key(KeyCode::Enter));
        assert_eq!(
            app.selection,
            Some(Selection::MoveFrom("b2".parse().unwrap()))
        );

        app.handle_key(key(KeyCode::Up));
        app.handle_key(key(KeyCode::Enter));

        assert_eq!(app.game().actions(), &["b2-b3".parse().unwrap()]);
        assert_eq!(app.game().position().side_to_move(), Player::Second);
        assert_eq!(app.selection, None);
    }

    #[test]
    fn illegal_destination_keeps_the_selected_piece() {
        let mut app = App::for_human_match();
        app.cursor = "b2".parse().unwrap();
        app.activate_board();
        app.cursor = "a2".parse().unwrap();
        app.activate_board();

        assert!(app.game().actions().is_empty());
        assert_eq!(
            app.selection,
            Some(Selection::MoveFrom("b2".parse().unwrap()))
        );
        assert!(app.notice.contains("Illegal"));
    }

    #[test]
    fn captured_piece_can_be_selected_from_the_hand_and_dropped() {
        let mut app = App::for_human_match();
        app.apply_action("b2-b3".parse().unwrap());
        app.apply_action("a4-a3".parse().unwrap());

        app.select_hand_piece(2);
        assert_eq!(app.selection, Some(Selection::Drop(HandPiece::Kodama)));
        app.cursor = "c2".parse().unwrap();
        let now = Instant::now();
        assert!(app.legal_destination("c2".parse().unwrap()));
        assert_eq!(
            square_highlight(&app, "c2".parse().unwrap(), now),
            SquareHighlight::Cursor
        );
        assert_eq!(
            square_highlight(&app, "a2".parse().unwrap(), now),
            SquareHighlight::LegalDestination
        );
        assert!(app.selected_hand_piece(Player::First, HandPiece::Kodama, 2));
        app.activate_board();

        assert_eq!(
            app.game().actions().last(),
            Some(&"kodama@c2".parse().unwrap())
        );
        assert_eq!(
            app.game().position().piece_at("c2".parse().unwrap()),
            Some(Piece::new(PieceKind::Kodama, Player::First))
        );
    }

    #[test]
    fn cpu_mode_places_the_human_at_the_bottom() {
        let (worker, _events, _commands) = AiWorker::stub();
        let app = App::for_cpu_match(worker);
        let Session::Match(session) = &app.session else {
            panic!("expected match session");
        };
        assert_eq!(session.controller(Player::First), Controller::Human);
        assert_eq!(session.controller(Player::Second), Controller::Cpu);
    }

    #[test]
    fn champion_predictions_are_computed_for_the_human_turn_without_playing() {
        let (worker, events, commands) = AiWorker::stub();
        let mut app = App::for_cpu_match(worker);
        let now = Instant::now();
        events
            .send(AiEvent::Ready {
                generation: 16,
                simulations: 400,
            })
            .unwrap();

        app.tick_at(now);
        match commands.try_recv().expect("human search request") {
            AiCommand::Search { request_id, game } => {
                assert_eq!(request_id, 0);
                assert!(game.actions().is_empty());
                assert_eq!(game.position().side_to_move(), Player::First);
            }
            _ => panic!("expected a search for the starting position"),
        }
        let human_action = app.game().legal_actions()[0];
        let analysis = ActionAnalysis {
            action: human_action,
            prior: 0.5,
            q_value: 0.25,
            visits: 400,
            visit_probability: 1.0,
        };
        events
            .send(AiEvent::SearchReady {
                request_id: 0,
                result: Box::new(SearchResult {
                    best_action: human_action,
                    selected_action: human_action,
                    root_value: 0.25,
                    policy: [0.0; yokai::POLICY_ACTIONS],
                    analysis: vec![analysis],
                }),
                search_started: now,
                search_finished: now + Duration::from_millis(20),
            })
            .unwrap();

        app.tick_at(now + Duration::from_millis(20));

        assert!(app.game().actions().is_empty());
        let view = analysis_view(&app);
        assert_eq!(view.actions, Some([analysis].as_slice()));
        assert_eq!(view.perspective, Some((Player::First, Controller::Human)));
        let Session::Match(session) = &app.session else {
            panic!("expected match session");
        };
        assert!(matches!(session.ai.as_ref().unwrap().state, AiState::Idle));
        assert!(commands.try_recv().is_err());
    }

    #[test]
    fn stale_human_predictions_are_ignored_after_the_human_moves() {
        let (worker, events, commands) = AiWorker::stub();
        let mut app = App::for_cpu_match(worker);
        let now = Instant::now();
        events
            .send(AiEvent::Ready {
                generation: 16,
                simulations: 400,
            })
            .unwrap();
        app.tick_at(now);
        assert!(matches!(
            commands.try_recv().expect("initial human search"),
            AiCommand::Search { request_id: 0, .. }
        ));

        let human_action = "b2-b3".parse().unwrap();
        app.apply_action(human_action);
        assert!(matches!(
            commands.try_recv().expect("human action advances the tree"),
            AiCommand::Advance { action, .. } if action == human_action
        ));
        assert!(matches!(
            commands.try_recv().expect("CPU search request"),
            AiCommand::Search { request_id: 1, .. }
        ));
        events
            .send(AiEvent::SearchReady {
                request_id: 0,
                result: Box::new(SearchResult {
                    best_action: human_action,
                    selected_action: human_action,
                    root_value: -0.5,
                    policy: [0.0; yokai::POLICY_ACTIONS],
                    analysis: Vec::new(),
                }),
                search_started: now,
                search_finished: now + Duration::from_millis(10),
            })
            .unwrap();

        app.tick_at(now + Duration::from_millis(10));

        assert!(analysis_view(&app).actions.is_none());
        let Session::Match(session) = &app.session else {
            panic!("expected match session");
        };
        assert!(matches!(
            session.ai.as_ref().unwrap().state,
            AiState::Thinking {
                request_id: 1,
                purpose: SearchPurpose::CpuMove,
                ..
            }
        ));
    }

    #[test]
    fn cpu_result_focuses_source_then_destination_before_being_applied() {
        let (worker, events, commands) = AiWorker::stub();
        let mut app = App::for_cpu_match(worker);
        let now = Instant::now();
        events
            .send(AiEvent::Ready {
                generation: 16,
                simulations: 400,
            })
            .unwrap();
        app.tick_at(now);
        assert!(matches!(
            commands.try_recv().expect("initial human search"),
            AiCommand::Search { request_id: 0, .. }
        ));

        let human_action = "b2-b3".parse().unwrap();
        app.apply_action(human_action);
        assert!(matches!(
            commands.try_recv().expect("human action advances the tree"),
            AiCommand::Advance { action, .. } if action == human_action
        ));
        assert!(matches!(
            commands.try_recv().expect("CPU search request"),
            AiCommand::Search { request_id: 1, .. }
        ));

        let cpu_action = app.game().legal_actions()[0];
        let analysis = ActionAnalysis {
            action: cpu_action,
            prior: 0.5,
            q_value: 0.25,
            visits: 400,
            visit_probability: 1.0,
        };
        events
            .send(AiEvent::SearchReady {
                request_id: 1,
                result: Box::new(SearchResult {
                    best_action: cpu_action,
                    selected_action: cpu_action,
                    root_value: 0.25,
                    policy: [0.0; yokai::POLICY_ACTIONS],
                    analysis: vec![analysis],
                }),
                search_started: now,
                search_finished: now + Duration::from_millis(20),
            })
            .unwrap();

        let result_received = now + Duration::from_millis(20);
        app.tick_at(result_received);
        assert_eq!(app.game().actions().len(), 1);
        assert_eq!(analysis_view(&app).actions, Some([analysis].as_slice()));
        let Action::Move { from, to } = cpu_action else {
            panic!("expected board move");
        };
        assert_eq!(app.selected_source(), Some(from));
        assert_eq!(app.focused_ai_destination(result_received), None);
        assert_eq!(
            square_highlight(&app, from, result_received),
            SquareHighlight::SelectedSource
        );
        assert_eq!(
            square_highlight(&app, to, result_received),
            SquareHighlight::LegalDestination
        );

        let destination_focus = result_received + AI_SOURCE_FOCUS_DURATION;
        app.tick_at(destination_focus);
        assert_eq!(app.game().actions().len(), 1);
        assert_eq!(app.focused_ai_destination(destination_focus), Some(to));
        assert_eq!(
            square_highlight(&app, to, destination_focus),
            SquareHighlight::FocusedDestination
        );

        app.tick_at(
            (result_received + AI_MOVE_DELAY)
                .checked_sub(Duration::from_millis(1))
                .unwrap(),
        );
        assert_eq!(app.game().actions().len(), 1);
        app.tick_at(result_received + AI_MOVE_DELAY);
        assert_eq!(
            app.game().actions(),
            &["b2-b3".parse().unwrap(), cpu_action]
        );
        assert!(app.notice.contains("CPU search 0.02s"));
        assert!(analysis_view(&app).actions.is_none());
        let Session::Match(session) = &app.session else {
            panic!("expected match session");
        };
        assert!(matches!(
            session.ai.as_ref().unwrap().state,
            AiState::Thinking {
                request_id: 2,
                purpose: SearchPurpose::HumanPrediction,
                ..
            }
        ));
        assert!(matches!(
            commands.try_recv().expect("CPU action advances the tree"),
            AiCommand::Advance { action, .. } if action == cpu_action
        ));
    }

    #[test]
    fn replay_seeking_reconstructs_positions_and_exposes_stored_analysis() {
        let mut game = Game::new(Player::First);
        let action: Action = "b2-b3".parse().unwrap();
        game.apply(action).unwrap();
        let analysis = ActionAnalysis {
            action,
            prior: 0.4,
            q_value: 0.2,
            visits: 12,
            visit_probability: 0.6,
        };
        let replay = Replay::from_game(&game, None).with_analyses(vec![vec![analysis]]);
        let mut app = App::for_replay(replay).expect("valid replay");

        assert_eq!(analysis_view(&app).actions, Some([analysis].as_slice()));
        app.seek_replay(1);
        assert_eq!(app.game().actions(), &[action]);
        assert!(analysis_view(&app).actions.is_none());
    }

    #[test]
    fn history_window_tracks_the_replay_cursor() {
        assert_eq!(visible_window(20, 0, 5), (0, 5));
        assert_eq!(visible_window(20, 10, 5), (8, 13));
        assert_eq!(visible_window(20, 20, 5), (15, 20));
    }

    #[test]
    fn minimum_supported_terminal_renders_every_primary_panel() {
        let app = App::for_human_match();
        let backend = TestBackend::new(MINIMUM_WIDTH, MINIMUM_HEIGHT);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        terminal.draw(|frame| render(frame, &app)).expect("render");

        let buffer = terminal.backend().buffer();
        let mut text = String::new();
        for y in 0..MINIMUM_HEIGHT {
            for x in 0..MINIMUM_WIDTH {
                text.push_str(buffer.cell((x, y)).expect("screen cell").symbol());
            }
            text.push('\n');
        }
        assert!(text.contains("Move history"));
        assert!(text.contains("Predictions"));
        assert!(text.contains("Player 1 · bottom"));
        assert!(text.contains("▲ KD"));
        assert!(text.contains("b2"));
    }
}
