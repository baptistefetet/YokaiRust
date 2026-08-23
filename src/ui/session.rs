//! Live-match and replay session state, including the CPU state machine.

use std::time::{Duration, Instant};

use yokai::{Action, ActionAnalysis, Game, HandPiece, Player, Replay, SearchResult, Square};

use super::ai::{AiEvent, AiWorker};
use super::render::{player_label, transition_message};
use super::{AI_MOVE_DELAY, AI_SOURCE_FOCUS_DURATION};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Controller {
    Human,
    Cpu,
}

impl Controller {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Cpu => "CPU",
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct PredictionSnapshot {
    pub(super) player: Player,
    pub(super) root_value: Option<f32>,
    pub(super) actions: Vec<ActionAnalysis>,
}

pub(super) struct MatchSession {
    pub(super) game: Game,
    pub(super) controllers: [Controller; 2],
    pub(super) predictions: Option<PredictionSnapshot>,
    pub(super) ai: Option<AiRuntime>,
}

impl MatchSession {
    pub(super) fn human_vs_human() -> Self {
        Self {
            game: Game::new(Player::First),
            controllers: [Controller::Human, Controller::Human],
            predictions: None,
            ai: None,
        }
    }

    pub(super) fn human_vs_cpu(worker: AiWorker) -> Self {
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

    pub(super) fn reset(&mut self) {
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

    pub(super) const fn controller(&self, player: Player) -> Controller {
        self.controllers[player.index()]
    }

    pub(super) fn request_cpu_move(&mut self, now: Instant) -> Result<(), String> {
        if self.game.outcome().is_terminal()
            || self.controller(self.game.position().side_to_move()) != Controller::Cpu
        {
            return Ok(());
        }
        self.request_search(now, SearchPurpose::CpuMove)
    }

    pub(super) fn request_human_predictions(&mut self, now: Instant) -> Result<(), String> {
        if self.game.outcome().is_terminal()
            || self.controller(self.game.position().side_to_move()) != Controller::Human
            || self.predictions.is_some()
            || !matches!(self.ai.as_ref().map(|ai| &ai.state), Some(AiState::Idle))
        {
            return Ok(());
        }
        self.request_search(now, SearchPurpose::HumanPrediction)
    }

    pub(super) fn request_search(
        &mut self,
        now: Instant,
        purpose: SearchPurpose,
    ) -> Result<(), String> {
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

    pub(super) fn tick(&mut self, now: Instant) -> Option<MatchUpdate> {
        let mut update = self.process_ai_events(now);
        if let Some(move_update) = self.apply_pending_cpu_move(now) {
            update = Some(move_update);
        }
        if let Err(message) = self.request_human_predictions(now) {
            update = Some(MatchUpdate::notice(format!("CPU error: {message}")));
        }
        update
    }

    pub(super) fn process_ai_events(&mut self, now: Instant) -> Option<MatchUpdate> {
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

    pub(super) fn process_ai_event(&mut self, event: AiEvent, now: Instant) -> Option<MatchUpdate> {
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
            } => {
                // A result is only consumed while its request is still
                // `Thinking`; anything stale or duplicated is ignored rather
                // than crashing the terminal UI.
                let purpose = ai.state.search_purpose(request_id)?;
                let player = self.game.position().side_to_move();
                let search_duration = search_finished.saturating_duration_since(search_started);
                self.predictions = Some(PredictionSnapshot {
                    player,
                    root_value: Some(result.root_value),
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
            AiEvent::Failed { .. } => None,
        }
    }

    pub(super) fn apply_pending_cpu_move(&mut self, now: Instant) -> Option<MatchUpdate> {
        let ai = self.ai.as_mut()?;
        if !matches!(ai.state, AiState::WaitingToPlay { apply_at, .. } if now >= apply_at) {
            return None;
        }
        let AiState::WaitingToPlay {
            result,
            search_duration,
            ..
        } = std::mem::replace(&mut ai.state, AiState::Idle)
        else {
            // Guarded by the `matches!` above; never panic inside the TUI.
            return None;
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

    pub(super) fn ai_status(&self, now: Instant) -> Option<String> {
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

pub(super) struct AiRuntime {
    pub(super) worker: AiWorker,
    pub(super) state: AiState,
    pub(super) generation: Option<u32>,
    pub(super) simulations: Option<u32>,
    pub(super) next_request_id: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SearchPurpose {
    HumanPrediction,
    CpuMove,
}

pub(super) enum AiState {
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
    pub(super) const fn matches_request(&self, expected: u64) -> bool {
        matches!(
            self,
            Self::Thinking { request_id, .. } | Self::WaitingToPlay { request_id, .. }
                if *request_id == expected
        )
    }

    pub(super) const fn search_purpose(&self, expected: u64) -> Option<SearchPurpose> {
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

    pub(super) fn pending_action(&self) -> Option<Action> {
        match self {
            Self::WaitingToPlay { result, .. } => Some(result.best_action),
            Self::Loading | Self::Idle | Self::Thinking { .. } | Self::Failed(_) => None,
        }
    }

    pub(super) fn focused_destination(&self, now: Instant) -> Option<Square> {
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

pub(super) struct MatchUpdate {
    pub(super) notice: String,
    pub(super) action: Option<Action>,
}

impl MatchUpdate {
    pub(super) fn notice(notice: String) -> Self {
        Self {
            notice,
            action: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ReplaySession {
    pub(super) replay: Replay,
    pub(super) game: Game,
    pub(super) ply: usize,
}

impl ReplaySession {
    pub(super) fn new(replay: Replay) -> Result<Self, yokai::ReplayError> {
        replay.to_game()?;
        Ok(Self {
            game: Game::new(replay.initial_player),
            replay,
            ply: 0,
        })
    }

    pub(super) fn seek(&mut self, target: usize) -> Result<(), yokai::MoveError> {
        let target = target.min(self.replay.actions.len());
        let mut game = Game::new(self.replay.initial_player);
        for &action in &self.replay.actions[..target] {
            game.apply(action)?;
        }
        self.game = game;
        self.ply = target;
        Ok(())
    }

    pub(super) fn current_analyses(&self) -> Option<&[ActionAnalysis]> {
        self.replay
            .analyses
            .as_ref()?
            .get(self.ply)
            .map(Vec::as_slice)
    }
}

pub(super) enum Session {
    Match(MatchSession),
    Replay(ReplaySession),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Focus {
    Board,
    Hand,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Selection {
    MoveFrom(Square),
    Drop(HandPiece),
}
