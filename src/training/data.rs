//! Validated self-play targets and a rolling, game-aware replay buffer.

use std::collections::VecDeque;

use rand::seq::SliceRandom;
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Action, Game, HISTORY_POSITIONS, Outcome, POLICY_ACTIONS, Player, PolicyIndex, Position,
    Replay, SearchResult,
};

const POLICY_TOLERANCE: f32 = 1.0e-4;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingExample {
    pub position: Position,
    pub repetition_count: u8,
    /// Runtime-only preceding states, reconstructed from the containing game.
    /// Keeping this out of JSON avoids duplicating history for every ply.
    #[serde(skip)]
    pub history: [Option<Position>; HISTORY_POSITIONS],
    #[serde(with = "policy_serde")]
    pub policy: [f32; POLICY_ACTIONS],
    /// Final game result from `position.side_to_move()`'s perspective.
    pub value: f32,
}

impl TrainingExample {
    #[must_use]
    pub fn mirrored(&self) -> Self {
        Self {
            position: self.position.mirrored_horizontally(),
            repetition_count: self.repetition_count,
            history: self
                .history
                .map(|position| position.map(Position::mirrored_horizontally)),
            policy: mirror_policy(&self.policy, self.position.side_to_move()),
            value: self.value,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelfPlayGame {
    pub generation: u32,
    pub seed: u64,
    pub outcome: Outcome,
    pub examples: Vec<TrainingExample>,
    /// Versioned game trace. Legacy buffers deserialize with no replay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<Replay>,
}

impl SelfPlayGame {
    /// Returns owned examples, optionally interleaving every horizontal mirror.
    #[must_use]
    pub fn augmented_examples(&self, mirror: bool) -> Vec<TrainingExample> {
        self.selected_examples(mirror, None)
    }

    /// Selects the complete game, or a tactical tail from a decisive game.
    ///
    /// A terminal window intentionally rejects draws: unlike a win, their last
    /// action is merely the action that repeated a position, not a forced tactic
    /// that the policy should learn to reproduce.
    #[must_use]
    pub fn selected_examples(
        &self,
        mirror: bool,
        terminal_window_plies: Option<usize>,
    ) -> Vec<TrainingExample> {
        let start = match terminal_window_plies {
            None => 0,
            Some(window) if matches!(self.outcome, Outcome::Win { .. }) => {
                self.examples.len().saturating_sub(window)
            }
            Some(_) => return Vec::new(),
        };
        let selected = (start..self.examples.len())
            .map(|index| self.example_with_history(index))
            .collect::<Vec<_>>();
        augment_examples(&selected, mirror)
    }

    fn example_with_history(&self, index: usize) -> TrainingExample {
        let mut example = self.examples[index].clone();
        example.history = std::array::from_fn(|offset| {
            index
                .checked_sub(offset + 1)
                .map(|previous| self.examples[previous].position)
        });
        example
    }
}

fn augment_examples(examples: &[TrainingExample], mirror: bool) -> Vec<TrainingExample> {
    if !mirror {
        return examples.to_vec();
    }
    let mut augmented = Vec::with_capacity(examples.len() * 2);
    for example in examples {
        augmented.push(example.clone());
        augmented.push(example.mirrored());
    }
    augmented
}

#[derive(Default)]
pub struct SelfPlayRecorder {
    pending: Vec<PendingExample>,
}

struct PendingExample {
    position: Position,
    repetition_count: u8,
    policy: [f32; POLICY_ACTIONS],
    action: Action,
}

impl SelfPlayRecorder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Records a non-terminal state and its complete MCTS policy.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingDataError`] for terminal states, invalid probability
    /// mass, or policy mass assigned to illegal actions.
    pub fn record(&mut self, game: &Game, search: &SearchResult) -> Result<(), TrainingDataError> {
        if game.outcome().is_terminal() {
            return Err(TrainingDataError::TerminalPolicyTarget);
        }
        validate_policy(game.position(), &search.policy)?;
        self.pending.push(PendingExample {
            position: *game.position(),
            repetition_count: game.current_repetition_count(),
            policy: search.policy,
            action: search.selected_action,
        });
        Ok(())
    }

    /// Assigns the final value to every recorded non-terminal state.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingDataError`] when the supplied result is not terminal
    /// or no policy target was recorded.
    pub fn finish(
        self,
        generation: u32,
        seed: u64,
        outcome: Outcome,
    ) -> Result<SelfPlayGame, TrainingDataError> {
        if !outcome.is_terminal() {
            return Err(TrainingDataError::NonTerminalGame);
        }
        if self.pending.is_empty() {
            return Err(TrainingDataError::EmptyGame);
        }
        let initial_position = self
            .pending
            .first()
            .map(|pending| pending.position)
            .ok_or(TrainingDataError::EmptyGame)?;
        let initial_player = initial_position.side_to_move();
        let actions = self
            .pending
            .iter()
            .map(|pending| pending.action)
            .collect::<Vec<_>>();
        let examples = self
            .pending
            .into_iter()
            .map(|pending| TrainingExample {
                position: pending.position,
                repetition_count: pending.repetition_count,
                history: [None; HISTORY_POSITIONS],
                policy: pending.policy,
                value: outcome_value(outcome, pending.position.side_to_move()),
            })
            .collect();
        Ok(SelfPlayGame {
            generation,
            seed,
            outcome,
            examples,
            replay: (initial_position == Position::initial(initial_player))
                .then(|| Replay::from_actions(initial_player, actions, outcome, Some(seed))),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ReplayBufferConfig {
    pub max_games: usize,
    pub generations_to_keep: u32,
}

impl Default for ReplayBufferConfig {
    fn default() -> Self {
        Self {
            max_games: 20_000,
            generations_to_keep: 20,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayBuffer {
    config: ReplayBufferConfig,
    games: VecDeque<SelfPlayGame>,
}

impl ReplayBuffer {
    /// Creates a rolling replay buffer.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingDataError`] when either retention limit is zero.
    pub fn new(config: ReplayBufferConfig) -> Result<Self, TrainingDataError> {
        if config.max_games == 0 || config.generations_to_keep == 0 {
            return Err(TrainingDataError::InvalidBufferConfiguration);
        }
        Ok(Self {
            config,
            games: VecDeque::new(),
        })
    }

    pub fn push(&mut self, game: SelfPlayGame) {
        let oldest_generation = game
            .generation
            .saturating_add(1)
            .saturating_sub(self.config.generations_to_keep);
        self.games
            .retain(|stored| stored.generation >= oldest_generation);
        self.games.push_back(game);
        while self.games.len() > self.config.max_games {
            self.games.pop_front();
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.games.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }

    #[must_use]
    pub fn contains(&self, generation: u32, seed: u64) -> bool {
        self.games
            .iter()
            .any(|game| game.generation == generation && game.seed == seed)
    }

    #[must_use]
    pub fn example_count(&self) -> usize {
        self.games.iter().map(|game| game.examples.len()).sum()
    }

    /// Splits whole games so positions from one game cannot leak across train
    /// and validation sets.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingDataError`] for a non-finite fraction outside `[0, 1]`.
    pub fn split(
        &self,
        validation_fraction: f32,
        seed: u64,
    ) -> Result<DatasetSplit, TrainingDataError> {
        if !validation_fraction.is_finite() || !(0.0..=1.0).contains(&validation_fraction) {
            return Err(TrainingDataError::InvalidValidationFraction);
        }
        let mut games = self.games.iter().cloned().collect::<Vec<_>>();
        games.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
        let validation_count = validation_game_count(games.len(), validation_fraction);
        let validation_games = games.split_off(games.len().saturating_sub(validation_count));
        Ok(DatasetSplit {
            training_games: games,
            validation_games,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DatasetSplit {
    pub training_games: Vec<SelfPlayGame>,
    pub validation_games: Vec<SelfPlayGame>,
}

impl DatasetSplit {
    #[must_use]
    pub fn training_examples(&self, mirror: bool) -> Vec<TrainingExample> {
        self.training_examples_with_window(mirror, None)
    }

    #[must_use]
    pub fn training_examples_with_window(
        &self,
        mirror: bool,
        terminal_window_plies: Option<usize>,
    ) -> Vec<TrainingExample> {
        self.training_games
            .iter()
            .flat_map(|game| game.selected_examples(mirror, terminal_window_plies))
            .collect()
    }

    #[must_use]
    pub fn validation_examples(&self) -> Vec<TrainingExample> {
        self.validation_examples_with_window(None)
    }

    #[must_use]
    pub fn validation_examples_with_window(
        &self,
        terminal_window_plies: Option<usize>,
    ) -> Vec<TrainingExample> {
        self.validation_games
            .iter()
            .flat_map(|game| game.selected_examples(false, terminal_window_plies))
            .collect()
    }

    #[must_use]
    pub fn selected_game_counts(&self, terminal_window_plies: Option<usize>) -> (usize, usize) {
        let is_selected = |game: &&SelfPlayGame| {
            terminal_window_plies.is_none() || matches!(game.outcome, Outcome::Win { .. })
        };
        (
            self.training_games.iter().filter(is_selected).count(),
            self.validation_games.iter().filter(is_selected).count(),
        )
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum TrainingDataError {
    #[error("terminal states cannot have policy targets")]
    TerminalPolicyTarget,
    #[error("self-play result must be terminal")]
    NonTerminalGame,
    #[error("a self-play game cannot contain zero training states")]
    EmptyGame,
    #[error("policy contains a non-finite or negative probability")]
    InvalidPolicyValue,
    #[error("policy probability mass must sum to one, got {0}")]
    InvalidPolicyMass(f32),
    #[error("policy assigns {0} probability mass to illegal actions")]
    IllegalPolicyMass(f32),
    #[error("replay buffer retention limits must be positive")]
    InvalidBufferConfiguration,
    #[error("validation fraction must be finite and between zero and one")]
    InvalidValidationFraction,
}

/// Mirrors a canonical policy horizontally while preserving invalid slots.
#[must_use]
pub fn mirror_policy(policy: &[f32; POLICY_ACTIONS], player: Player) -> [f32; POLICY_ACTIONS] {
    let mut mirrored = [0.0; POLICY_ACTIONS];
    for (raw_index, &probability) in policy.iter().enumerate() {
        let Ok(raw_index_u8) = u8::try_from(raw_index) else {
            continue;
        };
        let Some(index) = PolicyIndex::new(raw_index_u8) else {
            continue;
        };
        let Some(action) = Action::from_policy_index(index, player) else {
            continue;
        };
        let Some(mirrored_index) = action.mirrored_horizontally().policy_index(player) else {
            continue;
        };
        mirrored[mirrored_index.as_usize()] = probability;
    }
    mirrored
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn validation_game_count(game_count: usize, fraction: f32) -> usize {
    ((game_count as f32) * fraction).round() as usize
}

fn validate_policy(
    position: &Position,
    policy: &[f32; POLICY_ACTIONS],
) -> Result<(), TrainingDataError> {
    if policy
        .iter()
        .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(TrainingDataError::InvalidPolicyValue);
    }
    let total = policy.iter().sum::<f32>();
    if (total - 1.0).abs() > POLICY_TOLERANCE {
        return Err(TrainingDataError::InvalidPolicyMass(total));
    }
    let game = Game::from_position(*position);
    let legal_mass = game
        .legal_actions()
        .into_iter()
        .filter_map(|action| action.policy_index(position.side_to_move()))
        .map(|index| policy[index.as_usize()])
        .sum::<f32>();
    let illegal_mass = (total - legal_mass).max(0.0);
    if illegal_mass > POLICY_TOLERANCE {
        return Err(TrainingDataError::IllegalPolicyMass(illegal_mass));
    }
    Ok(())
}

fn outcome_value(outcome: Outcome, perspective: Player) -> f32 {
    match outcome {
        Outcome::Ongoing | Outcome::Draw { .. } => 0.0,
        Outcome::Win { player, .. } if player == perspective => 1.0,
        Outcome::Win { .. } => -1.0,
    }
}

mod policy_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

    use crate::POLICY_ACTIONS;

    pub fn serialize<S>(policy: &[f32; POLICY_ACTIONS], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        policy.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[f32; POLICY_ACTIONS], D::Error>
    where
        D: Deserializer<'de>,
    {
        Vec::<f32>::deserialize(deserializer)?
            .try_into()
            .map_err(|values: Vec<f32>| {
                D::Error::custom(format_args!(
                    "expected {POLICY_ACTIONS} policy values, got {}",
                    values.len()
                ))
            })
    }
}
