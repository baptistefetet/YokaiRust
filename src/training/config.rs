//! Human-readable training configuration and validation.

use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AlphaZeroNetworkConfig, ReplayBufferConfig, SelfPlayEvaluator, search::validate_draw_values,
};

/// Burn execution backend selected by the training configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Apple GPU acceleration through Metal/WGPU.
    Metal,
    /// Portable CPU execution, mainly for tests and debugging.
    Cpu,
}

/// Search, exploration and concurrency settings used to generate experience.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelfPlayConfig {
    /// Complete games generated before each optimization phase.
    pub games_per_generation: usize,
    /// Parallel game workers sharing the inference batching service.
    pub workers: usize,
    /// MCTS simulations performed for each regular move.
    pub simulations: u32,
    /// Leaf positions collected before one evaluator call inside one search.
    pub search_batch_size: usize,
    /// Safety limit preventing a faulty or cyclic game from running forever.
    pub max_game_plies: usize,
    /// Maximum number of positions combined in one backend inference call.
    pub inference_batch_size: usize,
    /// Maximum batching delay after the first inference request arrives.
    pub inference_wait_ms: u64,
    /// Opening plies during which visit counts are sampled instead of maximized.
    pub exploration_plies: usize,
    /// Visit-count sampling temperature during exploratory opening plies.
    pub exploration_temperature: f32,
    /// Visit-count sampling temperature after the exploratory opening.
    pub final_temperature: f32,
    /// Optional search-only penalty assigned to actions causing repetition.
    pub repetition_contempt: f32,
    /// Draw utility for the starter during self-play; the non-starter gets its negation.
    #[serde(default = "default_starter_draw_value")]
    pub starter_draw_value: f32,
    /// Fraction of games restarted from the recent visited-state archive.
    #[serde(default = "default_restart_fraction")]
    pub restart_fraction: f32,
    /// Optional larger MCTS budget for archive-restart trajectories.
    #[serde(default)]
    pub restart_simulations: Option<u32>,
    /// Optional generation-zero evaluator replacement.
    #[serde(default)]
    pub bootstrap: SelfPlayBootstrapConfig,
}

/// Optional evaluator used only before the first learned champion exists.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelfPlayBootstrapMode {
    /// Use the randomly initialized generation-zero neural network.
    #[default]
    Neural,
    /// Use random MCTS rollouts until a non-zero champion is accepted.
    RandomRolloutUntilFirstPromotion,
}

/// Settings for generation-zero self-play bootstrapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SelfPlayBootstrapConfig {
    /// Evaluator-selection policy.
    pub mode: SelfPlayBootstrapMode,
    /// Maximum rollout length before treating the leaf as neutral.
    pub rollout_max_plies: usize,
}

impl Default for SelfPlayBootstrapConfig {
    fn default() -> Self {
        Self {
            mode: SelfPlayBootstrapMode::Neural,
            rollout_max_plies: 512,
        }
    }
}

impl SelfPlayConfig {
    /// Selects bootstrap from the accepted source, never the attempt number.
    #[must_use]
    pub const fn evaluator_for_source_generation(
        &self,
        source_generation: u32,
    ) -> SelfPlayEvaluator {
        match (self.bootstrap.mode, source_generation) {
            (SelfPlayBootstrapMode::RandomRolloutUntilFirstPromotion, 0) => {
                SelfPlayEvaluator::RandomRollout {
                    max_plies: self.bootstrap.rollout_max_plies,
                }
            }
            _ => SelfPlayEvaluator::Neural,
        }
    }
}

/// Dataset, loss and Adam optimizer settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Number of optimizer updates performed after each self-play batch.
    pub steps_per_generation: usize,
    /// Emit training/validation metrics after this many optimizer updates.
    pub validation_interval_steps: usize,
    /// Number of sampled positions used by one gradient update.
    pub batch_size: usize,
    /// Initial learning rate, used until a configured champion milestone.
    pub learning_rate: f64,
    /// Optional lower rates selected from the accepted champion generation.
    #[serde(default)]
    pub learning_rate_schedule: Vec<LearningRateStage>,
    /// Coupled L2 regularization added to Adam's gradients before its moments.
    pub weight_decay: f32,
    /// Stable whole-game fraction reserved for metrics, never gradient updates.
    pub validation_fraction: f32,
    /// Whether each sampled example may be reflected left-to-right.
    pub mirror_augmentation: bool,
    /// Policy-loss multiplier for the non-starter's positions in drawn games.
    /// Value/WDL supervision always stays fully weighted.
    #[serde(default = "default_non_starter_draw_policy_weight")]
    pub non_starter_draw_policy_weight: f32,
    /// Multiplier for MSE between `P(win) - P(loss)` and the scalar result.
    /// The three-class WDL cross-entropy always remains fully weighted.
    #[serde(default)]
    pub scalar_value_loss_weight: f32,
    /// When set, train only on this many final positions from decisive games.
    /// `None` restores the regular `AlphaZero` dataset containing every position.
    pub terminal_window_plies: Option<usize>,
    /// Optional automatic bootstrap that expands a decisive-game tail before
    /// switching to the complete `AlphaZero` dataset.
    #[serde(default)]
    pub terminal_window_schedule: Option<TerminalWindowSchedule>,
    /// Retention limits for recent self-play games.
    pub replay_buffer: ReplayBufferConfig,
}

/// Temporary schedule that grows a decisive endgame-only training window.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerminalWindowSchedule {
    /// Number of final plies retained at the start of the schedule.
    pub initial_plies: usize,
    /// Multiplicative window growth applied after each generation.
    pub growth_factor: usize,
    /// Desired fraction in `[0, 1)` of training samples belonging to the decisive tail.
    /// Zero adds no extra tail samples.
    pub decisive_fraction: f32,
    /// This generation and all later ones use the complete replay buffer.
    pub full_dataset_generation: u32,
}

/// Learning-rate milestone keyed by the accepted source champion.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LearningRateStage {
    /// Use this rate when the optimization source is at least this generation.
    pub source_generation: u32,
    /// Adam learning rate selected at and after the milestone.
    pub learning_rate: f64,
}

impl OptimizationConfig {
    /// Selects the latest applicable learning-rate milestone.
    #[must_use]
    pub fn learning_rate_for_source_generation(&self, source_generation: u32) -> f64 {
        self.learning_rate_schedule
            .iter()
            .filter(|stage| stage.source_generation <= source_generation)
            .max_by_key(|stage| stage.source_generation)
            .map_or(self.learning_rate, |stage| stage.learning_rate)
    }

    /// Resolves the endgame window for a candidate generation.
    #[must_use]
    pub fn terminal_window_for_generation(&self, generation: u32) -> Option<usize> {
        let Some(schedule) = self.terminal_window_schedule else {
            return self.terminal_window_plies;
        };
        if generation >= schedule.full_dataset_generation {
            return None;
        }
        let exponent = generation.saturating_sub(1);
        Some(
            schedule
                .initial_plies
                .saturating_mul(schedule.growth_factor.saturating_pow(exponent)),
        )
    }
}

/// Fair model-comparison and productivity-diagnostic settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArenaConfig {
    /// Paired candidate-versus-champion strength games.
    pub games: usize,
    /// Parallel arena game workers.
    pub workers: usize,
    /// MCTS simulations per official arena move.
    pub simulations: u32,
    /// Leaf batch size inside each arena search.
    pub search_batch_size: usize,
    /// Maximum number of reproducible random opening plies. Each paired game
    /// shares the exact same opening before the networks exchange colors.
    #[serde(default = "default_arena_opening_plies")]
    pub opening_plies: usize,
    /// Minimum candidate score, counting a draw as one half, for promotion.
    /// A one-sided paired sign-flip test at 5% must also pass.
    pub score_threshold: f32,
    /// Noisy candidate self-play games used to catch exploration-only cycles.
    pub candidate_self_play_games: usize,
    /// Maximum noisy self-play draw rate allowed for promotion.
    pub max_candidate_self_play_draw_rate: f32,
}

/// Runtime storage roots; their contents are intentionally ignored by Git.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathsConfig {
    /// Directory containing versioned model generations and `latest` pointer.
    pub models: String,
    /// Directory containing games, replay buffer, reports and diagnostics.
    pub self_play: String,
}

/// Complete top-level training configuration loaded from TOML.
///
/// `config/training.toml` is the single source of truth for the active
/// experiment. There is deliberately no `Default` implementation: an
/// in-code default would inevitably drift away from the checked-in file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Root deterministic seed from which generation and game seeds derive.
    pub seed: u64,
    /// Hardware backend for both inference and optimization.
    pub backend: BackendKind,
    /// Serializable neural architecture dimensions.
    pub network: AlphaZeroNetworkConfig,
    /// Experience-generation settings.
    pub self_play: SelfPlayConfig,
    /// Dataset and gradient-update settings.
    pub optimization: OptimizationConfig,
    /// Candidate comparison and diagnostic settings.
    pub arena: ArenaConfig,
    /// Runtime artifact locations.
    pub paths: PathsConfig,
}

const fn default_arena_opening_plies() -> usize {
    4
}

const fn default_starter_draw_value() -> f32 {
    0.25
}

const fn default_restart_fraction() -> f32 {
    0.25
}

const fn default_non_starter_draw_policy_weight() -> f32 {
    1.0
}

impl TrainingConfig {
    /// Loads and validates a TOML training configuration, rejecting unknown keys.
    ///
    /// Strictness applies here, at the user-input boundary. JSON journals retain
    /// ordinary Serde compatibility with fields from older configurations.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingConfigError`] on I/O, TOML, or semantic errors.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TrainingConfigError> {
        let source = fs::read_to_string(path)?;
        let mut unknown = Vec::new();
        let config: Self =
            serde_ignored::deserialize(toml::de::Deserializer::parse(&source)?, |path| {
                unknown.push(path.to_string())
            })?;
        if !unknown.is_empty() {
            return Err(TrainingConfigError::UnknownFields(unknown.join(", ")));
        }
        config.validate()?;
        Ok(config)
    }

    /// Validates all bounds that would otherwise fail during a long run.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingConfigError::Invalid`] with a focused diagnostic.
    pub fn validate(&self) -> Result<(), TrainingConfigError> {
        for (positive, message) in [
            (self.network.filters > 0, "network.filters must be positive"),
            (
                self.network.residual_blocks > 0,
                "network.residual_blocks must be positive",
            ),
            (
                self.network.shared_hidden > 0,
                "network.shared_hidden must be positive",
            ),
            (
                self.network.value_hidden > 0,
                "network.value_hidden must be positive",
            ),
            (
                self.self_play.games_per_generation > 0,
                "self_play.games_per_generation must be positive",
            ),
            (
                self.self_play.workers > 0,
                "self_play.workers must be positive",
            ),
            (
                self.self_play.simulations > 0,
                "self_play.simulations must be positive",
            ),
            (
                self.self_play.search_batch_size > 0,
                "self_play.search_batch_size must be positive",
            ),
            (
                self.self_play.max_game_plies > 0,
                "self_play.max_game_plies must be positive",
            ),
            (
                self.self_play.inference_batch_size > 0,
                "self_play.inference_batch_size must be positive",
            ),
            (
                self.self_play.bootstrap.rollout_max_plies > 0,
                "self_play.bootstrap.rollout_max_plies must be positive",
            ),
            (
                self.optimization.steps_per_generation > 0,
                "optimization.steps_per_generation must be positive",
            ),
            (
                self.optimization.validation_interval_steps > 0,
                "optimization.validation_interval_steps must be positive",
            ),
            (
                self.optimization.batch_size > 0,
                "optimization.batch_size must be positive",
            ),
            (
                self.optimization.replay_buffer.max_games > 0,
                "optimization.replay_buffer.max_games must be positive",
            ),
            (
                self.optimization.replay_buffer.generations_to_keep > 0,
                "optimization.replay_buffer.generations_to_keep must be positive",
            ),
            (self.arena.workers > 0, "arena.workers must be positive"),
            (
                self.arena.simulations > 0,
                "arena.simulations must be positive",
            ),
            (
                self.arena.search_batch_size > 0,
                "arena.search_batch_size must be positive",
            ),
            (
                self.arena.candidate_self_play_games > 0,
                "arena.candidate_self_play_games must be positive",
            ),
        ] {
            ensure(positive, message)?;
        }
        ensure(
            self.self_play.exploration_temperature.is_finite()
                && self.self_play.exploration_temperature >= 0.0,
            "self_play.exploration_temperature must be finite and non-negative",
        )?;
        ensure(
            self.self_play.final_temperature.is_finite() && self.self_play.final_temperature >= 0.0,
            "self_play.final_temperature must be finite and non-negative",
        )?;
        validate_draw_values(
            self.self_play.repetition_contempt,
            self.self_play.starter_draw_value,
        )
        .map_err(TrainingConfigError::Invalid)?;
        ensure(
            self.self_play.restart_fraction.is_finite()
                && (0.0..=1.0).contains(&self.self_play.restart_fraction),
            "self_play.restart_fraction must be finite and in [0, 1]",
        )?;
        ensure(
            self.self_play
                .restart_simulations
                .is_none_or(|budget| budget >= self.self_play.simulations),
            "self_play.restart_simulations must be at least self_play.simulations",
        )?;
        let optimization = &self.optimization;
        ensure(
            optimization.validation_interval_steps <= optimization.steps_per_generation,
            "optimization.validation_interval_steps must not exceed optimization.steps_per_generation",
        )?;
        ensure(
            optimization.learning_rate.is_finite() && optimization.learning_rate > 0.0,
            "optimization.learning_rate must be finite and positive",
        )?;
        ensure(
            optimization.weight_decay.is_finite() && optimization.weight_decay >= 0.0,
            "optimization.weight_decay must be finite and non-negative",
        )?;
        let mut previous_generation = 0;
        let mut previous_learning_rate = optimization.learning_rate;
        for stage in &optimization.learning_rate_schedule {
            ensure(
                stage.source_generation > previous_generation,
                "optimization.learning_rate_schedule.source_generation must be positive and strictly increasing",
            )?;
            ensure(
                stage.learning_rate.is_finite()
                    && stage.learning_rate > 0.0
                    && stage.learning_rate < previous_learning_rate,
                "optimization.learning_rate_schedule.learning_rate must be finite, positive and strictly decreasing",
            )?;
            previous_generation = stage.source_generation;
            previous_learning_rate = stage.learning_rate;
        }
        ensure(
            optimization.validation_fraction.is_finite()
                && (0.0..1.0).contains(&optimization.validation_fraction),
            "optimization.validation_fraction must be finite and in [0, 1)",
        )?;
        ensure(
            optimization.non_starter_draw_policy_weight.is_finite()
                && (0.0..=1.0).contains(&optimization.non_starter_draw_policy_weight),
            "optimization.non_starter_draw_policy_weight must be finite and in [0, 1]",
        )?;
        ensure(
            optimization.scalar_value_loss_weight.is_finite()
                && (0.0..=1.0).contains(&optimization.scalar_value_loss_weight),
            "optimization.scalar_value_loss_weight must be finite and in [0, 1]",
        )?;
        validate_terminal_window(optimization)?;
        ensure(
            self.arena.games >= 2 && self.arena.games.is_multiple_of(2),
            "arena.games must be even and at least two",
        )?;
        ensure(
            self.arena.opening_plies < self.self_play.max_game_plies,
            "arena.opening_plies must be less than self_play.max_game_plies",
        )?;
        ensure(
            self.arena.score_threshold.is_finite()
                && (0.5..=1.0).contains(&self.arena.score_threshold),
            "arena.score_threshold must be finite and in [0.5, 1]",
        )?;
        ensure(
            self.arena.max_candidate_self_play_draw_rate.is_finite()
                && (0.0..=1.0).contains(&self.arena.max_candidate_self_play_draw_rate),
            "arena.max_candidate_self_play_draw_rate must be finite and in [0, 1]",
        )?;
        ensure(
            !self.paths.models.trim().is_empty(),
            "paths.models must not be empty",
        )?;
        ensure(
            !self.paths.self_play.trim().is_empty(),
            "paths.self_play must not be empty",
        )?;
        Ok(())
    }
}

fn ensure(condition: bool, message: &'static str) -> Result<(), TrainingConfigError> {
    if condition {
        Ok(())
    } else {
        Err(TrainingConfigError::Invalid(message))
    }
}

fn validate_terminal_window(optimization: &OptimizationConfig) -> Result<(), TrainingConfigError> {
    ensure(
        optimization.terminal_window_plies != Some(0),
        "optimization.terminal_window_plies must be positive when set",
    )?;
    if let Some(schedule) = optimization.terminal_window_schedule {
        ensure(
            optimization.terminal_window_plies.is_none(),
            "optimization.terminal_window_plies and terminal_window_schedule are mutually exclusive",
        )?;
        ensure(
            schedule.initial_plies > 0,
            "optimization.terminal_window_schedule.initial_plies must be positive",
        )?;
        ensure(
            schedule.growth_factor > 0,
            "optimization.terminal_window_schedule.growth_factor must be positive",
        )?;
        ensure(
            schedule.decisive_fraction.is_finite()
                && (0.0..1.0).contains(&schedule.decisive_fraction),
            "optimization.terminal_window_schedule.decisive_fraction must be finite and in [0, 1)",
        )?;
        ensure(
            schedule.full_dataset_generation >= 2,
            "optimization.terminal_window_schedule.full_dataset_generation must be at least two",
        )?;
    }
    Ok(())
}

/// Configuration loading, parsing and semantic-validation failures.
#[derive(Debug, Error)]
pub enum TrainingConfigError {
    /// Configuration file could not be read.
    #[error("training configuration I/O error: {0}")]
    Io(#[from] io::Error),
    /// Configuration text is not valid TOML for this schema.
    #[error("invalid training TOML: {0}")]
    Toml(#[from] toml::de::Error),
    /// The user-supplied TOML contains misspelled or unsupported keys.
    #[error("unknown training configuration field(s): {0}")]
    UnknownFields(String),
    /// Values parse correctly but violate a cross-field invariant.
    #[error("invalid training configuration: {0}")]
    Invalid(&'static str),
}
