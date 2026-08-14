//! Human-readable training configuration and validation.

use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AlphaZeroNetworkConfig, ReplayBufferConfig, SelfPlayEvaluator};

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
    /// L2-style `AdamW` regularization discouraging unnecessarily large weights.
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
    /// Desired fraction of training samples belonging to the decisive tail.
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
    pub score_threshold: f32,
    /// Noise-free candidate-versus-itself games used as a cycle diagnostic.
    pub mirror_games: usize,
    /// Diagnostic draw-rate reference for deterministic mirror games.
    pub max_mirror_draw_rate: f32,
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

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            backend: BackendKind::Metal,
            network: AlphaZeroNetworkConfig::new(),
            self_play: SelfPlayConfig {
                games_per_generation: 256,
                workers: 16,
                simulations: 200,
                search_batch_size: 8,
                max_game_plies: 512,
                inference_batch_size: 128,
                inference_wait_ms: 1,
                exploration_plies: 12,
                exploration_temperature: 1.0,
                final_temperature: 0.0,
                repetition_contempt: 0.0,
                starter_draw_value: default_starter_draw_value(),
                restart_fraction: default_restart_fraction(),
                restart_simulations: None,
                bootstrap: SelfPlayBootstrapConfig::default(),
            },
            optimization: OptimizationConfig {
                steps_per_generation: 400,
                validation_interval_steps: 100,
                batch_size: 256,
                learning_rate: 0.001,
                learning_rate_schedule: Vec::new(),
                weight_decay: 1.0e-4,
                validation_fraction: 0.1,
                mirror_augmentation: true,
                non_starter_draw_policy_weight: 1.0,
                scalar_value_loss_weight: 0.0,
                terminal_window_plies: None,
                terminal_window_schedule: None,
                replay_buffer: ReplayBufferConfig::default(),
            },
            arena: ArenaConfig {
                games: 200,
                workers: 128,
                simulations: 400,
                search_batch_size: 1,
                opening_plies: 4,
                score_threshold: 0.55,
                mirror_games: 4,
                max_mirror_draw_rate: 0.0,
                candidate_self_play_games: 64,
                max_candidate_self_play_draw_rate: 0.20,
            },
            paths: PathsConfig {
                models: "models".to_owned(),
                self_play: "data/self-play".to_owned(),
            },
        }
    }
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
    /// Loads and validates a TOML training configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingConfigError`] on I/O, TOML, or semantic errors.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TrainingConfigError> {
        let config: Self = toml::from_str(&fs::read_to_string(path)?)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates all bounds that would otherwise fail during a long run.
    ///
    /// # Errors
    ///
    /// Returns [`TrainingConfigError::Invalid`] with a focused diagnostic.
    #[allow(clippy::too_many_lines)]
    pub fn validate(&self) -> Result<(), TrainingConfigError> {
        if self.network.filters == 0
            || self.network.residual_blocks == 0
            || self.network.shared_hidden == 0
            || self.network.value_hidden == 0
        {
            return Err(TrainingConfigError::Invalid(
                "network dimensions must be positive",
            ));
        }
        if self.self_play.games_per_generation == 0
            || self.self_play.workers == 0
            || self.self_play.simulations == 0
            || self.self_play.search_batch_size == 0
            || self.self_play.max_game_plies == 0
            || self.self_play.inference_batch_size == 0
        {
            return Err(TrainingConfigError::Invalid(
                "self-play counts and limits must be positive",
            ));
        }
        for temperature in [
            self.self_play.exploration_temperature,
            self.self_play.final_temperature,
        ] {
            if !temperature.is_finite() || temperature < 0.0 {
                return Err(TrainingConfigError::Invalid(
                    "self-play temperatures must be finite and non-negative",
                ));
            }
        }
        if !self.self_play.repetition_contempt.is_finite()
            || !(0.0..=1.0).contains(&self.self_play.repetition_contempt)
        {
            return Err(TrainingConfigError::Invalid(
                "self-play repetition contempt must be finite and in [0, 1]",
            ));
        }
        if !self.self_play.starter_draw_value.is_finite()
            || !(0.0..1.0).contains(&self.self_play.starter_draw_value)
        {
            return Err(TrainingConfigError::Invalid(
                "self-play starter draw value must be finite and in [0, 1)",
            ));
        }
        if self.self_play.repetition_contempt > 0.0 && self.self_play.starter_draw_value > 0.0 {
            return Err(TrainingConfigError::Invalid(
                "self-play repetition contempt and starter draw value are mutually exclusive",
            ));
        }
        if !self.self_play.restart_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.self_play.restart_fraction)
        {
            return Err(TrainingConfigError::Invalid(
                "self-play restart fraction must be finite and in [0, 1]",
            ));
        }
        if self
            .self_play
            .restart_simulations
            .is_some_and(|simulations| simulations < self.self_play.simulations)
        {
            return Err(TrainingConfigError::Invalid(
                "restart simulations must be at least the regular self-play budget",
            ));
        }
        if self.self_play.bootstrap.rollout_max_plies == 0 {
            return Err(TrainingConfigError::Invalid(
                "self-play bootstrap rollout limit must be greater than zero",
            ));
        }
        let optimization = &self.optimization;
        if optimization.steps_per_generation == 0
            || optimization.validation_interval_steps == 0
            || optimization.validation_interval_steps > optimization.steps_per_generation
            || optimization.batch_size == 0
            || !optimization.learning_rate.is_finite()
            || optimization.learning_rate <= 0.0
            || !optimization.weight_decay.is_finite()
            || optimization.weight_decay < 0.0
        {
            return Err(TrainingConfigError::Invalid(
                "optimization steps, validation interval, learning rate and decay are invalid",
            ));
        }
        let mut previous_generation = 0;
        let mut previous_learning_rate = optimization.learning_rate;
        for stage in &optimization.learning_rate_schedule {
            if stage.source_generation <= previous_generation
                || !stage.learning_rate.is_finite()
                || stage.learning_rate <= 0.0
                || stage.learning_rate >= previous_learning_rate
            {
                return Err(TrainingConfigError::Invalid(
                    "learning-rate stages require strictly increasing source generations and strictly decreasing positive rates",
                ));
            }
            previous_generation = stage.source_generation;
            previous_learning_rate = stage.learning_rate;
        }
        if !optimization.validation_fraction.is_finite()
            || !(0.0..1.0).contains(&optimization.validation_fraction)
        {
            return Err(TrainingConfigError::Invalid(
                "validation fraction must be in [0, 1)",
            ));
        }
        if !optimization.non_starter_draw_policy_weight.is_finite()
            || !(0.0..=1.0).contains(&optimization.non_starter_draw_policy_weight)
        {
            return Err(TrainingConfigError::Invalid(
                "non-starter draw policy weight must be in [0, 1]",
            ));
        }
        if !optimization.scalar_value_loss_weight.is_finite()
            || !(0.0..=1.0).contains(&optimization.scalar_value_loss_weight)
        {
            return Err(TrainingConfigError::Invalid(
                "scalar value loss weight must be in [0, 1]",
            ));
        }
        validate_terminal_window(optimization)?;
        if optimization.replay_buffer.max_games == 0
            || optimization.replay_buffer.generations_to_keep == 0
        {
            return Err(TrainingConfigError::Invalid(
                "replay buffer limits must be positive",
            ));
        }
        if self.arena.games < 200
            || !self.arena.games.is_multiple_of(2)
            || self.arena.workers == 0
            || self.arena.simulations == 0
            || self.arena.search_batch_size == 0
            || self.arena.opening_plies >= self.self_play.max_game_plies
            || self.arena.mirror_games == 0
            || !self.arena.mirror_games.is_multiple_of(2)
            || self.arena.candidate_self_play_games == 0
            || !self.arena.score_threshold.is_finite()
            || !(0.5..=1.0).contains(&self.arena.score_threshold)
            || !self.arena.max_mirror_draw_rate.is_finite()
            || !(0.0..=1.0).contains(&self.arena.max_mirror_draw_rate)
            || !self.arena.max_candidate_self_play_draw_rate.is_finite()
            || !(0.0..=1.0).contains(&self.arena.max_candidate_self_play_draw_rate)
        {
            return Err(TrainingConfigError::Invalid(
                "arena requires at least 200 paired games, a bounded opening, a positive even mirror sample, simulations, and valid diagnostic thresholds",
            ));
        }
        if self.paths.models.trim().is_empty() || self.paths.self_play.trim().is_empty() {
            return Err(TrainingConfigError::Invalid(
                "model and self-play paths cannot be empty",
            ));
        }
        Ok(())
    }
}

fn validate_terminal_window(optimization: &OptimizationConfig) -> Result<(), TrainingConfigError> {
    if optimization.terminal_window_plies == Some(0) {
        return Err(TrainingConfigError::Invalid(
            "terminal training window must contain at least one ply",
        ));
    }
    if let Some(schedule) = optimization.terminal_window_schedule {
        if optimization.terminal_window_plies.is_some() {
            return Err(TrainingConfigError::Invalid(
                "static and scheduled terminal windows cannot both be configured",
            ));
        }
        if schedule.initial_plies == 0
            || schedule.growth_factor == 0
            || !schedule.decisive_fraction.is_finite()
            || !(0.0..1.0).contains(&schedule.decisive_fraction)
            || schedule.full_dataset_generation < 2
        {
            return Err(TrainingConfigError::Invalid(
                "terminal window schedule requires positive sizes, a decisive fraction in (0, 1), and a full-dataset generation of at least two",
            ));
        }
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
    /// Values parse correctly but violate a cross-field invariant.
    #[error("invalid training configuration: {0}")]
    Invalid(&'static str),
}
