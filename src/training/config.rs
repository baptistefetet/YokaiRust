//! Human-readable training configuration and validation.

use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{AlphaZeroNetworkConfig, ReplayBufferConfig};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Metal,
    Cpu,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelfPlayConfig {
    pub games_per_generation: usize,
    pub workers: usize,
    pub simulations: u32,
    pub search_batch_size: usize,
    pub max_game_plies: usize,
    pub inference_batch_size: usize,
    pub inference_wait_ms: u64,
    pub exploration_plies: usize,
    pub exploration_temperature: f32,
    pub final_temperature: f32,
    pub repetition_contempt: f32,
    /// Draw utility for the starter during self-play; the non-starter gets its negation.
    #[serde(default = "default_starter_draw_value")]
    pub starter_draw_value: f32,
    /// Fraction of games restarted near historical repetition failures.
    #[serde(default = "default_cycle_restart_fraction")]
    pub cycle_restart_fraction: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Number of optimizer updates performed after each self-play batch.
    pub steps_per_generation: usize,
    /// Emit training/validation metrics after this many optimizer updates.
    pub validation_interval_steps: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub validation_fraction: f32,
    pub mirror_augmentation: bool,
    /// Discounts policy imitation when the non-starter's drawn trajectory
    /// assigns MCTS visit mass to repetitions. Value/WDL supervision is kept.
    #[serde(default)]
    pub non_starter_draw_repetition_discount: f32,
    /// When set, train only on this many final positions from decisive games.
    /// `None` restores the regular `AlphaZero` dataset containing every position.
    pub terminal_window_plies: Option<usize>,
    /// Optional automatic bootstrap that expands a decisive-game tail before
    /// switching to the complete `AlphaZero` dataset.
    #[serde(default)]
    pub terminal_window_schedule: Option<TerminalWindowSchedule>,
    pub replay_buffer: ReplayBufferConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TerminalWindowSchedule {
    pub initial_plies: usize,
    pub growth_factor: usize,
    /// Desired fraction of training samples belonging to the decisive tail.
    pub decisive_fraction: f32,
    /// This generation and all later ones use the complete replay buffer.
    pub full_dataset_generation: u32,
}

impl OptimizationConfig {
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub games: usize,
    pub workers: usize,
    pub simulations: u32,
    pub search_batch_size: usize,
    /// Maximum number of reproducible random opening plies. Each paired game
    /// shares the exact same opening before the networks exchange colors.
    #[serde(default = "default_arena_opening_plies")]
    pub opening_plies: usize,
    pub score_threshold: f32,
    /// Noise-free candidate-versus-itself games used as a cycle diagnostic.
    pub mirror_games: usize,
    pub max_mirror_draw_rate: f32,
    /// Noisy candidate self-play games used to catch exploration-only cycles.
    pub candidate_self_play_games: usize,
    pub max_candidate_self_play_draw_rate: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathsConfig {
    pub models: String,
    pub self_play: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub seed: u64,
    pub backend: BackendKind,
    pub network: AlphaZeroNetworkConfig,
    pub self_play: SelfPlayConfig,
    pub optimization: OptimizationConfig,
    pub arena: ArenaConfig,
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
                cycle_restart_fraction: default_cycle_restart_fraction(),
            },
            optimization: OptimizationConfig {
                steps_per_generation: 400,
                validation_interval_steps: 100,
                batch_size: 256,
                learning_rate: 0.001,
                weight_decay: 1.0e-4,
                validation_fraction: 0.1,
                mirror_augmentation: true,
                non_starter_draw_repetition_discount: 0.0,
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

const fn default_cycle_restart_fraction() -> f32 {
    0.25
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
        if !self.self_play.cycle_restart_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.self_play.cycle_restart_fraction)
        {
            return Err(TrainingConfigError::Invalid(
                "self-play cycle restart fraction must be finite and in [0, 1]",
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
        if !optimization.validation_fraction.is_finite()
            || !(0.0..1.0).contains(&optimization.validation_fraction)
        {
            return Err(TrainingConfigError::Invalid(
                "validation fraction must be in [0, 1)",
            ));
        }
        if !optimization
            .non_starter_draw_repetition_discount
            .is_finite()
            || !(0.0..1.0).contains(&optimization.non_starter_draw_repetition_discount)
        {
            return Err(TrainingConfigError::Invalid(
                "non-starter draw repetition discount must be in [0, 1)",
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

#[derive(Debug, Error)]
pub enum TrainingConfigError {
    #[error("training configuration I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid training TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("invalid training configuration: {0}")]
    Invalid(&'static str),
}
