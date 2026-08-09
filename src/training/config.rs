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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub validation_fraction: f32,
    pub mirror_augmentation: bool,
    pub replay_buffer: ReplayBufferConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub games: usize,
    pub workers: usize,
    pub simulations: u32,
    pub search_batch_size: usize,
    pub promotion_score: f32,
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
                simulations: 400,
                search_batch_size: 8,
                max_game_plies: 512,
                inference_batch_size: 128,
                inference_wait_ms: 1,
                exploration_plies: 12,
                exploration_temperature: 1.0,
                final_temperature: 0.0,
            },
            optimization: OptimizationConfig {
                epochs: 10,
                batch_size: 256,
                learning_rate: 0.001,
                weight_decay: 1.0e-4,
                validation_fraction: 0.1,
                mirror_augmentation: true,
                replay_buffer: ReplayBufferConfig::default(),
            },
            arena: ArenaConfig {
                games: 200,
                workers: 128,
                simulations: 400,
                search_batch_size: 1,
                promotion_score: 0.55,
            },
            paths: PathsConfig {
                models: "models".to_owned(),
                self_play: "data/self-play".to_owned(),
            },
        }
    }
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
        let optimization = &self.optimization;
        if optimization.epochs == 0
            || optimization.batch_size == 0
            || !optimization.learning_rate.is_finite()
            || optimization.learning_rate <= 0.0
            || !optimization.weight_decay.is_finite()
            || optimization.weight_decay < 0.0
        {
            return Err(TrainingConfigError::Invalid(
                "optimization counts, learning rate and decay are invalid",
            ));
        }
        if !optimization.validation_fraction.is_finite()
            || !(0.0..1.0).contains(&optimization.validation_fraction)
        {
            return Err(TrainingConfigError::Invalid(
                "validation fraction must be in [0, 1)",
            ));
        }
        if optimization.replay_buffer.max_games == 0
            || optimization.replay_buffer.generations_to_keep == 0
        {
            return Err(TrainingConfigError::Invalid(
                "replay buffer limits must be positive",
            ));
        }
        if self.arena.games < 200
            || self.arena.workers == 0
            || self.arena.simulations == 0
            || self.arena.search_batch_size == 0
            || !self.arena.promotion_score.is_finite()
            || !(0.5..=1.0).contains(&self.arena.promotion_score)
        {
            return Err(TrainingConfigError::Invalid(
                "arena requires at least 200 games, simulations, and a score in [0.5, 1]",
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

#[derive(Debug, Error)]
pub enum TrainingConfigError {
    #[error("training configuration I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid training TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("invalid training configuration: {0}")]
    Invalid(&'static str),
}
