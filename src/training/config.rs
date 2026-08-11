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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub epochs: usize,
    pub early_stopping_patience: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub validation_fraction: f32,
    pub mirror_augmentation: bool,
    /// When set, train only on this many final positions from decisive games.
    /// `None` restores the regular `AlphaZero` dataset containing every position.
    pub terminal_window_plies: Option<usize>,
    pub replay_buffer: ReplayBufferConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArenaConfig {
    pub games: usize,
    pub workers: usize,
    pub simulations: u32,
    pub search_batch_size: usize,
    pub promotion_score: f32,
    /// Noise-free candidate-versus-itself games used as an anti-cycle gate.
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
pub struct CurriculumPhaseConfig {
    pub name: String,
    pub promotions_required: usize,
    pub simulations: u32,
    pub repetition_contempt: f32,
    /// Missing means that this phase uses every buffered position, including
    /// draws. A finite window selects only the tail of decisive games.
    pub terminal_window_plies: Option<usize>,
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
    #[serde(default)]
    pub curriculum: Vec<CurriculumPhaseConfig>,
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
                // A smaller self-play budget is intentional while the network is
                // learning: deeper searches currently converge on its repetition
                // cycle, whereas 200 simulations generate far more decisive games.
                simulations: 200,
                search_batch_size: 8,
                max_game_plies: 512,
                inference_batch_size: 128,
                inference_wait_ms: 1,
                exploration_plies: 12,
                exploration_temperature: 0.0,
                final_temperature: 0.0,
                repetition_contempt: 0.0,
            },
            optimization: OptimizationConfig {
                epochs: 10,
                early_stopping_patience: 2,
                batch_size: 256,
                learning_rate: 0.001,
                weight_decay: 1.0e-4,
                validation_fraction: 0.1,
                mirror_augmentation: true,
                terminal_window_plies: Some(8),
                replay_buffer: ReplayBufferConfig::default(),
            },
            arena: ArenaConfig {
                games: 200,
                workers: 128,
                simulations: 400,
                search_batch_size: 1,
                promotion_score: 0.55,
                mirror_games: 64,
                max_mirror_draw_rate: 0.35,
                candidate_self_play_games: 64,
                max_candidate_self_play_draw_rate: 0.20,
            },
            paths: PathsConfig {
                models: "models".to_owned(),
                self_play: "data/self-play".to_owned(),
            },
            curriculum: default_curriculum(),
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
        if !self.self_play.repetition_contempt.is_finite()
            || !(0.0..=1.0).contains(&self.self_play.repetition_contempt)
        {
            return Err(TrainingConfigError::Invalid(
                "self-play repetition contempt must be finite and in [0, 1]",
            ));
        }
        let optimization = &self.optimization;
        if optimization.epochs == 0
            || optimization.early_stopping_patience == 0
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
        if optimization.terminal_window_plies == Some(0) {
            return Err(TrainingConfigError::Invalid(
                "terminal training window must contain at least one ply",
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
            || self.arena.mirror_games == 0
            || !self.arena.mirror_games.is_multiple_of(2)
            || self.arena.candidate_self_play_games == 0
            || !self.arena.promotion_score.is_finite()
            || !(0.5..=1.0).contains(&self.arena.promotion_score)
            || !self.arena.max_mirror_draw_rate.is_finite()
            || !(0.0..=1.0).contains(&self.arena.max_mirror_draw_rate)
            || !self.arena.max_candidate_self_play_draw_rate.is_finite()
            || !(0.0..=1.0).contains(&self.arena.max_candidate_self_play_draw_rate)
        {
            return Err(TrainingConfigError::Invalid(
                "arena requires at least 200 games, a positive even mirror sample, simulations, and valid score/draw thresholds",
            ));
        }
        if self.paths.models.trim().is_empty() || self.paths.self_play.trim().is_empty() {
            return Err(TrainingConfigError::Invalid(
                "model and self-play paths cannot be empty",
            ));
        }
        validate_curriculum(&self.curriculum)?;
        Ok(())
    }
}

fn validate_curriculum(phases: &[CurriculumPhaseConfig]) -> Result<(), TrainingConfigError> {
    for phase in phases {
        if phase.name.trim().is_empty()
            || phase.promotions_required == 0
            || phase.simulations == 0
            || !phase.repetition_contempt.is_finite()
            || !(0.0..=1.0).contains(&phase.repetition_contempt)
            || phase.terminal_window_plies == Some(0)
        {
            return Err(TrainingConfigError::Invalid(
                "curriculum phases require a name, positive counts, and valid contempt/window values",
            ));
        }
    }
    Ok(())
}

fn default_curriculum() -> Vec<CurriculumPhaseConfig> {
    vec![
        CurriculumPhaseConfig {
            name: "terminal-8".to_owned(),
            promotions_required: 1,
            simulations: 200,
            repetition_contempt: 0.0,
            terminal_window_plies: Some(8),
        },
        CurriculumPhaseConfig {
            name: "terminal-16".to_owned(),
            promotions_required: 1,
            simulations: 400,
            repetition_contempt: 0.5,
            terminal_window_plies: Some(16),
        },
        CurriculumPhaseConfig {
            name: "terminal-32".to_owned(),
            promotions_required: 2,
            simulations: 400,
            repetition_contempt: 0.5,
            terminal_window_plies: Some(32),
        },
        CurriculumPhaseConfig {
            name: "terminal-64".to_owned(),
            promotions_required: 2,
            simulations: 400,
            repetition_contempt: 0.5,
            terminal_window_plies: Some(64),
        },
        CurriculumPhaseConfig {
            name: "full-dataset".to_owned(),
            promotions_required: 1,
            simulations: 400,
            repetition_contempt: 0.5,
            terminal_window_plies: None,
        },
    ]
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
