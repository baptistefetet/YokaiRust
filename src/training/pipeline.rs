//! One recoverable `AlphaZero` generation from self-play through arena decision.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

use burn::{module::AutodiffModule, prelude::Backend, tensor::backend::AutodiffBackend};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AlphaZeroNetworkConfig, ArenaError, ArenaResult, EpochReport, InferenceService,
    InferenceServiceError, ModelMetadata, ModelStoreError, NetworkEvaluator, Outcome, Player,
    ReplayBuffer, ReplayBufferConfig, SelfPlayError, SelfPlayGame, TrainingConfig, TrainingReport,
    generate_self_play_with_progress, load_champion, next_generation, publish_champion,
    run_arena_with_progress, save_generation, train_candidate_with_progress,
};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GameOutcomeStats {
    pub first_wins: usize,
    pub second_wins: usize,
    pub draws: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    pub champion_generation: u32,
    pub candidate_generation: u32,
    pub generated_games: usize,
    pub buffer_games: usize,
    pub buffer_examples: usize,
    pub self_play_outcomes: GameOutcomeStats,
    pub training: TrainingReport,
    pub arena: ArenaResult,
}

impl GenerationReport {
    #[must_use]
    pub const fn promoted(&self) -> bool {
        self.arena.promoted
    }
}

/// Coarse-grained events emitted by a complete `AlphaZero` generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrainingProgress {
    GenerationStarted {
        champion_generation: u32,
        candidate_generation: u32,
    },
    SelfPlayStarted {
        games: usize,
        workers: usize,
        simulations: u32,
    },
    SelfPlayAdvanced {
        completed: usize,
        total: usize,
    },
    SelfPlayFinished {
        games: usize,
        examples: usize,
        outcomes: GameOutcomeStats,
    },
    DatasetReady {
        buffer_games: usize,
        training_examples: usize,
        validation_examples: usize,
    },
    TrainingStarted {
        epochs: usize,
        batch_size: usize,
    },
    EpochFinished {
        total_epochs: usize,
        report: EpochReport,
    },
    CandidateSaved {
        generation: u32,
    },
    ArenaStarted {
        games: usize,
        simulations: u32,
    },
    ArenaAdvanced {
        completed: usize,
        total: usize,
    },
    ArenaFinished(ArenaResult),
    ChampionPromoted {
        generation: u32,
    },
    CandidateRejected {
        generation: u32,
    },
}

/// Creates generation zero only when no champion pointer exists.
///
/// # Errors
///
/// Returns [`PipelineError`] for model storage or loading failures.
pub fn bootstrap_champion<B: Backend>(
    root: impl AsRef<Path>,
    architecture: AlphaZeroNetworkConfig,
    device: &B::Device,
) -> Result<ModelMetadata, PipelineError> {
    let root = root.as_ref();
    if root.join("champion").exists() {
        let (_, metadata) = load_champion::<B>(root, device)?;
        return Ok(metadata);
    }
    let generation = next_generation(root)?;
    let model = architecture.init::<B>(device);
    let metadata = ModelMetadata::new(generation, architecture);
    save_generation(root, &metadata, &model)?;
    publish_champion(root, generation)?;
    Ok(metadata)
}

/// Runs a complete candidate generation and atomically promotes it on success.
///
/// # Errors
///
/// Returns [`PipelineError`] without changing the champion pointer when any
/// pre-arena stage fails or when the candidate is rejected.
pub fn run_generation<B>(
    config: &TrainingConfig,
    buffer: &mut ReplayBuffer,
    device: &B::Device,
) -> Result<GenerationReport, PipelineError>
where
    B: AutodiffBackend<FloatElem = f32> + 'static,
    B::InnerBackend: Backend<FloatElem = f32> + 'static,
    NetworkEvaluator<B::InnerBackend>: Send,
{
    run_generation_with_progress::<B, _>(config, buffer, device, &|_| {})
}

/// Runs a complete candidate generation with coarse-grained progress events.
///
/// The callback may be invoked concurrently by self-play and arena workers.
/// Events for those phases are deliberately throttled to roughly twenty lines.
///
/// # Errors
///
/// Returns [`PipelineError`] under the same conditions as [`run_generation`].
#[allow(clippy::too_many_lines)]
pub fn run_generation_with_progress<B, F>(
    config: &TrainingConfig,
    buffer: &mut ReplayBuffer,
    device: &B::Device,
    progress: &F,
) -> Result<GenerationReport, PipelineError>
where
    B: AutodiffBackend<FloatElem = f32> + 'static,
    B::InnerBackend: Backend<FloatElem = f32> + 'static,
    NetworkEvaluator<B::InnerBackend>: Send,
    F: Fn(TrainingProgress) + Sync,
{
    config.validate().map_err(PipelineError::Configuration)?;
    let models_root = Path::new(&config.paths.models);
    let (_, champion_metadata) = load_champion::<B::InnerBackend>(models_root, device)?;
    let candidate_generation = next_generation(models_root)?;
    progress(TrainingProgress::GenerationStarted {
        champion_generation: champion_metadata.generation,
        candidate_generation,
    });

    progress(TrainingProgress::SelfPlayStarted {
        games: config.self_play.games_per_generation,
        workers: config.self_play.workers,
        simulations: config.self_play.simulations,
    });
    let (champion_for_self_play, _) = load_champion::<B::InnerBackend>(models_root, device)?;
    let self_play_service = InferenceService::start(
        NetworkEvaluator::new(champion_for_self_play, device.clone()),
        config.self_play.inference_batch_size,
        Duration::from_millis(config.self_play.inference_wait_ms),
    )?;
    let self_play_client = self_play_service.client();
    let games = generate_self_play_with_progress(
        &self_play_client,
        &config.self_play,
        champion_metadata.generation,
        config
            .seed
            .wrapping_add(u64::from(candidate_generation) << 32),
        &|completed, total| {
            if progress_checkpoint(completed, total) {
                progress(TrainingProgress::SelfPlayAdvanced { completed, total });
            }
        },
    )?;
    drop(self_play_service);
    let self_play_outcomes = outcome_stats(&games);
    let generated_examples = games.iter().map(|game| game.examples.len()).sum();
    progress(TrainingProgress::SelfPlayFinished {
        games: games.len(),
        examples: generated_examples,
        outcomes: self_play_outcomes,
    });
    save_self_play_generation(&config.paths.self_play, candidate_generation, &games)?;
    for game in &games {
        buffer.push(game.clone());
    }
    save_replay_buffer(
        Path::new(&config.paths.self_play).join("buffer.json"),
        buffer,
    )?;

    let split = buffer.split(config.optimization.validation_fraction, config.seed)?;
    let training_examples = split.training_examples(config.optimization.mirror_augmentation);
    let validation_examples = split.validation_examples();
    if training_examples.is_empty() {
        return Err(PipelineError::EmptyTrainingSet);
    }
    progress(TrainingProgress::DatasetReady {
        buffer_games: buffer.len(),
        training_examples: training_examples.len(),
        validation_examples: validation_examples.len(),
    });
    progress(TrainingProgress::TrainingStarted {
        epochs: config.optimization.epochs,
        batch_size: config.optimization.batch_size,
    });
    let (champion_for_training, _) = load_champion::<B>(models_root, device)?;
    let (candidate, training) = train_candidate_with_progress(
        champion_for_training,
        &training_examples,
        &validation_examples,
        &config.optimization,
        config.seed.wrapping_add(u64::from(candidate_generation)),
        device,
        &|report| {
            progress(TrainingProgress::EpochFinished {
                total_epochs: config.optimization.epochs,
                report,
            });
        },
    );
    let candidate = candidate.valid();
    let candidate_metadata =
        ModelMetadata::new(candidate_generation, champion_metadata.architecture.clone());
    save_generation(models_root, &candidate_metadata, &candidate)?;
    progress(TrainingProgress::CandidateSaved {
        generation: candidate_generation,
    });

    progress(TrainingProgress::ArenaStarted {
        games: config.arena.games,
        simulations: config.arena.simulations,
    });
    let (champion_for_arena, _) = load_champion::<B::InnerBackend>(models_root, device)?;
    let candidate_service = InferenceService::start(
        NetworkEvaluator::new(candidate, device.clone()),
        config.self_play.inference_batch_size,
        Duration::from_millis(config.self_play.inference_wait_ms),
    )?;
    let champion_service = InferenceService::start(
        NetworkEvaluator::new(champion_for_arena, device.clone()),
        config.self_play.inference_batch_size,
        Duration::from_millis(config.self_play.inference_wait_ms),
    )?;
    let candidate_client = candidate_service.client();
    let champion_client = champion_service.client();
    let arena = run_arena_with_progress(
        &candidate_client,
        &champion_client,
        &config.arena,
        config.self_play.workers,
        config.self_play.max_game_plies,
        config
            .seed
            .wrapping_add(u64::from(candidate_generation) << 40),
        &|completed, total| {
            if progress_checkpoint(completed, total) {
                progress(TrainingProgress::ArenaAdvanced { completed, total });
            }
        },
    )?;
    drop(candidate_service);
    drop(champion_service);
    progress(TrainingProgress::ArenaFinished(arena));
    if arena.promoted {
        publish_champion(models_root, candidate_generation)?;
        progress(TrainingProgress::ChampionPromoted {
            generation: candidate_generation,
        });
    } else {
        progress(TrainingProgress::CandidateRejected {
            generation: candidate_generation,
        });
    }

    Ok(GenerationReport {
        champion_generation: champion_metadata.generation,
        candidate_generation,
        generated_games: games.len(),
        buffer_games: buffer.len(),
        buffer_examples: buffer.example_count(),
        self_play_outcomes,
        training,
        arena,
    })
}

fn progress_checkpoint(completed: usize, total: usize) -> bool {
    let interval = total.div_ceil(20).max(1);
    completed == 1 || completed == total || completed.is_multiple_of(interval)
}

/// Atomically stores the replay buffer for generation-boundary resume.
///
/// # Errors
///
/// Returns [`PipelineError`] on serialization or I/O failures.
pub fn save_replay_buffer(
    path: impl AsRef<Path>,
    buffer: &ReplayBuffer,
) -> Result<(), PipelineError> {
    atomic_json_write(path.as_ref(), buffer)
}

/// Loads a replay buffer, or creates an empty one when the file is absent.
///
/// # Errors
///
/// Returns [`PipelineError`] for malformed data, I/O, or invalid limits.
pub fn load_replay_buffer(
    path: impl AsRef<Path>,
    config: ReplayBufferConfig,
) -> Result<ReplayBuffer, PipelineError> {
    match fs::read(path) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes)?),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            ReplayBuffer::new(config).map_err(PipelineError::TrainingData)
        }
        Err(error) => Err(error.into()),
    }
}

fn save_self_play_generation(
    root: impl AsRef<Path>,
    generation: u32,
    games: &[SelfPlayGame],
) -> Result<(), PipelineError> {
    atomic_json_write(
        &root
            .as_ref()
            .join(format!("generation-{generation:06}.json")),
        games,
    )
}

fn atomic_json_write<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<(), PipelineError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = temporary_path(path);
    fs::write(&temporary, serde_json::to_vec(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn temporary_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("training-data");
    path.with_file_name(format!(".{name}-{}.tmp", std::process::id()))
}

fn outcome_stats(games: &[SelfPlayGame]) -> GameOutcomeStats {
    let mut stats = GameOutcomeStats::default();
    for game in games {
        match game.outcome {
            Outcome::Win {
                player: Player::First,
                ..
            } => stats.first_wins += 1,
            Outcome::Win {
                player: Player::Second,
                ..
            } => stats.second_wins += 1,
            Outcome::Draw { .. } => stats.draws += 1,
            Outcome::Ongoing => {}
        }
    }
    stats
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error(transparent)]
    Configuration(crate::TrainingConfigError),
    #[error(transparent)]
    Model(#[from] ModelStoreError),
    #[error(transparent)]
    Inference(#[from] InferenceServiceError),
    #[error(transparent)]
    SelfPlay(#[from] SelfPlayError),
    #[error(transparent)]
    Arena(#[from] ArenaError),
    #[error(transparent)]
    TrainingData(#[from] crate::TrainingDataError),
    #[error("training split produced no examples")]
    EmptyTrainingSet,
    #[error("training pipeline I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("training pipeline JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
