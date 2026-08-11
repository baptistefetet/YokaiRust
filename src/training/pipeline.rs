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
    AlphaZeroNetworkConfig, ArenaError, ArenaProgress, ArenaResult, EpochReport, InferenceService,
    InferenceServiceError, InferenceStats, ModelMetadata, ModelStoreError, NetworkEvaluator,
    Outcome, Player, ReplayBuffer, ReplayBufferConfig, ReplayError, SelfPlayError, SelfPlayGame,
    TrainingConfig, TrainingReport, generate_self_play_with_progress, load_champion,
    next_generation, publish_champion, run_arena_with_progress, save_generation,
    train_candidate_with_progress,
};

pub const CURRICULUM_STATE_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CurriculumState {
    pub format_version: u32,
    pub phase_index: usize,
    pub promotions_in_phase: usize,
    pub champion_generation: u32,
}

impl CurriculumState {
    const fn new(champion_generation: u32) -> Self {
        Self {
            format_version: CURRICULUM_STATE_FORMAT_VERSION,
            phase_index: 0,
            promotions_in_phase: 0,
            champion_generation,
        }
    }

    fn reconcile(&mut self, champion_generation: u32, config: &TrainingConfig) {
        if self.champion_generation == champion_generation {
            return;
        }
        if champion_generation > self.champion_generation {
            self.record_promotion(champion_generation, config);
        } else {
            // A deliberate champion rollback must not pretend that a new model
            // passed the curriculum gate.
            self.champion_generation = champion_generation;
        }
    }

    fn record_promotion(&mut self, generation: u32, config: &TrainingConfig) -> bool {
        self.champion_generation = generation;
        let Some(phase) = config.curriculum.get(self.phase_index) else {
            return false;
        };
        self.promotions_in_phase = self.promotions_in_phase.saturating_add(1);
        if self.promotions_in_phase >= phase.promotions_required
            && self.phase_index + 1 < config.curriculum.len()
        {
            self.phase_index += 1;
            self.promotions_in_phase = 0;
            true
        } else {
            self.promotions_in_phase = self.promotions_in_phase.min(phase.promotions_required);
            false
        }
    }
}

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
    pub candidate_mirror: ArenaResult,
    pub candidate_self_play: Option<GameOutcomeStats>,
}

impl GenerationReport {
    #[must_use]
    pub const fn promoted(&self) -> bool {
        self.arena.promoted
    }
}

/// Coarse-grained events emitted by a complete `AlphaZero` generation.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainingProgress {
    CurriculumPhaseStarted {
        phase_index: usize,
        phase_count: usize,
        name: String,
        promotions_in_phase: usize,
        promotions_required: usize,
        simulations: u32,
        repetition_contempt: f32,
        terminal_window_plies: Option<usize>,
    },
    CurriculumAdvanced {
        phase_index: usize,
        phase_count: usize,
        name: String,
    },
    GenerationStarted {
        champion_generation: u32,
        candidate_generation: u32,
    },
    SelfPlayStarted {
        games: usize,
        workers: usize,
        simulations: u32,
        search_batch_size: usize,
        repetition_contempt: f32,
    },
    SelfPlayAdvanced {
        completed: usize,
        total: usize,
    },
    SelfPlayFinished {
        games: usize,
        examples: usize,
        outcomes: GameOutcomeStats,
        inference: InferenceStats,
    },
    DatasetReady {
        buffer_games: usize,
        training_games: usize,
        validation_games: usize,
        training_examples: usize,
        validation_examples: usize,
        terminal_window_plies: Option<usize>,
    },
    TrainingStarted {
        epochs: usize,
        batch_size: usize,
    },
    EpochFinished {
        total_epochs: usize,
        report: EpochReport,
    },
    TrainingFinished {
        completed_epochs: usize,
        selected_epoch: usize,
    },
    CandidateSaved {
        generation: u32,
    },
    ArenaStarted {
        games: usize,
        workers: usize,
        simulations: u32,
        search_batch_size: usize,
    },
    ArenaAdvanced {
        progress: ArenaProgress,
    },
    ArenaFinished {
        result: ArenaResult,
        candidate_inference: InferenceStats,
        champion_inference: InferenceStats,
    },
    CandidateMirrorStarted {
        games: usize,
        simulations: u32,
        max_draw_rate: f32,
    },
    CandidateMirrorAdvanced {
        progress: ArenaProgress,
    },
    CandidateMirrorFinished {
        result: ArenaResult,
        draw_rate: f32,
        gate_passed: bool,
        candidate_promoted: bool,
    },
    CandidateSelfPlayStarted {
        games: usize,
        simulations: u32,
        max_draw_rate: f32,
    },
    CandidateSelfPlayAdvanced {
        completed: usize,
        total: usize,
    },
    CandidateSelfPlayFinished {
        outcomes: GameOutcomeStats,
        draw_rate: f32,
        gate_passed: bool,
        candidate_promoted: bool,
    },
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
    let base_config = config;
    base_config
        .validate()
        .map_err(PipelineError::Configuration)?;
    let models_root = Path::new(&base_config.paths.models);
    let (_, champion_metadata) = load_champion::<B::InnerBackend>(models_root, device)?;
    let candidate_generation = next_generation(models_root)?;
    progress(TrainingProgress::GenerationStarted {
        champion_generation: champion_metadata.generation,
        candidate_generation,
    });

    let curriculum_path = Path::new(&base_config.paths.self_play).join("curriculum-state.json");
    let mut curriculum_state = if base_config.curriculum.is_empty() {
        None
    } else {
        let mut state = load_curriculum_state(&curriculum_path, champion_metadata.generation)?;
        if state.phase_index >= base_config.curriculum.len() {
            state.phase_index = base_config.curriculum.len() - 1;
            state.promotions_in_phase = 0;
        }
        state.reconcile(champion_metadata.generation, base_config);
        save_curriculum_state(&curriculum_path, &state)?;
        Some(state)
    };
    let mut effective_config = base_config.clone();
    if let Some(state) = &curriculum_state {
        let phase = &base_config.curriculum[state.phase_index];
        effective_config.self_play.simulations = phase.simulations;
        effective_config.self_play.repetition_contempt = phase.repetition_contempt;
        effective_config.optimization.terminal_window_plies = phase.terminal_window_plies;
        progress(TrainingProgress::CurriculumPhaseStarted {
            phase_index: state.phase_index,
            phase_count: base_config.curriculum.len(),
            name: phase.name.clone(),
            promotions_in_phase: state.promotions_in_phase,
            promotions_required: phase.promotions_required,
            simulations: phase.simulations,
            repetition_contempt: phase.repetition_contempt,
            terminal_window_plies: phase.terminal_window_plies,
        });
    }
    let config = &effective_config;

    progress(TrainingProgress::SelfPlayStarted {
        games: config.self_play.games_per_generation,
        workers: config.self_play.workers,
        simulations: config.self_play.simulations,
        search_batch_size: config.self_play.search_batch_size,
        repetition_contempt: config.self_play.repetition_contempt,
    });
    let (champion_for_self_play, _) = load_champion::<B::InnerBackend>(models_root, device)?;
    let self_play_service = InferenceService::start_with_batching(
        NetworkEvaluator::new(champion_for_self_play, device.clone()),
        config
            .self_play
            .workers
            .saturating_mul(config.self_play.search_batch_size)
            .min(config.self_play.inference_batch_size),
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
    let self_play_inference = self_play_service.stats();
    drop(self_play_service);
    let self_play_outcomes = outcome_stats(&games);
    let generated_examples = games.iter().map(|game| game.examples.len()).sum();
    progress(TrainingProgress::SelfPlayFinished {
        games: games.len(),
        examples: generated_examples,
        outcomes: self_play_outcomes,
        inference: self_play_inference,
    });
    save_self_play_generation(&config.paths.self_play, candidate_generation, &games)?;
    save_self_play_replays(&config.paths.self_play, candidate_generation, &games)?;
    for game in &games {
        buffer.push(game.clone());
    }
    save_replay_buffer(
        Path::new(&config.paths.self_play).join("buffer.json"),
        buffer,
    )?;

    let split = buffer.split(config.optimization.validation_fraction, config.seed)?;
    let terminal_window_plies = config.optimization.terminal_window_plies;
    let (training_games, validation_games) = split.selected_game_counts(terminal_window_plies);
    let training_examples = split.training_examples_with_curriculum(
        config.optimization.mirror_augmentation,
        terminal_window_plies,
    );
    let validation_examples = split.validation_examples_with_curriculum(terminal_window_plies);
    if training_examples.is_empty() {
        return Err(PipelineError::EmptyTrainingSet);
    }
    progress(TrainingProgress::DatasetReady {
        buffer_games: buffer.len(),
        training_games,
        validation_games,
        training_examples: training_examples.len(),
        validation_examples: validation_examples.len(),
        terminal_window_plies,
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
    progress(TrainingProgress::TrainingFinished {
        completed_epochs: training.epochs.len(),
        selected_epoch: training.selected_epoch,
    });
    let candidate = candidate.valid();
    let candidate_metadata =
        ModelMetadata::new(candidate_generation, champion_metadata.architecture.clone());
    save_generation(models_root, &candidate_metadata, &candidate)?;
    progress(TrainingProgress::CandidateSaved {
        generation: candidate_generation,
    });

    progress(TrainingProgress::ArenaStarted {
        games: config.arena.games,
        workers: config.arena.workers,
        simulations: config.arena.simulations,
        search_batch_size: config.arena.search_batch_size,
    });
    let (champion_for_arena, _) = load_champion::<B::InnerBackend>(models_root, device)?;
    let arena_minimum_batch = config
        .arena
        .workers
        .div_ceil(2)
        .saturating_mul(config.arena.search_batch_size)
        .min(config.self_play.inference_batch_size);
    let candidate_service = InferenceService::start_with_batching(
        NetworkEvaluator::new(candidate, device.clone()),
        arena_minimum_batch,
        config.self_play.inference_batch_size,
        Duration::from_millis(config.self_play.inference_wait_ms),
    )?;
    let champion_service = InferenceService::start_with_batching(
        NetworkEvaluator::new(champion_for_arena, device.clone()),
        arena_minimum_batch,
        config.self_play.inference_batch_size,
        Duration::from_millis(config.self_play.inference_wait_ms),
    )?;
    let candidate_client = candidate_service.client();
    let champion_client = champion_service.client();
    let mut arena = run_arena_with_progress(
        &candidate_client,
        &champion_client,
        &config.arena,
        config.arena.workers,
        config.self_play.max_game_plies,
        config
            .seed
            .wrapping_add(u64::from(candidate_generation) << 40),
        &|arena_progress| {
            if progress_checkpoint(arena_progress.completed, arena_progress.total) {
                progress(TrainingProgress::ArenaAdvanced {
                    progress: arena_progress,
                });
            }
        },
    )?;
    let candidate_inference = candidate_service.stats();
    let champion_inference = champion_service.stats();
    progress(TrainingProgress::ArenaFinished {
        result: arena,
        candidate_inference,
        champion_inference,
    });

    progress(TrainingProgress::CandidateMirrorStarted {
        games: config.arena.mirror_games,
        simulations: config.arena.simulations,
        max_draw_rate: config.arena.max_mirror_draw_rate,
    });
    let mirror_config = crate::ArenaConfig {
        games: config.arena.mirror_games,
        promotion_score: 1.0,
        ..config.arena.clone()
    };
    let candidate_mirror = run_arena_with_progress(
        &candidate_client,
        &candidate_client,
        &mirror_config,
        config.arena.workers.min(config.arena.mirror_games),
        config.self_play.max_game_plies,
        config
            .seed
            .wrapping_add(u64::from(candidate_generation) << 48),
        &|mirror_progress| {
            if progress_checkpoint(mirror_progress.completed, mirror_progress.total) {
                progress(TrainingProgress::CandidateMirrorAdvanced {
                    progress: mirror_progress,
                });
            }
        },
    )?;
    let mirror_draw_rate = ratio(candidate_mirror.draws, config.arena.mirror_games);
    let mirror_gate_passed = mirror_draw_rate <= config.arena.max_mirror_draw_rate;
    arena.promoted &= mirror_gate_passed;
    progress(TrainingProgress::CandidateMirrorFinished {
        result: candidate_mirror,
        draw_rate: mirror_draw_rate,
        gate_passed: mirror_gate_passed,
        candidate_promoted: arena.promoted,
    });
    let candidate_self_play = if arena.promoted {
        progress(TrainingProgress::CandidateSelfPlayStarted {
            games: config.arena.candidate_self_play_games,
            simulations: config.self_play.simulations,
            max_draw_rate: config.arena.max_candidate_self_play_draw_rate,
        });
        let mut probe_config = config.self_play.clone();
        probe_config.games_per_generation = config.arena.candidate_self_play_games;
        probe_config.workers = probe_config.workers.min(probe_config.games_per_generation);
        let probe_games = generate_self_play_with_progress(
            &candidate_client,
            &probe_config,
            candidate_generation,
            config
                .seed
                .wrapping_add(u64::from(candidate_generation) << 56),
            &|completed, total| {
                if progress_checkpoint(completed, total) {
                    progress(TrainingProgress::CandidateSelfPlayAdvanced { completed, total });
                }
            },
        )?;
        let outcomes = outcome_stats(&probe_games);
        let draw_rate = ratio(outcomes.draws, probe_games.len());
        let gate_passed = draw_rate <= config.arena.max_candidate_self_play_draw_rate;
        arena.promoted &= gate_passed;
        progress(TrainingProgress::CandidateSelfPlayFinished {
            outcomes,
            draw_rate,
            gate_passed,
            candidate_promoted: arena.promoted,
        });
        Some(outcomes)
    } else {
        None
    };
    drop(candidate_service);
    drop(champion_service);
    if arena.promoted {
        publish_champion(models_root, candidate_generation)?;
        if let Some(state) = &mut curriculum_state {
            let advanced = state.record_promotion(candidate_generation, base_config);
            save_curriculum_state(&curriculum_path, state)?;
            if advanced {
                let phase = &base_config.curriculum[state.phase_index];
                progress(TrainingProgress::CurriculumAdvanced {
                    phase_index: state.phase_index,
                    phase_count: base_config.curriculum.len(),
                    name: phase.name.clone(),
                });
            }
        }
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
        candidate_mirror,
        candidate_self_play,
    })
}

#[allow(clippy::cast_precision_loss)]
fn ratio(numerator: usize, denominator: usize) -> f32 {
    numerator as f32 / denominator as f32
}

fn progress_checkpoint(completed: usize, total: usize) -> bool {
    let interval = total.div_ceil(20).max(1);
    completed == 1 || completed == total || completed.is_multiple_of(interval)
}

/// Loads the persisted automatic curriculum, creating phase zero when absent.
///
/// # Errors
///
/// Returns [`PipelineError`] for malformed or unsupported state files.
pub fn load_curriculum_state(
    path: impl AsRef<Path>,
    champion_generation: u32,
) -> Result<CurriculumState, PipelineError> {
    match fs::read(path) {
        Ok(bytes) => {
            let state: CurriculumState = serde_json::from_slice(&bytes)?;
            if state.format_version != CURRICULUM_STATE_FORMAT_VERSION {
                return Err(PipelineError::UnsupportedCurriculumState(
                    state.format_version,
                ));
            }
            Ok(state)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            Ok(CurriculumState::new(champion_generation))
        }
        Err(error) => Err(error.into()),
    }
}

/// Atomically persists an automatic curriculum phase.
///
/// # Errors
///
/// Returns [`PipelineError`] on serialization or I/O failures.
pub fn save_curriculum_state(
    path: impl AsRef<Path>,
    state: &CurriculumState,
) -> Result<(), PipelineError> {
    atomic_json_write(path.as_ref(), state)
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

fn save_self_play_replays(
    root: impl AsRef<Path>,
    generation: u32,
    games: &[SelfPlayGame],
) -> Result<(), PipelineError> {
    let directory = root
        .as_ref()
        .join("replays")
        .join(format!("generation-{generation:06}"));
    fs::create_dir_all(&directory)?;
    for (index, game) in games.iter().enumerate() {
        let Some(replay) = &game.replay else {
            continue;
        };
        replay.to_game()?;
        atomic_json_write(&directory.join(format!("game-{index:04}.json")), replay)?;
    }
    Ok(())
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
    #[error(transparent)]
    Replay(#[from] ReplayError),
    #[error("training split produced no examples")]
    EmptyTrainingSet,
    #[error("unsupported curriculum state version {0}")]
    UnsupportedCurriculumState(u32),
    #[error("training pipeline I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("training pipeline JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
