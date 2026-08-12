//! One recoverable `AlphaZero` candidate followed by guarded promotion.

use std::{
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

use burn::{module::AutodiffModule, prelude::Backend, tensor::backend::AutodiffBackend};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AlphaZeroNetworkConfig, AlphaZeroTrainingState, ArenaError, ArenaProgress, ArenaResult,
    DatasetDiagnostics, Evaluation, EvaluationError, EvaluationRequest, Evaluator, InferenceClient,
    InferenceService, InferenceServiceError, InferenceStats, ModelMetadata, ModelStoreError,
    NetworkEvaluator, Outcome, Player, ReplayBuffer, ReplayBufferConfig, ReplayError,
    SelfPlayError, SelfPlayEvaluator, SelfPlayGame, TrainingConfig, TrainingExample,
    TrainingReport, TrainingStepReport, dataset_diagnostics, generate_self_play_with_progress,
    generate_self_play_with_restarts_and_progress, load_champion, load_generation,
    load_training_generation, next_generation, planned_restart_count, publish_champion,
    run_arena_with_progress, save_generation, save_training_generation, train_state_with_progress,
};

/// Guard proving that rollout bootstrap never falls back to scalar-zero leaves.
#[derive(Clone, Copy)]
struct RolloutOnly;

impl Evaluator for RolloutOnly {
    fn evaluate_batch(
        &mut self,
        _requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Err(EvaluationError::Backend(
            "rollout bootstrap attempted neural inference".to_owned(),
        ))
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GameOutcomeStats {
    pub first_wins: usize,
    pub second_wins: usize,
    pub draws: usize,
    /// Wins by the side selected to make the first move.
    #[serde(default)]
    pub starter_wins: usize,
    /// Wins by the side that did not make the first move.
    #[serde(default)]
    pub non_starter_wins: usize,
    /// Wins from legacy self-play records without an initial-player replay.
    #[serde(default)]
    pub unclassified_wins: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationReport {
    /// Accepted checkpoint used as the official arena reference.
    pub source_generation: u32,
    /// Checkpoint that generated this generation's self-play games.
    #[serde(default)]
    pub self_play_source_generation: u32,
    /// Leaf evaluator used for this generation's persisted self-play.
    #[serde(default)]
    pub self_play_evaluator: SelfPlayEvaluator,
    pub candidate_generation: u32,
    pub generated_games: usize,
    #[serde(default)]
    pub restarted_games: usize,
    pub buffer_games: usize,
    pub buffer_examples: usize,
    #[serde(default)]
    pub terminal_window_plies: Option<usize>,
    #[serde(default)]
    pub terminal_extra_examples: usize,
    #[serde(default)]
    pub terminal_oversampling: bool,
    pub self_play_outcomes: GameOutcomeStats,
    #[serde(default)]
    pub initial_self_play_outcomes: GameOutcomeStats,
    #[serde(default)]
    pub restarted_self_play_outcomes: GameOutcomeStats,
    #[serde(default)]
    pub generated_dataset_diagnostics: DatasetDiagnostics,
    #[serde(default)]
    pub buffer_dataset_diagnostics: DatasetDiagnostics,
    pub training: TrainingReport,
    pub arena: ArenaResult,
    pub candidate_mirror: ArenaResult,
    pub candidate_self_play: GameOutcomeStats,
    #[serde(default)]
    pub promotion: PromotionDecision,
}

impl GenerationReport {
    #[must_use]
    pub const fn arena_threshold_reached(&self) -> bool {
        self.arena.threshold_reached
    }

    #[must_use]
    pub const fn promoted(&self) -> bool {
        self.promotion.promoted()
    }
}

/// Every independent condition that protects the published champion.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PromotionDecision {
    pub arena_passed: bool,
    pub mirror_draw_gate_passed: bool,
    pub exploratory_draw_gate_passed: bool,
}

impl PromotionDecision {
    #[must_use]
    pub const fn promoted(self) -> bool {
        self.arena_passed && self.mirror_draw_gate_passed && self.exploratory_draw_gate_passed
    }
}

/// Coarse-grained events emitted by a complete `AlphaZero` generation.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainingProgress {
    GenerationStarted {
        source_generation: u32,
        candidate_generation: u32,
    },
    SelfPlayStarted {
        evaluator: SelfPlayEvaluator,
        games: usize,
        workers: usize,
        simulations: u32,
        search_batch_size: usize,
        repetition_contempt: f32,
        starter_draw_value: f32,
        cycle_restart_simulations: u32,
        restart_archive: usize,
        planned_restarts: usize,
    },
    SelfPlayAdvanced {
        completed: usize,
        total: usize,
    },
    SelfPlayFinished {
        evaluator: SelfPlayEvaluator,
        games: usize,
        restarted_games: usize,
        examples: usize,
        outcomes: GameOutcomeStats,
        inference: InferenceStats,
    },
    SelfPlayResumed {
        evaluator: SelfPlayEvaluator,
        games: usize,
        examples: usize,
        restarted_games: usize,
    },
    DatasetReady {
        buffer_games: usize,
        training_games: usize,
        validation_games: usize,
        training_examples: usize,
        validation_examples: usize,
        terminal_window_plies: Option<usize>,
        terminal_extra_examples: usize,
        terminal_oversampling: bool,
    },
    TrainingStarted {
        steps: usize,
        batch_size: usize,
        learning_rate: f64,
        validation_interval_steps: usize,
        optimizer_resumed: bool,
    },
    TrainingAdvanced {
        total_steps: usize,
        report: TrainingStepReport,
    },
    TrainingFinished {
        completed_steps: usize,
    },
    CandidateSaved {
        generation: u32,
    },
    ArenaStarted {
        games: usize,
        workers: usize,
        simulations: u32,
        search_batch_size: usize,
        opening_plies: usize,
    },
    ArenaAdvanced {
        progress: ArenaProgress,
    },
    ArenaFinished {
        result: ArenaResult,
        candidate_inference: InferenceStats,
        reference_inference: InferenceStats,
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
        within_configured_limit: bool,
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
        within_configured_limit: bool,
    },
    ChampionPromoted {
        generation: u32,
    },
    CandidateRejected {
        generation: u32,
        decision: PromotionDecision,
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
    if root.join("latest").exists() || root.join("champion").exists() {
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

/// Backward-compatible name for [`bootstrap_champion`].
///
/// # Errors
///
/// Returns [`PipelineError`] under the same conditions as [`bootstrap_champion`].
pub fn bootstrap_latest<B: Backend>(
    root: impl AsRef<Path>,
    architecture: AlphaZeroNetworkConfig,
    device: &B::Device,
) -> Result<ModelMetadata, PipelineError> {
    bootstrap_champion::<B>(root, architecture, device)
}

/// Runs a complete candidate and atomically promotes it only on success.
///
/// # Errors
///
/// Official strength and both draw gates must pass before the champion changes.
/// A rejected candidate never becomes a training or self-play source.
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
    let (_, source_metadata) = load_champion::<B::InnerBackend>(models_root, device)?;
    let candidate_generation = next_generation(models_root)?;
    progress(TrainingProgress::GenerationStarted {
        source_generation: source_metadata.generation,
        candidate_generation,
    });
    let self_play_evaluator = config
        .self_play
        .evaluator_for_source_generation(source_metadata.generation);

    let persisted_games = load_self_play_generation(
        &config.paths.self_play,
        candidate_generation,
        source_metadata.generation,
        self_play_evaluator,
    )?;
    let games = if let Some(games) = persisted_games {
        let examples = games.iter().map(|game| game.examples.len()).sum();
        let restarted_games = games.iter().filter(|game| game.restart_ply > 0).count();
        progress(TrainingProgress::SelfPlayResumed {
            evaluator: self_play_evaluator,
            games: games.len(),
            examples,
            restarted_games,
        });
        games
    } else {
        let restart_archive = buffer.cycle_restart_replays()?;
        let planned_restarts = planned_restart_count(&config.self_play, restart_archive.len());
        progress(TrainingProgress::SelfPlayStarted {
            evaluator: self_play_evaluator,
            games: config.self_play.games_per_generation,
            workers: config.self_play.workers,
            simulations: config.self_play.simulations,
            search_batch_size: config.self_play.search_batch_size,
            repetition_contempt: config.self_play.repetition_contempt,
            starter_draw_value: config.self_play.starter_draw_value,
            cycle_restart_simulations: config
                .self_play
                .cycle_restart_simulations
                .unwrap_or(config.self_play.simulations),
            restart_archive: restart_archive.len(),
            planned_restarts,
        });
        let base_seed = config
            .seed
            .wrapping_add(u64::from(candidate_generation) << 32);
        let report_progress = |completed, total| {
            if progress_checkpoint(completed, total) {
                progress(TrainingProgress::SelfPlayAdvanced { completed, total });
            }
        };
        let (games, inference) = match self_play_evaluator {
            SelfPlayEvaluator::Neural => {
                let (source_for_self_play, _) = load_generation::<B::InnerBackend>(
                    models_root,
                    source_metadata.generation,
                    device,
                )?;
                let self_play_service = InferenceService::start_with_batching(
                    NetworkEvaluator::new(source_for_self_play, device.clone()),
                    config
                        .self_play
                        .workers
                        .saturating_mul(config.self_play.search_batch_size)
                        .min(config.self_play.inference_batch_size),
                    config.self_play.inference_batch_size,
                    Duration::from_millis(config.self_play.inference_wait_ms),
                )?;
                let games = generate_self_play_with_restarts_and_progress(
                    &self_play_service.client(),
                    &config.self_play,
                    source_metadata.generation,
                    base_seed,
                    &restart_archive,
                    &report_progress,
                )?;
                let inference = self_play_service.stats();
                drop(self_play_service);
                (games, inference)
            }
            SelfPlayEvaluator::RandomRollout { .. } => (
                generate_self_play_with_restarts_and_progress(
                    &RolloutOnly,
                    &config.self_play,
                    source_metadata.generation,
                    base_seed,
                    &restart_archive,
                    &report_progress,
                )?,
                InferenceStats::default(),
            ),
        };
        let outcomes = outcome_stats(&games);
        let examples = games.iter().map(|game| game.examples.len()).sum();
        let restarted_games = games.iter().filter(|game| game.restart_ply > 0).count();
        progress(TrainingProgress::SelfPlayFinished {
            evaluator: self_play_evaluator,
            games: games.len(),
            restarted_games,
            examples,
            outcomes,
            inference,
        });
        save_self_play_generation(&config.paths.self_play, candidate_generation, &games)?;
        save_self_play_replays(&config.paths.self_play, candidate_generation, &games)?;
        games
    };
    let self_play_outcomes = outcome_stats(&games);
    let initial_self_play_outcomes =
        outcome_stats(games.iter().filter(|game| game.restart_ply == 0));
    let restarted_self_play_outcomes =
        outcome_stats(games.iter().filter(|game| game.restart_ply > 0));
    let restarted_games = games.iter().filter(|game| game.restart_ply > 0).count();
    let generated_dataset_diagnostics = dataset_diagnostics(&games);
    for game in &games {
        if !buffer.contains(game.generation, game.seed) {
            buffer.push(game.clone());
        }
    }
    save_replay_buffer(
        Path::new(&config.paths.self_play).join("buffer.json"),
        buffer,
    )?;
    let buffer_dataset_diagnostics = buffer.diagnostics();

    let split = buffer.split(config.optimization.validation_fraction, config.seed)?;
    let terminal_window_plies = config
        .optimization
        .terminal_window_for_generation(candidate_generation);
    let terminal_oversampling =
        config.optimization.terminal_window_schedule.is_some() && terminal_window_plies.is_some();
    let (training_games, validation_games, mut training_examples, validation_examples) =
        if terminal_oversampling {
            let (training_games, validation_games) = split.selected_game_counts(None);
            (
                training_games,
                validation_games,
                split.training_examples(config.optimization.mirror_augmentation),
                split.validation_examples(),
            )
        } else {
            let (training_games, validation_games) =
                split.selected_game_counts(terminal_window_plies);
            (
                training_games,
                validation_games,
                split.training_examples_with_window(
                    config.optimization.mirror_augmentation,
                    terminal_window_plies,
                ),
                split.validation_examples_with_window(terminal_window_plies),
            )
        };
    let terminal_extra_examples = match (
        config.optimization.terminal_window_schedule,
        terminal_window_plies,
    ) {
        (Some(schedule), Some(window)) => {
            let tail = split.training_examples_with_window(
                config.optimization.mirror_augmentation,
                Some(window),
            );
            oversample_tail(&mut training_examples, &tail, schedule.decisive_fraction)
        }
        _ => 0,
    };
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
        terminal_extra_examples,
        terminal_oversampling,
    });
    let mut effective_optimization = config.optimization.clone();
    effective_optimization.learning_rate = config
        .optimization
        .learning_rate_for_source_generation(source_metadata.generation);
    let (training_state, optimizer_resumed) = if let Some((state, _)) = load_training_generation::<B>(
        models_root,
        source_metadata.generation,
        &effective_optimization,
        device,
    )? {
        (state, true)
    } else {
        let (source_for_training, _) =
            load_generation::<B>(models_root, source_metadata.generation, device)?;
        (
            AlphaZeroTrainingState::new(source_for_training, &effective_optimization),
            false,
        )
    };
    progress(TrainingProgress::TrainingStarted {
        steps: config.optimization.steps_per_generation,
        batch_size: config.optimization.batch_size,
        learning_rate: effective_optimization.learning_rate,
        validation_interval_steps: config.optimization.validation_interval_steps,
        optimizer_resumed,
    });
    let (training_state, training) = train_state_with_progress(
        training_state,
        &training_examples,
        &validation_examples,
        &effective_optimization,
        config.seed.wrapping_add(u64::from(candidate_generation)),
        device,
        &|report| {
            progress(TrainingProgress::TrainingAdvanced {
                total_steps: config.optimization.steps_per_generation,
                report,
            });
        },
    );
    progress(TrainingProgress::TrainingFinished {
        completed_steps: training.steps_completed,
    });
    let candidate_metadata =
        ModelMetadata::new(candidate_generation, source_metadata.architecture.clone());
    save_training_generation(models_root, &candidate_metadata, &training_state)?;
    let candidate = training_state.model.valid();
    progress(TrainingProgress::CandidateSaved {
        generation: candidate_generation,
    });
    progress(TrainingProgress::ArenaStarted {
        games: config.arena.games,
        workers: config.arena.workers,
        simulations: config.arena.simulations,
        search_batch_size: config.arena.search_batch_size,
        opening_plies: config.arena.opening_plies,
    });
    let (source_for_arena, _) =
        load_generation::<B::InnerBackend>(models_root, source_metadata.generation, device)?;
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
    let reference_service = InferenceService::start_with_batching(
        NetworkEvaluator::new(source_for_arena, device.clone()),
        arena_minimum_batch,
        config.self_play.inference_batch_size,
        Duration::from_millis(config.self_play.inference_wait_ms),
    )?;
    let candidate_client = candidate_service.client();
    let reference_client = reference_service.client();
    let arena = run_official_arena(
        &candidate_client,
        &reference_client,
        config,
        candidate_generation,
        progress,
    )?;
    let candidate_inference = candidate_service.stats();
    let reference_inference = reference_service.stats();
    progress(TrainingProgress::ArenaFinished {
        result: arena,
        candidate_inference,
        reference_inference,
    });
    let candidate_diagnostics =
        run_candidate_diagnostics(&candidate_client, config, candidate_generation, progress)?;
    drop(candidate_service);
    drop(reference_service);

    let promotion = promotion_decision(&arena, &candidate_diagnostics, config);
    if promotion.promoted() {
        publish_champion(models_root, candidate_generation)?;
        progress(TrainingProgress::ChampionPromoted {
            generation: candidate_generation,
        });
    } else {
        progress(TrainingProgress::CandidateRejected {
            generation: candidate_generation,
            decision: promotion,
        });
    }

    let report = GenerationReport {
        source_generation: source_metadata.generation,
        self_play_source_generation: source_metadata.generation,
        self_play_evaluator,
        candidate_generation,
        generated_games: games.len(),
        restarted_games,
        buffer_games: buffer.len(),
        buffer_examples: buffer.example_count(),
        terminal_window_plies,
        terminal_extra_examples,
        terminal_oversampling,
        self_play_outcomes,
        initial_self_play_outcomes,
        restarted_self_play_outcomes,
        generated_dataset_diagnostics,
        buffer_dataset_diagnostics,
        training,
        arena,
        candidate_mirror: candidate_diagnostics.mirror,
        candidate_self_play: candidate_diagnostics.exploratory,
        promotion,
    };
    save_generation_report(
        Path::new(&config.paths.self_play)
            .join("reports")
            .join(format!("generation-{candidate_generation:06}.json")),
        &report,
    )?;
    Ok(report)
}

struct CandidateDiagnostics {
    mirror: ArenaResult,
    exploratory: GameOutcomeStats,
}

fn promotion_decision(
    arena: &ArenaResult,
    diagnostics: &CandidateDiagnostics,
    config: &TrainingConfig,
) -> PromotionDecision {
    let mirror_games = diagnostics.mirror.candidate_wins
        + diagnostics.mirror.reference_wins
        + diagnostics.mirror.draws;
    let exploratory_games = diagnostics.exploratory.first_wins
        + diagnostics.exploratory.second_wins
        + diagnostics.exploratory.draws;
    let arena_passed = arena.threshold_reached;
    let mirror_draw_gate_passed =
        ratio(diagnostics.mirror.draws, mirror_games) <= config.arena.max_mirror_draw_rate;
    let exploratory_draw_gate_passed = ratio(diagnostics.exploratory.draws, exploratory_games)
        <= config.arena.max_candidate_self_play_draw_rate;
    PromotionDecision {
        arena_passed,
        mirror_draw_gate_passed,
        exploratory_draw_gate_passed,
    }
}

/// Plays the new network against its source network with official rules.
fn run_official_arena<F>(
    candidate: &InferenceClient,
    reference: &InferenceClient,
    config: &TrainingConfig,
    candidate_generation: u32,
    progress: &F,
) -> Result<ArenaResult, PipelineError>
where
    F: Fn(TrainingProgress) + Sync,
{
    Ok(run_arena_with_progress(
        candidate,
        reference,
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
    )?)
}

/// Measures deterministic and exploratory draw behavior before promotion.
fn run_candidate_diagnostics<F>(
    candidate: &InferenceClient,
    config: &TrainingConfig,
    candidate_generation: u32,
    progress: &F,
) -> Result<CandidateDiagnostics, PipelineError>
where
    F: Fn(TrainingProgress) + Sync,
{
    let mirror = run_mirror_diagnostic(candidate, config, candidate_generation, progress)?;
    let exploratory =
        run_exploratory_diagnostic(candidate, config, candidate_generation, progress)?;
    Ok(CandidateDiagnostics {
        mirror,
        exploratory,
    })
}

/// Checks deterministic candidate-versus-candidate repetition behavior.
fn run_mirror_diagnostic<F>(
    candidate: &InferenceClient,
    config: &TrainingConfig,
    candidate_generation: u32,
    progress: &F,
) -> Result<ArenaResult, PipelineError>
where
    F: Fn(TrainingProgress) + Sync,
{
    progress(TrainingProgress::CandidateMirrorStarted {
        games: config.arena.mirror_games,
        simulations: config.arena.simulations,
        max_draw_rate: config.arena.max_mirror_draw_rate,
    });
    let mirror_config = crate::ArenaConfig {
        games: config.arena.mirror_games,
        // Keep this diagnostic anchored to the real initial position. The
        // promotion arena above is the diversified strength measurement.
        opening_plies: 0,
        score_threshold: 1.0,
        ..config.arena.clone()
    };
    let result = run_arena_with_progress(
        candidate,
        candidate,
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
    let draw_rate = ratio(result.draws, config.arena.mirror_games);
    let within_configured_limit = draw_rate <= config.arena.max_mirror_draw_rate;
    progress(TrainingProgress::CandidateMirrorFinished {
        result,
        draw_rate,
        within_configured_limit,
    });
    Ok(result)
}

/// Checks repetition behavior under the actual noisy self-play settings.
fn run_exploratory_diagnostic<F>(
    candidate: &InferenceClient,
    config: &TrainingConfig,
    candidate_generation: u32,
    progress: &F,
) -> Result<GameOutcomeStats, PipelineError>
where
    F: Fn(TrainingProgress) + Sync,
{
    progress(TrainingProgress::CandidateSelfPlayStarted {
        games: config.arena.candidate_self_play_games,
        simulations: config.self_play.simulations,
        max_draw_rate: config.arena.max_candidate_self_play_draw_rate,
    });
    let mut probe_config = config.self_play.clone();
    probe_config.games_per_generation = config.arena.candidate_self_play_games;
    probe_config.workers = probe_config.workers.min(probe_config.games_per_generation);
    let probe_games = generate_self_play_with_progress(
        candidate,
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
    let within_configured_limit = draw_rate <= config.arena.max_candidate_self_play_draw_rate;
    progress(TrainingProgress::CandidateSelfPlayFinished {
        outcomes,
        draw_rate,
        within_configured_limit,
    });
    Ok(outcomes)
}

#[allow(clippy::cast_precision_loss)]
fn ratio(numerator: usize, denominator: usize) -> f32 {
    numerator as f32 / denominator as f32
}

#[allow(clippy::cast_precision_loss)]
fn oversample_tail(
    training: &mut Vec<TrainingExample>,
    tail: &[TrainingExample],
    desired_fraction: f32,
) -> usize {
    if tail.is_empty() {
        return 0;
    }
    let initial_len = training.len();
    let mut effective_tail = tail.len();
    let mut cursor = 0;
    while (effective_tail as f32) / (training.len() as f32) < desired_fraction {
        training.push(tail[cursor % tail.len()].clone());
        effective_tail += 1;
        cursor += 1;
    }
    training.len() - initial_len
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

fn save_generation_report(
    path: impl AsRef<Path>,
    report: &GenerationReport,
) -> Result<(), PipelineError> {
    atomic_json_write(path.as_ref(), report)
}

fn load_self_play_generation(
    root: impl AsRef<Path>,
    file_generation: u32,
    expected_source_generation: u32,
    expected_evaluator: SelfPlayEvaluator,
) -> Result<Option<Vec<SelfPlayGame>>, PipelineError> {
    let path = root
        .as_ref()
        .join(format!("generation-{file_generation:06}.json"));
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let games: Vec<SelfPlayGame> = serde_json::from_slice(&bytes)?;
    if games.is_empty()
        || games.iter().any(|game| {
            game.generation != expected_source_generation || game.evaluator != expected_evaluator
        })
    {
        return Err(PipelineError::InvalidPersistedSelfPlay {
            file_generation,
            expected_source_generation,
            expected_evaluator,
        });
    }
    Ok(Some(games))
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

fn outcome_stats<'a>(games: impl IntoIterator<Item = &'a SelfPlayGame>) -> GameOutcomeStats {
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
        if let Outcome::Win { player, .. } = game.outcome {
            match game.replay.as_ref().map(|replay| replay.initial_player) {
                Some(starter) if player == starter => stats.starter_wins += 1,
                Some(_) => stats.non_starter_wins += 1,
                None => stats.unclassified_wins += 1,
            }
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
    #[error(
        "persisted self-play generation {file_generation} is empty or was not produced by network {expected_source_generation} with {expected_evaluator:?}"
    )]
    InvalidPersistedSelfPlay {
        file_generation: u32,
        expected_source_generation: u32,
        expected_evaluator: SelfPlayEvaluator,
    },
    #[error("training pipeline I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("training pipeline JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
