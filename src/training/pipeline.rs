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

/// Official outcome counts, including absolute seat and starter-role views.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GameOutcomeStats {
    /// Games won by absolute [`Player::First`].
    pub first_wins: usize,
    /// Games won by absolute [`Player::Second`].
    pub second_wins: usize,
    /// Games ending by official repetition.
    pub draws: usize,
    /// Wins by the side selected to make the first move.
    #[serde(default)]
    pub starter_wins: usize,
    /// Wins by the side that did not make the first move.
    #[serde(default)]
    pub non_starter_wins: usize,
    /// Wins whose starter role could not be reconstructed from a replay.
    #[serde(default)]
    pub unclassified_wins: usize,
}

/// Persisted audit trail for every phase of one candidate attempt.
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
    /// Newly trained checkpoint identifier.
    pub candidate_generation: u32,
    /// Self-play trajectories generated or resumed for this attempt.
    pub generated_games: usize,
    /// Trajectories starting from a visited-state archive prefix.
    #[serde(default)]
    pub restarted_games: usize,
    /// Prefix occurrences available when restart slots were sampled.
    #[serde(default)]
    pub restart_archive_prefixes: usize,
    /// Shallowest sampled restart depth.
    #[serde(default)]
    pub restart_ply_min: Option<usize>,
    /// Deepest sampled restart depth.
    #[serde(default)]
    pub restart_ply_max: Option<usize>,
    /// Mean sampled restart depth.
    #[serde(default)]
    pub mean_restart_ply: f32,
    /// Complete games retained after adding this generation.
    pub buffer_games: usize,
    /// Nonterminal examples retained before augmentation.
    pub buffer_examples: usize,
    /// Optional decisive-tail length selected for this attempt.
    #[serde(default)]
    pub terminal_window_plies: Option<usize>,
    /// Extra duplicated tail examples added by scheduled oversampling.
    #[serde(default)]
    pub terminal_extra_examples: usize,
    /// Whether scheduled tail oversampling was active.
    #[serde(default)]
    pub terminal_oversampling: bool,
    /// Outcomes over all newly generated trajectories.
    pub self_play_outcomes: GameOutcomeStats,
    /// Outcomes over initial-position trajectories only.
    #[serde(default)]
    pub initial_self_play_outcomes: GameOutcomeStats,
    /// Outcomes over archive-restarted trajectories only.
    #[serde(default)]
    pub restarted_self_play_outcomes: GameOutcomeStats,
    /// Policy-target health for newly generated games.
    #[serde(default)]
    pub generated_dataset_diagnostics: DatasetDiagnostics,
    /// Policy-target health for the complete retained buffer.
    #[serde(default)]
    pub buffer_dataset_diagnostics: DatasetDiagnostics,
    /// Optimization losses and validation snapshots.
    pub training: TrainingReport,
    /// Official candidate-versus-champion comparison.
    pub arena: ArenaResult,
    /// Deterministic candidate-versus-itself diagnostic.
    pub candidate_mirror: ArenaResult,
    /// Outcomes from the noisy productivity probe.
    pub candidate_self_play: GameOutcomeStats,
    /// Individual checks and final publication decision.
    #[serde(default)]
    pub promotion: PromotionDecision,
}

impl GenerationReport {
    /// Reports whether the official strength arena passed.
    #[must_use]
    pub const fn arena_threshold_reached(&self) -> bool {
        self.arena.threshold_reached
    }

    /// Reports whether all promotion vetoes passed.
    #[must_use]
    pub const fn promoted(&self) -> bool {
        self.promotion.promoted()
    }
}

/// Strength and self-play-productivity checks for the published champion.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PromotionDecision {
    /// Candidate met the configured paired-arena score.
    pub arena_passed: bool,
    /// Whether the deterministic mirror draw rate stayed within its configured
    /// diagnostic limit. This is deliberately not a promotion veto: identical
    /// deterministic players can draw even when noisy self-play is productive.
    #[serde(default)]
    pub mirror_draw_limit_met: bool,
    /// Noisy self-play draw rate stayed below its productivity limit.
    pub exploratory_draw_gate_passed: bool,
}

impl PromotionDecision {
    /// Combines the two actual promotion gates.
    #[must_use]
    pub const fn promoted(self) -> bool {
        self.arena_passed && self.exploratory_draw_gate_passed
    }
}

/// Coarse-grained events emitted by a complete `AlphaZero` generation.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainingProgress {
    /// Candidate identifiers have been allocated and the attempt is beginning.
    GenerationStarted {
        /// Accepted checkpoint used as source and arena reference.
        source_generation: u32,
        /// New checkpoint identifier reserved for this attempt.
        candidate_generation: u32,
    },
    /// Self-play must be generated because no resumable file was found.
    SelfPlayStarted {
        /// Leaf evaluator selected for this source generation.
        evaluator: SelfPlayEvaluator,
        /// Number of scheduled trajectories.
        games: usize,
        /// Parallel game workers.
        workers: usize,
        /// Regular MCTS simulations per move.
        simulations: u32,
        /// Pending leaves evaluated together inside each search.
        search_batch_size: usize,
        /// Search-only immediate-repetition penalty.
        repetition_contempt: f32,
        /// Role-aware draw utility used in self-play search.
        starter_draw_value: f32,
        /// MCTS simulations per move on restarted trajectories.
        restart_simulations: u32,
        /// Prefix occurrences available to sample.
        restart_archive: usize,
        /// Trajectory slots assigned a restart.
        planned_restarts: usize,
    },
    /// A throttled snapshot of concurrent self-play completion.
    SelfPlayAdvanced {
        /// Finished trajectories.
        completed: usize,
        /// Total scheduled trajectories.
        total: usize,
    },
    /// Newly generated self-play has completed.
    SelfPlayFinished {
        /// Evaluator that produced the stored targets.
        evaluator: SelfPlayEvaluator,
        /// Completed trajectories.
        games: usize,
        /// Completed trajectories using a restart prefix.
        restarted_games: usize,
        /// Recorded nonterminal positions.
        examples: usize,
        /// Official results of generated games.
        outcomes: GameOutcomeStats,
        /// Shared inference batching and throughput measurements.
        inference: InferenceStats,
    },
    /// Previously completed self-play was validated and reused after resume.
    SelfPlayResumed {
        /// Evaluator recorded in the persisted file.
        evaluator: SelfPlayEvaluator,
        /// Persisted trajectories.
        games: usize,
        /// Persisted nonterminal positions.
        examples: usize,
        /// Persisted restarted trajectories.
        restarted_games: usize,
    },
    /// Replay-buffer partition and optional endgame sampling are ready.
    DatasetReady {
        /// Complete games retained in the rolling buffer.
        buffer_games: usize,
        /// Games contributing optimizer examples.
        training_games: usize,
        /// Games reserved for validation.
        validation_games: usize,
        /// Training positions after selection and augmentation.
        training_examples: usize,
        /// Unaugmented validation positions.
        validation_examples: usize,
        /// Decisive-tail length, or the full dataset when absent.
        terminal_window_plies: Option<usize>,
        /// Extra examples introduced by scheduled tail oversampling.
        terminal_extra_examples: usize,
        /// Whether tail examples augment rather than replace the full set.
        terminal_oversampling: bool,
    },
    /// Fixed-budget Adam optimization is beginning.
    TrainingStarted {
        /// Gradient updates scheduled.
        steps: usize,
        /// Sampled examples per update.
        batch_size: usize,
        /// Effective learning rate selected from champion milestones.
        learning_rate: f64,
        /// Updates between metric snapshots.
        validation_interval_steps: usize,
        /// Whether Adam moments were restored with the source checkpoint.
        optimizer_resumed: bool,
    },
    /// One periodic training/validation checkpoint is available.
    TrainingAdvanced {
        /// Total updates scheduled for the attempt.
        total_steps: usize,
        /// Metrics through the reported update.
        report: TrainingStepReport,
    },
    /// All optimizer updates completed.
    TrainingFinished {
        /// Updates actually applied.
        completed_steps: usize,
    },
    /// Candidate weights and resumable optimizer state are durable.
    CandidateSaved {
        /// Saved generation identifier.
        generation: u32,
    },
    /// Official paired strength comparison is beginning.
    ArenaStarted {
        /// Paired games scheduled.
        games: usize,
        /// Parallel match workers.
        workers: usize,
        /// MCTS simulations per move.
        simulations: u32,
        /// Pending leaves per search evaluation.
        search_batch_size: usize,
        /// Maximum shared random opening length.
        opening_plies: usize,
    },
    /// A throttled paired-arena snapshot is available.
    ArenaAdvanced {
        /// Current match totals.
        progress: ArenaProgress,
    },
    /// Official strength comparison completed.
    ArenaFinished {
        /// Candidate/reference outcomes and threshold status.
        result: ArenaResult,
        /// Candidate backend throughput and batching statistics.
        candidate_inference: InferenceStats,
        /// Reference backend throughput and batching statistics.
        reference_inference: InferenceStats,
    },
    /// Deterministic candidate-versus-itself diagnostic is beginning.
    CandidateMirrorStarted {
        /// Mirror games scheduled.
        games: usize,
        /// MCTS simulations per move.
        simulations: u32,
        /// Configured diagnostic draw-rate reference.
        max_draw_rate: f32,
    },
    /// A throttled deterministic-mirror snapshot is available.
    CandidateMirrorAdvanced {
        /// Current mirror totals.
        progress: ArenaProgress,
    },
    /// Deterministic mirror diagnostic completed.
    CandidateMirrorFinished {
        /// Detailed mirror result.
        result: ArenaResult,
        /// Observed official draw fraction.
        draw_rate: f32,
        /// Whether the diagnostic reference was met; this is not a veto.
        within_configured_limit: bool,
    },
    /// Noisy candidate self-play productivity probe is beginning.
    CandidateSelfPlayStarted {
        /// Probe trajectories scheduled.
        games: usize,
        /// MCTS simulations per move.
        simulations: u32,
        /// Maximum draw rate allowed by the promotion gate.
        max_draw_rate: f32,
    },
    /// A throttled productivity-probe snapshot is available.
    CandidateSelfPlayAdvanced {
        /// Finished probe trajectories.
        completed: usize,
        /// Total scheduled probe trajectories.
        total: usize,
    },
    /// Noisy productivity probe completed.
    CandidateSelfPlayFinished {
        /// Official outcomes of probe trajectories.
        outcomes: GameOutcomeStats,
        /// Observed draw fraction.
        draw_rate: f32,
        /// Whether the promotion draw gate passed.
        within_configured_limit: bool,
    },
    /// Candidate atomically became the accepted `latest` champion.
    ChampionPromoted {
        /// Published generation identifier.
        generation: u32,
    },
    /// Candidate remains stored for diagnosis but was not published.
    CandidateRejected {
        /// Rejected generation identifier.
        generation: u32,
        /// Individual gate results explaining the rejection.
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
    if root.join("latest").exists() {
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

/// Runs a complete candidate and atomically promotes it only on success.
///
/// # Errors
///
/// Official strength and the exploratory draw gate must pass before the
/// champion changes. Deterministic mirror draws remain diagnostic.
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

    let restart_archive = buffer.visited_restart_replays()?;
    let restart_archive_prefixes = restart_archive.len();
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
        let planned_restarts = planned_restart_count(&config.self_play, restart_archive.len());
        progress(TrainingProgress::SelfPlayStarted {
            evaluator: self_play_evaluator,
            games: config.self_play.games_per_generation,
            workers: config.self_play.workers,
            simulations: config.self_play.simulations,
            search_batch_size: config.self_play.search_batch_size,
            repetition_contempt: config.self_play.repetition_contempt,
            starter_draw_value: config.self_play.starter_draw_value,
            restart_simulations: config
                .self_play
                .restart_simulations
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
    let (restart_ply_min, restart_ply_max, mean_restart_ply) = restart_ply_stats(&games);
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
        restart_archive_prefixes,
        restart_ply_min,
        restart_ply_max,
        mean_restart_ply,
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
    let mirror_draw_limit_met =
        ratio(diagnostics.mirror.draws, mirror_games) <= config.arena.max_mirror_draw_rate;
    let exploratory_draw_gate_passed = ratio(diagnostics.exploratory.draws, exploratory_games)
        <= config.arena.max_candidate_self_play_draw_rate;
    PromotionDecision {
        arena_passed,
        mirror_draw_limit_met,
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

#[allow(clippy::cast_precision_loss)]
fn restart_ply_stats(games: &[SelfPlayGame]) -> (Option<usize>, Option<usize>, f32) {
    let mut count = 0_usize;
    let mut total = 0_usize;
    let mut minimum = None;
    let mut maximum = None;
    for ply in games
        .iter()
        .map(|game| game.restart_ply)
        .filter(|&ply| ply > 0)
    {
        count += 1;
        total = total.saturating_add(ply);
        minimum = Some(minimum.map_or(ply, |current: usize| current.min(ply)));
        maximum = Some(maximum.map_or(ply, |current: usize| current.max(ply)));
    }
    let mean = if count == 0 {
        0.0
    } else {
        total as f32 / count as f32
    };
    (minimum, maximum, mean)
}

/// Typed failures from any phase of a recoverable candidate attempt.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Training configuration failed semantic validation.
    #[error(transparent)]
    Configuration(crate::TrainingConfigError),
    /// Checkpoint load, save or publication failed.
    #[error(transparent)]
    Model(#[from] ModelStoreError),
    /// Shared inference worker could not start.
    #[error(transparent)]
    Inference(#[from] InferenceServiceError),
    /// Self-play generation failed.
    #[error(transparent)]
    SelfPlay(#[from] SelfPlayError),
    /// Candidate comparison failed.
    #[error(transparent)]
    Arena(#[from] ArenaError),
    /// Replay-buffer contents or targets were invalid.
    #[error(transparent)]
    TrainingData(#[from] crate::TrainingDataError),
    /// A stored complete game replay was invalid.
    #[error(transparent)]
    Replay(#[from] ReplayError),
    /// Dataset selection left no positions eligible for optimization.
    #[error("training split produced no examples")]
    EmptyTrainingSet,
    /// A resumable self-play file has incompatible provenance.
    #[error(
        "persisted self-play generation {file_generation} is empty or was not produced by network {expected_source_generation} with {expected_evaluator:?}"
    )]
    InvalidPersistedSelfPlay {
        /// Candidate attempt encoded by the filename.
        file_generation: u32,
        /// Champion generation that should have generated its games.
        expected_source_generation: u32,
        /// Evaluator that should have generated its targets.
        expected_evaluator: SelfPlayEvaluator,
    },
    /// Filesystem operation failed.
    #[error("training pipeline I/O error: {0}")]
    Io(#[from] io::Error),
    /// JSON artifact could not be serialized or parsed.
    #[error("training pipeline JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
