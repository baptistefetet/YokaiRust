use std::{
    collections::HashSet,
    fs,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use yokai::{
    Action, AlphaZeroNetworkConfig, ArenaConfig, BOARD_SQUARES, BackendKind, CpuBackend,
    CpuTrainingBackend, DatasetSplit, DrawReason, Game, Mcts, ModelMetadata, OptimizationConfig,
    PathsConfig, Piece, PieceKind, Player, Position, Replay, ReplayBuffer, ReplayBufferConfig,
    SearchConfig, SelfPlayConfig, SelfPlayGame, SelfPlayRecorder, Square, TerminalWindowSchedule,
    TrainingConfig, TrainingConfigError, TrainingDataError, TrainingProgress, UniformEvaluator,
    bootstrap_latest, generate_self_play, load_generation, load_latest, load_replay_buffer,
    run_arena, run_arena_with_progress, run_generation_with_progress, save_generation,
    train_candidate, validate_model,
};

fn square(row: u8, column: u8) -> Square {
    Square::new(row, column).expect("test square must be valid")
}

fn immediate_win_game() -> Game {
    let mut board = [None; BOARD_SQUARES];
    board[square(3, 0).index()] = Some(Piece::new(PieceKind::Koropokkuru, Player::First));
    board[square(1, 1).index()] = Some(Piece::new(PieceKind::Koropokkuru, Player::Second));
    board[square(2, 1).index()] = Some(Piece::new(PieceKind::Tanuki, Player::First));
    Game::from_position(
        Position::from_parts(board, [[0; 3]; 2], Player::First)
            .expect("test position must be valid"),
    )
}

fn recorded_game(generation: u32, seed: u64) -> SelfPlayGame {
    let mut game = immediate_win_game();
    let mut search = Mcts::new(
        UniformEvaluator,
        SearchConfig {
            simulations: 64,
            ..SearchConfig::default()
        },
        seed,
    )
    .expect("valid search");
    let result = search.search(&game, 0.0).expect("search must succeed");
    let mut recorder = SelfPlayRecorder::new();
    recorder
        .record(&game, &result)
        .expect("non-terminal target");
    game.apply(result.selected_action)
        .expect("selected action must be legal");
    recorder
        .finish(generation, seed, game.outcome())
        .expect("terminal self-play game")
}

#[test]
fn recorder_assigns_final_value_without_a_terminal_policy_target() {
    let self_play = recorded_game(0, 7);

    assert_eq!(self_play.examples.len(), 1);
    assert!((self_play.examples[0].value - 1.0).abs() < f32::EPSILON);
    assert!(self_play.replay.is_none());

    let terminal = {
        let mut game = immediate_win_game();
        game.apply(Action::Move {
            from: square(2, 1),
            to: square(1, 1),
        })
        .expect("winning move");
        game
    };
    let mut recorder = SelfPlayRecorder::new();
    let mut search = Mcts::new(UniformEvaluator, SearchConfig::default(), 1).expect("valid search");
    assert!(search.search(&terminal, 0.0).is_err());
    let dummy_result = Mcts::new(UniformEvaluator, SearchConfig::default(), 1)
        .expect("valid search")
        .search(&immediate_win_game(), 0.0)
        .expect("search result");
    assert_eq!(
        recorder.record(&terminal, &dummy_result),
        Err(TrainingDataError::TerminalPolicyTarget)
    );
}

#[test]
fn mirroring_a_training_example_twice_is_an_exact_bijection() {
    let original = recorded_game(0, 9).examples.remove(0);
    let mirrored = original.mirrored();

    assert_ne!(mirrored.position, original.position);
    assert_eq!(mirrored.mirrored(), original);
}

#[test]
fn replay_buffer_retains_generations_and_splits_whole_games() {
    let mut buffer = ReplayBuffer::new(ReplayBufferConfig {
        max_games: 4,
        generations_to_keep: 2,
    })
    .expect("valid replay buffer");
    buffer.push(recorded_game(0, 10));
    buffer.push(recorded_game(1, 11));
    buffer.push(recorded_game(1, 12));
    buffer.push(recorded_game(2, 13));
    buffer.push(recorded_game(2, 14));

    assert_eq!(buffer.len(), 4);
    assert_eq!(buffer.example_count(), 4);
    let split = buffer.split(0.5, 123).expect("valid split");
    let training_seeds = split
        .training_games
        .iter()
        .map(|game| game.seed)
        .collect::<HashSet<_>>();
    let validation_seeds = split
        .validation_games
        .iter()
        .map(|game| game.seed)
        .collect::<HashSet<_>>();

    assert_eq!(training_seeds.len(), 2);
    assert_eq!(validation_seeds.len(), 2);
    assert!(training_seeds.is_disjoint(&validation_seeds));
    assert!(!training_seeds.contains(&10));
    assert!(!validation_seeds.contains(&10));
    assert_eq!(split.training_examples(true).len(), 4);
    assert_eq!(split.validation_examples().len(), 2);
}

#[test]
fn replay_buffer_json_round_trip_preserves_fixed_policy_width() {
    let mut buffer = ReplayBuffer::new(ReplayBufferConfig::default()).expect("valid buffer");
    buffer.push(recorded_game(3, 99));

    let json = serde_json::to_string(&buffer).expect("buffer serialization");
    let decoded: ReplayBuffer = serde_json::from_str(&json).expect("buffer deserialization");

    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded.example_count(), 1);
}

#[test]
fn terminal_window_keeps_only_the_tail_of_decisive_games() {
    let mut decisive = recorded_game(3, 100);
    let example = decisive.examples[0].clone();
    decisive.examples = (1..=5)
        .map(|repetition_count| {
            let mut example = example.clone();
            example.repetition_count = repetition_count;
            example
        })
        .collect();
    let mut drawn = decisive.clone();
    drawn.seed = 101;
    drawn.outcome = yokai::Outcome::Draw {
        reason: DrawReason::ThreefoldRepetition,
    };
    let split = DatasetSplit {
        training_games: vec![decisive.clone(), drawn.clone()],
        validation_games: vec![decisive, drawn],
    };

    let training = split.training_examples_with_window(true, Some(2));
    let validation = split.validation_examples_with_window(Some(2));

    assert_eq!(split.selected_game_counts(Some(2)), (1, 1));
    assert_eq!(training.len(), 4, "two tail positions plus mirrors");
    assert_eq!(validation.len(), 2);
    assert_eq!(validation[0].repetition_count, 4);
    assert_eq!(validation[1].repetition_count, 5);
    assert_eq!(validation[1].history[0], Some(validation[0].position));
    assert!(validation[0].history[0].is_some());
    assert_eq!(split.training_examples(false).len(), 10);
    assert_eq!(split.validation_examples().len(), 10);
}

#[test]
fn checked_in_training_configuration_is_valid_and_strict() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/training.toml");
    let config = TrainingConfig::load(path).expect("checked-in config must be valid");

    assert_eq!(config.network.filters, 64);
    assert_eq!(config.network.residual_blocks, 4);
    assert_eq!(config.self_play.workers, 16);
    assert_eq!(config.self_play.inference_wait_ms, 1);
    assert_eq!(config.self_play.exploration_plies, 12);
    assert!((config.self_play.exploration_temperature - 1.0).abs() < f32::EPSILON);
    assert_eq!(config.optimization.steps_per_generation, 400);
    assert_eq!(config.optimization.validation_interval_steps, 100);
    assert_eq!(config.optimization.terminal_window_plies, None);
    assert_eq!(config.arena.games, 200);
    assert_eq!(config.arena.workers, 128);
    assert_eq!(config.arena.search_batch_size, 1);
    assert!((config.arena.score_threshold - 0.55).abs() < f32::EPSILON);
    assert_eq!(config.arena.mirror_games, 64);
    assert!((config.arena.max_mirror_draw_rate - 0.35).abs() < f32::EPSILON);
    assert_eq!(config.arena.candidate_self_play_games, 64);
    assert!((config.arena.max_candidate_self_play_draw_rate - 0.20).abs() < f32::EPSILON);

    let mut invalid = config;
    invalid.arena.games = 199;
    assert!(matches!(
        invalid.validate(),
        Err(TrainingConfigError::Invalid(_))
    ));

    invalid.arena.games = 200;
    invalid.optimization.terminal_window_plies = Some(0);
    assert!(matches!(
        invalid.validate(),
        Err(TrainingConfigError::Invalid(_))
    ));
}

#[test]
fn terminal_window_schedule_expands_then_restores_full_alphazero_data() {
    let mut optimization = TrainingConfig::default().optimization;
    optimization.terminal_window_schedule = Some(TerminalWindowSchedule {
        initial_plies: 1,
        growth_factor: 2,
        full_dataset_generation: 5,
    });

    assert_eq!(optimization.terminal_window_for_generation(1), Some(1));
    assert_eq!(optimization.terminal_window_for_generation(2), Some(2));
    assert_eq!(optimization.terminal_window_for_generation(3), Some(4));
    assert_eq!(optimization.terminal_window_for_generation(4), Some(8));
    assert_eq!(optimization.terminal_window_for_generation(5), None);
    assert_eq!(optimization.terminal_window_for_generation(12), None);
}

#[test]
fn tiny_cpu_corpus_overfits_above_ninety_five_percent_top1() {
    use burn::{module::AutodiffModule, prelude::Backend};

    let examples = recorded_game(0, 77).augmented_examples(true);
    let device = burn::backend::flex::FlexDevice;
    CpuTrainingBackend::seed(&device, 77);
    let network_config = AlphaZeroNetworkConfig::new()
        .with_filters(8)
        .with_residual_blocks(1)
        .with_value_hidden(8);
    let model = network_config.init::<CpuTrainingBackend>(&device);
    let initial = validate_model(&model.valid(), &examples, 2, &device);
    let optimization = OptimizationConfig {
        steps_per_generation: 80,
        validation_interval_steps: 1,
        batch_size: 2,
        learning_rate: 0.02,
        weight_decay: 0.0,
        validation_fraction: 0.5,
        mirror_augmentation: true,
        terminal_window_plies: None,
        terminal_window_schedule: None,
        replay_buffer: ReplayBufferConfig::default(),
    };

    let (trained, report) =
        train_candidate(model, &examples, &examples, &optimization, 77, &device);
    let trained = trained.valid();
    let final_metrics = validate_model(&trained, &examples, 2, &device);
    let selected_validation = report
        .selected()
        .and_then(|checkpoint| checkpoint.validation)
        .expect("final validation checkpoint");

    assert_eq!(report.checkpoints.len(), optimization.steps_per_generation);
    assert!((final_metrics.total_loss - selected_validation.total_loss).abs() < 1.0e-5);
    assert!(final_metrics.policy_top1_accuracy >= 0.95);
    assert!(final_metrics.policy_loss < initial.policy_loss);

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("yokai-overfit-test-{}-{nonce}", std::process::id()));
    let metadata = ModelMetadata::new(1, network_config);
    save_generation(&root, &metadata, &trained).expect("trained model save");
    let (reloaded, _) =
        load_generation::<CpuBackend>(&root, 1, &device).expect("trained model reload");
    let reloaded_metrics = validate_model(&reloaded, &examples, 2, &device);
    assert!((reloaded_metrics.policy_loss - final_metrics.policy_loss).abs() < f32::EPSILON);
    assert!((reloaded_metrics.value_loss - final_metrics.value_loss).abs() < f32::EPSILON);
    fs::remove_dir_all(root).expect("trained checkpoint cleanup");
}

#[test]
fn parallel_self_play_is_seed_ordered_and_reproducible() {
    let config = SelfPlayConfig {
        games_per_generation: 2,
        workers: 2,
        simulations: 16,
        search_batch_size: 1,
        max_game_plies: 256,
        inference_batch_size: 8,
        inference_wait_ms: 1,
        exploration_plies: 6,
        exploration_temperature: 1.0,
        final_temperature: 0.0,
        repetition_contempt: 0.0,
    };

    let first = generate_self_play(&UniformEvaluator, &config, 2, 500)
        .expect("self-play generation must finish");
    let second = generate_self_play(&UniformEvaluator, &config, 2, 500)
        .expect("self-play generation must be reproducible");

    assert_eq!(first, second);
    assert_eq!(
        first.iter().map(|game| game.seed).collect::<Vec<_>>(),
        vec![500, 501]
    );
    assert!(first.iter().all(|game| !game.examples.is_empty()));
}

#[test]
fn paired_arena_scores_identical_evaluators_at_one_half() {
    let result = run_arena(
        &UniformEvaluator,
        &UniformEvaluator,
        &ArenaConfig {
            games: 2,
            workers: 2,
            simulations: 16,
            search_batch_size: 1,
            score_threshold: 0.55,
            mirror_games: 2,
            max_mirror_draw_rate: 1.0,
            candidate_self_play_games: 2,
            max_candidate_self_play_draw_rate: 1.0,
        },
        2,
        256,
        900,
    )
    .expect("paired arena must finish");

    assert_eq!(
        result.candidate_wins + result.reference_wins + result.draws,
        2
    );
    assert_eq!(result.candidate_as_first.games(), 1);
    assert_eq!(result.candidate_as_second.games(), 1);
    assert!((result.score - 0.5).abs() < f32::EPSILON);
    assert!(!result.threshold_reached);
}

#[test]
fn arena_progress_reports_consistent_running_outcomes() {
    let snapshots = Arc::new(Mutex::new(Vec::new()));
    let recorded_snapshots = snapshots.clone();
    let result = run_arena_with_progress(
        &UniformEvaluator,
        &UniformEvaluator,
        &ArenaConfig {
            games: 4,
            workers: 2,
            simulations: 4,
            search_batch_size: 1,
            score_threshold: 0.55,
            mirror_games: 4,
            max_mirror_draw_rate: 1.0,
            candidate_self_play_games: 2,
            max_candidate_self_play_draw_rate: 1.0,
        },
        2,
        256,
        901,
        &move |progress| {
            recorded_snapshots
                .lock()
                .expect("arena progress mutex")
                .push(progress);
        },
    )
    .expect("arena with progress must finish");

    let snapshots = snapshots.lock().expect("arena progress mutex");
    assert_eq!(snapshots.len(), 4);
    for (index, snapshot) in snapshots.iter().enumerate() {
        assert_eq!(snapshot.completed, index + 1);
        assert_eq!(
            snapshot.candidate_wins + snapshot.reference_wins + snapshot.draws,
            snapshot.completed
        );
    }
    let final_snapshot = snapshots.last().expect("final arena progress");
    assert_eq!(final_snapshot.candidate_wins, result.candidate_wins);
    assert_eq!(final_snapshot.reference_wins, result.reference_wins);
    assert_eq!(final_snapshot.draws, result.draws);
    assert_eq!(result.candidate_as_first.games(), 2);
    assert_eq!(result.candidate_as_second.games(), 2);
    assert!((final_snapshot.score() - result.score).abs() < f32::EPSILON);
}

#[test]
#[allow(clippy::too_many_lines)]
fn short_cpu_alphazero_generation_publishes_before_diagnostics() {
    use burn::prelude::Backend;

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "yokai-pipeline-test-{}-{nonce}",
        std::process::id()
    ));
    let models = root.join("models");
    let self_play = root.join("self-play");
    let config = TrainingConfig {
        seed: 1234,
        backend: BackendKind::Cpu,
        network: AlphaZeroNetworkConfig::new()
            .with_filters(4)
            .with_residual_blocks(1)
            .with_value_hidden(4),
        self_play: SelfPlayConfig {
            games_per_generation: 2,
            workers: 4,
            simulations: 2,
            search_batch_size: 1,
            max_game_plies: 512,
            inference_batch_size: 16,
            inference_wait_ms: 0,
            exploration_plies: 4,
            exploration_temperature: 1.0,
            final_temperature: 0.0,
            repetition_contempt: 0.0,
        },
        optimization: OptimizationConfig {
            steps_per_generation: 1,
            validation_interval_steps: 1,
            batch_size: 16,
            learning_rate: 0.001,
            weight_decay: 0.0,
            validation_fraction: 0.5,
            mirror_augmentation: true,
            terminal_window_plies: None,
            terminal_window_schedule: None,
            replay_buffer: ReplayBufferConfig {
                max_games: 8,
                generations_to_keep: 2,
            },
        },
        arena: ArenaConfig {
            games: 200,
            workers: 2,
            simulations: 1,
            search_batch_size: 1,
            score_threshold: 0.55,
            mirror_games: 2,
            max_mirror_draw_rate: 1.0,
            candidate_self_play_games: 2,
            max_candidate_self_play_draw_rate: 1.0,
        },
        paths: PathsConfig {
            models: models.to_string_lossy().into_owned(),
            self_play: self_play.to_string_lossy().into_owned(),
        },
    };
    let device = burn::backend::flex::FlexDevice;
    CpuTrainingBackend::seed(&device, config.seed);
    bootstrap_latest::<CpuBackend>(&models, config.network.clone(), &device)
        .expect("initial network");
    let mut buffer = ReplayBuffer::new(config.optimization.replay_buffer).expect("buffer");

    let events = Arc::new(Mutex::new(Vec::new()));
    let recorded_events = events.clone();
    let progress = move |event| {
        recorded_events
            .lock()
            .expect("progress event mutex")
            .push(event);
    };
    let report = run_generation_with_progress::<CpuTrainingBackend, _>(
        &config,
        &mut buffer,
        &device,
        &progress,
    )
    .expect("short generation must finish");
    let reloaded_buffer = load_replay_buffer(
        self_play.join("buffer.json"),
        config.optimization.replay_buffer,
    )
    .expect("persisted buffer");

    assert_eq!(report.generated_games, 2);
    assert_eq!(
        report.self_play_outcomes.starter_wins
            + report.self_play_outcomes.non_starter_wins
            + report.self_play_outcomes.unclassified_wins,
        report.self_play_outcomes.first_wins + report.self_play_outcomes.second_wins,
    );
    assert_eq!(report.self_play_outcomes.unclassified_wins, 0);
    assert_eq!(
        report.arena.candidate_wins + report.arena.reference_wins + report.arena.draws,
        200
    );
    assert_eq!(
        report.candidate_mirror.candidate_wins
            + report.candidate_mirror.reference_wins
            + report.candidate_mirror.draws,
        2
    );
    assert_eq!(reloaded_buffer.len(), 2);
    let persisted_report: yokai::GenerationReport = serde_json::from_slice(
        &fs::read(self_play.join("reports/generation-000001.json"))
            .expect("persisted generation report"),
    )
    .expect("generation report JSON");
    assert_eq!(persisted_report.candidate_generation, 1);
    assert_eq!(persisted_report.training.steps_completed, 1);
    let replay_directory = self_play.join("replays/generation-000001");
    assert_eq!(
        fs::read_dir(replay_directory)
            .expect("persisted replay directory")
            .count(),
        2
    );
    let replay = Replay::read_json(self_play.join("replays/generation-000001/game-0000.json"))
        .expect("persisted self-play replay");
    assert!(replay.outcome.is_terminal());
    let events = events.lock().expect("progress event mutex");
    assert!(matches!(
        events.first(),
        Some(TrainingProgress::GenerationStarted { .. })
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        TrainingProgress::SelfPlayAdvanced {
            completed: 2,
            total: 2
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        TrainingProgress::TrainingAdvanced { total_steps: 1, .. }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        TrainingProgress::ArenaAdvanced { progress }
            if progress.completed == 200 && progress.total == 200
    )));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, TrainingProgress::LatestPublished { generation: 1 }))
    );
    assert!(events.iter().any(|event| matches!(
        event,
        TrainingProgress::TrainingStarted {
            optimizer_resumed: false,
            ..
        }
    )));
    assert!(
        models
            .join("generation-000001/training-model.bin")
            .is_file()
    );
    assert!(models.join("generation-000001/optimizer.bin").is_file());
    let (_, latest) = load_latest::<CpuBackend>(&models, &device).expect("latest model");
    assert_eq!(
        latest.generation, 1,
        "arena score cannot reject generation 1"
    );
    drop(events);

    // Simulate Ctrl+C after generation-2 self-play was persisted but before its
    // model was trained. Resuming must reuse those games instead of duplicating
    // an identical deterministic batch in the replay buffer.
    let mut persisted_games: Vec<SelfPlayGame> = serde_json::from_slice(
        &fs::read(self_play.join("generation-000001.json")).expect("generation-1 games"),
    )
    .expect("persisted game JSON");
    for game in &mut persisted_games {
        game.generation = 1;
        game.seed = game.seed.wrapping_add(10_000);
    }
    fs::write(
        self_play.join("generation-000002.json"),
        serde_json::to_vec(&persisted_games).expect("generation-2 serialization"),
    )
    .expect("generation-2 persistence");
    let resumed_events = Arc::new(Mutex::new(Vec::new()));
    let recorded_resumed_events = resumed_events.clone();
    run_generation_with_progress::<CpuTrainingBackend, _>(
        &config,
        &mut buffer,
        &device,
        &move |event| {
            recorded_resumed_events
                .lock()
                .expect("resumed progress event mutex")
                .push(event);
        },
    )
    .expect("resumed generation must finish");
    assert!(
        resumed_events
            .lock()
            .expect("resumed events")
            .iter()
            .any(|event| matches!(
                event,
                TrainingProgress::SelfPlayResumed {
                    games: 2,
                    examples: _
                }
            ))
    );
    assert!(
        resumed_events
            .lock()
            .expect("resumed events")
            .iter()
            .any(|event| matches!(
                event,
                TrainingProgress::TrainingStarted {
                    optimizer_resumed: true,
                    ..
                }
            ))
    );
    let (_, latest) = load_latest::<CpuBackend>(&models, &device).expect("resumed latest model");
    assert_eq!(latest.generation, 2);
    fs::remove_dir_all(root).expect("pipeline test cleanup");
}
