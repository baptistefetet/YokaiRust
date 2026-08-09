use std::{
    collections::HashSet,
    fs,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use yokai::{
    Action, AlphaZeroNetworkConfig, ArenaConfig, BOARD_SQUARES, BackendKind, CpuBackend,
    CpuTrainingBackend, Game, Mcts, ModelMetadata, OptimizationConfig, PathsConfig, Piece,
    PieceKind, Player, Position, ReplayBuffer, ReplayBufferConfig, SearchConfig, SelfPlayConfig,
    SelfPlayGame, SelfPlayRecorder, Square, TrainingConfig, TrainingConfigError, TrainingDataError,
    TrainingProgress, UniformEvaluator, bootstrap_champion, generate_self_play, load_generation,
    load_replay_buffer, run_arena, run_generation_with_progress, save_generation, train_candidate,
    validate_model,
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
fn checked_in_training_configuration_is_valid_and_strict() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config/training.toml");
    let config = TrainingConfig::load(path).expect("checked-in config must be valid");

    assert_eq!(config.network.filters, 64);
    assert_eq!(config.network.residual_blocks, 4);
    assert_eq!(config.arena.games, 200);
    assert!((config.arena.promotion_score - 0.55).abs() < f32::EPSILON);

    let mut invalid = config;
    invalid.arena.games = 199;
    assert!(matches!(
        invalid.validate(),
        Err(TrainingConfigError::Invalid(_))
    ));
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
        epochs: 80,
        batch_size: 2,
        learning_rate: 0.02,
        weight_decay: 0.0,
        validation_fraction: 0.5,
        mirror_augmentation: true,
        replay_buffer: ReplayBufferConfig::default(),
    };

    let (trained, report) =
        train_candidate(model, &examples, &examples, &optimization, 77, &device);
    let trained = trained.valid();
    let final_metrics = validate_model(&trained, &examples, 2, &device);

    assert_eq!(report.epochs.len(), optimization.epochs);
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
        max_game_plies: 256,
        inference_batch_size: 8,
        inference_wait_ms: 1,
        exploration_plies: 6,
        exploration_temperature: 1.0,
        final_temperature: 0.0,
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
            simulations: 16,
            promotion_score: 0.55,
        },
        2,
        256,
        900,
    )
    .expect("paired arena must finish");

    assert_eq!(
        result.candidate_wins + result.champion_wins + result.draws,
        2
    );
    assert!((result.score - 0.5).abs() < f32::EPSILON);
    assert!(!result.promoted);
}

#[test]
#[allow(clippy::too_many_lines)]
fn short_cpu_alphazero_generation_reaches_an_arena_decision() {
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
            max_game_plies: 128,
            inference_batch_size: 16,
            inference_wait_ms: 0,
            exploration_plies: 4,
            exploration_temperature: 1.0,
            final_temperature: 0.0,
        },
        optimization: OptimizationConfig {
            epochs: 1,
            batch_size: 16,
            learning_rate: 0.001,
            weight_decay: 0.0,
            validation_fraction: 0.5,
            mirror_augmentation: true,
            replay_buffer: ReplayBufferConfig {
                max_games: 8,
                generations_to_keep: 2,
            },
        },
        arena: ArenaConfig {
            games: 200,
            simulations: 1,
            promotion_score: 0.55,
        },
        paths: PathsConfig {
            models: models.to_string_lossy().into_owned(),
            self_play: self_play.to_string_lossy().into_owned(),
        },
    };
    let device = burn::backend::flex::FlexDevice;
    CpuTrainingBackend::seed(&device, config.seed);
    bootstrap_champion::<CpuBackend>(&models, config.network.clone(), &device)
        .expect("initial champion");
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
        report.arena.candidate_wins + report.arena.champion_wins + report.arena.draws,
        200
    );
    assert_eq!(reloaded_buffer.len(), 2);
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
        TrainingProgress::EpochFinished {
            total_epochs: 1,
            ..
        }
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        TrainingProgress::ArenaAdvanced {
            completed: 200,
            total: 200
        }
    )));
    assert!(matches!(
        events.last(),
        Some(
            TrainingProgress::ChampionPromoted { .. } | TrainingProgress::CandidateRejected { .. }
        )
    ));
    fs::remove_dir_all(root).expect("pipeline test cleanup");
}
