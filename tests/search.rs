//! Behavioral tests for PUCT perspective, noise, batching and subtree reuse.

use std::sync::{Arc, Mutex};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use yokai::{
    Action, BOARD_SQUARES, CachedEvaluator, Evaluation, EvaluationError, EvaluationRequest,
    Evaluator, Game, LeafEvaluation, Mcts, POLICY_ACTIONS, Piece, PieceKind, Player, Position,
    Replay, SearchConfig, Square, TemperatureSchedule, UniformEvaluator, random_rollout_value,
};

#[derive(Clone)]
struct BatchRecordingEvaluator {
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    fail_call: Option<usize>,
    calls: usize,
}

#[derive(Clone, Copy, Debug)]
struct CertainDrawEvaluator;

impl Evaluator for CertainDrawEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Ok(vec![
            Evaluation::from_wdl(
                [1.0 / 132.0; POLICY_ACTIONS],
                [0.0, 1.0, 0.0]
            );
            requests.len()
        ])
    }
}

impl Evaluator for BatchRecordingEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        self.batch_sizes
            .lock()
            .expect("batch recorder mutex")
            .push(requests.len());
        let call = self.calls;
        self.calls += 1;
        if self.fail_call == Some(call) {
            return Err(EvaluationError::Backend(
                "intentional batched evaluation failure".to_owned(),
            ));
        }
        Ok(vec![Evaluation::uniform(0.0); requests.len()])
    }
}

fn square(row: u8, column: u8) -> Square {
    Square::new(row, column).expect("test square must be valid")
}

fn position_with(
    pieces: &[(u8, u8, PieceKind, Player)],
    hands: [[u8; 3]; 2],
    side_to_move: Player,
) -> Position {
    let mut board = [None; BOARD_SQUARES];
    for &(row, column, kind, owner) in pieces {
        board[square(row, column).index()] = Some(Piece::new(kind, owner));
    }
    Position::from_parts(board, hands, side_to_move).expect("test position must be valid")
}

fn search_config(simulations: u32) -> SearchConfig {
    SearchConfig {
        simulations,
        ..SearchConfig::default()
    }
}

#[test]
fn mcts_finds_an_immediate_win_and_orients_its_value_for_the_root() {
    let position = position_with(
        &[
            (3, 0, PieceKind::Koropokkuru, Player::First),
            (1, 1, PieceKind::Koropokkuru, Player::Second),
            (2, 1, PieceKind::Tanuki, Player::First),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let game = Game::from_position(position);
    let winning_action = Action::Move {
        from: square(2, 1),
        to: square(1, 1),
    };
    let mut mcts = Mcts::new(
        UniformEvaluator,
        SearchConfig {
            simulations: 128,
            evaluation_batch_size: 8,
            ..SearchConfig::default()
        },
        7,
    )
    .expect("valid search");

    let result = mcts.search(&game, 0.0).expect("search must succeed");
    let winning_analysis = result
        .analysis
        .iter()
        .find(|entry| entry.action == winning_action)
        .expect("winning action must be analyzed");

    assert_eq!(result.best_action, winning_action);
    assert_eq!(result.selected_action, winning_action);
    assert!(winning_analysis.q_value > 0.99);
    assert!(result.root_value > 0.5);
}

#[test]
fn repetition_contempt_penalizes_only_the_player_causing_the_draw() {
    let position = position_with(
        &[
            (3, 1, PieceKind::Koropokkuru, Player::First),
            (0, 1, PieceKind::Koropokkuru, Player::Second),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let mut game = Game::from_position(position);
    let cycle = [
        Action::Move {
            from: square(3, 1),
            to: square(3, 0),
        },
        Action::Move {
            from: square(0, 1),
            to: square(0, 2),
        },
        Action::Move {
            from: square(3, 0),
            to: square(3, 1),
        },
        Action::Move {
            from: square(0, 2),
            to: square(0, 1),
        },
    ];
    for action in cycle.into_iter().chain(cycle.into_iter().take(3)) {
        game.apply(action).expect("repetition setup action");
    }
    let drawing_action = cycle[3];
    let request = EvaluationRequest::from_game(&game);
    let drawing_index = drawing_action
        .policy_index(game.position().side_to_move())
        .expect("drawing action has a policy index");

    assert!(!request.current_player_is_starter);
    assert_eq!(game.repetition_count_after(drawing_action), Some(3));
    assert_eq!(
        request.action_repetition_counts[drawing_index.as_usize()],
        3
    );

    let mut neutral = Mcts::new(UniformEvaluator, search_config(128), 71).expect("neutral search");
    let mut contempt = Mcts::new(
        UniformEvaluator,
        SearchConfig {
            simulations: 128,
            repetition_contempt: 1.0,
            ..SearchConfig::default()
        },
        71,
    )
    .expect("contempt search");
    let neutral_result = neutral.search(&game, 0.0).expect("neutral result");
    let contempt_result = contempt.search(&game, 0.0).expect("contempt result");
    let q_for = |result: &yokai::SearchResult| {
        result
            .analysis
            .iter()
            .find(|entry| entry.action == drawing_action)
            .expect("drawing action analysis")
            .q_value
    };

    assert!(q_for(&neutral_result).abs() < f32::EPSILON);
    assert!(q_for(&contempt_result) < -0.99);
    assert_ne!(contempt_result.best_action, drawing_action);
}

#[test]
fn role_aware_draw_value_rewards_starter_and_penalizes_non_starter() {
    let config = SearchConfig {
        simulations: 16,
        starter_draw_value: 0.25,
        ..SearchConfig::default()
    };
    let starter_game = Game::new(Player::First);
    let mut starter_search = Mcts::new(CertainDrawEvaluator, config, 73).expect("valid search");
    let starter_result = starter_search
        .search(&starter_game, 0.0)
        .expect("starter search");

    let mut non_starter_game = starter_game;
    let entry = non_starter_game.legal_actions()[0];
    non_starter_game.apply(entry).expect("legal entry move");
    let mut non_starter_search = Mcts::new(CertainDrawEvaluator, config, 74).expect("valid search");
    let non_starter_result = non_starter_search
        .search(&non_starter_game, 0.0)
        .expect("non-starter search");

    assert!(starter_result.root_value > 0.2);
    assert!(non_starter_result.root_value < -0.2);
}

#[test]
fn batched_mcts_evaluates_distinct_leaves_and_recovers_after_failure() {
    let game = Game::new(Player::First);
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let evaluator = BatchRecordingEvaluator {
        batch_sizes: batch_sizes.clone(),
        fail_call: Some(1),
        calls: 0,
    };
    let config = SearchConfig {
        simulations: 64,
        evaluation_batch_size: 8,
        ..SearchConfig::default()
    };
    let mut search = Mcts::new(evaluator, config, 17).expect("valid batched search");

    assert!(matches!(
        search.search(&game, 0.0),
        Err(yokai::SearchError::Evaluation(EvaluationError::Backend(_)))
    ));
    let result = search
        .search(&game, 0.0)
        .expect("virtual losses must be released after a backend failure");
    let recorded = batch_sizes.lock().expect("batch recorder mutex");

    assert!(game.legal_actions().contains(&result.best_action));
    assert_eq!(recorded[0], 1, "the unexpanded root is evaluated alone");
    assert!(recorded.iter().any(|size| *size > 1));
    assert!(recorded.iter().all(|size| *size <= 8));
}

#[derive(Clone, Copy, Debug)]
struct IllegalPolicyEvaluator;

impl Evaluator for IllegalPolicyEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        let mut policy = [0.0; POLICY_ACTIONS];
        policy[POLICY_ACTIONS - 1] = 1.0;
        Ok(vec![Evaluation::new(policy, 5.0); requests.len()])
    }
}

#[test]
fn illegal_policy_mass_is_masked_and_legal_actions_are_renormalized() {
    let game = Game::new(Player::First);
    let legal_actions = game.legal_actions();
    let mut mcts =
        Mcts::new(IllegalPolicyEvaluator, search_config(1), 11).expect("valid search config");

    let result = mcts.search(&game, 1.0).expect("search must succeed");
    let policy_sum = result.policy.iter().sum::<f32>();

    assert!((policy_sum - 1.0).abs() < 1.0e-6);
    assert_eq!(result.analysis.len(), legal_actions.len());
    assert!(
        result
            .analysis
            .iter()
            .all(|entry| legal_actions.contains(&entry.action))
    );
    assert!(
        result
            .analysis
            .windows(2)
            .all(|pair| (pair[0].prior - pair[1].prior).abs() < 1.0e-6)
    );
    assert!(result.policy[POLICY_ACTIONS - 1].abs() < f32::EPSILON);
}

#[test]
fn seeded_root_noise_and_temperature_sampling_are_reproducible() {
    let game = Game::new(Player::First);
    let noisy_config = SearchConfig {
        simulations: 48,
        ..SearchConfig::default()
    };
    let mut first = Mcts::new(UniformEvaluator, noisy_config, 42).expect("valid search");
    let mut second = Mcts::new(UniformEvaluator, noisy_config, 42).expect("valid search");

    let first_result = first
        .search_self_play(&game, 1.0)
        .expect("search must succeed");
    let second_result = second
        .search_self_play(&game, 1.0)
        .expect("search must succeed");

    assert_eq!(first_result, second_result);
    assert!(
        first_result
            .analysis
            .windows(2)
            .any(|pair| (pair[0].prior - pair[1].prior).abs() > 1.0e-5)
    );

    let mut analysis_search =
        Mcts::new(UniformEvaluator, search_config(1), 42).expect("valid search");
    let analysis_result = analysis_search
        .search(&game, 1.0)
        .expect("analysis search must succeed");
    assert!(
        analysis_result
            .analysis
            .windows(2)
            .all(|pair| (pair[0].prior - pair[1].prior).abs() < 1.0e-6)
    );
}

#[test]
fn rollout_search_is_seeded_uniform_and_never_queries_the_evaluator() {
    let game = Game::new(Player::First);
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let evaluator = BatchRecordingEvaluator {
        batch_sizes: batch_sizes.clone(),
        fail_call: Some(0),
        calls: 0,
    };
    let config = SearchConfig {
        simulations: 32,
        evaluation_batch_size: 4,
        leaf_evaluation: LeafEvaluation::RandomRollout { max_plies: 64 },
        ..SearchConfig::default()
    };
    let mut first = Mcts::new(evaluator, config, 91).expect("rollout search");
    let mut second = Mcts::new(UniformEvaluator, config, 91).expect("matching rollout search");

    let first_result = first
        .search_self_play(&game, 1.0)
        .expect("rollout search bypasses evaluator");
    let second_result = second
        .search_self_play(&game, 1.0)
        .expect("fixed seed reproduces rollouts");

    assert_eq!(first_result, second_result);
    assert!(batch_sizes.lock().expect("batch recorder").is_empty());
    assert!(
        first_result
            .analysis
            .windows(2)
            .all(|pair| { (pair[0].prior - pair[1].prior).abs() < f32::EPSILON })
    );
}

#[test]
fn rollout_value_uses_the_leaf_perspective_and_safety_limit() {
    let position = position_with(
        &[
            (3, 0, PieceKind::Koropokkuru, Player::First),
            (1, 1, PieceKind::Koropokkuru, Player::Second),
            (2, 1, PieceKind::Tanuki, Player::First),
        ],
        [[0; 3]; 2],
        Player::First,
    );
    let game = Game::from_position(position);
    let winning_action = Action::Move {
        from: square(2, 1),
        to: square(1, 1),
    };
    let winning_seed = (0..1_024)
        .find(|seed| {
            let mut rng = ChaCha8Rng::seed_from_u64(*seed);
            random_rollout_value(&game, 1, &mut rng)
                .is_ok_and(|value| (value - 1.0).abs() < f32::EPSILON)
        })
        .expect("some seeded uniform rollout must select the immediate win");
    let mut winning_rng = ChaCha8Rng::seed_from_u64(winning_seed);
    let winning_value = random_rollout_value(&game, 1, &mut winning_rng).expect("winning rollout");
    assert!((winning_value - 1.0).abs() < f32::EPSILON);

    let mut terminal = game;
    terminal.apply(winning_action).expect("immediate win");
    let mut terminal_rng = ChaCha8Rng::seed_from_u64(0);
    let losing_value =
        random_rollout_value(&terminal, 512, &mut terminal_rng).expect("terminal rollout");
    assert!(
        (losing_value + 1.0).abs() < f32::EPSILON,
        "the terminal side to move is the losing leaf perspective",
    );

    let mut limited_rng = ChaCha8Rng::seed_from_u64(7);
    let limited_value = random_rollout_value(&Game::new(Player::First), 1, &mut limited_rng)
        .expect("limited rollout");
    assert!(
        limited_value.abs() < f32::EPSILON,
        "an ongoing rollout at the safety limit is a draw value",
    );
}

#[test]
fn self_play_temperature_changes_after_the_configured_ply() {
    let schedule = TemperatureSchedule {
        exploration_plies: 3,
        exploration_temperature: 1.25,
        final_temperature: 0.05,
    };

    assert!((schedule.for_ply(0) - 1.25).abs() < f32::EPSILON);
    assert!((schedule.for_ply(2) - 1.25).abs() < f32::EPSILON);
    assert!((schedule.for_ply(3) - 0.05).abs() < f32::EPSILON);
}

#[test]
fn move_temperature_does_not_sharpen_the_policy_target() {
    let game = Game::new(Player::First);
    let config = search_config(64);
    let mut greedy = Mcts::new(UniformEvaluator, config, 29).expect("valid search");
    let mut sampling = Mcts::new(UniformEvaluator, config, 29).expect("valid search");

    let greedy_result = greedy.search(&game, 0.0).expect("greedy search");
    let sampling_result = sampling.search(&game, 1.0).expect("sampling search");

    assert!(greedy_result.policy.iter().zip(sampling_result.policy).all(
        |(&greedy_probability, sampling_probability)| {
            (greedy_probability - sampling_probability).abs() < f32::EPSILON
        }
    ));
    assert!(
        greedy_result
            .policy
            .iter()
            .filter(|&&probability| probability > 0.0)
            .count()
            > 1,
        "a greedy played move must retain the complete visit distribution as its target",
    );
    assert_eq!(greedy_result.selected_action, greedy_result.best_action);
}

#[test]
fn arena_tree_is_reused_after_the_played_action() {
    let mut game = Game::new(Player::First);
    let mut mcts = Mcts::new(UniformEvaluator, search_config(64), 5).expect("valid search");
    let first_result = mcts.search(&game, 0.0).expect("search must succeed");
    let nodes_before_move = mcts.node_count();

    game.apply(first_result.best_action)
        .expect("selected action must be legal");

    assert!(mcts.advance_root(first_result.best_action, &game));
    assert_eq!(mcts.node_count(), nodes_before_move);

    mcts.search(&game, 0.0).expect("reused search must succeed");
    assert!(mcts.node_count() >= nodes_before_move);
}

#[test]
fn replay_round_trip_preserves_optional_search_analysis() {
    let mut game = Game::new(Player::First);
    let mut mcts = Mcts::new(UniformEvaluator, search_config(16), 17).expect("valid search");
    let result = mcts.search(&game, 0.0).expect("search must succeed");
    game.apply(result.selected_action)
        .expect("selected action must be legal");
    let replay = Replay::from_game(&game, Some(17)).with_analyses(vec![result.analysis]);

    let decoded = Replay::from_json(&replay.to_json_pretty().expect("serializable replay"))
        .expect("valid replay");

    assert_eq!(decoded, replay);
    assert_eq!(decoded.analyses.as_ref().map(Vec::len), Some(1));
}

#[derive(Debug, Default)]
struct CountingEvaluator {
    batches: usize,
    positions: usize,
}

impl Evaluator for CountingEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        self.batches += 1;
        self.positions += requests.len();
        Ok(vec![Evaluation::uniform(0.25); requests.len()])
    }
}

#[test]
fn prediction_cache_deduplicates_within_and_across_batches() {
    let game = Game::new(Player::First);
    let request = EvaluationRequest::from_game(&game);
    let mut cache = CachedEvaluator::new(CountingEvaluator::default(), 8);

    let first = cache
        .evaluate_batch(&[request, request])
        .expect("evaluation must succeed");
    let second = cache
        .evaluate_batch(&[request])
        .expect("cached evaluation must succeed");

    assert_eq!(first, vec![Evaluation::uniform(0.25); 2]);
    assert_eq!(second, vec![Evaluation::uniform(0.25)]);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.inner().batches, 1);
    assert_eq!(cache.inner().positions, 1);
}

#[derive(Clone, Copy, Debug)]
struct BrokenBatchEvaluator;

impl Evaluator for BrokenBatchEvaluator {
    fn evaluate_batch(
        &mut self,
        _requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Ok(Vec::new())
    }
}

#[test]
fn prediction_cache_rejects_an_incomplete_backend_batch() {
    let game = Game::new(Player::First);
    let request = EvaluationRequest::from_game(&game);
    let mut cache = CachedEvaluator::new(BrokenBatchEvaluator, 8);

    assert_eq!(
        cache.evaluate_batch(&[request]),
        Err(EvaluationError::BatchSizeMismatch {
            expected: 1,
            actual: 0,
        })
    );
}
