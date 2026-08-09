use std::sync::{Arc, Mutex};

use yokai::{
    Action, BOARD_SQUARES, CachedEvaluator, Evaluation, EvaluationError, EvaluationRequest,
    Evaluator, Game, Mcts, POLICY_ACTIONS, Piece, PieceKind, Player, Position, Replay,
    SearchConfig, Square, TemperatureSchedule, UniformEvaluator,
};

#[derive(Clone)]
struct BatchRecordingEvaluator {
    batch_sizes: Arc<Mutex<Vec<usize>>>,
    fail_call: Option<usize>,
    calls: usize,
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
