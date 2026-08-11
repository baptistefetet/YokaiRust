use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use yokai::{
    AlphaZeroNetworkConfig, BOARD_SQUARES, CpuBackend, EncodedPosition, Evaluation,
    EvaluationError, EvaluationRequest, Evaluator, Game, HandPiece, INPUT_PLANES, InferenceService,
    MetalBackend, ModelMetadata, ModelStoreError, NetworkEvaluator, POLICY_ACTIONS, Piece,
    PieceKind, Player, Position, Square, encode_position, encoded_batch_tensor, load_generation,
    load_latest, publish_latest, save_generation,
};

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

#[test]
fn official_setup_has_the_same_canonical_encoding_for_both_players() {
    let first = encode_position(&Position::initial(Player::First), 1);
    let second = encode_position(&Position::initial(Player::Second), 1);

    assert_eq!(first, second);
    assert!((first.get(0, 3, 1) - 1.0).abs() < f32::EPSILON);
    assert!((first.get(5, 0, 1) - 1.0).abs() < f32::EPSILON);
}

#[test]
fn hand_and_repetition_planes_are_normalized_constants() {
    let position = position_with(
        &[
            (3, 1, PieceKind::Koropokkuru, Player::First),
            (0, 1, PieceKind::Koropokkuru, Player::Second),
        ],
        [[2, 1, 0], [0, 0, 1]],
        Player::First,
    );
    let encoded = encode_position(&position, 2);

    assert_plane_is_constant(&encoded, 10 + HandPiece::Tanuki.index(), 1.0);
    assert_plane_is_constant(&encoded, 10 + HandPiece::Kitsune.index(), 0.5);
    assert_plane_is_constant(&encoded, 13 + HandPiece::Kodama.index(), 0.5);
    assert_plane_is_constant(&encoded, INPUT_PLANES - 1, 2.0 / 3.0);
}

#[test]
fn horizontal_position_mirror_reverses_only_encoded_columns() {
    let position = position_with(
        &[
            (3, 0, PieceKind::Koropokkuru, Player::First),
            (0, 2, PieceKind::Koropokkuru, Player::Second),
            (2, 0, PieceKind::KodamaSamurai, Player::First),
        ],
        [[0, 1, 0], [0; 3]],
        Player::Second,
    );
    let original = encode_position(&position, 1);
    let mirrored = encode_position(&position.mirrored_horizontally(), 1);

    for plane in 0..INPUT_PLANES {
        for row in 0..4 {
            for column in 0..3 {
                assert!(
                    (original.get(plane, row, column) - mirrored.get(plane, row, 2 - column)).abs()
                        < f32::EPSILON
                );
            }
        }
    }
}

#[test]
fn residual_network_has_fixed_policy_and_bounded_value_outputs() {
    type Backend = CpuBackend;

    let device = burn::backend::flex::FlexDevice;
    let config = AlphaZeroNetworkConfig::new()
        .with_filters(8)
        .with_residual_blocks(2)
        .with_value_hidden(16);
    let model = config.init::<Backend>(&device);
    let positions = [
        encode_position(&Position::initial(Player::First), 1),
        encode_position(&Position::initial(Player::Second), 2),
    ];
    let output = model.forward(encoded_batch_tensor(&positions, &device));

    assert_eq!(output.policy_logits.dims(), [2, POLICY_ACTIONS]);
    assert_eq!(output.value.dims(), [2, 1]);
    let values = output
        .value
        .into_data()
        .to_vec::<f32>()
        .expect("f32 value output");
    assert!(values.iter().all(|value| (-1.0..=1.0).contains(value)));
}

#[test]
fn cpu_network_evaluator_batches_normalized_predictions() {
    use burn::prelude::Backend;

    let device = burn::backend::flex::FlexDevice;
    CpuBackend::seed(&device, 91);
    let model = AlphaZeroNetworkConfig::new()
        .with_filters(8)
        .with_residual_blocks(1)
        .with_value_hidden(8)
        .init::<CpuBackend>(&device);
    let mut evaluator = NetworkEvaluator::new(model, device);
    let first = EvaluationRequest::from_game(&Game::new(Player::First));
    let second = EvaluationRequest::from_game(&Game::new(Player::Second));

    let evaluations = evaluator
        .evaluate_batch(&[first, second])
        .expect("CPU inference must succeed");

    assert_eq!(evaluations.len(), 2);
    for evaluation in evaluations {
        assert!((evaluation.policy.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
        assert!((-1.0..=1.0).contains(&evaluation.value));
    }
}

#[test]
#[ignore = "requires an available Metal device"]
fn metal_network_evaluator_runs_a_real_batch() {
    use burn::prelude::Backend;

    let device = burn::backend::wgpu::WgpuDevice::default();
    MetalBackend::seed(&device, 91);
    let model = AlphaZeroNetworkConfig::new()
        .with_filters(8)
        .with_residual_blocks(1)
        .with_value_hidden(8)
        .init::<MetalBackend>(&device);
    let mut evaluator = NetworkEvaluator::new(model, device);
    let request = EvaluationRequest::from_game(&Game::new(Player::First));

    let evaluations = evaluator
        .evaluate_batch(&[request, request])
        .expect("Metal inference must succeed");

    assert_eq!(evaluations.len(), 2);
    assert!((evaluations[0].policy.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
}

#[derive(Clone)]
struct RecordingEvaluator {
    batch_sizes: Arc<Mutex<Vec<usize>>>,
}

impl Evaluator for RecordingEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        self.batch_sizes
            .lock()
            .expect("batch recorder mutex")
            .push(requests.len());
        Ok(vec![Evaluation::uniform(0.0); requests.len()])
    }
}

#[test]
fn inference_service_combines_concurrent_games_into_one_batch() {
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let service = InferenceService::start(
        RecordingEvaluator {
            batch_sizes: batch_sizes.clone(),
        },
        8,
        Duration::from_millis(50),
    )
    .expect("inference service must start");
    let barrier = Arc::new(Barrier::new(3));
    let request = EvaluationRequest::from_game(&Game::new(Player::First));
    let workers = (0..2)
        .map(|_| {
            let barrier = barrier.clone();
            let mut client = service.client();
            std::thread::spawn(move || {
                barrier.wait();
                client
                    .evaluate_batch(&[request])
                    .expect("batched request must succeed")
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    for worker in workers {
        assert_eq!(worker.join().expect("client thread").len(), 1);
    }

    assert_eq!(*batch_sizes.lock().expect("batch recorder mutex"), vec![2]);
    let stats = service.stats();
    assert_eq!(stats.jobs, 2);
    assert_eq!(stats.backend_batches, 1);
    assert_eq!(stats.positions, 2);
    assert_eq!(stats.maximum_batch_size, 2);
    assert!((stats.average_batch_size() - 2.0).abs() < f64::EPSILON);
}

#[test]
fn minimum_batch_runs_without_waiting_for_the_maximum() {
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let service = InferenceService::start_with_batching(
        RecordingEvaluator {
            batch_sizes: batch_sizes.clone(),
        },
        2,
        8,
        Duration::from_secs(5),
    )
    .expect("inference service must start");
    let barrier = Arc::new(Barrier::new(3));
    let request = EvaluationRequest::from_game(&Game::new(Player::First));
    let started = Instant::now();
    let workers = (0..2)
        .map(|_| {
            let barrier = barrier.clone();
            let mut client = service.client();
            std::thread::spawn(move || {
                barrier.wait();
                client
                    .evaluate_batch(&[request])
                    .expect("minimum batch request must succeed")
            })
        })
        .collect::<Vec<_>>();
    barrier.wait();
    for worker in workers {
        assert_eq!(worker.join().expect("client thread").len(), 1);
    }

    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(*batch_sizes.lock().expect("batch recorder mutex"), vec![2]);
}

#[test]
fn inference_service_never_exceeds_the_configured_backend_batch() {
    let batch_sizes = Arc::new(Mutex::new(Vec::new()));
    let service = InferenceService::start(
        RecordingEvaluator {
            batch_sizes: batch_sizes.clone(),
        },
        2,
        Duration::ZERO,
    )
    .expect("inference service must start");
    let request = EvaluationRequest::from_game(&Game::new(Player::First));
    let mut client = service.client();

    let evaluations = client
        .evaluate_batch(&[request; 5])
        .expect("large client request must be split");

    assert_eq!(evaluations.len(), 5);
    assert_eq!(
        *batch_sizes.lock().expect("batch recorder mutex"),
        vec![2, 2, 1]
    );
}

#[test]
fn checkpoint_round_trip_is_exact_and_latest_pointer_is_atomic() {
    use burn::prelude::Backend;

    let root = unique_test_directory();
    let device = burn::backend::flex::FlexDevice;
    CpuBackend::seed(&device, 123);
    let config = AlphaZeroNetworkConfig::new()
        .with_filters(8)
        .with_residual_blocks(1)
        .with_value_hidden(8);
    let model = config.init::<CpuBackend>(&device);
    let metadata = ModelMetadata::new(3, config);
    let encoded = [encode_position(&Position::initial(Player::First), 1)];
    let before = model
        .forward(encoded_batch_tensor(&encoded, &device))
        .policy_logits
        .into_data()
        .to_vec::<f32>()
        .expect("policy values");

    save_generation(&root, &metadata, &model).expect("generation save");
    publish_latest(&root, 3).expect("latest publication");
    let (loaded, loaded_metadata) = load_latest::<CpuBackend>(&root, &device).expect("latest load");
    let after = loaded
        .forward(encoded_batch_tensor(&encoded, &device))
        .policy_logits
        .into_data()
        .to_vec::<f32>()
        .expect("reloaded policy values");

    assert_eq!(loaded_metadata, metadata);
    assert!(
        before
            .iter()
            .zip(after)
            .all(|(left, right)| (*left - right).abs() < f32::EPSILON)
    );
    assert!(matches!(
        save_generation(&root, &metadata, &model),
        Err(ModelStoreError::GenerationExists(3))
    ));

    fs::remove_dir_all(root).expect("test checkpoint cleanup");
}

#[test]
fn checkpoint_rejects_incompatible_encoder_metadata() {
    let root = unique_test_directory();
    let device = burn::backend::flex::FlexDevice;
    let config = AlphaZeroNetworkConfig::new()
        .with_filters(4)
        .with_residual_blocks(1)
        .with_value_hidden(4);
    let model = config.init::<CpuBackend>(&device);
    let metadata = ModelMetadata::new(1, config);
    let directory = save_generation(&root, &metadata, &model).expect("generation save");
    let mut incompatible = metadata;
    incompatible.encoder_version = 999;
    fs::write(
        directory.join("metadata.json"),
        serde_json::to_vec_pretty(&incompatible).expect("metadata serialization"),
    )
    .expect("metadata rewrite");

    let error = load_generation::<CpuBackend>(&root, 1, &device)
        .expect_err("encoder mismatch must be rejected");
    assert!(matches!(error, ModelStoreError::Incompatible(_)));

    fs::remove_dir_all(root).expect("test checkpoint cleanup");
}

fn unique_test_directory() -> PathBuf {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "yokai-model-test-{}-{nonce}-{}",
        std::process::id(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    ))
}

fn assert_plane_is_constant(encoded: &EncodedPosition, plane: usize, expected: f32) {
    for row in 0..4 {
        for column in 0..3 {
            assert!((encoded.get(plane, row, column) - expected).abs() < 1.0e-6);
        }
    }
}
