//! Explicit local performance probes. They are ignored by the normal suite.

use std::{
    hint::black_box,
    path::Path,
    time::{Duration, Instant},
};

use burn::prelude::Backend;
use yokai::{
    AlphaZeroNetworkConfig, ArenaConfig, CpuBackend, EvaluationRequest, Evaluator, Game,
    InferenceService, MetalBackend, NetworkEvaluator, Player, SelfPlayConfig, generate_self_play,
    load_generation, run_arena,
};

const BATCH_SIZES: [usize; 6] = [1, 8, 16, 32, 64, 128];

#[test]
#[ignore = "manual CPU throughput benchmark"]
fn benchmark_cpu_inference_batches() {
    let device = burn::backend::flex::FlexDevice;
    CpuBackend::seed(&device, 42);
    benchmark_backend::<CpuBackend>("cpu", device);
}

#[test]
#[ignore = "manual Metal throughput benchmark"]
fn benchmark_metal_inference_batches() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    MetalBackend::seed(&device, 42);
    benchmark_backend::<MetalBackend>("metal", device);
}

#[test]
#[ignore = "manual Metal self-play concurrency benchmark"]
fn benchmark_metal_self_play_worker_counts() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    for workers in [16, 32, 64, 128] {
        benchmark_metal_self_play_case(&device, workers, 2);
    }
}

#[test]
#[ignore = "manual Metal self-play batching wait benchmark"]
fn benchmark_metal_self_play_wait_times() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    for wait_ms in [0, 1, 2, 4] {
        benchmark_metal_self_play_case(&device, 128, wait_ms);
    }
}

#[test]
#[ignore = "manual Metal full-generation concurrency benchmark"]
fn benchmark_metal_full_concurrency() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    for workers in [64, 128, 256] {
        benchmark_metal_self_play_case_with_games(&device, workers, 1, 256);
    }
}

#[test]
#[ignore = "manual full-size Metal self-play benchmark"]
fn benchmark_metal_default_self_play() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    benchmark_metal_self_play_case_with_config(&device, 16, 1, 256, 400, 8);
}

#[test]
#[ignore = "manual batched MCTS concurrency benchmark"]
fn benchmark_metal_search_batch_sizes() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    for (workers, search_batch_size) in [
        (128, 1),
        (64, 2),
        (32, 4),
        (16, 8),
        (8, 16),
        (32, 8),
        (16, 16),
    ] {
        benchmark_metal_self_play_case_with_config(&device, workers, 1, 128, 64, search_batch_size);
    }
}

#[test]
#[ignore = "manual full-size Metal arena benchmark"]
fn benchmark_metal_default_arena() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    MetalBackend::seed(&device, 42);
    let network = AlphaZeroNetworkConfig::new();
    let candidate = InferenceService::start_with_batching(
        NetworkEvaluator::new(network.init::<MetalBackend>(&device), device.clone()),
        64,
        128,
        Duration::from_millis(1),
    )
    .expect("candidate service must start");
    MetalBackend::seed(&device, 43);
    let champion = InferenceService::start_with_batching(
        NetworkEvaluator::new(network.init::<MetalBackend>(&device), device.clone()),
        64,
        128,
        Duration::from_millis(1),
    )
    .expect("champion service must start");
    let started = Instant::now();
    let result = run_arena(
        &candidate.client(),
        &champion.client(),
        &ArenaConfig {
            games: 200,
            workers: 128,
            simulations: 400,
            search_batch_size: 1,
            promotion_score: 0.55,
            mirror_games: 64,
            max_mirror_draw_rate: 0.35,
        },
        128,
        512,
        9_000,
    )
    .expect("benchmark arena must finish");
    let elapsed = started.elapsed();
    let candidate_stats = candidate.stats();
    let champion_stats = champion.stats();
    println!(
        "arena_games=200 simulations=400 search_batch=1 workers=128 wait_ms=1 wall_s={:.2} candidate/champion/draw={}/{}/{} candidate_avg_batch={:.1} champion_avg_batch={:.1} combined_positions={} candidate_pos_per_second={:.1} champion_pos_per_second={:.1}",
        elapsed.as_secs_f64(),
        result.candidate_wins,
        result.champion_wins,
        result.draws,
        candidate_stats.average_batch_size(),
        champion_stats.average_batch_size(),
        candidate_stats.positions + champion_stats.positions,
        candidate_stats.positions_per_backend_second(),
        champion_stats.positions_per_backend_second(),
    );
}

#[test]
#[ignore = "manual saved-generation arena diagnostic"]
fn compare_saved_generations_in_both_argument_orders() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let (generation_3, _) = load_generation::<MetalBackend>(Path::new("models"), 3, &device)
        .expect("generation 3 checkpoint");
    let (generation_4, _) = load_generation::<MetalBackend>(Path::new("models"), 4, &device)
        .expect("generation 4 checkpoint");
    let older = InferenceService::start_with_batching(
        NetworkEvaluator::new(generation_3, device.clone()),
        8,
        128,
        Duration::from_millis(1),
    )
    .expect("older inference service");
    let newer = InferenceService::start_with_batching(
        NetworkEvaluator::new(generation_4, device),
        8,
        128,
        Duration::from_millis(1),
    )
    .expect("newer inference service");
    let config = ArenaConfig {
        games: 20,
        workers: 16,
        simulations: 100,
        search_batch_size: 1,
        promotion_score: 0.55,
        mirror_games: 20,
        max_mirror_draw_rate: 1.0,
    };
    let newer_as_candidate = run_arena(
        &newer.client(),
        &older.client(),
        &config,
        config.workers,
        512,
        30_000,
    )
    .expect("newer candidate arena");
    let older_as_candidate = run_arena(
        &older.client(),
        &newer.client(),
        &config,
        config.workers,
        512,
        40_000,
    )
    .expect("older candidate arena");
    println!("newer-as-candidate={newer_as_candidate:?} older-as-candidate={older_as_candidate:?}");
}

#[test]
#[ignore = "manual saved-champion self-play diagnostic"]
#[allow(clippy::cast_precision_loss)]
fn compare_saved_champion_search_modes() {
    const GENERATION: u32 = 13;
    for (label, simulations, repetition_contempt) in [
        ("neutral-curriculum-200", 200, 0.0_f32),
        ("neutral-full-search-400", 400, 0.0_f32),
    ] {
        let workers = 16_usize;
        let search_batch_size = 8_usize;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let (model, _) = load_generation::<MetalBackend>(Path::new("models"), GENERATION, &device)
            .expect("saved champion checkpoint");
        let service = InferenceService::start_with_batching(
            NetworkEvaluator::new(model, device),
            workers.saturating_mul(search_batch_size).min(128),
            128,
            Duration::from_millis(1),
        )
        .expect("diagnostic inference service");
        let config = SelfPlayConfig {
            games_per_generation: 64,
            workers,
            simulations,
            search_batch_size,
            max_game_plies: 512,
            inference_batch_size: 128,
            inference_wait_ms: 1,
            exploration_plies: 12,
            exploration_temperature: 0.0,
            final_temperature: 0.0,
            repetition_contempt,
        };
        let started = Instant::now();
        let games = generate_self_play(&service.client(), &config, GENERATION, 50_000)
            .expect("diagnostic self-play");
        let draws = games
            .iter()
            .filter(|game| matches!(game.outcome, yokai::Outcome::Draw { .. }))
            .count();
        let examples = games.iter().map(|game| game.examples.len()).sum::<usize>();
        println!(
            "{label}: wall={:.2}s draws={draws}/{} examples={examples} avg_plies={:.1}",
            started.elapsed().as_secs_f64(),
            games.len(),
            examples as f64 / games.len() as f64,
        );
    }
}

#[test]
#[ignore = "manual saved-champion mirror arena diagnostic"]
fn compare_saved_champion_against_itself() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let (left_model, _) = load_generation::<MetalBackend>(Path::new("models"), 5, &device)
        .expect("left generation 5 checkpoint");
    let (right_model, _) = load_generation::<MetalBackend>(Path::new("models"), 5, &device)
        .expect("right generation 5 checkpoint");
    let left = InferenceService::start_with_batching(
        NetworkEvaluator::new(left_model, device.clone()),
        16,
        128,
        Duration::from_millis(1),
    )
    .expect("left inference service");
    let right = InferenceService::start_with_batching(
        NetworkEvaluator::new(right_model, device),
        16,
        128,
        Duration::from_millis(1),
    )
    .expect("right inference service");
    let config = ArenaConfig {
        games: 32,
        workers: 32,
        simulations: 400,
        search_batch_size: 1,
        promotion_score: 0.55,
        mirror_games: 32,
        max_mirror_draw_rate: 1.0,
    };
    let started = Instant::now();
    let result = run_arena(
        &left.client(),
        &right.client(),
        &config,
        config.workers,
        512,
        60_000,
    )
    .expect("generation 5 mirror arena");
    println!(
        "mirror arena wall={:.2}s result={result:?}",
        started.elapsed().as_secs_f64()
    );
}

fn benchmark_metal_self_play_case(
    device: &burn::backend::wgpu::WgpuDevice,
    workers: usize,
    wait_ms: u64,
) {
    benchmark_metal_self_play_case_with_config(device, workers, wait_ms, 128, 16, 1);
}

#[allow(clippy::cast_precision_loss)]
fn benchmark_metal_self_play_case_with_games(
    device: &burn::backend::wgpu::WgpuDevice,
    workers: usize,
    wait_ms: u64,
    games_per_generation: usize,
) {
    benchmark_metal_self_play_case_with_config(
        device,
        workers,
        wait_ms,
        games_per_generation,
        16,
        1,
    );
}

#[allow(clippy::cast_precision_loss)]
fn benchmark_metal_self_play_case_with_config(
    device: &burn::backend::wgpu::WgpuDevice,
    workers: usize,
    wait_ms: u64,
    games_per_generation: usize,
    simulations: u32,
    search_batch_size: usize,
) {
    MetalBackend::seed(device, 42);
    let network = AlphaZeroNetworkConfig::new();
    let evaluator = NetworkEvaluator::new(network.init::<MetalBackend>(device), device.clone());
    let service = InferenceService::start_with_batching(
        evaluator,
        workers.saturating_mul(search_batch_size).min(128),
        128,
        Duration::from_millis(wait_ms),
    )
    .expect("inference service must start");
    let config = SelfPlayConfig {
        games_per_generation,
        workers,
        simulations,
        search_batch_size,
        max_game_plies: 512,
        inference_batch_size: 128,
        inference_wait_ms: wait_ms,
        exploration_plies: 12,
        exploration_temperature: 1.0,
        final_temperature: 0.0,
        repetition_contempt: 0.0,
    };
    let started = Instant::now();
    let games = generate_self_play(&service.client(), &config, 0, 8_000)
        .expect("benchmark self-play must finish");
    let elapsed = started.elapsed();
    let stats = service.stats();
    let examples = games.iter().map(|game| game.examples.len()).sum::<usize>();
    println!(
        "games={games_per_generation:>3} simulations={simulations:>3} search_batch={search_batch_size:>2} workers={workers:>3} wait_ms={wait_ms:>2} wall_s={:>7.2} games_per_second={:>7.2} examples={} avg_batch={:>6.1} max_batch={:>3} inference_positions={} backend_positions_per_second={:>9.1} avg_wait_ms={:>7.2}",
        elapsed.as_secs_f64(),
        games_per_generation as f64 / elapsed.as_secs_f64(),
        examples,
        stats.average_batch_size(),
        stats.maximum_batch_size,
        stats.positions,
        stats.positions_per_backend_second(),
        stats.average_client_wait().as_secs_f64() * 1_000.0,
    );
}

#[allow(clippy::cast_precision_loss)]
fn benchmark_backend<B: Backend>(backend: &str, device: B::Device) {
    let model = AlphaZeroNetworkConfig::new().init::<B>(&device);
    let mut evaluator = NetworkEvaluator::new(model, device);
    let request = EvaluationRequest::from_game(&Game::new(Player::First));

    for batch_size in BATCH_SIZES {
        let requests = vec![request; batch_size];
        let warmup_started = Instant::now();
        black_box(
            evaluator
                .evaluate_batch(&requests)
                .expect("inference warmup must succeed"),
        );
        let warmup = warmup_started.elapsed();

        let iterations = (4_096 / batch_size).clamp(16, 256);
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(
                evaluator
                    .evaluate_batch(&requests)
                    .expect("timed inference must succeed"),
            );
        }
        let elapsed = started.elapsed();
        let positions = batch_size * iterations;
        let positions_per_second = positions as f64 / elapsed.as_secs_f64();
        let milliseconds_per_batch = 1_000.0 * elapsed.as_secs_f64() / iterations as f64;
        println!(
            "backend={backend} batch={batch_size:>3} warmup_ms={:>8.1} ms_per_batch={milliseconds_per_batch:>8.3} positions_per_second={positions_per_second:>10.1}",
            warmup.as_secs_f64() * 1_000.0,
        );
    }
}
