//! One-off common harness compiled against both recorded source revisions.
//! Paths are supplied by the experiment runner; nothing is published or trained
//! into the live model directory. GPU work must run without concurrent training.

use std::{
    env,
    error::Error,
    fs,
    hint::black_box,
    time::{Duration, Instant},
};

use burn::{module::AutodiffModule, prelude::Backend};
use serde_json::json;
use yokai::{
    EvaluationRequest, Evaluator, Game, InferenceService, MetalBackend, MetalTrainingBackend,
    NetworkEvaluator, Player, TrainingConfig, generate_self_play, load_generation,
    load_replay_buffer, load_training_generation, train_state_with_progress,
};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: measurement <config> <reference-root> <fixed-buffer> <output>".into());
    }
    let mut config = TrainingConfig::load(&args[0])?;
    let output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    let seed = 20_261_108;
    let device = burn::backend::wgpu::WgpuDevice::default();
    MetalBackend::seed(&device, seed);
    MetalTrainingBackend::seed(&device, seed);
    let (model, metadata) = load_generation::<MetalBackend>(&args[1], 21, &device)?;
    let mut evaluator = NetworkEvaluator::new(model, device.clone());
    let request = EvaluationRequest::from_game(&Game::new(Player::First));
    let mut inference = Vec::new();
    for batch in [8, 128] {
        let requests = vec![request; batch];
        for _ in 0..20 {
            black_box(evaluator.evaluate_batch(&requests)?);
        }
        let start = Instant::now();
        for _ in 0..200 {
            black_box(evaluator.evaluate_batch(&requests)?);
        }
        inference
            .push(json!({"batch":batch,"batches":200,"seconds":start.elapsed().as_secs_f64()}));
    }
    let service =
        InferenceService::start_with_batching(evaluator, 128, 128, Duration::from_millis(1))?;
    config.self_play.games_per_generation = 64;
    config.self_play.restart_fraction = 0.0;
    let start = Instant::now();
    let games = generate_self_play(&service.client(), &config.self_play, 21, seed)?;
    let self_play_seconds = start.elapsed().as_secs_f64();
    let stats = service.stats();
    let examples: usize = games.iter().map(|game| game.examples.len()).sum();
    let replays: Vec<_> = games.iter().map(|game| &game.replay).collect();
    drop(service);

    let buffer = load_replay_buffer(&args[2], config.optimization.replay_buffer)?;
    let split = buffer.split(config.optimization.validation_fraction, seed)?;
    let training = split.training_examples(true);
    let validation = split.validation_examples();
    config.optimization.learning_rate = 0.00025;
    let (state, _) = load_training_generation::<MetalTrainingBackend>(
        &args[1],
        21,
        &config.optimization,
        &device,
    )?
    .ok_or("reference training state missing")?;
    let start = Instant::now();
    let (state, training_report) = train_state_with_progress(
        state,
        &training,
        &validation,
        &config.optimization,
        seed,
        &device,
        &|_| {},
    );
    let optimizer_seconds = start.elapsed().as_secs_f64();
    let mut trained = NetworkEvaluator::new(state.model.valid(), device);
    let prediction = trained.evaluate_batch(&[request])?;
    let result = json!({
        "seed":seed,"reference":metadata,"inference":inference,
        "self_play":{"seconds":self_play_seconds,"games":games.len(),"examples":examples,
            "positions":stats.positions,"backend_batches":stats.backend_batches,
            "backend_seconds":stats.backend_time.as_secs_f64(),"replays":replays},
        "optimizer":{"seconds":optimizer_seconds,"training_examples":training.len(),
            "validation_examples":validation.len(),"report":training_report,
            "post_update_policy":prediction[0].policy.as_slice(),"post_update_value":prediction[0].value},
    });
    serde_json::to_writer_pretty(output, &result)?;
    println!(
        "self_play_s={self_play_seconds:.3} examples={examples} optimizer_s={optimizer_seconds:.3}"
    );
    Ok(())
}
