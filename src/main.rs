use std::{
    env,
    error::Error,
    io,
    path::Path,
    process::ExitCode,
    time::{Duration, Instant},
};

use burn::prelude::Backend;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use yokai::{
    BackendKind, CachedEvaluator, CpuBackend, CpuTrainingBackend, Game, Mcts, MetalBackend,
    MetalTrainingBackend, Replay, SearchConfig, TrainingConfig, TrainingProgress, UniformEvaluator,
    bootstrap_champion, load_replay_buffer, run_generation_with_progress,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    match arguments.next().as_deref() {
        None | Some("help" | "--help" | "-h") => print_help(),
        Some("analyze") => {
            let simulations = arguments
                .next()
                .map_or(Ok(200), |value| value.parse::<u32>())?;
            let seed = arguments
                .next()
                .map_or(Ok(42), |value| value.parse::<u64>())?;
            reject_extra_argument(arguments.next())?;
            analyze_initial_position(simulations, seed)?;
        }
        Some("replay") => {
            let path = arguments
                .next()
                .ok_or_else(|| invalid_input("replay requires a JSON file path"))?;
            reject_extra_argument(arguments.next())?;
            print_replay(&Replay::read_json(path)?);
        }
        Some("train") => {
            let arguments = arguments.collect::<Vec<_>>();
            train(&arguments)?;
        }
        Some(command) => return Err(invalid_input(format!("unknown command `{command}`")).into()),
    }
    Ok(())
}

fn train(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let mut config_path = "config/training.toml".to_owned();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--config" => {
                index += 1;
                config_path.clone_from(
                    arguments
                        .get(index)
                        .ok_or_else(|| invalid_input("--config requires a TOML path"))?,
                );
            }
            "--resume" => {
                index += 1;
                let resume = arguments
                    .get(index)
                    .ok_or_else(|| invalid_input("--resume requires `latest`"))?;
                if resume != "latest" {
                    return Err(invalid_input("only `--resume latest` is supported").into());
                }
            }
            "--headless" => {}
            argument => {
                return Err(
                    invalid_input(format!("unexpected train argument `{argument}`")).into(),
                );
            }
        }
        index += 1;
    }

    let config = TrainingConfig::load(&config_path)?;
    let started = Instant::now();
    eprintln!(
        "[{}] configuration={} backend={:?}",
        elapsed_text(started.elapsed()),
        config_path,
        config.backend
    );
    match config.backend {
        BackendKind::Cpu => {
            let device = burn::backend::flex::FlexDevice;
            CpuBackend::seed(&device, config.seed);
            CpuTrainingBackend::seed(&device, config.seed);
            let champion = bootstrap_champion::<CpuBackend>(
                &config.paths.models,
                config.network.clone(),
                &device,
            )?;
            eprintln!(
                "[{}] champion generation={} ready",
                elapsed_text(started.elapsed()),
                champion.generation
            );
            let mut buffer = load_replay_buffer(
                Path::new(&config.paths.self_play).join("buffer.json"),
                config.optimization.replay_buffer,
            )?;
            let progress = |event| print_training_progress(started, event);
            let report = run_generation_with_progress::<CpuTrainingBackend, _>(
                &config,
                &mut buffer,
                &device,
                &progress,
            )?;
            print_generation_report(&report);
        }
        BackendKind::Metal => {
            let device = burn::backend::wgpu::WgpuDevice::default();
            MetalBackend::seed(&device, config.seed);
            MetalTrainingBackend::seed(&device, config.seed);
            let champion = bootstrap_champion::<MetalBackend>(
                &config.paths.models,
                config.network.clone(),
                &device,
            )?;
            eprintln!(
                "[{}] champion generation={} ready",
                elapsed_text(started.elapsed()),
                champion.generation
            );
            let mut buffer = load_replay_buffer(
                Path::new(&config.paths.self_play).join("buffer.json"),
                config.optimization.replay_buffer,
            )?;
            let progress = |event| print_training_progress(started, event);
            let report = run_generation_with_progress::<MetalTrainingBackend, _>(
                &config,
                &mut buffer,
                &device,
                &progress,
            )?;
            print_generation_report(&report);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn print_training_progress(started: Instant, event: TrainingProgress) {
    let elapsed = elapsed_text(started.elapsed());
    match event {
        TrainingProgress::GenerationStarted {
            champion_generation,
            candidate_generation,
        } => eprintln!(
            "[{elapsed}] generation {candidate_generation} started from champion {champion_generation}"
        ),
        TrainingProgress::SelfPlayStarted {
            games,
            workers,
            simulations,
        } => eprintln!(
            "[{elapsed}] self-play started: {games} games, {workers} workers, {simulations} simulations/move"
        ),
        TrainingProgress::SelfPlayAdvanced { completed, total } => eprintln!(
            "[{elapsed}] self-play {completed}/{total} ({:.1}%)",
            percentage(completed, total)
        ),
        TrainingProgress::SelfPlayFinished {
            games,
            examples,
            outcomes,
        } => eprintln!(
            "[{elapsed}] self-play finished: {games} games, {examples} examples, first/second/draw={}/{}/{}",
            outcomes.first_wins, outcomes.second_wins, outcomes.draws
        ),
        TrainingProgress::DatasetReady {
            buffer_games,
            training_examples,
            validation_examples,
        } => eprintln!(
            "[{elapsed}] dataset ready: {buffer_games} games, {training_examples} train examples, {validation_examples} validation examples"
        ),
        TrainingProgress::TrainingStarted { epochs, batch_size } => {
            eprintln!("[{elapsed}] training started: {epochs} epochs, batch size {batch_size}");
        }
        TrainingProgress::EpochFinished {
            total_epochs,
            report,
        } => {
            eprintln!(
                "[{elapsed}] epoch {}/{}: train policy={:.4} value={:.4} entropy={:.4} illegal={:.4} top1={:.3}",
                report.epoch,
                total_epochs,
                report.training.policy_loss,
                report.training.value_loss,
                report.training.policy_entropy,
                report.training.illegal_policy_mass,
                report.training.policy_top1_accuracy
            );
            if let Some(validation) = report.validation {
                eprintln!(
                    "[{elapsed}] epoch {}/{}: valid policy={:.4} value={:.4} calibration={:.4} top1={:.3}",
                    report.epoch,
                    total_epochs,
                    validation.policy_loss,
                    validation.value_loss,
                    validation.value_calibration_error,
                    validation.policy_top1_accuracy
                );
            }
        }
        TrainingProgress::CandidateSaved { generation } => {
            eprintln!("[{elapsed}] candidate generation {generation} saved");
        }
        TrainingProgress::ArenaStarted { games, simulations } => {
            eprintln!("[{elapsed}] arena started: {games} games, {simulations} simulations/move");
        }
        TrainingProgress::ArenaAdvanced { completed, total } => eprintln!(
            "[{elapsed}] arena {completed}/{total} ({:.1}%)",
            percentage(completed, total)
        ),
        TrainingProgress::ArenaFinished(result) => eprintln!(
            "[{elapsed}] arena finished: candidate/champion/draw={}/{}/{}, score={:.3}",
            result.candidate_wins, result.champion_wins, result.draws, result.score
        ),
        TrainingProgress::ChampionPromoted { generation } => {
            eprintln!("[{elapsed}] generation {generation} promoted to champion");
        }
        TrainingProgress::CandidateRejected { generation } => {
            eprintln!("[{elapsed}] generation {generation} rejected; champion unchanged");
        }
    }
}

fn elapsed_text(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3_600,
        (seconds % 3_600) / 60,
        seconds % 60
    )
}

#[allow(clippy::cast_precision_loss)]
fn percentage(completed: usize, total: usize) -> f64 {
    100.0 * completed as f64 / total as f64
}

fn print_generation_report(report: &yokai::GenerationReport) {
    println!(
        "generation={} champion={} games={} buffer_examples={}",
        report.candidate_generation,
        report.champion_generation,
        report.generated_games,
        report.buffer_examples
    );
    if let Some(epoch) = report.training.epochs.last() {
        println!(
            "train policy_loss={:.4} value_loss={:.4} entropy={:.4} calibration={:.4} illegal_mass={:.4} top1={:.3}",
            epoch.training.policy_loss,
            epoch.training.value_loss,
            epoch.training.policy_entropy,
            epoch.training.value_calibration_error,
            epoch.training.illegal_policy_mass,
            epoch.training.policy_top1_accuracy
        );
        if let Some(validation) = epoch.validation {
            println!(
                "valid policy_loss={:.4} value_loss={:.4} entropy={:.4} calibration={:.4} illegal_mass={:.4} top1={:.3}",
                validation.policy_loss,
                validation.value_loss,
                validation.policy_entropy,
                validation.value_calibration_error,
                validation.illegal_policy_mass,
                validation.policy_top1_accuracy
            );
        }
    }
    println!(
        "arena candidate={} champion={} draws={} score={:.3} promoted={}",
        report.arena.candidate_wins,
        report.arena.champion_wins,
        report.arena.draws,
        report.arena.score,
        report.promoted()
    );
}

fn analyze_initial_position(simulations: u32, seed: u64) -> Result<(), Box<dyn Error>> {
    let mut starting_rng = ChaCha8Rng::seed_from_u64(seed);
    let game = Game::new_random(&mut starting_rng);
    let config = SearchConfig {
        simulations,
        ..SearchConfig::default()
    };
    let evaluator = CachedEvaluator::new(UniformEvaluator, 16_384);
    let mut search = Mcts::new(evaluator, config, seed)?;
    let result = search.search(&game, 0.0)?;

    println!(
        "starting_player={:?} simulations={} seed={} root_value={:+.3}",
        game.initial_player(),
        simulations,
        seed,
        result.root_value
    );
    println!("best={}", result.best_action);
    println!("{}", result.analysis_text());
    Ok(())
}

fn print_replay(replay: &Replay) {
    println!(
        "replay v{} rules v{} seed={:?} first={:?}",
        replay.format_version, replay.rules_version, replay.seed, replay.initial_player
    );
    for (ply, action) in replay.actions.iter().enumerate() {
        println!("{:>3}. {}", ply + 1, action);
        if let Some(analyses) = &replay.analyses {
            for entry in &analyses[ply] {
                println!(
                    "     {} prior={:.3} visits={} policy={:.3} q={:+.3}",
                    entry.action, entry.prior, entry.visits, entry.visit_probability, entry.q_value
                );
            }
        }
    }
    println!("outcome={:?}", replay.outcome);
}

fn reject_extra_argument(argument: Option<String>) -> Result<(), io::Error> {
    if let Some(argument) = argument {
        return Err(invalid_input(format!(
            "unexpected extra argument `{argument}`"
        )));
    }
    Ok(())
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn print_help() {
    println!(
        r"YokaiRust

Commands:
  yokai analyze [simulations] [seed]  Analyze the initial position with pure MCTS
  yokai replay <file.json>             Validate and print a recorded game
  yokai train [--config FILE] [--resume latest] [--headless]
                                       Run one AlphaZero generation"
    );
}
