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

struct TrainArguments {
    config_path: String,
    generations: usize,
}

fn parse_train_arguments(arguments: &[String]) -> Result<TrainArguments, io::Error> {
    let mut config_path = "config/training.toml".to_owned();
    let mut generations = 1_usize;
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
                    return Err(invalid_input("only `--resume latest` is supported"));
                }
            }
            "--generations" => {
                index += 1;
                generations = arguments
                    .get(index)
                    .ok_or_else(|| invalid_input("--generations requires a positive integer"))?
                    .parse()
                    .map_err(|_| invalid_input("--generations requires a positive integer"))?;
                if generations == 0 {
                    return Err(invalid_input("--generations must be greater than zero"));
                }
            }
            "--headless" => {}
            argument => {
                return Err(invalid_input(format!(
                    "unexpected train argument `{argument}`"
                )));
            }
        }
        index += 1;
    }
    Ok(TrainArguments {
        config_path,
        generations,
    })
}

fn train(arguments: &[String]) -> Result<(), Box<dyn Error>> {
    let TrainArguments {
        config_path,
        generations,
    } = parse_train_arguments(arguments)?;
    let config = TrainingConfig::load(&config_path)?;
    let started = Instant::now();
    eprintln!(
        "[{}] configuration={} backend={:?} generations={generations}",
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
                "[{}] champion network generation={} ready",
                elapsed_text(started.elapsed()),
                champion.generation
            );
            let mut buffer = load_replay_buffer(
                Path::new(&config.paths.self_play).join("buffer.json"),
                config.optimization.replay_buffer,
            )?;
            let progress = |event| print_training_progress(started, &event);
            for _ in 0..generations {
                let report = run_generation_with_progress::<CpuTrainingBackend, _>(
                    &config,
                    &mut buffer,
                    &device,
                    &progress,
                )?;
                print_generation_report(&report);
            }
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
                "[{}] champion network generation={} ready",
                elapsed_text(started.elapsed()),
                champion.generation
            );
            let mut buffer = load_replay_buffer(
                Path::new(&config.paths.self_play).join("buffer.json"),
                config.optimization.replay_buffer,
            )?;
            let progress = |event| print_training_progress(started, &event);
            for _ in 0..generations {
                let report = run_generation_with_progress::<MetalTrainingBackend, _>(
                    &config,
                    &mut buffer,
                    &device,
                    &progress,
                )?;
                print_generation_report(&report);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn print_training_progress(started: Instant, event: &TrainingProgress) {
    let elapsed = elapsed_text(started.elapsed());
    match event {
        TrainingProgress::GenerationStarted {
            source_generation,
            learner_generation,
            candidate_generation,
        } => eprintln!(
            "[{elapsed}] generation {candidate_generation} started: champion {source_generation}, learner {learner_generation}"
        ),
        TrainingProgress::SelfPlayStarted {
            games,
            workers,
            simulations,
            search_batch_size,
            repetition_contempt,
            starter_draw_value,
            restart_archive,
            planned_restarts,
        } => eprintln!(
            "[{elapsed}] self-play started: {games} games ({planned_restarts} targeted restarts from {restart_archive} prefixes), {workers} workers, {simulations} simulations/move, {search_batch_size} leaves/inference, repetition contempt={repetition_contempt:.2}, starter draw value={starter_draw_value:.2}"
        ),
        TrainingProgress::SelfPlayAdvanced { completed, total } => eprintln!(
            "[{elapsed}] self-play {completed}/{total} ({:.1}%)",
            percentage(*completed, *total)
        ),
        TrainingProgress::SelfPlayFinished {
            games,
            restarted_games,
            examples,
            outcomes,
            inference,
        } => {
            eprintln!(
                "[{elapsed}] self-play finished: {games} games ({restarted_games} targeted restarts), {examples} examples, absolute first/second/draw={}/{}/{}, mover starter/non-starter/unclassified={}/{}/{}",
                outcomes.first_wins,
                outcomes.second_wins,
                outcomes.draws,
                outcomes.starter_wins,
                outcomes.non_starter_wins,
                outcomes.unclassified_wins,
            );
            print_inference_stats(&elapsed, "self-play inference", inference);
        }
        TrainingProgress::SelfPlayResumed {
            games,
            examples,
            restarted_games,
        } => eprintln!(
            "[{elapsed}] self-play resumed from {games} persisted games ({restarted_games} targeted restarts, {examples} examples); generation will not be duplicated"
        ),
        TrainingProgress::DatasetReady {
            buffer_games,
            training_games,
            validation_games,
            training_examples,
            validation_examples,
            terminal_window_plies,
            terminal_extra_examples,
            terminal_oversampling,
        } => {
            let selection = if *terminal_oversampling {
                format!(
                    "all positions + {terminal_extra_examples} oversampled examples from the last {} decisive plies",
                    terminal_window_plies.unwrap_or_default(),
                )
            } else {
                terminal_window_plies.map_or_else(
                    || "all positions".to_owned(),
                    |plies| format!("last {plies} plies of decisive games"),
                )
            };
            eprintln!(
                "[{elapsed}] dataset ready: {buffer_games} buffered games; {selection}; train={training_games} games/{training_examples} examples, valid={validation_games} games/{validation_examples} examples"
            );
        }
        TrainingProgress::TrainingStarted {
            steps,
            batch_size,
            validation_interval_steps,
            optimizer_resumed,
        } => eprintln!(
            "[{elapsed}] training started: {steps} optimizer steps, batch size {batch_size}, metrics every {validation_interval_steps} steps, Adam resumed={optimizer_resumed}"
        ),
        TrainingProgress::TrainingAdvanced {
            total_steps,
            report,
        } => {
            eprintln!(
                "[{elapsed}] step {}/{}: train policy={:.4} WDL={:.4} policy_weight={:.3} entropy={:.4} illegal={:.4} policy_top1={:.3} WDL_top1={:.3} draw_error={:.3}",
                report.step,
                total_steps,
                report.training.policy_loss,
                report.training.value_loss,
                report.training.mean_policy_weight,
                report.training.policy_entropy,
                report.training.illegal_policy_mass,
                report.training.policy_top1_accuracy,
                report.training.wdl_top1_accuracy,
                report.training.draw_probability_error,
            );
            if let Some(validation) = report.validation {
                eprintln!(
                    "[{elapsed}] step {}/{}: valid policy={:.4} WDL={:.4} policy_weight={:.3} value_calibration={:.4} policy_top1={:.3} WDL_top1={:.3} draw_error={:.3}",
                    report.step,
                    total_steps,
                    validation.policy_loss,
                    validation.value_loss,
                    validation.mean_policy_weight,
                    validation.value_calibration_error,
                    validation.policy_top1_accuracy,
                    validation.wdl_top1_accuracy,
                    validation.draw_probability_error,
                );
            }
        }
        TrainingProgress::TrainingFinished { completed_steps } => {
            eprintln!("[{elapsed}] training finished after {completed_steps} optimizer steps");
        }
        TrainingProgress::CandidateSaved { generation } => {
            eprintln!("[{elapsed}] candidate generation {generation} saved");
        }
        TrainingProgress::ArenaStarted {
            games,
            workers,
            simulations,
            search_batch_size,
            opening_plies,
        } => eprintln!(
            "[{elapsed}] arena started: {games} games, {workers} workers, {simulations} simulations/move, {search_batch_size} leaf/inference, paired random openings=0-{opening_plies} plies; progress is completion-ordered"
        ),
        TrainingProgress::ArenaAdvanced { progress } => eprintln!(
            "[{elapsed}] arena {}/{} ({:.1}%): candidate/reference/draw={}/{}/{}, current_score={:.3}",
            progress.completed,
            progress.total,
            percentage(progress.completed, progress.total),
            progress.candidate_wins,
            progress.reference_wins,
            progress.draws,
            progress.score()
        ),
        TrainingProgress::ArenaFinished {
            result,
            candidate_inference,
            reference_inference,
        } => {
            eprintln!(
                "[{elapsed}] arena finished: candidate/reference/draw={}/{}/{}, score={:.3}, distinct_openings={}",
                result.candidate_wins,
                result.reference_wins,
                result.draws,
                result.score,
                result.distinct_openings,
            );
            print_arena_seats(&elapsed, result);
            print_inference_stats(&elapsed, "candidate inference", candidate_inference);
            print_inference_stats(&elapsed, "reference inference", reference_inference);
        }
        TrainingProgress::CandidateMirrorStarted {
            games,
            simulations,
            max_draw_rate,
        } => eprintln!(
            "[{elapsed}] candidate mirror started: {games} games, {simulations} simulations/move, maximum draw rate={:.1}%",
            max_draw_rate * 100.0
        ),
        TrainingProgress::CandidateMirrorAdvanced { progress } => eprintln!(
            "[{elapsed}] candidate mirror {}/{} ({:.1}%): draws={}",
            progress.completed,
            progress.total,
            percentage(progress.completed, progress.total),
            progress.draws,
        ),
        TrainingProgress::CandidateMirrorFinished {
            result,
            draw_rate,
            within_configured_limit,
        } => eprintln!(
            "[{elapsed}] candidate mirror finished: draws={}/{} ({:.1}%), gate_passed={within_configured_limit}",
            result.draws,
            result.candidate_wins + result.reference_wins + result.draws,
            draw_rate * 100.0,
        ),
        TrainingProgress::CandidateSelfPlayStarted {
            games,
            simulations,
            max_draw_rate,
        } => eprintln!(
            "[{elapsed}] candidate exploratory probe started: {games} games, {simulations} simulations/move, maximum draw rate={:.1}%",
            max_draw_rate * 100.0
        ),
        TrainingProgress::CandidateSelfPlayAdvanced { completed, total } => eprintln!(
            "[{elapsed}] candidate exploratory probe {completed}/{total} ({:.1}%)",
            percentage(*completed, *total),
        ),
        TrainingProgress::CandidateSelfPlayFinished {
            outcomes,
            draw_rate,
            within_configured_limit,
        } => eprintln!(
            "[{elapsed}] candidate exploratory probe finished: absolute first/second/draw={}/{}/{}, mover starter/non-starter/unclassified={}/{}/{}, draw_rate={:.1}%, gate_passed={within_configured_limit}",
            outcomes.first_wins,
            outcomes.second_wins,
            outcomes.draws,
            outcomes.starter_wins,
            outcomes.non_starter_wins,
            outcomes.unclassified_wins,
            draw_rate * 100.0,
        ),
        TrainingProgress::ChampionPromoted { generation } => {
            eprintln!("[{elapsed}] candidate {generation} promoted as champion");
        }
        TrainingProgress::CandidateRejected {
            generation,
            decision,
            learner_generation,
        } => {
            eprintln!(
                "[{elapsed}] candidate {generation} rejected: arena={} mirror_draw_gate={} exploratory_draw_gate={}; learner now {learner_generation}",
                decision.arena_passed,
                decision.mirror_draw_gate_passed,
                decision.exploratory_draw_gate_passed,
            );
        }
    }
}

fn print_inference_stats(elapsed: &str, label: &str, stats: &yokai::InferenceStats) {
    eprintln!(
        "[{elapsed}] {label}: positions={} batches={} avg_batch={:.1} max_batch={} backend={:.1}s throughput={:.0} pos/s avg_wait={:.2}ms",
        stats.positions,
        stats.backend_batches,
        stats.average_batch_size(),
        stats.maximum_batch_size,
        stats.backend_time.as_secs_f64(),
        stats.positions_per_backend_second(),
        stats.average_client_wait().as_secs_f64() * 1_000.0
    );
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
        "generation={} source_champion={} source_learner={} promoted={} games={} restarted_games={} buffer_examples={} terminal_window={:?} terminal_extra_examples={} terminal_oversampling={}",
        report.candidate_generation,
        report.source_generation,
        report.learner_source_generation,
        report.promoted(),
        report.generated_games,
        report.restarted_games,
        report.buffer_examples,
        report.terminal_window_plies,
        report.terminal_extra_examples,
        report.terminal_oversampling,
    );
    if let Some(checkpoint) = report.training.selected() {
        println!("training metrics at step={}", checkpoint.step);
        println!(
            "train policy_loss={:.4} WDL_loss={:.4} policy_weight={:.3} entropy={:.4} calibration={:.4} illegal_mass={:.4} policy_top1={:.3} WDL_top1={:.3} draw_error={:.3}",
            checkpoint.training.policy_loss,
            checkpoint.training.value_loss,
            checkpoint.training.mean_policy_weight,
            checkpoint.training.policy_entropy,
            checkpoint.training.value_calibration_error,
            checkpoint.training.illegal_policy_mass,
            checkpoint.training.policy_top1_accuracy,
            checkpoint.training.wdl_top1_accuracy,
            checkpoint.training.draw_probability_error,
        );
        if let Some(validation) = checkpoint.validation {
            println!(
                "valid policy_loss={:.4} WDL_loss={:.4} policy_weight={:.3} entropy={:.4} calibration={:.4} illegal_mass={:.4} policy_top1={:.3} WDL_top1={:.3} draw_error={:.3}",
                validation.policy_loss,
                validation.value_loss,
                validation.mean_policy_weight,
                validation.policy_entropy,
                validation.value_calibration_error,
                validation.illegal_policy_mass,
                validation.policy_top1_accuracy,
                validation.wdl_top1_accuracy,
                validation.draw_probability_error,
            );
        }
    }
    let initial = report.initial_self_play_outcomes;
    let restarted = report.restarted_self_play_outcomes;
    println!(
        "self-play initial starter/non-starter/draw={}/{}/{}; restarted starter/non-starter/draw={}/{}/{}",
        initial.starter_wins,
        initial.non_starter_wins,
        initial.draws,
        restarted.starter_wins,
        restarted.non_starter_wins,
        restarted.draws,
    );
    print_dataset_diagnostics("generated", report.generated_dataset_diagnostics);
    print_dataset_diagnostics("buffer", report.buffer_dataset_diagnostics);
    println!(
        "arena candidate={} previous={} draws={} score={:.3} threshold_reached={}",
        report.arena.candidate_wins,
        report.arena.reference_wins,
        report.arena.draws,
        report.arena.score,
        report.arena_threshold_reached()
    );
    print_arena_seats("summary", &report.arena);
    let mirror_games = report.candidate_mirror.candidate_wins
        + report.candidate_mirror.reference_wins
        + report.candidate_mirror.draws;
    println!(
        "candidate mirror draws={}/{} ({:.1}%)",
        report.candidate_mirror.draws,
        mirror_games,
        percentage(report.candidate_mirror.draws, mirror_games),
    );
    let outcomes = report.candidate_self_play;
    let games = outcomes.first_wins + outcomes.second_wins + outcomes.draws;
    println!(
        "candidate exploratory draws={}/{} ({:.1}%)",
        outcomes.draws,
        games,
        percentage(outcomes.draws, games),
    );
}

fn print_dataset_diagnostics(label: &str, diagnostics: yokai::DatasetDiagnostics) {
    for (bucket, metrics) in [
        ("all", diagnostics.all),
        ("draw", diagnostics.draws),
        ("draw-starter", diagnostics.draw_starter),
        ("draw-non-starter", diagnostics.draw_non_starter),
        ("decisive", diagnostics.decisive),
    ] {
        println!(
            "{label} {bucket}: positions={} target_entropy={:.3} max_probability={:.3} legal_coverage={:.3} repetition_mass={:.3} immediate_draw_positions={} immediate_draw_mass={:.3}",
            metrics.positions,
            metrics.mean_entropy,
            metrics.mean_max_probability,
            metrics.mean_legal_action_coverage,
            metrics.mean_repetition_mass,
            metrics.immediate_draw_positions,
            metrics.mean_immediate_draw_mass,
        );
    }
}

fn print_arena_seats(elapsed: &str, result: &yokai::ArenaResult) {
    let first = result.candidate_as_first;
    let second = result.candidate_as_second;
    eprintln!(
        "[{elapsed}] arena by seat: candidate as First W/L/D={}/{}/{} score={:.3}; as Second W/L/D={}/{}/{} score={:.3}",
        first.wins,
        first.losses,
        first.draws,
        first.score(),
        second.wins,
        second.losses,
        second.draws,
        second.score(),
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
  yokai train [--config FILE] [--resume latest] [--generations N] [--headless]
                                       Run N AlphaZero generations (default: 1)"
    );
}

#[cfg(test)]
mod tests {
    use super::parse_train_arguments;

    #[test]
    fn train_arguments_default_to_one_generation() {
        let parsed = parse_train_arguments(&[]).expect("default train arguments");
        assert_eq!(parsed.config_path, "config/training.toml");
        assert_eq!(parsed.generations, 1);
    }

    #[test]
    fn train_arguments_accept_an_explicit_generation_count() {
        let arguments = [
            "--config".to_owned(),
            "custom.toml".to_owned(),
            "--generations".to_owned(),
            "5".to_owned(),
            "--headless".to_owned(),
        ];
        let parsed = parse_train_arguments(&arguments).expect("explicit train arguments");
        assert_eq!(parsed.config_path, "custom.toml");
        assert_eq!(parsed.generations, 5);
    }

    #[test]
    fn train_arguments_reject_zero_generations() {
        let arguments = ["--generations".to_owned(), "0".to_owned()];
        assert!(parse_train_arguments(&arguments).is_err());
    }
}
