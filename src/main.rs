use std::{env, error::Error, io, process::ExitCode};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use yokai::{CachedEvaluator, Game, Mcts, Replay, SearchConfig, UniformEvaluator};

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
        Some(command) => return Err(invalid_input(format!("unknown command `{command}`")).into()),
    }
    Ok(())
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
        "YokaiRust\n\n\
         Commands:\n\
           yokai analyze [simulations] [seed]  Analyze the initial position with pure MCTS\n\
           yokai replay <file.json>             Validate and print a recorded game"
    );
}
