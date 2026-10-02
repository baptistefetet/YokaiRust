//! Reproducible CPU-only engine benchmarks, with no network or checkpoint.
//!
//! Run `cargo bench --no-default-features --bench engine -- all`, or replace
//! `all` with a name substring. `YOKAI_BENCH_ITERATIONS` overrides each case's
//! iteration count per sample. Five samples follow a short untimed warm-up.

use std::{env, error::Error, hint::black_box, time::Instant};

use yokai::{EvaluationRequest, Game, Mcts, Player, SearchConfig, UniformEvaluator};

const SAMPLES: usize = 5;
const PERFT_DEPTH: u8 = 6;

fn main() -> Result<(), Box<dyn Error>> {
    let filter = env::args()
        .skip(1)
        .find(|argument| !argument.starts_with('-'));
    let iterations = env::var("YOKAI_BENCH_ITERATIONS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if iterations == Some(0) {
        return Err("YOKAI_BENCH_ITERATIONS must be positive".into());
    }
    let initial = Game::new(Player::First);
    let middle = middle_game();
    let benchmarks = [
        (
            "request_initial",
            100_000,
            &initial,
            prepare_request as fn(&Game),
        ),
        ("request_middle", 50_000, &middle, prepare_request),
        ("perft_initial_depth6", 100, &initial, count_tree),
        ("mcts_initial_200", 500, &initial, search),
        ("mcts_middle_200", 500, &middle, search),
    ];
    let Some(filter) = filter else {
        eprintln!("Usage: cargo bench --no-default-features --bench engine -- <name>|all");
        for (name, _, _, _) in benchmarks {
            eprintln!("  {name}");
        }
        return Ok(());
    };
    println!(
        "CPU engine: samples={SAMPLES}, initial_legal={}, middle_ply={}, middle_legal={}",
        initial.legal_actions().len(),
        middle.actions().len(),
        middle.legal_actions().len(),
    );
    println!("MCTS: simulations=200, batch=8, seed=42, fresh tree, no root noise");
    let mut matched = false;
    for (name, default_iterations, game, benchmark) in benchmarks {
        if filter != "all" && !name.contains(&filter) {
            continue;
        }
        matched = true;
        if name.starts_with("perft_") {
            println!(
                "perft depth={PERFT_DEPTH}, leaves={}",
                perft(game, PERFT_DEPTH)
            );
        }
        measure(name, iterations.unwrap_or(default_iterations), || {
            benchmark(black_box(game));
        });
    }
    if !matched {
        return Err(format!("no benchmark matches `{filter}`").into());
    }
    Ok(())
}

fn measure(name: &str, iterations: usize, mut operation: impl FnMut()) {
    for _ in 0..(iterations / 10).max(1) {
        operation();
    }
    let mut times = [0.0_f64; SAMPLES];
    for elapsed in &mut times {
        let started = Instant::now();
        for _ in 0..iterations {
            operation();
        }
        *elapsed = started.elapsed().as_secs_f64() * 1.0e9 / iterations as f64;
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{name}: iterations={iterations}, median_ns={:.1}, min_ns={:.1}, max_ns={:.1}",
        times[SAMPLES / 2],
        times[0],
        times[SAMPLES - 1],
    );
}

fn prepare_request(game: &Game) {
    black_box(EvaluationRequest::from_game(game));
}

fn search(game: &Game) {
    let config = SearchConfig {
        simulations: 200,
        evaluation_batch_size: 8,
        dirichlet_weight: 0.0,
        ..SearchConfig::default()
    };
    let mut search = Mcts::new(UniformEvaluator, config, 42).expect("valid benchmark search");
    black_box(search.search(game, 0.0).expect("ongoing benchmark game"));
}

fn count_tree(game: &Game) {
    black_box(perft(game, PERFT_DEPTH));
}

// Count paths reaching exactly `depth` plies. Earlier terminal states contribute
// zero. Cloning Game deliberately includes the official repetition rules.
fn perft(game: &Game, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut leaves = 0;
    for action in game.legal_actions() {
        let mut child = game.clone();
        child.apply(action).expect("generated action must be legal");
        leaves += perft(&child, depth - 1);
    }
    leaves
}

fn middle_game() -> Game {
    // A fixed legal trace with captures and drops, independent of RNG versions,
    // local training files and the accepted champion. Absolute board notation.
    let mut game = Game::new(Player::First);
    for action in [
        "b1-c2",
        "b4-a3",
        "c2-b1",
        "b3-b2",
        "b1-c2",
        "a3-b4",
        "c2-b2",
        "kodama@b1",
        "kodama@b3",
        "c4-b3",
        "c1-c2",
        "kodama@c4",
        "c2-c3",
        "c4-c3",
        "b2-c1",
        "tanuki@c2",
        "c1-b1",
        "b3-a2",
        "b1-a2",
        "c2-b2",
        "a2-b2",
        "b4-c4",
    ] {
        game.apply(action.parse().expect("valid fixture notation"))
            .expect("legal fixture action");
    }
    assert!(!game.outcome().is_terminal());
    game
}
