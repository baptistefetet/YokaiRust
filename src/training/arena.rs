//! Paired, noise-free candidate versus champion evaluation.

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Evaluator, Game, Mcts, MoveError, Outcome, Player, SearchConfig, SearchError,
    training::config::ArenaConfig,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArenaResult {
    pub candidate_wins: usize,
    pub champion_wins: usize,
    pub draws: usize,
    pub score: f32,
    pub promoted: bool,
}

/// Runs paired seeds with alternating candidate colors, no root noise, and
/// temperature zero.
///
/// # Errors
///
/// Returns [`ArenaError`] if a worker, search, move, or safety limit fails.
pub fn run_arena<C, H>(
    candidate: &C,
    champion: &H,
    config: &ArenaConfig,
    workers: usize,
    max_game_plies: usize,
    base_seed: u64,
) -> Result<ArenaResult, ArenaError>
where
    C: Evaluator + Clone + Send + Sync,
    H: Evaluator + Clone + Send + Sync,
{
    run_arena_with_progress(
        candidate,
        champion,
        config,
        workers,
        max_game_plies,
        base_seed,
        &|_, _| {},
    )
}

/// Runs an arena and reports each successfully completed game.
///
/// The callback follows the same concurrent completion semantics as self-play.
///
/// # Errors
///
/// Returns [`ArenaError`] if a worker, search, move, or safety limit fails.
#[allow(clippy::too_many_arguments)]
pub fn run_arena_with_progress<C, H, F>(
    candidate: &C,
    champion: &H,
    config: &ArenaConfig,
    workers: usize,
    max_game_plies: usize,
    base_seed: u64,
    progress: &F,
) -> Result<ArenaResult, ArenaError>
where
    C: Evaluator + Clone + Send + Sync,
    H: Evaluator + Clone + Send + Sync,
    F: Fn(usize, usize) + Sync,
{
    if workers == 0 || config.games == 0 {
        return Err(ArenaError::InvalidConfiguration);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("yokai-arena-{index}"))
        .build()?;
    let completed = AtomicUsize::new(0);
    let outcomes = pool.install(|| {
        (0..config.games)
            .into_par_iter()
            .map(|game_index| {
                let paired_seed = base_seed.wrapping_add((game_index / 2) as u64);
                let candidate_player = if game_index % 2 == 0 {
                    Player::First
                } else {
                    Player::Second
                };
                let outcome = play_arena_game(
                    (*candidate).clone(),
                    (*champion).clone(),
                    candidate_player,
                    config.simulations,
                    max_game_plies,
                    paired_seed,
                )?;
                let completed = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress(completed, config.games);
                Ok(outcome)
            })
            .collect::<Result<Vec<_>, ArenaError>>()
    })?;

    let mut candidate_wins = 0;
    let mut champion_wins = 0;
    let mut draws = 0;
    for outcome in outcomes {
        match outcome {
            ArenaGameOutcome::CandidateWin => candidate_wins += 1,
            ArenaGameOutcome::ChampionWin => champion_wins += 1,
            ArenaGameOutcome::Draw => draws += 1,
        }
    }
    let score = score(candidate_wins, draws, config.games);
    Ok(ArenaResult {
        candidate_wins,
        champion_wins,
        draws,
        score,
        promoted: score >= config.promotion_score,
    })
}

fn play_arena_game<C: Evaluator, H: Evaluator>(
    candidate: C,
    champion: H,
    candidate_player: Player,
    simulations: u32,
    max_game_plies: usize,
    seed: u64,
) -> Result<ArenaGameOutcome, ArenaError> {
    let mut starting_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut game = Game::new_random(&mut starting_rng);
    let search_config = SearchConfig {
        simulations,
        ..SearchConfig::default()
    };
    let mut candidate_search = Mcts::new(candidate, search_config, seed.wrapping_mul(2))?;
    let mut champion_search = Mcts::new(
        champion,
        search_config,
        seed.wrapping_mul(2).wrapping_add(1),
    )?;

    while !game.outcome().is_terminal() {
        if game.actions().len() >= max_game_plies {
            return Err(ArenaError::PlyLimit(max_game_plies));
        }
        let result = if game.position().side_to_move() == candidate_player {
            candidate_search.search(&game, 0.0)?
        } else {
            champion_search.search(&game, 0.0)?
        };
        game.apply(result.best_action)?;
        // Both players keep a tree: the active search reuses its chosen child,
        // while the opponent can reuse the reply when it was already explored.
        let _candidate_reused = candidate_search.advance_root(result.best_action, &game);
        let _champion_reused = champion_search.advance_root(result.best_action, &game);
    }

    Ok(match game.outcome() {
        Outcome::Draw { .. } => ArenaGameOutcome::Draw,
        Outcome::Win { player, .. } if player == candidate_player => ArenaGameOutcome::CandidateWin,
        Outcome::Win { .. } => ArenaGameOutcome::ChampionWin,
        Outcome::Ongoing => unreachable!("arena loop ends only on a terminal game"),
    })
}

#[derive(Clone, Copy)]
enum ArenaGameOutcome {
    CandidateWin,
    ChampionWin,
    Draw,
}

fn score(wins: usize, draws: usize, games: usize) -> f32 {
    (count_as_f32(wins) + 0.5 * count_as_f32(draws)) / count_as_f32(games)
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f32(value: usize) -> f32 {
    value as f32
}

#[derive(Debug, Error)]
pub enum ArenaError {
    #[error("arena worker and game counts must be positive")]
    InvalidConfiguration,
    #[error(transparent)]
    Search(#[from] SearchError),
    #[error(transparent)]
    Move(#[from] MoveError),
    #[error("arena game exceeded the safety limit of {0} plies")]
    PlyLimit(usize),
    #[error("failed to create arena workers: {0}")]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),
}
