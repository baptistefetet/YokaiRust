//! Deterministic parallel self-play generation.

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    Evaluator, Game, Mcts, MoveError, SearchConfig, SearchError, SelfPlayGame, SelfPlayRecorder,
    TemperatureSchedule, TrainingDataError, training::config::SelfPlayConfig,
};

/// Plays one complete game with exploration enabled only at the MCTS root.
///
/// # Errors
///
/// Returns [`SelfPlayError`] for search, move, target, or safety-limit errors.
pub fn play_self_play_game<E: Evaluator>(
    evaluator: E,
    config: &SelfPlayConfig,
    generation: u32,
    seed: u64,
) -> Result<SelfPlayGame, SelfPlayError> {
    let mut starting_rng = ChaCha8Rng::seed_from_u64(seed);
    let mut game = Game::new_random(&mut starting_rng);
    let mut search = Mcts::new(
        evaluator,
        SearchConfig {
            simulations: config.simulations,
            evaluation_batch_size: config.search_batch_size,
            ..SearchConfig::default()
        },
        seed,
    )?;
    let schedule = TemperatureSchedule {
        exploration_plies: config.exploration_plies,
        exploration_temperature: config.exploration_temperature,
        final_temperature: config.final_temperature,
    };
    let mut recorder = SelfPlayRecorder::new();

    while !game.outcome().is_terminal() {
        if game.actions().len() >= config.max_game_plies {
            return Err(SelfPlayError::PlyLimit(config.max_game_plies));
        }
        let result = search.search_self_play_scheduled(&game, schedule)?;
        recorder.record(&game, &result)?;
        game.apply(result.selected_action)?;
        let _reused = search.advance_root(result.selected_action, &game);
    }
    recorder
        .finish(generation, seed, game.outcome())
        .map_err(Into::into)
}

/// Generates a stable seed-ordered batch on a dedicated Rayon pool.
///
/// # Errors
///
/// Returns [`SelfPlayError`] when the pool cannot start or any game fails.
pub fn generate_self_play<E>(
    evaluator: &E,
    config: &SelfPlayConfig,
    generation: u32,
    base_seed: u64,
) -> Result<Vec<SelfPlayGame>, SelfPlayError>
where
    E: Evaluator + Clone + Send + Sync,
{
    generate_self_play_with_progress(evaluator, config, generation, base_seed, &|_, _| {})
}

/// Generates self-play and reports each successfully completed game.
///
/// The callback can run concurrently on any self-play worker. `completed` is a
/// unique, monotonically allocated count in `1..=total`, although callbacks
/// from different threads may be observed a few lines out of order.
///
/// # Errors
///
/// Returns [`SelfPlayError`] when the pool cannot start or any game fails.
pub fn generate_self_play_with_progress<E, F>(
    evaluator: &E,
    config: &SelfPlayConfig,
    generation: u32,
    base_seed: u64,
    progress: &F,
) -> Result<Vec<SelfPlayGame>, SelfPlayError>
where
    E: Evaluator + Clone + Send + Sync,
    F: Fn(usize, usize) + Sync,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .thread_name(|index| format!("yokai-self-play-{index}"))
        .build()?;
    let completed = AtomicUsize::new(0);
    pool.install(|| {
        (0..config.games_per_generation)
            .into_par_iter()
            .map(|index| {
                let game = play_self_play_game(
                    (*evaluator).clone(),
                    config,
                    generation,
                    base_seed.wrapping_add(index as u64),
                )?;
                let completed = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress(completed, config.games_per_generation);
                Ok(game)
            })
            .collect()
    })
}

#[derive(Debug, Error)]
pub enum SelfPlayError {
    #[error(transparent)]
    Search(#[from] SearchError),
    #[error(transparent)]
    Move(#[from] MoveError),
    #[error(transparent)]
    Data(#[from] TrainingDataError),
    #[error("self-play exceeded the safety limit of {0} plies")]
    PlyLimit(usize),
    #[error("failed to create self-play workers: {0}")]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),
}
