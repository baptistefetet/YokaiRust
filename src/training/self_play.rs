//! Deterministic parallel self-play generation.

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use thiserror::Error;

use crate::{
    Evaluator, Game, Mcts, MoveError, Replay, ReplayError, SearchConfig, SearchError, SelfPlayGame,
    SelfPlayRecorder, TemperatureSchedule, TrainingDataError, training::config::SelfPlayConfig,
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
    play_self_play_game_from_restart(evaluator, config, generation, seed, None)
}

/// Plays one trajectory from the official initial state or a complete ongoing
/// historical prefix.
///
/// # Errors
///
/// Returns [`SelfPlayError`] for invalid prefixes, search, move, target, or
/// safety-limit errors.
pub fn play_self_play_game_from_restart<E: Evaluator>(
    evaluator: E,
    config: &SelfPlayConfig,
    generation: u32,
    seed: u64,
    restart: Option<&Replay>,
) -> Result<SelfPlayGame, SelfPlayError> {
    let mut game = if let Some(restart) = restart {
        let game = restart.to_game()?;
        if game.outcome().is_terminal() {
            return Err(SelfPlayError::TerminalRestart);
        }
        game
    } else {
        let mut starting_rng = ChaCha8Rng::seed_from_u64(seed);
        Game::new_random(&mut starting_rng)
    };
    let restart_ply = game.actions().len();
    let mut search = Mcts::new(
        evaluator,
        SearchConfig {
            simulations: config.simulations,
            evaluation_batch_size: config.search_batch_size,
            repetition_contempt: config.repetition_contempt,
            starter_draw_value: config.starter_draw_value,
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
        let trajectory_ply = game.actions().len().saturating_sub(restart_ply);
        let result = search.search_self_play(&game, schedule.for_ply(trajectory_ply))?;
        recorder.record(&game, &result)?;
        game.apply(result.selected_action)?;
        let _reused = search.advance_root(result.selected_action, &game);
    }
    recorder
        .finish_from_game(generation, seed, &game, restart_ply)
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
    generate_self_play_with_restarts_and_progress(
        evaluator,
        config,
        generation,
        base_seed,
        &[],
        &|_, _| {},
    )
}

/// Generates self-play with a deterministic fraction of games restarted from
/// cycle-adjacent historical prefixes.
///
/// # Errors
///
/// Returns [`SelfPlayError`] when the pool cannot start or any game fails.
pub fn generate_self_play_with_restarts<E>(
    evaluator: &E,
    config: &SelfPlayConfig,
    generation: u32,
    base_seed: u64,
    restart_archive: &[Replay],
) -> Result<Vec<SelfPlayGame>, SelfPlayError>
where
    E: Evaluator + Clone + Send + Sync,
{
    generate_self_play_with_restarts_and_progress(
        evaluator,
        config,
        generation,
        base_seed,
        restart_archive,
        &|_, _| {},
    )
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
    generate_self_play_with_restarts_and_progress(
        evaluator,
        config,
        generation,
        base_seed,
        &[],
        progress,
    )
}

/// Generates deterministic seed-ordered self-play with targeted restarts and
/// reports each completed trajectory.
///
/// # Errors
///
/// Returns [`SelfPlayError`] when the pool cannot start or any game fails.
pub fn generate_self_play_with_restarts_and_progress<E, F>(
    evaluator: &E,
    config: &SelfPlayConfig,
    generation: u32,
    base_seed: u64,
    restart_archive: &[Replay],
    progress: &F,
) -> Result<Vec<SelfPlayGame>, SelfPlayError>
where
    E: Evaluator + Clone + Send + Sync,
    F: Fn(usize, usize) + Sync,
{
    let starts = planned_restarts(config, base_seed, restart_archive);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.workers)
        .thread_name(|index| format!("yokai-self-play-{index}"))
        .build()?;
    let completed = AtomicUsize::new(0);
    pool.install(|| {
        (0..config.games_per_generation)
            .into_par_iter()
            .map(|index| {
                let game = play_self_play_game_from_restart(
                    (*evaluator).clone(),
                    config,
                    generation,
                    base_seed.wrapping_add(index as u64),
                    starts[index].as_ref(),
                )?;
                let completed = completed.fetch_add(1, Ordering::Relaxed) + 1;
                progress(completed, config.games_per_generation);
                Ok(game)
            })
            .collect()
    })
}

/// Number of trajectories that will use a restart for the supplied archive.
#[must_use]
pub fn planned_restart_count(config: &SelfPlayConfig, archive_len: usize) -> usize {
    if archive_len == 0 {
        return 0;
    }
    fraction_count(config.games_per_generation, config.cycle_restart_fraction)
}

fn planned_restarts(config: &SelfPlayConfig, seed: u64, archive: &[Replay]) -> Vec<Option<Replay>> {
    let mut starts = vec![None; config.games_per_generation];
    let count = planned_restart_count(config, archive.len());
    if count == 0 {
        return starts;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x4359_434c_455f_5253);
    let mut slots = (0..config.games_per_generation).collect::<Vec<_>>();
    slots.shuffle(&mut rng);
    let mut prefix_indices = (0..archive.len()).collect::<Vec<_>>();
    prefix_indices.shuffle(&mut rng);
    for (offset, slot) in slots.into_iter().take(count).enumerate() {
        starts[slot] = Some(archive[prefix_indices[offset % prefix_indices.len()]].clone());
    }
    starts
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn fraction_count(total: usize, fraction: f32) -> usize {
    ((total as f32) * fraction).round() as usize
}

#[derive(Debug, Error)]
pub enum SelfPlayError {
    #[error(transparent)]
    Search(#[from] SearchError),
    #[error(transparent)]
    Move(#[from] MoveError),
    #[error(transparent)]
    Data(#[from] TrainingDataError),
    #[error(transparent)]
    Replay(#[from] ReplayError),
    #[error("a targeted self-play restart must be ongoing")]
    TerminalRestart,
    #[error("self-play exceeded the safety limit of {0} plies")]
    PlyLimit(usize),
    #[error("failed to create self-play workers: {0}")]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),
}
