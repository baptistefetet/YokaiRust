//! Paired, noise-free new-network versus reference-network evaluation.

use std::{collections::HashSet, sync::Mutex};

use rand::{RngExt, SeedableRng};
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
    pub reference_wins: usize,
    pub draws: usize,
    pub score: f32,
    pub threshold_reached: bool,
    pub candidate_as_first: ArenaSeatResult,
    pub candidate_as_second: ArenaSeatResult,
    /// Number of different seeded opening histories represented by the games.
    /// Both games in a color-swapped pair deliberately count as one opening.
    #[serde(default)]
    pub distinct_openings: usize,
}

/// Candidate results for one absolute player assignment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArenaSeatResult {
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
}

impl ArenaSeatResult {
    #[must_use]
    pub const fn games(self) -> usize {
        self.wins + self.losses + self.draws
    }

    #[must_use]
    pub fn score(self) -> f32 {
        score(self.wins, self.draws, self.games())
    }
}

/// Consistent snapshot emitted as arena games finish concurrently.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArenaProgress {
    pub completed: usize,
    pub total: usize,
    pub candidate_wins: usize,
    pub reference_wins: usize,
    pub draws: usize,
}

impl ArenaProgress {
    #[must_use]
    pub fn score(self) -> f32 {
        score(self.candidate_wins, self.draws, self.completed)
    }
}

/// Runs paired, reproducible openings with alternating candidate colors, no
/// root noise, and temperature zero.
///
/// # Errors
///
/// Returns [`ArenaError`] if a worker, search, move, or safety limit fails.
pub fn run_arena<C, H>(
    candidate: &C,
    reference: &H,
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
        reference,
        config,
        workers,
        max_game_plies,
        base_seed,
        &|_| {},
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
    reference: &H,
    config: &ArenaConfig,
    workers: usize,
    max_game_plies: usize,
    base_seed: u64,
    progress: &F,
) -> Result<ArenaResult, ArenaError>
where
    C: Evaluator + Clone + Send + Sync,
    H: Evaluator + Clone + Send + Sync,
    F: Fn(ArenaProgress) + Sync,
{
    if workers == 0 || config.games == 0 {
        return Err(ArenaError::InvalidConfiguration);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("yokai-arena-{index}"))
        .build()?;
    let running = Mutex::new(ArenaProgress {
        total: config.games,
        ..ArenaProgress::default()
    });
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
                let (outcome, opening) = play_arena_game(
                    (*candidate).clone(),
                    (*reference).clone(),
                    candidate_player,
                    config,
                    max_game_plies,
                    paired_seed,
                )?;
                let mut running = running
                    .lock()
                    .map_err(|_| ArenaError::ProgressStatePoisoned)?;
                running.completed += 1;
                match outcome {
                    ArenaGameOutcome::CandidateWin => running.candidate_wins += 1,
                    ArenaGameOutcome::ReferenceWin => running.reference_wins += 1,
                    ArenaGameOutcome::Draw => running.draws += 1,
                }
                progress(*running);
                Ok((candidate_player, outcome, opening))
            })
            .collect::<Result<Vec<_>, ArenaError>>()
    })?;

    let mut candidate_wins = 0;
    let mut reference_wins = 0;
    let mut draws = 0;
    let mut candidate_as_first = ArenaSeatResult::default();
    let mut candidate_as_second = ArenaSeatResult::default();
    let mut openings = HashSet::new();
    for (candidate_player, outcome, opening) in outcomes {
        openings.insert(opening);
        let seat = match candidate_player {
            Player::First => &mut candidate_as_first,
            Player::Second => &mut candidate_as_second,
        };
        match outcome {
            ArenaGameOutcome::CandidateWin => {
                candidate_wins += 1;
                seat.wins += 1;
            }
            ArenaGameOutcome::ReferenceWin => {
                reference_wins += 1;
                seat.losses += 1;
            }
            ArenaGameOutcome::Draw => {
                draws += 1;
                seat.draws += 1;
            }
        }
    }
    let score = score(candidate_wins, draws, config.games);
    Ok(ArenaResult {
        candidate_wins,
        reference_wins,
        draws,
        score,
        threshold_reached: score >= config.score_threshold,
        candidate_as_first,
        candidate_as_second,
        distinct_openings: openings.len(),
    })
}

fn play_arena_game<C: Evaluator, H: Evaluator>(
    candidate: C,
    reference: H,
    candidate_player: Player,
    config: &ArenaConfig,
    max_game_plies: usize,
    seed: u64,
) -> Result<(ArenaGameOutcome, u64), ArenaError> {
    let mut game = random_opening_game(seed, config.opening_plies)?;
    let opening = game.history_fingerprint();
    let search_config = SearchConfig {
        simulations: config.simulations,
        evaluation_batch_size: config.search_batch_size,
        ..SearchConfig::default()
    };
    let mut candidate_search = Mcts::new(candidate, search_config, seed.wrapping_mul(2))?;
    let mut reference_search = Mcts::new(
        reference,
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
            reference_search.search(&game, 0.0)?
        };
        game.apply(result.best_action)?;
        // Both players keep a tree: the active search reuses its chosen child,
        // while the opponent can reuse the reply when it was already explored.
        let _candidate_reused = candidate_search.advance_root(result.best_action, &game);
        let _reference_reused = reference_search.advance_root(result.best_action, &game);
    }

    let outcome = match game.outcome() {
        Outcome::Draw { .. } => ArenaGameOutcome::Draw,
        Outcome::Win { player, .. } if player == candidate_player => ArenaGameOutcome::CandidateWin,
        Outcome::Win { .. } => ArenaGameOutcome::ReferenceWin,
        Outcome::Ongoing => unreachable!("arena loop ends only on a terminal game"),
    };
    Ok((outcome, opening))
}

fn random_opening_game(seed: u64, opening_plies: usize) -> Result<Game, MoveError> {
    let mut starting_rng = ChaCha8Rng::seed_from_u64(seed);
    // Consecutive pair seeds alternate the absolute initial player. Candidate
    // colors still swap inside every pair, so this additionally covers both
    // board orientations without relying on a lucky random sample.
    let starting_player = if seed.is_multiple_of(2) {
        Player::First
    } else {
        Player::Second
    };
    let mut game = Game::new(starting_player);
    let opening_length = starting_rng.random_range(0..=opening_plies);
    for _ in 0..opening_length {
        let non_terminal_actions = game
            .legal_actions()
            .iter()
            .copied()
            .filter(|&action| {
                let mut next = game.clone();
                next.apply(action).is_ok() && !next.outcome().is_terminal()
            })
            .collect::<Vec<_>>();
        if non_terminal_actions.is_empty() {
            break;
        }
        let action = non_terminal_actions[starting_rng.random_range(0..non_terminal_actions.len())];
        game.apply(action)?;
    }
    Ok(game)
}

#[derive(Clone, Copy)]
enum ArenaGameOutcome {
    CandidateWin,
    ReferenceWin,
    Draw,
}

fn score(wins: usize, draws: usize, games: usize) -> f32 {
    if games == 0 {
        return 0.0;
    }
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
    #[error("arena progress state was poisoned by a panicking worker")]
    ProgressStatePoisoned,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::Player;

    use super::random_opening_game;

    #[test]
    fn paired_openings_are_reproducible_and_diverse() {
        let left = random_opening_game(123, 4).expect("first opening");
        let right = random_opening_game(123, 4).expect("paired opening");
        assert_eq!(left.position_history(), right.position_history());
        assert_eq!(left.actions(), right.actions());
        assert_eq!(
            random_opening_game(2, 0).unwrap().initial_player(),
            Player::First
        );
        assert_eq!(
            random_opening_game(3, 0).unwrap().initial_player(),
            Player::Second
        );

        let openings = (0..100)
            .map(|seed| {
                random_opening_game(seed, 4)
                    .expect("seeded opening")
                    .history_fingerprint()
            })
            .collect::<HashSet<_>>();
        assert!(openings.len() >= 20, "only {} openings", openings.len());
    }
}
