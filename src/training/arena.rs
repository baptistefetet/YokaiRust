//! Paired, noise-free new-network versus reference-network evaluation.

use std::{collections::HashSet, sync::Mutex};

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Evaluator, Game, Mcts, MoveError, Outcome, Player, SearchConfig, SearchError,
    training::{config::ArenaConfig, data::count_as_f32},
};

/// Aggregate paired-match result used by the promotion gate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArenaResult {
    /// Games won by the candidate from either seat.
    pub candidate_wins: usize,
    /// Games won by the accepted reference champion.
    pub reference_wins: usize,
    /// Official drawn games.
    pub draws: usize,
    /// Candidate points divided by games, with a draw worth one half.
    pub score: f32,
    /// Whether `score` meets the configured promotion threshold.
    pub threshold_reached: bool,
    /// Candidate outcomes while playing as absolute First.
    pub candidate_as_first: ArenaSeatResult,
    /// Candidate outcomes while playing as absolute Second.
    pub candidate_as_second: ArenaSeatResult,
    /// Number of different seeded opening histories represented by the games.
    /// Both games in a color-swapped pair deliberately count as one opening.
    #[serde(default)]
    pub distinct_openings: usize,
    /// Number of color-swapped pairs scoring 0, 0.5, 1, 1.5 or 2 candidate
    /// points. These sufficient statistics preserve within-pair dependence.
    #[serde(default)]
    pub paired_score_counts: [usize; 5],
    /// One-sided exact sign-flip p-value for positive mean paired advantage.
    /// Absent in historical reports that did not retain paired scores.
    #[serde(default)]
    pub improvement_p_value: Option<f64>,
}

impl ArenaResult {
    /// Requires positive evidence at the 5% level in this fixed-size arena.
    /// This is a per-arena test, not a guarantee across repeated experiments.
    #[must_use]
    pub fn statistically_significant(&self) -> bool {
        self.improvement_p_value
            .is_some_and(|p| p.is_finite() && (0.0..=0.05).contains(&p))
    }
}

/// Candidate results for one absolute player assignment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArenaSeatResult {
    /// Candidate wins from this seat.
    pub wins: usize,
    /// Candidate losses from this seat.
    pub losses: usize,
    /// Candidate draws from this seat.
    pub draws: usize,
}

impl ArenaSeatResult {
    /// Returns total games represented by this seat breakdown.
    #[must_use]
    pub const fn games(self) -> usize {
        self.wins + self.losses + self.draws
    }

    /// Returns points per game, with each draw worth one half.
    #[must_use]
    pub fn score(self) -> f32 {
        score(self.wins, self.draws, self.games())
    }
}

/// Consistent snapshot emitted as arena games finish concurrently.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArenaProgress {
    /// Games that have reported an outcome.
    pub completed: usize,
    /// Total games scheduled for this arena.
    pub total: usize,
    /// Candidate wins observed so far.
    pub candidate_wins: usize,
    /// Reference wins observed so far.
    pub reference_wins: usize,
    /// Draws observed so far.
    pub draws: usize,
}

impl ArenaProgress {
    /// Returns the candidate's current score over completed games.
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
    if workers == 0 || config.games == 0 || !config.games.is_multiple_of(2) {
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
                // Adjacent games share their opening seed and swap colors. This
                // removes much of the first-player/opening luck from comparison.
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

    // Indexed Rayon collection preserves seed order, so adjacent results are
    // still the two colors of one opening even if workers finish out of order.
    let mut paired_score_counts = [0; 5];
    for pair in outcomes.chunks_exact(2) {
        let half_points = pair
            .iter()
            .map(|(_, outcome, _)| outcome.half_points())
            .sum::<usize>();
        paired_score_counts[half_points] += 1;
    }
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
        paired_score_counts,
        improvement_p_value: Some(paired_improvement_p_value(paired_score_counts)),
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

impl ArenaGameOutcome {
    fn half_points(self) -> usize {
        match self {
            Self::CandidateWin => 2,
            Self::Draw => 1,
            Self::ReferenceWin => 0,
        }
    }
}

/// Under the exchangeable-pair null, independently flip the sign of each
/// pair's advantage. Advantages are integer half-points in -2..=2, so dynamic
/// programming computes the exact tail without enumerating 2^pairs outcomes.
/// See scipy.stats.permutation_test, permutation_type="samples", one sample.
fn paired_improvement_p_value(counts: [usize; 5]) -> f64 {
    let positive = counts[3] + 2 * counts[4];
    let negative = counts[1] + 2 * counts[0];
    let mut probabilities = vec![1.0_f64];
    let mut radius = 0;
    for (magnitude, count) in [(1, counts[1] + counts[3]), (2, counts[0] + counts[4])] {
        for _ in 0..count {
            let mut next = vec![0.0; probabilities.len() + 2 * magnitude];
            for (index, probability) in probabilities.into_iter().enumerate() {
                next[index] += probability * 0.5;
                next[index + 2 * magnitude] += probability * 0.5;
            }
            probabilities = next;
            radius += magnitude;
        }
    }
    let tail_start = if positive >= negative {
        radius + (positive - negative)
    } else {
        radius - (negative - positive)
    };
    probabilities[tail_start..].iter().sum::<f64>().min(1.0)
}

fn score(wins: usize, draws: usize, games: usize) -> f32 {
    if games == 0 {
        return 0.0;
    }
    (count_as_f32(wins) + 0.5 * count_as_f32(draws)) / count_as_f32(games)
}

/// Failures that invalidate a model-comparison arena.
#[derive(Debug, Error)]
pub enum ArenaError {
    /// Worker count was zero, or the game count was not positive and even.
    #[error("arena requires workers and a positive even number of games")]
    InvalidConfiguration,
    /// MCTS failed in one worker.
    #[error(transparent)]
    Search(#[from] SearchError),
    /// The rules engine rejected a supposedly legal search choice.
    #[error(transparent)]
    Move(#[from] MoveError),
    /// One game exceeded the configured safety bound.
    #[error("arena game exceeded the safety limit of {0} plies")]
    PlyLimit(usize),
    /// Rayon could not create the dedicated worker pool.
    #[error("failed to create arena workers: {0}")]
    ThreadPool(#[from] rayon::ThreadPoolBuildError),
    /// A worker panic poisoned the shared progress mutex.
    #[error("arena progress state was poisoned by a panicking worker")]
    ProgressStatePoisoned,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::Player;

    use super::{paired_improvement_p_value, random_opening_game};

    #[test]
    fn a_fifty_five_percent_score_is_not_automatically_significant() {
        // 30 swept wins, 20 swept losses, 50 tied pairs: 110 / 200 points.
        let p = paired_improvement_p_value([20, 0, 50, 0, 30]);
        assert!((p - 0.101_319_375_532_270_33).abs() < 1e-12);
        assert_eq!(paired_improvement_p_value([0, 0, 100, 0, 0]), 1.0);
        assert!(paired_improvement_p_value([30, 0, 50, 0, 20]) > 0.5);
        assert!(paired_improvement_p_value([0, 0, 0, 0, 6]) < 0.05);
    }

    #[test]
    fn paired_test_matches_exhaustive_sign_flips_with_draws() {
        let advantages = [-2_i32, -1, 0, 1, 2, 2];
        let observed = advantages.iter().sum::<i32>();
        let mut extreme = 0;
        for mask in 0..1 << advantages.len() {
            let sum = advantages
                .iter()
                .enumerate()
                .map(|(i, value)| {
                    if mask & (1 << i) == 0 {
                        *value
                    } else {
                        -*value
                    }
                })
                .sum::<i32>();
            extreme += usize::from(sum >= observed);
        }
        let expected = extreme as f64 / (1 << advantages.len()) as f64;
        assert!((paired_improvement_p_value([1, 1, 1, 1, 2]) - expected).abs() < 1e-12);
    }

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
