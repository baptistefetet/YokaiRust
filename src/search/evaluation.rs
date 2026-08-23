//! Evaluator abstractions consumed by the MCTS driver in [`super`].
//!
//! Leaf evaluation sits behind the [`Evaluator`] and [`AsyncEvaluator`]
//! traits so the same tree code can use neural inference, random rollout
//! bootstrapping, a bounded cache, or small deterministic test doubles.

use std::collections::{HashMap, VecDeque};

use thiserror::Error;

use crate::{Game, HISTORY_POSITIONS, POLICY_ACTIONS, Position};

/// Input expected by a policy/value evaluator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EvaluationRequest {
    /// Path-independent board state to encode.
    pub position: Position,
    /// Number of occurrences of the current full position on this path.
    pub repetition_count: u8,
    /// Whether the side to move also made the trajectory's first move.
    pub current_player_is_starter: bool,
    /// Resulting occurrence count for each legal policy action; zero elsewhere.
    pub action_repetition_counts: [u8; POLICY_ACTIONS],
    /// Most recent position first; missing pre-game frames are `None`.
    pub history: [Option<Position>; HISTORY_POSITIONS],
}

impl EvaluationRequest {
    /// Collects all spatial, historical and per-action context from a game.
    #[must_use]
    pub fn from_game(game: &Game) -> Self {
        let positions = game.position_history();
        let player = game.position().side_to_move();
        let mut action_repetition_counts = [0; POLICY_ACTIONS];
        for action in game.legal_actions() {
            if let (Some(index), Some(count)) = (
                action.policy_index(player),
                game.repetition_count_after(action),
            ) {
                action_repetition_counts[index.as_usize()] = count;
            }
        }
        Self {
            position: *game.position(),
            repetition_count: game.current_repetition_count(),
            current_player_is_starter: player == game.initial_player(),
            action_repetition_counts,
            history: std::array::from_fn(|offset| {
                positions
                    .len()
                    .checked_sub(offset + 2)
                    .map(|index| positions[index])
            }),
        }
    }
}

/// Policy probabilities and a value from the current player's perspective.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// One probability per fixed action slot; illegal slots may be ignored later.
    pub policy: [f32; POLICY_ACTIONS],
    /// Win, draw and loss probabilities from the current player's perspective.
    pub wdl: [f32; 3],
    /// Neutral expected outcome, equal to `P(win) - P(loss)`.
    pub value: f32,
}

impl Evaluation {
    /// Builds an evaluation from a scalar value in `[-1, 1]`.
    ///
    /// A scalar carries no evidence that the position is a draw, so the
    /// expectation is preserved as a pure win/loss mixture with zero draw
    /// probability.
    #[must_use]
    pub const fn from_scalar(policy: [f32; POLICY_ACTIONS], value: f32) -> Self {
        let bounded = if value < -1.0 {
            -1.0
        } else if value > 1.0 {
            1.0
        } else {
            value
        };
        // A scalar carries no evidence that the position is a draw.
        // Preserve its expectation as a win/loss mixture instead of inventing
        // draw probability that role-aware search would then shape.
        let wdl = [1.0_f32.midpoint(bounded), 0.0, 1.0_f32.midpoint(-bounded)];
        Self {
            policy,
            wdl,
            value: bounded,
        }
    }

    /// Builds an evaluation and derives its neutral scalar expectation.
    #[must_use]
    pub const fn from_wdl(policy: [f32; POLICY_ACTIONS], wdl: [f32; 3]) -> Self {
        Self {
            policy,
            wdl,
            value: wdl[0] - wdl[2],
        }
    }

    /// Builds a uniform-policy evaluation useful for tests and rollouts.
    #[must_use]
    pub const fn uniform(value: f32) -> Self {
        Self::from_scalar([1.0 / (POLICY_ACTIONS as f32); POLICY_ACTIONS], value)
    }
}

/// Failures produced by an inference backend or adapter.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum EvaluationError {
    /// Backend-specific execution or tensor conversion failure.
    #[error("evaluator failed: {0}")]
    Backend(String),
    /// Evaluator violated the one-result-per-request contract.
    #[error("evaluator returned {actual} results for a batch of {expected}")]
    BatchSizeMismatch {
        /// Number of submitted requests.
        expected: usize,
        /// Number of returned evaluations.
        actual: usize,
    },
}

/// Batch-oriented interface shared by CPU, Metal, and test evaluators.
pub trait Evaluator {
    /// Evaluates every request in order.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError`] when the backend cannot produce the batch.
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError>;
}

/// Asynchronous evaluator used by browser backends whose device readback is a
/// JavaScript promise. The search algorithm remains identical to native PUCT;
/// only the inference boundary yields to the browser event loop.
#[allow(async_fn_in_trait)]
pub trait AsyncEvaluator {
    /// Evaluates every request in order without blocking the browser thread.
    ///
    /// # Errors
    ///
    /// Returns [`EvaluationError`] when the backend cannot produce the batch.
    async fn evaluate_batch_async(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError>;
}

/// Baseline evaluator returning a uniform policy and neutral value.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Ok(vec![Evaluation::uniform(0.0); requests.len()])
    }
}

/// FIFO prediction cache with deterministic eviction.
#[derive(Clone, Debug)]
pub struct CachedEvaluator<E> {
    inner: E,
    max_entries: usize,
    entries: HashMap<EvaluationRequest, Evaluation>,
    insertion_order: VecDeque<EvaluationRequest>,
}

impl<E> CachedEvaluator<E> {
    /// Wraps an evaluator with a bounded FIFO cache; zero disables storage.
    #[must_use]
    pub fn new(inner: E, max_entries: usize) -> Self {
        Self {
            inner,
            max_entries,
            entries: HashMap::with_capacity(max_entries),
            insertion_order: VecDeque::with_capacity(max_entries),
        }
    }

    /// Borrows the wrapped evaluator.
    #[must_use]
    pub const fn inner(&self) -> &E {
        &self.inner
    }

    /// Mutably borrows the wrapped evaluator, for example to inspect counters.
    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    /// Returns the number of cached unique requests.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Reports whether the cache currently contains no requests.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes all predictions while preserving allocated capacity.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.insertion_order.clear();
    }

    fn insert(&mut self, request: &EvaluationRequest, evaluation: Evaluation) {
        if self.max_entries == 0 || self.entries.contains_key(request) {
            return;
        }
        while self.entries.len() >= self.max_entries {
            let Some(oldest) = self.insertion_order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.insertion_order.push_back(*request);
        self.entries.insert(*request, evaluation);
    }
}

impl<E: Evaluator> Evaluator for CachedEvaluator<E> {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        let mut results = vec![None; requests.len()];
        let mut missing_requests = Vec::new();
        let mut missing_indices = Vec::<Vec<usize>>::new();
        let mut pending = HashMap::<EvaluationRequest, usize>::new();

        for (index, request) in requests.iter().copied().enumerate() {
            if let Some(cached) = self.entries.get(&request) {
                results[index] = Some(cached.clone());
            } else if let Some(&pending_index) = pending.get(&request) {
                missing_indices[pending_index].push(index);
            } else {
                pending.insert(request, missing_requests.len());
                missing_requests.push(request);
                missing_indices.push(vec![index]);
            }
        }

        if !missing_requests.is_empty() {
            let evaluated = self.inner.evaluate_batch(&missing_requests)?;
            if evaluated.len() != missing_requests.len() {
                return Err(EvaluationError::BatchSizeMismatch {
                    expected: missing_requests.len(),
                    actual: evaluated.len(),
                });
            }
            for ((request, indices), evaluation) in missing_requests
                .into_iter()
                .zip(missing_indices)
                .zip(evaluated)
            {
                self.insert(&request, evaluation.clone());
                for index in indices {
                    results[index] = Some(evaluation.clone());
                }
            }
        }

        results
            .into_iter()
            .map(|result| {
                result.ok_or_else(|| {
                    EvaluationError::Backend("cache did not fill an evaluation slot".to_owned())
                })
            })
            .collect()
    }
}
