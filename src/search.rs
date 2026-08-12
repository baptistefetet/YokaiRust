use std::collections::{HashMap, VecDeque};

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    Action, Game, HISTORY_POSITIONS, Outcome, POLICY_ACTIONS, Player, PolicyIndex, Position,
};

/// Input expected by a policy/value evaluator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EvaluationRequest {
    pub position: Position,
    pub repetition_count: u8,
    pub current_player_is_starter: bool,
    /// Resulting occurrence count for each legal policy action; zero elsewhere.
    pub action_repetition_counts: [u8; POLICY_ACTIONS],
    /// Most recent position first; missing pre-game frames are `None`.
    pub history: [Option<Position>; HISTORY_POSITIONS],
}

impl EvaluationRequest {
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
    pub policy: [f32; POLICY_ACTIONS],
    /// Win, draw and loss probabilities from the current player's perspective.
    pub wdl: [f32; 3],
    /// Neutral expected outcome, equal to `P(win) - P(loss)`.
    pub value: f32,
}

impl Evaluation {
    #[must_use]
    pub const fn new(policy: [f32; POLICY_ACTIONS], value: f32) -> Self {
        let bounded = if value < -1.0 {
            -1.0
        } else if value > 1.0 {
            1.0
        } else {
            value
        };
        // A legacy scalar carries no evidence that the position is a draw.
        // Preserve its expectation as a win/loss mixture instead of inventing
        // draw probability that role-aware search would then shape.
        let wdl = [1.0_f32.midpoint(bounded), 0.0, 1.0_f32.midpoint(-bounded)];
        Self {
            policy,
            wdl,
            value: bounded,
        }
    }

    #[must_use]
    pub const fn from_wdl(policy: [f32; POLICY_ACTIONS], wdl: [f32; 3]) -> Self {
        Self {
            policy,
            wdl,
            value: wdl[0] - wdl[2],
        }
    }

    #[must_use]
    pub const fn uniform(value: f32) -> Self {
        Self::new([1.0 / 132.0; POLICY_ACTIONS], value)
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum EvaluationError {
    #[error("evaluator failed: {0}")]
    Backend(String),
    #[error("evaluator returned {actual} results for a batch of {expected}")]
    BatchSizeMismatch { expected: usize, actual: usize },
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
    #[must_use]
    pub fn new(inner: E, max_entries: usize) -> Self {
        Self {
            inner,
            max_entries,
            entries: HashMap::with_capacity(max_entries),
            insertion_order: VecDeque::with_capacity(max_entries),
        }
    }

    #[must_use]
    pub const fn inner(&self) -> &E {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    pub simulations: u32,
    pub evaluation_batch_size: usize,
    pub c_puct: f32,
    pub dirichlet_alpha: f32,
    pub dirichlet_weight: f32,
    /// Search-only reward given to the opponent when a player causes a draw.
    /// Zero preserves the official game-theoretic value.
    pub repetition_contempt: f32,
    /// Self-play-only utility of a draw for the player who started the game.
    /// The non-starter receives the opposite utility. Zero is official play.
    pub starter_draw_value: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 100,
            evaluation_batch_size: 1,
            c_puct: 1.5,
            dirichlet_alpha: 0.3,
            dirichlet_weight: 0.25,
            repetition_contempt: 0.0,
            starter_draw_value: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemperatureSchedule {
    pub exploration_plies: usize,
    pub exploration_temperature: f32,
    pub final_temperature: f32,
}

impl Default for TemperatureSchedule {
    fn default() -> Self {
        Self {
            exploration_plies: 12,
            exploration_temperature: 1.0,
            final_temperature: 0.0,
        }
    }
}

impl TemperatureSchedule {
    #[must_use]
    pub fn for_ply(self, ply: usize) -> f32 {
        if ply < self.exploration_plies {
            self.exploration_temperature
        } else {
            self.final_temperature
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionAnalysis {
    pub action: Action,
    pub prior: f32,
    pub q_value: f32,
    pub visits: u32,
    pub visit_probability: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub best_action: Action,
    pub selected_action: Action,
    pub root_value: f32,
    /// Raw normalized MCTS visit counts used as the neural policy target.
    /// Move-selection temperature must not sharpen this training signal.
    pub policy: [f32; POLICY_ACTIONS],
    pub analysis: Vec<ActionAnalysis>,
}

impl SearchResult {
    #[must_use]
    pub fn analysis_text(&self) -> String {
        self.analysis
            .iter()
            .map(|entry| {
                format!(
                    "{} prior={:.3} visits={} policy={:.3} q={:+.3}",
                    entry.action, entry.prior, entry.visits, entry.visit_probability, entry.q_value
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SearchError {
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
    #[error("cannot search a terminal game")]
    TerminalGame,
    #[error("the root position has no legal action")]
    NoLegalAction,
    #[error("invalid search configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("tree action became illegal while reconstructing a simulation")]
    InvalidTreeAction,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    action: Option<Action>,
    first_child: usize,
    child_count: usize,
    visits: u32,
    value_sum: f32,
    prior: f32,
    expanded: bool,
    terminal: bool,
    in_flight: bool,
}

impl Node {
    const fn root() -> Self {
        Self {
            action: None,
            first_child: 0,
            child_count: 0,
            visits: 0,
            value_sum: 0.0,
            prior: 1.0,
            expanded: false,
            terminal: false,
            in_flight: false,
        }
    }

    const fn child(action: Action, prior: f32) -> Self {
        Self {
            action: Some(action),
            first_child: 0,
            child_count: 0,
            visits: 0,
            value_sum: 0.0,
            prior,
            expanded: false,
            terminal: false,
            in_flight: false,
        }
    }

    fn mean_value(self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / visits_as_f32(self.visits)
        }
    }
}

// A pending leaf temporarily looks favorable from its own perspective, hence
// unfavorable to its parent. This steers the next selection to another leaf
// while the whole group is waiting for one batched neural-network evaluation.
const VIRTUAL_LOSS: f32 = 1.0;

struct PendingSimulation {
    node_index: usize,
    path: Vec<usize>,
    game: Game,
}

enum PreparedSimulation {
    Pending(PendingSimulation),
    Completed,
    Unavailable,
}

/// PUCT Monte-Carlo tree search backed by an arena of contiguous nodes.
pub struct Mcts<E> {
    evaluator: E,
    config: SearchConfig,
    rng: ChaCha8Rng,
    arena: Vec<Node>,
    root: usize,
    root_position: Option<Position>,
    root_history_fingerprint: Option<u64>,
    root_ply: usize,
    root_noise_applied: bool,
}

impl<E: Evaluator> Mcts<E> {
    /// Creates a deterministic search instance.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfiguration`] for invalid parameters.
    pub fn new(evaluator: E, config: SearchConfig, seed: u64) -> Result<Self, SearchError> {
        validate_config(config)?;
        Ok(Self {
            evaluator,
            config,
            rng: ChaCha8Rng::seed_from_u64(seed),
            arena: vec![Node::root()],
            root: 0,
            root_position: None,
            root_history_fingerprint: None,
            root_ply: 0,
            root_noise_applied: false,
        })
    }

    #[must_use]
    pub const fn evaluator(&self) -> &E {
        &self.evaluator
    }

    pub fn evaluator_mut(&mut self) -> &mut E {
        &mut self.evaluator
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.arena.len()
    }

    pub fn reset(&mut self) {
        self.arena.clear();
        self.arena.push(Node::root());
        self.root = 0;
        self.root_position = None;
        self.root_history_fingerprint = None;
        self.root_ply = 0;
        self.root_noise_applied = false;
    }

    /// Moves the root to an explored child and returns whether it was reused.
    #[must_use]
    pub fn advance_root(&mut self, action: Action, game: &Game) -> bool {
        let child = self
            .children(self.root)
            .find(|&index| self.arena[index].action == Some(action));
        let Some(child) = child else {
            self.reset_to_game(game);
            return false;
        };

        self.root = child;
        self.root_position = Some(*game.position());
        self.root_history_fingerprint = Some(game.history_fingerprint());
        self.root_ply = game.actions().len();
        self.root_noise_applied = false;
        true
    }

    /// Runs PUCT simulations and returns the choice plus per-action diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for terminal games, invalid temperature,
    /// evaluator failures, or an inconsistent reused tree.
    pub fn search(&mut self, game: &Game, temperature: f32) -> Result<SearchResult, SearchError> {
        self.search_internal(game, temperature, false)
    }

    /// Runs a self-play search. This is the only entry point that injects
    /// Dirichlet noise into root priors.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::search`].
    pub fn search_self_play(
        &mut self,
        game: &Game,
        temperature: f32,
    ) -> Result<SearchResult, SearchError> {
        self.search_internal(game, temperature, true)
    }

    /// Runs self-play using the configured temperature for the current ply.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::search_self_play`].
    pub fn search_self_play_scheduled(
        &mut self,
        game: &Game,
        schedule: TemperatureSchedule,
    ) -> Result<SearchResult, SearchError> {
        self.search_self_play(game, schedule.for_ply(game.actions().len()))
    }

    fn search_internal(
        &mut self,
        game: &Game,
        temperature: f32,
        add_root_noise: bool,
    ) -> Result<SearchResult, SearchError> {
        if game.outcome().is_terminal() {
            return Err(SearchError::TerminalGame);
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(SearchError::InvalidConfiguration(
                "temperature must be finite and non-negative",
            ));
        }

        self.synchronize_root(game);
        if add_root_noise && self.arena[self.root].expanded && !self.root_noise_applied {
            self.apply_root_noise()?;
        }
        let mut remaining = self.config.simulations;
        while remaining > 0 {
            let requested = usize::try_from(remaining)
                .unwrap_or(usize::MAX)
                .min(self.config.evaluation_batch_size);
            let completed = self.run_simulation_batch(game, requested, add_root_noise)?;
            if completed == 0 {
                return Err(SearchError::InvalidConfiguration(
                    "batched search could not schedule a simulation",
                ));
            }
            remaining = remaining.saturating_sub(u32::try_from(completed).unwrap_or(u32::MAX));
        }
        self.build_result(game.position().side_to_move(), temperature)
    }

    fn run_simulation_batch(
        &mut self,
        root_game: &Game,
        requested: usize,
        add_root_noise: bool,
    ) -> Result<usize, SearchError> {
        let mut completed = 0;
        let mut pending = Vec::with_capacity(requested);
        while completed + pending.len() < requested {
            match self.prepare_simulation(root_game)? {
                PreparedSimulation::Pending(simulation) => pending.push(simulation),
                PreparedSimulation::Completed => completed += 1,
                PreparedSimulation::Unavailable => break,
            }
        }
        if pending.is_empty() {
            return Ok(completed);
        }

        let requests = pending
            .iter()
            .map(|simulation| EvaluationRequest::from_game(&simulation.game))
            .collect::<Vec<_>>();
        let evaluations = self.evaluator.evaluate_batch(&requests);
        if evaluations
            .as_ref()
            .is_ok_and(|evaluations| evaluations.len() != pending.len())
        {
            let actual = evaluations.as_ref().map_or(0, Vec::len);
            self.release_pending(&pending);
            return Err(EvaluationError::BatchSizeMismatch {
                expected: pending.len(),
                actual,
            }
            .into());
        }
        let evaluations = match evaluations {
            Ok(evaluations) => evaluations,
            Err(error) => {
                self.release_pending(&pending);
                return Err(error.into());
            }
        };
        self.release_pending(&pending);

        for (simulation, evaluation) in pending.into_iter().zip(evaluations) {
            let leaf_value = role_aware_value(
                &evaluation,
                simulation.game.position().side_to_move(),
                simulation.game.initial_player(),
                self.config.starter_draw_value,
            );
            self.expand(simulation.node_index, &simulation.game, &evaluation)?;
            if simulation.node_index == self.root && add_root_noise && !self.root_noise_applied {
                self.apply_root_noise()?;
            }
            self.backpropagate(&simulation.path, leaf_value);
            completed += 1;
        }
        Ok(completed)
    }

    fn prepare_simulation(&mut self, root_game: &Game) -> Result<PreparedSimulation, SearchError> {
        let mut game = root_game.clone();
        let mut node_index = self.root;
        let mut path = vec![node_index];

        loop {
            let node = self.arena[node_index];
            if node.in_flight {
                return Ok(PreparedSimulation::Unavailable);
            }
            if !node.expanded || node.terminal || node.child_count == 0 {
                break;
            }
            let Some(child) = self.select_child(node_index) else {
                return Ok(PreparedSimulation::Unavailable);
            };
            let action = self.arena[child]
                .action
                .ok_or(SearchError::InvalidTreeAction)?;
            game.apply(action)
                .map_err(|_| SearchError::InvalidTreeAction)?;
            node_index = child;
            path.push(node_index);
        }

        if game.outcome().is_terminal() {
            self.arena[node_index].terminal = true;
            let value = terminal_value(
                game.outcome(),
                game.position().side_to_move(),
                game.initial_player(),
                self.config.repetition_contempt,
                self.config.starter_draw_value,
            );
            self.backpropagate(&path, value);
            Ok(PreparedSimulation::Completed)
        } else if !self.arena[node_index].expanded {
            self.arena[node_index].in_flight = true;
            self.apply_virtual_loss(&path);
            Ok(PreparedSimulation::Pending(PendingSimulation {
                node_index,
                path,
                game,
            }))
        } else {
            self.backpropagate(&path, 0.0);
            Ok(PreparedSimulation::Completed)
        }
    }

    fn backpropagate(&mut self, path: &[usize], leaf_value: f32) {
        let mut value = leaf_value;
        for &visited in path.iter().rev() {
            let node = &mut self.arena[visited];
            node.visits += 1;
            node.value_sum += value;
            value = -value;
        }
    }

    fn apply_virtual_loss(&mut self, path: &[usize]) {
        for &visited in path {
            let node = &mut self.arena[visited];
            node.visits += 1;
            node.value_sum += VIRTUAL_LOSS;
        }
    }

    fn release_pending(&mut self, pending: &[PendingSimulation]) {
        for simulation in pending {
            self.arena[simulation.node_index].in_flight = false;
            for &visited in &simulation.path {
                let node = &mut self.arena[visited];
                node.visits = node.visits.saturating_sub(1);
                node.value_sum -= VIRTUAL_LOSS;
            }
        }
    }

    fn expand(
        &mut self,
        node_index: usize,
        game: &Game,
        evaluation: &Evaluation,
    ) -> Result<(), SearchError> {
        let legal_actions = game.legal_actions();
        if legal_actions.is_empty() {
            return Err(SearchError::NoLegalAction);
        }

        let player = game.position().side_to_move();
        let mut priors = legal_actions
            .iter()
            .map(|&action| {
                action.policy_index(player).map_or(0.0, |index| {
                    sanitize_prior(evaluation.policy[index.as_usize()])
                })
            })
            .collect::<Vec<_>>();
        normalize_or_uniform(&mut priors);

        let first_child = self.arena.len();
        self.arena.extend(
            legal_actions
                .into_iter()
                .zip(priors)
                .map(|(action, prior)| Node::child(action, prior)),
        );
        let child_count = self.arena.len() - first_child;
        let node = &mut self.arena[node_index];
        node.first_child = first_child;
        node.child_count = child_count;
        node.expanded = true;
        Ok(())
    }

    fn select_child(&self, parent_index: usize) -> Option<usize> {
        let parent_visits = visits_as_f32(self.arena[parent_index].visits.max(1));
        self.children(parent_index)
            .filter(|&child| !self.arena[child].in_flight)
            .max_by(|&left, &right| {
                self.puct_score(left, parent_visits)
                    .total_cmp(&self.puct_score(right, parent_visits))
                    .then_with(|| right.cmp(&left))
            })
    }

    fn puct_score(&self, child_index: usize, parent_visits: f32) -> f32 {
        let child = self.arena[child_index];
        let q_from_parent = -child.mean_value();
        let exploration = self.config.c_puct * child.prior * parent_visits.sqrt()
            / (1.0 + visits_as_f32(child.visits));
        q_from_parent + exploration
    }

    fn apply_root_noise(&mut self) -> Result<(), SearchError> {
        let children = self.children(self.root).collect::<Vec<_>>();
        if children.is_empty() {
            return Err(SearchError::NoLegalAction);
        }
        let gamma = Gamma::new(self.config.dirichlet_alpha, 1.0)
            .map_err(|_| SearchError::InvalidConfiguration("invalid Dirichlet alpha"))?;
        let mut noise = children
            .iter()
            .map(|_| gamma.sample(&mut self.rng))
            .collect::<Vec<f32>>();
        normalize_or_uniform(&mut noise);
        for (child_index, sampled_noise) in children.into_iter().zip(noise) {
            let child = &mut self.arena[child_index];
            child.prior = (1.0 - self.config.dirichlet_weight) * child.prior
                + self.config.dirichlet_weight * sampled_noise;
        }
        self.root_noise_applied = true;
        Ok(())
    }

    fn build_result(
        &mut self,
        player: Player,
        temperature: f32,
    ) -> Result<SearchResult, SearchError> {
        let children = self.children(self.root).collect::<Vec<_>>();
        if children.is_empty() {
            return Err(SearchError::NoLegalAction);
        }

        let best_child = children
            .iter()
            .copied()
            .max_by(|&left, &right| self.compare_final_children(left, right, player))
            .ok_or(SearchError::NoLegalAction)?;
        let policy_probabilities = self.visit_probabilities(&children, 1.0, best_child);
        let selection_probabilities = if (temperature - 1.0).abs() <= f32::EPSILON {
            policy_probabilities.clone()
        } else {
            self.visit_probabilities(&children, temperature, best_child)
        };
        let selected_child =
            children[sample_probability_index(&selection_probabilities, &mut self.rng)];
        let mut policy = [0.0; POLICY_ACTIONS];
        let mut analysis = children
            .iter()
            .copied()
            .zip(policy_probabilities.iter().copied())
            .filter_map(|(child_index, visit_probability)| {
                let child = self.arena[child_index];
                let action = child.action?;
                let policy_index = action.policy_index(player)?;
                policy[policy_index.as_usize()] = visit_probability;
                Some(ActionAnalysis {
                    action,
                    prior: child.prior,
                    q_value: -child.mean_value(),
                    visits: child.visits,
                    visit_probability,
                })
            })
            .collect::<Vec<_>>();
        analysis.sort_by(|left, right| {
            right
                .visits
                .cmp(&left.visits)
                .then_with(|| right.prior.total_cmp(&left.prior))
                .then_with(|| {
                    let left_index = left.action.policy_index(player).map(PolicyIndex::get);
                    let right_index = right.action.policy_index(player).map(PolicyIndex::get);
                    left_index.cmp(&right_index)
                })
        });

        Ok(SearchResult {
            best_action: self.arena[best_child]
                .action
                .ok_or(SearchError::InvalidTreeAction)?,
            selected_action: self.arena[selected_child]
                .action
                .ok_or(SearchError::InvalidTreeAction)?,
            root_value: self.arena[self.root].mean_value(),
            policy,
            analysis,
        })
    }

    fn compare_final_children(
        &self,
        left: usize,
        right: usize,
        player: Player,
    ) -> std::cmp::Ordering {
        let left_node = self.arena[left];
        let right_node = self.arena[right];
        left_node
            .visits
            .cmp(&right_node.visits)
            .then_with(|| left_node.prior.total_cmp(&right_node.prior))
            .then_with(|| {
                let left_policy = left_node
                    .action
                    .and_then(|action| action.policy_index(player))
                    .map(PolicyIndex::get);
                let right_policy = right_node
                    .action
                    .and_then(|action| action.policy_index(player))
                    .map(PolicyIndex::get);
                right_policy.cmp(&left_policy)
            })
    }

    fn visit_probabilities(
        &self,
        children: &[usize],
        temperature: f32,
        best_child: usize,
    ) -> Vec<f32> {
        if temperature <= f32::EPSILON {
            let best = children
                .iter()
                .position(|&child| child == best_child)
                .unwrap_or(0);
            let mut probabilities = vec![0.0; children.len()];
            probabilities[best] = 1.0;
            return probabilities;
        }

        let inverse_temperature = 1.0 / temperature;
        let mut probabilities = children
            .iter()
            .map(|&child| visits_as_f32(self.arena[child].visits).powf(inverse_temperature))
            .collect::<Vec<_>>();
        normalize_or_uniform(&mut probabilities);
        probabilities
    }

    fn synchronize_root(&mut self, game: &Game) {
        let fingerprint = game.history_fingerprint();
        if self.root_position == Some(*game.position())
            && self.root_history_fingerprint == Some(fingerprint)
        {
            return;
        }

        if game.actions().len() == self.root_ply + 1
            && let Some(&last_action) = game.actions().last()
            && self.advance_root(last_action, game)
        {
            return;
        }
        self.reset_to_game(game);
    }

    fn reset_to_game(&mut self, game: &Game) {
        self.arena.clear();
        self.arena.push(Node::root());
        self.root = 0;
        self.root_position = Some(*game.position());
        self.root_history_fingerprint = Some(game.history_fingerprint());
        self.root_ply = game.actions().len();
        self.root_noise_applied = false;
    }

    fn children(&self, node_index: usize) -> impl Iterator<Item = usize> + '_ {
        let node = self.arena[node_index];
        node.first_child..node.first_child + node.child_count
    }
}

fn validate_config(config: SearchConfig) -> Result<(), SearchError> {
    if config.simulations == 0 {
        return Err(SearchError::InvalidConfiguration(
            "simulations must be greater than zero",
        ));
    }
    if config.evaluation_batch_size == 0 {
        return Err(SearchError::InvalidConfiguration(
            "evaluation batch size must be greater than zero",
        ));
    }
    if !config.c_puct.is_finite() || config.c_puct <= 0.0 {
        return Err(SearchError::InvalidConfiguration(
            "c_puct must be finite and positive",
        ));
    }
    if !config.dirichlet_alpha.is_finite() || config.dirichlet_alpha <= 0.0 {
        return Err(SearchError::InvalidConfiguration(
            "Dirichlet alpha must be finite and positive",
        ));
    }
    if !config.dirichlet_weight.is_finite() || !(0.0..=1.0).contains(&config.dirichlet_weight) {
        return Err(SearchError::InvalidConfiguration(
            "Dirichlet weight must be between zero and one",
        ));
    }
    if !config.repetition_contempt.is_finite() || !(0.0..=1.0).contains(&config.repetition_contempt)
    {
        return Err(SearchError::InvalidConfiguration(
            "repetition contempt must be finite and between zero and one",
        ));
    }
    if !config.starter_draw_value.is_finite() || !(0.0..1.0).contains(&config.starter_draw_value) {
        return Err(SearchError::InvalidConfiguration(
            "starter draw value must be finite and in [0, 1)",
        ));
    }
    if config.repetition_contempt > 0.0 && config.starter_draw_value > 0.0 {
        return Err(SearchError::InvalidConfiguration(
            "repetition contempt and starter draw value are mutually exclusive",
        ));
    }
    Ok(())
}

fn sanitize_prior(prior: f32) -> f32 {
    if prior.is_finite() && prior > 0.0 {
        prior
    } else {
        0.0
    }
}

fn sanitize_value(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn normalize_or_uniform(values: &mut [f32]) {
    let sum = values.iter().sum::<f32>();
    if sum.is_finite() && sum > f32::EPSILON {
        for value in values {
            *value /= sum;
        }
    } else if !values.is_empty() {
        let value_count = u16::try_from(values.len())
            .expect("a Yokai policy cannot contain more than 132 legal actions");
        let uniform = 1.0 / f32::from(value_count);
        values.fill(uniform);
    }
}

/// Search visit counts are several orders of magnitude below the 24-bit range
/// represented exactly by `f32`; keeping this conversion in one place makes
/// that performance-oriented representation choice explicit.
#[allow(clippy::cast_precision_loss)]
fn visits_as_f32(visits: u32) -> f32 {
    visits as f32
}

fn sample_probability_index<R: Rng + ?Sized>(probabilities: &[f32], rng: &mut R) -> usize {
    let threshold = rng.random::<f32>();
    let mut cumulative = 0.0;
    for (index, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if threshold <= cumulative {
            return index;
        }
    }
    probabilities.len().saturating_sub(1)
}

fn role_aware_value(
    evaluation: &Evaluation,
    player_to_move: Player,
    initial_player: Player,
    starter_draw_value: f32,
) -> f32 {
    let draw_probability = if evaluation.wdl[1].is_finite() {
        evaluation.wdl[1].clamp(0.0, 1.0)
    } else {
        0.0
    };
    let draw_utility = if player_to_move == initial_player {
        starter_draw_value
    } else {
        -starter_draw_value
    };
    sanitize_value(evaluation.value + draw_utility * draw_probability)
}

fn terminal_value(
    outcome: Outcome,
    player_to_move: Player,
    initial_player: Player,
    repetition_contempt: f32,
    starter_draw_value: f32,
) -> f32 {
    match outcome {
        Outcome::Ongoing => 0.0,
        // `player_to_move` is the opponent of the player whose move completed
        // the repetition, so a positive value makes that action unattractive
        // to the player who caused the draw on the preceding tree edge.
        Outcome::Draw { .. } if starter_draw_value > 0.0 => {
            if player_to_move == initial_player {
                starter_draw_value
            } else {
                -starter_draw_value
            }
        }
        Outcome::Draw { .. } => repetition_contempt,
        Outcome::Win { player, .. } if player == player_to_move => 1.0,
        Outcome::Win { .. } => -1.0,
    }
}
