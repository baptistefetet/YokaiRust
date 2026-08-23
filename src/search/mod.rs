//! Deterministic PUCT Monte-Carlo tree search and evaluator abstractions.
//!
//! Search never touches Burn or a device: it sees leaf evaluation only
//! through the [`Evaluator`] trait, plus the encoder shape constants
//! (`POLICY_ACTIONS`, `HISTORY_POSITIONS`) shared via [`EvaluationRequest`].
//! That separation allows the same tree code to use neural inference, random
//! rollout bootstrapping, a cache, or small deterministic test doubles.
//!
//! The MCTS/AlphaZero vocabulary used here — PUCT, priors, Dirichlet noise,
//! temperature, virtual loss — is defined with context in
//! `docs/alphazero-guide.md` at the repository root.

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{Action, Game, Outcome, POLICY_ACTIONS, Player, PolicyIndex, Position};

pub mod evaluation;

pub use evaluation::{
    AsyncEvaluator, CachedEvaluator, Evaluation, EvaluationError, EvaluationRequest, Evaluator,
    UniformEvaluator,
};

/// PUCT budgets, exploration noise and draw-value conventions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// Complete tree traversals performed before choosing an action.
    pub simulations: u32,
    /// Pending leaves combined into one evaluator call.
    pub evaluation_batch_size: usize,
    /// Strength of the PUCT prior-driven exploration bonus.
    pub c_puct: f32,
    /// Concentration of root Dirichlet noise used only in neural self-play.
    pub dirichlet_alpha: f32,
    /// Fraction of root prior replaced by sampled Dirichlet noise.
    pub dirichlet_weight: f32,
    /// Search-only reward given to the opponent when a player causes a draw.
    /// Zero preserves the official game-theoretic value.
    pub repetition_contempt: f32,
    /// Self-play-only utility of a draw for the player who started the game.
    /// The non-starter receives the opposite utility. Zero is official play.
    pub starter_draw_value: f32,
    /// Source of priors and leaf values for non-terminal expansions.
    pub leaf_evaluation: LeafEvaluation,
}

/// Mechanism used to assign priors and values to newly reached leaves.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum LeafEvaluation {
    /// Query the configured policy/value evaluator.
    #[default]
    Evaluator,
    /// Play uniformly random legal moves to a terminal state or safety limit.
    RandomRollout {
        /// Maximum number of random actions before returning neutral value.
        max_plies: usize,
    },
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
            leaf_evaluation: LeafEvaluation::Evaluator,
        }
    }
}

/// Move-sampling temperatures before and after the exploratory opening.
///
/// The temperature `T` reshapes visit counts into selection probabilities via
/// `visits^(1/T)`: `T = 1` samples proportionally to visit counts, and as
/// `T → 0` the distribution collapses onto the most visited action (argmax).
/// Early plies keep `T = 1` so self-play games diverge and cover many
/// openings; after `exploration_plies` (default 12, roughly the opening phase
/// of this 3x4 game) play turns deterministic so middlegames and endgames are
/// learned from strong moves rather than random ones.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemperatureSchedule {
    /// Number of initial plies using `exploration_temperature`.
    pub exploration_plies: usize,
    /// Sampling temperature used to diversify opening self-play.
    pub exploration_temperature: f32,
    /// Later temperature, normally zero for a visit-count argmax.
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
    /// Returns the temperature applicable to a zero-based ply.
    #[must_use]
    pub fn for_ply(self, ply: usize) -> f32 {
        if ply < self.exploration_plies {
            self.exploration_temperature
        } else {
            self.final_temperature
        }
    }
}

/// Human-readable diagnostics for one legal root action.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionAnalysis {
    /// Domain action represented by this child.
    pub action: Action,
    /// Normalized evaluator prior after optional root noise.
    pub prior: f32,
    /// Mean value from the root player's perspective.
    pub q_value: f32,
    /// Number of simulations that traversed this action.
    pub visits: u32,
    /// Visit fraction used as the untempered neural policy target.
    pub visit_probability: f32,
}

/// Selected move, training target and diagnostics returned by one search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// Deterministic visit-count leader, independent of sampling temperature.
    pub best_action: Action,
    /// Action actually sampled under the requested temperature.
    pub selected_action: Action,
    /// Mean root value from the current player's perspective.
    pub root_value: f32,
    /// Raw normalized MCTS visit counts used as the neural policy target.
    /// Move-selection temperature must not sharpen this training signal.
    pub policy: [f32; POLICY_ACTIONS],
    /// Legal actions sorted for display, normally by descending visits.
    pub analysis: Vec<ActionAnalysis>,
}

impl SearchResult {
    /// Formats one compact diagnostics line per legal root action.
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

/// Invalid search requests, configuration or evaluator behavior.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SearchError {
    /// Leaf evaluation failed.
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
    /// Search was requested after the official game ended.
    #[error("cannot search a terminal game")]
    TerminalGame,
    /// An ongoing position unexpectedly exposes no action.
    #[error("the root position has no legal action")]
    NoLegalAction,
    /// A numeric budget or exploration setting is invalid.
    #[error("invalid search configuration: {0}")]
    InvalidConfiguration(&'static str),
    /// A reused tree edge no longer matches the reconstructed game path.
    #[error("tree action became illegal while reconstructing a simulation")]
    InvalidTreeAction,
    /// An expanded non-terminal node unexpectedly has no children.
    #[error("search tree contains an expanded node without children")]
    InconsistentTree,
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

// Classic MCTS "virtual loss": while a leaf waits for its batched network
// evaluation, every node on its path temporarily receives +1 from its own
// perspective. Because PUCT reads a child through negation
// (`-child.mean_value()`), that bonus makes the busy path look bad to its
// parent, steering the next simulation elsewhere until the batch returns.
// The +1 sign is therefore correct even though the concept is named a loss.
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

    /// Borrows the evaluator owned by the search instance.
    #[must_use]
    pub const fn evaluator(&self) -> &E {
        &self.evaluator
    }

    /// Mutably borrows the evaluator, commonly to inspect adapter state.
    pub fn evaluator_mut(&mut self) -> &mut E {
        &mut self.evaluator
    }

    /// Returns allocated nodes, including unreachable nodes from reused roots.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.arena.len()
    }

    /// Discards the complete tree while retaining evaluator and configuration.
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

    fn search_internal(
        &mut self,
        game: &Game,
        temperature: f32,
        add_root_noise: bool,
    ) -> Result<SearchResult, SearchError> {
        ensure_searchable(game, temperature)?;

        let add_root_noise =
            add_root_noise && matches!(self.config.leaf_evaluation, LeafEvaluation::Evaluator);
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
        let (pending, mut completed) = self.collect_pending(root_game, requested)?;
        if pending.is_empty() {
            return Ok(completed);
        }
        if let LeafEvaluation::RandomRollout { max_plies } = self.config.leaf_evaluation {
            completed += self.finish_rollouts(pending, max_plies)?;
            return Ok(completed);
        }
        let requests = evaluation_requests(&pending);
        let evaluations = self.evaluator.evaluate_batch(&requests);
        completed += self.integrate_evaluations(pending, evaluations, add_root_noise)?;
        Ok(completed)
    }

    /// Descends up to `requested` simulations, finishing terminal ones
    /// immediately and returning the leaves that need an evaluator.
    fn collect_pending(
        &mut self,
        root_game: &Game,
        requested: usize,
    ) -> Result<(Vec<PendingSimulation>, usize), SearchError> {
        let mut completed = 0;
        let mut pending = Vec::with_capacity(requested);
        while completed + pending.len() < requested {
            match self.prepare_simulation(root_game)? {
                PreparedSimulation::Pending(simulation) => pending.push(simulation),
                PreparedSimulation::Completed => completed += 1,
                PreparedSimulation::Unavailable => break,
            }
        }
        Ok((pending, completed))
    }

    /// Completes rollout-mode leaves without touching any evaluator.
    fn finish_rollouts(
        &mut self,
        pending: Vec<PendingSimulation>,
        max_plies: usize,
    ) -> Result<usize, SearchError> {
        self.release_pending(&pending);
        let mut completed = 0;
        for simulation in pending {
            let leaf_value = random_rollout_value(&simulation.game, max_plies, &mut self.rng)?;
            self.expand(
                simulation.node_index,
                &simulation.game,
                &Evaluation::uniform(0.0),
            )?;
            self.backpropagate(&simulation.path, leaf_value);
            completed += 1;
        }
        Ok(completed)
    }

    /// Validates the evaluator reply, releases the virtual losses, then
    /// expands and backpropagates every evaluated leaf. Shared verbatim by
    /// the synchronous and asynchronous batch drivers so they cannot drift.
    fn integrate_evaluations(
        &mut self,
        pending: Vec<PendingSimulation>,
        evaluations: Result<Vec<Evaluation>, EvaluationError>,
        add_root_noise: bool,
    ) -> Result<usize, SearchError> {
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

        let mut completed = 0;
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
            let value = match self.config.leaf_evaluation {
                LeafEvaluation::Evaluator => terminal_value(
                    game.outcome(),
                    game.position().side_to_move(),
                    game.initial_player(),
                    self.config.repetition_contempt,
                    self.config.starter_draw_value,
                ),
                LeafEvaluation::RandomRollout { .. } => {
                    official_value(game.outcome(), game.position().side_to_move())
                }
            };
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
            // `expand` never leaves an expanded, non-terminal node without
            // children, so reaching this state means the tree is corrupted.
            // Failing loudly beats silently counting a zero-value visit.
            Err(SearchError::InconsistentTree)
        }
    }

    fn backpropagate(&mut self, path: &[usize], leaf_value: f32) {
        let mut value = leaf_value;
        for &visited in path.iter().rev() {
            let node = &mut self.arena[visited];
            node.visits += 1;
            node.value_sum += value;
            // Each edge changes the side to move. Negating on every step keeps
            // every node value in that node's own player-to-move perspective.
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

    /// Computes the PUCT score ("Predictor + Upper Confidence bounds applied
    /// to Trees"), AlphaZero's child-selection rule:
    /// `Q + c_puct · prior · √N_parent / (1 + N_child)`.
    ///
    /// The `Q` term exploits what the search has already measured; the second
    /// term explores actions the network believes in (`prior`) but that have
    /// few visits so far. `c_puct` balances the two — the default `1.5` sits
    /// in the range AlphaZero-style engines typically use for small games.
    fn puct_score(&self, child_index: usize, parent_visits: f32) -> f32 {
        let child = self.arena[child_index];
        // Child Q is stored for the opponent, so the parent negates it before
        // adding the exploration bonus based on prior and visit imbalance.
        let q_from_parent = -child.mean_value();
        let exploration = self.config.c_puct * child.prior * parent_visits.sqrt()
            / (1.0 + visits_as_f32(child.visits));
        q_from_parent + exploration
    }

    /// Mixes Dirichlet noise into the root priors for self-play exploration.
    ///
    /// Sampling i.i.d. `Gamma(alpha, 1)` values and normalizing their sum to
    /// one is the standard way to draw a sample from a symmetric
    /// `Dirichlet(alpha)` distribution — which is why no explicit Dirichlet
    /// type appears below. With `alpha = 0.3` most of the noise mass lands on
    /// a few random actions, so each self-play game explores different
    /// openings while `1 - dirichlet_weight` (75%) of the learned prior is
    /// preserved. Official play (arena, TUI, web) never applies this noise.
    fn apply_root_noise(&mut self) -> Result<(), SearchError> {
        let children = self.children(self.root);
        if children.is_empty() {
            return Err(SearchError::NoLegalAction);
        }
        let gamma = Gamma::new(self.config.dirichlet_alpha, 1.0)
            .map_err(|_| SearchError::InvalidConfiguration("invalid Dirichlet alpha"))?;
        let mut noise = children
            .clone()
            .map(|_| gamma.sample(&mut self.rng))
            .collect::<Vec<f32>>();
        normalize_or_uniform(&mut noise);
        for (child_index, sampled_noise) in children.zip(noise) {
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

    /// Returns the contiguous arena indices of a node's children. The range
    /// is an owned value (nothing is borrowed from `self`), so callers may
    /// keep iterating it while mutating nodes.
    fn children(&self, node_index: usize) -> std::ops::Range<usize> {
        let node = self.arena[node_index];
        node.first_child..node.first_child + node.child_count
    }
}

impl<E: Evaluator + AsyncEvaluator> Mcts<E> {
    /// Runs the same deterministic PUCT search as [`Self::search`] while
    /// awaiting evaluator batches. This is intended for WebGPU and other
    /// browser backends whose tensor readback cannot be synchronous.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for terminal games, invalid temperature,
    /// evaluator failures, or an inconsistent reused tree.
    pub async fn search_async(
        &mut self,
        game: &Game,
        temperature: f32,
    ) -> Result<SearchResult, SearchError> {
        ensure_searchable(game, temperature)?;

        // Deliberately no Dirichlet root noise here: the asynchronous entry
        // point serves official (browser) play, never self-play generation.
        self.synchronize_root(game);
        let mut remaining = self.config.simulations;
        while remaining > 0 {
            let requested = usize::try_from(remaining)
                .unwrap_or(usize::MAX)
                .min(self.config.evaluation_batch_size);
            let completed = self.run_simulation_batch_async(game, requested).await?;
            if completed == 0 {
                return Err(SearchError::InvalidConfiguration(
                    "batched search could not schedule a simulation",
                ));
            }
            remaining = remaining.saturating_sub(u32::try_from(completed).unwrap_or(u32::MAX));
        }
        self.build_result(game.position().side_to_move(), temperature)
    }

    async fn run_simulation_batch_async(
        &mut self,
        root_game: &Game,
        requested: usize,
    ) -> Result<usize, SearchError> {
        let (pending, mut completed) = self.collect_pending(root_game, requested)?;
        if pending.is_empty() {
            return Ok(completed);
        }
        if let LeafEvaluation::RandomRollout { max_plies } = self.config.leaf_evaluation {
            completed += self.finish_rollouts(pending, max_plies)?;
            return Ok(completed);
        }
        let requests = evaluation_requests(&pending);
        // The `.await` on the evaluator is the only difference from the
        // synchronous driver; everything else is shared code.
        let evaluations = self.evaluator.evaluate_batch_async(&requests).await;
        completed += self.integrate_evaluations(pending, evaluations, false)?;
        Ok(completed)
    }
}

fn evaluation_requests(pending: &[PendingSimulation]) -> Vec<EvaluationRequest> {
    pending
        .iter()
        .map(|simulation| EvaluationRequest::from_game(&simulation.game))
        .collect()
}

fn ensure_searchable(game: &Game, temperature: f32) -> Result<(), SearchError> {
    if game.outcome().is_terminal() {
        return Err(SearchError::TerminalGame);
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(SearchError::InvalidConfiguration(
            "temperature must be finite and non-negative",
        ));
    }
    Ok(())
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
    if matches!(
        config.leaf_evaluation,
        LeafEvaluation::RandomRollout { max_plies: 0 }
    ) {
        return Err(SearchError::InvalidConfiguration(
            "random rollout limit must be greater than zero",
        ));
    }
    Ok(())
}

/// Plays uniformly random legal actions from a complete leaf game.
///
/// The returned value is always official `+1`, `0`, or `-1` from the leaf
/// player-to-move's perspective. Reaching the safety limit while ongoing is a
/// zero. The cloned [`Game`] preserves the leaf's full repetition history.
///
/// # Errors
///
/// Returns [`SearchError`] if an ongoing game has no legal action or a legal
/// action cannot be applied.
pub fn random_rollout_value<R: Rng + ?Sized>(
    game: &Game,
    max_plies: usize,
    rng: &mut R,
) -> Result<f32, SearchError> {
    let perspective = game.position().side_to_move();
    let mut rollout = game.clone();
    for _ in 0..max_plies {
        if rollout.outcome().is_terminal() {
            return Ok(official_value(rollout.outcome(), perspective));
        }
        let legal_actions = rollout.legal_actions();
        if legal_actions.is_empty() {
            return Err(SearchError::NoLegalAction);
        }
        let action = legal_actions[rng.random_range(0..legal_actions.len())];
        rollout
            .apply(action)
            .map_err(|_| SearchError::InvalidTreeAction)?;
    }
    Ok(official_value(rollout.outcome(), perspective))
}

fn official_value(outcome: Outcome, perspective: Player) -> f32 {
    match outcome {
        Outcome::Win { player, .. } if player == perspective => 1.0,
        Outcome::Win { .. } => -1.0,
        Outcome::Ongoing | Outcome::Draw { .. } => 0.0,
    }
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
        // Slices here are at most `POLICY_ACTIONS` (132) long, so the count
        // always converts losslessly to `f32` through `u16`.
        let value_count =
            u16::try_from(values.len()).expect("normalized slices stay far below the u16 range");
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
