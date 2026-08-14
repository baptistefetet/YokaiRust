//! Explicit `AlphaZero` loss and optimizer loop with diagnostic metrics.

use burn::{
    module::AutodiffModule,
    optim::{
        Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor,
        decay::WeightDecayConfig,
    },
    prelude::{Backend, Tensor, TensorData},
    tensor::{
        activation::{log_softmax, softmax},
        backend::AutodiffBackend,
    },
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::{
    AlphaZeroNetwork, POLICY_ACTIONS, TrainingExample, encode_position_with_history,
    encoded_batch_tensor, global_batch_tensor, policy_context_batch_tensor,
    training::config::OptimizationConfig,
};

/// Mean optimization losses and health metrics over a set of examples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LossMetrics {
    /// Sum of the weighted policy, WDL and auxiliary scalar objectives.
    pub total_loss: f32,
    /// Cross-entropy between network policy and MCTS visit target.
    pub policy_loss: f32,
    /// Cross-entropy between WDL logits and official result class.
    pub value_loss: f32,
    /// Unweighted MSE between `P(win) - P(loss)` and the scalar result.
    #[serde(default)]
    pub scalar_value_loss: f32,
    /// Entropy of predicted policy probabilities.
    pub policy_entropy: f32,
    /// Mean absolute scalar outcome error.
    pub value_calibration_error: f32,
    /// Predicted probability mass assigned to illegal actions.
    pub illegal_policy_mass: f32,
    /// Fraction whose highest policy logit matches the target's highest mass.
    pub policy_top1_accuracy: f32,
    #[serde(default)]
    /// Fraction whose most likely WDL class matches the official result.
    pub wdl_top1_accuracy: f32,
    #[serde(default)]
    /// Mean absolute error of predicted draw probability.
    pub draw_probability_error: f32,
    /// Mean multiplier applied to per-position policy cross-entropy.
    #[serde(default = "default_policy_weight")]
    pub mean_policy_weight: f32,
}

const fn default_policy_weight() -> f32 {
    1.0
}

/// Periodic metrics snapshot from one optimizer run.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingStepReport {
    /// One-based optimizer update count.
    pub step: usize,
    /// Metrics accumulated since the previous report.
    pub training: LossMetrics,
    /// Full validation metrics, absent when no validation examples exist.
    pub validation: Option<LossMetrics>,
}

/// Complete optimizer summary for one candidate generation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingReport {
    /// Periodic training and validation snapshots.
    pub checkpoints: Vec<TrainingStepReport>,
    /// Number of Adam updates actually applied.
    pub steps_completed: usize,
    /// Effective rate used for every update in this report.
    #[serde(default)]
    pub learning_rate: f64,
}

impl TrainingReport {
    /// Returns the final metrics snapshot selected for generation reporting.
    #[must_use]
    pub fn selected(&self) -> Option<&TrainingStepReport> {
        self.checkpoints.last()
    }
}

/// Adam specialized for the Yokai policy/value network.
pub type AlphaZeroOptimizer<B> = OptimizerAdaptor<Adam, AlphaZeroNetwork<B>, B>;

/// Model and optimizer moments that must advance together across generations.
#[derive(Clone)]
pub struct AlphaZeroTrainingState<B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    /// Trainable network parameters and batch-normalization state.
    pub model: AlphaZeroNetwork<B>,
    /// Adam first/second moments associated with exactly those parameters.
    pub optimizer: AlphaZeroOptimizer<B>,
}

impl<B> AlphaZeroTrainingState<B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    /// Couples a model with a fresh Adam optimizer configured for this run.
    #[must_use]
    pub fn new(model: AlphaZeroNetwork<B>, config: &OptimizationConfig) -> Self {
        Self {
            model,
            optimizer: new_optimizer(config),
        }
    }
}

/// Constructs the `AdamW` optimizer used for all candidate updates.
#[must_use]
pub fn new_optimizer<B>(config: &OptimizationConfig) -> AlphaZeroOptimizer<B>
where
    B: AutodiffBackend<FloatElem = f32>,
{
    AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.weight_decay)))
        .init()
}

/// Optimizes the next network initialized from the latest weights.
///
/// # Panics
///
/// Panics when the training dataset is empty.
#[must_use]
pub fn train_candidate<B>(
    model: AlphaZeroNetwork<B>,
    training_examples: &[TrainingExample],
    validation_examples: &[TrainingExample],
    config: &OptimizationConfig,
    seed: u64,
    device: &B::Device,
) -> (AlphaZeroNetwork<B>, TrainingReport)
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
{
    train_candidate_with_progress(
        model,
        training_examples,
        validation_examples,
        config,
        seed,
        device,
        &|_| {},
    )
}

/// Optimizes a candidate with a fresh Adam state.
///
/// The complete pipeline uses [`train_state_with_progress`] instead so Adam's
/// moments survive generation boundaries. This convenience API is useful for
/// isolated corpus tests.
///
/// # Panics
///
/// Panics when the training dataset is empty.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn train_candidate_with_progress<B, F>(
    model: AlphaZeroNetwork<B>,
    training_examples: &[TrainingExample],
    validation_examples: &[TrainingExample],
    config: &OptimizationConfig,
    seed: u64,
    device: &B::Device,
    progress: &F,
) -> (AlphaZeroNetwork<B>, TrainingReport)
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
    F: Fn(TrainingStepReport),
{
    let state = AlphaZeroTrainingState::new(model, config);
    let (state, report) = train_state_with_progress(
        state,
        training_examples,
        validation_examples,
        config,
        seed,
        device,
        progress,
    );
    (state.model, report)
}

/// Applies a fixed number of uniformly sampled mini-batch updates.
///
/// A fixed budget keeps the optimization pressure constant while the replay
/// buffer grows. The returned optimizer contains the moments required by the
/// next generation.
///
/// # Panics
///
/// Panics when the training dataset is empty.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn train_state_with_progress<B, F>(
    mut state: AlphaZeroTrainingState<B>,
    training_examples: &[TrainingExample],
    validation_examples: &[TrainingExample],
    config: &OptimizationConfig,
    seed: u64,
    device: &B::Device,
    progress: &F,
) -> (AlphaZeroTrainingState<B>, TrainingReport)
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
    F: Fn(TrainingStepReport),
{
    assert!(!training_examples.is_empty(), "training dataset is empty");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let report_capacity = config
        .steps_per_generation
        .div_ceil(config.validation_interval_steps);
    let mut checkpoints = Vec::with_capacity(report_capacity);
    let mut accumulator = MetricAccumulator::default();

    for step in 1..=config.steps_per_generation {
        let batch = (0..config.batch_size)
            .map(|_| {
                let index = rng.random_range(0..training_examples.len());
                training_examples[index].clone()
            })
            .collect::<Vec<_>>();
        let tensors = BatchTensors::<B>::new(&batch, config.non_starter_draw_policy_weight, device);
        let losses = forward_losses(&state.model, tensors, config.scalar_value_loss_weight);
        accumulator.add(read_metrics(&losses), batch.len());
        let gradients = GradientsParams::from_grads(losses.total.backward(), &state.model);
        state.model = state
            .optimizer
            .step(config.learning_rate, state.model, gradients);

        if step.is_multiple_of(config.validation_interval_steps)
            || step == config.steps_per_generation
        {
            let validation = if validation_examples.is_empty() {
                None
            } else {
                Some(validate_model_with_objective_weights(
                    &state.model.valid(),
                    validation_examples,
                    config.batch_size,
                    config.non_starter_draw_policy_weight,
                    config.scalar_value_loss_weight,
                    device,
                ))
            };
            let report = TrainingStepReport {
                step,
                training: std::mem::take(&mut accumulator).finish(),
                validation,
            };
            progress(report);
            checkpoints.push(report);
        }
    }

    (
        state,
        TrainingReport {
            checkpoints,
            steps_completed: config.steps_per_generation,
            learning_rate: config.learning_rate,
        },
    )
}

/// Computes metrics without gradients and without updating batch normalization.
///
/// # Panics
///
/// Panics when `batch_size` is zero.
#[must_use]
pub fn validate_model<B: Backend<FloatElem = f32>>(
    model: &AlphaZeroNetwork<B>,
    examples: &[TrainingExample],
    batch_size: usize,
    device: &B::Device,
) -> LossMetrics {
    validate_model_with_policy_weight(model, examples, batch_size, 1.0, device)
}

/// Computes validation metrics with the same draw-policy weighting as training.
///
/// # Panics
///
/// Panics when `batch_size` is zero.
#[must_use]
pub fn validate_model_with_policy_weight<B: Backend<FloatElem = f32>>(
    model: &AlphaZeroNetwork<B>,
    examples: &[TrainingExample],
    batch_size: usize,
    non_starter_draw_policy_weight: f32,
    device: &B::Device,
) -> LossMetrics {
    validate_model_with_objective_weights(
        model,
        examples,
        batch_size,
        non_starter_draw_policy_weight,
        0.0,
        device,
    )
}

fn validate_model_with_objective_weights<B: Backend<FloatElem = f32>>(
    model: &AlphaZeroNetwork<B>,
    examples: &[TrainingExample],
    batch_size: usize,
    non_starter_draw_policy_weight: f32,
    scalar_value_loss_weight: f32,
    device: &B::Device,
) -> LossMetrics {
    assert!(batch_size > 0, "validation batch size must be positive");
    let mut accumulator = MetricAccumulator::default();
    for batch in examples.chunks(batch_size) {
        let losses = forward_losses(
            model,
            BatchTensors::<B>::new(batch, non_starter_draw_policy_weight, device),
            scalar_value_loss_weight,
        );
        accumulator.add(read_metrics(&losses), batch.len());
    }
    accumulator.finish()
}

struct BatchTensors<B: Backend> {
    input: Tensor<B, 4>,
    global_features: Tensor<B, 2>,
    policy_context: Tensor<B, 2>,
    policy: Tensor<B, 2>,
    policy_weight: Tensor<B, 2>,
    value: Tensor<B, 2>,
    wdl: Tensor<B, 2>,
    illegal: Tensor<B, 2>,
}

impl<B: Backend> BatchTensors<B> {
    fn new(
        examples: &[TrainingExample],
        non_starter_draw_policy_weight: f32,
        device: &B::Device,
    ) -> Self {
        let encoded = examples
            .iter()
            .map(|example| {
                encode_position_with_history(
                    &example.position,
                    example.repetition_count,
                    example.current_player_is_starter,
                    &example.history,
                )
            })
            .collect::<Vec<_>>();
        let policy_contexts = examples
            .iter()
            .map(|example| example.action_repetition_counts)
            .collect::<Vec<_>>();
        let policy_supervision = examples
            .iter()
            .map(|example| policy_supervision(example, non_starter_draw_policy_weight))
            .collect::<Vec<_>>();
        let policy = policy_supervision
            .iter()
            .flat_map(|(target, _)| target.iter().copied())
            .collect::<Vec<_>>();
        let policy_weight = policy_supervision
            .iter()
            .map(|(_, weight)| *weight)
            .collect::<Vec<_>>();
        let value = examples
            .iter()
            .map(|example| example.value)
            .collect::<Vec<_>>();
        let wdl = examples
            .iter()
            .flat_map(|example| match example.value {
                value if value > 0.0 => [1.0, 0.0, 0.0],
                value if value < 0.0 => [0.0, 0.0, 1.0],
                _ => [0.0, 1.0, 0.0],
            })
            .collect::<Vec<_>>();
        let illegal = examples
            .iter()
            .flat_map(illegal_policy_mask)
            .collect::<Vec<_>>();
        Self {
            input: encoded_batch_tensor(&encoded, device),
            global_features: global_batch_tensor(&encoded, device),
            policy_context: policy_context_batch_tensor(&policy_contexts, device),
            policy: Tensor::from_data(
                TensorData::new(policy, [examples.len(), POLICY_ACTIONS]),
                device,
            ),
            policy_weight: Tensor::from_data(
                TensorData::new(policy_weight, [examples.len(), 1]),
                device,
            ),
            value: Tensor::from_data(TensorData::new(value, [examples.len(), 1]), device),
            wdl: Tensor::from_data(TensorData::new(wdl, [examples.len(), 3]), device),
            illegal: Tensor::from_data(
                TensorData::new(illegal, [examples.len(), POLICY_ACTIONS]),
                device,
            ),
        }
    }
}

struct BatchLosses<B: Backend> {
    total: Tensor<B, 1>,
    diagnostics: Tensor<B, 1>,
}

fn forward_losses<B: Backend<FloatElem = f32>>(
    model: &AlphaZeroNetwork<B>,
    batch: BatchTensors<B>,
    scalar_value_loss_weight: f32,
) -> BatchLosses<B> {
    let output = model.forward(batch.input, batch.global_features, batch.policy_context);

    // Cross-entropy for a target distribution p and predicted logits z is
    // -sum(p * log_softmax(z)). Keep one value per position first because
    // drawn non-starter targets may be omitted from policy imitation.
    let log_probabilities = log_softmax(output.policy_logits.clone(), 1);
    let probabilities = softmax(output.policy_logits.clone(), 1);
    let policy_loss_per_position = (log_probabilities.clone() * batch.policy.clone())
        .sum_dim(1)
        .neg();
    // The clamp also makes an artificial all-zero-weight test batch safe: its
    // policy loss is zero while WDL still trains normally.
    let policy_weight_sum = batch.policy_weight.clone().sum().clamp_min(1.0);
    let policy_loss =
        (policy_loss_per_position * batch.policy_weight.clone()).sum() / policy_weight_sum;
    // WDL means Win / Draw / Loss. Unlike policy loss, every official outcome
    // remains fully weighted, including draws.
    let wdl_log_probabilities = log_softmax(output.wdl_logits.clone(), 1);
    let value_loss = (wdl_log_probabilities * batch.wdl.clone())
        .sum_dim(1)
        .mean()
        .neg();
    let wdl_probabilities = softmax(output.wdl_logits.clone(), 1);
    let predicted_value =
        wdl_probabilities.clone().narrow(1, 0, 1) - wdl_probabilities.clone().narrow(1, 2, 1);
    let value_error = predicted_value - batch.value;
    let scalar_value_loss = (value_error.clone() * value_error.clone()).mean();
    let total = policy_loss.clone()
        + value_loss.clone()
        + scalar_value_loss.clone() * scalar_value_loss_weight;
    let entropy = (probabilities.clone() * log_probabilities)
        .sum_dim(1)
        .mean()
        .neg();
    let wdl_top1 = output
        .wdl_logits
        .clone()
        .argmax(1)
        .equal(batch.wdl.clone().argmax(1))
        .float()
        .mean();
    let calibration = value_error.abs().mean();
    let draw_probability_error = (wdl_probabilities.narrow(1, 1, 1) - batch.wdl.narrow(1, 1, 1))
        .abs()
        .mean();
    let mean_policy_weight = batch.policy_weight.mean();
    let illegal_mass = (probabilities * batch.illegal).sum_dim(1).mean();
    let top1 = output
        .policy_logits
        .argmax(1)
        .equal(batch.policy.argmax(1))
        .float()
        .mean();
    let diagnostics = Tensor::cat(
        vec![
            total.clone().detach(),
            policy_loss.detach(),
            value_loss.detach(),
            scalar_value_loss.detach(),
            entropy.detach(),
            calibration.detach(),
            illegal_mass.detach(),
            top1.detach(),
            wdl_top1.detach(),
            draw_probability_error.detach(),
            mean_policy_weight.detach(),
        ],
        0,
    );
    BatchLosses { total, diagnostics }
}

fn read_metrics<B: Backend<FloatElem = f32>>(losses: &BatchLosses<B>) -> LossMetrics {
    let values = losses
        .diagnostics
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("diagnostic tensor uses f32");
    LossMetrics {
        total_loss: values[0],
        policy_loss: values[1],
        value_loss: values[2],
        scalar_value_loss: values[3],
        policy_entropy: values[4],
        value_calibration_error: values[5],
        illegal_policy_mass: values[6],
        policy_top1_accuracy: values[7],
        wdl_top1_accuracy: values[8],
        draw_probability_error: values[9],
        mean_policy_weight: values[10],
    }
}

fn policy_supervision(
    example: &TrainingExample,
    non_starter_draw_weight: f32,
) -> ([f32; POLICY_ACTIONS], f32) {
    // A draw proves that the non-starter's complete policy trajectory failed to
    // convert, so the ordinary target is omitted. If an action would cause the
    // third repetition, however, the rules identify that exact failure. Remove
    // it and retain only MCTS's relative preference among the alternatives.
    if example.value != 0.0 || example.current_player_is_starter {
        return (example.policy, 1.0);
    }
    let mut target = example.policy;
    let mut found_immediate_draw = false;
    for (probability, repetition_count) in target.iter_mut().zip(example.action_repetition_counts) {
        if repetition_count >= 3 {
            *probability = 0.0;
            found_immediate_draw = true;
        }
    }
    let remaining_mass = target.iter().sum::<f32>();
    if found_immediate_draw && remaining_mass > f32::EPSILON {
        for probability in &mut target {
            *probability /= remaining_mass;
        }
        return (target, 1.0);
    }
    (example.policy, non_starter_draw_weight)
}

fn illegal_policy_mask(example: &TrainingExample) -> [f32; POLICY_ACTIONS] {
    let mut mask = [1.0; POLICY_ACTIONS];
    for action in example.position.legal_actions() {
        if let Some(index) = action.policy_index(example.position.side_to_move()) {
            mask[index.as_usize()] = 0.0;
        }
    }
    mask
}

#[derive(Default)]
struct MetricAccumulator {
    weighted: LossMetrics,
    examples: usize,
}

impl MetricAccumulator {
    fn add(&mut self, metrics: LossMetrics, examples: usize) {
        let weight = sample_count_as_f32(examples);
        self.weighted.total_loss += metrics.total_loss * weight;
        self.weighted.policy_loss += metrics.policy_loss * weight;
        self.weighted.value_loss += metrics.value_loss * weight;
        self.weighted.scalar_value_loss += metrics.scalar_value_loss * weight;
        self.weighted.policy_entropy += metrics.policy_entropy * weight;
        self.weighted.value_calibration_error += metrics.value_calibration_error * weight;
        self.weighted.illegal_policy_mass += metrics.illegal_policy_mass * weight;
        self.weighted.policy_top1_accuracy += metrics.policy_top1_accuracy * weight;
        self.weighted.wdl_top1_accuracy += metrics.wdl_top1_accuracy * weight;
        self.weighted.draw_probability_error += metrics.draw_probability_error * weight;
        self.weighted.mean_policy_weight += metrics.mean_policy_weight * weight;
        self.examples += examples;
    }

    fn finish(self) -> LossMetrics {
        if self.examples == 0 {
            return LossMetrics::default();
        }
        let divisor = sample_count_as_f32(self.examples);
        LossMetrics {
            total_loss: self.weighted.total_loss / divisor,
            policy_loss: self.weighted.policy_loss / divisor,
            value_loss: self.weighted.value_loss / divisor,
            scalar_value_loss: self.weighted.scalar_value_loss / divisor,
            policy_entropy: self.weighted.policy_entropy / divisor,
            value_calibration_error: self.weighted.value_calibration_error / divisor,
            illegal_policy_mass: self.weighted.illegal_policy_mass / divisor,
            policy_top1_accuracy: self.weighted.policy_top1_accuracy / divisor,
            wdl_top1_accuracy: self.weighted.wdl_top1_accuracy / divisor,
            draw_probability_error: self.weighted.draw_probability_error / divisor,
            mean_policy_weight: self.weighted.mean_policy_weight / divisor,
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn sample_count_as_f32(count: usize) -> f32 {
    count as f32
}

#[cfg(test)]
mod tests {
    use super::policy_supervision;
    use crate::{HISTORY_POSITIONS, POLICY_ACTIONS, Player, Position, TrainingExample};

    fn example(value: f32, starter: bool, repetition_mass: f32) -> TrainingExample {
        let mut policy = [0.0; POLICY_ACTIONS];
        policy[0] = repetition_mass;
        policy[1] = 1.0 - repetition_mass;
        let mut action_repetition_counts = [0; POLICY_ACTIONS];
        action_repetition_counts[0] = 2;
        TrainingExample {
            position: Position::initial(Player::First),
            repetition_count: 1,
            current_player_is_starter: starter,
            action_repetition_counts,
            history: [None; HISTORY_POSITIONS],
            policy,
            value,
        }
    }

    #[test]
    fn failed_draw_policy_is_omitted_until_the_rules_identify_the_exact_action() {
        let weight = 0.0;
        let ordinary_draw = example(0.0, false, 0.75);
        let (ordinary_target, ordinary_weight) = policy_supervision(&ordinary_draw, weight);

        assert!(
            ordinary_target
                .iter()
                .zip(ordinary_draw.policy)
                .all(|(actual, expected)| (*actual - expected).abs() < f32::EPSILON)
        );
        assert!(ordinary_weight.abs() < f32::EPSILON);

        let (starter_target, starter_weight) =
            policy_supervision(&example(0.0, true, 0.75), weight);
        assert!((starter_target[0] - 0.75).abs() < f32::EPSILON);
        assert!((starter_weight - 1.0).abs() < f32::EPSILON);

        let mut immediate_draw = ordinary_draw;
        immediate_draw.action_repetition_counts[0] = 3;
        let (corrected_target, corrected_weight) = policy_supervision(&immediate_draw, weight);
        assert!(corrected_target[0].abs() < f32::EPSILON);
        assert!((corrected_target[1] - 1.0).abs() < f32::EPSILON);
        assert!((corrected_weight - 1.0).abs() < f32::EPSILON);
    }
}
