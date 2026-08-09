//! Explicit `AlphaZero` loss and optimizer loop with diagnostic metrics.

use burn::{
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer, decay::WeightDecayConfig},
    prelude::{Backend, Tensor, TensorData},
    tensor::{
        activation::{log_softmax, softmax},
        backend::AutodiffBackend,
    },
};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::{
    AlphaZeroNetwork, POLICY_ACTIONS, TrainingExample, encode_position, encoded_batch_tensor,
    training::config::OptimizationConfig,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LossMetrics {
    pub total_loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub policy_entropy: f32,
    pub value_calibration_error: f32,
    pub illegal_policy_mass: f32,
    pub policy_top1_accuracy: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpochReport {
    pub epoch: usize,
    pub training: LossMetrics,
    pub validation: Option<LossMetrics>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrainingReport {
    pub epochs: Vec<EpochReport>,
    pub selected_epoch: usize,
}

impl TrainingReport {
    #[must_use]
    pub fn selected(&self) -> Option<&EpochReport> {
        self.epochs
            .iter()
            .find(|report| report.epoch == self.selected_epoch)
    }
}

/// Optimizes a candidate initialized from the champion weights.
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

/// Optimizes a candidate and reports metrics after every complete epoch.
///
/// # Panics
///
/// Panics when the training dataset is empty.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn train_candidate_with_progress<B, F>(
    mut model: AlphaZeroNetwork<B>,
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
    F: Fn(EpochReport),
{
    assert!(!training_examples.is_empty(), "training dataset is empty");
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(config.weight_decay)))
        .init();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut examples = training_examples.to_vec();
    let mut epochs = Vec::with_capacity(config.epochs);
    let mut best_model = None;
    let mut best_validation_loss = f32::INFINITY;
    let mut selected_epoch = 0;
    let mut epochs_without_improvement = 0;

    for epoch in 0..config.epochs {
        examples.shuffle(&mut rng);
        let mut accumulator = MetricAccumulator::default();
        for batch in examples.chunks(config.batch_size) {
            let tensors = BatchTensors::<B>::new(batch, device);
            let losses = forward_losses(&model, tensors);
            accumulator.add(read_metrics(&losses), batch.len());
            let gradients = GradientsParams::from_grads(losses.total.backward(), &model);
            model = optimizer.step(config.learning_rate, model, gradients);
        }

        let validation = if validation_examples.is_empty() {
            None
        } else {
            Some(validate_model(
                &model.valid(),
                validation_examples,
                config.batch_size,
                device,
            ))
        };
        let report = EpochReport {
            epoch: epoch + 1,
            training: accumulator.finish(),
            validation,
        };
        progress(report);
        epochs.push(report);

        if let Some(validation) = validation {
            if validation.total_loss < best_validation_loss {
                best_validation_loss = validation.total_loss;
                best_model = Some(model.clone());
                selected_epoch = epoch + 1;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
                if epochs_without_improvement >= config.early_stopping_patience {
                    break;
                }
            }
        }
    }

    if let Some(best_model) = best_model {
        model = best_model;
    } else {
        selected_epoch = epochs.len();
    }
    (
        model,
        TrainingReport {
            epochs,
            selected_epoch,
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
    assert!(batch_size > 0, "validation batch size must be positive");
    let mut accumulator = MetricAccumulator::default();
    for batch in examples.chunks(batch_size) {
        let losses = forward_losses(model, BatchTensors::<B>::new(batch, device));
        accumulator.add(read_metrics(&losses), batch.len());
    }
    accumulator.finish()
}

struct BatchTensors<B: Backend> {
    input: Tensor<B, 4>,
    policy: Tensor<B, 2>,
    value: Tensor<B, 2>,
    illegal: Tensor<B, 2>,
}

impl<B: Backend> BatchTensors<B> {
    fn new(examples: &[TrainingExample], device: &B::Device) -> Self {
        let encoded = examples
            .iter()
            .map(|example| encode_position(&example.position, example.repetition_count))
            .collect::<Vec<_>>();
        let policy = examples
            .iter()
            .flat_map(|example| example.policy.iter().copied())
            .collect::<Vec<_>>();
        let value = examples
            .iter()
            .map(|example| example.value)
            .collect::<Vec<_>>();
        let illegal = examples
            .iter()
            .flat_map(illegal_policy_mask)
            .collect::<Vec<_>>();
        Self {
            input: encoded_batch_tensor(&encoded, device),
            policy: Tensor::from_data(
                TensorData::new(policy, [examples.len(), POLICY_ACTIONS]),
                device,
            ),
            value: Tensor::from_data(TensorData::new(value, [examples.len(), 1]), device),
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
) -> BatchLosses<B> {
    let output = model.forward(batch.input);
    let log_probabilities = log_softmax(output.policy_logits.clone(), 1);
    let probabilities = softmax(output.policy_logits.clone(), 1);
    let policy_loss = (log_probabilities.clone() * batch.policy.clone())
        .sum_dim(1)
        .mean()
        .neg();
    let value_loss = (output.value.clone() - batch.value.clone())
        .powf_scalar(2.0)
        .mean();
    let total = policy_loss.clone() + value_loss.clone();
    let entropy = (probabilities.clone() * log_probabilities)
        .sum_dim(1)
        .mean()
        .neg();
    let calibration = (output.value - batch.value).abs().mean();
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
            entropy.detach(),
            calibration.detach(),
            illegal_mass.detach(),
            top1.detach(),
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
        policy_entropy: values[3],
        value_calibration_error: values[4],
        illegal_policy_mass: values[5],
        policy_top1_accuracy: values[6],
    }
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
        self.weighted.policy_entropy += metrics.policy_entropy * weight;
        self.weighted.value_calibration_error += metrics.value_calibration_error * weight;
        self.weighted.illegal_policy_mass += metrics.illegal_policy_mass * weight;
        self.weighted.policy_top1_accuracy += metrics.policy_top1_accuracy * weight;
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
            policy_entropy: self.weighted.policy_entropy / divisor,
            value_calibration_error: self.weighted.value_calibration_error / divisor,
            illegal_policy_mass: self.weighted.illegal_policy_mass / divisor,
            policy_top1_accuracy: self.weighted.policy_top1_accuracy / divisor,
        }
    }
}

#[allow(clippy::cast_precision_loss)]
fn sample_count_as_f32(count: usize) -> f32 {
    count as f32
}
