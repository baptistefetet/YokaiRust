//! Batched bridge between the Burn network and tree search.

use burn::{
    backend::{Autodiff, Flex, Metal},
    prelude::Backend,
    tensor::activation::softmax,
};

use crate::{
    Evaluation, EvaluationError, EvaluationRequest, Evaluator, POLICY_ACTIONS,
    neural::{encode_position_with_history, encoded_batch_tensor, model::AlphaZeroNetwork},
};

pub type CpuBackend = Flex<f32, i32>;
pub type MetalBackend = Metal<f32, i32>;
pub type CpuTrainingBackend = Autodiff<CpuBackend>;
pub type MetalTrainingBackend = Autodiff<MetalBackend>;

/// Synchronous batch evaluator. Policy and value are concatenated before the
/// device readback so each batch requires a single GPU synchronization.
pub struct NetworkEvaluator<B: Backend> {
    model: AlphaZeroNetwork<B>,
    device: B::Device,
}

impl<B: Backend> NetworkEvaluator<B> {
    #[must_use]
    pub const fn new(model: AlphaZeroNetwork<B>, device: B::Device) -> Self {
        Self { model, device }
    }

    #[must_use]
    pub const fn model(&self) -> &AlphaZeroNetwork<B> {
        &self.model
    }

    #[must_use]
    pub const fn device(&self) -> &B::Device {
        &self.device
    }

    #[must_use]
    pub fn into_model(self) -> AlphaZeroNetwork<B> {
        self.model
    }
}

impl<B: Backend> Evaluator for NetworkEvaluator<B> {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let encoded = requests
            .iter()
            .map(|request| {
                encode_position_with_history(
                    &request.position,
                    request.repetition_count,
                    &request.history,
                )
            })
            .collect::<Vec<_>>();
        let output = self
            .model
            .forward(encoded_batch_tensor(&encoded, &self.device));
        let output_width = POLICY_ACTIONS + 1;
        let predictions =
            burn::tensor::Tensor::cat(vec![softmax(output.policy_logits, 1), output.value], 1)
                .into_data()
                .to_vec::<f32>()
                .map_err(|error| EvaluationError::Backend(error.to_string()))?;

        if predictions.len() != requests.len() * output_width {
            return Err(EvaluationError::BatchSizeMismatch {
                expected: requests.len(),
                actual: predictions.len() / output_width,
            });
        }

        predictions
            .chunks_exact(output_width)
            .map(|prediction| {
                let policy = prediction[..POLICY_ACTIONS].try_into().map_err(|_| {
                    EvaluationError::Backend("invalid policy tensor width".to_owned())
                })?;
                Ok(Evaluation::new(policy, prediction[POLICY_ACTIONS]))
            })
            .collect()
    }
}
