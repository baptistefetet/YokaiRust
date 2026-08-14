//! Small residual policy/value network for the 3×4 board.
//!
//! One convolutional trunk extracts shared board features. Two heads then ask
//! different questions: the policy head returns one logit per encoded action,
//! while the WDL (Win / Draw / Loss) head returns three outcome logits. Burn's
//! `Tensor<B, 4>` is a rank-four tensor on backend `B`, analogous to a C++
//! template parameter constrained by the [`Backend`] trait.

use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::{Backend, Config, Module, Tensor},
    tensor::activation::relu,
};

use crate::{GLOBAL_INPUT_FEATURES, INPUT_PLANES, POLICY_ACTIONS, POLICY_CONTEXT_FEATURES};

const BOARD_VALUES: usize = 12;
/// Serializable architecture metadata stored next to every checkpoint.
#[derive(Config, Debug, PartialEq, Eq)]
pub struct AlphaZeroNetworkConfig {
    /// Channel width of the convolutional trunk.
    #[config(default = 64)]
    pub filters: usize,
    /// Number of skip-connected residual blocks.
    #[config(default = 4)]
    pub residual_blocks: usize,
    /// Width shared by the policy and value branches after board flattening.
    #[config(default = 64)]
    pub shared_hidden: usize,
    /// Width of the outcome-specific hidden layer.
    #[config(default = 64)]
    pub value_hidden: usize,
}

impl AlphaZeroNetworkConfig {
    /// Initializes a policy/value network.
    ///
    /// # Panics
    ///
    /// Panics if any configurable layer size is zero.
    #[must_use]
    pub fn init<B: Backend>(&self, device: &B::Device) -> AlphaZeroNetwork<B> {
        assert!(self.filters > 0, "network filters must be positive");
        assert!(
            self.residual_blocks > 0,
            "the residual tower must contain at least one block"
        );
        assert!(
            self.shared_hidden > 0,
            "the shared hidden size must be positive"
        );
        assert!(
            self.value_hidden > 0,
            "the value hidden size must be positive"
        );

        let input_conv = same_conv(INPUT_PLANES, self.filters, 3, device);
        let input_norm = BatchNormConfig::new(self.filters).init(device);
        let residual_tower = (0..self.residual_blocks)
            .map(|_| ResidualBlock::new(self.filters, device))
            .collect();

        AlphaZeroNetwork {
            input_conv,
            input_norm,
            residual_tower,
            shared_hidden: LinearConfig::new(
                self.filters * BOARD_VALUES + GLOBAL_INPUT_FEATURES,
                self.shared_hidden,
            )
            .init(device),
            policy_linear: LinearConfig::new(
                self.shared_hidden + POLICY_CONTEXT_FEATURES,
                POLICY_ACTIONS,
            )
            .init(device),
            value_hidden: LinearConfig::new(self.shared_hidden, self.value_hidden).init(device),
            value_output: LinearConfig::new(self.value_hidden, 3).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct ResidualBlock<B: Backend> {
    conv_1: Conv2d<B>,
    norm_1: BatchNorm<B>,
    conv_2: Conv2d<B>,
    norm_2: BatchNorm<B>,
}

impl<B: Backend> ResidualBlock<B> {
    fn new(filters: usize, device: &B::Device) -> Self {
        Self {
            conv_1: same_conv(filters, filters, 3, device),
            norm_1: BatchNormConfig::new(filters).init(device),
            conv_2: same_conv(filters, filters, 3, device),
            norm_2: BatchNormConfig::new(filters).init(device),
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        // The skip connection lets gradients bypass two convolutions. This is
        // the central ResNet idea and makes deeper stacks easier to optimize.
        let residual = input.clone();
        let hidden = relu(self.norm_1.forward(self.conv_1.forward(input)));
        relu(self.norm_2.forward(self.conv_2.forward(hidden)) + residual)
    }
}

/// Configurable `AlphaZero` network with a shared residual tower.
#[derive(Module, Debug)]
pub struct AlphaZeroNetwork<B: Backend> {
    input_conv: Conv2d<B>,
    input_norm: BatchNorm<B>,
    residual_tower: Vec<ResidualBlock<B>>,
    shared_hidden: Linear<B>,
    policy_linear: Linear<B>,
    value_hidden: Linear<B>,
    value_output: Linear<B>,
}

/// Unnormalized policy and Win/Draw/Loss logits for each batch item.
pub struct NetworkOutput<B: Backend> {
    /// One raw score per fixed policy slot and batch item.
    pub policy_logits: Tensor<B, 2>,
    /// Three raw scores ordered as win, draw and loss.
    pub wdl_logits: Tensor<B, 2>,
}

impl<B: Backend> AlphaZeroNetwork<B> {
    /// Runs the shared residual trunk followed by policy and WDL heads.
    ///
    /// `input` has shape `[batch, planes, rows, columns]`, `global_features`
    /// contains non-spatial hand/history and role state, and `policy_context`
    /// stores two repetition features for each fixed action slot. Returned
    /// values are logits: callers apply softmax before interpreting them as
    /// probabilities.
    #[must_use]
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
        global_features: Tensor<B, 2>,
        policy_context: Tensor<B, 2>,
    ) -> NetworkOutput<B> {
        let mut trunk = relu(self.input_norm.forward(self.input_conv.forward(input)));
        for block in &self.residual_tower {
            trunk = block.forward(trunk);
        }

        let shared = Tensor::cat(vec![trunk.flatten(1, 3), global_features], 1);
        let shared = relu(self.shared_hidden.forward(shared));

        // Only the policy branch consumes action-specific repetition context:
        // WDL describes the position, while the context distinguishes actions
        // that would create an immediate official draw.
        let policy = Tensor::cat(vec![shared.clone(), policy_context], 1);
        let policy_logits = self.policy_linear.forward(policy);

        let value = relu(self.value_hidden.forward(shared));
        let wdl_logits = self.value_output.forward(value);

        NetworkOutput {
            policy_logits,
            wdl_logits,
        }
    }
}

fn same_conv<B: Backend>(
    input_channels: usize,
    output_channels: usize,
    kernel_size: usize,
    device: &B::Device,
) -> Conv2d<B> {
    Conv2dConfig::new(
        [input_channels, output_channels],
        [kernel_size, kernel_size],
    )
    .with_padding(PaddingConfig2d::Same)
    .with_bias(false)
    .init(device)
}
