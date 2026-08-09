//! Small residual policy/value network for the 3×4 board.

use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::{Backend, Config, Module, Tensor},
    tensor::activation::relu,
};

use crate::{INPUT_PLANES, POLICY_ACTIONS};

const BOARD_VALUES: usize = 12;
const POLICY_CHANNELS: usize = 2;
const VALUE_CHANNELS: usize = 1;

/// Serializable architecture metadata stored next to every checkpoint.
#[derive(Config, Debug, PartialEq, Eq)]
pub struct AlphaZeroNetworkConfig {
    #[config(default = 64)]
    pub filters: usize,
    #[config(default = 4)]
    pub residual_blocks: usize,
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
            policy_conv: pointwise_conv(self.filters, POLICY_CHANNELS, device),
            policy_norm: BatchNormConfig::new(POLICY_CHANNELS).init(device),
            policy_linear: LinearConfig::new(POLICY_CHANNELS * BOARD_VALUES, POLICY_ACTIONS)
                .init(device),
            value_conv: pointwise_conv(self.filters, VALUE_CHANNELS, device),
            value_norm: BatchNormConfig::new(VALUE_CHANNELS).init(device),
            value_hidden: LinearConfig::new(VALUE_CHANNELS * BOARD_VALUES, self.value_hidden)
                .init(device),
            value_output: LinearConfig::new(self.value_hidden, 1).init(device),
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
    policy_conv: Conv2d<B>,
    policy_norm: BatchNorm<B>,
    policy_linear: Linear<B>,
    value_conv: Conv2d<B>,
    value_norm: BatchNorm<B>,
    value_hidden: Linear<B>,
    value_output: Linear<B>,
}

/// Unnormalized policy logits and a `tanh` value for each batch item.
pub struct NetworkOutput<B: Backend> {
    pub policy_logits: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
}

impl<B: Backend> AlphaZeroNetwork<B> {
    #[must_use]
    pub fn forward(&self, input: Tensor<B, 4>) -> NetworkOutput<B> {
        let mut trunk = relu(self.input_norm.forward(self.input_conv.forward(input)));
        for block in &self.residual_tower {
            trunk = block.forward(trunk);
        }

        let policy = relu(
            self.policy_norm
                .forward(self.policy_conv.forward(trunk.clone())),
        );
        let policy_logits = self.policy_linear.forward(policy.flatten(1, 3));

        let value = relu(self.value_norm.forward(self.value_conv.forward(trunk)));
        let value = relu(self.value_hidden.forward(value.flatten(1, 3)));
        let value = self.value_output.forward(value).tanh();

        NetworkOutput {
            policy_logits,
            value,
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

fn pointwise_conv<B: Backend>(
    input_channels: usize,
    output_channels: usize,
    device: &B::Device,
) -> Conv2d<B> {
    same_conv(input_channels, output_channels, 1, device)
}
