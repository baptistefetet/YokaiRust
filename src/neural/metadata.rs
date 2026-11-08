//! Model compatibility shared by native checkpoints and browser exports.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::model::AlphaZeroNetworkConfig;
use crate::{ENCODER_VERSION, MODEL_FORMAT_VERSION, POLICY_ACTIONS, RULES_VERSION};

/// Compatibility information stored alongside every set of weights.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Checkpoint storage schema version.
    pub format_version: u16,
    /// Neural feature-encoder version expected by the weights.
    pub encoder_version: u16,
    /// Game rules version that generated the training data.
    pub rules_version: u16,
    /// Fixed width of the policy output layer.
    pub policy_actions: usize,
    /// Monotonic candidate generation identifier.
    pub generation: u32,
    /// Layer dimensions needed to reconstruct the Burn module before loading.
    pub architecture: AlphaZeroNetworkConfig,
}

impl ModelMetadata {
    /// Creates metadata populated with every current compatibility version.
    #[must_use]
    pub fn new(generation: u32, architecture: AlphaZeroNetworkConfig) -> Self {
        Self {
            format_version: MODEL_FORMAT_VERSION,
            encoder_version: ENCODER_VERSION,
            rules_version: RULES_VERSION,
            policy_actions: POLICY_ACTIONS,
            generation,
            architecture,
        }
    }

    /// Checks that these weights use the current rules, encoder and network layout.
    ///
    /// # Errors
    ///
    /// Returns [`ModelMetadataError`] for incompatible versions or empty layers.
    pub fn validate(&self) -> Result<(), ModelMetadataError> {
        if self.format_version != MODEL_FORMAT_VERSION {
            return Err(ModelMetadataError(format!(
                "model format {} is not supported (expected {})",
                self.format_version, MODEL_FORMAT_VERSION
            )));
        }
        if self.encoder_version != ENCODER_VERSION {
            return Err(ModelMetadataError(format!(
                "encoder version {} does not match {}",
                self.encoder_version, ENCODER_VERSION
            )));
        }
        if self.rules_version != RULES_VERSION {
            return Err(ModelMetadataError(format!(
                "rules version {} does not match {}",
                self.rules_version, RULES_VERSION
            )));
        }
        if self.policy_actions != POLICY_ACTIONS {
            return Err(ModelMetadataError(format!(
                "policy width {} does not match {}",
                self.policy_actions, POLICY_ACTIONS
            )));
        }
        if self.architecture.filters == 0
            || self.architecture.residual_blocks == 0
            || self.architecture.shared_hidden == 0
            || self.architecture.value_hidden == 0
        {
            return Err(ModelMetadataError(
                "network dimensions must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Reason a model's metadata cannot be loaded by this build.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{0}")]
pub struct ModelMetadataError(String);
