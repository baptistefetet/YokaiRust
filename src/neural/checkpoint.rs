//! Atomic, versioned model generations and accepted-champion publication.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use burn::{
    module::{AutodiffModule, Module},
    optim::Optimizer,
    prelude::Backend,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    store::{ModuleSnapshot, SafetensorsStore},
    tensor::backend::AutodiffBackend,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ENCODER_VERSION, POLICY_ACTIONS, RULES_VERSION,
    neural::model::{AlphaZeroNetwork, AlphaZeroNetworkConfig},
    training::{
        config::OptimizationConfig,
        trainer::{AlphaZeroTrainingState, new_optimizer},
    },
};

/// Schema version of checkpoint directories and their metadata.
pub const MODEL_FORMAT_VERSION: u16 = 5;
const MODEL_FILE: &str = "model.safetensors";
const METADATA_FILE: &str = "metadata.json";
const LATEST_FILE: &str = "latest";
const TRAINING_MODEL_FILE: &str = "training-model";
const OPTIMIZER_FILE: &str = "optimizer";

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

    fn validate(&self) -> Result<(), ModelStoreError> {
        if self.format_version != MODEL_FORMAT_VERSION {
            return Err(ModelStoreError::Incompatible(format!(
                "model format {} is not supported (expected {})",
                self.format_version, MODEL_FORMAT_VERSION
            )));
        }
        if self.encoder_version != ENCODER_VERSION {
            return Err(ModelStoreError::Incompatible(format!(
                "encoder version {} does not match {}",
                self.encoder_version, ENCODER_VERSION
            )));
        }
        if self.rules_version != RULES_VERSION {
            return Err(ModelStoreError::Incompatible(format!(
                "rules version {} does not match {}",
                self.rules_version, RULES_VERSION
            )));
        }
        if self.policy_actions != POLICY_ACTIONS {
            return Err(ModelStoreError::Incompatible(format!(
                "policy width {} does not match {}",
                self.policy_actions, POLICY_ACTIONS
            )));
        }
        Ok(())
    }
}

/// Checkpoint compatibility, persistence and publication failures.
#[derive(Debug, Error)]
pub enum ModelStoreError {
    /// A checkpoint filesystem operation failed.
    #[error("model I/O error: {0}")]
    Io(#[from] io::Error),
    /// Metadata is not valid JSON for the current schema.
    #[error("invalid model metadata: {0}")]
    Json(#[from] serde_json::Error),
    /// Burn failed to serialize or restore tensors or optimizer state.
    #[error("Burn model storage failed: {0}")]
    Burn(String),
    /// Stored versions, architecture or tensor keys do not match this build.
    #[error("incompatible model: {0}")]
    Incompatible(String),
    /// Atomic save refused to overwrite an existing generation.
    #[error("model generation {0} already exists")]
    GenerationExists(u32),
    /// A requested generation directory does not exist.
    #[error("model generation {0} does not exist")]
    GenerationMissing(u32),
    /// The `latest` champion pointer is absent or does not contain a generation.
    #[error("champion pointer is missing or invalid")]
    InvalidLatest,
}

/// Writes a complete generation and atomically makes its final directory
/// visible. Existing generations are never overwritten.
///
/// # Errors
///
/// Returns [`ModelStoreError`] on I/O, serialization, storage, or collision.
pub fn save_generation<B: Backend>(
    root: impl AsRef<Path>,
    metadata: &ModelMetadata,
    model: &AlphaZeroNetwork<B>,
) -> Result<PathBuf, ModelStoreError> {
    save_generation_with_extra(root, metadata, model, |_| Ok(()))
}

/// Atomically stores inference weights together with resumable training state.
///
/// # Errors
///
/// Returns [`ModelStoreError`] on model, optimizer, serialization, or I/O
/// failures. The generation directory is made visible only after all files are
/// complete.
pub fn save_training_generation<B>(
    root: impl AsRef<Path>,
    metadata: &ModelMetadata,
    state: &AlphaZeroTrainingState<B>,
) -> Result<PathBuf, ModelStoreError>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
{
    let inference_model = state.model.clone().valid();
    save_generation_with_extra(root, metadata, &inference_model, |directory| {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .record(
                state.model.clone().into_record(),
                directory.join(TRAINING_MODEL_FILE),
            )
            .map_err(|error| ModelStoreError::Burn(error.to_string()))?;
        recorder
            .record(state.optimizer.to_record(), directory.join(OPTIMIZER_FILE))
            .map_err(|error| ModelStoreError::Burn(error.to_string()))?;
        Ok(())
    })
}

fn save_generation_with_extra<B, F>(
    root: impl AsRef<Path>,
    metadata: &ModelMetadata,
    model: &AlphaZeroNetwork<B>,
    save_extra: F,
) -> Result<PathBuf, ModelStoreError>
where
    B: Backend,
    F: FnOnce(&Path) -> Result<(), ModelStoreError>,
{
    metadata.validate()?;
    let root = root.as_ref();
    fs::create_dir_all(root)?;
    let final_directory = generation_directory(root, metadata.generation);
    if final_directory.exists() {
        return Err(ModelStoreError::GenerationExists(metadata.generation));
    }
    let temporary_directory = root.join(format!(
        ".generation-{:06}-{}.tmp",
        metadata.generation,
        std::process::id()
    ));
    if temporary_directory.exists() {
        return Err(ModelStoreError::GenerationExists(metadata.generation));
    }
    fs::create_dir(&temporary_directory)?;

    let mut store = SafetensorsStore::from_file(temporary_directory.join(MODEL_FILE))
        .overwrite(false)
        .metadata("yokai_model_format", MODEL_FORMAT_VERSION.to_string())
        .metadata("yokai_generation", metadata.generation.to_string());
    model
        .save_into(&mut store)
        .map_err(|error| ModelStoreError::Burn(error.to_string()))?;
    fs::write(
        temporary_directory.join(METADATA_FILE),
        serde_json::to_vec_pretty(metadata)?,
    )?;
    save_extra(&temporary_directory)?;
    fs::rename(&temporary_directory, &final_directory)?;
    Ok(final_directory)
}

/// Restores the trainable model and Adam moments for one generation.
///
/// Old inference-only checkpoints return `Ok(None)` so they remain usable; the
/// caller can then start a fresh optimizer explicitly.
///
/// # Errors
///
/// Returns [`ModelStoreError`] when metadata or either training-state file is
/// malformed, incompatible, or only partially present.
pub fn load_training_generation<B>(
    root: impl AsRef<Path>,
    generation: u32,
    optimization: &OptimizationConfig,
    device: &B::Device,
) -> Result<Option<(AlphaZeroTrainingState<B>, ModelMetadata)>, ModelStoreError>
where
    B: AutodiffBackend<FloatElem = f32>,
    B::InnerBackend: Backend<FloatElem = f32>,
{
    let directory = generation_directory(root.as_ref(), generation);
    if !directory.is_dir() {
        return Err(ModelStoreError::GenerationMissing(generation));
    }
    let metadata: ModelMetadata =
        serde_json::from_slice(&fs::read(directory.join(METADATA_FILE))?)?;
    metadata.validate()?;
    if metadata.generation != generation {
        return Err(ModelStoreError::Incompatible(format!(
            "directory generation {generation} contains generation {}",
            metadata.generation
        )));
    }

    let model_path = directory.join(format!("{TRAINING_MODEL_FILE}.bin"));
    let optimizer_path = directory.join(format!("{OPTIMIZER_FILE}.bin"));
    match (model_path.exists(), optimizer_path.exists()) {
        (false, false) => return Ok(None),
        (true, true) => {}
        _ => {
            return Err(ModelStoreError::Incompatible(
                "checkpoint contains only part of its training state".to_owned(),
            ));
        }
    }

    let recorder = BinFileRecorder::<FullPrecisionSettings>::default();
    let model_record = recorder
        .load(directory.join(TRAINING_MODEL_FILE), device)
        .map_err(|error| ModelStoreError::Burn(error.to_string()))?;
    let model = metadata
        .architecture
        .init::<B>(device)
        .load_record(model_record);
    let optimizer_record = recorder
        .load(directory.join(OPTIMIZER_FILE), device)
        .map_err(|error| ModelStoreError::Burn(error.to_string()))?;
    let optimizer = new_optimizer(optimization).load_record(optimizer_record);
    Ok(Some((
        AlphaZeroTrainingState { model, optimizer },
        metadata,
    )))
}

/// Loads and validates one generation on the requested backend.
///
/// # Errors
///
/// Returns [`ModelStoreError`] for absent, malformed, incompatible, or invalid
/// checkpoint data.
pub fn load_generation<B: Backend>(
    root: impl AsRef<Path>,
    generation: u32,
    device: &B::Device,
) -> Result<(AlphaZeroNetwork<B>, ModelMetadata), ModelStoreError> {
    let directory = generation_directory(root.as_ref(), generation);
    if !directory.is_dir() {
        return Err(ModelStoreError::GenerationMissing(generation));
    }
    let metadata: ModelMetadata =
        serde_json::from_slice(&fs::read(directory.join(METADATA_FILE))?)?;
    metadata.validate()?;
    if metadata.generation != generation {
        return Err(ModelStoreError::Incompatible(format!(
            "directory generation {generation} contains generation {}",
            metadata.generation
        )));
    }

    let mut model = metadata.architecture.init::<B>(device);
    let mut store = SafetensorsStore::from_file(directory.join(MODEL_FILE));
    let result = model
        .load_from(&mut store)
        .map_err(|error| ModelStoreError::Burn(error.to_string()))?;
    if !result.is_success() || !result.missing.is_empty() || !result.unused.is_empty() {
        return Err(ModelStoreError::Incompatible(result.to_string()));
    }
    Ok((model, metadata))
}

/// Atomically updates the champion pointer after verifying the generation.
///
/// # Errors
///
/// Returns [`ModelStoreError`] when the generation is absent or the pointer
/// cannot be written.
pub fn publish_champion(root: impl AsRef<Path>, generation: u32) -> Result<(), ModelStoreError> {
    publish_pointer(root.as_ref(), LATEST_FILE, generation)
}

/// Loads the generation referenced by the accepted-champion pointer.
///
/// # Errors
///
/// Returns [`ModelStoreError`] for an invalid pointer or checkpoint.
pub fn load_champion<B: Backend>(
    root: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(AlphaZeroNetwork<B>, ModelMetadata), ModelStoreError> {
    let root = root.as_ref();
    let pointer = fs::read_to_string(root.join(LATEST_FILE)).map_err(|error| {
        if error.kind() == io::ErrorKind::NotFound {
            ModelStoreError::InvalidLatest
        } else {
            ModelStoreError::Io(error)
        }
    })?;
    let generation = pointer
        .trim()
        .parse::<u32>()
        .map_err(|_| ModelStoreError::InvalidLatest)?;
    load_generation(root, generation, device)
}

/// Returns one more than the greatest generation directory currently present.
///
/// # Errors
///
/// Returns [`ModelStoreError::Io`] when the model directory cannot be read.
pub fn next_generation(root: impl AsRef<Path>) -> Result<u32, ModelStoreError> {
    Ok(stored_generations(root)?
        .last()
        .map_or(0, |generation| generation.saturating_add(1)))
}

/// Lists every complete saved model generation in ascending order.
///
/// # Errors
///
/// Returns [`ModelStoreError::Io`] when the model directory cannot be read.
pub fn stored_generations(root: impl AsRef<Path>) -> Result<Vec<u32>, ModelStoreError> {
    let root = root.as_ref();
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut generations = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(raw_generation) = name.strip_prefix("generation-") else {
            continue;
        };
        if let Ok(generation) = raw_generation.parse::<u32>() {
            generations.push(generation);
        }
    }
    generations.sort_unstable();
    generations.dedup();
    Ok(generations)
}

fn publish_pointer(root: &Path, filename: &str, generation: u32) -> Result<(), ModelStoreError> {
    if !generation_directory(root, generation).is_dir() {
        return Err(ModelStoreError::GenerationMissing(generation));
    }
    let temporary = root.join(format!(".{filename}-{}.tmp", std::process::id()));
    fs::write(&temporary, format!("{generation}\n"))?;
    fs::rename(temporary, root.join(filename))?;
    Ok(())
}

fn generation_directory(root: &Path, generation: u32) -> PathBuf {
    root.join(format!("generation-{generation:06}"))
}
