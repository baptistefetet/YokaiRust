//! Atomic, versioned model generations and latest-network publication.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use burn::{
    prelude::Backend,
    store::{ModuleSnapshot, SafetensorsStore},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ENCODER_VERSION, POLICY_ACTIONS, RULES_VERSION,
    neural::model::{AlphaZeroNetwork, AlphaZeroNetworkConfig},
};

pub const MODEL_FORMAT_VERSION: u16 = 1;
const MODEL_FILE: &str = "model.safetensors";
const METADATA_FILE: &str = "metadata.json";
const LATEST_FILE: &str = "latest";
const LEGACY_CHAMPION_FILE: &str = "champion";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub format_version: u16,
    pub encoder_version: u16,
    pub rules_version: u16,
    pub policy_actions: usize,
    pub generation: u32,
    pub architecture: AlphaZeroNetworkConfig,
}

impl ModelMetadata {
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

#[derive(Debug, Error)]
pub enum ModelStoreError {
    #[error("model I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("invalid model metadata: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Burn model storage failed: {0}")]
    Burn(String),
    #[error("incompatible model: {0}")]
    Incompatible(String),
    #[error("model generation {0} already exists")]
    GenerationExists(u32),
    #[error("model generation {0} does not exist")]
    GenerationMissing(u32),
    #[error("latest-network pointer is missing or invalid")]
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
    fs::rename(&temporary_directory, &final_directory)?;
    Ok(final_directory)
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

/// Atomically updates the latest-network pointer after verifying the generation.
///
/// # Errors
///
/// Returns [`ModelStoreError`] when the generation is absent or the pointer
/// cannot be written.
pub fn publish_latest(root: impl AsRef<Path>, generation: u32) -> Result<(), ModelStoreError> {
    let root = root.as_ref();
    if !generation_directory(root, generation).is_dir() {
        return Err(ModelStoreError::GenerationMissing(generation));
    }
    let temporary = root.join(format!(".{LATEST_FILE}-{}.tmp", std::process::id()));
    fs::write(&temporary, format!("{generation}\n"))?;
    fs::rename(temporary, root.join(LATEST_FILE))?;
    Ok(())
}

/// Loads the generation referenced by the latest-network pointer.
///
/// Repositories created before continuous updates used a `champion` file. It is
/// accepted as a read-only fallback and replaced by `latest` on the next update.
///
/// # Errors
///
/// Returns [`ModelStoreError`] for an invalid pointer or checkpoint.
pub fn load_latest<B: Backend>(
    root: impl AsRef<Path>,
    device: &B::Device,
) -> Result<(AlphaZeroNetwork<B>, ModelMetadata), ModelStoreError> {
    let root = root.as_ref();
    let pointer = match fs::read_to_string(root.join(LATEST_FILE)) {
        Ok(pointer) => pointer,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::read_to_string(root.join(LEGACY_CHAMPION_FILE)).map_err(|legacy_error| {
                if legacy_error.kind() == io::ErrorKind::NotFound {
                    ModelStoreError::InvalidLatest
                } else {
                    ModelStoreError::Io(legacy_error)
                }
            })?
        }
        Err(error) => return Err(ModelStoreError::Io(error)),
    };
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
    let root = root.as_ref();
    if !root.exists() {
        return Ok(0);
    }
    let mut greatest = None;
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(raw_generation) = name.strip_prefix("generation-") else {
            continue;
        };
        if let Ok(generation) = raw_generation.parse::<u32>() {
            greatest = Some(greatest.map_or(generation, |value: u32| value.max(generation)));
        }
    }
    Ok(greatest.map_or(0, |generation| generation.saturating_add(1)))
}

fn generation_directory(root: &Path, generation: u32) -> PathBuf {
    root.join(format!("generation-{generation:06}"))
}
