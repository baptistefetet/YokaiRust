//! Versioned, validated JSON representation of complete games.
//!
//! Deserialization alone is not trusted: [`Replay::to_game`] reapplies every
//! action through the rules engine and checks the claimed outcome. This keeps
//! old or manually edited files from silently creating impossible game states.

use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ActionAnalysis,
    game::{Action, Game, MoveError, Outcome, Player, RULES_VERSION},
};

/// Current schema version of serialized [`Replay`] files.
pub const REPLAY_FORMAT_VERSION: u16 = 1;

/// Portable record of one game and optional search information for each ply.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    /// Replay JSON schema version.
    pub format_version: u16,
    /// Rules implementation version used to validate the actions.
    pub rules_version: u16,
    /// Optional random seed that makes generated games reproducible.
    pub seed: Option<u64>,
    /// Absolute player who took the first turn.
    pub initial_player: Player,
    /// Applied actions in chronological order.
    pub actions: Vec<Action>,
    /// Claimed terminal status, checked by replaying all actions.
    pub outcome: Outcome,
    /// Optional MCTS alternatives aligned one list per played action.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analyses: Option<Vec<Vec<ActionAnalysis>>>,
}

impl Replay {
    /// Builds a replay from already collected components.
    #[must_use]
    pub fn from_actions(
        initial_player: Player,
        actions: Vec<Action>,
        outcome: Outcome,
        seed: Option<u64>,
    ) -> Self {
        Self {
            format_version: REPLAY_FORMAT_VERSION,
            rules_version: RULES_VERSION,
            seed,
            initial_player,
            actions,
            outcome,
            analyses: None,
        }
    }

    /// Snapshots a validated in-memory game into its portable representation.
    #[must_use]
    pub fn from_game(game: &Game, seed: Option<u64>) -> Self {
        Self::from_actions(
            game.initial_player(),
            game.actions().to_vec(),
            game.outcome(),
            seed,
        )
    }

    /// Attaches one complete legal-action analysis list per played move.
    /// Validation is deferred to the regular replay validation methods.
    #[must_use]
    pub fn with_analyses(mut self, analyses: Vec<Vec<ActionAnalysis>>) -> Self {
        self.analyses = Some(analyses);
        self
    }

    /// Replays and validates every stored action.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError`] for unsupported versions, illegal actions, or an
    /// outcome that does not match the reconstructed game.
    pub fn to_game(&self) -> Result<Game, ReplayError> {
        self.validate_versions()?;
        if let Some(analyses) = &self.analyses
            && analyses.len() != self.actions.len()
        {
            return Err(ReplayError::AnalysisCountMismatch {
                expected: self.actions.len(),
                actual: analyses.len(),
            });
        }
        let mut game = Game::new(self.initial_player);
        for (ply, &action) in self.actions.iter().enumerate() {
            if let Some(analyses) = &self.analyses {
                for entry in &analyses[ply] {
                    if !game.is_legal_action(entry.action) {
                        return Err(ReplayError::InvalidAnalysisAction {
                            ply,
                            action: entry.action,
                        });
                    }
                }
            }
            game.apply(action)
                .map_err(|source| ReplayError::InvalidAction { ply, source })?;
        }
        if game.outcome() != self.outcome {
            return Err(ReplayError::OutcomeMismatch {
                expected: self.outcome,
                actual: game.outcome(),
            });
        }
        Ok(game)
    }

    /// Serializes this replay as human-readable JSON.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError::Json`] if serialization fails.
    pub fn to_json_pretty(&self) -> Result<String, ReplayError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Parses JSON and fully validates the reconstructed game.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError`] for malformed JSON or invalid replay contents.
    pub fn from_json(input: &str) -> Result<Self, ReplayError> {
        let replay: Self = serde_json::from_str(input)?;
        replay.to_game()?;
        Ok(replay)
    }

    /// Validates and writes this replay to a JSON file.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError`] when validation, serialization, or I/O fails.
    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), ReplayError> {
        self.to_game()?;
        fs::write(path, self.to_json_pretty()?)?;
        Ok(())
    }

    /// Reads, parses, and validates a JSON replay file.
    ///
    /// # Errors
    ///
    /// Returns [`ReplayError`] when reading, parsing, or validation fails.
    pub fn read_json(path: impl AsRef<Path>) -> Result<Self, ReplayError> {
        let input = fs::read_to_string(path)?;
        Self::from_json(&input)
    }

    fn validate_versions(&self) -> Result<(), ReplayError> {
        if self.format_version != REPLAY_FORMAT_VERSION {
            return Err(ReplayError::UnsupportedFormatVersion(self.format_version));
        }
        if self.rules_version != RULES_VERSION {
            return Err(ReplayError::UnsupportedRulesVersion(self.rules_version));
        }
        Ok(())
    }
}

/// Failures encountered while parsing, validating or storing a replay.
#[derive(Debug, Error)]
pub enum ReplayError {
    /// File uses a replay schema this build does not understand.
    #[error("unsupported replay format version {0}")]
    UnsupportedFormatVersion(u16),
    /// File targets a different rules implementation.
    #[error("unsupported rules version {0}")]
    UnsupportedRulesVersion(u16),
    /// One stored action is illegal at its reconstructed ply.
    #[error("invalid action at ply {ply}: {source}")]
    InvalidAction {
        /// Zero-based action index.
        ply: usize,
        /// Rules-engine rejection.
        source: MoveError,
    },
    /// Optional analysis lists are not aligned one-to-one with actions.
    #[error("expected {expected} analysis lists, got {actual}")]
    AnalysisCountMismatch {
        /// Number of played actions.
        expected: usize,
        /// Number of stored analysis lists.
        actual: usize,
    },
    /// An analysis mentions an action illegal at that ply.
    #[error("analysis at ply {ply} contains illegal action {action}")]
    InvalidAnalysisAction {
        /// Zero-based action index.
        ply: usize,
        /// Illegal diagnostic alternative.
        action: Action,
    },
    /// Claimed outcome differs from rules-engine reconstruction.
    #[error("replay outcome mismatch: expected {expected:?}, got {actual:?}")]
    OutcomeMismatch {
        /// Outcome stored in JSON.
        expected: Outcome,
        /// Outcome produced by replaying actions.
        actual: Outcome,
    },
    /// JSON syntax or schema is invalid.
    #[error("invalid replay JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Replay file could not be read or written.
    #[error("replay I/O error: {0}")]
    Io(#[from] io::Error),
}
