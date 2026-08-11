use std::{fs, io, path::Path};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    ActionAnalysis,
    game::{Action, Game, MoveError, Outcome, Player, RULES_VERSION},
};

pub const REPLAY_FORMAT_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Replay {
    pub format_version: u16,
    pub rules_version: u16,
    pub seed: Option<u64>,
    pub initial_player: Player,
    pub actions: Vec<Action>,
    pub outcome: Outcome,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub analyses: Option<Vec<Vec<ActionAnalysis>>>,
}

impl Replay {
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

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("unsupported replay format version {0}")]
    UnsupportedFormatVersion(u16),
    #[error("unsupported rules version {0}")]
    UnsupportedRulesVersion(u16),
    #[error("invalid action at ply {ply}: {source}")]
    InvalidAction { ply: usize, source: MoveError },
    #[error("expected {expected} analysis lists, got {actual}")]
    AnalysisCountMismatch { expected: usize, actual: usize },
    #[error("analysis at ply {ply} contains illegal action {action}")]
    InvalidAnalysisAction { ply: usize, action: Action },
    #[error("replay outcome mismatch: expected {expected:?}, got {actual:?}")]
    OutcomeMismatch { expected: Outcome, actual: Outcome },
    #[error("invalid replay JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("replay I/O error: {0}")]
    Io(#[from] io::Error),
}
