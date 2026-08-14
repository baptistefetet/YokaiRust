//! Read-only diagnostics over the stable validation split.

use std::{
    fs, io,
    path::{Path, PathBuf},
};

use burn::prelude::Backend;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    AlphaZeroNetwork, DatasetSplit, LossMetrics, Outcome, TrainingExample,
    validate_model_with_policy_weight,
};

pub const ENDGAME_DISTANCE_REPORT_VERSION: u16 = 2;

/// Remaining plies between a recorded position and its official result.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EndgameDistance {
    #[serde(rename = "1")]
    One,
    #[serde(rename = "2-4")]
    TwoToFour,
    #[serde(rename = "5-8")]
    FiveToEight,
    #[serde(rename = "9-16")]
    NineToSixteen,
    #[serde(rename = "17+")]
    SeventeenPlus,
}

impl EndgameDistance {
    pub const ALL: [Self; 5] = [
        Self::One,
        Self::TwoToFour,
        Self::FiveToEight,
        Self::NineToSixteen,
        Self::SeventeenPlus,
    ];

    #[must_use]
    pub const fn from_remaining_plies(plies: usize) -> Self {
        match plies {
            0 | 1 => Self::One,
            2..=4 => Self::TwoToFour,
            5..=8 => Self::FiveToEight,
            9..=16 => Self::NineToSixteen,
            _ => Self::SeventeenPlus,
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::One => "1",
            Self::TwoToFour => "2-4",
            Self::FiveToEight => "5-8",
            Self::NineToSixteen => "9-16",
            Self::SeventeenPlus => "17+",
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::One => 0,
            Self::TwoToFour => 1,
            Self::FiveToEight => 2,
            Self::NineToSixteen => 3,
            Self::SeventeenPlus => 4,
        }
    }
}

/// Required validation metrics for one distance and outcome bucket.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EndgameDistanceMetrics {
    pub examples: usize,
    pub policy_loss: f32,
    pub policy_top1_accuracy: f32,
    pub wdl_loss: f32,
    pub wdl_top1_accuracy: f32,
    pub scalar_value_loss: f32,
}

impl EndgameDistanceMetrics {
    const fn new(examples: usize, metrics: LossMetrics) -> Self {
        Self {
            examples,
            policy_loss: metrics.policy_loss,
            policy_top1_accuracy: metrics.policy_top1_accuracy,
            wdl_loss: metrics.value_loss,
            wdl_top1_accuracy: metrics.wdl_top1_accuracy,
            scalar_value_loss: metrics.scalar_value_loss,
        }
    }
}

/// Metrics at one distance, with draws kept separate from decisive results.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EndgameDistanceBucketReport {
    pub distance: EndgameDistance,
    pub all: EndgameDistanceMetrics,
    pub draws: EndgameDistanceMetrics,
    pub decisive: EndgameDistanceMetrics,
}

/// One checkpoint evaluated on a frozen, whole-game validation split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EndgameDistanceReport {
    pub report_version: u16,
    pub checkpoint_generation: u32,
    pub split_seed: u64,
    pub validation_fraction: f32,
    pub buffer_games: usize,
    pub validation_games: usize,
    pub drawn_validation_games: usize,
    pub decisive_validation_games: usize,
    pub validation_examples: usize,
    pub non_starter_draw_policy_weight: f32,
    pub buckets: Vec<EndgameDistanceBucketReport>,
}

#[derive(Default)]
struct BucketExamples {
    all: Vec<TrainingExample>,
    draws: Vec<TrainingExample>,
    decisive: Vec<TrainingExample>,
}

/// Evaluates a checkpoint without changing training data, sampling, or model state.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn endgame_distance_report<B: Backend<FloatElem = f32>>(
    model: &AlphaZeroNetwork<B>,
    checkpoint_generation: u32,
    split: &DatasetSplit,
    split_seed: u64,
    validation_fraction: f32,
    batch_size: usize,
    non_starter_draw_policy_weight: f32,
    device: &B::Device,
) -> EndgameDistanceReport {
    let mut grouped: [BucketExamples; 5] = std::array::from_fn(|_| BucketExamples::default());
    let mut validation_examples = 0;
    let mut drawn_validation_games = 0;
    let mut decisive_validation_games = 0;

    for game in &split.validation_games {
        let is_draw = matches!(game.outcome, Outcome::Draw { .. });
        if is_draw {
            drawn_validation_games += 1;
        } else {
            decisive_validation_games += 1;
        }
        let examples = game.selected_examples(false, None);
        let game_examples = examples.len();
        validation_examples += game_examples;
        for (index, example) in examples.into_iter().enumerate() {
            let distance = EndgameDistance::from_remaining_plies(game_examples - index);
            let bucket = &mut grouped[distance.index()];
            bucket.all.push(example.clone());
            if is_draw {
                bucket.draws.push(example);
            } else {
                bucket.decisive.push(example);
            }
        }
    }

    let buckets = EndgameDistance::ALL
        .into_iter()
        .zip(grouped)
        .map(|(distance, examples)| EndgameDistanceBucketReport {
            distance,
            all: evaluate_examples(
                model,
                &examples.all,
                batch_size,
                non_starter_draw_policy_weight,
                device,
            ),
            draws: evaluate_examples(
                model,
                &examples.draws,
                batch_size,
                non_starter_draw_policy_weight,
                device,
            ),
            decisive: evaluate_examples(
                model,
                &examples.decisive,
                batch_size,
                non_starter_draw_policy_weight,
                device,
            ),
        })
        .collect();

    EndgameDistanceReport {
        report_version: ENDGAME_DISTANCE_REPORT_VERSION,
        checkpoint_generation,
        split_seed,
        validation_fraction,
        buffer_games: split.training_games.len() + split.validation_games.len(),
        validation_games: split.validation_games.len(),
        drawn_validation_games,
        decisive_validation_games,
        validation_examples,
        non_starter_draw_policy_weight,
        buckets,
    }
}

fn evaluate_examples<B: Backend<FloatElem = f32>>(
    model: &AlphaZeroNetwork<B>,
    examples: &[TrainingExample],
    batch_size: usize,
    non_starter_draw_policy_weight: f32,
    device: &B::Device,
) -> EndgameDistanceMetrics {
    EndgameDistanceMetrics::new(
        examples.len(),
        validate_model_with_policy_weight(
            model,
            examples,
            batch_size,
            non_starter_draw_policy_weight,
            device,
        ),
    )
}

/// Returns the stable output path for one checkpoint report.
#[must_use]
pub fn endgame_distance_report_path(self_play_root: impl AsRef<Path>, generation: u32) -> PathBuf {
    self_play_root
        .as_ref()
        .join("diagnostics")
        .join("endgame-distance")
        .join(format!("generation-{generation:06}.json"))
}

/// Atomically stores an endgame-distance report.
///
/// # Errors
///
/// Returns [`EndgameDiagnosticError`] when serialization or I/O fails.
pub fn save_endgame_distance_report(
    path: impl AsRef<Path>,
    report: &EndgameDistanceReport,
) -> Result<(), EndgameDiagnosticError> {
    let path = path.as_ref();
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("endgame-distance.json");
    let temporary = path.with_file_name(format!(".{name}-{}.tmp", std::process::id()));
    fs::write(&temporary, serde_json::to_vec(report)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[derive(Debug, Error)]
pub enum EndgameDiagnosticError {
    #[error("endgame diagnostic I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("endgame diagnostic JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
