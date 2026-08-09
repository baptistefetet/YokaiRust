//! Yokai No Mori 3x4 engine and supporting formats.
//!
//! The engine stores the board in an absolute orientation: First starts at the
//! bottom and moves toward row zero. Neural-network canonicalization belongs in
//! the policy/encoder layers, never in the rules themselves.

pub mod game;
pub mod neural;
pub mod notation;
pub mod policy;
pub mod replay;
pub mod search;
pub mod training;

pub use game::{
    Action, BOARD_HEIGHT, BOARD_SQUARES, BOARD_WIDTH, DrawReason, Game, HandPiece, MoveError,
    Outcome, Piece, PieceKind, Player, Position, PositionError, RULES_VERSION, Square, Transition,
    WinReason,
};
pub use neural::{
    ENCODER_VERSION, EncodedPosition, INPUT_PLANES,
    checkpoint::{
        MODEL_FORMAT_VERSION, ModelMetadata, ModelStoreError, load_champion, load_generation,
        next_generation, publish_champion, save_generation,
    },
    encode_game, encode_position, encoded_batch_tensor,
    evaluator::{
        CpuBackend, CpuTrainingBackend, MetalBackend, MetalTrainingBackend, NetworkEvaluator,
    },
    model::{AlphaZeroNetwork, AlphaZeroNetworkConfig, NetworkOutput},
    service::{InferenceClient, InferenceService, InferenceServiceError, InferenceStats},
};
pub use policy::{POLICY_ACTIONS, PolicyIndex};
pub use replay::{REPLAY_FORMAT_VERSION, Replay, ReplayError};
pub use search::{
    ActionAnalysis, CachedEvaluator, Evaluation, EvaluationError, EvaluationRequest, Evaluator,
    Mcts, SearchConfig, SearchError, SearchResult, TemperatureSchedule, UniformEvaluator,
};
pub use training::arena::{ArenaError, ArenaResult, run_arena, run_arena_with_progress};
pub use training::config::{
    ArenaConfig, BackendKind, OptimizationConfig, PathsConfig, SelfPlayConfig, TrainingConfig,
    TrainingConfigError,
};
pub use training::data::{
    DatasetSplit, ReplayBuffer, ReplayBufferConfig, SelfPlayGame, SelfPlayRecorder,
    TrainingDataError, TrainingExample, mirror_policy,
};
pub use training::pipeline::{
    GameOutcomeStats, GenerationReport, PipelineError, TrainingProgress, bootstrap_champion,
    load_replay_buffer, run_generation, run_generation_with_progress, save_replay_buffer,
};
pub use training::self_play::{
    SelfPlayError, generate_self_play, generate_self_play_with_progress, play_self_play_game,
};
pub use training::trainer::{
    EpochReport, LossMetrics, TrainingReport, train_candidate, train_candidate_with_progress,
    validate_model,
};
