//! Yokai No Mori 3x4 engine and supporting formats.
//!
//! The engine stores the board in an absolute orientation: First starts at the
//! bottom and moves toward row zero. Neural-network canonicalization belongs in
//! the policy/encoder layers, never in the rules themselves.
//!
//! Public items are documented deliberately: this crate is also meant to be a
//! readable Rust and `AlphaZero` learning project, not only an executable engine.

#![warn(missing_docs)]

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
    ENCODER_VERSION, EncodedPosition, GLOBAL_FEATURES_PER_FRAME, GLOBAL_INPUT_FEATURES,
    HISTORY_LENGTH, HISTORY_POSITIONS, INPUT_PLANES, POLICY_CONTEXT_FEATURES,
    checkpoint::{
        MODEL_FORMAT_VERSION, ModelMetadata, ModelStoreError, load_champion, load_generation,
        load_training_generation, next_generation, publish_champion, save_generation,
        save_training_generation, stored_generations,
    },
    encode_game, encode_position, encode_position_with_history, encoded_batch_tensor,
    evaluator::{
        CpuBackend, CpuTrainingBackend, MetalBackend, MetalTrainingBackend, NetworkEvaluator,
    },
    global_batch_tensor,
    model::{AlphaZeroNetwork, AlphaZeroNetworkConfig, NetworkOutput},
    policy_context_batch_tensor,
    service::{InferenceClient, InferenceService, InferenceServiceError, InferenceStats},
};
pub use policy::{POLICY_ACTIONS, PolicyIndex};
pub use replay::{REPLAY_FORMAT_VERSION, Replay, ReplayError};
pub use search::{
    ActionAnalysis, CachedEvaluator, Evaluation, EvaluationError, EvaluationRequest, Evaluator,
    LeafEvaluation, Mcts, SearchConfig, SearchError, SearchResult, TemperatureSchedule,
    UniformEvaluator, random_rollout_value,
};
pub use training::arena::{
    ArenaError, ArenaProgress, ArenaResult, ArenaSeatResult, run_arena, run_arena_with_progress,
};
pub use training::config::{
    ArenaConfig, BackendKind, LearningRateStage, OptimizationConfig, PathsConfig,
    SelfPlayBootstrapConfig, SelfPlayBootstrapMode, SelfPlayConfig, TerminalWindowSchedule,
    TrainingConfig, TrainingConfigError,
};
pub use training::data::{
    DatasetDiagnostics, DatasetSplit, PolicyTargetDiagnostics, ReplayBuffer, ReplayBufferConfig,
    SelfPlayEvaluator, SelfPlayGame, SelfPlayRecorder, TrainingDataError, TrainingExample,
    dataset_diagnostics, mirror_policy,
};
pub use training::diagnostics::{
    ENDGAME_DISTANCE_REPORT_VERSION, EndgameDiagnosticError, EndgameDistance,
    EndgameDistanceBucketReport, EndgameDistanceMetrics, EndgameDistanceReport,
    endgame_distance_report, endgame_distance_report_path, save_endgame_distance_report,
};
pub use training::pipeline::{
    GameOutcomeStats, GenerationReport, PipelineError, PromotionDecision, TrainingProgress,
    bootstrap_champion, load_replay_buffer, run_generation, run_generation_with_progress,
    save_replay_buffer,
};
pub use training::self_play::{
    SelfPlayError, generate_self_play, generate_self_play_with_progress,
    generate_self_play_with_restarts, generate_self_play_with_restarts_and_progress,
    planned_restart_count, play_self_play_game, play_self_play_game_from_restart,
};
pub use training::trainer::{
    AlphaZeroOptimizer, AlphaZeroTrainingState, LossMetrics, TrainingReport, TrainingStepReport,
    new_optimizer, train_candidate, train_candidate_with_progress, train_state_with_progress,
    validate_model, validate_model_with_policy_weight,
};
