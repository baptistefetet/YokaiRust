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
    ENCODER_VERSION, EncodedPosition, HISTORY_LENGTH, HISTORY_POSITIONS, INPUT_PLANES,
    POLICY_CONTEXT_FEATURES,
    checkpoint::{
        MODEL_FORMAT_VERSION, ModelMetadata, ModelStoreError, load_champion, load_generation,
        load_latest, load_learner, load_training_generation, next_generation, publish_champion,
        publish_latest, publish_learner, save_generation, save_training_generation,
    },
    encode_game, encode_position, encode_position_with_history, encoded_batch_tensor,
    evaluator::{
        CpuBackend, CpuTrainingBackend, MetalBackend, MetalTrainingBackend, NetworkEvaluator,
    },
    model::{AlphaZeroNetwork, AlphaZeroNetworkConfig, NetworkOutput},
    policy_context_batch_tensor,
    service::{InferenceClient, InferenceService, InferenceServiceError, InferenceStats},
};
pub use policy::{POLICY_ACTIONS, PolicyIndex};
pub use replay::{REPLAY_FORMAT_VERSION, Replay, ReplayError};
pub use search::{
    ActionAnalysis, CachedEvaluator, Evaluation, EvaluationError, EvaluationRequest, Evaluator,
    Mcts, SearchConfig, SearchError, SearchResult, TemperatureSchedule, UniformEvaluator,
};
pub use training::arena::{
    ArenaError, ArenaProgress, ArenaResult, ArenaSeatResult, run_arena, run_arena_with_progress,
};
pub use training::config::{
    ArenaConfig, BackendKind, OptimizationConfig, PathsConfig, SelfPlayConfig,
    TerminalWindowSchedule, TrainingConfig, TrainingConfigError,
};
pub use training::data::{
    CYCLE_RESTART_MAX_PLIES, CYCLE_RESTART_MIN_PLIES, DatasetDiagnostics, DatasetSplit,
    PolicyTargetDiagnostics, ReplayBuffer, ReplayBufferConfig, SelfPlayGame, SelfPlayRecorder,
    TrainingDataError, TrainingExample, dataset_diagnostics, mirror_policy,
};
pub use training::pipeline::{
    GameOutcomeStats, GenerationReport, PipelineError, PromotionDecision, TrainingProgress,
    bootstrap_champion, bootstrap_latest, load_replay_buffer, run_generation,
    run_generation_with_progress, save_replay_buffer,
};
pub use training::self_play::{
    SelfPlayError, generate_self_play, generate_self_play_with_progress,
    generate_self_play_with_restarts, generate_self_play_with_restarts_and_progress,
    planned_restart_count, play_self_play_game, play_self_play_game_from_restart,
};
pub use training::trainer::{
    AlphaZeroOptimizer, AlphaZeroTrainingState, LossMetrics, TrainingReport, TrainingStepReport,
    new_optimizer, train_candidate, train_candidate_with_progress, train_state_with_progress,
    validate_model,
};
