//! Browser adapter for the Rust rules, neural network and MCTS engine.

use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
    tensor::activation::softmax,
};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;
use yokai::{
    Action, AlphaZeroNetwork, AlphaZeroNetworkConfig, AsyncEvaluator, ENCODER_VERSION, Evaluation,
    EvaluationError, EvaluationRequest, Evaluator, Game, Mcts, Outcome, POLICY_ACTIONS, Piece,
    Player, RULES_VERSION, SearchConfig, SearchError, Transition, encode_position_with_history,
    encoded_batch_tensor, global_batch_tensor, policy_context_batch_tensor,
};

#[cfg(all(feature = "flex", feature = "webgpu"))]
compile_error!("enable either `flex` or `webgpu`, not both");
#[cfg(not(any(feature = "flex", feature = "webgpu")))]
compile_error!("enable either the `flex` or `webgpu` feature");

#[cfg(feature = "webgpu")]
use burn::backend::wgpu::{
    RuntimeOptions, Wgpu, WgpuDevice, graphics::AutoGraphicsApi, init_setup_async,
};

#[cfg(feature = "webgpu")]
type WebBackend = Wgpu<f32, i32>;
#[cfg(all(feature = "flex", not(feature = "webgpu")))]
type WebBackend = burn::backend::Flex<f32, i32>;

const MODEL_FORMAT_VERSION: u16 = 5;
const DEFAULT_SEED: u64 = 0x594f_4b41_4957_4542;

#[derive(Debug, Deserialize)]
struct WebModelMetadata {
    format_version: u16,
    encoder_version: u16,
    rules_version: u16,
    policy_actions: usize,
    generation: u32,
    architecture: AlphaZeroNetworkConfig,
}

impl WebModelMetadata {
    fn validate(&self) -> Result<(), JsValue> {
        let versions_match = self.format_version == MODEL_FORMAT_VERSION
            && self.encoder_version == ENCODER_VERSION
            && self.rules_version == RULES_VERSION
            && self.policy_actions == POLICY_ACTIONS;
        if versions_match {
            Ok(())
        } else {
            Err(js_error(format!(
                "incompatible model metadata (format={}, encoder={}, rules={}, policy={})",
                self.format_version, self.encoder_version, self.rules_version, self.policy_actions
            )))
        }
    }
}

struct BrowserEvaluator {
    model: AlphaZeroNetwork<WebBackend>,
    device: burn::tensor::Device<WebBackend>,
}

impl Evaluator for BrowserEvaluator {
    fn evaluate_batch(
        &mut self,
        _requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        Err(EvaluationError::Backend(
            "browser inference must use the asynchronous MCTS entry point".to_owned(),
        ))
    }
}

impl AsyncEvaluator for BrowserEvaluator {
    async fn evaluate_batch_async(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let encoded = requests
            .iter()
            .map(|request| {
                encode_position_with_history(
                    &request.position,
                    request.repetition_count,
                    request.current_player_is_starter,
                    &request.history,
                )
            })
            .collect::<Vec<_>>();
        let policy_contexts = requests
            .iter()
            .map(|request| request.action_repetition_counts)
            .collect::<Vec<_>>();
        let output = self.model.forward(
            encoded_batch_tensor(&encoded, &self.device),
            global_batch_tensor(&encoded, &self.device),
            policy_context_batch_tensor(&policy_contexts, &self.device),
        );
        let output_width = POLICY_ACTIONS + 3;
        let predictions = burn::tensor::Tensor::cat(
            vec![
                softmax(output.policy_logits, 1),
                softmax(output.wdl_logits, 1),
            ],
            1,
        )
        .into_data_async()
        .await
        .map_err(|error| EvaluationError::Backend(error.to_string()))?
        .to_vec::<f32>()
        .map_err(|error| EvaluationError::Backend(error.to_string()))?;

        if predictions.len() != requests.len() * output_width {
            return Err(EvaluationError::BatchSizeMismatch {
                expected: requests.len(),
                actual: predictions.len() / output_width,
            });
        }

        predictions
            .chunks_exact(output_width)
            .map(|prediction| {
                let policy = prediction[..POLICY_ACTIONS].try_into().map_err(|_| {
                    EvaluationError::Backend("invalid policy tensor width".to_owned())
                })?;
                let wdl = prediction[POLICY_ACTIONS..]
                    .try_into()
                    .map_err(|_| EvaluationError::Backend("invalid WDL tensor width".to_owned()))?;
                Ok(Evaluation::from_wdl(policy, wdl))
            })
            .collect()
    }
}

#[derive(Serialize)]
struct Snapshot<'a> {
    board: &'a [Option<Piece>; 12],
    hands: &'a [[u8; 3]; 2],
    side_to_move: Player,
    outcome: Outcome,
    legal_actions: Vec<Action>,
    ply: usize,
    human: Player,
}

#[derive(Serialize)]
struct AppliedAction<'a> {
    action: Action,
    captured: Option<yokai::PieceKind>,
    promoted: bool,
    state: Snapshot<'a>,
}

/// Complete one-player browser game. JavaScript only renders snapshots and
/// submits actions; legality, history, search and inference remain in Rust.
#[wasm_bindgen]
pub struct WebGame {
    game: Game,
    search: Mcts<BrowserEvaluator>,
    human: Player,
    generation: u32,
}

#[wasm_bindgen]
impl WebGame {
    /// Returns the initial or current state as JSON.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error if serialization unexpectedly fails.
    #[wasm_bindgen(js_name = snapshotJson)]
    pub fn snapshot_json(&self) -> Result<String, JsValue> {
        serde_json::to_string(&self.snapshot()).map_err(|error| js_error(error.to_string()))
    }

    /// Returns the generation identifier loaded into this browser game.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Starts a fresh one-player game with the human moving first.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error if the initial snapshot cannot be serialized.
    pub fn reset(&mut self) -> Result<String, JsValue> {
        self.game = Game::new(self.human);
        self.search.reset();
        self.snapshot_json()
    }

    /// Validates and applies a human action supplied in the snapshot format.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error when the turn or action is invalid, or when
    /// the resulting snapshot cannot be serialized.
    #[wasm_bindgen(js_name = playHuman)]
    pub fn play_human(&mut self, action_json: &str) -> Result<String, JsValue> {
        if self.game.position().side_to_move() != self.human {
            return Err(js_error("it is not the human player's turn"));
        }
        let action = serde_json::from_str::<Action>(action_json)
            .map_err(|error| js_error(format!("invalid action: {error}")))?;
        self.apply(action)
    }

    /// Searches, applies and returns the AI action.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error when it is not the AI turn, search fails, or
    /// the resulting snapshot cannot be serialized.
    #[wasm_bindgen(js_name = playAi)]
    pub async fn play_ai(&mut self) -> Result<String, JsValue> {
        if self.game.position().side_to_move() == self.human {
            return Err(js_error("it is not the AI player's turn"));
        }
        let result = self
            .search
            .search_async(&self.game, 0.0)
            .await
            .map_err(|error| search_error(&error))?;
        self.apply(result.selected_action)
    }
}

impl WebGame {
    fn snapshot(&self) -> Snapshot<'_> {
        Snapshot {
            board: self.game.position().board(),
            hands: self.game.position().hands(),
            side_to_move: self.game.position().side_to_move(),
            outcome: self.game.outcome(),
            legal_actions: self.game.legal_actions(),
            ply: self.game.actions().len(),
            human: self.human,
        }
    }

    fn apply(&mut self, action: Action) -> Result<String, JsValue> {
        let Transition {
            captured, promoted, ..
        } = self
            .game
            .apply(action)
            .map_err(|error| js_error(error.to_string()))?;
        let _ = self.search.advance_root(action, &self.game);
        serde_json::to_string(&AppliedAction {
            action,
            captured,
            promoted,
            state: self.snapshot(),
        })
        .map_err(|error| js_error(error.to_string()))
    }
}

/// Loads one exported champion and creates a new browser game.
///
/// # Errors
///
/// Returns a JavaScript error when metadata is incompatible, weights cannot be
/// decoded, backend initialization fails, or search configuration is invalid.
#[wasm_bindgen(js_name = createGame)]
pub async fn create_game(
    model_bytes: Vec<u8>,
    metadata_json: String,
    simulations: u32,
) -> Result<WebGame, JsValue> {
    console_error_panic_hook::set_once();
    if simulations == 0 {
        return Err(js_error("simulations must be greater than zero"));
    }
    let metadata: WebModelMetadata = serde_json::from_str(&metadata_json)
        .map_err(|error| js_error(format!("invalid model metadata: {error}")))?;
    metadata.validate()?;

    initialize_backend().await;

    let device = burn::tensor::Device::<WebBackend>::default();
    let record = BinBytesRecorder::<FullPrecisionSettings, Vec<u8>>::default()
        .load(model_bytes, &device)
        .map_err(|error| js_error(format!("could not decode model: {error}")))?;
    let model = metadata.architecture.init(&device).load_record(record);
    let evaluator = BrowserEvaluator { model, device };
    let config = SearchConfig {
        simulations,
        evaluation_batch_size: 8,
        dirichlet_weight: 0.0,
        repetition_contempt: 0.0,
        starter_draw_value: 0.0,
        ..SearchConfig::default()
    };
    let search =
        Mcts::new(evaluator, config, DEFAULT_SEED).map_err(|error| search_error(&error))?;
    let human = Player::First;
    Ok(WebGame {
        game: Game::new(human),
        search,
        human,
        generation: metadata.generation,
    })
}

/// Name of the inference backend compiled into this WASM module.
#[wasm_bindgen(js_name = backendName)]
#[must_use]
pub fn backend_name() -> String {
    #[cfg(feature = "webgpu")]
    {
        "WebGPU".to_owned()
    }
    #[cfg(all(feature = "flex", not(feature = "webgpu")))]
    {
        "CPU (Flex)".to_owned()
    }
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    js_sys::Error::new(message.as_ref()).into()
}

#[cfg(feature = "webgpu")]
async fn initialize_backend() {
    init_setup_async::<AutoGraphicsApi>(&WgpuDevice::default(), RuntimeOptions::default()).await;
}

#[cfg(all(feature = "flex", not(feature = "webgpu")))]
async fn initialize_backend() {
    core::future::ready(()).await;
}

fn search_error(error: &SearchError) -> JsValue {
    js_error(error.to_string())
}
