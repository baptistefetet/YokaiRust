//! Browser adapter for the Rust rules, neural network and MCTS engine.

use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use serde::Serialize;
use wasm_bindgen::prelude::*;
use yokai::{
    Action, BOARD_HEIGHT, BOARD_SQUARES, BOARD_WIDTH, DrawReason, Game, HandPiece, Mcts,
    ModelMetadata, NetworkEvaluator, Outcome, Piece, PieceKind, Player, SearchConfig, SearchError,
    Transition, WinReason,
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

const DEFAULT_SEED: u64 = 0x594f_4b41_4957_4542;

#[derive(Serialize)]
struct Snapshot<'a> {
    columns: u8,
    rows: u8,
    board: &'a [Option<Piece>; BOARD_SQUARES],
    hands: &'a [[u8; 3]; 2],
    /// Piece counted by each column of `hands`, so JavaScript does not repeat
    /// the engine's hand order.
    hand_kinds: [HandPiece; 3],
    /// The promoted face of every kind that has one, for drawing the choice
    /// between a plain and a promoting move.
    promotions: Vec<Promotion>,
    side_to_move: Player,
    outcome: Outcome,
    /// Human-readable end-of-game sentence; `null` while the game is ongoing.
    /// Rendering this text in Rust keeps JavaScript decoupled from the serde
    /// names of `WinReason`/`DrawReason` variants.
    outcome_text: Option<&'static str>,
    legal_actions: Vec<Action>,
    ply: usize,
    /// Sides played by a person. The champion plays every other side.
    humans: &'static [Player],
}

#[derive(Serialize)]
struct Promotion {
    kind: PieceKind,
    promoted: PieceKind,
}

fn promotions() -> Vec<Promotion> {
    HandPiece::ALL
        .into_iter()
        .filter_map(|piece| {
            let kind = piece.piece_kind();
            kind.promoted().map(|promoted| Promotion { kind, promoted })
        })
        .collect()
}

/// Sides played by people: First alone against the champion, or both sides.
fn human_sides(human_players: u8) -> Result<&'static [Player], JsValue> {
    match human_players {
        1 => Ok(&[Player::First]),
        2 => Ok(&[Player::First, Player::Second]),
        _ => Err(js_error("the number of human players must be 1 or 2")),
    }
}

const fn outcome_text(outcome: Outcome) -> Option<&'static str> {
    match outcome {
        Outcome::Ongoing => None,
        Outcome::Win { reason, .. } => Some(match reason {
            WinReason::KoropokkuruCaptured => "The Koropokkuru was captured.",
            WinReason::OpponentHasNoLegalAction => "The opponent has no legal action left.",
        }),
        Outcome::Draw { reason } => Some(match reason {
            DrawReason::ThreefoldRepetition => "The position was repeated three times.",
            DrawReason::MoveLimit => "The move limit was reached without a winner.",
        }),
    }
}

#[derive(Serialize)]
struct AppliedAction<'a> {
    action: Action,
    captured: Option<yokai::PieceKind>,
    promoted: bool,
    state: Snapshot<'a>,
}

/// Complete browser game for one person against the champion, or for two
/// people sharing the screen. JavaScript only renders snapshots and submits
/// actions; legality, history, search and inference remain in Rust.
#[wasm_bindgen]
pub struct WebGame {
    game: Game,
    search: Mcts<NetworkEvaluator<WebBackend>>,
    humans: &'static [Player],
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

    /// Starts a fresh game for one or two human players. First moves first;
    /// with one human player, the champion plays Second.
    ///
    /// # Errors
    ///
    /// Returns a JavaScript error if `human_players` is neither 1 nor 2, or if
    /// the initial snapshot cannot be serialized.
    pub fn reset(&mut self, human_players: u8) -> Result<String, JsValue> {
        self.humans = human_sides(human_players)?;
        self.game = Game::new(Player::First);
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
        if !self.human_to_move() {
            return Err(js_error("it is not a human player's turn"));
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
        if self.human_to_move() {
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
    fn human_to_move(&self) -> bool {
        self.humans.contains(&self.game.position().side_to_move())
    }

    fn snapshot(&self) -> Snapshot<'_> {
        Snapshot {
            columns: BOARD_WIDTH,
            rows: BOARD_HEIGHT,
            board: self.game.position().board(),
            hands: self.game.position().hands(),
            hand_kinds: HandPiece::ALL,
            promotions: promotions(),
            side_to_move: self.game.position().side_to_move(),
            outcome: self.game.outcome(),
            outcome_text: outcome_text(self.game.outcome()),
            legal_actions: self.game.legal_actions(),
            ply: self.game.actions().len(),
            humans: self.humans,
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

/// Loads one exported champion and creates a new browser game for one or two
/// human players.
///
/// # Errors
///
/// Returns a JavaScript error when `human_players` is neither 1 nor 2,
/// metadata is incompatible, weights cannot be decoded, backend initialization
/// fails, or search configuration is invalid.
#[wasm_bindgen(js_name = createGame)]
pub async fn create_game(
    model_bytes: Vec<u8>,
    metadata_json: String,
    simulations: u32,
    human_players: u8,
) -> Result<WebGame, JsValue> {
    console_error_panic_hook::set_once();
    if simulations == 0 {
        return Err(js_error("simulations must be greater than zero"));
    }
    let humans = human_sides(human_players)?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
        .map_err(|error| js_error(format!("invalid model metadata: {error}")))?;
    metadata
        .validate()
        .map_err(|error| js_error(format!("incompatible model: {error}")))?;

    initialize_backend().await;

    let device = burn::tensor::Device::<WebBackend>::default();
    let record = BinBytesRecorder::<FullPrecisionSettings, Vec<u8>>::default()
        .load(model_bytes, &device)
        .map_err(|error| js_error(format!("could not decode model: {error}")))?;
    let model = metadata.architecture.init(&device).load_record(record);
    let evaluator = NetworkEvaluator::new(model, device);
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
    Ok(WebGame {
        game: Game::new(Player::First),
        search,
        humans,
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
