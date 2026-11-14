//! Gameplay and asynchronous state-machine tests for the terminal interface.

use std::time::{Duration, Instant};

use ratatui::crossterm::event::{KeyCode, KeyEvent};
use yokai::{
    Action, ActionAnalysis, BOARD_SQUARES, Game, HandPiece, Piece, PieceKind, Player, Position,
    Replay, SearchResult, Square,
};

use super::ai::{AiCommand, AiEvent, AiWorker};
use super::app::*;
use super::render::*;
use super::session::*;
use super::{AI_MOVE_DELAY, AI_SOURCE_FOCUS_DURATION};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, ratatui::crossterm::event::KeyModifiers::NONE)
}

#[test]
fn human_match_selects_and_applies_only_a_legal_move() {
    let mut app = App::for_human_match();
    // The cursor starts on the Koropokkuru (c1); the Kodama stands on c3.
    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Enter));
    assert_eq!(
        app.selection,
        Some(Selection::MoveFrom("c3".parse().unwrap()))
    );

    // A Kodama cannot step backward onto the empty c2.
    app.handle_key(key(KeyCode::Down));
    app.handle_key(key(KeyCode::Enter));
    assert!(app.game().actions().is_empty());
    assert_eq!(
        app.selection,
        Some(Selection::MoveFrom("c3".parse().unwrap()))
    );

    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Enter));

    assert_eq!(app.game().actions(), &["c3-c4".parse().unwrap()]);
    assert_eq!(app.game().position().side_to_move(), Player::Second);
    assert_eq!(app.selection, None);
}

/// A human match from a sparse position: both Koropokkuru in corners plus
/// one piece of Player 1 on `square`, where the cursor also starts.
fn app_with_first_piece(kind: PieceKind, square: &str) -> App {
    let square: Square = square.parse().unwrap();
    let mut board = [None; BOARD_SQUARES];
    board["e1".parse::<Square>().unwrap().index()] =
        Some(Piece::new(PieceKind::Koropokkuru, Player::First));
    board["e6".parse::<Square>().unwrap().index()] =
        Some(Piece::new(PieceKind::Koropokkuru, Player::Second));
    board[square.index()] = Some(Piece::new(kind, Player::First));
    let position = Position::from_parts(board, [[0; 3]; 2], Player::First).unwrap();
    let mut app = App::for_match_session(MatchSession {
        game: Game::from_position(position),
        controllers: [Controller::Human, Controller::Human],
        predictions: None,
        ai: None,
    });
    app.cursor = square;
    app
}

#[test]
fn an_optional_promotion_is_asked_before_the_move_is_played() {
    for (answer, expected) in [('y', "b4-b5+"), ('n', "b4-b5")] {
        let mut app = app_with_first_piece(PieceKind::Oni, "b4");
        app.handle_key(key(KeyCode::Enter));
        app.handle_key(key(KeyCode::Up));
        app.handle_key(key(KeyCode::Enter));

        // The step is known but nothing is played until the question is answered.
        assert!(app.game().actions().is_empty());
        assert_eq!(
            app.selection,
            Some(Selection::Promotion {
                from: "b4".parse().unwrap(),
                to: "b5".parse().unwrap(),
            })
        );
        assert!(app.legal_destination("b5".parse().unwrap()));
        assert_eq!(app.selected_source(), Some("b4".parse().unwrap()));
        // The prompt is modal: movement keys do not reach the board.
        app.handle_key(key(KeyCode::Left));
        assert_eq!(app.cursor, "b5".parse().unwrap());

        // `N` answers the question here; it does not start a new game.
        app.handle_key(key(KeyCode::Char(answer)));
        assert_eq!(app.game().actions(), &[expected.parse().unwrap()]);
        assert_eq!(app.selection, None);
    }

    let mut app = app_with_first_piece(PieceKind::Oni, "b4");
    app.handle_key(key(KeyCode::Enter));
    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Enter));
    app.handle_key(key(KeyCode::Esc));
    assert!(app.game().actions().is_empty());
    assert_eq!(app.selection, None);
}

#[test]
fn a_move_with_a_single_promotion_variant_is_played_without_asking() {
    // A Kodama reaching the last row must promote.
    let mut app = app_with_first_piece(PieceKind::Kodama, "b5");
    app.handle_key(key(KeyCode::Enter));
    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Enter));
    assert_eq!(app.game().actions(), &["b5-b6+".parse().unwrap()]);

    // A Kirin has no promoted face, even inside the promotion zone.
    let mut app = app_with_first_piece(PieceKind::Kirin, "b5");
    app.handle_key(key(KeyCode::Enter));
    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Enter));
    assert_eq!(app.game().actions(), &["b5-b6".parse().unwrap()]);
}

#[test]
fn captured_piece_can_be_selected_from_the_hand_and_dropped() {
    let mut app = App::for_human_match();
    app.apply_action("c3-c4".parse().unwrap());
    app.apply_action("a6-a5".parse().unwrap());

    app.select_hand_piece(2);
    assert_eq!(app.selection, Some(Selection::Drop(HandPiece::Kodama)));
    app.cursor = "e2".parse().unwrap();
    let now = Instant::now();
    assert!(app.legal_destination("e2".parse().unwrap()));
    assert_eq!(
        square_highlight(&app, "e2".parse().unwrap(), now),
        SquareHighlight::Cursor
    );
    assert_eq!(
        square_highlight(&app, "a2".parse().unwrap(), now),
        SquareHighlight::LegalDestination
    );
    // Player 1 already has an unpromoted Kodama in the c file.
    assert!(!app.legal_destination("c2".parse().unwrap()));
    assert!(app.selected_hand_piece(Player::First, HandPiece::Kodama, 2));
    app.activate_board();

    assert_eq!(
        app.game().actions().last(),
        Some(&"kodama@e2".parse().unwrap())
    );
    assert_eq!(
        app.game().position().piece_at("e2".parse().unwrap()),
        Some(Piece::new(PieceKind::Kodama, Player::First))
    );
}

#[test]
fn champion_predictions_are_computed_for_the_human_turn_without_playing() {
    let (worker, events, commands) = AiWorker::stub();
    let mut app = App::for_cpu_match(worker);
    let now = Instant::now();
    events
        .send(AiEvent::Ready {
            generation: 16,
            simulations: 400,
        })
        .unwrap();

    app.tick_at(now);
    match commands.try_recv().expect("human search request") {
        AiCommand::Search { request_id, game } => {
            assert_eq!(request_id, 0);
            assert!(game.actions().is_empty());
            assert_eq!(game.position().side_to_move(), Player::First);
        }
        _ => panic!("expected a search for the starting position"),
    }
    let human_action = app.game().legal_actions()[0];
    let analysis = ActionAnalysis {
        action: human_action,
        prior: 0.5,
        q_value: 0.25,
        visits: 400,
        visit_probability: 1.0,
    };
    events
        .send(AiEvent::SearchReady {
            request_id: 0,
            result: Box::new(SearchResult {
                best_action: human_action,
                selected_action: human_action,
                root_value: 0.25,
                policy: [0.0; yokai::POLICY_ACTIONS],
                analysis: vec![analysis],
            }),
            search_started: now,
            search_finished: now + Duration::from_millis(20),
        })
        .unwrap();

    app.tick_at(now + Duration::from_millis(20));

    assert!(app.game().actions().is_empty());
    let view = analysis_view(&app);
    assert_eq!(view.actions, Some([analysis].as_slice()));
    assert_eq!(view.perspective, Some((Player::First, Controller::Human)));
    let Session::Match(session) = &app.session else {
        panic!("expected match session");
    };
    assert!(matches!(session.ai.as_ref().unwrap().state, AiState::Idle));
    assert!(commands.try_recv().is_err());
}

#[test]
fn stale_human_predictions_are_ignored_after_the_human_moves() {
    let (worker, events, commands) = AiWorker::stub();
    let mut app = App::for_cpu_match(worker);
    let now = Instant::now();
    events
        .send(AiEvent::Ready {
            generation: 16,
            simulations: 400,
        })
        .unwrap();
    app.tick_at(now);
    assert!(matches!(
        commands.try_recv().expect("initial human search"),
        AiCommand::Search { request_id: 0, .. }
    ));

    let human_action = "c3-c4".parse().unwrap();
    app.apply_action(human_action);
    assert!(matches!(
        commands.try_recv().expect("human action advances the tree"),
        AiCommand::Advance { action, .. } if action == human_action
    ));
    assert!(matches!(
        commands.try_recv().expect("CPU search request"),
        AiCommand::Search { request_id: 1, .. }
    ));
    events
        .send(AiEvent::SearchReady {
            request_id: 0,
            result: Box::new(SearchResult {
                best_action: human_action,
                selected_action: human_action,
                root_value: -0.5,
                policy: [0.0; yokai::POLICY_ACTIONS],
                analysis: Vec::new(),
            }),
            search_started: now,
            search_finished: now + Duration::from_millis(10),
        })
        .unwrap();

    app.tick_at(now + Duration::from_millis(10));

    assert!(analysis_view(&app).actions.is_none());
    let Session::Match(session) = &app.session else {
        panic!("expected match session");
    };
    assert!(matches!(
        session.ai.as_ref().unwrap().state,
        AiState::Thinking {
            request_id: 1,
            purpose: SearchPurpose::CpuMove,
            ..
        }
    ));
}

#[test]
fn cpu_result_focuses_source_then_destination_before_being_applied() {
    let (worker, events, commands) = AiWorker::stub();
    let mut app = App::for_cpu_match(worker);
    let now = Instant::now();
    events
        .send(AiEvent::Ready {
            generation: 16,
            simulations: 400,
        })
        .unwrap();
    app.tick_at(now);
    assert!(matches!(
        commands.try_recv().expect("initial human search"),
        AiCommand::Search { request_id: 0, .. }
    ));

    let human_action = "c3-c4".parse().unwrap();
    app.apply_action(human_action);
    assert!(matches!(
        commands.try_recv().expect("human action advances the tree"),
        AiCommand::Advance { action, .. } if action == human_action
    ));
    assert!(matches!(
        commands.try_recv().expect("CPU search request"),
        AiCommand::Search { request_id: 1, .. }
    ));

    let cpu_action = app.game().legal_actions()[0];
    let analysis = ActionAnalysis {
        action: cpu_action,
        prior: 0.5,
        q_value: 0.25,
        visits: 400,
        visit_probability: 1.0,
    };
    events
        .send(AiEvent::SearchReady {
            request_id: 1,
            result: Box::new(SearchResult {
                best_action: cpu_action,
                selected_action: cpu_action,
                root_value: 0.25,
                policy: [0.0; yokai::POLICY_ACTIONS],
                analysis: vec![analysis],
            }),
            search_started: now,
            search_finished: now + Duration::from_millis(20),
        })
        .unwrap();

    let result_received = now + Duration::from_millis(20);
    app.tick_at(result_received);
    assert_eq!(app.game().actions().len(), 1);
    assert_eq!(analysis_view(&app).actions, Some([analysis].as_slice()));
    let Action::Move { from, to, .. } = cpu_action else {
        panic!("expected board move");
    };
    assert_eq!(app.selected_source(), Some(from));
    assert_eq!(app.focused_ai_destination(result_received), None);
    assert_eq!(
        square_highlight(&app, from, result_received),
        SquareHighlight::SelectedSource
    );
    assert_eq!(
        square_highlight(&app, to, result_received),
        SquareHighlight::LegalDestination
    );

    let destination_focus = result_received + AI_SOURCE_FOCUS_DURATION;
    app.tick_at(destination_focus);
    assert_eq!(app.game().actions().len(), 1);
    assert_eq!(app.focused_ai_destination(destination_focus), Some(to));
    assert_eq!(
        square_highlight(&app, to, destination_focus),
        SquareHighlight::FocusedDestination
    );

    app.tick_at(
        (result_received + AI_MOVE_DELAY)
            .checked_sub(Duration::from_millis(1))
            .unwrap(),
    );
    assert_eq!(app.game().actions().len(), 1);
    app.tick_at(result_received + AI_MOVE_DELAY);
    assert_eq!(
        app.game().actions(),
        &["c3-c4".parse().unwrap(), cpu_action]
    );
    assert!(analysis_view(&app).actions.is_none());
    let Session::Match(session) = &app.session else {
        panic!("expected match session");
    };
    assert!(matches!(
        session.ai.as_ref().unwrap().state,
        AiState::Thinking {
            request_id: 2,
            purpose: SearchPurpose::HumanPrediction,
            ..
        }
    ));
    assert!(matches!(
        commands.try_recv().expect("CPU action advances the tree"),
        AiCommand::Advance { action, .. } if action == cpu_action
    ));
}

#[test]
fn replay_seeking_reconstructs_positions_and_exposes_stored_analysis() {
    let mut game = Game::new(Player::First);
    let action: Action = "c3-c4".parse().unwrap();
    game.apply(action).unwrap();
    let analysis = ActionAnalysis {
        action,
        prior: 0.4,
        q_value: 0.2,
        visits: 12,
        visit_probability: 0.6,
    };
    let replay = Replay::from_game(&game, None).with_analyses(vec![vec![analysis]]);
    let replay = Replay::from_json(&replay.to_json_pretty().unwrap()).unwrap();
    let mut app = App::for_replay(replay).expect("valid replay");

    assert_eq!(analysis_view(&app).actions, Some([analysis].as_slice()));
    app.seek_replay(1);
    assert_eq!(app.game().actions(), &[action]);
    assert!(analysis_view(&app).actions.is_none());
}

#[test]
fn history_window_tracks_the_replay_cursor() {
    assert_eq!(visible_window(20, 0, 5), (0, 5));
    assert_eq!(visible_window(20, 10, 5), (8, 13));
    assert_eq!(visible_window(20, 20, 5), (15, 20));
}
