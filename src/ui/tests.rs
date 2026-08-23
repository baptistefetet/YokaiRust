//! State-machine and rendering tests for the terminal interface.

use std::time::{Duration, Instant};

use ratatui::{
    Terminal,
    backend::TestBackend,
    crossterm::event::{KeyCode, KeyEvent},
};
use yokai::{
    Action, ActionAnalysis, Game, HandPiece, Piece, PieceKind, Player, Replay, SearchResult,
};

use super::ai::{AiCommand, AiEvent, AiWorker};
use super::app::*;
use super::render::*;
use super::session::*;
use super::*;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, ratatui::crossterm::event::KeyModifiers::NONE)
}

#[test]
fn human_match_selects_and_applies_only_a_legal_move() {
    let mut app = App::for_human_match();
    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Enter));
    assert_eq!(
        app.selection,
        Some(Selection::MoveFrom("b2".parse().unwrap()))
    );

    app.handle_key(key(KeyCode::Up));
    app.handle_key(key(KeyCode::Enter));

    assert_eq!(app.game().actions(), &["b2-b3".parse().unwrap()]);
    assert_eq!(app.game().position().side_to_move(), Player::Second);
    assert_eq!(app.selection, None);
}

#[test]
fn illegal_destination_keeps_the_selected_piece() {
    let mut app = App::for_human_match();
    app.cursor = "b2".parse().unwrap();
    app.activate_board();
    app.cursor = "a2".parse().unwrap();
    app.activate_board();

    assert!(app.game().actions().is_empty());
    assert_eq!(
        app.selection,
        Some(Selection::MoveFrom("b2".parse().unwrap()))
    );
    assert!(app.notice.contains("Illegal"));
}

#[test]
fn captured_piece_can_be_selected_from_the_hand_and_dropped() {
    let mut app = App::for_human_match();
    app.apply_action("b2-b3".parse().unwrap());
    app.apply_action("a4-a3".parse().unwrap());

    app.select_hand_piece(2);
    assert_eq!(app.selection, Some(Selection::Drop(HandPiece::Kodama)));
    app.cursor = "c2".parse().unwrap();
    let now = Instant::now();
    assert!(app.legal_destination("c2".parse().unwrap()));
    assert_eq!(
        square_highlight(&app, "c2".parse().unwrap(), now),
        SquareHighlight::Cursor
    );
    assert_eq!(
        square_highlight(&app, "a2".parse().unwrap(), now),
        SquareHighlight::LegalDestination
    );
    assert!(app.selected_hand_piece(Player::First, HandPiece::Kodama, 2));
    app.activate_board();

    assert_eq!(
        app.game().actions().last(),
        Some(&"kodama@c2".parse().unwrap())
    );
    assert_eq!(
        app.game().position().piece_at("c2".parse().unwrap()),
        Some(Piece::new(PieceKind::Kodama, Player::First))
    );
}

#[test]
fn cpu_mode_places_the_human_at_the_bottom() {
    let (worker, _events, _commands) = AiWorker::stub();
    let app = App::for_cpu_match(worker);
    let Session::Match(session) = &app.session else {
        panic!("expected match session");
    };
    assert_eq!(session.controller(Player::First), Controller::Human);
    assert_eq!(session.controller(Player::Second), Controller::Cpu);
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

    let human_action = "b2-b3".parse().unwrap();
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

    let human_action = "b2-b3".parse().unwrap();
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
    let Action::Move { from, to } = cpu_action else {
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
        &["b2-b3".parse().unwrap(), cpu_action]
    );
    assert!(app.notice.contains("CPU search 0.02s"));
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
    let action: Action = "b2-b3".parse().unwrap();
    game.apply(action).unwrap();
    let analysis = ActionAnalysis {
        action,
        prior: 0.4,
        q_value: 0.2,
        visits: 12,
        visit_probability: 0.6,
    };
    let replay = Replay::from_game(&game, None).with_analyses(vec![vec![analysis]]);
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

#[test]
fn minimum_supported_terminal_renders_every_primary_panel() {
    let app = App::for_human_match();
    let backend = TestBackend::new(MINIMUM_WIDTH, MINIMUM_HEIGHT);
    let mut terminal = Terminal::new(backend).expect("test terminal");
    terminal.draw(|frame| render(frame, &app)).expect("render");

    let buffer = terminal.backend().buffer();
    let mut text = String::new();
    for y in 0..MINIMUM_HEIGHT {
        for x in 0..MINIMUM_WIDTH {
            text.push_str(buffer.cell((x, y)).expect("screen cell").symbol());
        }
        text.push('\n');
    }
    assert!(text.contains("Move history"));
    assert!(text.contains("Predictions"));
    assert!(text.contains("Player 1 · bottom"));
    assert!(text.contains("▲ KD"));
    assert!(text.contains("b2"));
}
