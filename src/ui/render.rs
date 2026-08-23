//! Pure rendering: layout, widgets and the text formatting helpers.

use std::{cmp::Ordering, fmt, fmt::Write as _, time::Instant};

use ratatui::{
    Frame,
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Cell, Paragraph, Row, Table, Wrap},
};
use yokai::{
    Action, ActionAnalysis, DrawReason, HandPiece, Outcome, Piece, PieceKind, Player, Square,
    WinReason,
};

use super::app::App;
use super::session::{AiState, Controller, SearchPurpose, Session};
use super::{MINIMUM_HEIGHT, MINIMUM_WIDTH};

pub(super) fn render(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();
    let now = Instant::now();
    if area.width < MINIMUM_WIDTH || area.height < MINIMUM_HEIGHT {
        let message = format!(
            "Terminal too small\n\nMinimum: {MINIMUM_WIDTH}×{MINIMUM_HEIGHT}\nCurrent: {}×{}\n\nPress Q to quit",
            area.width, area.height
        );
        frame.render_widget(
            Paragraph::new(message)
                .alignment(Alignment::Center)
                .block(Block::bordered().title(" YokaiRust ")),
            area,
        );
        return;
    }

    let [header, main, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(18),
        Constraint::Length(3),
    ])
    .areas(area);
    render_header(frame, app, header, now);
    render_main(frame, app, main, now);
    render_footer(frame, app, footer);
}

pub(super) fn render_header(frame: &mut Frame<'_>, app: &App, area: Rect, now: Instant) {
    let mode = match &app.session {
        Session::Match(session) => {
            let mut label = format!(
                "P1 {} vs P2 {}",
                session.controllers[0].label(),
                session.controllers[1].label()
            );
            if let Some(ai) = &session.ai
                && let Some(generation) = ai.generation
            {
                let _ = write!(label, " g{generation}");
                if let Some(simulations) = ai.simulations {
                    let _ = write!(label, "/{simulations}");
                }
            }
            label
        }
        Session::Replay(session) => {
            format!("Replay {}/{}", session.ply, session.replay.actions.len())
        }
    };
    let status = match &app.session {
        Session::Replay(session) if session.ply < session.replay.actions.len() => {
            format!("before {}", session.replay.actions[session.ply])
        }
        Session::Replay(_) => outcome_text(app.game().outcome()),
        Session::Match(session) => match app.game().outcome() {
            Outcome::Ongoing
                if session.controller(app.game().position().side_to_move()) == Controller::Cpu =>
            {
                session
                    .ai_status(now)
                    .unwrap_or_else(|| "CPU turn".to_owned())
            }
            Outcome::Ongoing => format!(
                "Turn: {}",
                short_player_label(app.game().position().side_to_move())
            ),
            outcome => outcome_text(outcome),
        },
    };
    let line = Line::from(vec![
        Span::styled(mode, Style::default().fg(Color::Cyan).bold()),
        Span::raw("  │  "),
        Span::styled(status, Style::default().fg(Color::Yellow)),
        Span::raw("  │  "),
        Span::raw(app.notice.as_str()),
    ]);
    frame.render_widget(
        Paragraph::new(line).block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" YokaiRust "),
        ),
        area,
    );
}

pub(super) fn render_main(frame: &mut Frame<'_>, app: &App, area: Rect, now: Instant) {
    if area.width >= 100 {
        let [board, history, analysis] = Layout::horizontal([
            Constraint::Length(34),
            Constraint::Length(24),
            Constraint::Min(42),
        ])
        .areas(area);
        render_board_column(frame, app, board, now);
        render_history(frame, app, history);
        render_analysis(frame, app, analysis);
    } else {
        let [board, sidebar] =
            Layout::horizontal([Constraint::Length(34), Constraint::Min(36)]).areas(area);
        let [history, analysis] =
            Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(sidebar);
        render_board_column(frame, app, board, now);
        render_history(frame, app, history);
        render_analysis(frame, app, analysis);
    }
}

pub(super) fn render_board_column(frame: &mut Frame<'_>, app: &App, area: Rect, now: Instant) {
    let [second_hand, board, first_hand] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(12),
        Constraint::Length(3),
    ])
    .areas(area);
    render_hand(frame, app, Player::Second, second_hand);
    render_board(frame, app, board, now);
    render_hand(frame, app, Player::First, first_hand);
}

pub(super) fn render_hand(frame: &mut Frame<'_>, app: &App, player: Player, area: Rect) {
    let position = app.game().position();
    let spans = HandPiece::ALL
        .iter()
        .enumerate()
        .flat_map(|(index, &piece)| {
            let selected = app.selected_hand_piece(player, piece, index);
            let style = if selected {
                Style::default().fg(Color::Black).bg(Color::Yellow).bold()
            } else if position.hand_count(player, piece) > 0 {
                Style::default().fg(player_color(player)).bold()
            } else {
                Style::default().fg(Color::DarkGray)
            };
            [
                Span::styled(
                    format!(
                        "{} {}×{}",
                        index + 1,
                        hand_piece_code(piece),
                        position.hand_count(player, piece)
                    ),
                    style,
                ),
                Span::raw("  "),
            ]
        })
        .collect::<Vec<_>>();
    let title = format!(" {} ", player_label(player));
    frame.render_widget(
        Paragraph::new(Line::from(spans)).block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(player_color(player)))
                .title(title),
        ),
        area,
    );
}

pub(super) fn render_board(frame: &mut Frame<'_>, app: &App, area: Rect, now: Instant) {
    let rows = Layout::vertical([Constraint::Length(3); 4]).split(area);
    for (row, row_area) in rows.iter().enumerate() {
        let columns = Layout::horizontal([Constraint::Ratio(1, 3); 3]).split(*row_area);
        for (column, &cell_area) in columns.iter().enumerate() {
            let row = u8::try_from(row).expect("board row fits in u8");
            let column = u8::try_from(column).expect("board column fits in u8");
            let square = Square::new(row, column).expect("rendered board square");
            render_square(frame, app, square, cell_area, now);
        }
    }
}

pub(super) fn render_square(
    frame: &mut Frame<'_>,
    app: &App,
    square: Square,
    area: Rect,
    now: Instant,
) {
    let piece = app.game().position().piece_at(square);
    let (border_color, border_type) = match square_highlight(app, square, now) {
        SquareHighlight::SelectedSource => (Color::Yellow, BorderType::Double),
        SquareHighlight::FocusedDestination | SquareHighlight::Cursor => {
            (Color::Cyan, BorderType::Double)
        }
        SquareHighlight::LegalDestination => (Color::Green, BorderType::Thick),
        SquareHighlight::LastDestination => (Color::Yellow, BorderType::Thick),
        SquareHighlight::LastSource => (Color::DarkGray, BorderType::Thick),
        SquareHighlight::None => (Color::Gray, BorderType::Plain),
    };
    let content = piece.map_or_else(
        || "·".to_owned(),
        |piece| format!("{} {}", owner_arrow(piece.owner), piece_code(piece)),
    );
    let piece_style = piece.map_or_else(
        || Style::default().fg(Color::DarkGray),
        |piece| Style::default().fg(player_color(piece.owner)).bold(),
    );
    frame.render_widget(
        Paragraph::new(Span::styled(content, piece_style))
            .alignment(Alignment::Center)
            .block(
                Block::bordered()
                    .border_type(border_type)
                    .border_style(Style::default().fg(border_color))
                    .title(Span::styled(
                        square.to_string(),
                        Style::default().fg(Color::DarkGray),
                    )),
            ),
        area,
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SquareHighlight {
    SelectedSource,
    FocusedDestination,
    Cursor,
    LegalDestination,
    LastDestination,
    LastSource,
    None,
}

pub(super) fn square_highlight(app: &App, square: Square, now: Instant) -> SquareHighlight {
    if app.selected_source() == Some(square) {
        return SquareHighlight::SelectedSource;
    }
    if app.focused_ai_destination(now) == Some(square) {
        return SquareHighlight::FocusedDestination;
    }
    let cursor = app.board_cursor(square);
    if cursor && app.selection.is_some() {
        return SquareHighlight::Cursor;
    }
    if app.legal_destination(square) {
        return SquareHighlight::LegalDestination;
    }
    let last_action = app.last_action();
    if last_action.is_some_and(|action| action.destination() == square) {
        return SquareHighlight::LastDestination;
    }
    if last_action
        .is_some_and(|action| matches!(action, Action::Move { from, .. } if from == square))
    {
        return SquareHighlight::LastSource;
    }
    if cursor {
        return SquareHighlight::Cursor;
    }
    SquareHighlight::None
}

pub(super) fn render_history(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let (actions, applied) = match &app.session {
        Session::Match(session) => (session.game.actions(), session.game.actions().len()),
        Session::Replay(session) => (session.replay.actions.as_slice(), session.ply),
    };
    let available_rows = usize::from(area.height.saturating_sub(3)).max(1);
    let (start, end) = visible_window(actions.len(), applied, available_rows);
    let rows = actions[start..end]
        .iter()
        .enumerate()
        .map(|(offset, action)| {
            let index = start + offset;
            let style = match index.cmp(&applied) {
                Ordering::Equal => Style::default().fg(Color::Yellow).bold(),
                Ordering::Less => Style::default().fg(Color::White),
                Ordering::Greater => Style::default().fg(Color::DarkGray),
            };
            Row::new([
                Cell::from(format!("{}.", index + 1)),
                Cell::from(action.to_string()),
            ])
            .style(style)
        });
    let table = Table::new(rows, [Constraint::Length(4), Constraint::Min(8)])
        .header(
            Row::new(["#", "Move"])
                .style(Style::default().fg(Color::Cyan).bold())
                .bottom_margin(0),
        )
        .block(
            Block::bordered()
                .border_type(BorderType::Rounded)
                .title(" Move history "),
        );
    frame.render_widget(table, area);
}

pub(super) fn render_analysis(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let view = analysis_view(app);
    let title = view.perspective.map_or_else(
        || " Predictions ".to_owned(),
        |(player, controller)| {
            format!(
                " Predictions · {} · {} ",
                short_player_label(player),
                controller.label()
            )
        },
    );
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let [summary_area, table_area] =
        Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner);
    // The root value is `P(win) - P(loss)` from the analyzed side's view.
    let summary = view.root_value.map_or_else(
        || "Root value —".to_owned(),
        |value| format!("Root value {value:+.3}"),
    );
    frame.render_widget(
        Paragraph::new(summary).style(Style::default().fg(Color::DarkGray)),
        summary_area,
    );

    let Some(actions) = view.actions else {
        frame.render_widget(
            Paragraph::new(view.note)
                .style(Style::default().fg(Color::DarkGray).italic())
                .wrap(Wrap { trim: true }),
            table_area,
        );
        return;
    };
    let max_rows = usize::from(table_area.height.saturating_sub(1));
    let played_action = match &app.session {
        Session::Replay(session) => session.replay.actions.get(session.ply).copied(),
        Session::Match(_) => None,
    };
    let offset = app
        .analysis_offset
        .min(actions.len().saturating_sub(max_rows.max(1)));
    let rows = actions.iter().skip(offset).take(max_rows).map(|entry| {
        let style = if Some(entry.action) == played_action {
            Style::default().fg(Color::Yellow).bold()
        } else {
            Style::default()
        };
        Row::new([
            Cell::from(entry.action.to_string()),
            Cell::from(format!("{:.2}", entry.prior)),
            Cell::from(entry.visits.to_string()),
            Cell::from(format!("{:.2}", entry.visit_probability)),
            Cell::from(format!("{:+.2}", entry.q_value)),
        ])
        .style(style)
    });
    let table = Table::new(
        rows,
        [
            Constraint::Min(8),
            Constraint::Length(5),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(5),
        ],
    )
    .header(
        Row::new(["Move", "Prior", "Visits", "Policy", "Q"])
            .style(Style::default().fg(Color::Cyan).bold()),
    );
    frame.render_widget(table, table_area);
}

pub(super) struct AnalysisView<'a> {
    pub(super) perspective: Option<(Player, Controller)>,
    pub(super) root_value: Option<f32>,
    pub(super) actions: Option<&'a [ActionAnalysis]>,
    pub(super) note: &'static str,
}

pub(super) fn analysis_view(app: &App) -> AnalysisView<'_> {
    match &app.session {
        Session::Match(session) => session.predictions.as_ref().map_or(
            AnalysisView {
                perspective: Some((
                    session.game.position().side_to_move(),
                    session.controller(session.game.position().side_to_move()),
                )),
                root_value: None,
                actions: None,
                note: match session.ai.as_ref().map(|ai| &ai.state) {
                    None => "No model analysis runs in human/human mode.",
                    Some(AiState::Loading) => "Loading the latest champion…",
                    Some(AiState::Thinking {
                        purpose: SearchPurpose::HumanPrediction,
                        ..
                    }) => "Champion is analyzing the human position…",
                    Some(AiState::Thinking {
                        purpose: SearchPurpose::CpuMove,
                        ..
                    }) => "Champion is choosing the CPU move…",
                    Some(AiState::Failed(_)) => "Champion analysis is unavailable.",
                    Some(AiState::Idle | AiState::WaitingToPlay { .. }) => {
                        "Champion analysis is starting…"
                    }
                },
            },
            |predictions| AnalysisView {
                perspective: Some((predictions.player, session.controller(predictions.player))),
                root_value: predictions.root_value,
                actions: Some(&predictions.actions),
                note: "",
            },
        ),
        Session::Replay(session) => AnalysisView {
            perspective: None,
            root_value: None,
            actions: session.current_analyses(),
            note: if session.ply == session.replay.actions.len() {
                "End of replay."
            } else if session.replay.analyses.is_some() {
                "No analysis is stored for this position."
            } else {
                "This replay contains no prediction data."
            },
        },
    }
}

pub(super) fn render_footer(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let help = match app.session {
        Session::Match(_) => "Q quit · N new · Arrows move · Enter play · Tab hand · Esc cancel",
        Session::Replay(_) => "Q quit · ←/→ moves · ↑/↓ analysis · Home/End or g/G bounds",
    };
    frame.render_widget(
        Paragraph::new(help)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Gray))
            .block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .title(" Controls "),
            ),
        area,
    );
}

pub(super) fn visible_window(total: usize, applied: usize, capacity: usize) -> (usize, usize) {
    if total <= capacity {
        return (0, total);
    }
    let anchor = applied.min(total.saturating_sub(1));
    let start = anchor
        .saturating_sub(capacity / 2)
        .min(total.saturating_sub(capacity));
    (start, start + capacity)
}

pub(super) const fn player_color(player: Player) -> Color {
    match player {
        Player::First => Color::Cyan,
        Player::Second => Color::Magenta,
    }
}

pub(super) const fn player_label(player: Player) -> &'static str {
    match player {
        Player::First => "Player 1 · bottom",
        Player::Second => "Player 2 · top",
    }
}

pub(super) const fn short_player_label(player: Player) -> &'static str {
    match player {
        Player::First => "P1 bottom",
        Player::Second => "P2 top",
    }
}

pub(super) const fn owner_arrow(player: Player) -> &'static str {
    match player {
        Player::First => "▲",
        Player::Second => "▼",
    }
}

pub(super) const fn hand_piece_code(piece: HandPiece) -> &'static str {
    match piece {
        HandPiece::Tanuki => "TA",
        HandPiece::Kitsune => "KI",
        HandPiece::Kodama => "KD",
    }
}

pub(super) const fn piece_code(piece: Piece) -> &'static str {
    match piece.kind {
        PieceKind::Koropokkuru => "KO",
        PieceKind::Tanuki => "TA",
        PieceKind::Kitsune => "KI",
        PieceKind::Kodama => "KD",
        PieceKind::KodamaSamurai => "SA",
    }
}

pub(super) fn outcome_text(outcome: Outcome) -> String {
    match outcome {
        Outcome::Ongoing => "Game in progress".to_owned(),
        Outcome::Win { player, reason } => {
            format!(
                "{} wins ({})",
                player_label(player),
                win_reason_text(reason)
            )
        }
        Outcome::Draw { reason } => format!("Draw ({})", draw_reason_text(reason)),
    }
}

pub(super) const fn win_reason_text(reason: WinReason) -> &'static str {
    match reason {
        WinReason::KoropokkuruCaptured => "Koropokkuru captured",
        WinReason::KoropokkuruReachedGoal => "Koropokkuru reached the goal",
        WinReason::OpponentHasNoLegalAction => "opponent has no legal move",
    }
}

pub(super) const fn draw_reason_text(reason: DrawReason) -> &'static str {
    match reason {
        DrawReason::ThreefoldRepetition => "threefold repetition",
    }
}

pub(super) fn transition_message(transition: yokai::Transition) -> String {
    let mut message = format!(
        "{} plays {}",
        player_label(transition.player),
        transition.action
    );
    if let Some(captured) = transition.captured {
        let _ = write!(message, " · captures {}", PieceKindLabel(captured));
    }
    if transition.promoted {
        message.push_str(" · promotes to samurai");
    }
    if transition.outcome.is_terminal() {
        let _ = write!(message, " · {}", outcome_text(transition.outcome));
    }
    message
}

pub(super) struct PieceKindLabel(PieceKind);

impl fmt::Display for PieceKindLabel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.0 {
            PieceKind::Koropokkuru => "Koropokkuru",
            PieceKind::Tanuki => "Tanuki",
            PieceKind::Kitsune => "Kitsune",
            PieceKind::Kodama => "Kodama",
            PieceKind::KodamaSamurai => "Kodama samurai",
        })
    }
}
