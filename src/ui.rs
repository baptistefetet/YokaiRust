//! Ratatui terminal interface: entry points and the event loop.
//!
//! The rules remain in the library. This module tree only coordinates a live
//! match or a replay cursor:
//!
//! - [`session`] owns the match/replay state machines and the CPU worker
//!   states;
//! - [`app`] owns input handling and the shared application state;
//! - [`render`] draws everything and hosts the text formatting helpers;
//! - [`ai`] runs champion loading and MCTS on a background thread.

mod ai;
mod app;
mod render;
mod session;
#[cfg(test)]
mod tests;

use std::{error::Error, io, str::FromStr, time::Duration};

use ratatui::crossterm::event::{self, Event, KeyEventKind};
use yokai::{Replay, TrainingConfig};

use self::ai::AiWorker;
use self::app::App;
use self::render::render;

const EVENT_POLL_INTERVAL: Duration = Duration::from_millis(100);
pub(crate) const AI_SOURCE_FOCUS_DURATION: Duration = Duration::from_millis(800);
pub(crate) const AI_MOVE_DELAY: Duration = Duration::from_millis(1_500);
pub(crate) const ACTIVE_TRAINING_CONFIG: &str = "config/training.toml";
pub(crate) const MINIMUM_WIDTH: u16 = 70;
pub(crate) const MINIMUM_HEIGHT: u16 = 24;

/// Match type accepted by the `play` command.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum PlayMode {
    /// Two people alternate on the same terminal.
    #[default]
    HumanVsHuman,
    /// The human is First at the bottom; the latest champion is Second.
    HumanVsCpu,
}

impl FromStr for PlayMode {
    type Err = io::Error;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input {
            "human-vs-human" | "hvh" => Ok(Self::HumanVsHuman),
            "human-vs-cpu" | "hvc" => Ok(Self::HumanVsCpu),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unknown play mode `{input}`"),
            )),
        }
    }
}

/// Starts the fullscreen interface for a local match.
pub(crate) fn play(mode: PlayMode) -> Result<(), Box<dyn Error>> {
    let app = match mode {
        PlayMode::HumanVsHuman => App::for_human_match(),
        PlayMode::HumanVsCpu => {
            let config = TrainingConfig::load(ACTIVE_TRAINING_CONFIG)?;
            App::for_cpu_match(AiWorker::spawn(config)?)
        }
    };
    run(app)?;
    Ok(())
}

/// Starts the fullscreen replay viewer for an already validated replay.
pub(crate) fn watch(replay: Replay) -> io::Result<()> {
    let app = App::for_replay(replay).map_err(io::Error::other)?;
    run(app)
}

fn run(mut app: App) -> io::Result<()> {
    ratatui::run(|terminal| {
        while !app.should_quit {
            app.tick();
            terminal.draw(|frame| render(frame, &app))?;
            if event::poll(EVENT_POLL_INTERVAL)?
                && let Event::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                app.handle_key(key);
            }
        }
        Ok(())
    })
}
