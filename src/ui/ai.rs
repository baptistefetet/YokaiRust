//! Background champion loading and MCTS execution for the interactive UI.

use std::{io, thread, time::Instant};

use burn::prelude::Backend;
use crossbeam_channel::{Receiver, Sender, unbounded};
use yokai::{
    BackendKind, CpuBackend, Game, Mcts, MetalBackend, NetworkEvaluator, SearchConfig,
    SearchResult, TrainingConfig, load_champion,
};

pub(super) enum AiEvent {
    Ready {
        generation: u32,
        simulations: u32,
    },
    SearchReady {
        request_id: u64,
        result: Box<SearchResult>,
        search_started: Instant,
        search_finished: Instant,
    },
    Failed {
        request_id: Option<u64>,
        message: String,
    },
}

pub(super) enum AiCommand {
    Search { request_id: u64, game: Game },
    Advance { action: yokai::Action, game: Game },
    Reset,
    Shutdown,
}

pub(super) struct AiWorker {
    commands: Sender<AiCommand>,
    events: Receiver<AiEvent>,
    thread: Option<thread::JoinHandle<()>>,
}

impl AiWorker {
    pub(super) fn spawn(config: TrainingConfig) -> io::Result<Self> {
        let (commands, command_receiver) = unbounded();
        let (event_sender, events) = unbounded();
        let worker = thread::Builder::new()
            .name("yokai-ui-ai".to_owned())
            .spawn(move || worker_main(&config, &command_receiver, &event_sender))?;
        Ok(Self {
            commands,
            events,
            thread: Some(worker),
        })
    }

    pub(super) fn search(&self, request_id: u64, game: &Game) -> Result<(), String> {
        self.commands
            .send(AiCommand::Search {
                request_id,
                game: game.clone(),
            })
            .map_err(|_| "AI worker stopped".to_owned())
    }

    pub(super) fn advance(&self, action: yokai::Action, game: &Game) {
        let _ = self.commands.send(AiCommand::Advance {
            action,
            game: game.clone(),
        });
    }

    pub(super) fn reset(&self) {
        let _ = self.commands.send(AiCommand::Reset);
    }

    pub(super) fn try_event(&self) -> Option<AiEvent> {
        self.events.try_recv().ok()
    }

    #[cfg(test)]
    pub(super) fn stub() -> (Self, Sender<AiEvent>, Receiver<AiCommand>) {
        let (commands, command_receiver) = unbounded();
        let (event_sender, events) = unbounded();
        (
            Self {
                commands,
                events,
                thread: None,
            },
            event_sender,
            command_receiver,
        )
    }
}

impl Drop for AiWorker {
    fn drop(&mut self) {
        let _ = self.commands.send(AiCommand::Shutdown);
        if let Some(worker) = self.thread.take() {
            let _ = worker.join();
        }
    }
}

fn worker_main(config: &TrainingConfig, commands: &Receiver<AiCommand>, events: &Sender<AiEvent>) {
    let result = match config.backend {
        BackendKind::Cpu => {
            let device = burn::backend::flex::FlexDevice;
            CpuBackend::seed(&device, config.seed);
            run_with_backend::<CpuBackend>(config, device, commands, events)
        }
        BackendKind::Metal => {
            let device = burn::backend::wgpu::WgpuDevice::default();
            MetalBackend::seed(&device, config.seed);
            run_with_backend::<MetalBackend>(config, device, commands, events)
        }
    };
    if let Err(message) = result {
        let _ = events.send(AiEvent::Failed {
            request_id: None,
            message,
        });
    }
}

fn run_with_backend<B: Backend<FloatElem = f32>>(
    config: &TrainingConfig,
    device: B::Device,
    commands: &Receiver<AiCommand>,
    events: &Sender<AiEvent>,
) -> Result<(), String> {
    let (model, metadata) =
        load_champion::<B>(&config.paths.models, &device).map_err(|error| error.to_string())?;
    let search_config = SearchConfig {
        simulations: config.arena.simulations,
        evaluation_batch_size: config.arena.search_batch_size,
        ..SearchConfig::default()
    };
    let evaluator = NetworkEvaluator::new(model, device);
    let mut search =
        Mcts::new(evaluator, search_config, config.seed).map_err(|error| error.to_string())?;
    let _ = events.send(AiEvent::Ready {
        generation: metadata.generation,
        simulations: search_config.simulations,
    });

    while let Ok(command) = commands.recv() {
        match command {
            AiCommand::Search { request_id, game } => {
                let search_started = Instant::now();
                match search.search(&game, 0.0) {
                    Ok(result) => {
                        let _ = events.send(AiEvent::SearchReady {
                            request_id,
                            result: Box::new(result),
                            search_started,
                            search_finished: Instant::now(),
                        });
                    }
                    Err(error) => {
                        let _ = events.send(AiEvent::Failed {
                            request_id: Some(request_id),
                            message: error.to_string(),
                        });
                    }
                }
            }
            AiCommand::Advance { action, game } => {
                let _reused = search.advance_root(action, &game);
            }
            AiCommand::Reset => search.reset(),
            AiCommand::Shutdown => break,
        }
    }
    Ok(())
}
