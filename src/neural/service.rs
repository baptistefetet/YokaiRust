//! Multi-producer batching service for GPU-friendly inference.

use std::{thread, time::Duration};

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded, unbounded};
use thiserror::Error;

use crate::{Evaluation, EvaluationError, EvaluationRequest, Evaluator};

enum Message {
    Evaluate {
        requests: Vec<EvaluationRequest>,
        response: Sender<Result<Vec<Evaluation>, EvaluationError>>,
    },
    Shutdown,
}

type ResponseSender = Sender<Result<Vec<Evaluation>, EvaluationError>>;
type InferenceJob = (Vec<EvaluationRequest>, ResponseSender);

/// Cloneable evaluator handle used independently by self-play games.
#[derive(Clone)]
pub struct InferenceClient {
    sender: Sender<Message>,
}

impl Evaluator for InferenceClient {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let (response, receiver) = bounded(1);
        self.sender
            .send(Message::Evaluate {
                requests: requests.to_vec(),
                response,
            })
            .map_err(|_| EvaluationError::Backend("inference service stopped".to_owned()))?;
        receiver
            .recv()
            .map_err(|_| EvaluationError::Backend("inference worker stopped".to_owned()))?
    }
}

#[derive(Debug, Error)]
pub enum InferenceServiceError {
    #[error("maximum inference batch size must be greater than zero")]
    EmptyBatch,
    #[error("failed to start inference thread: {0}")]
    Spawn(#[from] std::io::Error),
}

/// Owns the inference worker and joins it on drop.
pub struct InferenceService {
    sender: Sender<Message>,
    worker: Option<thread::JoinHandle<()>>,
}

impl InferenceService {
    /// Starts a named worker around any evaluator backend.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceServiceError`] for an empty maximum batch or when the
    /// operating system cannot create the worker thread.
    pub fn start<E: Evaluator + Send + 'static>(
        evaluator: E,
        max_batch_size: usize,
        max_wait: Duration,
    ) -> Result<Self, InferenceServiceError> {
        if max_batch_size == 0 {
            return Err(InferenceServiceError::EmptyBatch);
        }
        let (sender, receiver) = unbounded();
        let worker = thread::Builder::new()
            .name("yokai-inference".to_owned())
            .spawn(move || worker_loop(evaluator, &receiver, max_batch_size, max_wait))?;
        Ok(Self {
            sender,
            worker: Some(worker),
        })
    }

    #[must_use]
    pub fn client(&self) -> InferenceClient {
        InferenceClient {
            sender: self.sender.clone(),
        }
    }
}

impl Drop for InferenceService {
    fn drop(&mut self) {
        let _ = self.sender.send(Message::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn worker_loop<E: Evaluator>(
    mut evaluator: E,
    receiver: &Receiver<Message>,
    max_batch_size: usize,
    max_wait: Duration,
) {
    let mut stop_after_batch = false;
    while !stop_after_batch {
        let Ok(message) = receiver.recv() else {
            return;
        };
        let Message::Evaluate { requests, response } = message else {
            return;
        };
        let mut jobs = vec![(requests, response)];
        let mut position_count = jobs[0].0.len();

        while position_count < max_batch_size {
            match receiver.recv_timeout(max_wait) {
                Ok(Message::Evaluate { requests, response }) => {
                    position_count += requests.len();
                    jobs.push((requests, response));
                }
                Ok(Message::Shutdown) => {
                    stop_after_batch = true;
                    break;
                }
                Err(RecvTimeoutError::Timeout | RecvTimeoutError::Disconnected) => break,
            }
        }

        evaluate_jobs(&mut evaluator, jobs, max_batch_size);
    }
}

fn evaluate_jobs<E: Evaluator>(evaluator: &mut E, jobs: Vec<InferenceJob>, max_batch_size: usize) {
    let requests = jobs
        .iter()
        .flat_map(|(requests, _)| requests.iter().copied())
        .collect::<Vec<_>>();
    let evaluations = requests.chunks(max_batch_size).try_fold(
        Vec::with_capacity(requests.len()),
        |mut all, batch| {
            let evaluated = evaluator.evaluate_batch(batch)?;
            if evaluated.len() != batch.len() {
                return Err(EvaluationError::BatchSizeMismatch {
                    expected: batch.len(),
                    actual: evaluated.len(),
                });
            }
            all.extend(evaluated);
            Ok(all)
        },
    );
    match evaluations {
        Ok(evaluations) => {
            let mut offset = 0;
            for (requests, response) in jobs {
                let end = offset + requests.len();
                let _ = response.send(Ok(evaluations[offset..end].to_vec()));
                offset = end;
            }
        }
        Err(error) => {
            for (_, response) in jobs {
                let _ = response.send(Err(error.clone()));
            }
        }
    }
}
