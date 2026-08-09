//! Multi-producer batching service for GPU-friendly inference.

use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
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
    stats: Arc<SharedInferenceStats>,
}

impl Evaluator for InferenceClient {
    fn evaluate_batch(
        &mut self,
        requests: &[EvaluationRequest],
    ) -> Result<Vec<Evaluation>, EvaluationError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let (response, receiver) = bounded(1);
        self.sender
            .send(Message::Evaluate {
                requests: requests.to_vec(),
                response,
            })
            .map_err(|_| EvaluationError::Backend("inference service stopped".to_owned()))?;
        let result = receiver
            .recv()
            .map_err(|_| EvaluationError::Backend("inference worker stopped".to_owned()))?;
        self.stats.jobs.fetch_add(1, Ordering::Relaxed);
        self.stats
            .client_wait_nanos
            .fetch_add(duration_nanos(started.elapsed()), Ordering::Relaxed);
        result
    }
}

/// Snapshot of batching and backend utilization since service startup.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceStats {
    pub jobs: u64,
    pub backend_batches: u64,
    pub positions: u64,
    pub maximum_batch_size: usize,
    pub backend_time: Duration,
    pub cumulative_client_wait: Duration,
}

impl InferenceStats {
    #[must_use]
    pub fn average_batch_size(self) -> f64 {
        if self.backend_batches == 0 {
            return 0.0;
        }
        count_as_f64(self.positions) / count_as_f64(self.backend_batches)
    }

    #[must_use]
    pub fn positions_per_backend_second(self) -> f64 {
        let seconds = self.backend_time.as_secs_f64();
        if seconds <= f64::EPSILON {
            return 0.0;
        }
        count_as_f64(self.positions) / seconds
    }

    #[must_use]
    pub fn average_client_wait(self) -> Duration {
        if self.jobs == 0 {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(self.cumulative_client_wait.as_secs_f64() / count_as_f64(self.jobs))
    }
}

#[derive(Default)]
struct SharedInferenceStats {
    jobs: AtomicU64,
    backend_batches: AtomicU64,
    positions: AtomicU64,
    maximum_batch_size: AtomicUsize,
    backend_nanos: AtomicU64,
    client_wait_nanos: AtomicU64,
}

impl SharedInferenceStats {
    fn snapshot(&self) -> InferenceStats {
        InferenceStats {
            jobs: self.jobs.load(Ordering::Relaxed),
            backend_batches: self.backend_batches.load(Ordering::Relaxed),
            positions: self.positions.load(Ordering::Relaxed),
            maximum_batch_size: self.maximum_batch_size.load(Ordering::Relaxed),
            backend_time: Duration::from_nanos(self.backend_nanos.load(Ordering::Relaxed)),
            cumulative_client_wait: Duration::from_nanos(
                self.client_wait_nanos.load(Ordering::Relaxed),
            ),
        }
    }
}

#[derive(Debug, Error)]
pub enum InferenceServiceError {
    #[error("maximum inference batch size must be greater than zero")]
    EmptyBatch,
    #[error("minimum inference batch must be in 1..=maximum batch size")]
    InvalidMinimumBatch,
    #[error("failed to start inference thread: {0}")]
    Spawn(#[from] std::io::Error),
}

/// Owns the inference worker and joins it on drop.
pub struct InferenceService {
    sender: Sender<Message>,
    worker: Option<thread::JoinHandle<()>>,
    stats: Arc<SharedInferenceStats>,
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
        Self::start_with_batching(evaluator, max_batch_size, max_batch_size, max_wait)
    }

    /// Starts a worker that executes as soon as `minimum_batch_size` positions
    /// are queued, or after `max_wait` measured from the first request.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceServiceError`] for invalid batch bounds or when the
    /// operating system cannot create the worker thread.
    pub fn start_with_batching<E: Evaluator + Send + 'static>(
        evaluator: E,
        minimum_batch_size: usize,
        max_batch_size: usize,
        max_wait: Duration,
    ) -> Result<Self, InferenceServiceError> {
        if max_batch_size == 0 {
            return Err(InferenceServiceError::EmptyBatch);
        }
        if minimum_batch_size == 0 || minimum_batch_size > max_batch_size {
            return Err(InferenceServiceError::InvalidMinimumBatch);
        }
        let (sender, receiver) = unbounded();
        let stats = Arc::new(SharedInferenceStats::default());
        let worker_stats = stats.clone();
        let worker = thread::Builder::new()
            .name("yokai-inference".to_owned())
            .spawn(move || {
                worker_loop(
                    evaluator,
                    &receiver,
                    minimum_batch_size,
                    max_batch_size,
                    max_wait,
                    &worker_stats,
                );
            })?;
        Ok(Self {
            sender,
            worker: Some(worker),
            stats,
        })
    }

    #[must_use]
    pub fn client(&self) -> InferenceClient {
        InferenceClient {
            sender: self.sender.clone(),
            stats: self.stats.clone(),
        }
    }

    #[must_use]
    pub fn stats(&self) -> InferenceStats {
        self.stats.snapshot()
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
    minimum_batch_size: usize,
    max_batch_size: usize,
    max_wait: Duration,
    stats: &SharedInferenceStats,
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
        let deadline = Instant::now() + max_wait;

        while position_count < max_batch_size {
            let message = if position_count >= minimum_batch_size {
                receiver.try_recv().ok()
            } else {
                let remaining = deadline.saturating_duration_since(Instant::now());
                receiver.recv_timeout(remaining).ok()
            };
            match message {
                Some(Message::Evaluate { requests, response }) => {
                    position_count += requests.len();
                    jobs.push((requests, response));
                }
                Some(Message::Shutdown) => {
                    stop_after_batch = true;
                    break;
                }
                None => break,
            }
        }

        evaluate_jobs(&mut evaluator, jobs, max_batch_size, stats);
    }
}

fn evaluate_jobs<E: Evaluator>(
    evaluator: &mut E,
    jobs: Vec<InferenceJob>,
    max_batch_size: usize,
    stats: &SharedInferenceStats,
) {
    let requests = jobs
        .iter()
        .flat_map(|(requests, _)| requests.iter().copied())
        .collect::<Vec<_>>();
    let evaluations = requests.chunks(max_batch_size).try_fold(
        Vec::with_capacity(requests.len()),
        |mut all, batch| {
            let started = Instant::now();
            let evaluated = evaluator.evaluate_batch(batch)?;
            stats.backend_batches.fetch_add(1, Ordering::Relaxed);
            stats
                .positions
                .fetch_add(count_as_u64(batch.len()), Ordering::Relaxed);
            stats
                .maximum_batch_size
                .fetch_max(batch.len(), Ordering::Relaxed);
            stats
                .backend_nanos
                .fetch_add(duration_nanos(started.elapsed()), Ordering::Relaxed);
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

fn duration_nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn count_as_u64(count: usize) -> u64 {
    u64::try_from(count).unwrap_or(u64::MAX)
}

#[allow(clippy::cast_precision_loss)]
fn count_as_f64(count: u64) -> f64 {
    count as f64
}
