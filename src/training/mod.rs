//! `AlphaZero` self-play data and optimization pipeline.
//!
//! The training loop (self-play → replay buffer → optimization → arena →
//! guarded promotion) and its vocabulary are explained step by step in
//! `docs/alphazero-guide.md`; `docs/reading-guide.md` suggests a reading
//! order through these modules for developers coming from C++.

pub mod arena;
pub mod config;
pub mod data;
pub mod diagnostics;
pub mod pipeline;
pub mod self_play;
pub mod trainer;
