//! Converts the accepted Safetensors champion to Burn's browser byte format.

use std::{env, error::Error, fs, path::PathBuf, process::ExitCode};

use burn::{
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use yokai::{CpuBackend, TrainingConfig, load_champion};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    let config_path = arguments
        .next()
        .unwrap_or_else(|| "config/training.toml".to_owned());
    let output = arguments
        .next()
        .map_or_else(|| PathBuf::from("web/generated/model"), PathBuf::from);
    if let Some(extra) = arguments.next() {
        return Err(format!("unexpected argument `{extra}`").into());
    }

    let config = TrainingConfig::load(config_path)?;
    let device = burn::backend::flex::FlexDevice;
    let (model, metadata) = load_champion::<CpuBackend>(&config.paths.models, &device)?;
    let bytes =
        BinBytesRecorder::<FullPrecisionSettings>::default().record(model.into_record(), ())?;

    fs::create_dir_all(&output)?;
    fs::write(output.join("champion.bin"), bytes)?;
    fs::write(
        output.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    println!(
        "exported generation {} to {}",
        metadata.generation,
        output.display()
    );
    Ok(())
}
