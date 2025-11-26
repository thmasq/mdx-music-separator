mod audio;
mod inference;
mod utils;

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};

use crate::inference::MDXModel;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the ONNX model file
    #[arg(short, long)]
    model: PathBuf,

    /// Input audio files
    #[arg(short, long = "input-audio", required = true, num_args = 1.., value_name = "FILES")]
    input: Vec<PathBuf>,

    /// Output folder
    #[arg(short, long, default_value = "results")]
    output_dir: PathBuf,

    /// FFT size (usually 6144 for Kim_Vocal_1, 7680 for others)
    #[arg(long, default_value_t = 6144)]
    n_fft: usize,

    /// Hop length (usually 1024)
    #[arg(long, default_value_t = 1024)]
    hop_length: usize,

    /// Segment size (number of frames per inference pass, e.g., 256)
    #[arg(long, default_value_t = 256)]
    segment_size: usize,

    /// Batch size (number of segments to process in parallel)
    #[arg(long, default_value_t = 4)]
    batch_size: usize,

    /// Overlap amount (default is segment_size / 2 if not set, but here we explicitly set a default)
    #[arg(long, default_value_t = 128)]
    overlap: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create output directory if it doesn't exist
    if !args.output_dir.exists() {
        fs::create_dir_all(&args.output_dir).context("Failed to create output directory")?;
    }

    println!("Loading model from {:?}...", args.model);
    let mut model = MDXModel::load(
        &args.model,
        args.n_fft,
        args.hop_length,
        args.segment_size,
        args.batch_size,
        args.overlap,
    )
    .context("Failed to initialize MDX model")?;
    println!("Model loaded successfully.");

    for input_path in &args.input {
        process_file(&input_path, &args, &mut model)?;
    }

    Ok(())
}

fn process_file(path: &Path, args: &Args, model: &mut MDXModel) -> Result<()> {
    println!("Processing {:?}...", path);

    // 1. Load Audio
    let (audio, sample_rate) = audio::load_wav(path.to_str().unwrap())
        .with_context(|| format!("Failed to load audio file {:?}", path))?;

    // 3. Run Inference
    let result = model.demix(&audio)?;

    // 5. Save Result
    let filename = path.file_stem().unwrap().to_str().unwrap();
    let model_name = args.model.file_stem().unwrap().to_str().unwrap();
    let output_filename = format!("{}_{}.wav", filename, model_name);
    let output_path = args.output_dir.join(output_filename);

    audio::save_wav(output_path.to_str().unwrap(), result.view(), sample_rate)?;

    println!("Saved to {:?}", output_path);

    Ok(())
}
