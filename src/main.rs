mod audio;
mod inference;
mod utils;

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array2;
use std::fs;
use std::path::{Path, PathBuf}; // You might need ndarray-stats or implement simple stats manually. 
// For simplicity, I will implement manual mean/std to avoid extra deps if you prefer,
// but ndarray-stats is recommended. I'll use manual here to keep it "pure".

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
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Create output directory if it doesn't exist
    if !args.output_dir.exists() {
        fs::create_dir_all(&args.output_dir).context("Failed to create output directory")?;
    }

    println!("Loading model from {:?}...", args.model);
    let mut model = MDXModel::load(&args.model, args.n_fft, args.hop_length, args.segment_size)
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
    let (mut audio, sample_rate) = audio::load_wav(path.to_str().unwrap())
        .with_context(|| format!("Failed to load audio file {:?}", path))?;

    // 2. Normalize Audio
    // Python logic: mix = (mix - mix.mean()) / mix.std()
    let (mean, std) = calculate_stats(&audio);

    // Avoid division by zero
    let std = if std == 0.0 { 1.0 } else { std };

    // Apply normalization
    audio.mapv_inplace(|x| (x - mean) / std);

    // 3. Run Inference
    let mut result = model.demix(&audio)?;

    // 4. Denormalize Audio
    // Python logic: res = res * std + mean
    result.mapv_inplace(|x| (x * std) + mean);

    // 5. Save Result
    let filename = path.file_stem().unwrap().to_str().unwrap();
    let model_name = args.model.file_stem().unwrap().to_str().unwrap();
    let output_filename = format!("{}_{}.wav", filename, model_name);
    let output_path = args.output_dir.join(output_filename);

    audio::save_wav(output_path.to_str().unwrap(), result.view(), sample_rate)?;

    println!("Saved to {:?}", output_path);

    Ok(())
}

/// Helper to calculate mean and std deviation of an ndarray
fn calculate_stats(arr: &Array2<f32>) -> (f32, f32) {
    let len = arr.len() as f32;
    if len == 0.0 {
        return (0.0, 1.0);
    }

    let mean = arr.sum() / len;

    let variance = arr.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / len;

    (mean, variance.sqrt())
}
