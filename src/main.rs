use clap::Parser;
use mdx_music_separator::{
    inference::{MdxNet, save_audio_wav},
    model::ConvTdfNetTrimModelConfig,
};
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

// Define the backend.
type Backend = burn_ndarray::NdArray<f32>;

/// Command-line arguments for the MDX-B inference tool
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    /// List of input audio files to process
    #[arg(long, required = true, num_args = 1..)]
    input_audio: Vec<PathBuf>,

    /// Directory to save the output files
    #[arg(long, default_value = "./results")]
    output_folder: PathBuf,

    /// Directory where the converted PyTorch models are stored (.pt files)
    #[arg(long, default_value = "./models")]
    model_folder: PathBuf,

    /// FFT window size
    #[arg(long, default_value_t = 7680)] // Default for MDX-B
    n_fft: usize,

    /// Frequency dimension
    #[arg(long, default_value_t = 3072)] // Default for MDX-B
    dim_f: usize,

    /// Overlap margin percentage
    #[arg(long, default_value_t = 0.01)]
    margin: f64,
}

fn main() {
    run_app().unwrap_or_else(|err| {
        println!("\n--- APPLICATION ERROR ---");
        println!("{}", err);
    });
}

fn run_app() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let device = Default::default();

    // --- 1. Define Model and Inference Config ---
    let config = ConvTdfNetTrimModelConfig {
        l_layers: 11, // This is fixed for this arch
        n_fft: cli.n_fft,
        hop: 1024, // This is fixed
        dim_f: cli.dim_f,
        margin: cli.margin,
    };

    println!("Initializing MDXNet...");
    println!(
        "Model Config: n_fft={}, dim_f={}, margin={}",
        config.n_fft, config.dim_f, config.margin
    );

    // --- 2. Load Models ---
    // Models are sourced from the user provided directory.
    // The inference module will look for *.pt files in this folder.
    let mdx_net = MdxNet::<Backend>::new(cli.model_folder.to_str().unwrap(), config, &device)?;
    println!(
        "\nSuccessfully initialized MdxNet with {} model(s) from {:?}.",
        mdx_net.models.len(),
        cli.model_folder
    );

    // --- 3. Create Output Directory ---
    fs::create_dir_all(&cli.output_folder)?;

    // --- 4. Run Inference Loop ---
    for input_path in &cli.input_audio {
        let input_path_str = input_path.to_str().unwrap_or("invalid_path");
        println!("\nProcessing file: {}", input_path_str);

        let file_name = input_path.file_name().ok_or("Invalid input file path")?;
        let output_path = cli.output_folder.join(file_name);
        let output_path_str = output_path.to_str().unwrap();

        let start_time = Instant::now();

        match mdx_net.run_inference(input_path_str) {
            Ok((processed_wave, input_spec)) => {
                let duration = start_time.elapsed();
                println!("Inference complete in: {:.2?}", duration);

                // --- 5. Save Output ---
                println!("Saving output to: {}", output_path_str);
                if let Err(e) = save_audio_wav(output_path_str, processed_wave, input_spec) {
                    eprintln!("Error saving file {}: {}", output_path_str, e);
                }
            }
            Err(e) => {
                eprintln!("Failed to process {}: {}", input_path_str, e);
            }
        }
    }

    println!("\nAll files processed.");
    Ok(())
}
