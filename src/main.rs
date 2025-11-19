use clap::Parser;
use glob::glob;
use indicatif::{ProgressBar, ProgressStyle};
use mdx_music_separator::{
    inference::{MdxNet, save_audio_wav},
    model::ConvTdfNetTrimModelConfig,
};
use std::error::Error;
use std::fs::{self, File};
use std::io::copy;
use std::path::{Path, PathBuf};
use std::time::Instant;

// Define the backend.
type Backend = burn_ndarray::NdArray<f32>;

const MDX_B_MODEL_URL: &str = "https://huggingface.co/zfturbo/MVSEP-MDX23-music-separation-models/resolve/main/MDX-B-Inst-Voc-Models.zip";

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

    /// Directory where models are stored or will be downloaded
    #[arg(long, default_value = "./models")]
    model_folder: PathBuf,

    /// FFT window size
    #[arg(long, default_value_t = 7680)] // Default for MDX-B
    n_fft: usize,

    /// Frequency dimension
    #[arg(long, default_value_t = 4096)] // Default for MDX-B
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

/// Downloads and unzips the MDX-B models if they are not already present.
fn download_models_if_needed(model_dir: &Path) -> Result<(), Box<dyn Error>> {
    // Check if models already exist
    let model_pattern = model_dir.join("*.pt");
    if let Ok(paths) = glob(model_pattern.to_str().unwrap()) {
        if paths.count() > 0 {
            println!("Models found in {:?}. Skipping download.", model_dir);
            return Ok(());
        }
    }

    println!("Models not found. Downloading from Hugging Face...");
    fs::create_dir_all(model_dir)?;

    let zip_path = model_dir.join("MDX-B-Inst-Voc-Models.zip");

    // Download with progress bar
    let client = reqwest::blocking::Client::builder()
        .timeout(None) // Allow long downloads
        .build()?;
    let mut response = client.get(MDX_B_MODEL_URL).send()?.error_for_status()?;
    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
        .progress_chars("#>-"));
    pb.set_message("Downloading MDX-B-Inst-Voc-Models.zip");

    let mut dest_file = File::create(&zip_path)?;
    let mut download_progress = pb.wrap_read(&mut response);

    copy(&mut download_progress, &mut dest_file)?;
    pb.finish_with_message("Download complete.");

    // Unzip
    println!("Unzipping models...");
    let file = File::open(&zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => model_dir.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            copy(&mut file, &mut outfile)?;
        }
    }

    // Clean up zip file
    fs::remove_file(&zip_path)?;
    println!(
        "Models successfully downloaded and unzipped to {:?}.",
        model_dir
    );

    Ok(())
}

fn run_app() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let device = Default::default();

    // --- 1. Download models if needed ---
    download_models_if_needed(&cli.model_folder)?;

    // --- 2. Define Model and Inference Config ---
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

    // --- 3. Load Models ---
    let mdx_net = MdxNet::<Backend>::new(cli.model_folder.to_str().unwrap(), config, &device)?;
    println!(
        "\nSuccessfully initialized MdxNet with {} model(s).",
        mdx_net.models.len()
    );

    // --- 4. Create Output Directory ---
    fs::create_dir_all(&cli.output_folder)?;

    // --- 5. Run Inference Loop ---
    for input_path in &cli.input_audio {
        let input_path_str = input_path.to_str().unwrap_or("invalid_path");
        println!("\nProcessing file: {}", input_path_str);

        let file_name = input_path.file_name().ok_or("Invalid input file path")?;
        let output_path = cli.output_folder.join(file_name);
        let output_path_str = output_path.to_str().unwrap();

        let start_time = Instant::now();

        let (processed_wave, input_spec) = mdx_net.run_inference(input_path_str)?;

        let duration = start_time.elapsed();
        println!("Inference complete in: {:.2?}", duration);

        // --- 6. Save Output ---
        println!("Saving output to: {}", output_path_str);
        save_audio_wav(output_path_str, processed_wave, input_spec)?;
    }

    println!("\nAll files processed.");
    Ok(())
}
