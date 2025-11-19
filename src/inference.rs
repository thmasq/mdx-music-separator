use crate::{
    audio::reflect_pad_1d,
    model::{ConvTdfNetTrimModel, ConvTdfNetTrimModelConfig},
};
use burn::{
    prelude::*,
    record::{FullPrecisionSettings, Recorder},
    tensor::Tensor,
};
use burn_import::pytorch::PyTorchFileRecorder;
use glob::glob;
use std::error::Error;

/// Manages loading models and running the full inference pipeline.
/// This is an equivalent to the `MDXNet` class in Python.
#[derive(Debug)]
pub struct MdxNet<B: Backend> {
    pub models: Vec<ConvTdfNetTrimModel<B>>,
    device: B::Device,
}

impl<B: Backend> MdxNet<B> {
    /// Creates a new `MdxNet` and loads all models from a directory.
    pub fn new(
        model_dir: &str,
        config: ConvTdfNetTrimModelConfig,
        device: &B::Device,
    ) -> Result<Self, Box<dyn Error>> {
        println!("Loading models from: {}", model_dir);
        let mut models = Vec::new();

        let pattern = format!("{}{}*.pt", model_dir, std::path::MAIN_SEPARATOR);
        for entry in glob(&pattern)? {
            let path = entry?;
            let model_path_str = path.to_str().unwrap_or("invalid_path");
            println!("- Loading weights from: {}", model_path_str);

            let record = PyTorchFileRecorder::<FullPrecisionSettings>::default()
                .load(path.clone().into(), device)
                .map_err(|e| format!("Failed to load weights from {}: {:?}", model_path_str, e))?;

            let model = config.init(device).load_record(record);
            models.push(model);
        }

        if models.is_empty() {
            return Err(format!("No models found in directory: {}", model_dir).into());
        }

        println!("Loaded {} model(s) successfully.", models.len());
        Ok(Self {
            models,
            device: device.clone(),
        })
    }

    /// Runs inference on an audio file.
    pub fn run_inference(
        &self,
        audio_path: &str,
    ) -> Result<(Tensor<B, 2>, hound::WavSpec), Box<dyn Error>> {
        // 1. Load audio
        let (mix, spec) = load_audio_wav(audio_path, &self.device)?;
        println!(
            "Loaded audio: {} channels, {} samples, {} Hz",
            spec.channels,
            mix.dims()[1],
            spec.sample_rate
        );

        // 2. Run demix (ensembling)
        let processed_wave = self.demix(mix);

        // 3. Post-process (e.g., normalization - optional, Python code does this)
        // Let's add simple peak normalization
        let max_val = processed_wave.clone().abs().max().into_scalar().to_f64();
        let processed_wave = if max_val > 1.0 {
            processed_wave.div_scalar(max_val)
        } else {
            processed_wave
        };

        Ok((processed_wave, spec))
    }

    /// Performs ensembled demixing by averaging the output of all models.
    pub fn demix(&self, mix: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut all_outputs = Vec::new();
        for (i, model) in self.models.iter().enumerate() {
            println!("Running model {}/{}...", i + 1, self.models.len());
            let output = self.demix_base(model, mix.clone());
            all_outputs.push(output);
        }

        // Average the results
        let stacked = Tensor::stack(all_outputs, 0);
        let ensembled = stacked.mean_dim(0);
        ensembled
    }

    /// The core inference pipeline (padding, chunking, overlap-add).
    /// This is an equivalent to `demix_base` from the Python MDX_Model.
    fn demix_base(&self, model: &ConvTdfNetTrimModel<B>, mix: Tensor<B, 2>) -> Tensor<B, 2> {
        let [_n_channels, n_samples] = mix.dims();
        let chunk_size = model.chunk_size;
        let margin_size = model.margin_size;

        // 1. Calculate padding
        // This logic matches the Python implementation
        let n_iters = (n_samples + margin_size * 2 + chunk_size - 1) / chunk_size;
        let pad_len = n_iters * chunk_size - n_samples;

        // 2. Apply reflection padding
        // [2, n_samples] -> [2, margin + n_samples + pad_len + margin]
        let pad_left = margin_size;
        let pad_right = margin_size + pad_len;

        let ch1_vec = mix
            .clone()
            .slice([0..1])
            .reshape([-1])
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| format!("{:?}", e))
            .expect("Failed to get ch1 vec for padding");
        let ch2_vec = mix
            .slice([1..2])
            .reshape([-1])
            .into_data()
            .into_vec::<f32>()
            .map_err(|e| format!("{:?}", e))
            .expect("Failed to get ch2 vec for padding");

        // Pad the Vecs using our audio helper
        let ch1_padded = reflect_pad_1d(&ch1_vec, pad_left, pad_right);
        let ch2_padded = reflect_pad_1d(&ch2_vec, pad_left, pad_right);

        // Use length from padded vecs as authoritative padded length
        let total_padded_len = ch1_padded.len();

        // Reconstruct the padded tensor
        let ch1_tensor: Tensor<B, 1> =
            Tensor::<B, 1>::from_floats(ch1_padded.as_slice(), &self.device);
        let ch2_tensor: Tensor<B, 1> =
            Tensor::<B, 1>::from_floats(ch2_padded.as_slice(), &self.device);

        // <-- Important: annotate the const-generic dimension here so the compiler knows
        // the stacked tensor has rank 2 (channels x samples).
        let mix_padded: Tensor<B, 2> = Tensor::stack(vec![ch1_tensor, ch2_tensor], 0);

        // 3. Process in chunks (overlap-add)
        // make explicit the chunk vector element type so stack() can infer correctly
        let mut all_chunks: Vec<Tensor<B, 2>> = Vec::new();
        let window = model.overlap_window.clone();

        for i in 0..n_iters {
            let start = i * chunk_size;
            let end = start + chunk_size;

            // Slice chunk: [2, chunk_size]
            let chunk: Tensor<B, 2> = mix_padded.clone().slice([0..2, start..end]);

            // Apply window: [2, chunk_size]
            // window is [chunk_size] -> unsqueeze_dim(0) -> [1, chunk_size], but broadcast/mul works
            let chunk_windowed: Tensor<B, 2> = chunk * window.clone().unsqueeze_dim(0);

            // Run inference
            // Add batch dim: [1, 2, chunk_size]
            let spec = model.stft(chunk_windowed.unsqueeze());
            // Model forward: [1, 4, F, T]
            let processed_spec = model.forward(spec);
            // iSTFT: [1, 2, chunk_size]
            // Remove batch dim: [2, chunk_size]
            let processed_chunk: Tensor<B, 2> = model.istft(processed_spec).squeeze();

            // Apply window again: [2, chunk_size]
            let processed_chunk_windowed: Tensor<B, 2> =
                processed_chunk * window.clone().unsqueeze_dim(0);

            // Pad this chunk back to the total padded length for summation
            let pad_left_for_chunk = start;
            let pad_right_for_chunk = total_padded_len - end;
            let final_chunk: Tensor<B, 2> = processed_chunk_windowed
                .pad((0, 0, pad_left_for_chunk, pad_right_for_chunk), 0.0f32);

            all_chunks.push(final_chunk);
        }

        // 4. Sum all processed chunks
        // stack(all_chunks, 0) -> Tensor with one extra leading dim; sum_dim(0) -> back to [2, total_padded_len]
        let summed_chunks = Tensor::stack(all_chunks, 0).sum_dim(0);

        // 5. Slice off the margins
        let result = summed_chunks.slice([
            0..2,
            margin_size..(total_padded_len - pad_len - margin_size),
        ]);

        result
    }
}

/// Loads a WAV file into a Burn tensor.
/// Converts to stereo f32 tensor in range [-1.0, 1.0].
pub fn load_audio_wav<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<(Tensor<B, 2>, hound::WavSpec), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let n_channels = spec.channels as usize;

    // Read samples and convert to f32
    let samples_interleaved: Vec<f32> = reader
        .samples::<i32>() // Assuming 32-bit int, adjust if needed
        .map(|s| s.unwrap_or(0) as f32 / i32::MAX as f32)
        .collect();

    let n_samples = samples_interleaved.len() / n_channels;
    let mut data = vec![vec![0.0f32; n_samples]; n_channels];

    for (i, frame) in samples_interleaved.chunks_exact(n_channels).enumerate() {
        for (ch, sample) in frame.iter().enumerate() {
            data[ch][i] = *sample;
        }
    }

    // Convert to tensor
    let ch1_data = Tensor::<B, 1>::from_floats(data[0].as_slice(), device);

    let tensor = if n_channels == 1 {
        // Mono to Stereo
        println!("Warning: Input audio is mono, converting to stereo.");
        Tensor::stack(vec![ch1_data.clone(), ch1_data], 0)
    } else {
        // Get stereo
        let ch2_data = Tensor::<B, 1>::from_floats(data[1].as_slice(), device);
        Tensor::stack(vec![ch1_data, ch2_data], 0)
    };

    Ok((tensor, spec))
}

/// Saves a Burn tensor as a WAV file.
/// Assumes f32 tensor in range [-1.0, 1.0].
pub fn save_audio_wav<B: Backend>(
    path: &str,
    audio: Tensor<B, 2>,
    spec: hound::WavSpec,
) -> Result<(), Box<dyn Error>> {
    let [n_channels, n_samples] = audio.dims();
    if n_channels != 2 {
        return Err("Audio tensor must be stereo to save.".into());
    }

    let out_spec = hound::WavSpec {
        channels: 2,
        sample_rate: spec.sample_rate,
        bits_per_sample: 32, // Saving as 32-bit int
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, out_spec)?;

    // De-interleave and convert to i32
    // Use .reshape([-1]) to ensure we have a 1D tensor [N]
    // 1. .into_data() returns `TensorData`
    // 2. .into_vec::<f32>() is a method on `TensorData`
    let ch1_samples: Vec<f32> = audio
        .clone()
        .slice([0..1])
        .reshape([-1])
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| format!("{:?}", e))?; // Manually convert DataError to String via Debug
    let ch2_samples: Vec<f32> = audio
        .slice([1..2])
        .reshape([-1])
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| format!("{:?}", e))?; // Manually convert DataError to String via Debug

    for i in 0..n_samples {
        let sample_l = (ch1_samples[i] * i32::MAX as f32) as i32;
        let sample_r = (ch2_samples[i] * i32::MAX as f32) as i32;
        writer.write_sample(sample_l)?;
        writer.write_sample(sample_r)?;
    }

    writer.finalize()?;
    Ok(())
}
