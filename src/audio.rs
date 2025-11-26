use anyhow::Result;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use ndarray::{Array2, ArrayView2};

/// Loads a WAV file into an ndarray with shape (Channels, Samples).
///
/// # Arguments
/// * `path` - Path to the WAV file.
///
/// # Returns
/// * A tuple containing:
///   - Audio tensor of shape (Channels, Samples)
///   - Sample rate
pub fn load_wav(path: &str) -> Result<(Array2<f32>, u32)> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    // Read all samples as floats
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(Result::ok)
            .collect(),
        SampleFormat::Int => {
            let max_val = 2u32.pow(spec.bits_per_sample as u32 - 1) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|x| x as f32 / max_val)
                .collect()
        }
    };

    let len = samples.len() / channels;

    // De-interleave: Hound gives [L, R, L, R], we want [[L, L...], [R, R...]]
    let mut audio = Array2::<f32>::zeros((channels, len));
    for (i, sample) in samples.iter().enumerate() {
        let ch = i % channels;
        let t = i / channels;
        audio[[ch, t]] = *sample;
    }

    Ok((audio, spec.sample_rate))
}

/// Saves an ndarray of shape (Channels, Samples) to a WAV file.
///
/// # Arguments
/// * `path` - Output path.
/// * `data` - Audio data (Channels, Samples).
/// * `sample_rate` - Sample rate (e.g., 44100).
pub fn save_wav(path: &str, data: ArrayView2<f32>, sample_rate: u32) -> Result<()> {
    let (channels, len) = data.dim();

    let spec = WavSpec {
        channels: channels as u16,
        sample_rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };

    let mut writer = WavWriter::create(path, spec)?;

    // Interleave: We have [[L...], [R...]], WAV wants [L, R, L, R]
    for t in 0..len {
        for c in 0..channels {
            writer.write_sample(data[[c, t]])?;
        }
    }

    Ok(())
}
