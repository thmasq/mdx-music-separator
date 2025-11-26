use crate::utils::STFT;
use anyhow::{Context, Result};
use ndarray::{Array2, Array3, Array4, ArrayViewD, s};
use num_complex::Complex;
use ort::{
    inputs,
    session::{
        Session,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
    value::Value,
};
use std::path::Path;

pub struct MDXModel {
    session: Session,
    stft: STFT,
    segment_size: usize,
}

impl MDXModel {
    /// Load an MDX-Net ONNX model from a file.
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx file.
    /// * `n_fft` - FFT size (e.g., 6144, 7680 depending on the specific MDX model).
    /// * `hop_length` - Hop length (e.g., 1024).
    /// * `segment_size` - The number of time frames the model processes in one pass (e.g., 256).
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        n_fft: usize,
        hop_length: usize,
        segment_size: usize,
    ) -> Result<Self> {
        // Initialize the ORT Session
        let session = SessionBuilder::new()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(16)?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model via ORT")?;

        let stft = STFT::new(n_fft, hop_length);

        Ok(Self {
            session,
            stft,
            segment_size,
        })
    }

    /// Run inference on the provided audio.
    ///
    /// # Arguments
    /// * `audio` - Input audio tensor of shape (Channels, Samples).
    ///
    /// # Returns
    /// * Separated audio tensor of shape (Channels, Samples).
    pub fn demix(&mut self, audio: &Array2<f32>) -> Result<Array2<f32>> {
        // 1. Compute STFT
        // Output shape: (Channels, Freq, Time)
        let spec_complex = self.stft.forward(audio.view());
        let (channels, n_freq, n_frames) = spec_complex.dim();

        // --- MDX-Net frequency correction ---
        // STFT gives n_fft/2+1 bins = 3073 for n_fft=6144.
        // MDX-Net expects exactly 3072 (Nyquist removed).
        let model_n_freq = n_freq - 1; // 3072

        // Trim Nyquist
        let spec_trimmed = spec_complex.slice(s![.., 0..model_n_freq, ..]);

        // 2. Prepare Input for ONNX
        // MDX models typically expect shape (Batch, Channels*2, Freq, Time)
        // Where Channels*2 comes from concatenating Real and Imaginary parts.

        let mut out_spec = Array3::<Complex<f32>>::zeros((channels, model_n_freq, n_frames));

        let overlap = 0;
        let step = self.segment_size - overlap;

        for t in (0..n_frames).step_by(step) {
            let end = std::cmp::min(t + self.segment_size, n_frames);
            let width = end - t;

            // Prepare input tensor (1, Channels*2, Freq, SegmentSize)
            // We use a fixed size tensor and zero-pad if the last chunk is smaller than segment_size
            let mut input_tensor =
                Array4::<f32>::zeros((1, channels * 2, model_n_freq, self.segment_size));

            // Fill input tensor
            for c in 0..channels {
                for f in 0..model_n_freq {
                    for dt in 0..width {
                        let val = spec_trimmed[[c, f, t + dt]];
                        // Real part channel
                        input_tensor[[0, c, f, dt]] = val.re;
                        // Imag part channel (offset by number of channels)
                        input_tensor[[0, c + channels, f, dt]] = val.im;
                    }
                }
            }

            // 3. Run Inference using ORT
            // Create an ORT Value (Tensor) from the ndarray view.
            // This explicit conversion is required because inputs! macro doesn't auto-convert views.
            let input_value = Value::from_array(input_tensor)?;

            // Run the session using the inputs! macro
            let outputs = self.session.run(inputs![input_value])?;

            // Extract the first output tensor
            let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
            let output_view = ArrayViewD::from_shape(shape_usize, data)?;

            // output_view shape is typically (Batch, Channels*2, Freq, Time)
            let out_shape = output_view.shape();

            // Iterate and write back to out_spec
            for c in 0..channels {
                for f in 0..model_n_freq {
                    for dt in 0..width {
                        // Handle output shapes (Batch may or may not be squeezed by the model)

                        let re = if out_shape.len() == 4 {
                            output_view[[0, c, f, dt]]
                        } else {
                            output_view[[c, f, dt]]
                        };

                        let im = if out_shape.len() == 4 {
                            output_view[[0, c + channels, f, dt]]
                        } else {
                            output_view[[c + channels, f, dt]]
                        };

                        out_spec[[c, f, t + dt]] = Complex::new(re, im);
                    }
                }
            }
        }

        // 4. Inverse STFT
        let mut out_padded = Array3::<Complex<f32>>::zeros((channels, model_n_freq + 1, n_frames));
        out_padded
            .slice_mut(s![.., 0..model_n_freq, ..])
            .assign(&out_spec);

        let output_audio = self
            .stft
            .inverse(out_padded.view(), Some(audio.len() / channels));

        Ok(output_audio)
    }
}
