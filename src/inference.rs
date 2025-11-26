use crate::utils::STFT;
use anyhow::{Context, Result};
use ndarray::{Array2, Array3, Array4, ArrayViewD, s};
use num_complex::Complex;
use std::path::Path;
use tract_onnx::prelude::*;

pub struct MDXModel {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
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
        // Load the ONNX model
        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("Failed to load ONNX model")?
            .into_optimized()?
            .into_runnable()?;

        let stft = STFT::new(n_fft, hop_length);

        Ok(Self {
            model,
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
    pub fn demix(&self, audio: &Array2<f32>) -> Result<Array2<f32>> {
        // 1. Compute STFT
        // Output shape: (Channels, Freq, Time)
        let spec_complex = self.stft.forward(audio.view());
        let (channels, n_freq, n_frames) = spec_complex.dim();

        // --- MDX-Net frequency correction ---
        // STFT gives n_fft/2+1 bins = 3073 for n_fft=6144.
        // MDX-Net expects exactly 3072 (Nyquist removed).
        let model_n_freq = n_freq - 1; // 3072

        // Trim Nyquist
        let spec_trimmed = spec_complex.slice(s![.., 0..model_n_freq, ..]).to_owned();

        // 2. Prepare Input for ONNX
        // MDX models typically expect shape (Batch, Channels*2, Freq, Time)
        // Where Channels*2 comes from concatenating Real and Imaginary parts.
        // PyTorch Logic: torch.cat([x.real, x.imag], dim=1)
        // Order: [Ch0_Re, Ch1_Re, Ch0_Im, Ch1_Im] (assuming stereo input)

        // We process in chunks (segments) along the Time axis to avoid memory issues
        // and match model training dimensions.

        let mut out_spec = Array3::<Complex<f32>>::zeros((channels, model_n_freq, n_frames));

        // Pad spectrogram if necessary to fit segment size
        // (Simple implementation: processing valid chunks, edge handling might need zero-padding in production)

        let overlap = 0; // Some models use overlap-add on the spectrogram, standard MDX usually cuts
        let step = self.segment_size - overlap;

        for t in (0..n_frames).step_by(step) {
            let end = std::cmp::min(t + self.segment_size, n_frames);
            let width = end - t;

            // If the last chunk is too small, we might need to pad it.
            // For now, let's assume the model can handle dynamic width OR we pad to segment_size.
            // Tract handles dynamic shapes well, but fixed shapes are faster.
            // Most MDX ONNX exports have fixed input sizes. We will pad with zeros.

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

            // 3. Run Inference
            let tract_input = input_tensor.into_tensor();
            let result = self.model.run(tvec!(tract_input.into()))?;

            // FIX: Use ArrayViewD (dynamic) instead of forcing dimensionality.
            // This allows the output to be 3D or 4D depending on the model,
            // and allows indexing with a slice `&[usize]`.
            let output_tensor: ArrayViewD<f32> = result[0].to_array_view::<f32>()?;

            // Output usually: (Batch, Channels*2, Freq, Time) or (Batch, Channels, Freq, Time) depending on model
            // MDX usually outputs the residual or the target directly in same shape.

            // Iterate and write back to out_spec
            // The model output usually corresponds to the specific target (e.g. vocals).

            // Reshape view to (Batch, 2*Chan, Freq, Time) - Tract might return reduced dims
            let out_shape = output_tensor.shape();
            // Assuming output is (1, 4, Freq, Time) collapsed to (4, Freq, Time) if batch is 1

            // Map output back to Complex
            for c in 0..channels {
                for f in 0..model_n_freq {
                    for dt in 0..width {
                        // Read from output tensor
                        // Depending on if tract squeezed the batch dim:
                        let re_idx = if out_shape.len() == 4 {
                            vec![0, c, f, dt]
                        } else {
                            vec![c, f, dt]
                        };
                        let im_idx = if out_shape.len() == 4 {
                            vec![0, c + channels, f, dt]
                        } else {
                            vec![c + channels, f, dt]
                        };

                        // Safety: The model output segment might be larger than 'width' if we padded input
                        // We only read the valid 'width' part.

                        // FIX: Indexing with slice now works because output_tensor is ArrayViewD
                        let re = output_tensor[&*re_idx];
                        let im = output_tensor[&*im_idx];

                        // MDX-Net 'v1' ONNX models usually predict the *residual* or the *source*.
                        // Assuming standard "Kim" models, they predict the source directly.
                        // However, some versions predict the "other" stem.
                        // Let's assume direct prediction of the target for now.

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
