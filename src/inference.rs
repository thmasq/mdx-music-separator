use crate::utils::STFT;
use anyhow::{Context, Result};
use ndarray::{Array3, ArrayViewD, Axis, Zip, s};
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
    batch_size: usize,
    overlap: usize,
}

impl MDXModel {
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        n_fft: usize,
        hop_length: usize,
        segment_size: usize,
        batch_size: usize,
        overlap: usize,
    ) -> Result<Self> {
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
            batch_size,
            overlap,
        })
    }

    pub fn demix(&mut self, audio: &ndarray::Array2<f32>) -> Result<ndarray::Array2<f32>> {
        // 1. Compute STFT
        let spec_complex = self.stft.forward(audio.view());
        let (channels, n_freq, n_frames) = spec_complex.dim();

        // --- MDX-Net frequency correction ---
        let model_n_freq = n_freq - 1;
        let spec_trimmed = spec_complex.slice(s![.., 0..model_n_freq, ..]);

        // 2. Prepare Output Accumulators (Overlap-Add)
        let mut out_spec = Array3::<Complex<f32>>::zeros((channels, model_n_freq, n_frames));
        let mut divider = Array3::<f32>::zeros((channels, model_n_freq, n_frames));

        // 3. Batched Inference Loop
        let step = self.segment_size.saturating_sub(self.overlap).max(1);

        let mut batch_tensors: Vec<Array3<f32>> = Vec::with_capacity(self.batch_size);
        let mut batch_indices = Vec::with_capacity(self.batch_size);

        for t in (0..n_frames).step_by(step) {
            let end = std::cmp::min(t + self.segment_size, n_frames);
            let width = end - t;

            let mut chunk = Array3::<f32>::zeros((channels * 2, model_n_freq, self.segment_size));

            for c in 0..channels {
                for f in 0..model_n_freq {
                    for dt in 0..width {
                        let val = spec_trimmed[[c, f, t + dt]];
                        chunk[[c, f, dt]] = val.re;
                        chunk[[c + channels, f, dt]] = val.im;
                    }
                }
            }

            batch_tensors.push(chunk);
            batch_indices.push((t, width));

            if batch_tensors.len() >= self.batch_size {
                self.process_batch(
                    &batch_tensors,
                    &batch_indices,
                    &mut out_spec,
                    &mut divider,
                    channels,
                    model_n_freq,
                )?;
                batch_tensors.clear();
                batch_indices.clear();
            }
        }

        if !batch_tensors.is_empty() {
            self.process_batch(
                &batch_tensors,
                &batch_indices,
                &mut out_spec,
                &mut divider,
                channels,
                model_n_freq,
            )?;
        }

        // 4. Normalize by divider
        Zip::from(&mut out_spec)
            .and(&divider)
            .for_each(|out, &div| {
                if div > 0.0 {
                    *out = *out / div;
                }
            });

        // 5. Inverse STFT
        let mut out_padded = Array3::<Complex<f32>>::zeros((channels, model_n_freq + 1, n_frames));
        out_padded
            .slice_mut(s![.., 0..model_n_freq, ..])
            .assign(&out_spec);

        let output_audio = self
            .stft
            .inverse(out_padded.view(), Some(audio.len() / channels));

        Ok(output_audio)
    }

    fn process_batch(
        &mut self,
        tensors: &[Array3<f32>],
        indices: &[(usize, usize)],
        out_spec: &mut Array3<Complex<f32>>,
        divider: &mut Array3<f32>,
        channels: usize,
        model_n_freq: usize,
    ) -> Result<()> {
        if tensors.is_empty() {
            return Ok(());
        }

        let views: Vec<_> = tensors.iter().map(|a| a.view()).collect();
        let input_tensor = ndarray::stack(Axis(0), &views)?;

        let input_value = Value::from_array(input_tensor)?;
        let outputs = self.session.run(inputs![input_value])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let output_view = ArrayViewD::from_shape(shape_usize, data)?;

        for (batch_idx, &(t, width)) in indices.iter().enumerate() {
            for c in 0..channels {
                for f in 0..model_n_freq {
                    for dt in 0..width {
                        // Access logic for batched output
                        let re = output_view[[batch_idx, c, f, dt]];
                        let im = output_view[[batch_idx, c + channels, f, dt]];

                        let val = Complex::new(re, im);
                        out_spec[[c, f, t + dt]] += val;
                        divider[[c, f, t + dt]] += 1.0;
                    }
                }
            }
        }

        Ok(())
    }
}
