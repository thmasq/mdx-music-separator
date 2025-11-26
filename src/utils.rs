use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Zip, s};
use num_complex::Complex;
use rustfft::{Fft, FftPlanner, num_complex::Complex as FftComplex};
use std::sync::Arc;

/// A helper struct to handle Short-Time Fourier Transform operations
/// matching PyTorch's defaults (Hann window, center=True, reflect padding).
pub struct STFT {
    pub n_fft: usize,
    pub hop_length: usize,
    pub window: Array1<f32>,
    fft_forward: Arc<dyn Fft<f32>>,
    fft_inverse: Arc<dyn Fft<f32>>,
}

impl STFT {
    /// Create a new STFT instance.
    ///
    /// # Arguments
    /// * `n_fft` - The size of the FFT.
    /// * `hop_length` - The stride between windows.
    pub fn new(n_fft: usize, hop_length: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(n_fft);
        let fft_inverse = planner.plan_fft_inverse(n_fft);

        // Create Hann window using apodize
        // PyTorch default is periodic=True (window function has denominator N).
        // Standard signal processing (and apodize) is usually symmetric (denominator N-1).
        // To get a periodic window of size N, we generate a symmetric window of size N+1
        // and take the first N samples. This ensures numerical matching with torch.hann_window(N, periodic=True).
        let window = apodize::hanning_iter(n_fft + 1)
            .take(n_fft)
            .map(|x| x as f32)
            .collect::<Array1<f32>>();

        Self {
            n_fft,
            hop_length,
            window,
            fft_forward,
            fft_inverse,
        }
    }

    /// Perform STFT on a batch of audio signals.
    ///
    /// Matches: torch.stft(..., center=True, pad_mode='reflect', return_complex=True)
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (Batch, Time)
    ///
    /// # Returns
    /// * Output tensor of shape (Batch, Frequency, Frames) as Complex<f32>
    pub fn forward(&self, input: ArrayView2<f32>) -> Array3<Complex<f32>> {
        let (batch_size, original_len) = input.dim();

        // Calculate padding (center=True)
        // PyTorch pads n_fft // 2 on both sides
        let pad = self.n_fft / 2;

        // Calculate number of frames
        // PyTorch formula: floor((original_len + 2*pad - n_fft) / hop_length) + 1
        // We use the explicit formula to handle odd n_fft/padding cases correctly
        let n_frames = (original_len + 2 * pad - self.n_fft) / self.hop_length + 1;

        // Frequency bins (n_fft // 2 + 1)
        let n_freqs = self.n_fft / 2 + 1;

        // Initialize output array
        let mut output = Array3::<Complex<f32>>::zeros((batch_size, n_freqs, n_frames));

        // Process each item in the batch
        for (b, signal) in input.outer_iter().enumerate() {
            // 1. Apply Reflection Padding
            let padded_signal = pad_reflect(signal, pad);

            // 2. Windowing and FFT
            let mut frame_buffer = vec![FftComplex { re: 0.0, im: 0.0 }; self.n_fft];

            for t in 0..n_frames {
                let start = t * self.hop_length;

                // Copy chunk and apply window
                for i in 0..self.n_fft {
                    // Safety check for padding edge cases, though n_frames calculation should prevent this
                    if start + i < padded_signal.len() {
                        let sig_val = padded_signal[start + i];
                        let win_val = self.window[i];
                        frame_buffer[i] = FftComplex {
                            re: sig_val * win_val,
                            im: 0.0,
                        };
                    }
                }

                // Perform FFT
                self.fft_forward.process(&mut frame_buffer);

                // Store only the positive frequencies (first n_freqs)
                for f in 0..n_freqs {
                    output[[b, f, t]] = Complex::new(frame_buffer[f].re, frame_buffer[f].im);
                }
            }
        }

        output
    }

    /// Perform Inverse STFT.
    ///
    /// Matches: torch.istft(..., center=True)
    ///
    /// # Arguments
    /// * `input` - Spectrogram of shape (Batch, Frequency, Frames)
    /// * `length` - Optional length to trim the output to (original signal length).
    ///              If None, calculates based on frame count.
    pub fn inverse(&self, input: ArrayView3<Complex<f32>>, length: Option<usize>) -> Array2<f32> {
        let (batch_size, n_freqs, n_frames) = input.dim();
        assert_eq!(
            n_freqs,
            self.n_fft / 2 + 1,
            "Input frequency dimension mismatch"
        );

        let expected_len = self.n_fft + self.hop_length * (n_frames - 1);
        let trim_len = length.unwrap_or(expected_len - self.n_fft); // Approximate if not provided
        let pad = self.n_fft / 2; // Because center=True was used in forward

        let output_len = expected_len;
        let mut output_signal = Array2::<f32>::zeros((batch_size, output_len));
        let mut norm_signal = Array1::<f32>::zeros(output_len);

        // Pre-calculate window squared for NOLA (Overlap-Add) normalization
        let window_sq = &self.window * &self.window;

        // Construct the normalization signal (sum of squared windows)
        for t in 0..n_frames {
            let start = t * self.hop_length;
            let end = start + self.n_fft;
            if end <= output_len {
                let mut slice = norm_signal.slice_mut(s![start..end]);
                slice += &window_sq;
            }
        }

        let mut frame_buffer = vec![FftComplex { re: 0.0, im: 0.0 }; self.n_fft];

        for (b, spectrogram) in input.outer_iter().enumerate() {
            let mut batch_out = output_signal.row_mut(b);

            for t in 0..n_frames {
                let start = t * self.hop_length;

                // Reconstruct full symmetric spectrum
                // Copy positive frequencies
                for f in 0..n_freqs {
                    let c = spectrogram[[f, t]];
                    frame_buffer[f] = FftComplex { re: c.re, im: c.im };
                }

                // Reconstruct negative frequencies (conjugate symmetry)
                // We iterate positive bins and set their corresponding negative bin.
                // Bin 0 is DC, Bin n_freqs-1 might be Nyquist (if even).
                for f in 1..n_freqs {
                    let idx = self.n_fft - f;
                    // Only set if we haven't wrapped around to positive freq part
                    // (This logic works for both Even and Odd n_fft)
                    if idx > f {
                        let c = frame_buffer[f];
                        frame_buffer[idx] = FftComplex {
                            re: c.re,
                            im: -c.im,
                        };
                    }
                }

                // Ensure Nyquist bin is real if n_fft is even.
                // This is crucial for numerical matching with PyTorch's real-signal assumption.
                if self.n_fft % 2 == 0 {
                    frame_buffer[self.n_fft / 2].im = 0.0;
                }

                // IFFT
                self.fft_inverse.process(&mut frame_buffer);

                // Overlap Add
                // rustfft inverse is unnormalized (returns Sum), so we scale by 1/N
                let scale = 1.0 / self.n_fft as f32;

                for i in 0..self.n_fft {
                    if start + i < output_len {
                        // Apply window and add
                        let val = frame_buffer[i].re * scale;
                        batch_out[start + i] += val * self.window[i];
                    }
                }
            }
        }

        // Normalize by envelope (NOLA)
        Zip::from(&mut output_signal)
            .and_broadcast(&norm_signal)
            .for_each(|out, &norm| {
                // Avoid division by zero
                if norm > 1e-10 {
                    *out /= norm;
                }
            });

        // Trim padding (center=True reverses to center crop)
        // We padded `pad` on left and right.
        let start = pad;
        let end = start + trim_len;

        // Ensure we don't slice out of bounds
        let safe_end = std::cmp::min(end, output_len);
        let safe_start = std::cmp::min(start, safe_end);

        output_signal.slice_move(s![.., safe_start..safe_end])
    }
}

/// Helper for Reflection Padding (1D)
/// Mimics torch.nn.functional.pad(..., mode='reflect')
fn pad_reflect(input: ArrayView1<f32>, pad: usize) -> Array1<f32> {
    let len = input.len();
    let output_len = len + 2 * pad;
    let mut output = Array1::<f32>::zeros(output_len);

    // Copy center
    output.slice_mut(s![pad..pad + len]).assign(&input);

    // Reflect Left: inputs[1], inputs[2]...
    // Boundary (index 0) is not repeated in standard reflect
    for i in 0..pad {
        // input index: 1 + i
        // output index: pad - 1 - i
        let input_idx = 1 + i;
        let output_idx = pad - 1 - i;
        if input_idx < len {
            output[output_idx] = input[input_idx];
        }
    }

    // Reflect Right: inputs[len-2], inputs[len-3]...
    for i in 0..pad {
        // input index: len - 2 - i
        // output index: pad + len + i
        let input_idx = len - 2 - i;
        let output_idx = pad + len + i;
        if input_idx < len {
            // Check for usize underflow conceptually, though indices are strict
            output[output_idx] = input[input_idx];
        }
    }

    output
}
