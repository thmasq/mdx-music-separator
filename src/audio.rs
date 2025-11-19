use burn::{
    prelude::{Backend, Tensor},
    tensor::Int,
};
// --- NEW IMPORTS ---
use kofft::{
    fft::Complex32,
    stft::{istft as kofft_istft, stft as kofft_stft},
};
// --- END NEW IMPORTS ---
use std::f32::consts::PI;

/// Generates a Hanning window. (Unchanged, still needed by model.rs)
pub fn hanning_window<B: Backend>(n: usize, periodic: bool, device: &B::Device) -> Tensor<B, 1> {
    let n_f = n as f32;
    let n_use = if periodic { n_f } else { n_f - 1.0 };

    if n_use == 0.0 {
        return Tensor::zeros([0], device);
    }

    let arange = Tensor::<B, 1, Int>::arange(0..n as i64, device).float();

    let factor = arange.mul_scalar(2.0 * PI).div_scalar(n_use);
    factor.cos().mul_scalar(-0.5).add_scalar(0.5)
}

/// A simple implementation of reflection padding for a 1D vector. (Unchanged)
/// (Making this function public)
pub fn reflect_pad_1d(data: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let len = data.len();
    if len == 0 {
        return vec![0.0; pad_left + pad_right];
    }
    let mut padded = Vec::with_capacity(len + pad_left + pad_right);

    // Left padding
    padded.extend((0..pad_left).map(|i| data[(pad_left - 1 - i).min(len - 1) % len]));
    // Original data
    padded.extend_from_slice(data);
    // Right padding
    padded.extend((0..pad_right).map(|i| data[len - 1 - (i % len)]));

    padded
}

/// Custom STFT implementation using `kofft`.
///
/// This function is not fully parallel and involves CPU-GPU data transfers
/// but bridges the gap left by Burn not having a native STFT.
pub fn stft<B: Backend>(
    x: Tensor<B, 2>,
    n_fft: usize,
    hop: usize,
    _win_length: Option<usize>,
    window: Option<Tensor<B, 1>>,
    center: bool,
    // Remove pad_mode, it doesn't exist
    _pad_mode: Option<()>,
    _normalized: bool,
    _onesided: bool,
    _return_complex: bool,
) -> Tensor<B, 4> {
    let [batch_size, _n_samples] = x.dims();
    let device = x.device();
    let n_bins = n_fft / 2 + 1;

    // 1. Get window Vec<f32>
    let window_vec = window
        .unwrap()
        .into_data()
        .into_vec::<f32>()
        .expect("Failed to get window vec");

    // 2. Process each item in the batch
    let mut batch_outputs = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let mut audio_ch = x
            .clone()
            .slice([i..i + 1])
            .reshape([-1])
            .into_data()
            .into_vec::<f32>()
            .expect("Failed to get audio vec");

        // 3. Centered padding (manual reflection padding)
        // This matches the behavior of torch.stft(center=True)
        if center {
            let pad_len = n_fft / 2;
            audio_ch = reflect_pad_1d(&audio_ch, pad_len, pad_len);
        }

        // 4. Perform STFT using kofft
        // 4.1. Prepare output frames
        // This calculation is from the kofft example
        let n_frames = (audio_ch.len() - n_fft) / hop + 1;
        let n_bins_kofft = n_fft / 2 + 1; // kofft returns this many bins

        let mut frames: Vec<Vec<Complex32>> =
            vec![vec![Complex32::new(0.0, 0.0); n_bins_kofft]; n_frames];

        // 4.2. Call kofft_stft
        // Signature: stft(signal: &[f32], window: &[f32], hop_size: usize, frames: &mut [Vec<Complex32>])
        kofft_stft(&audio_ch, &window_vec, hop, frames.as_mut_slice()).expect("kofft STFT failed");

        // The output `frames` is now populated.
        // `frames` has shape [n_frames, n_bins]
        let stft_output = frames;

        // 5. Convert kofft output to a flat Vec<f32> for Burn
        // We need to transpose from [n_frames, n_bins] to [n_bins, n_frames]
        // and also handle the complex [f32; 2]
        let mut flat_tensor_data = vec![0.0f32; n_bins * n_frames * 2];
        for t in 0..n_frames {
            for f in 0..n_bins {
                if f >= n_bins_kofft {
                    continue;
                } // Should not happen if n_bins == n_bins_kofft
                let complex_val = stft_output[t][f];
                // [n_bins, n_frames, 2]
                let out_idx = (f * n_frames + t) * 2;
                flat_tensor_data[out_idx] = complex_val.re; // real
                flat_tensor_data[out_idx + 1] = complex_val.im; // imag
            }
        }

        // 6. Convert to tensor [n_bins, n_frames, 2]
        let stft_tensor = Tensor::<B, 3>::from_floats(flat_tensor_data.as_slice(), &device)
            .reshape([n_bins, n_frames, 2]);

        batch_outputs.push(stft_tensor);
    }

    // 7. Stack batch: [B, n_bins, n_frames, 2]
    Tensor::stack(batch_outputs, 0)
}

/// Custom iSTFT implementation using `kofft`.
pub fn istft<B: Backend>(
    x: Tensor<B, 4>,
    n_fft: usize,
    hop: usize,
    _win_length: Option<usize>,
    window: Option<Tensor<B, 1>>,
    center: bool,
    _normalized: bool,
    _onesided: bool,
    length: Option<usize>,
    _return_complex: bool,
) -> Tensor<B, 2> {
    let [batch_size, n_bins, n_frames, _] = x.dims();
    let device = x.device();

    // 1. Get window Vec<f32>
    let window_vec = window
        .unwrap()
        .into_data()
        .into_vec::<f32>()
        .expect("Failed to get window vec");

    // 2. Process each item in the batch
    let mut batch_outputs = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        // Slice: [n_bins, n_frames, 2]
        let stft_tensor: Tensor<B, 3> = x.clone().slice([i..i + 1]).squeeze();

        // Convert to Vec<f32>
        let flat_stft_vec = stft_tensor
            .into_data()
            .into_vec::<f32>()
            .expect("Failed to get STFT vec");

        // 3. Convert flat Vec<f32> back to kofft's expected format:
        // &[Vec<Complex32>] (shape [n_frames, n_bins])
        let mut kofft_input: Vec<Vec<Complex32>> =
            vec![vec![Complex32::new(0.0, 0.0); n_bins]; n_frames];

        for f in 0..n_bins {
            for t in 0..n_frames {
                let in_idx = (f * n_frames + t) * 2;
                kofft_input[t][f] =
                    Complex32::new(flat_stft_vec[in_idx], flat_stft_vec[in_idx + 1]);
            }
        }

        // 4. Perform iSTFT
        // We need to know the output length for the buffer
        let expected_len = n_fft + (n_frames - 1) * hop;
        let mut audio_out = vec![0.0f32; expected_len];

        // Call kofft_istft
        // Signature: istft(frames: &[Vec<Complex32>], window: &[f32], hop_size: usize, output: &mut [f32])
        kofft_istft(&kofft_input, &window_vec, hop, &mut audio_out).expect("kofft iSTFT failed");

        // 5. Trim/pad to final length
        // This logic is identical to the previous implementation
        let mut final_audio = if center {
            let pad_len = n_fft / 2;
            let expected_len = audio_out.len();
            if expected_len > pad_len * 2 {
                audio_out[pad_len..expected_len - pad_len].to_vec()
            } else {
                Vec::new() // Padded length was less than trim length
            }
        } else {
            audio_out
        };

        if let Some(l) = length {
            if final_audio.len() > l {
                final_audio.truncate(l);
            } else if final_audio.len() < l {
                final_audio.resize(l, 0.0);
            }
        }

        batch_outputs.push(Tensor::<B, 1>::from_floats(final_audio.as_slice(), &device));
    }

    // 6. Stack batch: [B, n_samples]
    Tensor::stack(batch_outputs, 0)
}
