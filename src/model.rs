use crate::audio::{hanning_window, istft, stft};
use burn::{
    module::Module,
    nn::{
        PaddingConfig2d, // Moved from conv
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::{Backend, Config, Tensor},
};

// Define constants from the Python model
const DIM_C: usize = 4;
const DIM_T: usize = 256;

/// Configuration for the MDX-B model.
///
/// This struct holds the parameters required to build the model,
/// matching the `__init__` method of the Python `Conv_TDF_net_trim_model`.
#[derive(Config, Debug, Copy)]
pub struct ConvTdfNetTrimModelConfig {
    /// Number of layers in the U-Net encoder/decoder (L=11 in Python -> n=5)
    #[config(default = 11)]
    pub l_layers: usize,

    /// FFT window size (e.g., 6144 or 7680)
    #[config(default = 6144)]
    pub n_fft: usize,

    /// Hop length
    #[config(default = 1024)]
    pub hop: usize,

    /// Frequency dimension
    #[config(default = 3072)]
    pub dim_f: usize,

    /// Overlap margin (percentage)
    #[config(default = 0.01)] // 1% overlap margin
    pub margin: f64,
}

impl ConvTdfNetTrimModelConfig {
    /// Initializes the model struct from the configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTdfNetTrimModel<B> {
        let n = self.l_layers / 2; // n=5
        let n_bins = self.n_fft / 2 + 1;

        // --- Layer Definitions (Based on U-Net structure) ---
        // These parameters (channels, kernels) are not in inference.py,
        // so we define a plausible U-Net architecture.
        // These *must* match the parameters of the model you are loading weights from.

        let channels: [usize; 6] = [4, 32, 64, 128, 256, 512]; // Guessed channel sizes
        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding_same = PaddingConfig2d::Same;
        let padding_explicit = PaddingConfig2d::Explicit(1, 1);

        // --- Entry Conv ---
        let first_conv = Conv2dConfig::new(
            [channels[0], channels[1]], // 4 -> 32
            kernel_size,
        )
        .with_padding(padding_same.clone())
        .init(device);

        // --- Encoder (n=5) ---
        let mut ds_dense = Vec::with_capacity(n);
        let mut ds = Vec::with_capacity(n);
        for i in 0..n {
            ds_dense.push(
                Conv2dConfig::new([channels[i + 1], channels[i + 1]], kernel_size)
                    .with_padding(padding_same.clone())
                    .init(device),
            );
            ds.push(
                Conv2dConfig::new([channels[i + 1], channels[i + 2]], kernel_size)
                    .with_stride(stride)
                    .with_padding(padding_explicit.clone())
                    .init(device),
            );
        }

        // --- Bottleneck ---
        let mid_dense = Conv2dConfig::new([channels[n], channels[n]], kernel_size)
            .with_padding(padding_same.clone())
            .init(device);

        // --- Decoder (n=5) ---
        let mut us = Vec::with_capacity(n);
        let mut us_dense = Vec::with_capacity(n);
        for i in (0..n).rev() {
            us.push(
                ConvTranspose2dConfig::new([channels[i + 2], channels[i + 1]], kernel_size)
                    .with_stride(stride)
                    .with_padding([1, 1])
                    .with_padding_out([1, 1])
                    .init(device),
            );
            us_dense.push(
                Conv2dConfig::new([channels[i + 1] * 2, channels[i + 1]], kernel_size) // *2 for skip connection
                    .with_padding(padding_same.clone())
                    .init(device),
            );
        }

        // --- Exit Conv ---
        let final_conv = Conv2dConfig::new([channels[1], channels[0]], kernel_size) // 32 -> 4
            .with_padding(padding_same)
            .init(device);

        // --- STFT Hann Window ---
        let stft_window = hanning_window(self.n_fft, false, &device.clone());

        // --- Frequency Padding ---
        // out_c = 4 (since target_name != '*')
        let freq_pad = Tensor::<B, 4>::zeros([1, DIM_C, n_bins - self.dim_f, DIM_T], device);

        // --- Overlap-Add Window & Sizes ---
        let chunk_size = self.hop * (DIM_T - 1);
        let margin_size = (self.margin * chunk_size as f64) as usize;
        let overlap_window = hanning_window(chunk_size, false, &device.clone());

        ConvTdfNetTrimModel {
            first_conv,
            ds_dense,
            ds,
            mid_dense,
            us,
            us_dense,
            final_conv,
            n,
            n_fft: self.n_fft,
            hop: self.hop,
            dim_f: self.dim_f,
            dim_t: DIM_T,
            n_bins,
            chunk_size,
            margin_size,
            stft_window,
            freq_pad,
            overlap_window,
        }
    }
}

///
/// MDX-B Model (Conv_TDF_net_trim_model) implemented in Burn.
///
#[derive(Module, Debug)]
pub struct ConvTdfNetTrimModel<B: Backend> {
    // --- Layers ---
    first_conv: Conv2d<B>,
    ds_dense: Vec<Conv2d<B>>,
    ds: Vec<Conv2d<B>>,
    mid_dense: Conv2d<B>,
    us: Vec<ConvTranspose2d<B>>,
    us_dense: Vec<Conv2d<B>>,
    final_conv: Conv2d<B>,

    // --- STFT/iSTFT Parameters ---
    n: usize,
    n_fft: usize,
    hop: usize,
    dim_f: usize,
    dim_t: usize,
    n_bins: usize,
    pub chunk_size: usize,
    pub margin_size: usize,
    stft_window: Tensor<B, 1>,
    freq_pad: Tensor<B, 4>,
    pub overlap_window: Tensor<B, 1>,
}

impl<B: Backend> ConvTdfNetTrimModel<B> {
    /// STFT function matching the Python implementation.
    /// Input shape: [B, 2, chunk_size]
    /// Output shape: [B, 4, dim_f, dim_t]
    pub fn stft(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, _channels, _chunk_size] = x.dims();

        // 1. Reshape [B, 2, chunk_size] -> [B*2, chunk_size]
        // Use -1isize for automatic dimension calculation
        let x_reshaped: Tensor<B, 2> = x.reshape([-164, self.chunk_size as i64]);

        // 2. Perform STFT
        // Input: [B*2, chunk_size]
        // Output: [B*2, n_bins, dim_t, 2]
        let x_stft: Tensor<B, 4> = stft(
            x_reshaped,
            self.n_fft,
            self.hop,
            Some(self.n_fft), // win_length
            Some(self.stft_window.clone()),
            true,  // center
            None,  // pad_mode
            false, // normalized
            true,  // onesided
            false, // return_complex
        );

        // 3. Permute [B*2, n_bins, dim_t, 2] -> [B*2, 2, n_bins, dim_t]
        let x_permuted = x_stft.permute([0, 3, 1, 2]);

        // 4. Reshape [B*2, 2, n_bins, dim_t] -> [B, 2, 2, n_bins, dim_t]
        let x_reshaped_2 = x_permuted.reshape([batch_size, 2, 2, self.n_bins, self.dim_t]);

        // 5. Reshape [B, 2, 2, n_bins, dim_t] -> [B, 4, n_bins, dim_t]
        let x_final_shape: Tensor<B, 4> =
            x_reshaped_2.reshape([batch_size, DIM_C, self.n_bins, self.dim_t]);

        // 6. Slice to dim_f: [B, 4, n_bins, dim_t] -> [B, 4, dim_f, dim_t]
        x_final_shape.slice([0..batch_size, 0..DIM_C, 0..self.dim_f, 0..self.dim_t])
    }

    /// iSTFT function matching the Python implementation.
    /// Input shape: [B, 4, dim_f, dim_t]
    /// Output shape: [B, 2, chunk_size]
    pub fn istft(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, _, _, _] = x.dims();

        // 1. Concatenate frequency padding
        // freq_pad is [1, 4, n_bins - dim_f, dim_t], we need [B, 4, ...]
        // The `repeat` API takes a slice for new dims.
        // We repeat `batch_size` times on the 0th dim, and 1 for all others.
        let freq_pad_cloned = self.freq_pad.clone().repeat(&[batch_size]);
        // x: [B, 4, dim_f, dim_t]
        // freq_pad: [B, 4, n_bins - dim_f, dim_t]
        // output: [B, 4, n_bins, dim_t]
        let x_padded = Tensor::cat(vec![x, freq_pad_cloned], 2);

        // 2. Reshape [B, 4, n_bins, dim_t] -> [B, 2, 2, n_bins, dim_t]
        let x_reshaped = x_padded.reshape([batch_size, 2, 2, self.n_bins, self.dim_t]);

        // 3. Reshape [B, 2, 2, n_bins, dim_t] -> [B*2, 2, n_bins, dim_t]
        let x_reshaped_2 = x_reshaped.reshape([-1i64, 2, self.n_bins as i64, self.dim_t as i64]);

        // 4. Permute [B*2, 2, n_bins, dim_t] -> [B*2, n_bins, dim_t, 2]
        // .contiguous() is not needed in Burn as layout is handled.
        let x_permuted = x_reshaped_2.permute([0, 2, 3, 1]);

        // 5. Perform iSTFT
        // Input: [B*2, n_bins, dim_t, 2]
        // Output: [B*2, chunk_size]
        let x_istft: Tensor<B, 2> = istft(
            x_permuted,
            self.n_fft,
            self.hop,
            Some(self.n_fft), // win_length
            Some(self.stft_window.clone()),
            true,                  // center
            false,                 // normalized
            true,                  // onesided
            Some(self.chunk_size), // length
            false,                 // return_complex
        );

        // 6. Reshape [B*2, chunk_size] -> [B, 2, chunk_size]
        x_istft.reshape([batch_size, 2, self.chunk_size])
    }

    /// Forward pass through the U-Net.
    /// Input shape: [B, 4, 3072, 256] (B, C, F, T)
    /// Output shape: [B, 4, 3072, 256] (B, C, F, T)
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // [B, C, F, T]
        let x = self.first_conv.forward(x);

        // Transpose: [B, C, F, T] -> [B, C, T, F]
        // We treat (T, F) as (H, W) for 2D convolutions
        // .transpose() swaps the last two dimensions
        let x = x.transpose();

        let mut ds_outputs = Vec::with_capacity(self.n);

        // --- Encoder ---
        let mut x_enc = x;
        for i in 0..self.n {
            x_enc = self.ds_dense[i].forward(x_enc);
            ds_outputs.push(x_enc.clone());
            x_enc = self.ds[i].forward(x_enc);
        }

        // --- Bottleneck ---
        let mut x_dec = self.mid_dense.forward(x_enc);

        // --- Decoder ---
        for i in 0..self.n {
            // Upsample
            x_dec = self.us[i].forward(x_dec);

            // Get skip connection (from encoder, in reverse order)
            let skip_con = ds_outputs.pop().unwrap();

            // Concatenate skip connection
            // us_dense[i] has input channels C_in * 2
            x_dec = Tensor::cat(vec![x_dec, skip_con], 1);

            // Dense block
            x_dec = self.us_dense[i].forward(x_dec);
        }

        // Transpose back: [B, C, T, F] -> [B, C, F, T]
        // .transpose() swaps the last two dimensions
        let x_dec = x_dec.transpose();

        // --- Final Conv ---
        self.final_conv.forward(x_dec)
    }
}
