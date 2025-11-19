use crate::audio::{hanning_window, istft, stft};
use burn::{
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
    },
    prelude::{Backend, Config, Tensor},
};

// Define constants from the Python model
const DIM_C: usize = 4;
const DIM_T: usize = 256;

/// Configuration for the MDX-B model.
#[derive(Config, Debug, Copy)]
pub struct ConvTdfNetTrimModelConfig {
    /// Number of layers in the U-Net encoder/decoder
    #[config(default = 11)]
    pub l_layers: usize,

    /// FFT window size
    #[config(default = 6144)]
    pub n_fft: usize,

    /// Hop length
    #[config(default = 1024)]
    pub hop: usize,

    /// Frequency dimension
    #[config(default = 3072)]
    pub dim_f: usize,

    /// Overlap margin (percentage)
    #[config(default = 0.01)]
    pub margin: f64,
}

/// A Dense Block consisting of 3 consecutive convolutions.
#[derive(Module, Debug)]
pub struct DenseBlock<B: Backend> {
    pub layer_0: Conv2d<B>,
    pub layer_1: Conv2d<B>,
    pub layer_2: Conv2d<B>,
}

impl<B: Backend> DenseBlock<B> {
    pub fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        x = self.layer_0.forward(x);
        x = self.layer_1.forward(x);
        x = self.layer_2.forward(x);
        x
    }
}

impl ConvTdfNetTrimModelConfig {
    /// Initializes the model struct from the configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvTdfNetTrimModel<B> {
        let n = self.l_layers / 2; // n=5
        let n_bins = self.n_fft / 2 + 1;

        // --- Validation Check ---
        if n_bins < self.dim_f {
            panic!(
                "Invalid Configuration: n_fft ({}) produces {} bins, which is less than dim_f ({}). \
                 For 'Trim' models, n_bins must be >= dim_f.",
                self.n_fft, n_bins, self.dim_f
            );
        }

        // Channels matching the KimModelFused architecture
        let channels: [usize; 7] = [4, 48, 96, 144, 192, 240, 288];

        let kernel_size = [3, 3];
        let stride = [2, 2];
        let padding_same = PaddingConfig2d::Same;

        // --- Entry Conv ---
        let first_conv = Conv2dConfig::new(
            [channels[0], channels[1]], // 4 -> 48
            [1, 1],                     // kernel_size=1
        )
        .with_padding(padding_same.clone())
        .init(device);

        // --- Encoder (n=5) ---
        let mut ds_dense = Vec::with_capacity(n);
        let mut ds = Vec::with_capacity(n);

        for i in 0..n {
            let c_in = channels[i + 1];

            // Dense Block
            ds_dense.push(DenseBlock {
                layer_0: Conv2dConfig::new([c_in, c_in], kernel_size)
                    .with_padding(padding_same.clone())
                    .init(device),
                layer_1: Conv2dConfig::new([c_in, c_in], kernel_size)
                    .with_padding(padding_same.clone())
                    .init(device),
                layer_2: Conv2dConfig::new([c_in, c_in], kernel_size)
                    .with_padding(padding_same.clone())
                    .init(device),
            });

            // Downsample
            ds.push(
                Conv2dConfig::new([channels[i + 1], channels[i + 2]], [2, 2])
                    .with_stride(stride)
                    .with_padding(PaddingConfig2d::Explicit(0, 0))
                    .init(device),
            );
        }

        // --- Bottleneck ---
        let c_mid = channels[n + 1]; // 288
        let mid_dense = DenseBlock {
            layer_0: Conv2dConfig::new([c_mid, c_mid], kernel_size)
                .with_padding(padding_same.clone())
                .init(device),
            layer_1: Conv2dConfig::new([c_mid, c_mid], kernel_size)
                .with_padding(padding_same.clone())
                .init(device),
            layer_2: Conv2dConfig::new([c_mid, c_mid], kernel_size)
                .with_padding(padding_same.clone())
                .init(device),
        };

        // --- Decoder (n=5) ---
        let mut us = Vec::with_capacity(n);
        let mut us_dense = Vec::with_capacity(n);

        // Iterate backwards: i = 4, 3, 2, 1, 0
        for i in (0..n).rev() {
            let c_deep = channels[i + 2];
            let c_shallow = channels[i + 1];

            // Upsample
            us.push(
                ConvTranspose2dConfig::new([c_deep, c_shallow], [2, 2])
                    .with_stride(stride)
                    .with_padding([0, 0])
                    .init(device),
            );

            // Dense Block
            us_dense.push(DenseBlock {
                layer_0: Conv2dConfig::new([c_shallow, c_shallow], kernel_size)
                    .with_padding(padding_same.clone())
                    .init(device),
                layer_1: Conv2dConfig::new([c_shallow, c_shallow], kernel_size)
                    .with_padding(padding_same.clone())
                    .init(device),
                layer_2: Conv2dConfig::new([c_shallow, c_shallow], kernel_size)
                    .with_padding(padding_same.clone())
                    .init(device),
            });
        }

        // --- Exit Conv ---
        let final_conv = Conv2dConfig::new(
            [channels[1], channels[0]], // 48 -> 4
            [1, 1],                     // kernel_size=1
        )
        .with_padding(padding_same)
        .init(device);

        // --- STFT Hann Window ---
        let stft_window = hanning_window(self.n_fft, false, &device.clone());

        // --- Frequency Padding ---
        // Safe subtraction now guaranteed by check above
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
    ds_dense: Vec<DenseBlock<B>>,
    ds: Vec<Conv2d<B>>,
    mid_dense: DenseBlock<B>,
    us: Vec<ConvTranspose2d<B>>,
    us_dense: Vec<DenseBlock<B>>,
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
    pub fn stft(&self, x: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, _channels, _chunk_size] = x.dims();
        let x_reshaped: Tensor<B, 2> = x.reshape([-1, self.chunk_size as i64]);
        let x_stft: Tensor<B, 4> = stft(
            x_reshaped,
            self.n_fft,
            self.hop,
            Some(self.n_fft),
            Some(self.stft_window.clone()),
            true,
            None,
            false,
            true,
            false,
        );
        let x_permuted = x_stft.permute([0, 3, 1, 2]);
        let x_reshaped_2 = x_permuted.reshape([batch_size, 2, 2, self.n_bins, self.dim_t]);
        let x_final_shape: Tensor<B, 4> =
            x_reshaped_2.reshape([batch_size, DIM_C, self.n_bins, self.dim_t]);
        x_final_shape.slice([0..batch_size, 0..DIM_C, 0..self.dim_f, 0..self.dim_t])
    }

    /// iSTFT function matching the Python implementation.
    pub fn istft(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, _, _, _] = x.dims();
        let freq_pad_cloned = self.freq_pad.clone().repeat(&[batch_size]);
        let x_padded = Tensor::cat(vec![x, freq_pad_cloned], 2);
        let x_reshaped = x_padded.reshape([batch_size, 2, 2, self.n_bins, self.dim_t]);
        let x_reshaped_2 = x_reshaped.reshape([-1, 2, self.n_bins as i64, self.dim_t as i64]);
        let x_permuted = x_reshaped_2.permute([0, 2, 3, 1]);
        let x_istft: Tensor<B, 2> = istft(
            x_permuted,
            self.n_fft,
            self.hop,
            Some(self.n_fft),
            Some(self.stft_window.clone()),
            true,
            false,
            true,
            Some(self.chunk_size),
            false,
        );
        x_istft.reshape([batch_size, 2, self.chunk_size])
    }

    /// Forward pass through the U-Net.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // [B, C, F, T]
        let x = self.first_conv.forward(x);

        // Transpose: [B, C, F, T] -> [B, C, T, F]
        // We treat (T, F) as (H, W) for 2D convolutions
        let x = x.transpose();

        let mut ds_outputs = Vec::with_capacity(self.n);

        // --- Encoder ---
        let mut x_enc = x;
        for i in 0..self.n {
            // Run through dense block
            x_enc = self.ds_dense[i].forward(x_enc);
            ds_outputs.push(x_enc.clone());

            // Downsample
            x_enc = self.ds[i].forward(x_enc);
        }

        // --- Bottleneck ---
        let mut x_dec = self.mid_dense.forward(x_enc);

        // --- Decoder ---
        for i in 0..self.n {
            // Upsample
            x_dec = self.us[i].forward(x_dec);

            // Get skip connection
            let skip_con = ds_outputs.pop().unwrap();

            // Add skip connection (Summation for Kim models)
            x_dec = x_dec + skip_con;

            // Dense block
            x_dec = self.us_dense[i].forward(x_dec);
        }

        // Transpose back: [B, C, T, F] -> [B, C, F, T]
        let x_dec = x_dec.transpose();

        // --- Final Conv ---
        self.final_conv.forward(x_dec)
    }
}
