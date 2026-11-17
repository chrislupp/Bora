/*
    Bora: Rust Port of NeuralFoil

    Copyright 2026 Christopher A. Lupp

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    This work has been cleared for public release, distribution unlimited, case
    number: AFRL-2026-2087.

    The views expressed are those of the authors and do not reflect the
    official guidance or position of the United States Government, the
    Department of Defense or of the United States Air Force.

    Statement from DoD: The Appearance of external hyperlinks does not
    constitute endorsement by the United States Department of Defense (DoD) of
    the linked websites, of the information, products, or services contained
    therein. The DoD does not exercise any editorial, security, or other
    control over the information you may find at these locations.
*/

//! The NeuralFoil analysis model.

use std::f64::consts::LN_10;
use std::path::{Path, PathBuf};

use crate::input_index as idx;
use crate::{
    AeroGradients, AeroInput, AeroOutput, AeroResult, BoundaryLayerData, BoundaryLayerGradients,
    Error, Gradient, InputDistribution, ModelSize, NUM_BL_POINTS, NUM_CST_WEIGHTS_PER_SIDE,
    NUM_INPUTS, NUM_LATENT_INPUTS, NUM_LATENT_OUTPUTS, Network, Result,
};

const N: usize = NUM_BL_POINTS;
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Source-tree weight directory, used when no other location is given.
const SOURCE_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data");

/// NeuralFoil airfoil analysis model.
///
/// A Rust port of NeuralFoil (<https://github.com/peterdsharpe/NeuralFoil>): a
/// physics-informed neural network surrogate of XFoil. Given an airfoil shape
/// in Kulfan (CST) parameterization and an operating condition, it predicts
/// lift, drag, and moment coefficients, transition locations, and boundary
/// layer distributions — optionally with analytic gradients with respect to
/// all inputs, enabling gradient-based design optimization.
///
/// Instances are immutable after construction and safe to share across
/// threads.
#[derive(Debug, Clone)]
pub struct NeuralFoil {
    size: ModelSize,
    network: Network,
    distribution: InputDistribution,
}

/// Matches NeuralFoil's numerically-stable sigmoid: inputs are clipped to
/// suppress overflow, with the clip bound derived from the float64 range.
fn stable_sigmoid(x: f64) -> f64 {
    let ln_eps = (10.0 / f64::MAX).ln();
    let x = x.clamp(ln_eps, -ln_eps);
    1.0 / (1.0 + (-x).exp())
}

/// Builds the 25-dimensional scaled latent input vector from user inputs.
fn build_latent_input(input: &AeroInput) -> [f64; NUM_LATENT_INPUTS] {
    let mut x = [0.0; NUM_LATENT_INPUTS];
    x[..8].copy_from_slice(&input.kulfan.upper_weights);
    x[8..16].copy_from_slice(&input.kulfan.lower_weights);
    x[16] = input.kulfan.leading_edge_weight;
    x[17] = input.kulfan.te_thickness * 50.0;
    let alpha_rad = input.alpha * DEG_TO_RAD;
    x[18] = (2.0 * alpha_rad).sin();
    x[19] = alpha_rad.cos();
    x[20] = 1.0 - x[19] * x[19];
    x[21] = (input.re.ln() - 12.5) / 3.5;
    x[22] = (input.n_crit - 9.0) / 4.5;
    x[23] = input.xtr_upper;
    x[24] = input.xtr_lower;
    x
}

/// Builds the seed Jacobian d(latent input)/d(user inputs), row-major
/// `NUM_LATENT_INPUTS x NUM_INPUTS`.
fn build_seed_jacobian(input: &AeroInput) -> Vec<f64> {
    let mut jx = vec![0.0; NUM_LATENT_INPUTS * NUM_INPUTS];
    let mut set = |row: usize, col: usize, value: f64| jx[row * NUM_INPUTS + col] = value;

    for i in 0..NUM_CST_WEIGHTS_PER_SIDE {
        set(i, idx::UPPER_WEIGHTS + i, 1.0);
        set(8 + i, idx::LOWER_WEIGHTS + i, 1.0);
    }
    set(16, idx::LEADING_EDGE_WEIGHT, 1.0);
    set(17, idx::TE_THICKNESS, 50.0);
    let alpha_rad = input.alpha * DEG_TO_RAD;
    set(18, idx::ALPHA, 2.0 * DEG_TO_RAD * (2.0 * alpha_rad).cos());
    set(19, idx::ALPHA, -DEG_TO_RAD * alpha_rad.sin());
    set(20, idx::ALPHA, DEG_TO_RAD * (2.0 * alpha_rad).sin());
    set(21, idx::RE, 1.0 / (3.5 * input.re));
    set(22, idx::N_CRIT, 1.0 / 4.5);
    set(23, idx::XTR_UPPER, 1.0);
    set(24, idx::XTR_LOWER, 1.0);
    jx
}

/// Applies the alpha-symmetry flip to the latent input: the airfoil is
/// mirrored (upper/lower surfaces swapped and negated) and alpha is negated.
/// The callback receives (dest_index, source_index, sign); the same linear map
/// applies to latent values and to rows of a seed Jacobian.
fn apply_input_flip(mut apply: impl FnMut(usize, usize, f64)) {
    for i in 0..8 {
        apply(i, 8 + i, -1.0); // kulfan_upper <- -kulfan_lower
        apply(8 + i, i, -1.0); // kulfan_lower <- -kulfan_upper
    }
    apply(16, 16, -1.0); // leading edge weight
    apply(17, 17, 1.0); // TE thickness
    apply(18, 18, -1.0); // sin(2a)
    apply(19, 19, 1.0); // cos(a)
    apply(20, 20, 1.0); // 1 - cos(a)^2
    apply(21, 21, 1.0); // scaled ln(Re)
    apply(22, 22, 1.0); // scaled n_crit
    apply(23, 24, 1.0); // xtr_upper <- xtr_lower
    apply(24, 23, 1.0); // xtr_lower <- xtr_upper
}

/// Applies the inverse of the output flip to results of a flipped evaluation,
/// bringing them back to the original airfoil's orientation. The callback
/// receives (dest_index, source_index, sign) and covers every output index.
fn apply_output_unflip(mut apply: impl FnMut(usize, usize, f64)) {
    apply(0, 0, 1.0); // analysis confidence
    apply(1, 1, -1.0); // CL
    apply(2, 2, 1.0); // CD
    apply(3, 3, -1.0); // CM
    apply(4, 5, 1.0); // Top_Xtr <- Bot_Xtr
    apply(5, 4, 1.0); // Bot_Xtr <- Top_Xtr
    for i in 0..2 * N {
        apply(6 + i, 6 + 3 * N + i, 1.0); // upper theta, H <- lower theta, H
        apply(6 + 3 * N + i, 6 + i, 1.0); // lower theta, H <- upper theta, H
    }
    for i in 0..N {
        apply(6 + 2 * N + i, 6 + 5 * N + i, -1.0); // upper ue <- -lower ue
        apply(6 + 5 * N + i, 6 + 2 * N + i, -1.0); // lower ue <- -upper ue
    }
}

/// Decodes the fused latent outputs into physical quantities, optionally
/// propagating the latent Jacobian (`jy_fused`, row-major
/// `NUM_LATENT_OUTPUTS x NUM_INPUTS`) into per-output gradients.
fn decode_outputs(
    y_fused: &[f64],
    input: &AeroInput,
    jy_fused: Option<&[f64]>,
    mut gradients: Option<&mut AeroGradients>,
) -> AeroOutput {
    let mut output = AeroOutput {
        // Scalar outputs, with the same scaling/squashing as upstream NeuralFoil.
        analysis_confidence: stable_sigmoid(y_fused[0]),
        cl: y_fused[1] / 2.0,
        cd: ((y_fused[2] - 2.0) * 2.0).exp(),
        cm: y_fused[3] / 20.0,
        top_xtr: y_fused[4].clamp(0.0, 1.0),
        bot_xtr: y_fused[5].clamp(0.0, 1.0),
        ..AeroOutput::default()
    };

    let jac_row =
        |i: usize| &jy_fused.expect("gradients require a Jacobian")[i * NUM_INPUTS..][..NUM_INPUTS];
    let scale_row = |i: usize, factor: f64| -> Gradient {
        let row = jac_row(i);
        std::array::from_fn(|k| factor * row[k])
    };

    if let Some(g) = gradients.as_deref_mut() {
        let conf = output.analysis_confidence;
        // Clipped transition locations: zero gradient where the clip is active.
        let clip_active = |v: f64| if v > 0.0 && v < 1.0 { 1.0 } else { 0.0 };
        g.analysis_confidence = scale_row(0, conf * (1.0 - conf));
        g.cl = scale_row(1, 0.5);
        g.cd = scale_row(2, 2.0 * output.cd);
        g.cm = scale_row(3, 1.0 / 20.0);
        g.top_xtr = scale_row(4, clip_active(y_fused[4]));
        g.bot_xtr = scale_row(5, clip_active(y_fused[5]));
    }

    // Boundary layer distributions. Latent layout per surface: theta encoding,
    // then H encoding, then ue/vinf, each N wide. Upper surface starts at 6,
    // lower surface at 6 + 3 * N.
    let decode_surface =
        |offset: usize,
         bl: &mut BoundaryLayerData,
         mut bl_grad: Option<&mut BoundaryLayerGradients>| {
            for i in 0..N {
                let i_theta = offset + i;
                let i_h = offset + N + i;
                let i_ue = offset + 2 * N + i;

                let ue = y_fused[i_ue];
                let pow_term = 10.0_f64.powf(y_fused[i_theta]);
                let theta = (pow_term - 0.1) / (ue.abs() * input.re);
                let h = 2.6 * y_fused[i_h].exp();

                bl.ue_over_vinf[i] = ue;
                bl.theta[i] = theta;
                bl.h[i] = h;

                if let Some(bl_grad) = bl_grad.as_deref_mut() {
                    let (row_theta, row_h, row_ue) =
                        (jac_row(i_theta), jac_row(i_h), jac_row(i_ue));

                    let dtheta_dlatent = LN_10 * pow_term / (ue.abs() * input.re);
                    let dtheta_due = -theta / ue;
                    for k in 0..NUM_INPUTS {
                        bl_grad.theta[i][k] =
                            dtheta_dlatent * row_theta[k] + dtheta_due * row_ue[k];
                        bl_grad.h[i][k] = h * row_h[k];
                        bl_grad.ue_over_vinf[i][k] = row_ue[k];
                    }
                    // theta also depends on Re directly through the decoding.
                    bl_grad.theta[i][idx::RE] += -theta / input.re;
                }
            }
        };

    decode_surface(
        6,
        &mut output.upper,
        gradients.as_deref_mut().map(|g| &mut g.upper),
    );
    decode_surface(
        6 + 3 * N,
        &mut output.lower,
        gradients.map(|g| &mut g.lower),
    );
    output
}

/// Resolves the weight directory: explicit argument, then `BORA_DATA_DIR`,
/// then the source-tree `data/` directory.
fn resolve_data_dir(explicit_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = explicit_dir {
        return Ok(dir.to_path_buf());
    }
    if let Some(dir) = std::env::var_os("BORA_DATA_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let source_dir = Path::new(SOURCE_DATA_DIR);
    if source_dir.exists() {
        return Ok(source_dir.to_path_buf());
    }
    Err(Error::DataDirNotFound)
}

impl NeuralFoil {
    /// Constructs an analysis model, loading weights from the default location.
    ///
    /// The weight directory is resolved in this order:
    ///   1. The `BORA_DATA_DIR` environment variable, if set.
    ///   2. The crate's source-tree `data/` directory, baked in at build time.
    ///
    /// # Errors
    ///
    /// Returns an error if the weight files cannot be found or parsed.
    pub fn new(size: ModelSize) -> Result<Self> {
        Self::load(size, None)
    }

    /// Constructs an analysis model, loading weights from `data_dir`, which must
    /// contain the HDF5 (`.h5`) weight files.
    ///
    /// # Errors
    ///
    /// Returns an error if the weight files cannot be found or parsed.
    pub fn with_data_dir(size: ModelSize, data_dir: impl AsRef<Path>) -> Result<Self> {
        Self::load(size, Some(data_dir.as_ref()))
    }

    fn load(size: ModelSize, data_dir: Option<&Path>) -> Result<Self> {
        let dir = resolve_data_dir(data_dir)?;
        let network = Network::load(dir.join(format!("nn-{size}.h5")))?;
        let distribution = InputDistribution::load(dir.join("scaled_input_distribution.h5"))?;
        Ok(Self {
            size,
            ..Self::from_parts(network, distribution)?
        })
    }

    /// Constructs an analysis model from pre-loaded components. The model size
    /// is reported as [`ModelSize::XLarge`].
    ///
    /// # Errors
    ///
    /// [`Error::DimensionMismatch`] unless the network maps 25 inputs to 198
    /// outputs and the distribution is 25-dimensional.
    pub fn from_parts(network: Network, distribution: InputDistribution) -> Result<Self> {
        if network.input_size() != NUM_LATENT_INPUTS {
            return Err(Error::DimensionMismatch("network input"));
        }
        if network.output_size() != NUM_LATENT_OUTPUTS {
            return Err(Error::DimensionMismatch("network output"));
        }
        if distribution.size() != NUM_LATENT_INPUTS {
            return Err(Error::DimensionMismatch("input distribution"));
        }
        Ok(Self {
            size: ModelSize::XLarge,
            network,
            distribution,
        })
    }

    /// Analyzes an airfoil at an operating condition.
    pub fn analyze(&self, input: &AeroInput) -> AeroOutput {
        let x = build_latent_input(input);

        // Flipped input: embeds the alpha-symmetry invariant of airfoil analysis.
        let mut x_flipped = [0.0; NUM_LATENT_INPUTS];
        apply_input_flip(|dest, src, sign| x_flipped[dest] = sign * x[src]);

        let mut y = [0.0; NUM_LATENT_OUTPUTS];
        let mut y_flipped = [0.0; NUM_LATENT_OUTPUTS];
        self.network.evaluate(&x, &mut y);
        self.network.evaluate(&x_flipped, &mut y_flipped);

        // Analysis-confidence correction: decays confidence with distance from
        // the training data distribution (this term was baked into training).
        let scale = 2.0 * NUM_LATENT_INPUTS as f64;
        y[0] -= self.distribution.squared_mahalanobis(&x) / scale;
        y_flipped[0] -= self.distribution.squared_mahalanobis(&x_flipped) / scale;

        // Un-flip the outputs of the flipped evaluation, then fuse by averaging.
        let mut y_fused = [0.0; NUM_LATENT_OUTPUTS];
        apply_output_unflip(|dest, src, sign| {
            y_fused[dest] = 0.5 * (y[dest] + sign * y_flipped[src]);
        });

        decode_outputs(&y_fused, input, None, None)
    }

    /// Analyzes an airfoil and computes analytic gradients of every output with
    /// respect to all [`NUM_INPUTS`] user inputs (CST weights, leading edge
    /// weight, TE thickness, alpha, Re, n_crit, xtr_upper, xtr_lower).
    pub fn analyze_with_gradients(&self, input: &AeroInput) -> AeroResult {
        let x = build_latent_input(input);
        let jx = build_seed_jacobian(input);

        let mut x_flipped = [0.0; NUM_LATENT_INPUTS];
        let mut jx_flipped = vec![0.0; NUM_LATENT_INPUTS * NUM_INPUTS];
        apply_input_flip(|dest, src, sign| {
            x_flipped[dest] = sign * x[src];
            for k in 0..NUM_INPUTS {
                jx_flipped[dest * NUM_INPUTS + k] = sign * jx[src * NUM_INPUTS + k];
            }
        });

        let mut y = [0.0; NUM_LATENT_OUTPUTS];
        let mut jy = vec![0.0; NUM_LATENT_OUTPUTS * NUM_INPUTS];
        let mut y_flipped = [0.0; NUM_LATENT_OUTPUTS];
        let mut jy_flipped = vec![0.0; NUM_LATENT_OUTPUTS * NUM_INPUTS];
        self.network
            .evaluate_with_jacobian(&x, &jx, NUM_INPUTS, &mut y, &mut jy);
        self.network.evaluate_with_jacobian(
            &x_flipped,
            &jx_flipped,
            NUM_INPUTS,
            &mut y_flipped,
            &mut jy_flipped,
        );

        // Analysis-confidence correction and its gradient, for both passes.
        let apply_mahalanobis =
            |x_pass: &[f64], jx_pass: &[f64], y_pass: &mut [f64], jy_pass: &mut [f64]| {
                let scale = 2.0 * NUM_LATENT_INPUTS as f64;
                y_pass[0] -= self.distribution.squared_mahalanobis(x_pass) / scale;
                let mut grad_latent = [0.0; NUM_LATENT_INPUTS];
                self.distribution
                    .squared_mahalanobis_gradient(x_pass, &mut grad_latent);
                for (k, jy_k) in jy_pass[..NUM_INPUTS].iter_mut().enumerate() {
                    let dot: f64 = (0..NUM_LATENT_INPUTS)
                        .map(|i| grad_latent[i] * jx_pass[i * NUM_INPUTS + k])
                        .sum();
                    *jy_k -= dot / scale;
                }
            };
        apply_mahalanobis(&x, &jx, &mut y, &mut jy);
        apply_mahalanobis(&x_flipped, &jx_flipped, &mut y_flipped, &mut jy_flipped);

        let mut y_fused = [0.0; NUM_LATENT_OUTPUTS];
        let mut jy_fused = vec![0.0; NUM_LATENT_OUTPUTS * NUM_INPUTS];
        apply_output_unflip(|dest, src, sign| {
            y_fused[dest] = 0.5 * (y[dest] + sign * y_flipped[src]);
            for k in 0..NUM_INPUTS {
                jy_fused[dest * NUM_INPUTS + k] =
                    0.5 * (jy[dest * NUM_INPUTS + k] + sign * jy_flipped[src * NUM_INPUTS + k]);
            }
        });

        let mut gradient = AeroGradients::default();
        let output = decode_outputs(&y_fused, input, Some(&jy_fused), Some(&mut gradient));
        AeroResult { output, gradient }
    }

    /// Returns the model size this instance was constructed with (or
    /// [`ModelSize::XLarge`] when constructed from pre-loaded components).
    pub fn model_size(&self) -> ModelSize {
        self.size
    }
}
