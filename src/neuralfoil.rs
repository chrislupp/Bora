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
    AeroGradients, AeroInput, AeroOutput, AeroResult, AeroScalarGradients, AeroScalarResult,
    AeroScalars, AeroTangent, BATCH_BLOCK_SIZE, BatchWorkspace, BoundaryLayerData,
    BoundaryLayerGradients, Error, Gradient, InputDistribution, ModelSize, NUM_BL_POINTS,
    NUM_CST_WEIGHTS_PER_SIDE, NUM_INPUTS, NUM_LATENT_INPUTS, NUM_LATENT_OUTPUTS, Network, Result,
};

const N: usize = NUM_BL_POINTS;
/// Number of scalar network outputs, which lead the latent output vector.
const NUM_SCALARS: usize = 6;
/// Divisor of the squared Mahalanobis distance in the analysis-confidence
/// correction.
const CONFIDENCE_SCALE: f64 = 2.0 * NUM_LATENT_INPUTS as f64;
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
fn build_seed_jacobian(input: &AeroInput) -> [f64; NUM_LATENT_INPUTS * NUM_INPUTS] {
    let mut jx = [0.0; NUM_LATENT_INPUTS * NUM_INPUTS];
    seed_jacobian_entries(input, |row, col, value| jx[row * NUM_INPUTS + col] = value);
    jx
}

/// Enumerates the nonzero entries of the seed Jacobian d(latent input)/d(user
/// inputs). The callback receives (latent row, user input column, value);
/// each (row, col) pair occurs once. Every latent row has a single nonzero
/// except that alpha feeds three rows (18, 19, 20).
fn seed_jacobian_entries(input: &AeroInput, mut set: impl FnMut(usize, usize, f64)) {
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
}

/// Builds the latent input tangent d(latent input)/d(user inputs) * direction,
/// i.e. the product of the seed Jacobian with `direction`, without forming
/// the matrix.
fn build_latent_tangent(input: &AeroInput, direction: &Gradient) -> [f64; NUM_LATENT_INPUTS] {
    let mut dx = [0.0; NUM_LATENT_INPUTS];
    seed_jacobian_entries(input, |row, col, value| dx[row] += value * direction[col]);
    dx
}

/// Applies the alpha-symmetry flip to a latent input vector.
fn flip_latent_input(x: &[f64; NUM_LATENT_INPUTS]) -> [f64; NUM_LATENT_INPUTS] {
    let mut x_flipped = [0.0; NUM_LATENT_INPUTS];
    apply_input_flip(|dest, src, sign| x_flipped[dest] = sign * x[src]);
    x_flipped
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
    apply_scalar_output_unflip(&mut apply);
    for i in 0..2 * N {
        apply(6 + i, 6 + 3 * N + i, 1.0); // upper theta, H <- lower theta, H
        apply(6 + 3 * N + i, 6 + i, 1.0); // lower theta, H <- upper theta, H
    }
    for i in 0..N {
        apply(6 + 2 * N + i, 6 + 5 * N + i, -1.0); // upper ue <- -lower ue
        apply(6 + 5 * N + i, 6 + 2 * N + i, -1.0); // lower ue <- -upper ue
    }
}

/// The part of [`apply_output_unflip`] covering the [`NUM_SCALARS`] scalar
/// outputs; every (dest, source) pair lies below `NUM_SCALARS`.
fn apply_scalar_output_unflip(mut apply: impl FnMut(usize, usize, f64)) {
    apply(0, 0, 1.0); // analysis confidence
    apply(1, 1, -1.0); // CL
    apply(2, 2, 1.0); // CD
    apply(3, 3, -1.0); // CM
    apply(4, 5, 1.0); // Top_Xtr <- Bot_Xtr
    apply(5, 4, 1.0); // Bot_Xtr <- Top_Xtr
}

/// Decodes the first [`NUM_SCALARS`] fused latent outputs into the scalar
/// outputs, with the same scaling/squashing as upstream NeuralFoil.
fn decode_scalars(y_fused: &[f64]) -> AeroScalars {
    AeroScalars {
        analysis_confidence: stable_sigmoid(y_fused[0]),
        cl: y_fused[1] / 2.0,
        cd: ((y_fused[2] - 2.0) * 2.0).exp(),
        cm: y_fused[3] / 20.0,
        top_xtr: y_fused[4].clamp(0.0, 1.0),
        bot_xtr: y_fused[5].clamp(0.0, 1.0),
    }
}

/// Propagates the gradients of the fused latent scalar outputs (`jac_row(i)`
/// for latent output `i`, w.r.t. the user inputs) through
/// [`decode_scalars`]; `output` is its result for `y_fused`.
fn decode_scalar_gradients<'a>(
    y_fused: &[f64],
    output: &AeroScalars,
    jac_row: impl Fn(usize) -> &'a [f64],
) -> AeroScalarGradients {
    let scale_row = |i: usize, factor: f64| -> Gradient {
        let row = jac_row(i);
        std::array::from_fn(|k| factor * row[k])
    };
    let conf = output.analysis_confidence;
    // Clipped transition locations: zero gradient where the clip is active.
    let clip_active = |v: f64| if v > 0.0 && v < 1.0 { 1.0 } else { 0.0 };
    AeroScalarGradients {
        analysis_confidence: scale_row(0, conf * (1.0 - conf)),
        cl: scale_row(1, 0.5),
        cd: scale_row(2, 2.0 * output.cd),
        cm: scale_row(3, 1.0 / 20.0),
        top_xtr: scale_row(4, clip_active(y_fused[4])),
        bot_xtr: scale_row(5, clip_active(y_fused[5])),
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
    let scalars = decode_scalars(y_fused);
    let mut output = AeroOutput {
        analysis_confidence: scalars.analysis_confidence,
        cl: scalars.cl,
        cd: scalars.cd,
        cm: scalars.cm,
        top_xtr: scalars.top_xtr,
        bot_xtr: scalars.bot_xtr,
        ..AeroOutput::default()
    };

    let jac_row =
        |i: usize| &jy_fused.expect("gradients require a Jacobian")[i * NUM_INPUTS..][..NUM_INPUTS];

    if let Some(g) = gradients.as_deref_mut() {
        let sg = decode_scalar_gradients(y_fused, &scalars, jac_row);
        g.analysis_confidence = sg.analysis_confidence;
        g.cl = sg.cl;
        g.cd = sg.cd;
        g.cm = sg.cm;
        g.top_xtr = sg.top_xtr;
        g.bot_xtr = sg.bot_xtr;
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

/// Propagates the fused latent output tangent `dy_fused` through the decoding
/// of [`decode_outputs`], given the decoded `output` at the same point. Uses
/// the same derivative formulas as the gradient path of [`decode_outputs`],
/// contracted with one direction (`d_re` is the direction's Re component, for
/// theta's direct dependence on Re).
fn decode_tangent(
    y_fused: &[f64],
    dy_fused: &[f64],
    input: &AeroInput,
    d_re: f64,
    output: &AeroOutput,
) -> AeroOutput {
    let conf = output.analysis_confidence;
    // Clipped transition locations: zero derivative where the clip is active.
    let clip_active = |v: f64| if v > 0.0 && v < 1.0 { 1.0 } else { 0.0 };
    let mut tangent = AeroOutput {
        analysis_confidence: conf * (1.0 - conf) * dy_fused[0],
        cl: 0.5 * dy_fused[1],
        cd: 2.0 * output.cd * dy_fused[2],
        cm: 1.0 / 20.0 * dy_fused[3],
        top_xtr: clip_active(y_fused[4]) * dy_fused[4],
        bot_xtr: clip_active(y_fused[5]) * dy_fused[5],
        ..AeroOutput::default()
    };

    let decode_surface = |offset: usize, bl: &BoundaryLayerData, bl_dot: &mut BoundaryLayerData| {
        for i in 0..N {
            let i_theta = offset + i;
            let i_h = offset + N + i;
            let i_ue = offset + 2 * N + i;

            let ue = y_fused[i_ue];
            let pow_term = 10.0_f64.powf(y_fused[i_theta]);
            let theta = bl.theta[i];
            let dtheta_dlatent = LN_10 * pow_term / (ue.abs() * input.re);
            let dtheta_due = -theta / ue;
            // theta also depends on Re directly through the decoding.
            bl_dot.theta[i] = dtheta_dlatent * dy_fused[i_theta]
                + dtheta_due * dy_fused[i_ue]
                + (-theta / input.re) * d_re;
            bl_dot.h[i] = bl.h[i] * dy_fused[i_h];
            bl_dot.ue_over_vinf[i] = dy_fused[i_ue];
        }
    };
    decode_surface(6, &output.upper, &mut tangent.upper);
    decode_surface(6 + 3 * N, &output.lower, &mut tangent.lower);
    tangent
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
        let x_flipped = flip_latent_input(&x);

        // Both inputs share one traversal of the weights; each goes through the
        // same operations as a single evaluation, which keeps the fused result
        // exactly symmetric.
        let mut y = [0.0; NUM_LATENT_OUTPUTS];
        let mut y_flipped = [0.0; NUM_LATENT_OUTPUTS];
        self.network
            .evaluate_pair([&x, &x_flipped], [&mut y, &mut y_flipped]);

        // Analysis-confidence correction: decays confidence with distance from
        // the training data distribution (this term was baked into training).
        y[0] -= self.distribution.squared_mahalanobis(&x) / CONFIDENCE_SCALE;
        y_flipped[0] -= self.distribution.squared_mahalanobis(&x_flipped) / CONFIDENCE_SCALE;

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
        let mut jx_flipped = [0.0; NUM_LATENT_INPUTS * NUM_INPUTS];
        apply_input_flip(|dest, src, sign| {
            x_flipped[dest] = sign * x[src];
            for k in 0..NUM_INPUTS {
                jx_flipped[dest * NUM_INPUTS + k] = sign * jx[src * NUM_INPUTS + k];
            }
        });

        // Two single-input passes: batching both inputs into one traversal of
        // the weights measured no faster for the small models and ~50% slower
        // for xxxlarge, where two inputs' padded Jacobians (2 x 98 KB per
        // layer) overflow the L1 data cache.
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
                let scale = CONFIDENCE_SCALE;
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

    /// Analyzes an airfoil, computing only the scalar outputs (analysis
    /// confidence, CL, CD, CM and the transition locations).
    ///
    /// The results are bitwise identical to the corresponding fields of
    /// [`analyze`](Self::analyze), which additionally evaluates and decodes
    /// the 192 boundary layer outputs of the network; skipping them makes this
    /// ~1.1x (xxxlarge) to ~2.2x (xxsmall) faster.
    pub fn analyze_scalars(&self, input: &AeroInput) -> AeroScalars {
        let x = build_latent_input(input);
        let x_flipped = flip_latent_input(&x);

        let mut y = [0.0; NUM_SCALARS];
        let mut y_flipped = [0.0; NUM_SCALARS];
        self.network
            .evaluate_first_pair([&x, &x_flipped], NUM_SCALARS, [&mut y, &mut y_flipped]);

        y[0] -= self.distribution.squared_mahalanobis(&x) / CONFIDENCE_SCALE;
        y_flipped[0] -= self.distribution.squared_mahalanobis(&x_flipped) / CONFIDENCE_SCALE;

        let mut y_fused = [0.0; NUM_SCALARS];
        apply_scalar_output_unflip(|dest, src, sign| {
            y_fused[dest] = 0.5 * (y[dest] + sign * y_flipped[src]);
        });
        decode_scalars(&y_fused)
    }

    /// Analyzes an airfoil and computes analytic gradients of the scalar
    /// outputs (analysis confidence, CL, CD, CM and the transition locations)
    /// with respect to all [`NUM_INPUTS`] user inputs.
    ///
    /// Use this instead of [`analyze_with_gradients`](Self::analyze_with_gradients)
    /// when the boundary layer gradients are not needed, e.g. in optimization
    /// with aerodynamic-coefficient objectives and constraints: it
    /// differentiates the network in reverse mode (one adjoint per scalar
    /// output) rather than forward mode (one seed per input), which is ~4x
    /// (xxxlarge) to ~8x (xxsmall) faster.
    ///
    /// The outputs are bitwise identical to [`analyze`](Self::analyze) and the
    /// gradients agree with those of `analyze_with_gradients` up to rounding
    /// (the sums are formed in a different order).
    pub fn analyze_scalars_with_gradients(&self, input: &AeroInput) -> AeroScalarResult {
        const L: usize = NUM_LATENT_INPUTS;
        let x = build_latent_input(input);
        let x_flipped = flip_latent_input(&x);

        // Values and gradients w.r.t. the latent inputs, row-major
        // `NUM_SCALARS x NUM_LATENT_INPUTS`, for both passes.
        let mut y = [0.0; NUM_SCALARS];
        let mut y_flipped = [0.0; NUM_SCALARS];
        let mut g = [0.0; NUM_SCALARS * L];
        let mut g_flipped = [0.0; NUM_SCALARS * L];
        self.network.evaluate_first_with_gradients_pair(
            [&x, &x_flipped],
            NUM_SCALARS,
            [&mut y, &mut y_flipped],
            [&mut g, &mut g_flipped],
        );

        // Analysis-confidence correction and its latent gradient.
        let apply_mahalanobis = |x_pass: &[f64], y_pass: &mut [f64], g_pass: &mut [f64]| {
            let mut grad_latent = [0.0; L];
            let distance = self
                .distribution
                .squared_mahalanobis_with_gradient(x_pass, &mut grad_latent);
            y_pass[0] -= distance / CONFIDENCE_SCALE;
            for (g, d) in g_pass[..L].iter_mut().zip(grad_latent) {
                *g -= d / CONFIDENCE_SCALE;
            }
        };
        apply_mahalanobis(&x, &mut y, &mut g);
        apply_mahalanobis(&x_flipped, &mut y_flipped, &mut g_flipped);

        // Fuse in the latent space of the original input: the flipped pass
        // sees x_flipped = F x, so its gradients map back through F^T.
        let mut y_fused = [0.0; NUM_SCALARS];
        let mut g_fused = [[0.0; L]; NUM_SCALARS];
        apply_scalar_output_unflip(|dest, src, sign| {
            y_fused[dest] = 0.5 * (y[dest] + sign * y_flipped[src]);
            let row = &mut g_fused[dest];
            for (r, g) in row.iter_mut().zip(&g[dest * L..][..L]) {
                *r = 0.5 * g;
            }
            let g_src = &g_flipped[src * L..][..L];
            apply_input_flip(|i_flip, i, flip_sign| {
                row[i] += 0.5 * sign * flip_sign * g_src[i_flip];
            });
        });

        // Chain to the user inputs through the sparse seed Jacobian.
        let mut jy_fused = [[0.0; NUM_INPUTS]; NUM_SCALARS];
        seed_jacobian_entries(input, |row, col, value| {
            for (jy, g) in jy_fused.iter_mut().zip(&g_fused) {
                jy[col] += g[row] * value;
            }
        });

        let output = decode_scalars(&y_fused);
        let gradient = decode_scalar_gradients(&y_fused, &output, |i| &jy_fused[i]);
        AeroScalarResult { output, gradient }
    }

    /// Analyzes an airfoil and computes the directional derivative of every
    /// output along `direction`, a vector in the same [`NUM_INPUTS`]-entry
    /// user-input space as [`Gradient`] (ordering given by
    /// [`input_index`](crate::input_index)).
    ///
    /// This is a forward-mode Jacobian-vector product: for every output X,
    /// `tangent.X == sum_k gradient.X[k] * direction[k]`, where `gradient` is
    /// what [`analyze_with_gradients`](Self::analyze_with_gradients) returns
    /// (up to floating-point summation order). A unit vector at
    /// [`input_index::ALPHA`](crate::input_index::ALPHA) gives d/d(alpha) of
    /// every output. It propagates a single direction instead of all
    /// [`NUM_INPUTS`], so it costs about two [`analyze`](Self::analyze) calls
    /// (1.8-2.0x on Apple M1 Pro) versus 11-13 for `analyze_with_gradients`.
    /// `output` is
    /// bitwise identical to [`analyze`](Self::analyze).
    pub fn analyze_with_tangent(&self, input: &AeroInput, direction: &Gradient) -> AeroTangent {
        let x = build_latent_input(input);
        let dx = build_latent_tangent(input, direction);
        // The flip is linear, so it maps tangents like values.
        let x_flipped = flip_latent_input(&x);
        let dx_flipped = flip_latent_input(&dx);

        // Two value+tangent passes. Sharing one traversal of the weights
        // between them (two `dot2` calls per weight row) measured 1.1-1.2x slower
        // on Apple M1 Pro for every model size, and a single 4-vector dot
        // (64 accumulators, which spill) ~3x slower.
        let mut y = [0.0; NUM_LATENT_OUTPUTS];
        let mut dy = [0.0; NUM_LATENT_OUTPUTS];
        let mut y_flipped = [0.0; NUM_LATENT_OUTPUTS];
        let mut dy_flipped = [0.0; NUM_LATENT_OUTPUTS];
        self.network.evaluate_with_tangent(&x, &dx, &mut y, &mut dy);
        self.network.evaluate_with_tangent(
            &x_flipped,
            &dx_flipped,
            &mut y_flipped,
            &mut dy_flipped,
        );

        // Analysis-confidence correction and its directional derivative, for
        // both passes.
        let apply_mahalanobis =
            |x_pass: &[f64], dx_pass: &[f64], y_pass: &mut [f64], dy_pass: &mut [f64]| {
                let mut grad_latent = [0.0; NUM_LATENT_INPUTS];
                let distance = self
                    .distribution
                    .squared_mahalanobis_with_gradient(x_pass, &mut grad_latent);
                y_pass[0] -= distance / CONFIDENCE_SCALE;
                let dot: f64 = grad_latent.iter().zip(dx_pass).map(|(g, d)| g * d).sum();
                dy_pass[0] -= dot / CONFIDENCE_SCALE;
            };
        apply_mahalanobis(&x, &dx, &mut y, &mut dy);
        apply_mahalanobis(&x_flipped, &dx_flipped, &mut y_flipped, &mut dy_flipped);

        // Un-flip the outputs of the flipped evaluation, then fuse by averaging.
        let mut y_fused = [0.0; NUM_LATENT_OUTPUTS];
        let mut dy_fused = [0.0; NUM_LATENT_OUTPUTS];
        apply_output_unflip(|dest, src, sign| {
            y_fused[dest] = 0.5 * (y[dest] + sign * y_flipped[src]);
            dy_fused[dest] = 0.5 * (dy[dest] + sign * dy_flipped[src]);
        });

        let output = decode_outputs(&y_fused, input, None, None);
        let tangent = decode_tangent(&y_fused, &dy_fused, input, direction[idx::RE], &output);
        AeroTangent { output, tangent }
    }

    /// Analyzes a batch of operating points (and/or airfoils) at once.
    ///
    /// Returns one output per input, in order: `analyze_batch(inputs)[i]` equals
    /// [`analyze(&inputs[i])`](Self::analyze) to within a few ulps of the
    /// network outputs (relative differences around 1e-14), but not bit for
    /// bit: the batched network kernel ([`Network::evaluate_batch`]) sums in a
    /// different order and evaluates the activation function with a
    /// vectorizable `exp`. A point's result does not depend on its position in
    /// the batch or on the other points beyond that rounding level.
    ///
    /// Points are processed four at a time (each with its flipped image, so
    /// one network block). For a 64-point batch this is 1.9x (xxsmall) to
    /// 2.4x (xxxlarge) faster per point than calling [`analyze`](Self::analyze)
    /// in a loop on Apple M1 Pro; a final partial chunk gains less, and a
    /// single point costs about one `analyze` call. Any length works,
    /// including zero.
    ///
    /// Allocates the returned vector and a scratch workspace; use
    /// [`analyze_batch_into`](Self::analyze_batch_into) to reuse both across
    /// calls.
    pub fn analyze_batch(&self, inputs: &[AeroInput]) -> Vec<AeroOutput> {
        let mut outputs = vec![AeroOutput::default(); inputs.len()];
        self.analyze_batch_into(inputs, &mut outputs, &mut BatchWorkspace::default());
        outputs
    }

    /// Like [`analyze_batch`](Self::analyze_batch), but writes the results to
    /// `outputs` (`outputs[i]` for `inputs[i]`) and takes its scratch space from
    /// `workspace`, so repeated calls with a kept workspace do not allocate.
    ///
    /// # Panics
    ///
    /// If `outputs` and `inputs` differ in length.
    pub fn analyze_batch_into(
        &self,
        inputs: &[AeroInput],
        outputs: &mut [AeroOutput],
        workspace: &mut BatchWorkspace,
    ) {
        self.batch_into(
            inputs,
            outputs,
            workspace,
            NUM_LATENT_OUTPUTS,
            |y_fused, input| decode_outputs(y_fused, input, None, None),
        );
    }

    /// Analyzes a batch of operating points (and/or airfoils) at once,
    /// computing only the scalar outputs: the batched counterpart of
    /// [`analyze_scalars`](Self::analyze_scalars), which skips the 192
    /// boundary layer rows of the network's output layer.
    ///
    /// `analyze_scalars_batch(inputs)[i]` equals
    /// [`analyze_scalars(&inputs[i])`](Self::analyze_scalars) to within a few
    /// ulps (see [`analyze_batch`](Self::analyze_batch) for why it is not bit
    /// for bit), and is bitwise identical to the scalar fields of
    /// `analyze_batch(inputs)[i]`. For a 64-point batch it is 2.1x
    /// (xxsmall) to 2.5x (xxlarge, xxxlarge) faster per point than a loop over
    /// `analyze_scalars` on Apple M1 Pro.
    ///
    /// Allocates the returned vector and a scratch workspace; use
    /// [`analyze_scalars_batch_into`](Self::analyze_scalars_batch_into) to
    /// reuse both across calls.
    pub fn analyze_scalars_batch(&self, inputs: &[AeroInput]) -> Vec<AeroScalars> {
        let mut outputs = vec![AeroScalars::default(); inputs.len()];
        self.analyze_scalars_batch_into(inputs, &mut outputs, &mut BatchWorkspace::default());
        outputs
    }

    /// Like [`analyze_scalars_batch`](Self::analyze_scalars_batch), but writes
    /// the results to `outputs` (`outputs[i]` for `inputs[i]`) and takes its
    /// scratch space from `workspace`, so repeated calls with a kept workspace
    /// do not allocate.
    ///
    /// # Panics
    ///
    /// If `outputs` and `inputs` differ in length.
    pub fn analyze_scalars_batch_into(
        &self,
        inputs: &[AeroInput],
        outputs: &mut [AeroScalars],
        workspace: &mut BatchWorkspace,
    ) {
        self.batch_into(inputs, outputs, workspace, NUM_SCALARS, |y_fused, _| {
            decode_scalars(y_fused)
        });
    }

    /// Shared body of the batched analysis calls: evaluates the first `n_out`
    /// latent outputs (all of them, or the [`NUM_SCALARS`] scalars) of every
    /// input and its alpha-flipped image with the batched kernel, applies the
    /// confidence correction, fuses, and decodes each point with `decode`.
    fn batch_into<T>(
        &self,
        inputs: &[AeroInput],
        outputs: &mut [T],
        workspace: &mut BatchWorkspace,
        n_out: usize,
        decode: impl Fn(&[f64], &AeroInput) -> T,
    ) {
        // Each point needs two network passes (the input and its alpha-flipped
        // image), evaluated as two columns of the same batch: a chunk of
        // `CHUNK` points fills one `BATCH_BLOCK_SIZE`-column block, with the
        // points in columns 0..m and their flipped images in columns m..2m.
        const CHUNK: usize = BATCH_BLOCK_SIZE / 2;
        const N_IN: usize = NUM_LATENT_INPUTS;
        const N_OUT: usize = NUM_LATENT_OUTPUTS;
        assert_eq!(
            inputs.len(),
            outputs.len(),
            "batched analysis: inputs and outputs differ in length"
        );
        let unflip: fn(&mut dyn FnMut(usize, usize, f64)) = if n_out == NUM_SCALARS {
            |apply| apply_scalar_output_unflip(apply)
        } else {
            |apply| apply_output_unflip(apply)
        };

        let mut x_block = [0.0; N_IN * 2 * CHUNK];
        let mut y_block = [0.0; N_OUT * 2 * CHUNK];
        let mut corrections = [0.0; 2 * CHUNK];
        for (inputs, outputs) in inputs.chunks(CHUNK).zip(outputs.chunks_mut(CHUNK)) {
            let m = inputs.len();
            let n_cols = 2 * m;

            for (k, input) in inputs.iter().enumerate() {
                let x = build_latent_input(input);
                let x_flipped = flip_latent_input(&x);
                for j in 0..N_IN {
                    x_block[j * n_cols + k] = x[j];
                    x_block[j * n_cols + m + k] = x_flipped[j];
                }
                // Analysis-confidence correction (see `analyze`).
                let distance = |x: &[f64]| self.distribution.squared_mahalanobis_unbuffered(x);
                corrections[k] = distance(&x) / CONFIDENCE_SCALE;
                corrections[m + k] = distance(&x_flipped) / CONFIDENCE_SCALE;
            }

            self.network
                .evaluate_first_batch(&x_block, n_out, &mut y_block, n_cols, workspace);

            for (k, (input, output)) in inputs.iter().zip(outputs.iter_mut()).enumerate() {
                let y = |i: usize| y_block[i * n_cols + k];
                let y_flipped = |i: usize| y_block[i * n_cols + m + k];
                let mut y_fused = [0.0; N_OUT];
                unflip(&mut |dest, src, sign| {
                    let mut value = y(dest);
                    let mut flipped = y_flipped(src);
                    if dest == 0 {
                        value -= corrections[k];
                    }
                    if src == 0 {
                        flipped -= corrections[m + k];
                    }
                    y_fused[dest] = 0.5 * (value + sign * flipped);
                });
                *output = decode(&y_fused[..n_out], input);
            }
        }
    }

    /// Returns the model size this instance was constructed with (or
    /// [`ModelSize::XLarge`] when constructed from pre-loaded components).
    pub fn model_size(&self) -> ModelSize {
        self.size
    }
}
