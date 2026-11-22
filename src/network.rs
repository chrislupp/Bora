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

//! Neural network weights, evaluation, and training distribution statistics.

use std::path::Path;

use hdf5_pure::File;

use crate::{Error, Result};

/// A single dense layer: y = W x + b, with W stored row-major.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Layer {
    /// Output dimension.
    pub rows: usize,
    /// Input dimension.
    pub cols: usize,
    /// `rows * cols` entries, row-major.
    pub weights: Vec<f64>,
    /// `rows` entries.
    pub biases: Vec<f64>,
}

/// A multilayer perceptron with swish activations between layers (no
/// activation after the final layer), matching NeuralFoil's architecture.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Network {
    layers: Vec<Layer>,
}

/// Statistics of the training data distribution in the scaled input space,
/// used for the Mahalanobis-distance analysis-confidence correction.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InputDistribution {
    /// Mean of the scaled inputs.
    mean: Vec<f64>,
    /// Inverse covariance, row-major `size() x size()`.
    inv_cov: Vec<f64>,
}

/// Number of points (columns) the batched kernel
/// ([`Network::evaluate_batch`]) pushes through the weight matrices together.
///
/// Each output row is accumulated for a block of this many points at once, four
/// rows at a time, so every weight loaded from memory feeds `4 * BATCH_BLOCK_SIZE`
/// independent fused multiply-add chains (the single-point `dot` runs four).
///
/// Chosen by measurement on an Apple M1 Pro (single thread, 4096 points, minimum
/// over repetitions, speedup of the per-point network time over a loop over
/// [`Network::evaluate`]); `B x R` is block size x output rows per pass:
///
/// | model    | B4xR4 | B8xR4 | B16xR4 | B32xR4 | B16xR2 | B32xR1 |
/// |----------|-------|-------|--------|--------|--------|--------|
/// | xxsmall  | 3.1x  | 4.2x  | 3.2x   | 2.5x   | 4.2x   | 3.0x   |
/// | medium   | 3.4x  | 4.9x  | 3.9x   | 3.2x   | 5.0x   | 3.7x   |
/// | xlarge   | 3.2x  | 5.1x  | 4.0x   | 3.3x   | 5.2x   | 3.9x   |
/// | xxxlarge | 3.5x  | 6.0x  | 4.6x   | 3.6x   | 6.4x   | 3.8x   |
///
/// Eight points by four rows (32 accumulators) ties with 16 x 2 within noise
/// for every model size and is the smaller block, which keeps the padding waste
/// of a partial final block low. The vectorizable activation matters as much as
/// the blocking: the same 8 x 4 kernel with the libm `exp` reaches only 1.6x
/// (xxsmall) to 2.4x (xxxlarge).
pub const BATCH_BLOCK_SIZE: usize = 8;

/// A partial final block with fewer than this many points is evaluated one point
/// at a time (with the single-point kernel) instead of being padded to a full
/// [`BATCH_BLOCK_SIZE`] block. One padded block costs about as much as two to
/// three single-point evaluations.
const MIN_PADDED_BLOCK_POINTS: usize = 3;

/// Reusable scratch space for [`Network::evaluate_batch`] and
/// [`NeuralFoil::analyze_batch_into`](crate::NeuralFoil::analyze_batch_into).
///
/// Holds the hidden-layer activations of one [`BATCH_BLOCK_SIZE`]-point block.
/// It grows on first use to fit the widest layer of the network it is used with
/// and is never shrunk, so a caller that evaluates batches repeatedly (e.g. once
/// per solver iteration) can keep one workspace and avoid re-allocating it. A
/// workspace may be used with any network.
#[derive(Debug, Clone, Default)]
pub struct BatchWorkspace {
    /// Block activations, neuron-major: `current[neuron][point]`.
    current: Vec<[f64; BATCH_BLOCK_SIZE]>,
    /// Block activations of the layer being computed.
    next: Vec<[f64; BATCH_BLOCK_SIZE]>,
}

fn open_file(path: &Path, what: &'static str) -> Result<File> {
    File::open(path).map_err(|source| Error::Open {
        what,
        path: path.to_path_buf(),
        source,
    })
}

fn open_bytes(data: &[u8], what: &'static str) -> Result<File> {
    File::from_bytes(data.to_vec())
        .map_err(|err| Error::Format(format!("not a valid {what} HDF5 image: {err}")))
}

/// Reads an entire dataset as f64 (converting from the stored type) and
/// returns it with its extents.
fn read_f64_dataset(file: &File, path: &str) -> Result<(Vec<f64>, Vec<usize>)> {
    let dataset = file
        .dataset(path)
        .map_err(|_| Error::Format(format!("Bora data file is missing dataset '{path}'")))?;
    let shape = dataset.shape().map_err(|err| {
        Error::Format(format!("cannot read the shape of dataset '{path}': {err}"))
    })?;
    let values = dataset
        .read_f64()
        .map_err(|err| Error::Format(format!("failed to read dataset '{path}': {err}")))?;
    Ok((values, shape.into_iter().map(|d| d as usize).collect()))
}

/// Reads and validates the `/layers/<i>/{weights,biases}` network layout.
fn read_layers(file: &File) -> Result<Vec<Layer>> {
    let n_layers = file
        .group("layers")
        .and_then(|group| group.groups())
        .map_err(|_| Error::Format("Bora network file has no 'layers' group".into()))?
        .len();
    if n_layers == 0 || n_layers > 1000 {
        return Err(Error::Format(format!(
            "Bora network file has an implausible layer count: {n_layers}"
        )));
    }

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let (weights, dims) = read_f64_dataset(file, &format!("layers/{i}/weights"))?;
        if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
            return Err(Error::Format(format!(
                "layer {i} 'weights' must be a non-empty matrix"
            )));
        }
        let (biases, bias_dims) = read_f64_dataset(file, &format!("layers/{i}/biases"))?;
        if bias_dims != [dims[0]] {
            return Err(Error::Format(format!(
                "layer {i} 'biases' must be a vector with one entry per weight row"
            )));
        }
        layers.push(Layer {
            rows: dims[0],
            cols: dims[1],
            weights,
            biases,
        });
    }

    if layers.windows(2).any(|pair| pair[1].cols != pair[0].rows) {
        return Err(Error::Format(
            "Bora network file has inconsistent layer dimensions".into(),
        ));
    }
    Ok(layers)
}

/// Reads and validates the `/mean` and `/inv_cov` distribution layout.
fn read_distribution(file: &File) -> Result<InputDistribution> {
    let (mean, dims) = read_f64_dataset(file, "mean")?;
    if dims.len() != 1 || dims[0] == 0 || dims[0] > 10000 {
        return Err(Error::Format(
            "Bora distribution file has an implausible 'mean' dimension".into(),
        ));
    }
    let n = mean.len();
    let (inv_cov, dims) = read_f64_dataset(file, "inv_cov")?;
    if dims != [n, n] {
        return Err(Error::Format(
            "Bora distribution file 'inv_cov' must be a square matrix matching 'mean'".into(),
        ));
    }
    Ok(InputDistribution { mean, inv_cov })
}

/// Computes `a * b + c`, fused into a single FMA instruction on targets with
/// hardware support. Elsewhere `f64::mul_add` would fall back to a slow
/// software routine, so the plain expression is used instead.
#[inline(always)]
fn mul_add(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
    {
        a * b + c
    }
}

/// Dot product with four independent accumulators: breaks the floating-point
/// dependency chain so the loop is throughput-bound instead of latency-bound
/// (the compiler may not reassociate floating-point sums itself). The fused
/// multiply-adds are worth ~25% of `Network::evaluate` on aarch64.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut s = [0.0; 4];
    let (a_chunks, a_rest) = a.as_chunks::<4>();
    let (b_chunks, b_rest) = b.as_chunks::<4>();
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        s[0] = mul_add(ac[0], bc[0], s[0]);
        s[1] = mul_add(ac[1], bc[1], s[1]);
        s[2] = mul_add(ac[2], bc[2], s[2]);
        s[3] = mul_add(ac[3], bc[3], s[3]);
    }
    for (x, y) in a_rest.iter().zip(b_rest) {
        s[0] += x * y;
    }
    (s[0] + s[1]) + (s[2] + s[3])
}

/// Computes `(dot(a, b), dot(a, c))` in one pass over `a`, so each weight row
/// is loaded once for both the value and the tangent product. The first result
/// uses exactly the accumulation order of [`dot`], so values are bit-identical
/// to [`Network::evaluate`].
#[inline]
fn dot2(a: &[f64], b: &[f64], c: &[f64]) -> (f64, f64) {
    let n = a.len().min(b.len()).min(c.len());
    let (a, b, c) = (&a[..n], &b[..n], &c[..n]);
    let mut s = [0.0; 4];
    let mut t = [0.0; 4];
    let (a_chunks, a_rest) = a.as_chunks::<4>();
    let (b_chunks, b_rest) = b.as_chunks::<4>();
    let (c_chunks, c_rest) = c.as_chunks::<4>();
    for ((ac, bc), cc) in a_chunks.iter().zip(b_chunks).zip(c_chunks) {
        s[0] = mul_add(ac[0], bc[0], s[0]);
        s[1] = mul_add(ac[1], bc[1], s[1]);
        s[2] = mul_add(ac[2], bc[2], s[2]);
        s[3] = mul_add(ac[3], bc[3], s[3]);
        t[0] = mul_add(ac[0], cc[0], t[0]);
        t[1] = mul_add(ac[1], cc[1], t[1]);
        t[2] = mul_add(ac[2], cc[2], t[2]);
        t[3] = mul_add(ac[3], cc[3], t[3]);
    }
    for ((x, y), z) in a_rest.iter().zip(b_rest).zip(c_rest) {
        s[0] += x * y;
        t[0] += x * z;
    }
    ((s[0] + s[1]) + (s[2] + s[3]), (t[0] + t[1]) + (t[2] + t[3]))
}

/// swish(x) = x * sigmoid(x), the network's hidden-layer activation.
#[inline]
fn swish(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// d/dx swish(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
#[inline]
fn swish_derivative(x: f64) -> f64 {
    let sig = 1.0 / (1.0 + (-x).exp());
    sig * (1.0 + x * (1.0 - sig))
}

/// Branch-free `exp(x)` that the compiler can vectorize, unlike `f64::exp`
/// (a libm call). Accurate to about one ulp (max. relative error 2.3e-16 on
/// [-700, 700]); the argument is clamped to [-708, 709], so it never returns
/// zero or infinity, which is harmless where it is used (`1 + exp(-x)`) but makes
/// it unsuitable as a general `exp`.
///
/// Uses `exp(x) = 2^n exp(r)` with `n = round(x / ln 2)` and
/// `|r| <= ln(2) / 2`, a degree-13 Taylor polynomial for `exp(r)`, and builds
/// `2^n` from its bit pattern. Only `mul_add` and integer bit operations are
/// used, so it is portable and fast on non-FMA targets too.
#[inline(always)]
fn exp_vectorizable(x: f64) -> f64 {
    // Adding 1.5 * 2^52 rounds to the nearest integer and leaves that integer
    // in the low mantissa bits.
    const ROUNDER: f64 = 6_755_399_441_055_744.0;
    const LN_2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN_2_LO: f64 = 1.908_214_929_270_587_7e-10;
    // 1 / k! for k = 13 down to 0.
    const INV_FACTORIALS: [f64; 14] = [
        1.0 / 6_227_020_800.0,
        1.0 / 479_001_600.0,
        1.0 / 39_916_800.0,
        1.0 / 3_628_800.0,
        1.0 / 362_880.0,
        1.0 / 40_320.0,
        1.0 / 5_040.0,
        1.0 / 720.0,
        1.0 / 120.0,
        1.0 / 24.0,
        1.0 / 6.0,
        1.0 / 2.0,
        1.0,
        1.0,
    ];
    let x = x.clamp(-708.0, 709.0);
    let shifted = mul_add(x, std::f64::consts::LOG2_E, ROUNDER);
    let n = shifted - ROUNDER;
    let r = mul_add(n, -LN_2_LO, mul_add(n, -LN_2_HI, x));
    let mut p = INV_FACTORIALS[0];
    for &c in &INV_FACTORIALS[1..] {
        p = mul_add(p, r, c);
    }
    // The low bits of `shifted` hold n in two's complement; (n + 1023) << 52
    // is the bit pattern of 2^n for -1022 <= n <= 1023.
    let scale = f64::from_bits(shifted.to_bits().wrapping_add(1023) << 52);
    p * scale
}

/// Applies swish to a block of values with [`exp_vectorizable`]; agrees with
/// [`swish`] to about one ulp.
#[inline(always)]
fn swish_block<const B: usize>(values: &mut [f64; B]) {
    for v in values.iter_mut() {
        *v /= 1.0 + exp_vectorizable(-*v);
    }
}

/// One dense layer applied to a block of `B` points stored neuron-major:
/// `output[i][p] = act(bias[i] + sum_j weight[i][j] * input[j][p])`, with swish
/// as `act` if `activate`, else the identity.
///
/// Output rows are computed four at a time: each step of the inner loop loads
/// one input row `input[j]` (`B` values) and one weight per output row, and
/// performs `4 * B` independent fused multiply-adds, which the compiler
/// vectorizes across the block. This is the building block for batched
/// variants (tangent or reverse sweeps) that keep the same block layout.
///
/// `input` must have at least `layer.cols` rows and `output` at least
/// `layer.rows`.
#[inline]
fn dense_block<const B: usize>(
    layer: &Layer,
    input: &[[f64; B]],
    output: &mut [[f64; B]],
    activate: bool,
) {
    let cols = layer.cols;
    let input = &input[..cols];
    let output = &mut output[..layer.rows];

    let w_quads = layer.weights.chunks_exact(4 * cols);
    let w_rest = w_quads.remainder();
    let b_quads = layer.biases.chunks_exact(4);
    let b_rest = b_quads.remainder();
    let mut out_quads = output.chunks_exact_mut(4);
    for ((w4, b4), out4) in w_quads.zip(b_quads).zip(&mut out_quads) {
        let (w0, w123) = w4.split_at(cols);
        let (w1, w23) = w123.split_at(cols);
        let (w2, w3) = w23.split_at(cols);
        let mut a0 = [b4[0]; B];
        let mut a1 = [b4[1]; B];
        let mut a2 = [b4[2]; B];
        let mut a3 = [b4[3]; B];
        for ((((x, &v0), &v1), &v2), &v3) in input.iter().zip(w0).zip(w1).zip(w2).zip(w3) {
            for p in 0..B {
                a0[p] = mul_add(v0, x[p], a0[p]);
                a1[p] = mul_add(v1, x[p], a1[p]);
                a2[p] = mul_add(v2, x[p], a2[p]);
                a3[p] = mul_add(v3, x[p], a3[p]);
            }
        }
        if activate {
            swish_block(&mut a0);
            swish_block(&mut a1);
            swish_block(&mut a2);
            swish_block(&mut a3);
        }
        out4.copy_from_slice(&[a0, a1, a2, a3]);
    }

    let out_rest = out_quads.into_remainder();
    for ((w_row, &bias), out) in w_rest.chunks_exact(cols).zip(b_rest).zip(out_rest) {
        let mut acc = [bias; B];
        for (x, &v) in input.iter().zip(w_row) {
            for p in 0..B {
                acc[p] = mul_add(v, x[p], acc[p]);
            }
        }
        if activate {
            swish_block(&mut acc);
        }
        *out = acc;
    }
}

impl Network {
    /// Loads a network from a Bora HDF5 weight file (`.h5`).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or has the wrong layout.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = open_file(path.as_ref(), "Bora network weight")?;
        Ok(Self {
            layers: read_layers(&file)?,
        })
    }

    /// Parses a network from an in-memory HDF5 file image, e.g. one embedded
    /// with `include_bytes!`.
    ///
    /// # Errors
    ///
    /// Returns an error if the image is not a valid Bora network file.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let file = open_bytes(data, "Bora network weight")?;
        Ok(Self {
            layers: read_layers(&file)?,
        })
    }

    /// Evaluates the network.
    ///
    /// # Panics
    ///
    /// If `x` is shorter than [`input_size`](Self::input_size) or `y` is
    /// shorter than [`output_size`](Self::output_size).
    pub fn evaluate(&self, x: &[f64], y: &mut [f64]) {
        let mut current = x[..self.input_size()].to_vec();
        let mut next = Vec::new();

        for (l, layer) in self.layers.iter().enumerate() {
            let is_last = l == self.layers.len() - 1;
            next.clear();
            next.extend(
                layer
                    .weights
                    .chunks_exact(layer.cols)
                    .zip(&layer.biases)
                    .map(|(w_row, bias)| {
                        let sum = bias + dot(w_row, &current);
                        if is_last { sum } else { swish(sum) }
                    }),
            );
            std::mem::swap(&mut current, &mut next);
        }

        y[..current.len()].copy_from_slice(&current);
    }

    /// Evaluates the network and propagates an input Jacobian.
    ///
    /// Computes y = net(x) and jy = d(net)/dx * jx via forward-mode
    /// differentiation, where `jx` holds the derivatives of `x` with respect to
    /// `n_seed` independent variables.
    ///
    /// - `jx`: input Jacobian, row-major `input_size() x n_seed`.
    /// - `y`: output vector (`output_size()` entries), overwritten.
    /// - `jy`: output Jacobian, row-major `output_size() x n_seed`, overwritten.
    ///
    /// # Panics
    ///
    /// If any slice is shorter than the dimensions above.
    pub fn evaluate_with_jacobian(
        &self,
        x: &[f64],
        jx: &[f64],
        n_seed: usize,
        y: &mut [f64],
        jy: &mut [f64],
    ) {
        // Values and Jacobians are propagated together layer-by-layer.
        // Jacobians are stored row-major: entry (i, k) is d(value_i)/d(seed_k).
        let mut current = x[..self.input_size()].to_vec();
        let mut current_jac = jx[..self.input_size() * n_seed].to_vec();
        let mut next = Vec::new();
        let mut next_jac = Vec::new();

        for (l, layer) in self.layers.iter().enumerate() {
            let is_last = l == self.layers.len() - 1;
            next.clear();
            next.resize(layer.rows, 0.0);
            next_jac.clear();
            next_jac.resize(layer.rows * n_seed, 0.0);

            let rows = layer
                .weights
                .chunks_exact(layer.cols)
                .zip(&layer.biases)
                .zip(next.iter_mut().zip(next_jac.chunks_exact_mut(n_seed)));
            for ((w_row, bias), (out, jac_row)) in rows {
                let sum = bias + dot(w_row, &current);

                // Jacobian rows accumulate two weighted input rows per pass to
                // shorten the dependency chain on each jac_row entry.
                let (w_pairs, w_rest) = w_row.as_chunks::<2>();
                for (&[w0, w1], jac_in) in w_pairs.iter().zip(current_jac.chunks_exact(2 * n_seed))
                {
                    let (jac_in0, jac_in1) = jac_in.split_at(n_seed);
                    for ((dest, a), b) in jac_row.iter_mut().zip(jac_in0).zip(jac_in1) {
                        *dest += w0 * a + w1 * b;
                    }
                }
                if let [w] = w_rest {
                    let jac_in = &current_jac[(layer.cols - 1) * n_seed..layer.cols * n_seed];
                    for (dest, a) in jac_row.iter_mut().zip(jac_in) {
                        *dest += w * a;
                    }
                }

                if is_last {
                    *out = sum;
                } else {
                    *out = swish(sum);
                    let d = swish_derivative(sum);
                    jac_row.iter_mut().for_each(|v| *v *= d);
                }
            }
            std::mem::swap(&mut current, &mut next);
            std::mem::swap(&mut current_jac, &mut next_jac);
        }

        y[..current.len()].copy_from_slice(&current);
        jy[..current_jac.len()].copy_from_slice(&current_jac);
    }

    /// Evaluates the network and propagates a single input tangent
    /// (a Jacobian-vector product).
    ///
    /// Computes y = net(x) and dy = d(net)/dx * dx via forward-mode
    /// differentiation along the one direction `dx`. Equivalent to
    /// [`evaluate_with_jacobian`](Self::evaluate_with_jacobian) with
    /// `n_seed = 1`, but each weight row is read once for both the value and
    /// the tangent, and the sigmoid is evaluated once per neuron. `y` is
    /// bit-identical to [`evaluate`](Self::evaluate).
    ///
    /// - `dx`: input tangent (`input_size()` entries).
    /// - `y`: output vector (`output_size()` entries), overwritten.
    /// - `dy`: output tangent (`output_size()` entries), overwritten.
    ///
    /// # Panics
    ///
    /// If `x` or `dx` is shorter than [`input_size`](Self::input_size), or `y`
    /// or `dy` is shorter than [`output_size`](Self::output_size).
    pub fn evaluate_with_tangent(&self, x: &[f64], dx: &[f64], y: &mut [f64], dy: &mut [f64]) {
        let n_in = self.input_size();
        let mut current = x[..n_in].to_vec();
        let mut current_dot = dx[..n_in].to_vec();
        let mut next = Vec::new();
        let mut next_dot = Vec::new();

        for (l, layer) in self.layers.iter().enumerate() {
            let is_last = l == self.layers.len() - 1;
            next.clear();
            next_dot.clear();
            for (w_row, bias) in layer.weights.chunks_exact(layer.cols).zip(&layer.biases) {
                let (wx, wdx) = dot2(w_row, &current, &current_dot);
                let sum = bias + wx;
                if is_last {
                    next.push(sum);
                    next_dot.push(wdx);
                } else {
                    // One exp per neuron: swish and its derivative share the
                    // sigmoid. The value matches `swish` exactly.
                    let denom = 1.0 + (-sum).exp();
                    let sig = 1.0 / denom;
                    next.push(sum / denom);
                    next_dot.push(wdx * (sig * (1.0 + sum * (1.0 - sig))));
                }
            }
            std::mem::swap(&mut current, &mut next);
            std::mem::swap(&mut current_dot, &mut next_dot);
        }

        y[..current.len()].copy_from_slice(&current);
        dy[..current_dot.len()].copy_from_slice(&current_dot);
    }

    /// Returns the input dimension of the network.
    pub fn input_size(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.cols)
    }

    /// Returns the output dimension of the network.
    pub fn output_size(&self) -> usize {
        self.layers.last().map_or(0, |layer| layer.rows)
    }

    /// Returns the layers of the network.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Evaluates the network like [`evaluate`](Self::evaluate) (with
    /// bit-identical results) and records in `tape` what
    /// [`backpropagate`](Self::backpropagate) needs for a subsequent
    /// reverse-mode sweep: the swish derivative of every hidden neuron.
    ///
    /// The tape's buffers are reused, so evaluating many points with one tape
    /// does not allocate after the first call.
    ///
    /// # Panics
    ///
    /// If `x` is shorter than [`input_size`](Self::input_size) or `y` is
    /// shorter than [`output_size`](Self::output_size).
    pub fn evaluate_with_tape(&self, x: &[f64], y: &mut [f64], tape: &mut NetworkTape) {
        let NetworkTape {
            derivatives,
            current,
            next,
            n_layers,
        } = tape;
        derivatives.clear();
        current.clear();
        current.extend_from_slice(&x[..self.input_size()]);

        for (l, layer) in self.layers.iter().enumerate() {
            let is_last = l == self.layers.len() - 1;
            next.clear();
            let rows = layer.weights.chunks_exact(layer.cols).zip(&layer.biases);
            if is_last {
                next.extend(rows.map(|(w_row, bias)| bias + dot(w_row, current)));
            } else {
                next.reserve(layer.rows);
                derivatives.reserve(layer.rows);
                for (w_row, bias) in rows {
                    let sum = bias + dot(w_row, current);
                    // One exponential per neuron serves both the activation and
                    // its derivative; the expressions match `swish` and
                    // `swish_derivative` exactly.
                    let denom = 1.0 + (-sum).exp();
                    let sig = 1.0 / denom;
                    next.push(sum / denom);
                    derivatives.push(sig * (1.0 + sum * (1.0 - sig)));
                }
            }
            std::mem::swap(current, next);
        }
        *n_layers = self.layers.len();

        y[..current.len()].copy_from_slice(current);
    }

    /// Reverse-mode sweep: computes `x_bar = J^T y_bar` for `n_cotangents`
    /// output cotangents at once, where `J = d(net)/dx` is the network Jacobian
    /// at the point most recently recorded into `tape` by
    /// [`evaluate_with_tape`](Self::evaluate_with_tape).
    ///
    /// - `y_bar`: output cotangents, row-major `n_cotangents x output_size()`
    ///   (each cotangent contiguous).
    /// - `x_bar`: input cotangents, row-major `n_cotangents x input_size()`,
    ///   overwritten.
    ///
    /// The cost is about one forward evaluation per cotangent, minus the final
    /// layer's rows whose cotangents are all zero (which are skipped), so
    /// sparse output cotangents are cheap. The tape keeps its recording and
    /// can be swept again with other cotangents.
    ///
    /// # Panics
    ///
    /// If `tape` was not recorded by a network with this layer structure, or
    /// if `y_bar` or `x_bar` is shorter than the dimensions above.
    pub fn backpropagate(
        &self,
        tape: &mut NetworkTape,
        y_bar: &[f64],
        n_cotangents: usize,
        x_bar: &mut [f64],
    ) {
        let n_hidden: usize = self.layers.iter().rev().skip(1).map(|l| l.rows).sum();
        assert!(
            tape.n_layers == self.layers.len() && tape.derivatives.len() == n_hidden,
            "tape was not recorded by this network"
        );
        let NetworkTape {
            derivatives,
            current,
            next,
            ..
        } = tape;
        let n = n_cotangents;

        // `current` holds the cotangents of the current layer's output,
        // row-major `n x layer.rows`.
        current.clear();
        current.extend_from_slice(&y_bar[..n * self.output_size()]);
        let mut d_end = derivatives.len();

        for (l, layer) in self.layers.iter().enumerate().rev() {
            if l != self.layers.len() - 1 {
                // Through the swish activation: z_bar = a_bar * swish'(z).
                let d = &derivatives[d_end - layer.rows..d_end];
                d_end -= layer.rows;
                for cotangent in current.chunks_exact_mut(layer.rows) {
                    for (c, d) in cotangent.iter_mut().zip(d) {
                        *c *= d;
                    }
                }
            }
            next.clear();
            next.resize(n * layer.cols, 0.0);
            accumulate_transposed(layer, current, n, next);
            std::mem::swap(current, next);
        }

        x_bar[..current.len()].copy_from_slice(current);
    }

    /// Evaluates the network for a batch of `n_points` points at once.
    ///
    /// Equivalent to calling [`evaluate`](Self::evaluate) on every point, but
    /// pushes blocks of [`BATCH_BLOCK_SIZE`] points through each weight matrix
    /// together, which is several times faster per point for large batches.
    /// Results agree with [`evaluate`](Self::evaluate) to a few ulps, not bit
    /// for bit: the batched kernel sums in a different order and evaluates the
    /// activation with a vectorizable `exp`. A point's result does not depend on
    /// the other points in the batch beyond that rounding level.
    ///
    /// Layout: matrices with one column per point, stored row-major
    /// (feature-major):
    ///
    /// - `x`: `input_size() x n_points`; input `j` of point `p` is
    ///   `x[j * n_points + p]`.
    /// - `y`: `output_size() x n_points`, overwritten; output `i` of point `p`
    ///   is `y[i * n_points + p]`.
    ///
    /// Points are processed in blocks of [`BATCH_BLOCK_SIZE`] consecutive
    /// columns. A partial final block is zero-padded (the padding is discarded)
    /// unless it has only one or two points, which are evaluated one at a time
    /// with the single-point kernel (bit-identical to `evaluate`). `workspace`
    /// holds the block activations; it is grown on first use and can be reused
    /// across calls to avoid allocating.
    ///
    /// # Panics
    ///
    /// If `x` is shorter than `input_size() * n_points` or `y` is shorter than
    /// `output_size() * n_points`.
    pub fn evaluate_batch(
        &self,
        x: &[f64],
        y: &mut [f64],
        n_points: usize,
        workspace: &mut BatchWorkspace,
    ) {
        const B: usize = BATCH_BLOCK_SIZE;
        let (n_in, n_out) = (self.input_size(), self.output_size());
        let x = &x[..n_in * n_points];
        let y = &mut y[..n_out * n_points];
        if n_points == 0 || self.layers.is_empty() {
            return;
        }

        let width = self
            .layers
            .iter()
            .map(|layer| layer.rows.max(layer.cols))
            .max()
            .unwrap_or(0);
        for buffer in [&mut workspace.current, &mut workspace.next] {
            if buffer.len() < width {
                buffer.resize(width, [0.0; B]);
            }
        }

        let mut start = 0;
        while start < n_points {
            let count = B.min(n_points - start);
            if count < MIN_PADDED_BLOCK_POINTS {
                for p in start..start + count {
                    self.evaluate_column(x, y, n_points, p, workspace);
                }
            } else {
                // Gather the block's inputs (zero-padding unused lanes).
                for (j, row) in workspace.current[..n_in].iter_mut().enumerate() {
                    *row = [0.0; B];
                    row[..count].copy_from_slice(&x[j * n_points + start..][..count]);
                }
                let last = self.layers.len() - 1;
                for (l, layer) in self.layers.iter().enumerate() {
                    dense_block(layer, &workspace.current, &mut workspace.next, l != last);
                    std::mem::swap(&mut workspace.current, &mut workspace.next);
                }
                for (i, row) in workspace.current[..n_out].iter().enumerate() {
                    y[i * n_points + start..][..count].copy_from_slice(&row[..count]);
                }
            }
            start += count;
        }
    }

    /// Evaluates column `p` of a feature-major batch (see
    /// [`evaluate_batch`](Self::evaluate_batch)) with the single-point kernel,
    /// performing the same operations as [`evaluate`](Self::evaluate) but using
    /// `workspace` instead of allocating.
    fn evaluate_column(
        &self,
        x: &[f64],
        y: &mut [f64],
        n_points: usize,
        p: usize,
        workspace: &mut BatchWorkspace,
    ) {
        let mut current = workspace.current.as_flattened_mut();
        let mut next = workspace.next.as_flattened_mut();
        let mut len = self.input_size();
        for (j, value) in current[..len].iter_mut().enumerate() {
            *value = x[j * n_points + p];
        }
        let last = self.layers.len() - 1;
        for (l, layer) in self.layers.iter().enumerate() {
            let rows = layer.weights.chunks_exact(layer.cols).zip(&layer.biases);
            for ((w_row, bias), out) in rows.zip(next.iter_mut()) {
                let sum = bias + dot(w_row, &current[..len]);
                *out = if l == last { sum } else { swish(sum) };
            }
            len = layer.rows;
            std::mem::swap(&mut current, &mut next);
        }
        for (i, value) in current[..len].iter().enumerate() {
            y[i * n_points + p] = *value;
        }
    }
}

impl InputDistribution {
    /// Loads distribution statistics from a Bora HDF5 file (`.h5`).
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or has the wrong layout.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        read_distribution(&open_file(path.as_ref(), "Bora input distribution")?)
    }

    /// Parses distribution statistics from an in-memory HDF5 file image.
    ///
    /// # Errors
    ///
    /// Returns an error if the image is not a valid Bora distribution file.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        read_distribution(&open_bytes(data, "Bora input distribution")?)
    }

    /// Computes S (x - mean), with S the inverse covariance.
    fn weighted_offset(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.size();
        let centered: Vec<f64> = x[..n].iter().zip(&self.mean).map(|(x, m)| x - m).collect();
        let weighted = self
            .inv_cov
            .chunks_exact(n)
            .map(|row| row.iter().zip(&centered).map(|(s, c)| s * c).sum())
            .collect();
        (centered, weighted)
    }

    /// Computes the squared Mahalanobis distance of a point (`size()` entries)
    /// from the training data distribution.
    ///
    /// # Panics
    ///
    /// If `x` is shorter than [`size`](Self::size).
    pub fn squared_mahalanobis(&self, x: &[f64]) -> f64 {
        let (centered, weighted) = self.weighted_offset(x);
        centered.iter().zip(&weighted).map(|(c, w)| c * w).sum()
    }

    /// Computes the gradient of [`squared_mahalanobis`](Self::squared_mahalanobis)
    /// at `x`, writing `size()` entries to `grad`.
    ///
    /// # Panics
    ///
    /// If `x` or `grad` is shorter than [`size`](Self::size).
    pub fn squared_mahalanobis_gradient(&self, x: &[f64], grad: &mut [f64]) {
        // d/dx (x - m)^T S (x - m) = 2 S (x - m), with S symmetric.
        let (_, weighted) = self.weighted_offset(x);
        for (g, w) in grad.iter_mut().zip(weighted) {
            *g = 2.0 * w;
        }
    }

    /// Returns the dimensionality of the input space.
    pub fn size(&self) -> usize {
        self.mean.len()
    }

    /// Same value as [`squared_mahalanobis`](Self::squared_mahalanobis), bit
    /// for bit (identical operations in identical order), without allocating:
    /// the centered offsets are recomputed per row instead of being stored.
    pub(crate) fn squared_mahalanobis_unbuffered(&self, x: &[f64]) -> f64 {
        let n = self.size();
        let x = &x[..n];
        let centered = |j: usize| x[j] - self.mean[j];
        self.inv_cov
            .chunks_exact(n)
            .enumerate()
            .map(|(i, row)| {
                let weighted: f64 = row.iter().enumerate().map(|(j, s)| s * centered(j)).sum();
                centered(i) * weighted
            })
            .sum()
    }
}

/// Recording of a network evaluation for reverse-mode differentiation, filled
/// by [`Network::evaluate_with_tape`] and consumed by
/// [`Network::backpropagate`].
///
/// Holds the swish derivative of every hidden neuron plus scratch buffers, all
/// reused across calls. A default-constructed tape is empty and can be used
/// with any network.
#[derive(Debug, Clone, Default)]
pub struct NetworkTape {
    /// Swish derivatives of all hidden layers, concatenated in layer order.
    derivatives: Vec<f64>,
    /// Scratch buffers (layer values or cotangents).
    current: Vec<f64>,
    next: Vec<f64>,
    /// Layer count of the network that recorded the tape.
    n_layers: usize,
}

/// Accumulates `out += g W` for the `n` cotangent rows of `g` (row-major
/// `n x layer.rows`) into `out` (row-major `n x layer.cols`), i.e. multiplies
/// each cotangent by the transpose of the row-major weights.
///
/// Loops over blocks of four weight rows, each streamed contiguously and
/// reused for every cotangent while in L1; the inner loop is a vectorizable
/// multi-row axpy over the columns that loads and stores each output entry
/// once per block. Blocks whose cotangents are all zero are skipped.
fn accumulate_transposed(layer: &Layer, g: &[f64], n: usize, out: &mut [f64]) {
    let (rows, cols) = (layer.rows, layer.cols);
    let mut blocks = layer.weights.chunks_exact(4 * cols);
    for (b, w_block) in blocks.by_ref().enumerate() {
        let (w01, w23) = w_block.split_at(2 * cols);
        let (w0, w1) = w01.split_at(cols);
        let (w2, w3) = w23.split_at(cols);
        for (g_j, out_j) in g.chunks_exact(rows).zip(out.chunks_exact_mut(cols)).take(n) {
            let [a0, a1, a2, a3] = [g_j[4 * b], g_j[4 * b + 1], g_j[4 * b + 2], g_j[4 * b + 3]];
            if a0 == 0.0 && a1 == 0.0 && a2 == 0.0 && a3 == 0.0 {
                continue;
            }
            let ws = w0.iter().zip(w1).zip(w2.iter().zip(w3));
            for (o, ((u0, u1), (u2, u3))) in out_j.iter_mut().zip(ws) {
                *o = mul_add(
                    a3,
                    *u3,
                    mul_add(a2, *u2, mul_add(a1, *u1, mul_add(a0, *u0, *o))),
                );
            }
        }
    }
    let first_rest = rows - rows % 4;
    for (r, w_row) in blocks.remainder().chunks_exact(cols).enumerate() {
        for (g_j, out_j) in g.chunks_exact(rows).zip(out.chunks_exact_mut(cols)).take(n) {
            let a = g_j[first_rest + r];
            if a == 0.0 {
                continue;
            }
            for (o, u) in out_j.iter_mut().zip(w_row) {
                *o = mul_add(a, *u, *o);
            }
        }
    }
}
