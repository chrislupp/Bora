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

use std::marker::PhantomData;
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
/// Each output row is accumulated for a block of this many points at once,
/// four rows at a time, so every weight loaded from memory feeds
/// `4 * BATCH_BLOCK_SIZE` independent multiply-add chains.
///
/// Chosen by measurement on an Apple M1 Pro (single thread, `cargo bench`,
/// minimum over repetitions): speedup per point of
/// [`NeuralFoil::analyze_batch_into`](crate::NeuralFoil::analyze_batch_into)
/// on 64 points over a loop over [`NeuralFoil::analyze`](crate::NeuralFoil::analyze)
/// (which evaluates both of its network passes in one traversal of the
/// weights), for block size `B` x output rows per pass `R`:
///
/// | model    | B4xR4 | B8xR4 | B16xR4 | B32xR4 | B8xR2 | B16xR2 | B16xR1 | B32xR1 |
/// |----------|-------|-------|--------|--------|-------|--------|--------|--------|
/// | xxsmall  | 1.41x | 1.95x | 1.57x  | 1.39x  | 1.54x | 1.95x  | 1.55x  | 0.61x  |
/// | medium   | 1.45x | 2.06x | 1.66x  | 1.49x  | 1.57x | 2.11x  | 1.58x  | 0.58x  |
/// | xlarge   | 1.44x | 2.29x | 1.69x  | 1.54x  | 1.52x | 2.34x  | 1.50x  | 0.53x  |
/// | xxxlarge | 1.38x | 2.42x | 1.73x  | 1.44x  | 1.47x | 2.56x  | 1.46x  | 0.49x  |
///
/// Sixteen points by two rows is 2-6% faster than eight by four on full
/// blocks, but its larger block is mostly padding for short batches: two to
/// four points ran at 0.6-1.3x `analyze` with 16 x 2 versus 1.1-2.5x with
/// 8 x 4, so 8 x 4 is kept. The vectorizable activation matters as much as
/// the blocking, and the four rows must be unrolled by hand: the same kernel
/// with generic `[[f64; B]; R]` accumulators was ~3x slower.
pub const BATCH_BLOCK_SIZE: usize = 8;

/// A partial final block with fewer than this many points is evaluated one point
/// at a time (with the single-point kernel) instead of being padded to a full
/// [`BATCH_BLOCK_SIZE`] block. One padded block costs about as much as two to
/// three single-point evaluations.
const MIN_PADDED_BLOCK_POINTS: usize = 3;

/// Reusable scratch space for [`Network::evaluate_batch`] and the batched
/// analysis calls
/// ([`NeuralFoil::analyze_batch_into`](crate::NeuralFoil::analyze_batch_into),
/// [`NeuralFoil::analyze_scalars_batch_into`](crate::NeuralFoil::analyze_scalars_batch_into)).
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
        source: source.into(),
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

/// How the kernels compute `a * b + c`.
///
/// The strategy is a type parameter of every kernel and pass rather than a
/// `cfg` inside one helper, because the runtime-dispatched x86-64 path (see
/// [`dispatch`]) is compiled with FMA enabled while `cfg(target_feature)`
/// still reflects the crate-wide flags. Each strategy is a separate type, so
/// every monomorphized kernel contains only its own instruction sequence.
trait MulAdd {
    /// Returns `a * b + c`.
    fn mul_add(a: f64, b: f64, c: f64) -> f64;
}

/// A single fused multiply-add instruction. Only for code compiled with
/// hardware FMA: elsewhere `f64::mul_add` is a slow software routine.
struct Fused;

impl MulAdd for Fused {
    #[inline(always)]
    fn mul_add(a: f64, b: f64, c: f64) -> f64 {
        a.mul_add(b, c)
    }
}

/// A separate multiply and add, for code compiled without hardware FMA.
#[cfg_attr(any(target_arch = "aarch64", target_feature = "fma"), allow(dead_code))]
struct Unfused;

impl MulAdd for Unfused {
    #[inline(always)]
    fn mul_add(a: f64, b: f64, c: f64) -> f64 {
        a * b + c
    }
}

/// The strategy for the crate's static target features: aarch64 always has
/// FMA; x86-64 has it only when built with e.g. `-C target-cpu=native`.
#[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
type Baseline = Fused;
#[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
type Baseline = Unfused;

/// One network pass with its arguments, generic over the multiply-add
/// strategy so that [`dispatch`] can choose the strategy (and target
/// features) at run time.
trait Pass {
    /// Runs the pass. Implementations are `#[inline(always)]` and so are
    /// all the functions they call, so that the whole pass is compiled into
    /// (and with the target features of) the dispatching function.
    fn run<M: MulAdd>(self);
}

/// Runs `pass` with FMA and 256-bit vectors when the crate was compiled
/// without them but the CPU supports them (x86-64), otherwise with the
/// [`Baseline`] strategy. The feature check is made once per call, outside
/// the pass.
#[inline(always)]
fn dispatch<P: Pass>(pass: P) {
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    if avx2_fma::detected() {
        // SAFETY: `detected()` returns true only if `is_x86_feature_detected!`
        // reported both AVX2 and FMA on the running CPU, which is the
        // precondition for calling a function compiled with
        // `#[target_feature(enable = "avx2,fma")]`.
        unsafe { avx2_fma::run(pass) };
        return;
    }
    pass.run::<Baseline>();
}

/// The runtime-dispatched AVX2+FMA path for x86-64 builds whose static
/// target features lack them.
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx2", target_feature = "fma"))
))]
mod avx2_fma {
    use std::sync::OnceLock;

    use super::{Fused, Pass};

    /// Hidden testing switch, not public API: when set (to anything) at the
    /// first network evaluation, the process keeps the baseline path even on
    /// CPUs with AVX2 and FMA, so that both paths can be tested and timed.
    pub(super) const DISABLE_VAR: &str = "BORA_DISABLE_AVX2_FMA";

    /// Whether to take the AVX2+FMA path. Decided once per process, so all
    /// entry points use the same arithmetic and their bitwise relationships
    /// (e.g. pair vs. single evaluation) hold.
    #[inline]
    pub(super) fn detected() -> bool {
        static DETECTED: OnceLock<bool> = OnceLock::new();
        *DETECTED.get_or_init(|| {
            std::env::var_os(DISABLE_VAR).is_none()
                && is_x86_feature_detected!("avx2")
                && is_x86_feature_detected!("fma")
        })
    }

    /// Runs `pass` with FMA. The pass is inlined (see [`Pass::run`]), so it
    /// is compiled with AVX2 and FMA enabled.
    #[target_feature(enable = "avx2,fma")]
    pub(super) fn run<P: Pass>(pass: P) {
        pass.run::<Fused>();
    }
}

/// Independent accumulators per dot product. A single accumulator chain is
/// bound by multiply-add latency (the compiler may not reassociate
/// floating-point sums itself); 16 chains keep the vector units busy. Measured
/// on Apple M1 Pro, 16 beat 4 (by ~1.5x), 8 and 32 for every model size up to
/// xxlarge and tied with 8 on xxxlarge; 32 loses on the narrow networks, whose
/// rows (25-64 entries) leave most of its accumulators idle.
const DOT_LANES: usize = 16;

/// Largest seed count served by the register-accumulated Jacobian kernel;
/// wider seed sets use [`Network::jacobian_pass_general`]. Measured on Apple
/// M1 Pro, the register kernel was 1.5-4x faster than the general one for
/// every seed count tried (1, 5, 16, 23, 30).
const MAX_REGISTER_SEEDS: usize = 32;

/// Largest number of adjoint lanes (output gradients) per reverse sweep;
/// more are split across several sweeps. Lane counts are padded to a
/// multiple of 2 (one 128-bit vector of f64).
const MAX_ADJOINT_LANES: usize = 16;

/// Weight rows per adjoint update (see [`adjoint_rows`]). Measured on Apple M1
/// Pro with 12 lanes (the two-input scalar gradient pass), 4 beat 2 by ~5-25%
/// and 8 (whose 96 coefficients no longer fit in registers) by ~15-35% on
/// xlarge and xxxlarge; all three tied on the smaller models.
const ADJOINT_ROW_BLOCK: usize = 4;

/// Sums an accumulator array pairwise (`K` must be a power of two).
#[inline(always)]
fn tree_sum<const K: usize>(mut s: [f64; K]) -> f64 {
    const { assert!(K.is_power_of_two()) };
    let mut width = K;
    while width > 1 {
        width /= 2;
        for k in 0..width {
            s[k] += s[k + width];
        }
    }
    s[0]
}

/// Dot product over the common length of `w` and `x`, using `DOT_LANES`
/// independent accumulators (the remainder is folded into the first
/// accumulators) and a pairwise final reduction.
#[inline(always)]
fn dot<M: MulAdd>(w: &[f64], x: &[f64]) -> f64 {
    let n = w.len().min(x.len());
    let (w, x) = (&w[..n], &x[..n]);
    let (w_chunks, w_rest) = w.as_chunks::<DOT_LANES>();
    let (x_chunks, x_rest) = x.as_chunks::<DOT_LANES>();
    let mut s = [0.0; DOT_LANES];
    for (wc, xc) in w_chunks.iter().zip(x_chunks) {
        for k in 0..DOT_LANES {
            s[k] = M::mul_add(wc[k], xc[k], s[k]);
        }
    }
    for (k, (wv, xv)) in w_rest.iter().zip(x_rest).enumerate() {
        s[k] = M::mul_add(*wv, *xv, s[k]);
    }
    tree_sum(s)
}

/// [`dot`] of one weight row with two inputs, loading the row once. Each lane
/// goes through exactly the operation sequence of [`dot`], so when `a` and `b`
/// are at least as long as `w` the results are bitwise identical to
/// `[dot(w, a), dot(w, b)]`.
#[inline(always)]
fn dot2<M: MulAdd>(w: &[f64], a: &[f64], b: &[f64]) -> [f64; 2] {
    let n = w.len().min(a.len()).min(b.len());
    let (w, a, b) = (&w[..n], &a[..n], &b[..n]);
    let (w_chunks, w_rest) = w.as_chunks::<DOT_LANES>();
    let (a_chunks, a_rest) = a.as_chunks::<DOT_LANES>();
    let (b_chunks, b_rest) = b.as_chunks::<DOT_LANES>();
    let mut s = [0.0; DOT_LANES];
    let mut t = [0.0; DOT_LANES];
    for ((wc, ac), bc) in w_chunks.iter().zip(a_chunks).zip(b_chunks) {
        for k in 0..DOT_LANES {
            s[k] = M::mul_add(wc[k], ac[k], s[k]);
            t[k] = M::mul_add(wc[k], bc[k], t[k]);
        }
    }
    for (k, ((wv, av), bv)) in w_rest.iter().zip(a_rest).zip(b_rest).enumerate() {
        s[k] = M::mul_add(*wv, *av, s[k]);
        t[k] = M::mul_add(*wv, *bv, t[k]);
    }
    [tree_sum(s), tree_sum(t)]
}

/// One row of a forward-mode Jacobian product: returns `sum_j w[j] * jx[j]`,
/// where every `jx[j]` is a derivative row padded to `P` seeds.
///
/// The `P`-wide result stays in registers across the whole weight row and is
/// stored once by the caller, rather than read and written per column; the
/// `P` accumulators are independent chains, so a plain multiply-add chain per
/// seed keeps the vector units busy. Measured on Apple M1 Pro, this beat
/// combining four columns pairwise before accumulating by ~8%, and a loop over
/// single columns by ~2%.
///
/// Panics if `jx` is shorter than `w`.
#[inline(always)]
fn jacobian_row<const P: usize, M: MulAdd>(w: &[f64], jx: &[[f64; P]]) -> [f64; P] {
    let (w_quads, w_rest) = w.as_chunks::<4>();
    let (j_quads, j_rest) = jx[..w.len()].as_chunks::<4>();
    let mut acc = [0.0; P];
    for (wq, jq) in w_quads.iter().zip(j_quads) {
        for k in 0..P {
            acc[k] = M::mul_add(wq[0], jq[0][k], acc[k]);
            acc[k] = M::mul_add(wq[1], jq[1][k], acc[k]);
            acc[k] = M::mul_add(wq[2], jq[2][k], acc[k]);
            acc[k] = M::mul_add(wq[3], jq[3][k], acc[k]);
        }
    }
    for (wv, jr) in w_rest.iter().zip(j_rest) {
        for k in 0..P {
            acc[k] = M::mul_add(*wv, jr[k], acc[k]);
        }
    }
    acc
}

/// The dot-product kernel for evaluating `B` inputs per traversal of the
/// weights.
///
/// Each implementation is a separate type so that every monomorphized pass
/// contains only its own kernel: selecting the kernel with `if B == 2` inside a
/// generic function measured ~15% slower for the single-input pass.
trait Lanes<const B: usize> {
    /// [`dot`] of one weight row with each of the `B` inputs.
    fn dots<M: MulAdd>(w: &[f64], x: [&[f64]; B]) -> [f64; B];
}

/// The kernel for a single input.
struct OneInput;

impl Lanes<1> for OneInput {
    #[inline(always)]
    fn dots<M: MulAdd>(w: &[f64], [x]: [&[f64]; 1]) -> [f64; 1] {
        [dot::<M>(w, x)]
    }
}

/// The kernel for two inputs sharing each weight row load.
struct TwoInputs;

impl Lanes<2> for TwoInputs {
    #[inline(always)]
    fn dots<M: MulAdd>(w: &[f64], [a, b]: [&[f64]; 2]) -> [f64; 2] {
        dot2::<M>(w, a, b)
    }
}

#[inline(always)]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// swish(x) = x * sigmoid(x), the network's hidden-layer activation.
#[inline(always)]
fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

/// Returns (swish(x), d/dx swish(x)) from a single sigmoid evaluation, with
/// d/dx swish(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))). The value is
/// bitwise identical to [`swish`].
#[inline(always)]
fn swish_with_derivative(x: f64) -> (f64, f64) {
    let sig = sigmoid(x);
    (x * sig, sig * (1.0 + x * (1.0 - sig)))
}

/// Applies the first `rows` rows of one dense layer (and the swish
/// activation, if `activate`) to `B` inputs in a single traversal of the
/// weights.
#[inline(always)]
fn dense_layer<const B: usize, L: Lanes<B>, M: MulAdd>(
    layer: &Layer,
    rows: usize,
    activate: bool,
    x: [&[f64]; B],
    y: [&mut [f64]; B],
) {
    let weights = layer.weights[..rows * layer.cols].chunks_exact(layer.cols);
    for (r, (w, bias)) in weights.zip(&layer.biases[..rows]).enumerate() {
        let sums = L::dots::<M>(w, x);
        for b in 0..B {
            let s = bias + sums[b];
            y[b][r] = if activate { swish(s) } else { s };
        }
    }
}

/// Applies one hidden dense layer and the swish activation to `B` inputs in a
/// single traversal of the weights, also storing the activation slopes
/// d(swish)/d(pre-activation) for a later reverse sweep. The values are
/// bitwise identical to those of [`dense_layer`].
#[inline(always)]
fn dense_layer_slopes<const B: usize, L: Lanes<B>, M: MulAdd>(
    layer: &Layer,
    x: [&[f64]; B],
    y: [&mut [f64]; B],
    slopes: [&mut [f64]; B],
) {
    let rows = layer.weights.chunks_exact(layer.cols).zip(&layer.biases);
    for (r, (w, bias)) in rows.enumerate() {
        let sums = L::dots::<M>(w, x);
        for b in 0..B {
            (y[b][r], slopes[b][r]) = swish_with_derivative(bias + sums[b]);
        }
    }
}

/// One step of a reverse (adjoint) sweep through a dense layer: adds
/// `sum_i c[i] * w[i][j]` to `out[j]` for every column `j`, where `w` are `R`
/// weight rows and `c[i]` the `P`-lane adjoint of row `i`'s pre-activation.
///
/// Blocking `R` rows per update reads and writes each `out[j]` once per block
/// rather than once per row, and the `R * P` coefficients stay in registers.
///
/// Panics if any `w[i]` is shorter than `out`.
#[inline(always)]
fn adjoint_rows<const R: usize, const P: usize, M: MulAdd>(
    w: [&[f64]; R],
    c: [[f64; P]; R],
    out: &mut [[f64; P]],
) {
    let w = w.map(|w| &w[..out.len()]);
    for (j, o) in out.iter_mut().enumerate() {
        let mut acc = *o;
        for i in 0..R {
            for k in 0..P {
                acc[k] = M::mul_add(w[i][j], c[i][k], acc[k]);
            }
        }
        *o = acc;
    }
}

/// Applies one dense layer (and the swish activation, if `activate`) to one
/// input and propagates its `P`-wide (padded) Jacobian.
#[inline(always)]
fn dense_layer_jacobian<const P: usize, M: MulAdd>(
    layer: &Layer,
    activate: bool,
    x: &[f64],
    jx: &[[f64; P]],
    y: &mut [f64],
    jy: &mut [[f64; P]],
) {
    let rows = layer.weights.chunks_exact(layer.cols).zip(&layer.biases);
    for ((w, bias), (y, jy)) in rows.zip(y.iter_mut().zip(jy)) {
        let s = bias + dot::<M>(w, x);
        let mut jac_row = jacobian_row::<P, M>(w, jx);
        if activate {
            let (value, slope) = swish_with_derivative(s);
            *y = value;
            jac_row.iter_mut().for_each(|v| *v *= slope);
        } else {
            *y = s;
        }
        *jy = jac_row;
    }
}

/// Applies one dense layer (and the swish activation, if `activate`) to
/// `V / 2` inputs and propagates one tangent per input, in a single traversal
/// of the weights.
///
/// The lanes alternate value and tangent: `x[2p]` is input `p` and `x[2p + 1]`
/// its tangent, and likewise for `y`. The values are bitwise identical to
/// those of [`dense_layer`]; the tangent of a pre-activation is the weight row
/// times the input tangent (the bias has none), scaled by the swish slope.
#[inline(always)]
fn dense_layer_tangent<const V: usize, L: Lanes<V>, M: MulAdd>(
    layer: &Layer,
    activate: bool,
    x: [&[f64]; V],
    y: [&mut [f64]; V],
) {
    let rows = layer.weights.chunks_exact(layer.cols).zip(&layer.biases);
    for (r, (w, bias)) in rows.enumerate() {
        let sums = L::dots::<M>(w, x);
        for p in 0..V / 2 {
            let s = bias + sums[2 * p];
            if activate {
                let (value, slope) = swish_with_derivative(s);
                y[2 * p][r] = value;
                y[2 * p + 1][r] = slope * sums[2 * p + 1];
            } else {
                y[2 * p][r] = s;
                y[2 * p + 1][r] = sums[2 * p + 1];
            }
        }
    }
}

/// Branch-free `exp(x)` that the compiler can vectorize, unlike `f64::exp`
/// (a libm call). Accurate to about one ulp (see the
/// `exp_vectorizable_is_accurate` test: max. relative error on [-700, 700] of
/// 2.3e-16 with either multiply-add strategy); the argument is clamped to
/// [-708, 709], so it never returns zero or infinity, which is harmless where
/// it is used (`1 + exp(-x)`) but makes it unsuitable as a general `exp`.
///
/// Uses `exp(x) = 2^n exp(r)` with `n = round(x / ln 2)` and
/// `|r| <= ln(2) / 2`, a degree-13 Taylor polynomial for `exp(r)`, and builds
/// `2^n` from its bit pattern. Only multiply-adds and integer bit operations
/// are used. The reduction `r = (x - n ln2_hi) - n ln2_lo` (Cody-Waite) stays
/// accurate without FMA: `ln2_hi` has 32 significant bits, so `n * ln2_hi` is
/// exact for the `|n| <= 1023` that occur, and `x - n * ln2_hi` is exact as
/// well because the two nearly cancel.
#[inline(always)]
fn exp_vectorizable<M: MulAdd>(x: f64) -> f64 {
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
    let shifted = M::mul_add(x, std::f64::consts::LOG2_E, ROUNDER);
    let n = shifted - ROUNDER;
    let r = M::mul_add(n, -LN_2_LO, M::mul_add(n, -LN_2_HI, x));
    let mut p = INV_FACTORIALS[0];
    for &c in &INV_FACTORIALS[1..] {
        p = M::mul_add(p, r, c);
    }
    // The low bits of `shifted` hold n in two's complement; (n + 1023) << 52
    // is the bit pattern of 2^n for -1022 <= n <= 1023.
    let scale = f64::from_bits(shifted.to_bits().wrapping_add(1023) << 52);
    p * scale
}

/// Applies swish to a block of values with [`exp_vectorizable`]; agrees with
/// [`swish`] to about one ulp.
#[inline(always)]
fn swish_block<const B: usize, M: MulAdd>(values: &mut [f64; B]) {
    for v in values.iter_mut() {
        *v /= 1.0 + exp_vectorizable::<M>(-*v);
    }
}

/// The first `rows` rows of one dense layer applied to a block of `B` points
/// stored neuron-major:
/// `output[i][p] = act(bias[i] + sum_j weight[i][j] * input[j][p])`, with
/// swish as `act` if `activate`, else the identity.
///
/// Output rows are computed four at a time: each step of the inner loop loads
/// one input row `input[j]` (`B` values) and one weight per output row, and
/// performs `4 * B` independent multiply-adds, which the compiler vectorizes
/// across the block.
///
/// `input` must have at least `layer.cols` rows and `output` at least `rows`.
#[inline(always)]
fn dense_block<const B: usize, M: MulAdd>(
    layer: &Layer,
    rows: usize,
    activate: bool,
    input: &[[f64; B]],
    output: &mut [[f64; B]],
) {
    let cols = layer.cols;
    let input = &input[..cols];
    let output = &mut output[..rows];

    let w_quads = layer.weights[..rows * cols].chunks_exact(4 * cols);
    let w_rest = w_quads.remainder();
    let (b_quads, b_rest) = layer.biases[..rows].as_chunks::<4>();
    let (out_quads, out_rest) = output.as_chunks_mut::<4>();
    for ((w4, b4), out4) in w_quads.zip(b_quads).zip(out_quads) {
        let (w0, w123) = w4.split_at(cols);
        let (w1, w23) = w123.split_at(cols);
        let (w2, w3) = w23.split_at(cols);
        let mut a0 = [b4[0]; B];
        let mut a1 = [b4[1]; B];
        let mut a2 = [b4[2]; B];
        let mut a3 = [b4[3]; B];
        for ((((x, &v0), &v1), &v2), &v3) in input.iter().zip(w0).zip(w1).zip(w2).zip(w3) {
            for p in 0..B {
                a0[p] = M::mul_add(v0, x[p], a0[p]);
                a1[p] = M::mul_add(v1, x[p], a1[p]);
                a2[p] = M::mul_add(v2, x[p], a2[p]);
                a3[p] = M::mul_add(v3, x[p], a3[p]);
            }
        }
        if activate {
            swish_block::<B, M>(&mut a0);
            swish_block::<B, M>(&mut a1);
            swish_block::<B, M>(&mut a2);
            swish_block::<B, M>(&mut a3);
        }
        *out4 = [a0, a1, a2, a3];
    }

    for ((w_row, &bias), out) in w_rest.chunks_exact(cols).zip(b_rest).zip(out_rest) {
        let mut acc = [bias; B];
        for (x, &v) in input.iter().zip(w_row) {
            for p in 0..B {
                acc[p] = M::mul_add(v, x[p], acc[p]);
            }
        }
        if activate {
            swish_block::<B, M>(&mut acc);
        }
        *out = acc;
    }
}

/// Views the first `len` entries of each of the `B` consecutive `stride`-long
/// lanes of `buf`.
fn lanes<T, const B: usize>(buf: &[T], stride: usize, len: usize) -> [&[T]; B] {
    std::array::from_fn(|b| &buf[b * stride..][..len])
}

/// Mutable counterpart of [`lanes`].
fn lanes_mut<T, const B: usize>(buf: &mut [T], stride: usize, len: usize) -> [&mut [T]; B] {
    let mut chunks = buf.chunks_exact_mut(stride);
    std::array::from_fn(|_| &mut chunks.next().expect("buffer holds B lanes")[..len])
}

/// [`Network::value_pass`] with its arguments.
struct ValuePass<'a, const B: usize, L> {
    network: &'a Network,
    x: [&'a [f64]; B],
    n_out: usize,
    y: [&'a mut [f64]; B],
    kernel: PhantomData<L>,
}

impl<'a, const B: usize, L> ValuePass<'a, B, L> {
    fn new(network: &'a Network, x: [&'a [f64]; B], n_out: usize, y: [&'a mut [f64]; B]) -> Self {
        let kernel = PhantomData;
        Self {
            network,
            x,
            n_out,
            y,
            kernel,
        }
    }
}

impl<const B: usize, L: Lanes<B>> Pass for ValuePass<'_, B, L> {
    #[inline(always)]
    fn run<M: MulAdd>(self) {
        self.network
            .value_pass::<B, L, M>(self.x, self.n_out, self.y);
    }
}

/// [`Network::gradient_pass`] with its arguments.
struct GradientPass<'a, const B: usize, L> {
    network: &'a Network,
    x: [&'a [f64]; B],
    n_out: usize,
    y: [&'a mut [f64]; B],
    grad: [&'a mut [f64]; B],
    kernel: PhantomData<L>,
}

impl<'a, const B: usize, L> GradientPass<'a, B, L> {
    fn new(
        network: &'a Network,
        x: [&'a [f64]; B],
        n_out: usize,
        y: [&'a mut [f64]; B],
        grad: [&'a mut [f64]; B],
    ) -> Self {
        let kernel = PhantomData;
        Self {
            network,
            x,
            n_out,
            y,
            grad,
            kernel,
        }
    }
}

impl<const B: usize, L: Lanes<B>> Pass for GradientPass<'_, B, L> {
    #[inline(always)]
    fn run<M: MulAdd>(self) {
        let Self {
            network,
            x,
            n_out,
            y,
            grad,
            ..
        } = self;
        network.gradient_pass::<B, L, M>(x, n_out, y, grad);
    }
}

/// [`Network::jacobian_pass`] with its arguments.
struct JacobianPass<'a> {
    network: &'a Network,
    x: &'a [f64],
    jx: &'a [f64],
    n_seed: usize,
    y: &'a mut [f64],
    jy: &'a mut [f64],
}

impl Pass for JacobianPass<'_> {
    #[inline(always)]
    fn run<M: MulAdd>(self) {
        let Self {
            network,
            x,
            jx,
            n_seed,
            y,
            jy,
        } = self;
        network.jacobian_pass::<M>(x, jx, n_seed, y, jy);
    }
}

/// [`Network::tangent_pass`] with its arguments.
struct TangentPass<'a, const V: usize, L> {
    network: &'a Network,
    x: [&'a [f64]; V],
    y: [&'a mut [f64]; V],
    kernel: PhantomData<L>,
}

impl<const V: usize, L: Lanes<V>> Pass for TangentPass<'_, V, L> {
    #[inline(always)]
    fn run<M: MulAdd>(self) {
        self.network.tangent_pass::<V, L, M>(self.x, self.y);
    }
}

/// [`Network::batch_pass`] with its arguments.
struct BatchPass<'a> {
    network: &'a Network,
    x: &'a [f64],
    n_points: usize,
    n_out: usize,
    y: &'a mut [f64],
    workspace: &'a mut BatchWorkspace,
}

impl Pass for BatchPass<'_> {
    #[inline(always)]
    fn run<M: MulAdd>(self) {
        let Self {
            network,
            x,
            n_points,
            n_out,
            y,
            workspace,
        } = self;
        network.batch_pass::<M>(x, n_points, n_out, y, workspace);
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
        dispatch(ValuePass::<1, OneInput>::new(
            self,
            [x],
            self.output_size(),
            [y],
        ));
    }

    /// Evaluates only the first `n_outputs` outputs of the network, skipping
    /// the remaining rows of the output layer.
    ///
    /// The values written to `y[..n_outputs]` are bitwise identical to the
    /// corresponding entries of [`evaluate`](Self::evaluate); entries from
    /// `n_outputs` on are left untouched.
    ///
    /// # Panics
    ///
    /// If `n_outputs` exceeds [`output_size`](Self::output_size), `x` is
    /// shorter than [`input_size`](Self::input_size), or `y` is shorter than
    /// `n_outputs`.
    pub fn evaluate_first(&self, x: &[f64], n_outputs: usize, y: &mut [f64]) {
        dispatch(ValuePass::<1, OneInput>::new(self, [x], n_outputs, [y]));
    }

    /// [`evaluate_first`](Self::evaluate_first) at two inputs in one traversal
    /// of the weights; `y[i]` receives the outputs for `x[i]`, bitwise
    /// identical to `evaluate_first(x[i], n_outputs, y[i])`.
    pub(crate) fn evaluate_first_pair(&self, x: [&[f64]; 2], n_outputs: usize, y: [&mut [f64]; 2]) {
        dispatch(ValuePass::<2, TwoInputs>::new(self, x, n_outputs, y));
    }

    /// Evaluates the first `n_outputs` outputs of the network and their
    /// gradients with respect to the network inputs, by reverse-mode
    /// differentiation.
    ///
    /// The cost is one forward pass plus one reverse sweep through the
    /// weights per (up to) 16 outputs, independent of the input dimension,
    /// which makes this much cheaper than
    /// [`evaluate_with_jacobian`](Self::evaluate_with_jacobian) when only a
    /// few outputs are needed.
    ///
    /// - `y`: the first `n_outputs` outputs, bitwise identical to the
    ///   corresponding entries of [`evaluate`](Self::evaluate).
    /// - `grad`: row-major `n_outputs x input_size()`; entry `[q][i]` is
    ///   d(output q)/d(input i). These agree with the forward-mode Jacobian
    ///   of [`evaluate_with_jacobian`](Self::evaluate_with_jacobian) (seeded
    ///   with the identity) up to rounding, since the sums are formed in a
    ///   different order.
    ///
    /// Entries of `y` and `grad` past those dimensions are left untouched.
    ///
    /// # Panics
    ///
    /// If `n_outputs` exceeds [`output_size`](Self::output_size), `x` is
    /// shorter than [`input_size`](Self::input_size), `y` is shorter than
    /// `n_outputs`, or `grad` is shorter than `n_outputs * input_size()`.
    pub fn evaluate_first_with_gradients(
        &self,
        x: &[f64],
        n_outputs: usize,
        y: &mut [f64],
        grad: &mut [f64],
    ) {
        dispatch(GradientPass::<1, OneInput>::new(
            self,
            [x],
            n_outputs,
            [y],
            [grad],
        ));
    }

    /// [`evaluate_first_with_gradients`](Self::evaluate_first_with_gradients)
    /// at two inputs, with the forward sweep sharing one traversal of the
    /// weights; `y[i]` and `grad[i]` receive the results for `x[i]`.
    pub(crate) fn evaluate_first_with_gradients_pair(
        &self,
        x: [&[f64]; 2],
        n_outputs: usize,
        y: [&mut [f64]; 2],
        grad: [&mut [f64]; 2],
    ) {
        dispatch(GradientPass::<2, TwoInputs>::new(
            self, x, n_outputs, y, grad,
        ));
    }

    /// Evaluates the network at two inputs in one traversal of the weights,
    /// which is faster than two [`evaluate`](Self::evaluate) calls because
    /// each weight row is loaded once for both inputs.
    ///
    /// `y[i]` receives the outputs for `x[i]`, bitwise identical to
    /// `evaluate(x[i], y[i])`.
    ///
    /// # Panics
    ///
    /// If either `x[i]` is shorter than [`input_size`](Self::input_size) or
    /// either `y[i]` is shorter than [`output_size`](Self::output_size).
    pub fn evaluate_pair(&self, x: [&[f64]; 2], y: [&mut [f64]; 2]) {
        dispatch(ValuePass::<2, TwoInputs>::new(
            self,
            x,
            self.output_size(),
            y,
        ));
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
    /// The values written to `y` are bitwise identical to those of
    /// [`evaluate`](Self::evaluate).
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
        dispatch(JacobianPass {
            network: self,
            x,
            jx,
            n_seed,
            y,
            jy,
        });
    }

    /// Evaluates the network and propagates a single input tangent (a
    /// Jacobian-vector product).
    ///
    /// Computes y = net(x) and dy = d(net)/dx * dx via forward-mode
    /// differentiation along the one direction `dx`. Equivalent to
    /// [`evaluate_with_jacobian`](Self::evaluate_with_jacobian) with
    /// `n_seed = 1` up to rounding, but each weight row is loaded once for
    /// both the value and the tangent (see [`evaluate_pair`](Self::evaluate_pair)).
    ///
    /// - `dx`: input tangent (`input_size()` entries).
    /// - `y`: output vector (`output_size()` entries), overwritten; bitwise
    ///   identical to [`evaluate`](Self::evaluate).
    /// - `dy`: output tangent (`output_size()` entries), overwritten.
    ///
    /// # Panics
    ///
    /// If `x` or `dx` is shorter than [`input_size`](Self::input_size), or `y`
    /// or `dy` is shorter than [`output_size`](Self::output_size).
    pub fn evaluate_with_tangent(&self, x: &[f64], dx: &[f64], y: &mut [f64], dy: &mut [f64]) {
        dispatch(TangentPass::<2, TwoInputs> {
            network: self,
            x: [x, dx],
            y: [y, dy],
            kernel: PhantomData,
        });
    }

    /// Evaluates the network for a batch of `n_points` points at once.
    ///
    /// Equivalent to calling [`evaluate`](Self::evaluate) on every point, but
    /// pushes blocks of [`BATCH_BLOCK_SIZE`] points through each weight matrix
    /// together, which is faster per point for batches of more than a few
    /// points. Results agree with [`evaluate`](Self::evaluate) to a few ulps,
    /// not bit for bit: the batched kernel sums in a different order and
    /// evaluates the activation with a vectorizable `exp`. A point's result
    /// does not depend on the other points in the batch beyond that rounding
    /// level.
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
    /// with the single-point kernel (bitwise identical to `evaluate`).
    /// `workspace` holds the block activations; it is grown on first use and
    /// can be reused across calls to avoid allocating.
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
        dispatch(BatchPass {
            network: self,
            x,
            n_points,
            n_out: self.output_size(),
            y,
            workspace,
        });
    }

    /// [`evaluate_batch`](Self::evaluate_batch) computing only the first
    /// `n_outputs` outputs of every point (skipping the remaining rows of the
    /// output layer); `y` is `n_outputs x n_points`, and its entries agree with
    /// the corresponding ones of `evaluate_batch` bit for bit.
    pub(crate) fn evaluate_first_batch(
        &self,
        x: &[f64],
        n_outputs: usize,
        y: &mut [f64],
        n_points: usize,
        workspace: &mut BatchWorkspace,
    ) {
        dispatch(BatchPass {
            network: self,
            x,
            n_points,
            n_out: n_outputs,
            y,
            workspace,
        });
    }

    /// The widest layer input or output, i.e. the activation buffer size.
    fn max_width(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.rows.max(layer.cols))
            .max()
            .unwrap_or(0)
    }

    /// Evaluates the network at `B` inputs in one traversal of the weights,
    /// computing only the first `n_out` outputs.
    #[inline(always)]
    fn value_pass<const B: usize, L: Lanes<B>, M: MulAdd>(
        &self,
        x: [&[f64]; B],
        n_out: usize,
        y: [&mut [f64]; B],
    ) {
        let (n_in, width) = (self.input_size(), self.max_width());
        assert!(
            n_out <= self.output_size(),
            "requested {n_out} outputs of a network with {}",
            self.output_size()
        );
        let x = x.map(|x| &x[..n_in]);
        let y = y.map(|y| &mut y[..n_out]);
        if self.layers.is_empty() {
            return;
        }
        // One allocation: two halves (current and next layer), each holding
        // B lanes of `width` entries.
        let mut buf = vec![0.0; 2 * B * width];
        let (mut cur, mut next) = buf.split_at_mut(B * width);
        for (lane, x) in cur.chunks_exact_mut(width).zip(x) {
            lane[..n_in].copy_from_slice(x);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let activate = l + 1 < self.layers.len();
            let rows = if activate { layer.rows } else { n_out };
            dense_layer::<B, L, M>(
                layer,
                rows,
                activate,
                lanes(cur, width, layer.cols),
                lanes_mut(next, width, rows),
            );
            std::mem::swap(&mut cur, &mut next);
        }
        for (lane, y) in cur.chunks_exact(width).zip(y) {
            y.copy_from_slice(&lane[..n_out]);
        }
    }

    /// Forward-mode pass for `V / 2` inputs with one tangent each, in one
    /// traversal of the weights; lanes alternate value and tangent (see
    /// [`dense_layer_tangent`]).
    #[inline(always)]
    fn tangent_pass<const V: usize, L: Lanes<V>, M: MulAdd>(
        &self,
        x: [&[f64]; V],
        y: [&mut [f64]; V],
    ) {
        let (n_in, n_out, width) = (self.input_size(), self.output_size(), self.max_width());
        let x = x.map(|x| &x[..n_in]);
        let y = y.map(|y| &mut y[..n_out]);
        if self.layers.is_empty() {
            return;
        }
        // One allocation: two halves (current and next layer), each holding
        // V lanes of `width` entries.
        let mut buf = vec![0.0; 2 * V * width];
        let (mut cur, mut next) = buf.split_at_mut(V * width);
        for (lane, x) in cur.chunks_exact_mut(width).zip(x) {
            lane[..n_in].copy_from_slice(x);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let activate = l + 1 < self.layers.len();
            dense_layer_tangent::<V, L, M>(
                layer,
                activate,
                lanes(cur, width, layer.cols),
                lanes_mut(next, width, layer.rows),
            );
            std::mem::swap(&mut cur, &mut next);
        }
        for (lane, y) in cur.chunks_exact(width).zip(y) {
            y.copy_from_slice(&lane[..n_out]);
        }
    }

    /// Batched forward pass over feature-major `x` (`input_size() x
    /// n_points`), writing the first `n_out` outputs of every point to `y`
    /// (`n_out x n_points`); see [`evaluate_batch`](Self::evaluate_batch).
    #[inline(always)]
    fn batch_pass<M: MulAdd>(
        &self,
        x: &[f64],
        n_points: usize,
        n_out: usize,
        y: &mut [f64],
        workspace: &mut BatchWorkspace,
    ) {
        const B: usize = BATCH_BLOCK_SIZE;
        let (n_in, width) = (self.input_size(), self.max_width());
        assert!(
            n_out <= self.output_size(),
            "requested {n_out} outputs of a network with {}",
            self.output_size()
        );
        let x = &x[..n_in * n_points];
        let y = &mut y[..n_out * n_points];
        if n_points == 0 || self.layers.is_empty() {
            return;
        }
        for buffer in [&mut workspace.current, &mut workspace.next] {
            if buffer.len() < width {
                buffer.resize(width, [0.0; B]);
            }
        }

        let last = self.layers.len() - 1;
        let mut start = 0;
        while start < n_points {
            let count = B.min(n_points - start);
            if count == 2 {
                // Both columns share one traversal of the weights, as in
                // `evaluate_pair` (a single point of `analyze_batch` is one
                // such column pair).
                self.evaluate_columns::<2, TwoInputs, M>(x, n_points, start, n_out, y, workspace);
            } else if count < MIN_PADDED_BLOCK_POINTS {
                self.evaluate_columns::<1, OneInput, M>(x, n_points, start, n_out, y, workspace);
            } else {
                // Gather the block's inputs (zero-padding unused lanes).
                for (j, row) in workspace.current[..n_in].iter_mut().enumerate() {
                    *row = [0.0; B];
                    row[..count].copy_from_slice(&x[j * n_points + start..][..count]);
                }
                for (l, layer) in self.layers.iter().enumerate() {
                    let rows = if l == last { n_out } else { layer.rows };
                    dense_block::<B, M>(
                        layer,
                        rows,
                        l != last,
                        &workspace.current,
                        &mut workspace.next,
                    );
                    std::mem::swap(&mut workspace.current, &mut workspace.next);
                }
                for (i, row) in workspace.current[..n_out].iter().enumerate() {
                    y[i * n_points + start..][..count].copy_from_slice(&row[..count]);
                }
            }
            start += count;
        }
    }

    /// Evaluates the first `n_out` outputs of the `C` columns starting at
    /// `start` of a feature-major batch (see
    /// [`evaluate_batch`](Self::evaluate_batch)) with the single-point kernels,
    /// performing the same operations as [`evaluate`](Self::evaluate) (and
    /// [`evaluate_pair`](Self::evaluate_pair)) but using `workspace` instead of
    /// allocating.
    #[inline(always)]
    fn evaluate_columns<const C: usize, L: Lanes<C>, M: MulAdd>(
        &self,
        x: &[f64],
        n_points: usize,
        start: usize,
        n_out: usize,
        y: &mut [f64],
        workspace: &mut BatchWorkspace,
    ) {
        // The workspace holds BATCH_BLOCK_SIZE >= C lanes of `width` entries.
        let width = self.max_width();
        let mut cur = workspace.current.as_flattened_mut();
        let mut next = workspace.next.as_flattened_mut();
        let n_in = self.input_size();
        for (c, lane) in cur.chunks_exact_mut(width).take(C).enumerate() {
            for (j, value) in lane[..n_in].iter_mut().enumerate() {
                *value = x[j * n_points + start + c];
            }
        }
        let last = self.layers.len() - 1;
        for (l, layer) in self.layers.iter().enumerate() {
            let rows = if l == last { n_out } else { layer.rows };
            dense_layer::<C, L, M>(
                layer,
                rows,
                l != last,
                lanes(cur, width, layer.cols),
                lanes_mut(next, width, rows),
            );
            std::mem::swap(&mut cur, &mut next);
        }
        for (c, lane) in cur.chunks_exact(width).take(C).enumerate() {
            for (i, value) in lane[..n_out].iter().enumerate() {
                y[i * n_points + start + c] = *value;
            }
        }
    }

    /// Reverse-mode pass at `B` inputs: evaluates the first `n_out` outputs in
    /// one forward traversal of the weights (storing the activation slopes),
    /// then propagates the adjoints of those outputs back to the inputs.
    ///
    /// `grad[b]` receives the gradients for input `b`, row-major
    /// `n_out x input_size()`.
    #[inline(always)]
    fn gradient_pass<const B: usize, L: Lanes<B>, M: MulAdd>(
        &self,
        x: [&[f64]; B],
        n_out: usize,
        y: [&mut [f64]; B],
        grad: [&mut [f64]; B],
    ) {
        let (n_in, width) = (self.input_size(), self.max_width());
        assert!(
            n_out <= self.output_size(),
            "requested {n_out} outputs of a network with {}",
            self.output_size()
        );
        let x = x.map(|x| &x[..n_in]);
        let y = y.map(|y| &mut y[..n_out]);
        let mut grad = grad.map(|g| &mut g[..n_out * n_in]);
        let Some((last, hidden)) = self.layers.split_last() else {
            return;
        };

        // Forward sweep. Slopes of hidden layer l are stored as B consecutive
        // lanes of `rows` entries each.
        let n_hidden: usize = hidden.iter().map(|layer| layer.rows).sum();
        let mut slopes = vec![0.0; B * n_hidden];
        let mut buf = vec![0.0; 2 * B * width];
        let (mut cur, mut next) = buf.split_at_mut(B * width);
        for (lane, x) in cur.chunks_exact_mut(width).zip(x) {
            lane[..n_in].copy_from_slice(x);
        }
        let mut offset = 0;
        for layer in hidden {
            dense_layer_slopes::<B, L, M>(
                layer,
                lanes(cur, width, layer.cols),
                lanes_mut(next, width, layer.rows),
                lanes_mut(&mut slopes[offset..], layer.rows, layer.rows),
            );
            offset += B * layer.rows;
            std::mem::swap(&mut cur, &mut next);
        }
        dense_layer::<B, L, M>(
            last,
            n_out,
            false,
            lanes(cur, width, last.cols),
            lanes_mut(next, width, n_out),
        );
        for (lane, y) in next.chunks_exact(width).zip(y) {
            y.copy_from_slice(&lane[..n_out]);
        }

        // Reverse sweeps: one adjoint lane per (input, output) pair, in groups
        // of at most MAX_ADJOINT_LANES. Lanes of different inputs share a
        // sweep (and so each weight row load): for two inputs with 6 outputs
        // each, one 12-lane sweep measured ~10% faster than two 6-lane sweeps
        // for medium, ~17% for xlarge and ~25% for xxxlarge on Apple M1 Pro
        // (a tie for xxsmall).
        let all_lanes: Vec<(usize, usize)> = (0..B)
            .flat_map(|b| (0..n_out).map(move |q| (b, q)))
            .collect();
        for chunk in all_lanes.chunks(MAX_ADJOINT_LANES) {
            self.adjoint_sweep::<M>(&slopes, B, chunk, &mut grad);
        }
    }

    /// Dispatches a reverse sweep to the kernel with the narrowest padded lane
    /// count that holds `lanes`.
    #[inline(always)]
    fn adjoint_sweep<M: MulAdd>(
        &self,
        slopes: &[f64],
        n_passes: usize,
        lanes: &[(usize, usize)],
        grad: &mut [&mut [f64]],
    ) {
        match lanes.len().next_multiple_of(2) {
            0 | 2 => self.adjoint_sweep_padded::<2, M>(slopes, n_passes, lanes, grad),
            4 => self.adjoint_sweep_padded::<4, M>(slopes, n_passes, lanes, grad),
            6 => self.adjoint_sweep_padded::<6, M>(slopes, n_passes, lanes, grad),
            8 => self.adjoint_sweep_padded::<8, M>(slopes, n_passes, lanes, grad),
            10 => self.adjoint_sweep_padded::<10, M>(slopes, n_passes, lanes, grad),
            12 => self.adjoint_sweep_padded::<12, M>(slopes, n_passes, lanes, grad),
            14 => self.adjoint_sweep_padded::<14, M>(slopes, n_passes, lanes, grad),
            16 => self.adjoint_sweep_padded::<16, M>(slopes, n_passes, lanes, grad),
            _ => unreachable!("adjoint lane groups hold at most {MAX_ADJOINT_LANES} lanes"),
        }
    }

    /// Reverse sweep for up to `P` adjoint lanes, where lane `k` is the
    /// gradient of output `lanes[k].1` of forward pass `lanes[k].0`.
    ///
    /// The adjoint of each layer input is held as one `P`-lane row per
    /// neuron; padding lanes stay zero.
    #[inline(always)]
    fn adjoint_sweep_padded<const P: usize, M: MulAdd>(
        &self,
        slopes: &[f64],
        n_passes: usize,
        lanes: &[(usize, usize)],
        grad: &mut [&mut [f64]],
    ) {
        debug_assert!(lanes.len() <= P);
        let (n_in, width) = (self.input_size(), self.max_width());
        let Some((last, hidden)) = self.layers.split_last() else {
            return;
        };
        let lane_pass: [usize; P] = std::array::from_fn(|k| lanes.get(k).map_or(0, |l| l.0));

        // The output layer has no activation: the adjoints of its input are
        // the selected weight rows.
        let mut buf = vec![[0.0; P]; 2 * width];
        let (mut cur, mut next) = buf.split_at_mut(width);
        for (k, &(_, q)) in lanes.iter().enumerate() {
            let w = &last.weights[q * last.cols..][..last.cols];
            for (a, w) in cur.iter_mut().zip(w) {
                a[k] = *w;
            }
        }

        let mut offset = slopes.len();
        for layer in hidden.iter().rev() {
            let (rows, cols) = (layer.rows, layer.cols);
            offset -= n_passes * rows;
            let layer_slopes = &slopes[offset..][..n_passes * rows];
            // Adjoint of neuron r's pre-activation, per lane.
            let coefficients = |r: usize| -> [f64; P] {
                std::array::from_fn(|k| cur[r][k] * layer_slopes[lane_pass[k] * rows + r])
            };
            let out = &mut next[..cols];
            out.fill([0.0; P]);
            let mut r = 0;
            while r + ADJOINT_ROW_BLOCK <= rows {
                adjoint_rows::<ADJOINT_ROW_BLOCK, P, M>(
                    std::array::from_fn(|i| &layer.weights[(r + i) * cols..][..cols]),
                    std::array::from_fn(|i| coefficients(r + i)),
                    out,
                );
                r += ADJOINT_ROW_BLOCK;
            }
            for r in r..rows {
                adjoint_rows::<1, P, M>(
                    [&layer.weights[r * cols..][..cols]],
                    [coefficients(r)],
                    out,
                );
            }
            std::mem::swap(&mut cur, &mut next);
        }

        for (k, &(b, q)) in lanes.iter().enumerate() {
            let g = &mut grad[b][q * n_in..][..n_in];
            for (g, a) in g.iter_mut().zip(cur.iter()) {
                *g = a[k];
            }
        }
    }

    /// Dispatches a Jacobian pass to the register kernel with the narrowest
    /// padded width that holds `n_seed`, or to the general kernel.
    #[inline(always)]
    fn jacobian_pass<M: MulAdd>(
        &self,
        x: &[f64],
        jx: &[f64],
        n_seed: usize,
        y: &mut [f64],
        jy: &mut [f64],
    ) {
        match n_seed.next_multiple_of(4) {
            0 | 4 => self.jacobian_pass_padded::<4, M>(x, jx, n_seed, y, jy),
            8 => self.jacobian_pass_padded::<8, M>(x, jx, n_seed, y, jy),
            12 => self.jacobian_pass_padded::<12, M>(x, jx, n_seed, y, jy),
            16 => self.jacobian_pass_padded::<16, M>(x, jx, n_seed, y, jy),
            20 => self.jacobian_pass_padded::<20, M>(x, jx, n_seed, y, jy),
            24 => self.jacobian_pass_padded::<24, M>(x, jx, n_seed, y, jy),
            28 => self.jacobian_pass_padded::<28, M>(x, jx, n_seed, y, jy),
            MAX_REGISTER_SEEDS => {
                self.jacobian_pass_padded::<MAX_REGISTER_SEEDS, M>(x, jx, n_seed, y, jy);
            }
            _ => self.jacobian_pass_general::<M>(x, jx, n_seed, y, jy),
        }
    }

    /// Forward-mode Jacobian pass with Jacobian rows padded from `n_seed` to
    /// `P` seeds and accumulated in registers.
    #[inline(always)]
    fn jacobian_pass_padded<const P: usize, M: MulAdd>(
        &self,
        x: &[f64],
        jx: &[f64],
        n_seed: usize,
        y: &mut [f64],
        jy: &mut [f64],
    ) {
        debug_assert!(n_seed <= P);
        let (n_in, n_out, width) = (self.input_size(), self.output_size(), self.max_width());
        let (x, jx) = (&x[..n_in], &jx[..n_in * n_seed]);
        if self.layers.is_empty() {
            return;
        }
        // Values and Jacobians are propagated together layer by layer, each in
        // one allocation split into current and next layer. Jacobian entry
        // [i][k] is d(value_i)/d(seed_k); padding seeds stay zero.
        let mut values = vec![0.0; 2 * width];
        let mut jacobians = vec![[0.0; P]; 2 * width];
        let (mut cur, mut next) = values.split_at_mut(width);
        let (mut cur_jac, mut next_jac) = jacobians.split_at_mut(width);
        cur[..n_in].copy_from_slice(x);
        for (i, row) in cur_jac[..n_in].iter_mut().enumerate() {
            row[..n_seed].copy_from_slice(&jx[i * n_seed..][..n_seed]);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let activate = l + 1 < self.layers.len();
            dense_layer_jacobian::<P, M>(
                layer,
                activate,
                &cur[..layer.cols],
                &cur_jac[..layer.cols],
                &mut next[..layer.rows],
                &mut next_jac[..layer.rows],
            );
            std::mem::swap(&mut cur, &mut next);
            std::mem::swap(&mut cur_jac, &mut next_jac);
        }
        y[..n_out].copy_from_slice(&cur[..n_out]);
        let jy = &mut jy[..n_out * n_seed];
        for (i, row) in cur_jac[..n_out].iter().enumerate() {
            jy[i * n_seed..][..n_seed].copy_from_slice(&row[..n_seed]);
        }
    }

    /// Forward-mode Jacobian pass for any `n_seed > 0`, with Jacobians stored
    /// row-major at their natural width. Used for seed counts above
    /// [`MAX_REGISTER_SEEDS`].
    #[inline(always)]
    fn jacobian_pass_general<M: MulAdd>(
        &self,
        x: &[f64],
        jx: &[f64],
        n_seed: usize,
        y: &mut [f64],
        jy: &mut [f64],
    ) {
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
                let sum = bias + dot::<M>(w_row, &current);

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
                    let (value, slope) = swish_with_derivative(sum);
                    *out = value;
                    jac_row.iter_mut().for_each(|v| *v *= slope);
                }
            }
            std::mem::swap(&mut current, &mut next);
            std::mem::swap(&mut current_jac, &mut next_jac);
        }

        y[..current.len()].copy_from_slice(&current);
        jy[..current_jac.len()].copy_from_slice(&current_jac);
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

    /// [`squared_mahalanobis`](Self::squared_mahalanobis) and its gradient
    /// (written to `grad`) from one evaluation of S (x - mean); both are
    /// bitwise identical to the separate calls.
    pub(crate) fn squared_mahalanobis_with_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
        let (centered, weighted) = self.weighted_offset(x);
        for (g, w) in grad.iter_mut().zip(&weighted) {
            *g = 2.0 * w;
        }
        centered.iter().zip(&weighted).map(|(c, w)| c * w).sum()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic network with 7 inputs and 5 outputs.
    fn test_network() -> Network {
        let dims = [(20, 7), (33, 20), (5, 33)];
        let mut seed = 0.37_f64;
        let mut next = || {
            seed = (seed * 9.73 + 0.11).fract();
            seed - 0.5
        };
        let layers = dims
            .iter()
            .map(|&(rows, cols)| Layer {
                rows,
                cols,
                weights: (0..rows * cols).map(|_| next()).collect(),
                biases: (0..rows).map(|_| next()).collect(),
            })
            .collect();
        Network { layers }
    }

    /// The fused and unfused strategies (one of which is not the baseline
    /// of this build) agree up to rounding.
    #[test]
    fn multiply_add_strategies_agree() {
        let network = test_network();
        let x = [0.3, -1.2, 0.7, 2.0, -0.4, 0.05, 1.1];
        let (mut fused, mut unfused) = ([0.0; 5], [0.0; 5]);
        network.value_pass::<1, OneInput, Fused>([&x], 5, [&mut fused]);
        network.value_pass::<1, OneInput, Unfused>([&x], 5, [&mut unfused]);
        for (a, b) in fused.iter().zip(&unfused) {
            assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "{a} vs {b}");
        }
    }

    /// Max. relative error of [`exp_vectorizable`] against `f64::exp` over a
    /// dense sweep of [-700, 700].
    fn exp_vectorizable_max_error<M: MulAdd>() -> f64 {
        const STEPS: i32 = 2_000_000;
        (-STEPS..=STEPS)
            .map(|i| {
                // Irrational step offset, so the sweep does not only hit
                // multiples of simple fractions of ln 2.
                let x = 700.0 * f64::from(i) / f64::from(STEPS) + 1e-3 * std::f64::consts::SQRT_2;
                let x = x.clamp(-700.0, 700.0);
                let exact = x.exp();
                ((exp_vectorizable::<M>(x) - exact) / exact).abs()
            })
            .fold(0.0, f64::max)
    }

    /// The vectorizable `exp` of the batched kernel is accurate to about one
    /// ulp with either multiply-add strategy: without FMA, its Cody-Waite
    /// reduction still forms `n * ln2_hi` exactly.
    #[test]
    fn exp_vectorizable_is_accurate() {
        let fused = exp_vectorizable_max_error::<Fused>();
        let unfused = exp_vectorizable_max_error::<Unfused>();
        println!("exp_vectorizable max. relative error: fused {fused:e}, unfused {unfused:e}");
        assert!(fused < 4.0 * f64::EPSILON, "fused: {fused:e}");
        assert!(unfused < 4.0 * f64::EPSILON, "unfused: {unfused:e}");
        // Clamped arguments stay finite and positive.
        for x in [-1e6, -745.0, 710.0, 1e6] {
            let v = exp_vectorizable::<Baseline>(x);
            assert!(v.is_finite() && v > 0.0, "{x}: {v}");
        }
    }

    /// The batched kernel agrees with the single-point kernel for both
    /// multiply-add strategies (one of which is not the baseline of this
    /// build).
    #[test]
    fn batch_strategies_agree() {
        let network = test_network();
        let n = BATCH_BLOCK_SIZE + 5;
        let x: Vec<f64> = (0..7 * n)
            .map(|i| ((i * 37 % 23) as f64 - 11.0) / 7.0)
            .collect();
        for check in [Network::batch_pass::<Fused>, Network::batch_pass::<Unfused>] {
            let mut y = vec![0.0; 5 * n];
            check(&network, &x, n, 5, &mut y, &mut BatchWorkspace::default());
            for p in 0..n {
                let point: Vec<f64> = (0..7).map(|j| x[j * n + p]).collect();
                let mut expected = [0.0; 5];
                network.value_pass::<1, OneInput, Unfused>([&point], 5, [&mut expected]);
                for (i, e) in expected.iter().enumerate() {
                    let a = y[i * n + p];
                    assert!((a - e).abs() <= 1e-12 * e.abs().max(1.0), "{a} vs {e}");
                }
            }
        }
    }

    /// The x86-64 runtime dispatch takes the AVX2+FMA path exactly when the
    /// CPU supports it and the testing switch is unset. Run with
    /// `--nocapture` to see which path this process uses.
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    #[test]
    fn dispatch_follows_cpu_features() {
        let supported = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        let disabled = std::env::var_os(avx2_fma::DISABLE_VAR).is_some();
        println!(
            "CPU avx2+fma: {supported}, {} set: {disabled}, AVX2+FMA path taken: {}",
            avx2_fma::DISABLE_VAR,
            avx2_fma::detected()
        );
        assert_eq!(avx2_fma::detected(), supported && !disabled);
    }
}
