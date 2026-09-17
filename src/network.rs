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
    let mut a_chunks = a.chunks_exact(4);
    let mut b_chunks = b.chunks_exact(4);
    for (ac, bc) in (&mut a_chunks).zip(&mut b_chunks) {
        s[0] = mul_add(ac[0], bc[0], s[0]);
        s[1] = mul_add(ac[1], bc[1], s[1]);
        s[2] = mul_add(ac[2], bc[2], s[2]);
        s[3] = mul_add(ac[3], bc[3], s[3]);
    }
    for (x, y) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        s[0] += x * y;
    }
    (s[0] + s[1]) + (s[2] + s[3])
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
                let mut w_pairs = w_row.chunks_exact(2);
                let mut jac_pairs = current_jac.chunks_exact(2 * n_seed);
                for (w, jac_in) in (&mut w_pairs).zip(&mut jac_pairs) {
                    let (w0, w1) = (w[0], w[1]);
                    let (jac_in0, jac_in1) = jac_in.split_at(n_seed);
                    for ((dest, a), b) in jac_row.iter_mut().zip(jac_in0).zip(jac_in1) {
                        *dest += w0 * a + w1 * b;
                    }
                }
                if let [w] = w_pairs.remainder() {
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

    /// Returns the dimensionality of the input space.
    pub fn size(&self) -> usize {
        self.mean.len()
    }
}
