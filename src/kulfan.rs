//! Kulfan (CST) geometry utilities.

use std::f64::consts::PI;

use crate::{Error, KulfanParameters, NUM_CST_WEIGHTS_PER_SIDE, Result};

/// A single (x, y) airfoil coordinate point.
pub type Point2 = [f64; 2];

const W: usize = NUM_CST_WEIGHTS_PER_SIDE;

/// Binomial coefficients comb(W - 1, k) for the Bernstein basis of order 7.
fn bernstein_coefficients() -> [f64; W] {
    let mut coefficients = [0.0; W];
    coefficients[0] = 1.0;
    for k in 1..W {
        coefficients[k] = coefficients[k - 1] * (W - k) as f64 / k as f64;
    }
    coefficients
}

/// Bernstein basis functions of order W - 1 evaluated at x.
fn bernstein_basis(x: f64) -> [f64; W] {
    let coefficients = bernstein_coefficients();
    std::array::from_fn(|k| coefficients[k] * x.powf(k as f64) * (1.0 - x).powf((W - 1 - k) as f64))
}

/// Class function C(x) = x^0.5 * (1 - x)^1.0 for conventional airfoils.
fn class_function(x: f64) -> f64 {
    x.sqrt() * (1.0 - x)
}

/// Leading-edge modification mode shape: x * (1 - x)^(W + 0.5).
fn leading_edge_mode(x: f64) -> f64 {
    x * (1.0 - x).max(0.0).powf(W as f64 + 0.5)
}

/// Solves the linear system a * x = b via Gaussian elimination with partial
/// pivoting. `a` is row-major n x n.
fn solve_linear_system(mut a: Vec<f64>, mut b: Vec<f64>, n: usize) -> Result<Vec<f64>> {
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row * n + col].abs() > a[pivot * n + col].abs() {
                pivot = row;
            }
        }
        if a[pivot * n + col].abs() < 1e-14 {
            return Err(Error::SingularFit);
        }
        if pivot != col {
            for j in 0..n {
                a.swap(col * n + j, pivot * n + j);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..n {
            let factor = a[row * n + col] / a[col * n + col];
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for j in row + 1..n {
            sum -= a[row * n + j] * x[j];
        }
        x[row] = sum / a[row * n + row];
    }
    Ok(x)
}

/// Solves min ||A x - b||_2 via the normal equations. `a` is row-major m x n.
fn solve_least_squares(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut ata = vec![0.0; n * n];
    let mut atb = vec![0.0; n];
    for (a_row, &b_row) in a.chunks_exact(n).zip(b) {
        for i in 0..n {
            atb[i] += a_row[i] * b_row;
            for j in i..n {
                ata[i * n + j] += a_row[i] * a_row[j];
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            ata[i * n + j] = ata[j * n + i];
        }
    }
    solve_linear_system(ata, atb, n)
}

/// Generates airfoil coordinates from Kulfan (CST) parameters.
///
/// Coordinates are returned in Selig format: from the upper surface trailing
/// edge, around the leading edge, to the lower surface trailing edge, using
/// cosine spacing in x/c. The parameterization matches AeroSandbox's
/// `get_kulfan_coordinates()` with N1=0.5, N2=1.0 and the leading-edge
/// modification term.
///
/// Returns `2 * n_points_per_side - 1` points.
///
/// # Errors
///
/// [`Error::InvalidArgument`] if `n_points_per_side < 2`.
pub fn generate_kulfan_coordinates(
    parameters: &KulfanParameters,
    n_points_per_side: usize,
) -> Result<Vec<Point2>> {
    if n_points_per_side < 2 {
        return Err(Error::InvalidArgument(
            "n_points_per_side must be at least 2".into(),
        ));
    }

    // Cosine-spaced x/c stations from 0 to 1.
    let x: Vec<f64> = (0..n_points_per_side)
        .map(|i| {
            let t = i as f64 / (n_points_per_side - 1) as f64;
            0.5 * (1.0 - (PI * t).cos())
        })
        .collect();

    let surface_y = |xi: f64, weights: &[f64; W], te_sign: f64| {
        let basis = bernstein_basis(xi);
        let shape: f64 = weights.iter().zip(&basis).map(|(w, b)| w * b).sum();
        class_function(xi) * shape
            + te_sign * xi * parameters.te_thickness / 2.0
            + parameters.leading_edge_weight * leading_edge_mode(xi)
    };

    // Selig format: upper surface TE -> LE, then lower surface LE -> TE.
    let upper = x
        .iter()
        .rev()
        .map(|&xi| [xi, surface_y(xi, &parameters.upper_weights, 1.0)]);
    let lower = x
        .iter()
        .skip(1)
        .map(|&xi| [xi, surface_y(xi, &parameters.lower_weights, -1.0)]);
    Ok(upper.chain(lower).collect())
}

/// Fits Kulfan (CST) parameters to a set of airfoil coordinates.
///
/// Solves the linear least-squares problem of AeroSandbox's
/// `get_kulfan_parameters(method="least_squares")`. If the fitted trailing edge
/// thickness is negative, the problem is re-solved with TE thickness fixed to
/// zero. Coordinates must be in Selig format and should be normalized (leading
/// edge at the origin, trailing edge at x=1); no normalization is performed.
///
/// # Errors
///
/// [`Error::InvalidArgument`] if there are too few coordinates, or
/// [`Error::SingularFit`] if the fit problem is singular.
pub fn fit_kulfan_parameters(coordinates: &[Point2]) -> Result<KulfanParameters> {
    let m = coordinates.len();
    if m < 2 * W + 2 {
        return Err(Error::InvalidArgument(
            "too few coordinates for a Kulfan fit".into(),
        ));
    }

    // The leading edge splits the coordinate list into upper and lower
    // surfaces (Selig ordering: upper surface comes first).
    let mut le_index = 0;
    for (i, point) in coordinates.iter().enumerate().skip(1) {
        if point[0] < coordinates[le_index][0] {
            le_index = i;
        }
    }

    // Unknown ordering matches AeroSandbox: 8 lower weights, 8 upper weights,
    // leading edge weight, TE thickness.
    let n_unknowns = 2 * W + 2;
    let mut a = vec![0.0; m * n_unknowns];
    let mut b = vec![0.0; m];
    for (row, (a_row, point)) in a.chunks_exact_mut(n_unknowns).zip(coordinates).enumerate() {
        let xi = point[0];
        let is_upper = row <= le_index;
        let basis = bernstein_basis(xi);
        let c = class_function(xi);
        let offset = if is_upper { W } else { 0 };
        for k in 0..W {
            a_row[offset + k] = c * basis[k];
        }
        a_row[2 * W] = leading_edge_mode(xi);
        a_row[2 * W + 1] = if is_upper { xi / 2.0 } else { -xi / 2.0 };
        b[row] = point[1];
    }

    let mut solution = solve_least_squares(&a, &b, n_unknowns)?;

    // A negative fitted TE thickness is non-physical: refit with it fixed to 0.
    if solution[2 * W + 1] < 0.0 {
        let a_reduced: Vec<f64> = a
            .chunks_exact(n_unknowns)
            .flat_map(|row| row[..n_unknowns - 1].iter().copied())
            .collect();
        solution = solve_least_squares(&a_reduced, &b, n_unknowns - 1)?;
        solution.push(0.0);
    }

    Ok(KulfanParameters {
        lower_weights: std::array::from_fn(|k| solution[k]),
        upper_weights: std::array::from_fn(|k| solution[W + k]),
        leading_edge_weight: solution[2 * W],
        te_thickness: solution[2 * W + 1],
    })
}
