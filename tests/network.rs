mod common;

use bora::{InputDistribution, Network};
use common::{Failures, WEIGHTS_DATA_DIR};

fn weights_path(name: &str) -> String {
    format!("{WEIGHTS_DATA_DIR}/{name}")
}

#[test]
fn network_from_bytes_matches_load() {
    let path = weights_path("nn-xxsmall.h5");
    let from_disk = Network::load(&path).unwrap();
    let from_memory = Network::from_bytes(&std::fs::read(&path).unwrap()).unwrap();

    assert_eq!(from_memory.layers(), from_disk.layers());
    assert_eq!(from_disk.input_size(), 25);
    assert_eq!(from_disk.output_size(), 198);
}

#[test]
fn network_from_bytes_rejects_garbage() {
    assert!(Network::from_bytes(&[0x42; 256]).is_err());
}

#[test]
fn input_distribution_from_bytes_matches_load() {
    let path = weights_path("scaled_input_distribution.h5");
    let from_disk = InputDistribution::load(&path).unwrap();
    let from_memory = InputDistribution::from_bytes(&std::fs::read(&path).unwrap()).unwrap();

    assert_eq!(from_memory.size(), from_disk.size());
    let x = vec![0.5; from_disk.size()];
    assert_eq!(
        from_memory.squared_mahalanobis(&x),
        from_disk.squared_mahalanobis(&x)
    );
}

#[test]
fn input_distribution_load_rejects_network_file() {
    assert!(InputDistribution::load(weights_path("nn-xxsmall.h5")).is_err());
}

/// Deterministic pseudo-random values in [-scale, scale) (64-bit LCG).
fn pseudo_random(seed: u64, n: usize, scale: f64) -> Vec<f64> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            scale * (2.0 * (state >> 11) as f64 / (1u64 << 53) as f64 - 1.0)
        })
        .collect()
}

fn network(name: &str) -> Network {
    Network::load(weights_path(&format!("nn-{name}.h5"))).unwrap()
}

/// Evaluates the Jacobian of the network along one seed column (`n_seed = 1`).
fn jacobian_column(net: &Network, x: &[f64], column: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; net.output_size()];
    let mut jy = vec![0.0; net.output_size()];
    net.evaluate_with_jacobian(x, column, 1, &mut y, &mut jy);
    jy
}

/// Extracts seed column `k` of a row-major `rows x n_seed` matrix.
fn column(matrix: &[f64], n_seed: usize, k: usize) -> Vec<f64> {
    matrix.chunks_exact(n_seed).map(|row| row[k]).collect()
}

#[test]
fn evaluate_pair_matches_evaluate() {
    for name in ["xxsmall", "medium", "xlarge"] {
        let net = network(name);
        let n_out = net.output_size();
        for case in 0..4 {
            let xa = pseudo_random(2 * case, net.input_size(), 1.0);
            let xb = pseudo_random(2 * case + 1, net.input_size(), 1.0);

            let (mut ya, mut yb) = (vec![0.0; n_out], vec![0.0; n_out]);
            net.evaluate(&xa, &mut ya);
            net.evaluate(&xb, &mut yb);
            let (mut pa, mut pb) = (vec![0.0; n_out], vec![0.0; n_out]);
            net.evaluate_pair([&xa, &xb], [&mut pa, &mut pb]);

            // Each input goes through the single-input operation sequence.
            assert_eq!(pa, ya, "{name} case {case}, first input");
            assert_eq!(pb, yb, "{name} case {case}, second input");
        }
    }
}

#[test]
fn evaluate_pair_accepts_identical_inputs_and_long_slices() {
    let net = network("xxsmall");
    let mut x = pseudo_random(7, net.input_size(), 1.0);
    let mut expected = vec![0.0; net.output_size()];
    net.evaluate(&x, &mut expected);

    // Extra trailing entries are ignored, as with `evaluate`.
    x.push(1e300);
    let mut ya = vec![-1.0; net.output_size() + 3];
    let mut yb = vec![-1.0; net.output_size() + 3];
    net.evaluate_pair([&x, &x], [&mut ya, &mut yb]);
    assert_eq!(ya[..net.output_size()], expected[..]);
    assert_eq!(yb[..net.output_size()], expected[..]);
    assert_eq!(ya[net.output_size()..], [-1.0; 3]);
}

#[test]
#[should_panic]
fn evaluate_pair_panics_on_short_input() {
    let net = network("xxsmall");
    let xa = vec![0.0; net.input_size()];
    let xb = vec![0.0; net.input_size() - 1];
    let mut ya = vec![0.0; net.output_size()];
    let mut yb = vec![0.0; net.output_size()];
    net.evaluate_pair([&xa, &xb], [&mut ya, &mut yb]);
}

#[test]
fn evaluate_with_jacobian_values_match_evaluate() {
    for name in ["xxsmall", "medium"] {
        let net = network(name);
        let x = pseudo_random(11, net.input_size(), 1.0);
        let mut expected = vec![0.0; net.output_size()];
        net.evaluate(&x, &mut expected);

        for n_seed in [0, 1, 5, 23, 33] {
            let jx = pseudo_random(12, net.input_size() * n_seed, 1.0);
            let mut y = vec![0.0; net.output_size()];
            let mut jy = vec![0.0; net.output_size() * n_seed];
            net.evaluate_with_jacobian(&x, &jx, n_seed, &mut y, &mut jy);
            assert_eq!(y, expected, "{name} with {n_seed} seeds");
        }
    }
}

#[test]
fn evaluate_with_jacobian_columns_are_independent_of_seed_count() {
    // Every seed count, including those padded internally and those wider than
    // the padded kernels, must agree with evaluating one seed column at a time.
    let net = network("medium");
    let (n_in, n_out) = (net.input_size(), net.output_size());
    let x = pseudo_random(21, n_in, 1.0);
    for n_seed in [1, 2, 5, 8, 23, 24, 30, 32, 33, 40] {
        let jx = pseudo_random(100 + n_seed as u64, n_in * n_seed, 1.0);
        let mut y = vec![0.0; n_out];
        let mut jy = vec![0.0; n_out * n_seed];
        net.evaluate_with_jacobian(&x, &jx, n_seed, &mut y, &mut jy);

        let mut failures = Failures::default();
        for k in 0..n_seed {
            let reference = jacobian_column(&net, &x, &column(&jx, n_seed, k));
            let actual = column(&jy, n_seed, k);
            for (i, (a, r)) in actual.iter().zip(&reference).enumerate() {
                failures.expect_near(*a, *r, 1e-12 * r.abs().max(1.0), || {
                    format!("{n_seed} seeds, output {i}, seed {k}")
                });
            }
        }
        failures.assert_none();
    }
}

#[test]
fn evaluate_with_jacobian_matches_finite_differences() {
    let net = network("xxsmall");
    let (n_in, n_out) = (net.input_size(), net.output_size());
    let x = pseudo_random(31, n_in, 1.0);
    let h = 1e-6;
    let mut failures = Failures::default();
    for n_seed in [1, 5, 24, 30, 40] {
        let jx = pseudo_random(200 + n_seed as u64, n_in * n_seed, 1.0);
        let mut y = vec![0.0; n_out];
        let mut jy = vec![0.0; n_out * n_seed];
        net.evaluate_with_jacobian(&x, &jx, n_seed, &mut y, &mut jy);

        for k in 0..n_seed {
            // Central difference of the network along seed direction k.
            let direction = column(&jx, n_seed, k);
            let shifted = |t: f64| -> Vec<f64> {
                let xt: Vec<f64> = x.iter().zip(&direction).map(|(x, d)| x + t * d).collect();
                let mut yt = vec![0.0; n_out];
                net.evaluate(&xt, &mut yt);
                yt
            };
            let (plus, minus) = (shifted(h), shifted(-h));
            for i in 0..n_out {
                let numeric = (plus[i] - minus[i]) / (2.0 * h);
                let analytic = jy[i * n_seed + k];
                failures.expect_near(analytic, numeric, 1e-6 * numeric.abs().max(1.0), || {
                    format!("{n_seed} seeds, output {i}, seed {k}")
                });
            }
        }
    }
    failures.assert_none();
}

#[test]
fn evaluate_with_jacobian_leaves_trailing_entries_untouched() {
    let net = network("xxsmall");
    let (n_in, n_out) = (net.input_size(), net.output_size());
    let x = pseudo_random(41, n_in, 1.0);
    for n_seed in [3, 23, 35] {
        let jx = pseudo_random(42, n_in * n_seed, 1.0);
        let mut y = vec![-1.0; n_out + 2];
        let mut jy = vec![-1.0; n_out * n_seed + 2];
        net.evaluate_with_jacobian(&x, &jx, n_seed, &mut y, &mut jy);
        assert_eq!(y[n_out..], [-1.0; 2]);
        assert_eq!(jy[n_out * n_seed..], [-1.0; 2]);
    }
}
