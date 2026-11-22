//! Timing harness for the network kernels and the end-to-end analysis calls.
//!
//! Run with `cargo bench` (all model sizes) or `cargo bench -- medium xlarge`
//! to restrict the run to the named sizes. Times are best-of-several averages
//! in microseconds per call; std only, so no extra dependencies are needed.

use std::hint::black_box;
use std::time::Instant;

use bora::{AeroInput, ModelSize, NUM_INPUTS, NUM_LATENT_INPUTS, NUM_LATENT_OUTPUTS, NeuralFoil};

/// Minimum wall time of one timed batch, in seconds.
const MIN_BATCH_SECONDS: f64 = 0.05;
/// Number of timed batches; the fastest is reported.
const REPETITIONS: usize = 7;
/// Number of scalar network outputs (confidence, CL, CD, CM, transitions).
const SCALARS: usize = 6;

/// Returns the best-of-`REPETITIONS` time per call of `f`, in microseconds.
/// The batch size doubles until a batch takes at least `MIN_BATCH_SECONDS`.
fn time<F: FnMut()>(mut f: F) -> f64 {
    let mut n = 1_u64;
    loop {
        let start = Instant::now();
        for _ in 0..n {
            f();
        }
        if start.elapsed().as_secs_f64() > MIN_BATCH_SECONDS {
            break;
        }
        n *= 2;
    }
    let mut best = f64::MAX;
    for _ in 0..REPETITIONS {
        let start = Instant::now();
        for _ in 0..n {
            f();
        }
        best = best.min(start.elapsed().as_secs_f64() / n as f64);
    }
    best * 1e6
}

fn sample_input() -> AeroInput {
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.18, 0.22, 0.25, 0.24, 0.23, 0.2, 0.19, 0.2];
    input.kulfan.lower_weights = [-0.16, -0.1, -0.12, -0.05, -0.08, -0.03, 0.02, 0.05];
    input.kulfan.leading_edge_weight = 0.1;
    input.kulfan.te_thickness = 0.002;
    input.alpha = 4.0;
    input.re = 2e6;
    input
}

fn main() {
    // `cargo bench` passes flags such as `--bench`; bare words select sizes.
    let filters: Vec<String> = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with('-'))
        .collect();
    let sizes: Vec<ModelSize> = ModelSize::ALL
        .into_iter()
        .filter(|size| filters.is_empty() || filters.iter().any(|f| f == size.as_str()))
        .collect();

    let input = sample_input();
    let x: Vec<f64> = (0..NUM_LATENT_INPUTS)
        .map(|i| 0.1 * ((i * 7 % 11) as f64 - 5.0) / 5.0)
        .collect();
    let mut jx = vec![0.0; NUM_LATENT_INPUTS * NUM_INPUTS];
    for i in 0..NUM_LATENT_INPUTS {
        jx[i * NUM_INPUTS + i % NUM_INPUTS] = 1.0 + 0.1 * i as f64;
    }
    let mut y = vec![0.0; NUM_LATENT_OUTPUTS];
    let mut jy = vec![0.0; NUM_LATENT_OUTPUTS * NUM_INPUTS];
    // Second input of the pair pass: the alpha-symmetry flip negates most
    // latent inputs, so a negated copy is representative.
    let x2: Vec<f64> = x.iter().map(|v| -v).collect();
    let mut y2 = y.clone();

    let mut grad = vec![0.0; SCALARS * NUM_LATENT_INPUTS];

    println!(
        "{:<9} {:>24} | {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9} {:>9} {:>9}",
        "size",
        "hidden layers",
        "analyze",
        "w/ grads",
        "scalars",
        "s/ grads",
        "evaluate",
        "eval+jac",
        "pair",
        "first+grd"
    );
    for size in sizes {
        let model = NeuralFoil::new(size).expect("model weights load");
        let net = bora::Network::load(format!("{}/data/nn-{size}.h5", env!("CARGO_MANIFEST_DIR")))
            .expect("network weights load");
        let layers = net.layers();
        let shape: Vec<String> = layers[..layers.len() - 1]
            .iter()
            .map(|layer| layer.rows.to_string())
            .collect();

        let t_analyze = time(|| {
            black_box(model.analyze(black_box(&input)));
        });
        let t_gradients = time(|| {
            black_box(model.analyze_with_gradients(black_box(&input)));
        });
        let t_evaluate = time(|| {
            net.evaluate(black_box(&x), &mut y);
            black_box(&y);
        });
        let t_jacobian = time(|| {
            net.evaluate_with_jacobian(black_box(&x), black_box(&jx), NUM_INPUTS, &mut y, &mut jy);
            black_box(&jy);
        });
        let t_pair = time(|| {
            net.evaluate_pair([black_box(&x), black_box(&x2)], [&mut y, &mut y2]);
            black_box((&y, &y2));
        });
        let t_scalars = time(|| {
            black_box(model.analyze_scalars(black_box(&input)));
        });
        let t_scalar_gradients = time(|| {
            black_box(model.analyze_scalars_with_gradients(black_box(&input)));
        });
        let t_first_gradients = time(|| {
            net.evaluate_first_with_gradients(black_box(&x), SCALARS, &mut y, &mut grad);
            black_box(&grad);
        });

        println!(
            "{:<9} {:>24} | {:>9.2} {:>9.2} {:>9.2} {:>9.2} | {:>9.2} {:>9.2} {:>9.2} {:>9.2}",
            size.as_str(),
            shape.join("-"),
            t_analyze,
            t_gradients,
            t_scalars,
            t_scalar_gradients,
            t_evaluate,
            t_jacobian,
            t_pair,
            t_first_gradients
        );
    }
    println!(
        "(microseconds per call; scalars / s/ grads: analyze_scalars / \
         analyze_scalars_with_gradients; evaluate: one input, eval+jac: one input with 23 seeds, \
         pair: evaluate_pair with two inputs, first+grd: evaluate_first_with_gradients for the \
         {SCALARS} scalar outputs of one input)"
    );
}
