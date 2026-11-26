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

//! Batched evaluation (`Network::evaluate_batch`, `NeuralFoil::analyze_batch`,
//! `NeuralFoil::analyze_scalars_batch`) against the single-point calls.

mod common;

use bora::{
    AeroInput, AeroOutput, AeroScalars, BATCH_BLOCK_SIZE, BatchWorkspace, ModelSize,
    NUM_LATENT_INPUTS, NUM_LATENT_OUTPUTS, Network,
};
use common::{Failures, WEIGHTS_DATA_DIR, model};

/// Relative tolerance between batched and single-point results.
const RTOL: f64 = 1e-12;

/// Batch lengths covering empty, single, around one network block (in points
/// and in columns: each point is two columns), and a long batch.
const LENGTHS: [usize; 11] = [
    0,
    1,
    2,
    3,
    BATCH_BLOCK_SIZE / 2,
    BATCH_BLOCK_SIZE / 2 + 1,
    BATCH_BLOCK_SIZE - 1,
    BATCH_BLOCK_SIZE,
    BATCH_BLOCK_SIZE + 1,
    2 * BATCH_BLOCK_SIZE + 3,
    301,
];

/// Deterministic xorshift generator of uniform numbers in [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
}

/// Inputs varying the Kulfan weights, alpha, Re, n_crit, and forced transition.
fn random_inputs(n: usize, seed: u64) -> Vec<AeroInput> {
    let mut rng = Rng(seed);
    let upper = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    let lower = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    (0..n)
        .map(|_| {
            let mut input = AeroInput::default();
            for i in 0..upper.len() {
                input.kulfan.upper_weights[i] = upper[i] + rng.uniform(-0.1, 0.1);
                input.kulfan.lower_weights[i] = lower[i] + rng.uniform(-0.1, 0.1);
            }
            input.kulfan.leading_edge_weight = rng.uniform(-0.2, 0.5);
            input.kulfan.te_thickness = rng.uniform(0.0, 0.01);
            input.alpha = rng.uniform(-15.0, 20.0);
            input.re = 10f64.powf(rng.uniform(4.0, 8.0));
            input.n_crit = rng.uniform(4.0, 14.0);
            // Forced transition on about half of the cases.
            if rng.next() < 0.5 {
                input.xtr_upper = rng.uniform(0.0, 1.0);
                input.xtr_lower = rng.uniform(0.0, 1.0);
            }
            input
        })
        .collect()
}

/// Every field of an output, with its name and an absolute scale below which
/// differences are compared absolutely: quantities decoded linearly from the
/// network outputs (CL, CM, transition locations, ue) inherit the absolute
/// rounding error of their O(1) latent value, so relative comparison of values
/// near zero would only measure cancellation. `re` is the input Reynolds number.
fn fields(output: &AeroOutput, re: f64) -> Vec<(String, f64, f64)> {
    let mut list = vec![
        (
            "analysis_confidence".to_string(),
            output.analysis_confidence,
            1.0,
        ),
        ("cl".to_string(), output.cl, 1.0),
        ("cd".to_string(), output.cd, 0.0),
        ("cm".to_string(), output.cm, 0.05),
        ("top_xtr".to_string(), output.top_xtr, 1.0),
        ("bot_xtr".to_string(), output.bot_xtr, 1.0),
    ];
    for (side, bl) in [("upper", &output.upper), ("lower", &output.lower)] {
        for i in 0..bl.theta.len() {
            // theta = (10^t - 0.1) / (|ue| Re) inherits the absolute error of
            // the latent ue relative to |ue| (small near stagnation), and that
            // of 10^t on the scale of the 0.1 term.
            let ue = bl.ue_over_vinf[i].abs();
            let theta_scale = (bl.theta[i].abs() / ue).max(0.1 / (ue * re));
            list.push((format!("{side}.theta[{i}]"), bl.theta[i], theta_scale));
            list.push((format!("{side}.h[{i}]"), bl.h[i], 0.0));
            list.push((format!("{side}.ue_over_vinf[{i}]"), bl.ue_over_vinf[i], 1.0));
        }
    }
    list
}

fn compare_outputs(
    failures: &mut Failures,
    actual: &AeroOutput,
    expected: &AeroOutput,
    re: f64,
    context: &str,
) {
    let pairs = fields(actual, re).into_iter().zip(fields(expected, re));
    for ((name, a, _), (_, e, scale)) in pairs {
        failures.expect_near(a, e, RTOL * e.abs().max(scale), || {
            format!("{context}: {name}")
        });
    }
}

#[test]
fn analyze_batch_matches_analyze_for_all_sizes_and_lengths() {
    for size in ModelSize::ALL {
        let model = model(size.as_str());
        let mut failures = Failures::default();
        for (seed, &n) in LENGTHS.iter().enumerate() {
            let inputs = random_inputs(n, 0x9E37_79B9_7F4A_7C15 ^ seed as u64);
            let batch = model.analyze_batch(&inputs);
            assert_eq!(batch.len(), n);
            for (i, (input, output)) in inputs.iter().zip(&batch).enumerate() {
                let context = format!("{size}, length {n}, point {i}");
                compare_outputs(
                    &mut failures,
                    output,
                    &model.analyze(input),
                    input.re,
                    &context,
                );
            }
        }
        failures.assert_none();
    }
}

#[test]
fn analyze_batch_into_matches_analyze_batch_and_reuses_workspace() {
    let mut workspace = BatchWorkspace::default();
    // Alternate sizes so one workspace serves networks of different widths.
    for size in [
        ModelSize::XXXLarge,
        ModelSize::XXSmall,
        ModelSize::Medium,
        ModelSize::XLarge,
    ] {
        let model = model(size.as_str());
        let inputs = random_inputs(37, 7);
        let mut outputs = vec![AeroOutput::default(); inputs.len()];
        model.analyze_batch_into(&inputs, &mut outputs, &mut workspace);
        assert_eq!(outputs, model.analyze_batch(&inputs), "{size}");
    }
}

/// The scalar fields of an [`AeroOutput`], for comparison with [`AeroScalars`].
fn scalars_of(output: &AeroOutput) -> AeroScalars {
    let mut scalars = AeroScalars::default();
    scalars.analysis_confidence = output.analysis_confidence;
    scalars.cl = output.cl;
    scalars.cd = output.cd;
    scalars.cm = output.cm;
    scalars.top_xtr = output.top_xtr;
    scalars.bot_xtr = output.bot_xtr;
    scalars
}

/// An [`AeroOutput`] carrying only the given scalars, so that the scalar
/// fields can be compared with [`compare_outputs`].
fn output_of(scalars: &AeroScalars) -> AeroOutput {
    let mut output = AeroOutput::default();
    output.analysis_confidence = scalars.analysis_confidence;
    output.cl = scalars.cl;
    output.cd = scalars.cd;
    output.cm = scalars.cm;
    output.top_xtr = scalars.top_xtr;
    output.bot_xtr = scalars.bot_xtr;
    output
}

#[test]
fn analyze_scalars_batch_matches_analyze_scalars_for_all_sizes_and_lengths() {
    for size in ModelSize::ALL {
        let model = model(size.as_str());
        let mut failures = Failures::default();
        for (seed, &n) in LENGTHS.iter().enumerate() {
            let inputs = random_inputs(n, 0x2545_F491_4F6C_DD1D ^ seed as u64);
            let batch = model.analyze_scalars_batch(&inputs);
            assert_eq!(batch.len(), n);
            for (i, (input, scalars)) in inputs.iter().zip(&batch).enumerate() {
                let context = format!("{size}, length {n}, point {i}");
                compare_outputs(
                    &mut failures,
                    &output_of(scalars),
                    &output_of(&model.analyze_scalars(input)),
                    input.re,
                    &context,
                );
            }
            // The scalar path skips the boundary layer rows of the output
            // layer, which does not change the scalar rows' arithmetic.
            let full: Vec<AeroScalars> = model
                .analyze_batch(&inputs)
                .iter()
                .map(scalars_of)
                .collect();
            assert_eq!(batch, full, "{size}, length {n}");
        }
        failures.assert_none();
    }
}

#[test]
fn analyze_scalars_batch_into_matches_analyze_scalars_batch_and_reuses_workspace() {
    let mut workspace = BatchWorkspace::default();
    for size in [ModelSize::XLarge, ModelSize::XXSmall, ModelSize::Large] {
        let model = model(size.as_str());
        let inputs = random_inputs(29, 11);
        let mut outputs = vec![AeroScalars::default(); inputs.len()];
        model.analyze_scalars_batch_into(&inputs, &mut outputs, &mut workspace);
        assert_eq!(outputs, model.analyze_scalars_batch(&inputs), "{size}");
    }
}

#[test]
#[should_panic(expected = "differ in length")]
fn analyze_scalars_batch_into_rejects_mismatched_lengths() {
    let inputs = random_inputs(3, 1);
    let mut outputs = vec![AeroScalars::default(); 4];
    model("xxsmall").analyze_scalars_batch_into(
        &inputs,
        &mut outputs,
        &mut BatchWorkspace::default(),
    );
}

#[test]
#[should_panic(expected = "differ in length")]
fn analyze_batch_into_rejects_mismatched_lengths() {
    let inputs = random_inputs(3, 1);
    let mut outputs = vec![AeroOutput::default(); 2];
    model("xxsmall").analyze_batch_into(&inputs, &mut outputs, &mut BatchWorkspace::default());
}

#[test]
fn batch_results_do_not_depend_on_position_or_neighbours() {
    let probes = random_inputs(5, 42);
    // Extreme neighbours: large angles, very low and high Re, forced transition.
    let mut extreme = random_inputs(2 * BATCH_BLOCK_SIZE + 1, 99);
    for (i, input) in extreme.iter_mut().enumerate() {
        input.alpha = if i % 2 == 0 { 45.0 } else { -40.0 };
        input.re = if i % 3 == 0 { 1e3 } else { 1e9 };
        input.xtr_upper = 0.0;
    }

    for size in [ModelSize::XXSmall, ModelSize::Large, ModelSize::XXXLarge] {
        let model = model(size.as_str());
        let mut failures = Failures::default();
        for (p, probe) in probes.iter().enumerate() {
            let reference = model.analyze(probe);
            // Place the probe at every offset within the first blocks, among
            // the extreme neighbours, and at the end of batches of all lengths.
            for offset in 0..=extreme.len() {
                let mut batch = extreme.clone();
                batch.insert(offset, *probe);
                let output = model.analyze_batch(&batch)[offset];
                let context = format!("{size}, probe {p} at offset {offset}");
                compare_outputs(&mut failures, &output, &reference, probe.re, &context);
                let tail = model.analyze_batch(&batch[..=offset])[offset];
                compare_outputs(
                    &mut failures,
                    &tail,
                    &reference,
                    probe.re,
                    &format!("{context} (tail)"),
                );
            }
        }
        failures.assert_none();
    }
}

/// Deterministic latent inputs in [-2, 2].
fn latent_points(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    (0..n * NUM_LATENT_INPUTS)
        .map(|_| rng.uniform(-2.0, 2.0))
        .collect()
}

#[test]
fn network_evaluate_batch_matches_evaluate() {
    let mut workspace = BatchWorkspace::default();
    for size in ModelSize::ALL {
        let network = Network::load(format!("{WEIGHTS_DATA_DIR}/nn-{size}.h5")).unwrap();
        let mut failures = Failures::default();
        for &n in &[0, 1, 2, 3, 7, 8, 9, 16, 17, 100] {
            // Point-major inputs for `evaluate`; feature-major for the batch.
            let points = latent_points(n, 0x1234_5678 + n as u64);
            let mut x = vec![0.0; NUM_LATENT_INPUTS * n];
            for p in 0..n {
                for j in 0..NUM_LATENT_INPUTS {
                    x[j * n + p] = points[p * NUM_LATENT_INPUTS + j];
                }
            }
            let mut y = vec![f64::NAN; NUM_LATENT_OUTPUTS * n];
            network.evaluate_batch(&x, &mut y, n, &mut workspace);

            for p in 0..n {
                let mut expected = [0.0; NUM_LATENT_OUTPUTS];
                network.evaluate(&points[p * NUM_LATENT_INPUTS..], &mut expected);
                for (i, &e) in expected.iter().enumerate() {
                    let actual = y[i * n + p];
                    if n == 1 {
                        // A lone point uses the single-point kernel.
                        assert_eq!(actual.to_bits(), e.to_bits(), "{size}, output {i}");
                    }
                    // Latent outputs are O(1); compare absolutely below 1.
                    failures.expect_near(actual, e, RTOL * e.abs().max(1.0), || {
                        format!("{size}, n = {n}, point {p}, output {i}")
                    });
                }
            }
        }
        failures.assert_none();
    }
}

#[test]
fn network_evaluate_batch_leaves_extra_output_untouched() {
    let network = Network::load(format!("{WEIGHTS_DATA_DIR}/nn-xsmall.h5")).unwrap();
    let n = BATCH_BLOCK_SIZE + 3;
    let x = latent_points(n, 5);
    let mut y = vec![-7.0; NUM_LATENT_OUTPUTS * n + 4];
    network.evaluate_batch(&x, &mut y, n, &mut BatchWorkspace::default());
    assert!(y[..NUM_LATENT_OUTPUTS * n].iter().all(|v| v.is_finite()));
    assert_eq!(y[NUM_LATENT_OUTPUTS * n..], [-7.0; 4]);
}
