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

//! Checks the scalar-only paths (`analyze_scalars`,
//! `analyze_scalars_with_gradients`, `Network::evaluate_first`,
//! `Network::evaluate_first_with_gradients`) against the full paths.

mod common;

use bora::{
    AeroInput, AeroOutput, AeroScalarGradients, AeroScalars, Gradient, ModelSize,
    NUM_LATENT_INPUTS, NUM_LATENT_OUTPUTS, Network, input_index,
};
use common::{Failures, WEIGHTS_DATA_DIR, input_from_array, model, output_ref_data};

/// Tolerance on a reverse-mode gradient entry relative to the largest entry
/// of the same forward-mode gradient vector. Reverse and forward mode form
/// the same sums in a different order, so they differ by rounding only. The
/// worst cases measured over every model size and test input are 5.6e-14
/// (model level) and 3.2e-14 (network level), i.e. a few hundred ulps of the
/// largest entry; the tolerance leaves ~20x headroom for other inputs.
/// Normalizing by the largest entry rather than by each entry is needed
/// because the absolute rounding error is set by the largest terms: entries
/// ~1e-6 of the largest differ by up to ~6e-11 relative to themselves.
const GRADIENT_TOLERANCE: f64 = 1e-12;

fn base_input() -> AeroInput {
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.leading_edge_weight = 0.02;
    input.kulfan.te_thickness = 0.0025;
    input.alpha = 3.0;
    input.re = 1e6;
    input
}

/// Modifies a test input in place.
type Modifier = fn(&mut AeroInput);

/// A spread of operating conditions around [`base_input`] (negative, zero and
/// high alpha, low and high Re, low n_crit, forced transition), plus every
/// Python reference case.
fn test_inputs() -> Vec<(String, AeroInput)> {
    let base = base_input();
    let variants: [(&str, Modifier); 9] = [
        ("base", |_| {}),
        ("alpha -8", |i| i.alpha = -8.0),
        ("alpha -2.5", |i| i.alpha = -2.5),
        ("alpha 0", |i| i.alpha = 0.0),
        ("alpha 14", |i| i.alpha = 14.0),
        ("Re 6e4", |i| i.re = 6e4),
        ("Re 2e7, alpha -1", |i| {
            i.re = 2e7;
            i.alpha = -1.0;
        }),
        ("n_crit 4", |i| i.n_crit = 4.0),
        ("forced transition", |i| {
            i.xtr_upper = 0.1;
            i.xtr_lower = 0.05;
        }),
    ];
    let mut inputs: Vec<(String, AeroInput)> = variants
        .into_iter()
        .map(|(name, modify)| {
            let mut input = base;
            modify(&mut input);
            (name.to_string(), input)
        })
        .collect();
    let data = output_ref_data();
    for c in 0..data.n_cases {
        let input = input_from_array(&data.inputs[c * bora::NUM_INPUTS..]);
        inputs.push((data.names[c].clone(), input));
    }
    inputs
}

fn scalar_values(o: &AeroScalars) -> [f64; 6] {
    [
        o.analysis_confidence,
        o.cl,
        o.cd,
        o.cm,
        o.top_xtr,
        o.bot_xtr,
    ]
}

fn output_values(o: &AeroOutput) -> [f64; 6] {
    [
        o.analysis_confidence,
        o.cl,
        o.cd,
        o.cm,
        o.top_xtr,
        o.bot_xtr,
    ]
}

fn scalar_gradients(g: &AeroScalarGradients) -> [Gradient; 6] {
    [
        g.analysis_confidence,
        g.cl,
        g.cd,
        g.cm,
        g.top_xtr,
        g.bot_xtr,
    ]
}

const SCALAR_NAMES: [&str; 6] = [
    "analysis_confidence",
    "CL",
    "CD",
    "CM",
    "Top_Xtr",
    "Bot_Xtr",
];

#[test]
fn analyze_scalars_matches_analyze_bitwise() {
    let inputs = test_inputs();
    for size in ModelSize::ALL {
        let model = model(size.as_str());
        for (name, input) in &inputs {
            let full = output_values(&model.analyze(input));
            let scalars = scalar_values(&model.analyze_scalars(input));
            assert_eq!(scalars, full, "{size} {name}");
        }
    }
}

#[test]
fn scalar_gradients_match_full_gradients() {
    let inputs = test_inputs();
    let mut failures = Failures::default();
    let mut worst = (0.0_f64, String::new());
    let (mut clipped_cases, mut negative_alpha_cases) = (0, 0);
    for size in ModelSize::ALL {
        let model = model(size.as_str());
        for (name, input) in &inputs {
            let full = model.analyze_with_gradients(input);
            let scalar = model.analyze_scalars_with_gradients(input);

            // Values come from the same operation sequence as `analyze`.
            let full_values = output_values(&full.output);
            assert_eq!(scalar_values(&scalar.output), full_values, "{size} {name}");
            assert_eq!(
                scalar_values(&scalar.output),
                output_values(&model.analyze(input)),
                "{size} {name}"
            );

            let full_grads = [
                full.gradient.analysis_confidence,
                full.gradient.cl,
                full.gradient.cd,
                full.gradient.cm,
                full.gradient.top_xtr,
                full.gradient.bot_xtr,
            ];
            for (o, (actual, expected)) in scalar_gradients(&scalar.gradient)
                .iter()
                .zip(&full_grads)
                .enumerate()
            {
                let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                if scale == 0.0 {
                    // Clipped transition location: the gradient is exactly zero.
                    assert_eq!(actual, &[0.0; bora::NUM_INPUTS], "{size} {name}");
                    clipped_cases += 1;
                    continue;
                }
                for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
                    let error = (a - e).abs() / scale;
                    if error > worst.0 {
                        worst = (
                            error,
                            format!("{size} {name} {} input {k}", SCALAR_NAMES[o]),
                        );
                    }
                    failures.expect_near(*a, *e, GRADIENT_TOLERANCE * scale, || {
                        format!("{size} {name}: d{}/d(input {k})", SCALAR_NAMES[o])
                    });
                }
            }
            if input.alpha < 0.0 {
                negative_alpha_cases += 1;
            }
        }
    }
    eprintln!(
        "worst scalar gradient error relative to the gradient's largest entry: {:.2e} ({})",
        worst.0, worst.1
    );
    // The inputs must exercise the transition clip and negative alpha.
    assert!(
        clipped_cases > 0,
        "no test input activates a transition clip"
    );
    assert!(negative_alpha_cases > 0);
    failures.assert_none();
}

#[test]
fn scalar_gradients_include_alpha_and_reynolds_chain() {
    // The alpha column collects three latent rows and Re enters through a
    // logarithm; check both against the full path at a negative alpha with a
    // tight tolerance, entry by entry.
    let model = model("xlarge");
    let mut input = base_input();
    input.alpha = -6.5;
    let full = model.analyze_with_gradients(&input).gradient;
    let scalar = model.analyze_scalars_with_gradients(&input).gradient;
    for k in [input_index::ALPHA, input_index::RE] {
        for (a, e) in [(scalar.cl[k], full.cl[k]), (scalar.cd[k], full.cd[k])] {
            assert!(e != 0.0);
            assert!((a - e).abs() <= GRADIENT_TOLERANCE * e.abs(), "{a} vs {e}");
        }
    }
}

fn network(size: ModelSize) -> Network {
    Network::load(format!("{WEIGHTS_DATA_DIR}/nn-{size}.h5")).unwrap()
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

#[test]
fn evaluate_first_matches_evaluate_bitwise() {
    for size in [ModelSize::XXSmall, ModelSize::Medium, ModelSize::XLarge] {
        let net = network(size);
        let n_out = net.output_size();
        let x = pseudo_random(3, net.input_size(), 1.0);
        let mut expected = vec![0.0; n_out];
        net.evaluate(&x, &mut expected);
        for n in [0, 1, 6, 17, 100, n_out] {
            let mut y = vec![-1.0; n_out + 1];
            net.evaluate_first(&x, n, &mut y);
            assert_eq!(y[..n], expected[..n], "{size} with {n} outputs");
            assert!(
                y[n..].iter().all(|v| *v == -1.0),
                "{size}: trailing entries"
            );
        }
    }
}

#[test]
#[should_panic]
fn evaluate_first_panics_on_too_many_outputs() {
    let net = network(ModelSize::XXSmall);
    let x = vec![0.0; net.input_size()];
    let mut y = vec![0.0; net.output_size() + 1];
    net.evaluate_first(&x, net.output_size() + 1, &mut y);
}

#[test]
#[should_panic]
fn evaluate_first_with_gradients_panics_on_short_gradient() {
    let net = network(ModelSize::XXSmall);
    let x = vec![0.0; net.input_size()];
    let mut y = vec![0.0; 6];
    let mut grad = vec![0.0; 6 * net.input_size() - 1];
    net.evaluate_first_with_gradients(&x, 6, &mut y, &mut grad);
}

#[test]
fn evaluate_first_with_gradients_matches_forward_mode() {
    let mut worst = 0.0_f64;
    let mut failures = Failures::default();
    for size in ModelSize::ALL {
        let net = network(size);
        let (n_in, n_out) = (net.input_size(), net.output_size());
        assert_eq!((n_in, n_out), (NUM_LATENT_INPUTS, NUM_LATENT_OUTPUTS));
        // Identity seeds: the forward-mode Jacobian is d(outputs)/d(inputs).
        let mut identity = vec![0.0; n_in * n_in];
        for i in 0..n_in {
            identity[i * n_in + i] = 1.0;
        }
        for case in 0..2 {
            let x = pseudo_random(50 + case, n_in, 1.0);
            let mut y_ref = vec![0.0; n_out];
            let mut jac_ref = vec![0.0; n_out * n_in];
            net.evaluate_with_jacobian(&x, &identity, n_in, &mut y_ref, &mut jac_ref);

            // Output counts below, at and above one reverse sweep's lanes.
            for n in [0, 1, 6, 16, 17, 40] {
                let mut y = vec![-1.0; n + 1];
                let mut grad = vec![-1.0; n * n_in + 1];
                net.evaluate_first_with_gradients(&x, n, &mut y, &mut grad);
                assert_eq!(y[..n], y_ref[..n], "{size} with {n} outputs");
                assert_eq!((y[n], grad[n * n_in]), (-1.0, -1.0), "trailing entries");

                for q in 0..n {
                    let expected = &jac_ref[q * n_in..][..n_in];
                    let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                    for (i, (a, e)) in grad[q * n_in..][..n_in].iter().zip(expected).enumerate() {
                        worst = worst.max((a - e).abs() / scale);
                        failures.expect_near(*a, *e, GRADIENT_TOLERANCE * scale, || {
                            format!("{size} case {case}, {n} outputs: d(output {q})/d(input {i})")
                        });
                    }
                }
            }
        }
    }
    eprintln!("worst network gradient error relative to the row's largest entry: {worst:.2e}");
    failures.assert_none();
}
