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

//! Checks the forward-mode Jacobian-vector product (`analyze_with_tangent`,
//! `Network::evaluate_with_tangent`) against the full forward-mode gradients
//! contracted with the same direction.

mod common;

use bora::{
    AeroGradients, AeroInput, AeroOutput, Gradient, ModelSize, NUM_INPUTS, Network, input_index,
};
use common::{Failures, WEIGHTS_DATA_DIR, input_from_array, model, output_ref_data};

/// Relative tolerance of the tangent against the contracted gradient. The two
/// differ only in floating-point summation order.
const REL_TOL: f64 = 1e-10;

/// Groups every output field by quantity, in a fixed order: each scalar is a
/// group of one, each boundary-layer array (per surface) a group of
/// `NUM_BL_POINTS` stations.
fn group_output(out: &AeroOutput) -> Vec<(String, Vec<f64>)> {
    let mut v = vec![
        (
            "analysis_confidence".to_string(),
            vec![out.analysis_confidence],
        ),
        ("cl".to_string(), vec![out.cl]),
        ("cd".to_string(), vec![out.cd]),
        ("cm".to_string(), vec![out.cm]),
        ("top_xtr".to_string(), vec![out.top_xtr]),
        ("bot_xtr".to_string(), vec![out.bot_xtr]),
    ];
    for (side, bl) in [("upper", &out.upper), ("lower", &out.lower)] {
        v.push((format!("{side} theta"), bl.theta.to_vec()));
        v.push((format!("{side} h"), bl.h.to_vec()));
        v.push((format!("{side} ue_over_vinf"), bl.ue_over_vinf.to_vec()));
    }
    v
}

/// Groups every gradient in the same order as [`group_output`].
fn group_gradients(g: &AeroGradients) -> Vec<Vec<Gradient>> {
    let mut v = vec![
        vec![g.analysis_confidence],
        vec![g.cl],
        vec![g.cd],
        vec![g.cm],
        vec![g.top_xtr],
        vec![g.bot_xtr],
    ];
    for bl in [&g.upper, &g.lower] {
        v.push(bl.theta.to_vec());
        v.push(bl.h.to_vec());
        v.push(bl.ue_over_vinf.to_vec());
    }
    v
}

/// Test inputs spanning alpha, Re, and airfoil shapes: synthetic cases with
/// non-default Kulfan weights plus a few reference cases.
fn test_inputs() -> Vec<(String, AeroInput)> {
    let mut cases = Vec::new();
    let base_upper = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    let base_lower = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    for (j, (alpha, re)) in [(-8.0, 5e4), (0.0, 3e5), (3.0, 1e6), (9.5, 4e6), (15.0, 2e7)]
        .into_iter()
        .enumerate()
    {
        let mut input = AeroInput {
            alpha,
            re,
            n_crit: 7.0 + j as f64,
            ..AeroInput::default()
        };
        let s = j as f64;
        input.kulfan.upper_weights =
            std::array::from_fn(|i| base_upper[i] * (1.0 + 0.1 * (s + i as f64).sin()));
        input.kulfan.lower_weights =
            std::array::from_fn(|i| base_lower[i] + 0.03 * (0.7 * s + i as f64).cos());
        input.kulfan.leading_edge_weight = 0.05 * (s - 2.0);
        input.kulfan.te_thickness = 0.001 * s;
        if j == 3 {
            // Forced transition, so the xtr clip is inactive on the upper side.
            input.xtr_upper = 0.4;
            input.xtr_lower = 0.7;
        }
        cases.push((format!("synthetic {j} (alpha {alpha}, Re {re:e})"), input));
    }

    let data = output_ref_data();
    for case in [1, 21, 41, 61] {
        let input = input_from_array(&data.inputs[case * NUM_INPUTS..]);
        cases.push((format!("reference {}", data.names[case]), input));
    }
    cases
}

/// Unit directions at alpha, Re, and two CST weights, plus a dense direction
/// with every component nonzero (Re scaled so its contribution is comparable).
fn test_directions(input: &AeroInput) -> Vec<(String, Gradient)> {
    let unit = |k: usize| {
        let mut d = [0.0; NUM_INPUTS];
        d[k] = 1.0;
        d
    };
    let mut dense: Gradient = std::array::from_fn(|k| (0.9 * k as f64 + 0.3).sin());
    dense[input_index::RE] *= 0.3 * input.re;
    dense[input_index::TE_THICKNESS] *= 0.01;
    vec![
        ("alpha".into(), unit(input_index::ALPHA)),
        ("Re".into(), unit(input_index::RE)),
        (
            "upper weight 3".into(),
            unit(input_index::UPPER_WEIGHTS + 3),
        ),
        (
            "lower weight 5".into(),
            unit(input_index::LOWER_WEIGHTS + 5),
        ),
        ("dense".into(), dense),
    ]
}

fn check_model(size: ModelSize) {
    let model = model(size.as_str());
    let mut failures = Failures::default();

    for (case_name, input) in test_inputs() {
        let plain = model.analyze(&input);
        let full = model.analyze_with_gradients(&input);
        let grads = group_gradients(&full.gradient);

        for (dir_name, direction) in test_directions(&input) {
            let result = model.analyze_with_tangent(&input, &direction);
            assert_eq!(
                result.output, plain,
                "{size} {case_name} {dir_name}: tangent output differs from analyze"
            );

            let tangents = group_output(&result.tangent);
            assert_eq!(tangents.len(), grads.len());
            for ((name, actual), g) in tangents.iter().zip(&grads) {
                // Contracted gradient and its summation-order error scale
                // (the summed magnitudes of its terms) at each entry.
                let (expected, scale): (Vec<f64>, Vec<f64>) = g
                    .iter()
                    .map(|g| {
                        let terms = g.iter().zip(&direction).map(|(g, d)| g * d);
                        (terms.clone().sum::<f64>(), terms.map(f64::abs).sum::<f64>())
                    })
                    .unzip();
                // Boundary-layer arrays are compared norm-wise: the latent
                // outputs of the plain and flipped passes can cancel at a
                // single station (by ~1e5 for some Re derivatives), so the
                // error floor is set by the array's largest entry scale.
                let array_scale = scale.iter().copied().fold(0.0, f64::max);
                for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    failures.expect_near(*a, *e, REL_TOL * array_scale, || {
                        format!("{size} {case_name} d/d({dir_name}) {name}[{i}]")
                    });
                }
            }
        }
    }
    failures.assert_none();
}

#[test]
fn tangent_matches_contracted_gradients_xxsmall() {
    check_model(ModelSize::XXSmall);
}

#[test]
fn tangent_matches_contracted_gradients_xsmall() {
    check_model(ModelSize::XSmall);
}

#[test]
fn tangent_matches_contracted_gradients_small() {
    check_model(ModelSize::Small);
}

#[test]
fn tangent_matches_contracted_gradients_medium() {
    check_model(ModelSize::Medium);
}

#[test]
fn tangent_matches_contracted_gradients_large() {
    check_model(ModelSize::Large);
}

#[test]
fn tangent_matches_contracted_gradients_xlarge() {
    check_model(ModelSize::XLarge);
}

#[test]
fn tangent_matches_contracted_gradients_xxlarge() {
    check_model(ModelSize::XXLarge);
}

#[test]
fn tangent_matches_contracted_gradients_xxxlarge() {
    check_model(ModelSize::XXXLarge);
}

#[test]
fn zero_direction_gives_zero_tangent() {
    let model = model("xxsmall");
    let input = test_inputs().remove(2).1;
    let result = model.analyze_with_tangent(&input, &[0.0; NUM_INPUTS]);
    assert_eq!(result.output, model.analyze(&input));
    for (name, values) in group_output(&result.tangent) {
        assert!(values.iter().all(|v| *v == 0.0), "{name}: {values:?}");
    }
}

#[test]
fn network_tangent_matches_single_seed_jacobian() {
    for size in ModelSize::ALL {
        let net = Network::load(format!("{WEIGHTS_DATA_DIR}/nn-{size}.h5")).unwrap();
        let (n_in, n_out) = (net.input_size(), net.output_size());
        let x: Vec<f64> = (0..n_in).map(|i| (0.37 * i as f64).sin()).collect();
        let dx: Vec<f64> = (0..n_in).map(|i| (0.11 * i as f64 + 0.5).cos()).collect();

        let mut y_ref = vec![0.0; n_out];
        net.evaluate(&x, &mut y_ref);
        let mut y_jac = vec![0.0; n_out];
        let mut dy_jac = vec![0.0; n_out];
        net.evaluate_with_jacobian(&x, &dx, 1, &mut y_jac, &mut dy_jac);
        let mut y = vec![0.0; n_out];
        let mut dy = vec![0.0; n_out];
        net.evaluate_with_tangent(&x, &dx, &mut y, &mut dy);

        assert_eq!(y, y_ref, "{size}: tangent value differs from evaluate");
        let mut failures = Failures::default();
        for (i, (a, e)) in dy.iter().zip(&dy_jac).enumerate() {
            failures.expect_near(*a, *e, REL_TOL * e.abs().max(1e-3), || {
                format!("{size} dy[{i}]")
            });
        }
        failures.assert_none();
    }
}
