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

//! Verifies Bora's analytic gradients against jacobians of the Python
//! NeuralFoil computed with CasADi. The reference data lives in
//! tests/data/gradient_ref_data.h5 (committed to the repository) and is
//! regenerated with scripts/generate_gradient_ref_data.py only when the
//! reference data needs updating.

mod common;

use bora::{AeroGradients, Gradient, NUM_INPUTS};
use common::{Failures, gradient_ref_data, input_from_array, model};

/// Collects Bora's gradients in the reference jacobian row order: the six
/// scalars, then per surface and quantity the recorded stations.
fn jacobian_rows<'a>(gradient: &'a AeroGradients, stations: &[usize]) -> Vec<&'a Gradient> {
    let mut rows = vec![
        &gradient.analysis_confidence,
        &gradient.cl,
        &gradient.cd,
        &gradient.cm,
        &gradient.top_xtr,
        &gradient.bot_xtr,
    ];
    for surface in [&gradient.upper, &gradient.lower] {
        for quantity in [&surface.theta, &surface.h, &surface.ue_over_vinf] {
            rows.extend(stations.iter().map(|&station| &quantity[station]));
        }
    }
    rows
}

#[test]
fn matches_python_neuralfoil_jacobians() {
    let data = gradient_ref_data();
    assert!(data.n_cases > 0);

    let mut failures = Failures::default();
    for c in 0..data.n_cases {
        let model = model(&data.model_sizes[c]);
        let result =
            model.analyze_with_gradients(&input_from_array(&data.inputs[c * NUM_INPUTS..]));
        let rows = jacobian_rows(&result.gradient, &data.bl_stations);
        assert_eq!(rows.len(), data.n_rows);

        let expected_case = &data.jacobians[c * data.n_rows * NUM_INPUTS..];
        for (row, actual_row) in rows.iter().enumerate() {
            for k in 0..NUM_INPUTS {
                let expected = expected_case[row * NUM_INPUTS + k];
                let tolerance = 1e-6 * expected.abs().max(1e-8);
                failures.expect_near(actual_row[k], expected, tolerance, || {
                    format!("{} jacobian row {row}, input {k}", data.names[c])
                });
            }
        }
    }
    failures.assert_none();
}
