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

use bora::{
    KulfanParameters, NUM_CST_WEIGHTS_PER_SIDE, bl_x_points, fit_kulfan_parameters,
    generate_kulfan_coordinates,
};

fn test_parameters() -> KulfanParameters {
    KulfanParameters {
        upper_weights: [0.18, 0.22, 0.20, 0.19, 0.21, 0.18, 0.17, 0.19],
        lower_weights: [-0.12, -0.10, -0.08, -0.05, -0.02, 0.00, 0.02, 0.03],
        leading_edge_weight: 0.05,
        te_thickness: 0.002,
    }
}

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual {actual:e}, expected {expected:e} (tolerance {tolerance:e})"
    );
}

#[test]
fn coordinates_have_expected_structure() {
    let coordinates = generate_kulfan_coordinates(&test_parameters(), 100).unwrap();
    assert_eq!(coordinates.len(), 199);

    // Selig format: starts and ends at the trailing edge, leading edge in the
    // middle, and the trailing edge gap matches te_thickness.
    assert_eq!(coordinates[0][0], 1.0);
    assert_eq!(coordinates[198][0], 1.0);
    assert_eq!(coordinates[99], [0.0, 0.0]);
    assert_near(coordinates[0][1] - coordinates[198][1], 0.002, 1e-12);
}

#[test]
fn coordinates_reject_too_few_points() {
    assert!(generate_kulfan_coordinates(&test_parameters(), 1).is_err());
}

#[test]
fn fit_round_trip_recovers_parameters() {
    let original = test_parameters();
    let coordinates = generate_kulfan_coordinates(&original, 150).unwrap();
    let fitted = fit_kulfan_parameters(&coordinates).unwrap();

    for i in 0..NUM_CST_WEIGHTS_PER_SIDE {
        assert_near(fitted.upper_weights[i], original.upper_weights[i], 1e-8);
        assert_near(fitted.lower_weights[i], original.lower_weights[i], 1e-8);
    }
    assert_near(
        fitted.leading_edge_weight,
        original.leading_edge_weight,
        1e-8,
    );
    assert_near(fitted.te_thickness, original.te_thickness, 1e-8);
}

#[test]
fn fit_clamps_negative_te_thickness() {
    let parameters = KulfanParameters {
        te_thickness: 0.0,
        ..test_parameters()
    };
    let coordinates = generate_kulfan_coordinates(&parameters, 150).unwrap();
    let fitted = fit_kulfan_parameters(&coordinates).unwrap();
    assert!(fitted.te_thickness >= 0.0);
}

#[test]
fn fit_rejects_too_few_coordinates() {
    assert!(fit_kulfan_parameters(&[[1.0, 0.0], [0.0, 0.0]]).is_err());
}

#[test]
fn bl_x_points_are_uniform_midpoints() {
    let points = bl_x_points();
    assert_eq!(points[0], 1.0 / 64.0);
    assert_eq!(points[31], 63.0 / 64.0);
}
