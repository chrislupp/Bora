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
