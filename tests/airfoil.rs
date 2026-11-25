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

//! Coordinate-based analysis against NeuralFoil's `get_aero_from_coordinates`,
//! and the `.dat` reader.

mod common;

use bora::{
    Error, FittedAirfoil, OperatingPoint, Point2, generate_kulfan_coordinates,
    normalize_coordinates, parse_dat, read_dat_file,
};
use common::{Failures, corrected_ref_data, model};

/// The coordinates of every reference case.
fn reference_coordinates(data: &common::CorrectedRefData) -> Vec<Vec<Point2>> {
    let mut start = 0;
    (0..data.airfoil_counts.n_rows)
        .map(|c| {
            let count = data.airfoil_counts.row(c)[0] as usize;
            let points = (start..start + count)
                .map(|i| {
                    let row = data.airfoil_coordinates.row(i);
                    [row[0], row[1]]
                })
                .collect();
            start += count;
            points
        })
        .collect()
}

#[test]
fn matches_python_neuralfoil_for_transformed_coordinates() {
    let data = corrected_ref_data();
    let model = model(data.model_size);
    let coordinates = reference_coordinates(&data);
    assert!(!coordinates.is_empty());

    let names = [
        "analysis_confidence",
        "cl",
        "cd",
        "cm",
        "top_xtr",
        "bot_xtr",
        "x_translation",
        "y_translation",
        "scale_factor",
        "rotation_angle",
    ];
    let mut failures = Failures::default();
    for (c, points) in coordinates.iter().enumerate() {
        let airfoil = FittedAirfoil::from_coordinates(points).unwrap();
        let condition = data.airfoil_conditions.row(c);
        let aero = model.analyze_airfoil(
            &airfoil,
            &OperatingPoint {
                alpha: condition[0],
                re: condition[1],
                ..OperatingPoint::default()
            },
        );
        let actual = [
            aero.analysis_confidence,
            aero.cl,
            aero.cd,
            aero.cm,
            aero.top_xtr,
            aero.bot_xtr,
            airfoil.x_translation,
            airfoil.y_translation,
            airfoil.scale_factor,
            airfoil.rotation_angle,
        ];
        let expected = data.airfoil_outputs.row(c);
        for i in 0..actual.len() {
            // The Kulfan fit solves the normal equations where NumPy uses an
            // SVD, which limits the agreement of the fitted weights.
            let tolerance = 1e-6 * expected[i].abs().max(1e-3);
            failures.expect_near(actual[i], expected[i], tolerance, || {
                format!("{} {}", data.airfoil_names[c], names[i])
            });
        }
    }
    failures.assert_none();
}

#[test]
fn normalization_puts_the_chord_on_the_unit_interval() {
    let data = corrected_ref_data();
    for points in reference_coordinates(&data) {
        let normalized = normalize_coordinates(&points).unwrap();
        let (first, last) = (
            normalized.coordinates[0],
            normalized.coordinates[points.len() - 1],
        );
        let trailing_edge = [(first[0] + last[0]) / 2.0, (first[1] + last[1]) / 2.0];
        assert!((trailing_edge[0] - 1.0).abs() < 1e-12);
        assert!(trailing_edge[1].abs() < 1e-12);
        let nearest = normalized
            .coordinates
            .iter()
            .map(|p| p[0].hypot(p[1]))
            .fold(f64::INFINITY, f64::min);
        assert!(nearest < 1e-12, "leading edge at the origin");
    }
}

#[test]
fn normalized_airfoil_needs_no_transfer() {
    let model = model("medium");
    let kulfan = bora::KulfanParameters {
        upper_weights: [0.2; 8],
        lower_weights: [-0.1; 8],
        ..Default::default()
    };
    let points = generate_kulfan_coordinates(&kulfan, 80).unwrap();
    let airfoil = FittedAirfoil::from_coordinates(&points).unwrap();
    assert!(airfoil.rotation_angle.abs() < 1e-9);
    assert!((airfoil.scale_factor - 1.0).abs() < 1e-12);

    let point = OperatingPoint {
        alpha: 2.0,
        ..OperatingPoint::default()
    };
    let direct = model.analyze(&airfoil.input(&point));
    let via_airfoil = model.analyze_airfoil(&airfoil, &point);
    assert!((direct.cm - via_airfoil.cm).abs() < 1e-9);
    assert_eq!(direct.cl, via_airfoil.cl);
}

#[test]
fn rejects_degenerate_coordinates() {
    assert!(matches!(
        normalize_coordinates(&[]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        normalize_coordinates(&[[0.5, 0.5]; 5]),
        Err(Error::InvalidArgument(_))
    ));
    assert!(FittedAirfoil::from_coordinates(&[[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]).is_err());
}

#[test]
fn parses_dat_text() {
    let text = "NACA 0012 AIRFOILS\n\n 1.000000  0.001260\n0.5,0.05\n0.0;0.0\t\n\
                # comment\n0.5 | -0.05\n1.0 -1.26e-3 extra\n1.000000 -0.001260\n";
    let points = parse_dat(text).unwrap();
    assert_eq!(
        points,
        vec![
            [1.0, 0.00126],
            [0.5, 0.05],
            [0.0, 0.0],
            [0.5, -0.05],
            [1.0, -0.00126]
        ]
    );
    assert!(matches!(
        parse_dat("just a name\n"),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn reads_dat_files() {
    let data = corrected_ref_data();
    let points = &reference_coordinates(&data)[0];
    let mut text = String::from("Reference airfoil\n");
    for p in points {
        text.push_str(&format!("{:e} {:e}\n", p[0], p[1]));
    }
    let path = std::env::temp_dir().join(format!("bora_test_{}.dat", std::process::id()));
    std::fs::write(&path, text).unwrap();
    let read = read_dat_file(&path);
    let fitted = FittedAirfoil::from_dat_file(&path);
    std::fs::remove_file(&path).unwrap();

    assert_eq!(&read.unwrap(), points);
    assert_eq!(
        fitted.unwrap(),
        FittedAirfoil::from_coordinates(points).unwrap()
    );
    assert!(matches!(
        read_dat_file("/nonexistent/airfoil.dat"),
        Err(Error::Io { .. })
    ));
}
