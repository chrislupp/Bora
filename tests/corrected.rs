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

//! The compressibility, post-stall and control surface corrections against
//! AeroSandbox's `KulfanAirfoil.get_aero_from_neuralfoil`, and their
//! gradients against finite differences.

mod common;

use bora::{
    AeroInput, ControlSurface, CorrectedInput, CorrectedOutput, NUM_BL_POINTS,
    NUM_CORRECTED_INPUTS, NUM_INPUTS, input_index,
};
use common::{Failures, corrected_ref_data, input_from_array, model};

const SCALAR_NAMES: [&str; 10] = [
    "analysis_confidence",
    "cl",
    "cd",
    "cm",
    "cpmin",
    "top_xtr",
    "bot_xtr",
    "mach_crit",
    "mach_dd",
    "cpmin_0",
];

/// The scalar outputs in the order of the reference data.
fn scalars(output: &CorrectedOutput) -> [f64; 10] {
    [
        output.analysis_confidence,
        output.cl,
        output.cd,
        output.cm,
        output.cpmin,
        output.top_xtr,
        output.bot_xtr,
        output.mach_crit,
        output.mach_dd,
        output.cpmin_0,
    ]
}

/// Builds the corrected input of reference case `c`.
fn reference_input(data: &common::CorrectedRefData, c: usize) -> CorrectedInput {
    // Layout: the 18 shape parameters, alpha, Re, mach, include_360, count.
    let row = data.corrected_inputs.row(c);
    let mut values = [0.0; NUM_INPUTS];
    values[..20].copy_from_slice(&row[..20]);
    let defaults = AeroInput::default();
    values[input_index::N_CRIT] = defaults.n_crit;
    values[input_index::XTR_UPPER] = defaults.xtr_upper;
    values[input_index::XTR_LOWER] = defaults.xtr_lower;
    let aero = input_from_array(&values);
    let surfaces = data.control_surfaces.row(c);
    CorrectedInput {
        aero,
        mach: row[20],
        include_360_deg_effects: row[21] != 0.0,
        control_surfaces: (0..row[22] as usize)
            .map(|i| ControlSurface {
                deflection: surfaces[2 * i],
                hinge_point: surfaces[2 * i + 1],
            })
            .collect(),
    }
}

#[test]
fn matches_aerosandbox_reference_values() {
    let data = corrected_ref_data();
    let model = model(data.model_size);
    assert!(data.corrected_inputs.n_rows > 0);

    let mut failures = Failures::default();
    for c in 0..data.corrected_inputs.n_rows {
        let input = reference_input(&data, c);
        let expected = data.corrected_outputs.row(c);
        let actual = scalars(&model.analyze_corrected(&input));
        for (i, &value) in actual.iter().enumerate() {
            let tolerance = 1e-6 * expected[i].abs().max(1e-6);
            failures.expect_near(value, expected[i], tolerance, || {
                format!("{} {}", data.corrected_names[c], SCALAR_NAMES[i])
            });
        }
        failures.expect_near(
            input.aero.kulfan.max_thickness(),
            expected[10],
            1e-12,
            || format!("{} max_thickness", data.corrected_names[c]),
        );
    }
    failures.assert_none();
}

#[test]
fn boundary_layer_is_the_network_output_at_the_shifted_alpha() {
    let data = corrected_ref_data();
    let model = model(data.model_size);
    let mut input = reference_input(&data, 0);
    input.control_surfaces = vec![ControlSurface {
        deflection: 8.0,
        hinge_point: 0.0,
    }];

    // A hinge at the leading edge moves the whole airfoil: alpha + deflection.
    let corrected = model.analyze_corrected(&input);
    let mut shifted = input.aero;
    shifted.alpha += 8.0;
    let base = model.analyze(&shifted);
    assert_eq!(corrected.upper, base.upper);
    assert_eq!(corrected.lower, base.lower);
    assert_eq!(corrected.analysis_confidence, base.analysis_confidence);
}

#[test]
fn alpha_is_periodic() {
    let model = model("medium");
    let data = corrected_ref_data();
    let input = reference_input(&data, 2);
    let mut turned = input.clone();
    turned.aero.alpha += 720.0;
    let (a, b) = (
        model.analyze_corrected(&input),
        model.analyze_corrected(&turned),
    );
    for (x, y) in scalars(&a).iter().zip(scalars(&b)) {
        assert!((x - y).abs() <= 1e-9 * x.abs().max(1.0));
    }
}

#[test]
fn batch_matches_single_point_calls() {
    let data = corrected_ref_data();
    let model = model(data.model_size);
    let inputs: Vec<CorrectedInput> = (0..data.corrected_inputs.n_rows)
        .map(|c| reference_input(&data, c))
        .collect();
    let batch = model.analyze_corrected_batch(&inputs);
    assert_eq!(batch.len(), inputs.len());

    let mut failures = Failures::default();
    for (c, (input, batched)) in inputs.iter().zip(&batch).enumerate() {
        let single = scalars(&model.analyze_corrected(input));
        for (i, (s, b)) in single.iter().zip(scalars(batched)).enumerate() {
            failures.expect_near(b, *s, 1e-10 * s.abs().max(1e-3), || {
                format!("{} {}", data.corrected_names[c], SCALAR_NAMES[i])
            });
        }
    }
    failures.assert_none();
    assert!(model.analyze_corrected_batch(&[]).is_empty());
}

/// Adds `delta` to input `k` of a corrected input (the 23 inputs, then Mach).
fn perturbed(input: &CorrectedInput, k: usize, delta: f64) -> CorrectedInput {
    let mut out = input.clone();
    let a = &mut out.aero;
    match k {
        0..8 => a.kulfan.upper_weights[k] += delta,
        8..16 => a.kulfan.lower_weights[k - 8] += delta,
        input_index::LEADING_EDGE_WEIGHT => a.kulfan.leading_edge_weight += delta,
        input_index::TE_THICKNESS => a.kulfan.te_thickness += delta,
        input_index::ALPHA => a.alpha += delta,
        input_index::RE => a.re += delta,
        input_index::N_CRIT => a.n_crit += delta,
        input_index::XTR_UPPER => a.xtr_upper += delta,
        input_index::XTR_LOWER => a.xtr_lower += delta,
        input_index::MACH => out.mach += delta,
        _ => unreachable!(),
    }
    out
}

/// Central finite difference of the scalar outputs between two inputs.
fn difference(
    model: &bora::NeuralFoil,
    plus: &CorrectedInput,
    minus: &CorrectedInput,
    step: f64,
) -> [f64; 10] {
    let (p, m) = (
        scalars(&model.analyze_corrected(plus)),
        scalars(&model.analyze_corrected(minus)),
    );
    std::array::from_fn(|i| (p[i] - m[i]) / (2.0 * step))
}

#[test]
fn gradients_match_finite_differences() {
    let data = corrected_ref_data();
    let model = model(data.model_size);

    // The cases of the first airfoil cover every branch of the corrections
    // (each wave drag regime, post-stall, control surfaces); those of the
    // second add camber.
    let n_conditions = data.corrected_inputs.n_rows / 4;
    let mut failures = Failures::default();
    for c in 0..2 * n_conditions {
        let mut input = reference_input(&data, c);
        // Free transition leaves the forced locations without effect at
        // xtr = 1; move them inside the chord, where they matter and can be
        // perturbed in both directions.
        input.aero.xtr_upper = 0.6;
        input.aero.xtr_lower = 0.7;
        let result = model.analyze_corrected_with_gradients(&input);
        let grad = &result.gradient;
        let gradients = [
            &grad.analysis_confidence,
            &grad.cl,
            &grad.cd,
            &grad.cm,
            &grad.cpmin,
            &grad.top_xtr,
            &grad.bot_xtr,
            &grad.mach_crit,
            &grad.mach_dd,
            &grad.cpmin_0,
        ];

        let plain = scalars(&model.analyze_corrected(&input));
        for (i, (a, b)) in scalars(&result.output).iter().zip(plain).enumerate() {
            failures.expect_near(*a, b, 1e-12 * b.abs().max(1e-3), || {
                format!("{} value of {}", data.corrected_names[c], SCALAR_NAMES[i])
            });
        }

        let check = |failures: &mut Failures, fd: [f64; 10], analytic: [f64; 10], what: String| {
            for i in 0..10 {
                let scale = fd[i].abs().max(analytic[i].abs());
                failures.expect_near(analytic[i], fd[i], 1e-4 * scale + 1e-9, || {
                    format!("{} d{}/d{what}", data.corrected_names[c], SCALAR_NAMES[i])
                });
            }
        };

        for k in 0..NUM_CORRECTED_INPUTS {
            let step = if k == input_index::RE {
                1e-5 * input.aero.re
            } else {
                1e-6
            };
            let fd = difference(
                model,
                &perturbed(&input, k, step),
                &perturbed(&input, k, -step),
                step,
            );
            check(
                &mut failures,
                fd,
                gradients.map(|g| g[k]),
                format!("input {k}"),
            );
        }

        assert_eq!(grad.deflection.len(), input.control_surfaces.len());
        for (j, derivative) in grad.deflection.iter().enumerate() {
            let step = 1e-6;
            let (mut plus, mut minus) = (input.clone(), input.clone());
            plus.control_surfaces[j].deflection += step;
            minus.control_surfaces[j].deflection -= step;
            let fd = difference(model, &plus, &minus, step);
            check(
                &mut failures,
                fd,
                scalars(derivative),
                format!("deflection {j}"),
            );

            // The boundary layer follows the alpha shift of the surface.
            let (p, m) = (
                model.analyze_corrected(&plus),
                model.analyze_corrected(&minus),
            );
            for s in [0, NUM_BL_POINTS / 2, NUM_BL_POINTS - 1] {
                let fd = (p.upper.ue_over_vinf[s] - m.upper.ue_over_vinf[s]) / (2.0 * step);
                let analytic = derivative.upper.ue_over_vinf[s];
                failures.expect_near(analytic, fd, 2e-5 * fd.abs() + 1e-9, || {
                    format!(
                        "{} d(upper ue {s})/d(deflection {j})",
                        data.corrected_names[c]
                    )
                });
            }
        }
    }
    failures.assert_none();
}

#[test]
fn boundary_layer_gradients_are_those_of_the_network() {
    let data = corrected_ref_data();
    let model = model(data.model_size);
    let input = reference_input(&data, 3);
    let corrected = model.analyze_corrected_with_gradients(&input);
    let base = model.analyze_with_gradients(&input.aero);
    assert_eq!(corrected.gradient.upper, base.gradient.upper);
    assert_eq!(corrected.gradient.lower, base.gradient.lower);
    assert_eq!(
        corrected.gradient.analysis_confidence[input_index::MACH],
        0.0
    );
    assert_eq!(
        corrected.gradient.analysis_confidence[..NUM_INPUTS],
        base.gradient.analysis_confidence
    );
}

#[test]
fn default_corrected_input_wraps_an_aero_input() {
    let input = CorrectedInput::default();
    assert_eq!(input.aero, AeroInput::default());
    assert_eq!(input.mach, 0.0);
    assert!(input.control_surfaces.is_empty());
    assert!(input.include_360_deg_effects);
    assert_eq!(CorrectedInput::from(AeroInput::default()), input);
}
