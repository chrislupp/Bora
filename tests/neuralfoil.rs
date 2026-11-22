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

mod common;

use bora::{AeroOutput, ModelSize, NUM_CST_WEIGHTS_PER_SIDE, NUM_LATENT_OUTPUTS, NeuralFoil};
use common::{Failures, input_from_array, model, output_ref_data};

/// Flattens an AeroOutput into the reference data output ordering.
fn flatten(output: &AeroOutput) -> Vec<f64> {
    let mut flat = vec![
        output.analysis_confidence,
        output.cl,
        output.cd,
        output.cm,
        output.top_xtr,
        output.bot_xtr,
    ];
    for surface in [&output.upper, &output.lower] {
        for quantity in [&surface.theta, &surface.h, &surface.ue_over_vinf] {
            flat.extend_from_slice(quantity);
        }
    }
    flat
}

#[test]
fn matches_python_neuralfoil_reference_values() {
    let data = output_ref_data();
    assert!(data.n_cases > 0);
    assert_eq!(data.n_outputs, NUM_LATENT_OUTPUTS);

    let mut failures = Failures::default();
    for c in 0..data.n_cases {
        let model = model(&data.model_sizes[c]);
        let input = input_from_array(&data.inputs[c * bora::NUM_INPUTS..]);
        let actual = flatten(&model.analyze(&input));

        for (i, &value) in actual.iter().enumerate() {
            let expected = data.outputs[c * data.n_outputs + i];
            let tolerance = 1e-6 * expected.abs().max(1e-8);
            failures.expect_near(value, expected, tolerance, || {
                format!("{} output index {i}", data.names[c])
            });
        }
    }
    failures.assert_none();
}

#[test]
fn gradient_variant_returns_same_values() {
    let model = model("medium");
    let input = input_from_array(&output_ref_data().inputs);

    let plain = flatten(&model.analyze(&input));
    let with_grad = flatten(&model.analyze_with_gradients(&input).output);
    let mut failures = Failures::default();
    for (i, (p, g)) in plain.iter().zip(&with_grad).enumerate() {
        failures.expect_near(*g, *p, 1e-12 * p.abs().max(1e-3), || {
            format!("output index {i}")
        });
    }
    failures.assert_none();
}

#[test]
fn alpha_symmetry_invariant() {
    // Mirroring the airfoil and negating alpha must exactly negate CL and CM
    // and swap the surfaces, by construction of the fused evaluation.
    let model = model("medium");
    let input = input_from_array(&output_ref_data().inputs);

    let mut mirrored = input;
    for i in 0..NUM_CST_WEIGHTS_PER_SIDE {
        mirrored.kulfan.upper_weights[i] = -input.kulfan.lower_weights[i];
        mirrored.kulfan.lower_weights[i] = -input.kulfan.upper_weights[i];
    }
    mirrored.kulfan.leading_edge_weight = -input.kulfan.leading_edge_weight;
    mirrored.alpha = -input.alpha;
    std::mem::swap(&mut mirrored.xtr_upper, &mut mirrored.xtr_lower);

    let base = model.analyze(&input);
    let flipped = model.analyze(&mirrored);

    assert!((flipped.cl + base.cl).abs() <= 1e-12);
    assert!((flipped.cm + base.cm).abs() <= 1e-12);
    assert!((flipped.cd - base.cd).abs() <= 1e-14);
    assert!((flipped.analysis_confidence - base.analysis_confidence).abs() <= 1e-12);
    assert!((flipped.top_xtr - base.bot_xtr).abs() <= 1e-12);
    assert!((flipped.bot_xtr - base.top_xtr).abs() <= 1e-12);
}

#[test]
fn errors_on_missing_data() {
    assert!(NeuralFoil::with_data_dir(ModelSize::Medium, "/nonexistent/path").is_err());
}

#[test]
fn model_size_round_trips_through_strings() {
    for size in ModelSize::ALL {
        assert_eq!(size.as_str().parse::<ModelSize>().unwrap(), size);
    }
    assert!("huge".parse::<ModelSize>().is_err());
    assert_eq!(model("small").model_size(), ModelSize::Small);
}

#[test]
fn version_is_not_empty() {
    assert!(!bora::version().is_empty());
}
