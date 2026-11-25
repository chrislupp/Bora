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

//! Tests for the Philote discipline wrapper, exercised directly (no gRPC).

use std::collections::HashMap;

use bora::philote::BoraDiscipline;
use bora::{
    AeroInput, CorrectedInput, Gradient, ModelSize, NUM_BL_POINTS, NeuralFoil, input_index,
};
use ndarray::{ArrayD, IxDyn};
use philote_mdo::{ArrayMap, Discipline, ExplicitDiscipline};

/// (name, size, offset into the Gradient vector)
const INPUT_SPECS: [(&str, usize, usize); 9] = [
    ("upper_weights", 8, input_index::UPPER_WEIGHTS),
    ("lower_weights", 8, input_index::LOWER_WEIGHTS),
    ("leading_edge_weight", 1, input_index::LEADING_EDGE_WEIGHT),
    ("TE_thickness", 1, input_index::TE_THICKNESS),
    ("alpha", 1, input_index::ALPHA),
    ("Re", 1, input_index::RE),
    ("n_crit", 1, input_index::N_CRIT),
    ("xtr_upper", 1, input_index::XTR_UPPER),
    ("xtr_lower", 1, input_index::XTR_LOWER),
];

const SCALAR_OUTPUTS: [&str; 6] = [
    "CL",
    "CD",
    "CM",
    "Top_Xtr",
    "Bot_Xtr",
    "analysis_confidence",
];

/// A NACA-4412-like test case matching examples/example.rs.
fn test_case() -> AeroInput {
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.leading_edge_weight = 0.02;
    input.kulfan.te_thickness = 0.0025;
    input.alpha = 3.0;
    input.re = 1e6;
    input
}

fn make_inputs(input: &AeroInput) -> ArrayMap {
    let values: [(&str, Vec<f64>); 9] = [
        ("upper_weights", input.kulfan.upper_weights.to_vec()),
        ("lower_weights", input.kulfan.lower_weights.to_vec()),
        (
            "leading_edge_weight",
            vec![input.kulfan.leading_edge_weight],
        ),
        ("TE_thickness", vec![input.kulfan.te_thickness]),
        ("alpha", vec![input.alpha]),
        ("Re", vec![input.re]),
        ("n_crit", vec![input.n_crit]),
        ("xtr_upper", vec![input.xtr_upper]),
        ("xtr_lower", vec![input.xtr_lower]),
    ];
    values
        .into_iter()
        .map(|(name, v)| {
            let n = v.len();
            (
                name.to_string(),
                ArrayD::from_shape_vec(IxDyn(&[n]), v).unwrap(),
            )
        })
        .collect()
}

/// The discipline's default configuration (no boundary layer outputs) takes
/// the scalar-only gradient path, so that is the bitwise reference.
fn scalar_gradients(model: &NeuralFoil, input: &AeroInput) -> [Gradient; 6] {
    let g = model.analyze_scalars_with_gradients(input).gradient;
    [
        g.cl,
        g.cd,
        g.cm,
        g.top_xtr,
        g.bot_xtr,
        g.analysis_confidence,
    ]
}

#[tokio::test]
async fn compute_matches_direct_analysis() {
    let input = test_case();
    let outputs = BoraDiscipline::new()
        .compute(&make_inputs(&input))
        .await
        .unwrap();

    let expected = NeuralFoil::new(ModelSize::XLarge).unwrap().analyze(&input);
    assert_eq!(outputs["CL"][[0]], expected.cl);
    assert_eq!(outputs["CD"][[0]], expected.cd);
    assert_eq!(outputs["CM"][[0]], expected.cm);
    assert_eq!(outputs["Top_Xtr"][[0]], expected.top_xtr);
    assert_eq!(outputs["Bot_Xtr"][[0]], expected.bot_xtr);
    assert_eq!(
        outputs["analysis_confidence"][[0]],
        expected.analysis_confidence
    );
    assert!(!outputs.contains_key("upper_theta"));
}

#[tokio::test]
async fn compute_partials_matches_direct_gradients() {
    let input = test_case();
    let partials = BoraDiscipline::new()
        .compute_partials(&make_inputs(&input))
        .await
        .unwrap();

    let model = NeuralFoil::new(ModelSize::XLarge).unwrap();
    let expected = scalar_gradients(&model, &input);
    for (output, gradient) in SCALAR_OUTPUTS.iter().zip(&expected) {
        for (name, size, offset) in INPUT_SPECS {
            let jac = &partials[&(output.to_string(), name.to_string())];
            assert_eq!(jac.shape(), [size]);
            let actual: Vec<f64> = jac.iter().copied().collect();
            assert_eq!(
                actual,
                gradient[offset..offset + size],
                "{output} wrt {name}"
            );
        }
    }
}

#[tokio::test]
async fn options_select_model_and_boundary_layer_outputs() {
    let mut discipline = BoraDiscipline::new();
    discipline.initialize().unwrap();
    let options = HashMap::from([
        ("model_size".to_string(), serde_json::json!("medium")),
        (
            "include_boundary_layer".to_string(),
            serde_json::json!(true),
        ),
    ]);
    discipline.set_options(&options).unwrap();
    discipline.setup().unwrap();
    discipline.setup_partials().unwrap();

    assert_eq!(discipline.get_variable_definitions().unwrap().len(), 9 + 12);
    assert_eq!(discipline.get_partials_definitions().unwrap().len(), 12 * 9);

    let input = test_case();
    let inputs = make_inputs(&input);
    let outputs = discipline.compute(&inputs).await.unwrap();
    let partials = discipline.compute_partials(&inputs).await.unwrap();

    let model = NeuralFoil::new(ModelSize::Medium).unwrap();
    let expected = model.analyze_with_gradients(&input);
    assert_eq!(outputs["CL"][[0]], expected.output.cl);
    let theta: Vec<f64> = outputs["lower_theta"].iter().copied().collect();
    assert_eq!(theta, expected.output.lower.theta);

    // Boundary layer partials are row-major [NUM_BL_POINTS x input size].
    let upper_h_weights = &partials[&("upper_H".to_string(), "upper_weights".to_string())];
    assert_eq!(upper_h_weights.shape(), [NUM_BL_POINTS, 8]);
    assert_eq!(upper_h_weights[[5, 3]], expected.gradient.upper.h[5][3]);
    let lower_ue_alpha = &partials[&("lower_ue_over_vinf".to_string(), "alpha".to_string())];
    assert_eq!(lower_ue_alpha.shape(), [NUM_BL_POINTS]);
    assert_eq!(
        lower_ue_alpha[[7]],
        expected.gradient.lower.ue_over_vinf[7][input_index::ALPHA]
    );
}

#[tokio::test]
async fn corrections_option_adds_mach_and_corrected_outputs() {
    let mut discipline = BoraDiscipline::new();
    discipline.initialize().unwrap();
    let options = HashMap::from([
        ("model_size".to_string(), serde_json::json!("medium")),
        ("include_corrections".to_string(), serde_json::json!(true)),
        (
            "include_360_deg_effects".to_string(),
            serde_json::json!(false),
        ),
    ]);
    discipline.set_options(&options).unwrap();
    discipline.setup().unwrap();
    discipline.setup_partials().unwrap();

    // Ten inputs (with mach) and ten scalar outputs.
    assert_eq!(
        discipline.get_variable_definitions().unwrap().len(),
        10 + 10
    );
    assert_eq!(
        discipline.get_partials_definitions().unwrap().len(),
        10 * 10
    );

    let corrected = CorrectedInput {
        aero: test_case(),
        mach: 0.5,
        control_surfaces: Vec::new(),
        include_360_deg_effects: false,
    };
    let mut inputs = make_inputs(&corrected.aero);
    inputs.insert(
        "mach".to_string(),
        ArrayD::from_shape_vec(IxDyn(&[1]), vec![corrected.mach]).unwrap(),
    );
    let outputs = discipline.compute(&inputs).await.unwrap();
    let partials = discipline.compute_partials(&inputs).await.unwrap();

    let model = NeuralFoil::new(ModelSize::Medium).unwrap();
    let expected = model.analyze_corrected_with_gradients(&corrected);
    assert_eq!(outputs["CL"][[0]], expected.output.cl);
    assert_eq!(outputs["CD"][[0]], expected.output.cd);
    assert_eq!(outputs["Cpmin"][[0]], expected.output.cpmin);
    assert_eq!(outputs["mach_crit"][[0]], expected.output.mach_crit);
    assert_eq!(outputs["mach_dd"][[0]], expected.output.mach_dd);
    assert!(expected.output.cl > model.analyze(&corrected.aero).cl * 1.1);

    let key = |output: &str, input: &str| (output.to_string(), input.to_string());
    assert_eq!(
        partials[&key("CD", "mach")][[0]],
        expected.gradient.cd[input_index::MACH]
    );
    assert_eq!(
        partials[&key("CL", "alpha")][[0]],
        expected.gradient.cl[input_index::ALPHA]
    );
    assert_eq!(
        partials[&key("mach_crit", "upper_weights")][[2]],
        expected.gradient.mach_crit[input_index::UPPER_WEIGHTS + 2]
    );

    // Without the option there is no Mach input.
    assert!(
        discipline
            .compute(&make_inputs(&corrected.aero))
            .await
            .is_err()
    );
}

#[test]
fn rejects_unknown_model_size() {
    let mut discipline = BoraDiscipline::new();
    let options = HashMap::from([("model_size".to_string(), serde_json::json!("huge"))]);
    assert!(discipline.set_options(&options).is_err());
}
