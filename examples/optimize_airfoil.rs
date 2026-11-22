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

//! Gradient-based airfoil optimization example.
//!
//! Starting from a NACA 0012, minimizes drag at a fixed lift target
//! (CL = 0.8, Re = 1e6) by shaping the airfoil and trimming alpha, using
//! Bora's analytic gradients with a simple Adam optimizer.
//!
//! Constraints are handled with quadratic penalties:
//!   - CL must meet the target,
//!   - the analysis confidence must stay high, which keeps the optimizer
//!     inside the region where the neural network is trustworthy (and, as a
//!     side effect, keeps the shape physically reasonable).

use std::fs;

use bora::{
    AeroInput, AeroOutput, ModelSize, NUM_CST_WEIGHTS_PER_SIDE, NeuralFoil,
    generate_kulfan_coordinates, input_index,
};

const CL_TARGET: f64 = 0.8;
const REYNOLDS: f64 = 1e6;
const CONFIDENCE_FLOOR: f64 = 0.95;
const CL_PENALTY_WEIGHT: f64 = 1.0;
const CONFIDENCE_PENALTY_WEIGHT: f64 = 0.1;

/// Design variables: 16 CST weights, leading edge weight, and alpha (18 total,
/// mapped onto the corresponding entries of Bora's input vector).
const NUM_DESIGN_VARS: usize = 18;

fn design_to_input_index(design_index: usize) -> usize {
    if design_index < 17 {
        design_index // CST weights and LE weight
    } else {
        input_index::ALPHA
    }
}

struct Objective {
    value: f64,
    gradient: [f64; NUM_DESIGN_VARS],
    aero: AeroOutput,
}

fn evaluate(model: &NeuralFoil, input: &AeroInput) -> Objective {
    let result = model.analyze_with_gradients(input);
    let aero = result.output;

    let cl_error = aero.cl - CL_TARGET;
    let confidence_deficit = (CONFIDENCE_FLOOR - aero.analysis_confidence).max(0.0);

    let value = aero.cd
        + CL_PENALTY_WEIGHT * cl_error * cl_error
        + CONFIDENCE_PENALTY_WEIGHT * confidence_deficit * confidence_deficit;

    let g = &result.gradient;
    let gradient = std::array::from_fn(|i| {
        let k = design_to_input_index(i);
        g.cd[k] + 2.0 * CL_PENALTY_WEIGHT * cl_error * g.cl[k]
            - 2.0 * CONFIDENCE_PENALTY_WEIGHT * confidence_deficit * g.analysis_confidence[k]
    });
    Objective {
        value,
        gradient,
        aero,
    }
}

fn apply_design(design: &[f64; NUM_DESIGN_VARS], input: &mut AeroInput) {
    let w = NUM_CST_WEIGHTS_PER_SIDE;
    input.kulfan.upper_weights.copy_from_slice(&design[..w]);
    input
        .kulfan
        .lower_weights
        .copy_from_slice(&design[w..2 * w]);
    input.kulfan.leading_edge_weight = design[16];
    input.alpha = design[17];
}

fn format_weights(weights: &[f64]) -> String {
    weights
        .iter()
        .map(|w| format!("{w:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = NeuralFoil::new(ModelSize::XLarge)?;

    // Starting point: NACA 0012 (fitted Kulfan parameters), alpha = 4 deg.
    let mut input = AeroInput {
        re: REYNOLDS,
        ..AeroInput::default()
    };
    input.kulfan.te_thickness = 0.002548;
    #[rustfmt::skip]
    let mut design: [f64; NUM_DESIGN_VARS] = [
        0.172884, 0.151563, 0.173763, 0.127681, 0.164818, 0.126376, 0.145898, 0.1392, // upper
        -0.172884, -0.151563, -0.173763, -0.127681, -0.164818, -0.126376, -0.145898, -0.1392, // lower
        0.0, // leading edge weight
        4.0, // alpha [deg]
    ];

    // Adam optimizer state.
    const LEARNING_RATE: f64 = 5e-3;
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPSILON: f64 = 1e-9;
    let mut m = [0.0; NUM_DESIGN_VARS];
    let mut v = [0.0; NUM_DESIGN_VARS];

    println!("Minimizing CD at CL = {CL_TARGET:.2}, Re = {REYNOLDS:.1e} (xlarge model)\n");
    println!(
        "{:>6}  {:>10}  {:>9}  {:>9}  {:>8}  {:>11}",
        "iter", "objective", "CL", "CD", "L/D", "confidence"
    );

    apply_design(&design, &mut input);
    let mut objective = evaluate(&model, &input);
    for iteration in 1..=500 {
        for i in 0..NUM_DESIGN_VARS {
            let g = objective.gradient[i];
            m[i] = BETA1 * m[i] + (1.0 - BETA1) * g;
            v[i] = BETA2 * v[i] + (1.0 - BETA2) * g * g;
            let m_hat = m[i] / (1.0 - BETA1.powi(iteration));
            let v_hat = v[i] / (1.0 - BETA2.powi(iteration));
            design[i] -= LEARNING_RATE * m_hat / (v_hat.sqrt() + EPSILON);
        }
        apply_design(&design, &mut input);
        objective = evaluate(&model, &input);

        if iteration % 50 == 0 || iteration == 1 {
            let aero = &objective.aero;
            println!(
                "{iteration:>6}  {:>10.6}  {:>9.5}  {:>9.5}  {:>8.2}  {:>11.4}",
                objective.value,
                aero.cl,
                aero.cd,
                aero.cl / aero.cd,
                aero.analysis_confidence
            );
        }
    }

    println!("\nOptimized airfoil (Kulfan parameters):");
    println!("  upper_weights = {{{}}}", format_weights(&design[..8]));
    println!("  lower_weights = {{{}}}", format_weights(&design[8..16]));
    println!("  leading_edge_weight = {:.6}", design[16]);
    println!("  alpha = {:.3} deg", design[17]);

    // Export the optimized shape as Selig-format coordinates.
    let coordinates = generate_kulfan_coordinates(&input.kulfan, 100)?;
    let out_path = "optimized_airfoil.dat";
    let mut contents = String::from("Bora optimized airfoil\n");
    for [x, y] in coordinates {
        contents.push_str(&format!("{x:.6} {y:.6}\n"));
    }
    fs::write(out_path, contents)?;
    println!("\nWrote {out_path}");
    Ok(())
}
