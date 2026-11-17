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

//! Basic usage example: analyze an airfoil and print coefficients and
//! selected gradients.

use bora::{AeroInput, ModelSize, NUM_CST_WEIGHTS_PER_SIDE, NeuralFoil, input_index as idx};

fn main() -> bora::Result<()> {
    // A NACA-4412-like airfoil in Kulfan (CST) parameterization.
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.leading_edge_weight = 0.02;
    input.kulfan.te_thickness = 0.0025;
    input.alpha = 3.0; // deg
    input.re = 1e6;

    let model = NeuralFoil::new(ModelSize::XLarge)?;
    let result = model.analyze_with_gradients(&input);
    let aero = &result.output;

    println!(
        "Bora {} (NeuralFoil port), model size '{}'\n",
        bora::version(),
        model.model_size()
    );
    println!("alpha = {:.2} deg, Re = {:.2e}", input.alpha, input.re);
    println!("  CL  = {:9.5}", aero.cl);
    println!("  CD  = {:9.5}", aero.cd);
    println!("  CM  = {:9.5}", aero.cm);
    println!("  L/D = {:9.3}", aero.cl / aero.cd);
    println!(
        "  transition: upper x/c = {:.3}, lower x/c = {:.3}",
        aero.top_xtr, aero.bot_xtr
    );
    println!("  analysis confidence = {:.4}\n", aero.analysis_confidence);

    let gradient = &result.gradient;
    println!("Gradients (analytic, for optimization):");
    println!("  dCL/dalpha = {:9.5} per deg", gradient.cl[idx::ALPHA]);
    println!("  dCM/dalpha = {:9.5} per deg", gradient.cm[idx::ALPHA]);
    println!("  dCD/dRe    = {:9.3e}", gradient.cd[idx::RE]);
    print!("  dCL/d(upper CST weight 0..7):\n   ");
    for i in 0..NUM_CST_WEIGHTS_PER_SIDE {
        print!(" {:8.4}", gradient.cl[idx::UPPER_WEIGHTS + i]);
    }
    println!();
    Ok(())
}
