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

//! Corrected analysis: a Mach sweep with the compressibility corrections, a
//! deflected flap, and the post-stall model at large angles of attack.

use bora::{AeroInput, ControlSurface, CorrectedInput, ModelSize, NeuralFoil, input_index as idx};

fn main() -> bora::Result<()> {
    // A NACA-4412-like airfoil in Kulfan (CST) parameterization.
    let mut aero = AeroInput::default();
    aero.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    aero.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    aero.kulfan.leading_edge_weight = 0.02;
    aero.kulfan.te_thickness = 0.0025;
    aero.alpha = 1.0; // deg
    aero.re = 5e6;

    let model = NeuralFoil::new(ModelSize::XLarge)?;
    let mut input = CorrectedInput::from(aero);

    let first = model.analyze_corrected(&input);
    println!(
        "alpha = {:.1} deg, Re = {:.1e}: Mcrit = {:.3}, Mdd = {:.3}\n",
        aero.alpha, aero.re, first.mach_crit, first.mach_dd
    );
    println!("  Mach       CL        CD        CM     Cpmin   dCD/dMach");
    for mach in [0.0, 0.3, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.2] {
        input.mach = mach;
        let result = model.analyze_corrected_with_gradients(&input);
        let out = &result.output;
        println!(
            "  {mach:4.2}  {:7.4}  {:8.5}  {:8.4}  {:8.3}  {:10.5}",
            out.cl,
            out.cd,
            out.cm,
            out.cpmin,
            result.gradient.cd[idx::MACH]
        );
    }

    // A plain flap hinged at 75% chord, at Mach 0.3.
    input.mach = 0.3;
    println!("\n  Flap (deg)       CL        CD        CM   dCL/d(flap)");
    for deflection in [-10.0, 0.0, 10.0, 20.0] {
        input.control_surfaces = vec![ControlSurface {
            deflection,
            hinge_point: 0.75,
        }];
        let result = model.analyze_corrected_with_gradients(&input);
        println!(
            "  {deflection:10.1}  {:7.4}  {:8.5}  {:8.4}  {:12.5}",
            result.output.cl, result.output.cd, result.output.cm, result.gradient.deflection[0].cl
        );
    }

    // Around the full circle, where the post-stall model takes over.
    input.control_surfaces.clear();
    input.mach = 0.0;
    println!("\n  alpha (deg)       CL        CD  confidence");
    for alpha in [0.0, 15.0, 25.0, 45.0, 90.0, 135.0, 180.0] {
        input.aero.alpha = alpha;
        let out = model.analyze_corrected(&input);
        println!(
            "  {alpha:11.1}  {:7.4}  {:8.5}  {:10.4}",
            out.cl, out.cd, out.analysis_confidence
        );
    }
    Ok(())
}
