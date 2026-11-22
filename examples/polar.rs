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

//! Polar sweep example: analyze one airfoil across a range of angles of
//! attack at several Reynolds numbers and print the lift curve, drag polar,
//! transition locations, and the analytic lift-curve slope.
//!
//! Usage: `polar [--json]`
//!
//! With `--json`, the airfoil outline and the sweeps are written to standard
//! output as JSON for plotting. The figures in the documentation are generated
//! this way (`docs/src/data/polar.json`).

use bora::{
    AeroInput, AeroScalarResult, ModelSize, NeuralFoil, generate_kulfan_coordinates, input_index,
};

const REYNOLDS_NUMBERS: [f64; 3] = [2e5, 1e6, 5e6];
const ALPHA_START: f64 = -8.0;
const ALPHA_END: f64 = 18.0;

/// Angles of attack from `ALPHA_START` to `ALPHA_END` in increments of `step`.
fn alphas(step: f64) -> impl Iterator<Item = f64> {
    let count = ((ALPHA_END - ALPHA_START) / step).round() as usize;
    (0..=count).map(move |i| ALPHA_START + i as f64 * step)
}

fn print_table(model: &NeuralFoil, input: &AeroInput) {
    println!(
        "Polar at Re = {:.1e}, n_crit = {:.1} ({} model)\n",
        input.re,
        input.n_crit,
        model.model_size()
    );
    println!(
        "{:>6}  {:>8}  {:>8}  {:>8}  {:>7}  {:>7}  {:>7}  {:>10}  {:>10}",
        "alpha", "CL", "CD", "CM", "L/D", "xtr_top", "xtr_bot", "dCL/dalpha", "confidence"
    );
    for alpha in alphas(2.0) {
        let AeroScalarResult { output, gradient } =
            model.analyze_scalars_with_gradients(&AeroInput { alpha, ..*input });
        println!(
            "{alpha:>6.1}  {:>8.4}  {:>8.5}  {:>8.4}  {:>7.1}  {:>7.3}  {:>7.3}  {:>10.5}  {:>10.4}",
            output.cl,
            output.cd,
            output.cm,
            output.cl / output.cd,
            output.top_xtr,
            output.bot_xtr,
            gradient.cl[input_index::ALPHA],
            output.analysis_confidence
        );
    }
    println!();
}

fn sweep_json(model: &NeuralFoil, input: &AeroInput) -> String {
    let rows: Vec<String> = alphas(0.25)
        .map(|alpha| {
            let AeroScalarResult { output, gradient } =
                model.analyze_scalars_with_gradients(&AeroInput { alpha, ..*input });
            format!(
                "        {{\"alpha\": {alpha}, \"cl\": {:.5}, \"cd\": {:.6}, \"cm\": {:.5}, \
                 \"top_xtr\": {:.4}, \"bot_xtr\": {:.4}, \"confidence\": {:.4}, \
                 \"dcl_dalpha\": {:.8}}}",
                output.cl,
                output.cd,
                output.cm,
                output.top_xtr,
                output.bot_xtr,
                output.analysis_confidence,
                gradient.cl[input_index::ALPHA]
            )
        })
        .collect();
    format!(
        "    {{\n      \"re\": {:e},\n      \"polar\": [\n{}\n      ]\n    }}",
        input.re,
        rows.join(",\n")
    )
}

fn print_json(model: &NeuralFoil, input: &AeroInput) -> bora::Result<()> {
    let outline: Vec<String> = generate_kulfan_coordinates(&input.kulfan, 81)?
        .iter()
        .map(|[x, y]| format!("[{x:.5}, {y:.5}]"))
        .collect();
    let series: Vec<String> = REYNOLDS_NUMBERS
        .iter()
        .map(|&re| sweep_json(model, &AeroInput { re, ..*input }))
        .collect();

    println!("{{");
    println!("  \"model\": \"{}\",", model.model_size());
    println!("  \"n_crit\": {},", input.n_crit);
    println!("  \"outline\": [{}],", outline.join(", "));
    println!("  \"series\": [\n{}\n  ]", series.join(",\n"));
    println!("}}");
    Ok(())
}

fn main() -> bora::Result<()> {
    // The cambered airfoil of the basic example, in Kulfan (CST) parameters.
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.leading_edge_weight = 0.02;
    input.kulfan.te_thickness = 0.0025;

    let model = NeuralFoil::new(ModelSize::XLarge)?;

    if std::env::args().any(|arg| arg == "--json") {
        print_json(&model, &input)
    } else {
        for re in REYNOLDS_NUMBERS {
            print_table(&model, &AeroInput { re, ..input });
        }
        Ok(())
    }
}
