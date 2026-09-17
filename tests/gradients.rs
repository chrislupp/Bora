//! Checks the analytic gradients against central finite differences.

mod common;

use bora::{AeroInput, NUM_CST_WEIGHTS_PER_SIDE, NUM_INPUTS, input_index};
use common::{Failures, input_from_array, model, output_ref_data};

fn array_from_input(input: &AeroInput) -> [f64; NUM_INPUTS] {
    let mut values = [0.0; NUM_INPUTS];
    let w = NUM_CST_WEIGHTS_PER_SIDE;
    values[input_index::UPPER_WEIGHTS..][..w].copy_from_slice(&input.kulfan.upper_weights);
    values[input_index::LOWER_WEIGHTS..][..w].copy_from_slice(&input.kulfan.lower_weights);
    values[input_index::LEADING_EDGE_WEIGHT] = input.kulfan.leading_edge_weight;
    values[input_index::TE_THICKNESS] = input.kulfan.te_thickness;
    values[input_index::ALPHA] = input.alpha;
    values[input_index::RE] = input.re;
    values[input_index::N_CRIT] = input.n_crit;
    values[input_index::XTR_UPPER] = input.xtr_upper;
    values[input_index::XTR_LOWER] = input.xtr_lower;
    values
}

/// Checks one analytic gradient against a central finite difference.
fn expect_gradient_near(
    failures: &mut Failures,
    analytic: f64,
    numeric: f64,
    fd_scale: f64,
    what: impl FnOnce() -> String,
) {
    // The FD truncation/roundoff error floor scales with the output magnitude.
    let tolerance = 5e-5 * analytic.abs().max(numeric.abs()).max(fd_scale);
    failures.expect_near(analytic, numeric, tolerance, what);
}

fn check_case(case: usize) {
    let data = output_ref_data();
    assert!(case < data.n_cases);
    let name = &data.names[case];
    // Model size in the reference case is ignored; gradients use the medium model.
    let model = model("medium");

    let input = input_from_array(&data.inputs[case * NUM_INPUTS..]);
    let result = model.analyze_with_gradients(&input);
    let g = &result.gradient;
    let base_values = array_from_input(&input);
    let mut failures = Failures::default();

    for k in 0..NUM_INPUTS {
        // Relative step for the large-magnitude Re input; a smaller step for
        // TE thickness, which is scaled by 50 in the latent space and feeds
        // steep exponential decodings; absolute step otherwise.
        let h = match k {
            input_index::RE => 1e-4 * base_values[k],
            input_index::TE_THICKNESS => 1e-7,
            _ => 1e-6 * base_values[k].abs().max(1.0),
        };

        let mut perturbed = base_values;
        perturbed[k] = base_values[k] + h;
        let plus = model.analyze(&input_from_array(&perturbed));
        perturbed[k] = base_values[k] - h;
        let minus = model.analyze(&input_from_array(&perturbed));
        let fd = |p: f64, m: f64| (p - m) / (2.0 * h);
        let suffix = format!("{name}: d/d(input {k})");

        let mut check = |analytic: f64, p: f64, m: f64, fd_scale: f64, output: &str| {
            expect_gradient_near(&mut failures, analytic, fd(p, m), fd_scale, || {
                format!("{output} {suffix}")
            });
        };
        check(g.cl[k], plus.cl, minus.cl, plus.cl.abs() * 1e-7 / h, "CL");
        check(g.cd[k], plus.cd, minus.cd, plus.cd.abs() * 1e-7 / h, "CD");
        check(g.cm[k], plus.cm, minus.cm, plus.cm.abs() * 1e-7 / h, "CM");
        check(
            g.analysis_confidence[k],
            plus.analysis_confidence,
            minus.analysis_confidence,
            1e-7 / h,
            "analysis_confidence",
        );
        check(
            g.top_xtr[k],
            plus.top_xtr,
            minus.top_xtr,
            1e-7 / h,
            "Top_Xtr",
        );
        check(
            g.bot_xtr[k],
            plus.bot_xtr,
            minus.bot_xtr,
            1e-7 / h,
            "Bot_Xtr",
        );

        // Spot-check the boundary layer gradients at a few stations.
        for i in [0, 15, 31] {
            check(
                g.upper.theta[i][k],
                plus.upper.theta[i],
                minus.upper.theta[i],
                plus.upper.theta[i].abs() * 1e-7 / h,
                &format!("upper theta {i}"),
            );
            check(
                g.lower.h[i][k],
                plus.lower.h[i],
                minus.lower.h[i],
                plus.lower.h[i].abs() * 1e-7 / h,
                &format!("lower H {i}"),
            );
            check(
                g.upper.ue_over_vinf[i][k],
                plus.upper.ue_over_vinf[i],
                minus.upper.ue_over_vinf[i],
                1e-7 / h,
                &format!("upper ue {i}"),
            );
        }
    }
    failures.assert_none();
}

// Reference cases 1, 21, 41, 61 span different airfoils and conditions.

#[test]
fn matches_finite_differences_case_1() {
    check_case(1);
}

#[test]
fn matches_finite_differences_case_21() {
    check_case(21);
}

#[test]
fn matches_finite_differences_case_41() {
    check_case(41);
}

#[test]
fn matches_finite_differences_case_61() {
    check_case(61);
}
