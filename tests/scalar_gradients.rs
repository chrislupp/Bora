//! Checks the reverse-mode scalar gradients (`analyze_with_scalar_gradients`,
//! `Network::backpropagate`) against the forward-mode gradients of
//! `analyze_with_gradients` / `Network::evaluate_with_jacobian`.

mod common;

use bora::{
    AeroInput, Gradient, ModelSize, NUM_INPUTS, NUM_LATENT_INPUTS, NUM_LATENT_OUTPUTS, NetworkTape,
    ScalarOutput,
};
use common::{Failures, input_from_array, model, output_ref_data};

/// Relative tolerance on each gradient entry, scaled by the largest entry of
/// the forward-mode gradient (reverse and forward mode sum the same terms in a
/// different order, so they agree to rounding, not bit for bit).
const RELATIVE_TOLERANCE: f64 = 1e-10;

/// A cambered airfoil (the crate-level documentation example).
fn base_airfoil() -> AeroInput {
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.te_thickness = 0.0025;
    input.alpha = 3.0;
    input
}

/// Inputs spanning alpha, Re, n_crit, forced transition, and non-default
/// Kulfan parameters, plus a few reference airfoils.
fn test_inputs() -> Vec<(String, AeroInput)> {
    let mut cases = Vec::new();
    let base = base_airfoil();
    for alpha in [-10.0, -2.0, 0.0, 3.0, 9.0, 16.0] {
        cases.push((format!("alpha {alpha}"), AeroInput { alpha, ..base }));
    }
    for re in [3e4, 2e5, 5e6, 5e7] {
        cases.push((format!("Re {re:e}"), AeroInput { re, ..base }));
    }
    for n_crit in [3.0, 12.0] {
        cases.push((format!("n_crit {n_crit}"), AeroInput { n_crit, ..base }));
    }
    for (xtr_upper, xtr_lower) in [(0.05, 1.0), (0.3, 0.6), (1.0, 0.02)] {
        cases.push((
            format!("forced transition {xtr_upper}/{xtr_lower}"),
            AeroInput {
                xtr_upper,
                xtr_lower,
                ..base
            },
        ));
    }
    let mut shaped = base;
    shaped.kulfan.leading_edge_weight = 0.15;
    shaped.kulfan.te_thickness = 0.008;
    shaped.kulfan.upper_weights[2] += 0.1;
    shaped.kulfan.lower_weights[5] -= 0.07;
    shaped.alpha = 5.5;
    shaped.re = 8e5;
    shaped.n_crit = 7.0;
    cases.push(("modified Kulfan".into(), shaped));
    cases.push(("flat plate".into(), AeroInput::default()));

    let data = output_ref_data();
    for c in [0, 13, 27, 40, 55] {
        let case = c.min(data.n_cases - 1);
        cases.push((
            format!("reference {}", data.names[case]),
            input_from_array(&data.inputs[case * NUM_INPUTS..]),
        ));
    }
    cases
}

fn expect_gradient(
    failures: &mut Failures,
    actual: &Gradient,
    expected: &Gradient,
    what: impl Fn() -> String,
) {
    let scale = expected.iter().fold(0.0_f64, |m, g| m.max(g.abs()));
    let tolerance = RELATIVE_TOLERANCE * scale;
    for k in 0..NUM_INPUTS {
        failures.expect_near(actual[k], expected[k], tolerance, || {
            format!("{} d/d(input {k})", what())
        });
    }
}

fn check_model(size: ModelSize) {
    let model = model(size.as_str());
    let mut failures = Failures::default();
    let (mut clip_active, mut clip_inactive) = (0, 0);

    for (name, input) in test_inputs() {
        let forward = model.analyze_with_gradients(&input);
        let reverse = model.analyze_with_scalar_gradients(&input, &ScalarOutput::ALL);

        assert_eq!(
            reverse.output,
            model.analyze(&input),
            "{size} {name}: output"
        );
        assert_eq!(reverse.gradients.len(), ScalarOutput::ALL.len());
        for (output, gradient) in ScalarOutput::ALL.iter().zip(&reverse.gradients) {
            expect_gradient(
                &mut failures,
                gradient,
                output.gradient_of(&forward.gradient),
                || format!("{size} {name}: {output:?}"),
            );
        }

        for xtr in [reverse.output.top_xtr, reverse.output.bot_xtr] {
            if xtr == 0.0 || xtr == 1.0 {
                clip_active += 1;
            } else {
                clip_inactive += 1;
            }
        }
        for (output, gradient) in [
            (ScalarOutput::TopXtr, &reverse.gradients[4]),
            (ScalarOutput::BotXtr, &reverse.gradients[5]),
        ] {
            let value = output.value_of(&reverse.output);
            if value == 0.0 || value == 1.0 {
                assert!(
                    gradient.iter().all(|&g| g == 0.0),
                    "{size} {name}: {output:?} is clipped but has a nonzero gradient"
                );
            }
        }
    }
    failures.assert_none();
    // The inputs must exercise both sides of the transition clip.
    assert!(
        clip_active > 0,
        "{size}: no case with an active transition clip"
    );
    assert!(
        clip_inactive > 0,
        "{size}: no case with an inactive transition clip"
    );
}

#[test]
fn matches_forward_mode_xxsmall() {
    check_model(ModelSize::XXSmall);
}

#[test]
fn matches_forward_mode_xsmall() {
    check_model(ModelSize::XSmall);
}

#[test]
fn matches_forward_mode_small() {
    check_model(ModelSize::Small);
}

#[test]
fn matches_forward_mode_medium() {
    check_model(ModelSize::Medium);
}

#[test]
fn matches_forward_mode_large() {
    check_model(ModelSize::Large);
}

#[test]
fn matches_forward_mode_xlarge() {
    check_model(ModelSize::XLarge);
}

#[test]
fn matches_forward_mode_xxlarge() {
    check_model(ModelSize::XXLarge);
}

#[test]
fn matches_forward_mode_xxxlarge() {
    check_model(ModelSize::XXXLarge);
}

#[test]
fn returns_gradients_in_request_order() {
    let model = model("medium");
    let input = base_airfoil();
    let forward = model.analyze_with_gradients(&input);
    let mut failures = Failures::default();

    let requests: [&[ScalarOutput]; 5] = [
        &[ScalarOutput::Cl],
        &[ScalarOutput::Cd],
        &[ScalarOutput::Cl, ScalarOutput::Cd],
        &[
            ScalarOutput::Cd,
            ScalarOutput::AnalysisConfidence,
            ScalarOutput::Cd,
        ],
        &[
            ScalarOutput::BotXtr,
            ScalarOutput::Cm,
            ScalarOutput::Cl,
            ScalarOutput::Cl,
            ScalarOutput::TopXtr,
        ],
    ];
    for request in requests {
        let reverse = model.analyze_with_scalar_gradients(&input, request);
        assert_eq!(reverse.output, model.analyze(&input));
        assert_eq!(reverse.gradients.len(), request.len());
        for (output, gradient) in request.iter().zip(&reverse.gradients) {
            expect_gradient(
                &mut failures,
                gradient,
                output.gradient_of(&forward.gradient),
                || format!("{request:?}: {output:?}"),
            );
        }
        // A repeated output yields identical gradients.
        for (i, a) in request.iter().enumerate() {
            for (j, b) in request.iter().enumerate() {
                if a == b {
                    assert_eq!(reverse.gradients[i], reverse.gradients[j]);
                }
            }
        }
    }
    failures.assert_none();

    // The gradient of a single output does not depend on what else is requested.
    let alone = model.analyze_with_scalar_gradients(&input, &[ScalarOutput::Cd]);
    let together = model.analyze_with_scalar_gradients(&input, &ScalarOutput::ALL);
    assert_eq!(alone.gradients[0], together.gradients[2]);
}

#[test]
fn empty_request_returns_output_only() {
    let model = model("small");
    let input = base_airfoil();
    let result = model.analyze_with_scalar_gradients(&input, &[]);
    assert!(result.gradients.is_empty());
    assert_eq!(result.output, model.analyze(&input));
}

#[test]
fn network_backpropagate_matches_forward_jacobian() {
    for size in [ModelSize::XXSmall, ModelSize::Large] {
        let network =
            bora::Network::load(format!("{}/nn-{size}.h5", common::WEIGHTS_DATA_DIR)).unwrap();
        let x: Vec<f64> = (0..NUM_LATENT_INPUTS)
            .map(|i| 0.3 * ((i as f64) * 0.7).sin())
            .collect();

        // Forward mode with an identity seed gives the full Jacobian.
        let mut identity = vec![0.0; NUM_LATENT_INPUTS * NUM_LATENT_INPUTS];
        for i in 0..NUM_LATENT_INPUTS {
            identity[i * NUM_LATENT_INPUTS + i] = 1.0;
        }
        let mut y = vec![0.0; NUM_LATENT_OUTPUTS];
        let mut jacobian = vec![0.0; NUM_LATENT_OUTPUTS * NUM_LATENT_INPUTS];
        network.evaluate_with_jacobian(&x, &identity, NUM_LATENT_INPUTS, &mut y, &mut jacobian);

        let mut y_plain = vec![0.0; NUM_LATENT_OUTPUTS];
        network.evaluate(&x, &mut y_plain);
        let mut y_taped = vec![0.0; NUM_LATENT_OUTPUTS];
        let mut tape = NetworkTape::default();
        network.evaluate_with_tape(&x, &mut y_taped, &mut tape);
        assert_eq!(y_taped, y_plain, "{size}: taped evaluation differs");

        // Three dense cotangents (exercising every weight row, including the
        // final layer's rows beyond the last full block of four).
        let n = 3;
        let y_bar: Vec<f64> = (0..n * NUM_LATENT_OUTPUTS)
            .map(|i| ((i as f64) * 0.37).cos())
            .collect();
        let mut x_bar = vec![0.0; n * NUM_LATENT_INPUTS];
        network.backpropagate(&mut tape, &y_bar, n, &mut x_bar);

        let mut failures = Failures::default();
        for j in 0..n {
            for i in 0..NUM_LATENT_INPUTS {
                let expected: f64 = (0..NUM_LATENT_OUTPUTS)
                    .map(|o| {
                        y_bar[j * NUM_LATENT_OUTPUTS + o] * jacobian[o * NUM_LATENT_INPUTS + i]
                    })
                    .sum();
                failures.expect_near(
                    x_bar[j * NUM_LATENT_INPUTS + i],
                    expected,
                    1e-10 * expected.abs().max(1.0),
                    || format!("{size}: cotangent {j}, x_bar[{i}]"),
                );
            }
        }
        failures.assert_none();

        // The tape can be swept again, and a single cotangent matches.
        let mut x_bar_single = vec![0.0; NUM_LATENT_INPUTS];
        network.backpropagate(
            &mut tape,
            &y_bar[NUM_LATENT_OUTPUTS..2 * NUM_LATENT_OUTPUTS],
            1,
            &mut x_bar_single,
        );
        assert_eq!(
            x_bar_single,
            x_bar[NUM_LATENT_INPUTS..2 * NUM_LATENT_INPUTS]
        );
    }
}

#[test]
#[should_panic(expected = "tape was not recorded by this network")]
fn backpropagate_rejects_foreign_tape() {
    let small = bora::Network::load(format!("{}/nn-xxsmall.h5", common::WEIGHTS_DATA_DIR)).unwrap();
    let large = bora::Network::load(format!("{}/nn-large.h5", common::WEIGHTS_DATA_DIR)).unwrap();
    let mut tape = NetworkTape::default();
    let mut y = vec![0.0; NUM_LATENT_OUTPUTS];
    small.evaluate_with_tape(&[0.1; NUM_LATENT_INPUTS], &mut y, &mut tape);
    let mut x_bar = vec![0.0; NUM_LATENT_INPUTS];
    large.backpropagate(&mut tape, &y, 1, &mut x_bar);
}
