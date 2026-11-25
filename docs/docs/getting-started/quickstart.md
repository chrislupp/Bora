---
sidebar_position: 2
title: "Quickstart"
description: "Analyze an airfoil with Bora and read off coefficients and their gradients."
---

# Quickstart

This page analyzes one airfoil at one operating point and reads off the
coefficients and a gradient. It assumes Bora is already a dependency of your
project; see [Installation](./installation.md) if not.

## A complete program

```rust title="src/main.rs"
use bora::{AeroInput, ModelSize, NeuralFoil, input_index};

fn main() -> bora::Result<()> {
    // Airfoil shape as Kulfan (CST) parameters: 8 weights per surface.
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.leading_edge_weight = 0.02;
    input.kulfan.te_thickness = 0.0025;

    // Operating point.
    input.alpha = 3.0; // degrees
    input.re = 1e6;

    // Load the network once, then analyze as often as needed.
    let model = NeuralFoil::new(ModelSize::XLarge)?;
    let result = model.analyze_with_gradients(&input);

    println!(
        "CL = {:.4}, CD = {:.5}, CM = {:.4}",
        result.output.cl, result.output.cd, result.output.cm
    );
    println!(
        "dCL/dalpha = {:.4} per degree",
        result.gradient.cl[input_index::ALPHA]
    );
    Ok(())
}
```

```bash
cargo run --release
```

```text
CL = 0.9516, CD = 0.00729, CM = -0.1301
dCL/dalpha = 0.1019 per degree
```

This cambered section is the one used in every example in these docs and in
the figure on the home page.

## What each step does

### Describe the airfoil

Bora takes the shape as 18 numbers: eight weights for each surface, a
leading-edge weight and the trailing-edge thickness
(`bora::KulfanParameters`). If you have coordinates instead, fit the
parameters with `bora::fit_kulfan_parameters`.
[Airfoil geometry](../guide/geometry.mdx) explains the parameterization and the
fit.

### Set the operating point

`AeroInput` also carries the angle of attack in degrees (`alpha`), the
chord Reynolds number (`re`), the critical amplification factor (`n_crit`) and
forced transition locations (`xtr_upper`, `xtr_lower`).
`AeroInput::default()` sets `re = 1e6`, `n_crit = 9.0` and free transition
(`xtr_upper = xtr_lower = 1.0`), so you only set what differs.

### Load the model

`NeuralFoil::new` reads the NeuralFoil network weights from disk. This is the only step
that can fail, and it returns a `bora::Result`. The model is immutable
afterwards and can be shared between threads.

There are eight network sizes, from `ModelSize::XXSmall` to
`ModelSize::XXXLarge`. `XLarge` is NeuralFoil's default and a sound first
choice; [Choosing a network size](../guide/model-sizes.md) has the numbers.

### Analyze

There are four main analysis calls. All of them are infallible.

| Call | Returns | Gradients |
|------|---------|-----------|
| `analyze` | Everything, including the boundary layer | No |
| `analyze_with_gradients` | Everything | Of every output |
| `analyze_scalars` | The six scalar outputs | No |
| `analyze_scalars_with_gradients` | The six scalar outputs | Of the six scalars |

The six scalar outputs are `cl`, `cd`, `cm`, `top_xtr`, `bot_xtr` and
`analysis_confidence`. If those are all you need, as in most optimization
problems, the scalar calls return identical values and the gradient version is
four to eight times faster. See [Gradients](../guide/gradients.md).

Two more forms cover other needs: `analyze_with_tangent` returns the
derivative of every output along one input direction at about twice the cost
of `analyze` (see [Gradients](../guide/gradients.md#directional-derivatives)),
and `analyze_batch` and `analyze_scalars_batch` analyze many inputs at once,
about twice as fast per input as a loop (see
[Many inputs at once](../guide/analysis.md#many-inputs-at-once)).

## Check the confidence

Every result carries `analysis_confidence`. Read it before you trust the other
numbers:

```rust
let aero = model.analyze_scalars(&input);
if aero.analysis_confidence < 0.9 {
    eprintln!("low confidence ({:.2}): treat this result as extrapolation",
        aero.analysis_confidence);
}
```

[Analysis confidence](../guide/confidence.md) explains what the number means
and how to use it as an optimization constraint.

## Next

- [Inputs and outputs](../guide/analysis.md) for every field and its units.
- [Polar sweep](../examples/polar.mdx) for a lift curve and drag polar.
- [Multipoint optimization with OpenMDAO](../examples/openmdao.mdx) for a
  complete gradient-based design problem.
