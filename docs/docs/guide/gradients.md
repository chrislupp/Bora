---
sidebar_position: 3
title: "Gradients"
description: "How Bora returns exact derivatives of its outputs, how they are laid out, and which of the two gradient calls to use."
---

# Gradients

Bora can return the derivative of every output with respect to all 23 inputs.
The derivatives are analytic: they are propagated through the network and the
input and output transformations by hand-written differentiation code, not
estimated by finite differences. They are exact to rounding error.

NeuralFoil's Python implementation leaves differentiation to an external
automatic-differentiation tool ([Sharpe, 2023](../references.md#neuralfoil-repository)).
Having the derivatives built in, with no extra dependency, is the main thing
Bora adds to the model.

## Layout

A gradient is a fixed-size array with one entry per input:

```rust
pub const NUM_INPUTS: usize = 23;
pub type Gradient = [f64; NUM_INPUTS];
```

The constants in `bora::input_index` name the entries.

| Index | Constant | Input | Derivative is per |
|------:|----------|-------|-------------------|
| 0–7 | `UPPER_WEIGHTS` + k | Upper-surface weight k | unit weight |
| 8–15 | `LOWER_WEIGHTS` + k | Lower-surface weight k | unit weight |
| 16 | `LEADING_EDGE_WEIGHT` | Leading-edge weight | unit weight |
| 17 | `TE_THICKNESS` | Trailing-edge thickness | chord length |
| 18 | `ALPHA` | Angle of attack | degree |
| 19 | `RE` | Reynolds number | unit Reynolds number |
| 20 | `N_CRIT` | Critical amplification factor | unit |
| 21 | `XTR_UPPER` | Forced transition, upper | chord fraction |
| 22 | `XTR_LOWER` | Forced transition, lower | chord fraction |

The gradient types mirror the output types field for field, so the derivative
of an output lives under the same name:

```rust
use bora::input_index as idx;

let result = model.analyze_with_gradients(&input);

// Lift-curve slope, per degree.
let cl_alpha = result.gradient.cl[idx::ALPHA];

// Sensitivity of drag to the third upper-surface weight.
let dcd_dw = result.gradient.cd[idx::UPPER_WEIGHTS + 2];

// Sensitivity of the last upper-surface momentum thickness to Reynolds number.
let dtheta_dre = result.gradient.upper.theta[bora::NUM_BL_POINTS - 1][idx::RE];
```

:::info Units
Derivatives with respect to `alpha` are per degree, not per radian. Derivatives
with respect to `re` are per unit of Reynolds number and therefore very small
numbers; scale your design variables accordingly if you optimize over
Reynolds number.
:::

## Two gradient calls

| | `analyze_with_gradients` | `analyze_scalars_with_gradients` |
|---|---|---|
| Outputs differentiated | All 198 | The 6 scalars |
| Differentiation mode | Forward: one pass per input | Reverse: one pass per output |
| Returns | `AeroResult` | `AeroScalarResult` |
| Time, NeuralFoil xlarge network | 322 µs | 72 µs |

The cost of forward-mode differentiation grows with the number of inputs and
that of reverse mode with the number of outputs. With 23 inputs, 198 outputs
in total and 6 scalar outputs, reverse mode is the cheaper way to
differentiate the scalars and forward mode the cheaper way to differentiate
everything.

**Use the scalar call unless you need boundary-layer derivatives.** An
objective or constraint built from lift, drag, moment, transition location or
confidence needs nothing else, and the scalar call is four to eight times
faster depending on the network size (see
[Choosing a network size](./model-sizes.md#speed)). The field names are the
same, so switching is a one-line change:

```rust
let result = model.analyze_scalars_with_gradients(&input);
let dcd_dalpha = result.gradient.cd[idx::ALPHA];
```

The scalar values from both calls are bitwise identical to `analyze`. The
gradients from the two calls agree to rounding, not bitwise, because the same
sums are accumulated in a different order; across the test suite the
difference stays below 10⁻¹³ of each gradient's largest entry.

## Transition locations

The two transition outputs are clipped to the interval from 0 to 1. Where the
clip is active, typically a surface that stays laminar to the trailing edge
and reports exactly 1, the output does not change with the inputs and its
gradient is exactly zero. An optimizer that constrains a transition location
should expect a flat region there.

## How the gradients are verified

Three independent checks run in the test suite:

1. Against Jacobians of Python NeuralFoil computed with CasADi, to a relative
   tolerance of 10⁻⁶ (`tests/reference_gradients.rs`).
2. Against central finite differences of Bora's own outputs
   (`tests/gradients.rs`).
3. The reverse-mode scalar gradients against the forward-mode ones
   (`tests/scalars.rs`).

## Using the gradients in an optimizer

[Airfoil optimization](../examples/optimization.mdx) is a complete example. The
pattern is to build the objective from the outputs and assemble its gradient
from the corresponding rows by the chain rule:

```rust
let result = model.analyze_scalars_with_gradients(&input);
let aero = result.output;
let g = &result.gradient;

// Minimize drag with a quadratic penalty on missing a lift target.
let cl_error = aero.cl - CL_TARGET;
let objective = aero.cd + WEIGHT * cl_error * cl_error;

let gradient: [f64; bora::NUM_INPUTS] =
    std::array::from_fn(|k| g.cd[k] + 2.0 * WEIGHT * cl_error * g.cl[k]);
```
