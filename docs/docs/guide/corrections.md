---
sidebar_position: 3
title: "Mach number, stall and flaps"
description: "The corrected analysis: compressibility, post-stall behaviour over 360 degrees and control surfaces, with gradients."
---

# Mach number, stall and flaps

The network is a surrogate of XFoil at zero Mach number, trained on angles of
attack up to about ±28°. AeroSandbox wraps it in a set of corrections that
extend it to compressible flow, to the full circle of angle of attack and to
sections with deflected control surfaces; this is what its
`get_aero_from_neuralfoil` returns
([Sharpe, 2021](../references.md#aerosandbox)). Bora ports those corrections
as `analyze_corrected`, and adds their analytic derivatives.

```rust
use bora::{ControlSurface, CorrectedInput, ModelSize, NeuralFoil};

let model = NeuralFoil::new(ModelSize::XLarge)?;
let input = CorrectedInput {
    aero: input,          // the AeroInput of a plain analysis
    mach: 0.6,
    control_surfaces: vec![ControlSurface { deflection: 5.0, hinge_point: 0.75 }],
    include_360_deg_effects: true,
};
let aero = model.analyze_corrected(&input);
println!("CL = {:.4}, CD = {:.5}, Mcrit = {:.3}", aero.cl, aero.cd, aero.mach_crit);
```

`CorrectedInput::from(aero_input)` gives Mach 0, no control surfaces and the
360° effects on, which are the upstream defaults.

## Inputs

| Field | Meaning | Default |
|-------|---------|--------:|
| `aero` | Shape and operating point, see [Inputs and outputs](./analysis.md) | |
| `mach` | Freestream Mach number | 0 |
| `control_surfaces` | Any number of `ControlSurface { deflection, hinge_point }` | none |
| `include_360_deg_effects` | Blend into the post-stall model at large angles | `true` |

A control surface has a `deflection` in degrees, positive trailing edge down,
and a `hinge_point` as a fraction of chord (default 0.75).

## Outputs

`CorrectedOutput` has the fields of `AeroOutput` and four more:

| Field | Meaning |
|-------|---------|
| `cl`, `cd`, `cm` | Corrected coefficients; `cd` includes wave drag |
| `top_xtr`, `bot_xtr` | Transition locations, moved to the leading or trailing edge where the flow is separated |
| `cpmin_0` | Minimum pressure coefficient in incompressible flow |
| `cpmin` | Minimum pressure coefficient at the given Mach number |
| `mach_crit` | Critical Mach number |
| `mach_dd` | Drag-divergence Mach number |
| `analysis_confidence`, `upper`, `lower` | The network's own outputs, not corrected |

## What the corrections do

The description below follows the AeroSandbox source
([Sharpe, 2021](../references.md#aerosandbox)), which also names the data
behind each fit.

**Control surfaces.** A deflected surface is not modelled as a change of
shape. It shifts the angle of attack that the network sees, by the deflection
times an effectiveness that falls from 1 for a hinge at the leading edge to 0
at the trailing edge, and it multiplies the drag by a factor that grows with
the square of the deflection. The boundary-layer outputs and the confidence
are therefore those of the undeflected section at the shifted angle.

**Post-stall.** Beyond about 20° in either direction the lift, drag and
moment blend smoothly into a model of fully separated flow that holds at every
angle of attack ([Truong, 2020](../references.md#truong-2020)). The angle of
attack is periodic, so 370° is 10°. Below stall the blend leaves the network's
values almost untouched (a relative change of order 10⁻⁶ at small angles).

**Compressibility.** The minimum pressure coefficient is taken from the
largest edge velocity in the boundary-layer output. It gives the critical Mach
number, and from that the drag-divergence Mach number. Lift, moment and
minimum pressure are scaled by the Prandtl-Glauert factor. Above the critical
Mach number a wave drag is added, rising steeply through drag divergence to a
peak near Mach 1.1 that scales with the thickness of the section. Lift falls
beyond drag divergence to represent buffet, the lift slope drops to its
supersonic value above Mach 1, and the moment is shifted as the aerodynamic
centre moves aft.

These are empirical corrections on an incompressible viscous analysis. They
are meant to give an optimizer the right trends, a drag rise in roughly the
right place and a smooth function everywhere. They are not a transonic flow
solution, and nothing here predicts shock position or shock-induced
separation.

:::note Mach 0 is not exactly the plain analysis
The corrections are smooth blends that never switch off completely. With
`mach: 0.0`, no control surfaces and `include_360_deg_effects: false`, the
lift and moment coefficients are still about 0.4% smaller in magnitude than
those of `analyze`, because the Prandtl-Glauert factor is smoothed through
Mach 1. Bora reproduces this upstream behaviour exactly. Use `analyze` when
you want the network's own result.
:::

## A Mach sweep

`examples/compressible.rs` sweeps the quickstart airfoil through Mach number
at 1° and a Reynolds number of 5 × 10⁶:

```bash
cargo run --release --example compressible
```

```text
alpha = 1.0 deg, Re = 5.0e6: Mcrit = 0.576, Mdd = 0.644

  Mach       CL        CD        CM     Cpmin   dCD/dMach
  0.00   0.7438   0.00561   -0.1340    -1.031     0.00000
  0.30   0.7777   0.00561   -0.1402    -1.078     0.00000
  0.50   0.8491   0.00561   -0.1532    -1.177     0.00000
  0.60   0.9077   0.00564   -0.1705    -1.259     0.00428
  0.65   0.9298   0.00810   -0.2032    -1.311     0.16916
  0.70   0.5788   0.02939   -0.2456    -1.371     0.64762
  0.75   0.5222   0.06907   -0.2947    -1.440     0.89968
  0.80   0.5533   0.11541   -0.3305    -1.517     0.91621
  0.90   0.6473   0.18551   -0.3803    -1.677     0.38737
  1.20   0.6608   0.09285   -0.3524    -1.438    -0.00117
```

Drag is flat up to the critical Mach number of 0.58 and rises steeply past
drag divergence at 0.64, where the lift also breaks. The same program deflects
a flap hinged at 75% of chord at Mach 0.3,

```text
  Flap (deg)       CL        CD        CM   dCL/d(flap)
       -10.0   0.1223   0.00885   -0.1359       0.06701
         0.0   0.7777   0.00561   -0.1402       0.06465
        10.0   1.3921   0.01155   -0.1376       0.05571
        20.0   1.8458   0.05084   -0.1126       0.03105
```

and takes the section around the circle at Mach 0:

```text
  alpha (deg)       CL        CD  confidence
          0.0   0.6312   0.00544      0.9630
         15.0   1.8243   0.03299      0.9452
         25.0   0.9512   0.47996      0.7986
         45.0   1.1161   1.22437      0.0000
         90.0   0.0896   2.08338      0.0000
        135.0  -0.9939   1.22890      0.0000
        180.0   0.0000   0.00338      0.0000
```

The confidence is the network's and falls to zero outside its training range.
There the coefficients come from the post-stall model, which is fitted to a
NACA 0012 and does not depend on the section, so read them as a plausible
continuation for a solver that must not fail at large angles, not as a
prediction for your airfoil.

## Gradients

`analyze_corrected_with_gradients` returns a `CorrectedResult` with the same
outputs and their derivatives:

```rust
use bora::input_index;

let result = model.analyze_corrected_with_gradients(&input);
let dcd_dmach = result.gradient.cd[input_index::MACH];
let dcl_dalpha = result.gradient.cl[input_index::ALPHA];
let dcl_dflap = result.gradient.deflection[0].cl;
```

Each scalar output has a `CorrectedGradient`, an array of
`NUM_CORRECTED_INPUTS` = 24 entries: the 23 entries of a
[`Gradient`](./gradients.md#layout) followed by the Mach number at
`input_index::MACH`. The boundary-layer gradients keep the 23-entry layout,
since the boundary layer does not depend on Mach number. `deflection` holds
one `CorrectedOutput` per control surface, whose fields are the derivatives of
the outputs with respect to that surface's deflection in degrees. There is no
derivative with respect to the hinge position.

The derivatives are exact, with one qualification. The supersonic wave drag
depends on the maximum thickness of the section, which Bora, like AeroSandbox,
takes as the largest of 101 sampled thicknesses. Its derivative is that of the
thickness at the thickest sample, and it jumps when another sample becomes the
thickest. This affects `cd` and `cm` above the drag-divergence Mach number
only.

The corrections need the edge velocities, so the call costs the same as
`analyze_with_gradients`. There is no scalar-only, reverse-mode form.

## Many inputs at once

`analyze_corrected_batch(&inputs)` takes a slice of `CorrectedInput`s and
returns one `CorrectedOutput` each, evaluating the network with the batched
kernel of [`analyze_batch`](./analysis.md#many-inputs-at-once).
