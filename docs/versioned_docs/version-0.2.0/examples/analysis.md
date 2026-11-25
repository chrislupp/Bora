---
sidebar_position: 1
title: "Analysis with gradients"
description: "The basic example: one airfoil, one operating point, coefficients and selected gradients."
---

# Analysis with gradients

`examples/example.rs` analyzes one airfoil at one operating point and prints
the coefficients with a selection of their derivatives. Run it from the
repository root:

```bash
cargo run --release --example example
```

```text
Bora 0.1.0 (NeuralFoil port), model size 'xlarge'

alpha = 3.00 deg, Re = 1.00e6
  CL  =   0.95162
  CD  =   0.00729
  CM  =  -0.13014
  L/D =   130.585
  transition: upper x/c = 0.490, lower x/c = 0.921
  analysis confidence = 0.9643

Gradients (analytic, for optimization):
  dCL/dalpha =   0.10185 per deg
  dCM/dalpha =   0.00158 per deg
  dCD/dRe    = -9.517e-10
  dCL/d(upper CST weight 0..7):
      0.0876  -0.0515   0.1400   0.4751   0.5213   0.3174   0.1829   0.2422
```

## The code

The example builds the input, loads the default network and makes a single
call. This is the core of it; the full file also prints the moment,
transition locations and confidence.

```rust title="examples/example.rs (abridged)"
use bora::{AeroInput, ModelSize, NUM_CST_WEIGHTS_PER_SIDE, NeuralFoil, input_index as idx};

fn main() -> bora::Result<()> {
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

    println!("  CL  = {:9.5}", aero.cl);
    println!("  CD  = {:9.5}", aero.cd);
    println!("  L/D = {:9.3}", aero.cl / aero.cd);

    let gradient = &result.gradient;
    println!("  dCL/dalpha = {:9.5} per deg", gradient.cl[idx::ALPHA]);
    println!("  dCD/dRe    = {:9.3e}", gradient.cd[idx::RE]);
    for i in 0..NUM_CST_WEIGHTS_PER_SIDE {
        print!(" {:8.4}", gradient.cl[idx::UPPER_WEIGHTS + i]);
    }
    println!();
    Ok(())
}
```

## Reading the output

**The lift-curve slope** of 0.102 per degree is 5.84 per radian, a little
under the thin-airfoil value of $2\pi \approx 6.28$, as expected for a viscous
section.

**The derivative of drag with respect to Reynolds number** is about
−10⁻⁹ per unit Reynolds number. It looks negligible only because of the
units: raising the Reynolds number from 1 to 2 million at this slope would
lower the drag coefficient by about 0.001, or 13%. When Reynolds number is a
design variable, optimize over its logarithm or rescale it.

**The shape sensitivities** show where lift is made. The derivative of lift
with respect to the upper-surface weights peaks at weights 3 and 4, which act
around mid-chord, and is small or negative near the nose. A design change
there buys little lift.

**Transition** occurs at 49% chord on the upper surface and 92% on the lower.
The [polar sweep](./polar.mdx) shows how those move with angle of attack.
