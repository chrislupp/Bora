---
sidebar_position: 1
title: "Basic Usage"
---

# Basic Usage

Two worked examples ship with the repository under `examples/`. Run them with
Cargo from the repository root.

## Airfoil analysis with gradients

`examples/example.rs` analyzes a cambered airfoil at alpha = 3°, Re = 10⁶ and
prints coefficients and selected gradients:

```bash
cargo run --release --example example
```

The core of the example (abridged; the full file also prints CM, transition
locations, and confidence):

```rust
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

Full output (xlarge model):

```
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

## Gradient-based airfoil optimization

`examples/optimize_airfoil.rs` minimizes drag at a fixed lift target
(CL = 0.8, Re = 10⁶), starting from a NACA 0012. Design variables are the 16
CST weights, the leading edge weight, and alpha; constraints are enforced with
quadratic penalties and the design is updated with Adam using Bora's analytic
gradients:

```bash
cargo run --release --example optimize_airfoil
```

Each iteration evaluates the objective and its gradient from a single
`analyze_scalars_with_gradients()` call. The objective uses only scalar
outputs, so the reverse-mode scalar path is enough, and it is ~4.5x faster than
`analyze_with_gradients()` for the xlarge model:

```rust
let result = model.analyze_scalars_with_gradients(&input);
let aero = result.output;

let cl_error = aero.cl - CL_TARGET;
let confidence_deficit = (CONFIDENCE_FLOOR - aero.analysis_confidence).max(0.0);

let value = aero.cd
    + CL_PENALTY_WEIGHT * cl_error * cl_error
    + CONFIDENCE_PENALTY_WEIGHT * confidence_deficit * confidence_deficit;

let g = &result.gradient;
let gradient: [f64; NUM_DESIGN_VARS] = std::array::from_fn(|i| {
    let k = design_to_input_index(i); // CST/LE weights map 1:1; last entry is input_index::ALPHA
    g.cd[k] + 2.0 * CL_PENALTY_WEIGHT * cl_error * g.cl[k]
        - 2.0 * CONFIDENCE_PENALTY_WEIGHT * confidence_deficit * g.analysis_confidence[k]
});
```

Output (xlarge model):

```
Minimizing CD at CL = 0.80, Re = 1.0e6 (xlarge model)

  iter   objective         CL         CD       L/D   confidence
     1    0.123803    0.45873    0.00734     62.50       0.9635
    50    0.006915    0.78887    0.00679    116.17       0.9562
   100    0.006122    0.79903    0.00612    130.54       0.9685
   150    0.005748    0.79911    0.00575    139.05       0.9723
   200    0.005572    0.79914    0.00557    143.43       0.9714
   250    0.005477    0.79970    0.00548    146.00       0.9687
   300    0.005411    0.79971    0.00541    147.80       0.9665
   350    0.005349    0.79970    0.00535    149.52       0.9633
   400    0.005289    0.79968    0.00529    151.21       0.9591
   450    0.005230    0.79965    0.00523    152.89       0.9537
   500    0.005174    0.79959    0.00517    154.56       0.9473

Optimized airfoil (Kulfan parameters):
  upper_weights = {0.097579, 0.146209, 0.215940, 0.161252, 0.254478, 0.300737, 0.213287, 0.125876}
  lower_weights = {-0.058255, 0.027221, -0.067101, -0.050751, -0.087094, -0.033425, -0.016107, -0.032550}
  leading_edge_weight = 0.191559
  alpha = 3.997 deg

Wrote optimized_airfoil.dat
```

Drag drops by about 30% at the lift target in 500 iterations (a fraction of a
second of compute). Two details worth copying into your own optimization:

- **Penalize low analysis confidence.** This keeps the optimizer inside the
  region where the neural network is trustworthy and prevents degenerate
  shapes.
- **Let alpha trim the lift constraint.** Including alpha as a design variable
  alongside the shape makes the CL target easy to hold.

The example writes the optimized shape to `optimized_airfoil.dat` (Selig
format) in the current working directory via
`bora::generate_kulfan_coordinates()`.
