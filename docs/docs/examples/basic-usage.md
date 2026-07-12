---
sidebar_position: 1
title: "Basic Usage"
---

# Basic Usage

Two worked examples ship with the repository (built when `BUILD_EXAMPLES=ON`,
the default).

## Airfoil analysis with gradients

`examples/example.cpp` analyzes a cambered airfoil at alpha = 3°, Re = 10⁶ and
prints coefficients and selected gradients:

```cpp
const bora::NeuralFoil model(bora::ModelSize::kXLarge);
const bora::AeroResult result = model.AnalyzeWithGradients(input);
```

Expected output (xlarge model):

```
alpha = 3.00 deg, Re = 1.00e+06
  CL  =   0.95162
  CD  =   0.00729
  CM  =  -0.13014
  L/D =   130.585
  transition: upper x/c = 0.490, lower x/c = 0.921
  analysis confidence = 0.9643

Gradients (analytic, for optimization):
  dCL/dalpha =   0.10185 per deg
  ...
```

## Gradient-based airfoil optimization

`examples/optimize_airfoil.cpp` minimizes drag at a fixed lift target
(CL = 0.8, Re = 10⁶), starting from a NACA 0012. Design variables are the 16
CST weights, the leading edge weight, and alpha; constraints are enforced with
quadratic penalties and the design is updated with Adam using Bora's analytic
gradients:

```
  iter   objective         CL         CD       L/D   confidence
     1    0.123803    0.45873    0.00734     62.50       0.9635
   100    0.006122    0.79903    0.00612    130.54       0.9685
   500    0.005174    0.79959    0.00517    154.56       0.9473
```

Drag drops by about 30% at the lift target in 500 iterations (a fraction of a
second of compute). Two details worth copying into your own optimization:

- **Penalize low analysis confidence.** This keeps the optimizer inside the
  region where the neural network is trustworthy and prevents degenerate
  shapes.
- **Let alpha trim the lift constraint.** Including alpha as a design variable
  alongside the shape makes the CL target easy to hold.

The example writes the optimized shape to `optimized_airfoil.dat` (Selig
format) via `bora::GenerateKulfanCoordinates()`.
