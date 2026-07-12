---
sidebar_position: 1
title: "Core Concepts"
---

# Core Concepts

Bora is a C++20 port of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil),
a physics-informed neural network surrogate of XFoil. This page explains the
model and how it maps onto Bora's API.

## How NeuralFoil works

NeuralFoil replaces an XFoil run with the evaluation of a multilayer
perceptron trained on tens of millions of XFoil results. The network takes a
25-dimensional encoding of the airfoil shape and operating condition and
returns aerodynamic coefficients, transition locations, and boundary layer
distributions. Two evaluations are fused per analysis (the airfoil and its
mirror image at negated alpha) so that the symmetry
$C_L(-\alpha) = -C_L(\alpha)$ for a mirrored airfoil holds exactly.

Because the surrogate is smooth, it is well suited to gradient-based
optimization — there is no solver noise or convergence failure, and
derivatives are exact.

## Shape parameterization

Airfoils are described by Kulfan (CST) parameters (`bora::KulfanParameters`):
8 Bernstein shape weights per surface, a leading-edge modification weight, and
the trailing-edge thickness. Utilities are provided to generate coordinates
from parameters (`bora::GenerateKulfanCoordinates`) and to fit parameters to
existing coordinates (`bora::FitKulfanParameters`).

## Inputs and outputs

`bora::AeroInput` bundles the shape with the operating condition: alpha
(degrees), Reynolds number, critical amplification factor `n_crit`, and forced
transition locations. `bora::AeroOutput` contains:

- `CL`, `CD`, `CM` — force and moment coefficients
- `Top_Xtr`, `Bot_Xtr` — transition locations (x/c)
- `upper` / `lower` — boundary layer data at 32 stations per surface
  (momentum thickness `theta`, shape factor `H`, edge velocity ratio
  `ue_over_vinf`); station locations come from `bora::BlXPoints()`
- `analysis_confidence` — see below

## Gradients

`AnalyzeWithGradients()` returns a `bora::AeroResult`, which adds a
`bora::Gradient` (a 23-element array) for every output. Entry indices are
named in `bora::input_index`:

| Index | Input |
|-------|-------|
| 0–7 | Upper surface CST weights |
| 8–15 | Lower surface CST weights |
| 16 | Leading edge weight |
| 17 | TE thickness |
| 18 | Alpha (per degree) |
| 19 | Reynolds number |
| 20 | n_crit |
| 21, 22 | xtr_upper, xtr_lower |

Gradients are computed by forward-mode differentiation through the network and
all output decodings, so they are exact to machine precision (the two clipped
transition outputs have zero gradient where the clip is active).

## Analysis confidence

The network reports a confidence in [0, 1] that decays with the (Mahalanobis)
distance of the query from the training data distribution. Treat results with
low confidence as extrapolation. In optimization, penalizing low confidence is
an effective way to keep the search inside the region where the surrogate is
trustworthy.

## Model sizes

All eight upstream network sizes are available via `bora::ModelSize`, from
`kXXSmall` (3 layers, fastest) to `kXXXLarge` (7 layers, 512 wide, most
accurate). `kXLarge` is the upstream default and a good starting point.

## Error handling

Constructors and loaders throw `std::runtime_error` with a descriptive message
(missing weight files, malformed data, dimension mismatches). Analysis calls
do not throw.
