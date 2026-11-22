---
sidebar_position: 1
title: "Core Concepts"
---

# Core Concepts

Bora is a Rust port of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil),
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
8 Bernstein shape weights per surface (`upper_weights`, `lower_weights`,
ordered from leading edge to trailing edge), a leading-edge modification weight
(`leading_edge_weight`), and the trailing-edge thickness (`te_thickness`).
Utilities are provided to generate Selig-format coordinates from parameters
(`bora::generate_kulfan_coordinates`) and to fit parameters to existing
coordinates (`bora::fit_kulfan_parameters`).

## Inputs and outputs

`bora::AeroInput` bundles the shape (`kulfan`) with the operating condition:
`alpha` (degrees), Reynolds number `re`, critical amplification factor
`n_crit`, and forced transition locations `xtr_upper` / `xtr_lower`.
`AeroInput::default()` sets `re = 1e6`, `n_crit = 9.0`, and both transition
locations to 1.0 (free transition). `bora::AeroOutput` contains:

- `cl`, `cd`, `cm` — force and moment coefficients
- `top_xtr`, `bot_xtr` — transition locations (x/c)
- `upper` / `lower` — `BoundaryLayerData` at `NUM_BL_POINTS` (32) stations per
  surface (momentum thickness `theta`, shape factor `h`, edge velocity ratio
  `ue_over_vinf`); station locations come from `bora::bl_x_points()`
- `analysis_confidence` — see below

## Gradients

`analyze_with_gradients()` returns a `bora::AeroResult` whose `output` field
is the `AeroOutput` and whose `gradient` field (`AeroGradients`) holds a
`bora::Gradient` — a `[f64; NUM_INPUTS]` array with `NUM_INPUTS = 23` — for
every output, mirroring the `AeroOutput` layout. Entry indices are named in
`bora::input_index`:

| Index | Constant | Input |
|-------|----------|-------|
| 0–7 | `UPPER_WEIGHTS` (+ i) | Upper surface CST weights |
| 8–15 | `LOWER_WEIGHTS` (+ i) | Lower surface CST weights |
| 16 | `LEADING_EDGE_WEIGHT` | Leading edge weight |
| 17 | `TE_THICKNESS` | TE thickness |
| 18 | `ALPHA` | Alpha (per degree) |
| 19 | `RE` | Reynolds number |
| 20 | `N_CRIT` | n_crit |
| 21 | `XTR_UPPER` | Upper forced transition location |
| 22 | `XTR_LOWER` | Lower forced transition location |

For example, $\partial C_L / \partial \alpha$ is
`result.gradient.cl[input_index::ALPHA]`, and the sensitivity of the drag
coefficient to the third upper-surface weight is
`result.gradient.cd[input_index::UPPER_WEIGHTS + 2]`.

Gradients are computed by algorithmic differentiation through the network and
all output decodings, so they are exact to machine precision (the two clipped
transition outputs have zero gradient where the clip is active).

## Scalar-only analysis

Most design studies only need the six scalar outputs: `cl`, `cd`, `cm`,
`top_xtr`, `bot_xtr`, and `analysis_confidence`. For those, use the scalar
paths, which skip the 192 boundary layer outputs of the network:

| Call | Returns | Use when |
|------|---------|----------|
| `analyze_scalars()` | `AeroScalars` | Values only, no boundary layer |
| `analyze_scalars_with_gradients()` | `AeroScalarResult` (`output: AeroScalars`, `gradient: AeroScalarGradients`) | Gradients of the scalar outputs, e.g. optimization objectives and constraints built from coefficients |
| `analyze()` / `analyze_with_gradients()` | `AeroOutput` / `AeroResult` | You need the boundary layer distributions or their gradients |

The scalar types have the same field names as `AeroOutput` and
`AeroGradients`, so switching is usually a one-line change:

```rust
let result = model.analyze_scalars_with_gradients(&input);
let dcd_dalpha = result.gradient.cd[input_index::ALPHA];
```

`analyze_scalars_with_gradients()` differentiates the network in reverse mode,
with one adjoint per scalar output. `analyze_with_gradients()` uses forward
mode, with one seed per input, which suits its 198 outputs. For six outputs,
reverse mode is much cheaper. Measured on an Apple M1 Pro (µs per call):

| Model | `analyze` | `analyze_scalars` | `analyze_with_gradients` | `analyze_scalars_with_gradients` |
|-------|----------:|------------------:|-------------------------:|---------------------------------:|
| xxsmall | 4.9 | 2.2 | 57 | 7.0 |
| medium | 9.2 | 5.8 | 104 | 19 |
| xlarge | 27 | 21 | 321 | 71 |
| xxxlarge | 355 | 329 | 4473 | 1126 |

The scalar values are bitwise identical to those of `analyze()`. The scalar
gradients agree with `analyze_with_gradients()` up to rounding, because the same
sums are formed in a different order. Across the test suite's inputs, the
differences stay below 1e-13 of each gradient's largest entry. Run `cargo bench` to time every call on your
machine.

## Analysis confidence

The network reports a confidence in [0, 1] that decays with the (Mahalanobis)
distance of the query from the training data distribution. Treat results with
low confidence as extrapolation. In optimization, penalizing low confidence is
an effective way to keep the search inside the region where the surrogate is
trustworthy.

## Model sizes

All eight upstream network sizes are available via `bora::ModelSize`, from
`XXSmall` (3 layers, fastest) to `XXXLarge` (7 layers, 512 wide, most
accurate). `XLarge` is the upstream default (and `ModelSize::default()`) and a
good starting point. `ModelSize::ALL` lists every size; sizes convert to and
from their lower-case upstream names (`"xlarge"`) via `as_str()` / `Display`
and `FromStr` (parsing an unknown name yields `bora::Error::InvalidArgument`):

```rust
// Inside a function returning bora::Result<()>
let size: bora::ModelSize = "large".parse()?;
assert_eq!(size.to_string(), "large");
```

## Error handling

Fallible operations return `bora::Result<T>`, an alias for
`Result<T, bora::Error>`. This covers model construction
(`NeuralFoil::new`, `with_data_dir`, `from_parts`), weight loading
(`Network::load`, `InputDistribution::load`, and their `from_bytes`
counterparts), and the Kulfan utilities. `bora::Error` is an enum with a
descriptive `Display` message; its variants distinguish unreadable HDF5 files
(`Open`), malformed data (`Format`), an unresolvable weight directory
(`DataDirNotFound`), inconsistent dimensions (`DimensionMismatch`), invalid
arguments (`InvalidArgument`), and a singular Kulfan fit (`SingularFit`).

Analysis calls (`analyze`, `analyze_with_gradients`, and their `_scalars`
counterparts) are infallible and return
their results directly, so once a model is constructed no further error
handling is needed.
