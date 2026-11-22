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

### Directional derivatives

When only the derivative along one direction is needed, for example
$\partial C_L / \partial \alpha$ inside a Newton iteration on lift,
`analyze_with_tangent(&input, &direction)` is much cheaper than
`analyze_with_gradients()`. `direction` is a `bora::Gradient` in the same input
space as the table above, and the returned `bora::AeroTangent` holds the
`output` (bitwise identical to `analyze()`) and a `tangent` (`AeroOutput`)
whose every field is the directional derivative of the corresponding output,
`tangent.X = sum_k gradient.X[k] * direction[k]`:

```rust
let mut direction = [0.0; bora::NUM_INPUTS];
direction[input_index::ALPHA] = 1.0;
let result = model.analyze_with_tangent(&input, &direction);
let (cl, dcl_dalpha) = (result.output.cl, result.tangent.cl);
```

It propagates one direction through the network instead of all 23 (forward
mode, one tangent alongside each value), and covers the boundary layer outputs
too. Measured on an Apple M1 Pro (µs per call):

| Model | `analyze` | `analyze_with_tangent` | `analyze_with_gradients` |
|-------|----------:|-----------------------:|-------------------------:|
| xxsmall | 4.9 | 9.0 (1.8×) | 56 (11×) |
| medium | 9.2 | 17.6 (1.9×) | 104 (11×) |
| xlarge | 27 | 54 (2.0×) | 321 (12×) |
| xxxlarge | 358 | 701 (2.0×) | 4470 (12×) |

The tangent agrees with the contracted `analyze_with_gradients()` result up to
rounding. To differentiate the six scalar outputs with respect to all inputs,
`analyze_scalars_with_gradients()` (below) is cheaper than 23 tangents.

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

Analysis calls (`analyze`, `analyze_with_gradients`, `analyze_with_tangent`,
the batched calls, and their `_scalars` counterparts) are infallible and return
their results directly, so once a model is constructed no further error
handling is needed.

## Batch analysis

Upstream NeuralFoil evaluates arrays of operating points in one vectorized
call. The Bora equivalent is `NeuralFoil::analyze_batch`, which analyzes a
slice of `AeroInput`s (any mix of airfoils and operating conditions) and
returns one `AeroOutput` per input, in order:

```rust
use bora::{AeroInput, AeroOutput, BatchWorkspace, ModelSize, NeuralFoil};

let model = NeuralFoil::new(ModelSize::XLarge)?;
let inputs: Vec<AeroInput> = (0..64)
    .map(|i| AeroInput { alpha: -4.0 + 0.25 * i as f64, ..AeroInput::default() })
    .collect();
let outputs = model.analyze_batch(&inputs);

// In a solver loop: keep the output buffer and workspace across iterations.
let mut results = vec![AeroOutput::default(); inputs.len()];
let mut workspace = BatchWorkspace::default();
model.analyze_batch_into(&inputs, &mut results, &mut workspace);

// Scalar outputs only (no boundary layer), returning `AeroScalars`.
let scalars = model.analyze_scalars_batch(&inputs);
```

The batched path pushes blocks of `BATCH_BLOCK_SIZE` (8) network evaluations
(four points and their alpha-flipped images) through each weight matrix
together, with many independent multiply-add chains and a vectorized
activation function, instead of streaming the weights once per point. On
x86-64 it uses AVX2 and FMA when the CPU has them, like the single-point
calls. Measured on an Apple M1 Pro (single thread), per-point time for a
64-point batch:

| Model | `analyze` | `analyze_batch` | Speedup | `analyze_scalars` | `analyze_scalars_batch` | Speedup |
|-------|----------:|----------------:|--------:|------------------:|------------------------:|--------:|
| `XXSmall` | 4.9 µs | 2.5 µs | 1.9x | 2.2 µs | 1.1 µs | 2.1x |
| `XSmall` | 5.9 µs | 2.9 µs | 2.0x | 3.1 µs | 1.5 µs | 2.1x |
| `Small` | 7.6 µs | 3.8 µs | 2.0x | 4.3 µs | 2.0 µs | 2.1x |
| `Medium` | 9.2 µs | 4.4 µs | 2.1x | 5.9 µs | 2.6 µs | 2.3x |
| `Large` | 22.9 µs | 9.9 µs | 2.3x | 16.8 µs | 7.0 µs | 2.4x |
| `XLarge` | 27.2 µs | 11.8 µs | 2.3x | 21.6 µs | 8.9 µs | 2.4x |
| `XXLarge` | 84 µs | 34 µs | 2.5x | 72.5 µs | 29.0 µs | 2.5x |
| `XXXLarge` | 357 µs | 147 µs | 2.4x | 331 µs | 134 µs | 2.5x |

Batches of two points are about 1.1-1.2x faster per point, three points
1.5-1.9x; a single point costs about the same as `analyze`. Results agree with
`analyze` (and `analyze_scalars`) to within a few ulps of the network outputs
(the batched kernel sums in a different order and uses its own vectorizable
`exp`), and a point's result does not depend on its position in the batch or
on the other points.

The underlying kernel, `Network::evaluate_batch`, is public: it maps an
`input_size() x n_points` matrix to an `output_size() x n_points` matrix, both
row-major with one column per point.
