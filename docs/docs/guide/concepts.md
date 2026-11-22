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

Gradients are computed by forward-mode differentiation through the network and
all output decodings, so they are exact to machine precision (the two clipped
transition outputs have zero gradient where the clip is active).

### Directional derivatives

When only the derivative along one direction is needed, for example
$\partial C_L / \partial \alpha$ inside a Newton iteration on lift,
`analyze_with_tangent(&input, &direction)` is much cheaper than
`analyze_with_gradients()`. `direction` is a `bora::Gradient` in the same input
space as the table above, and the returned `bora::AeroTangent` holds the
`output` (identical to `analyze()`) and a `tangent` (`AeroOutput`) whose every
field is the directional derivative of the corresponding output,
`tangent.X = sum_k gradient.X[k] * direction[k]`:

```rust
let mut direction = [0.0; bora::NUM_INPUTS];
direction[input_index::ALPHA] = 1.0;
let result = model.analyze_with_tangent(&input, &direction);
let (cl, dcl_dalpha) = (result.output.cl, result.tangent.cl);
```

It propagates one direction through the network instead of all 23, costing
about 1.1–1.7× an `analyze()` call versus roughly 13× for
`analyze_with_gradients()` (measured on an Apple M1 Pro; the ratio is smallest
for the larger models).

### Gradients of a few scalar outputs

When only some scalar outputs need gradients — for example lift and drag in a
vortex-lattice or strip-theory loop — use `analyze_with_scalar_gradients()`.
It takes the outputs to differentiate as a slice of `bora::ScalarOutput`
(`AnalysisConfidence`, `Cl`, `Cd`, `Cm`, `TopXtr`, `BotXtr`) and returns a
`bora::ScalarGradients` with the full `AeroOutput` and one `Gradient` per
requested output, in the order requested:

```rust
use bora::{ScalarOutput, input_index};

let result = model.analyze_with_scalar_gradients(&input, &[ScalarOutput::Cl, ScalarOutput::Cd]);
let (dcl, dcd) = (&result.gradients[0], &result.gradients[1]);
let dcl_dalpha = dcl[input_index::ALPHA];
```

These gradients are computed by reverse-mode differentiation: one network
evaluation plus one backward sweep per requested output, instead of
propagating all 23 input directions through all 198 latent outputs. They equal
the corresponding `analyze_with_gradients()` fields to rounding. Measured on an
Apple M1 Pro, `[Cl, Cd]` costs about 1.4–1.7× an `analyze()` call and all six
scalars about 2–2.6×, versus 13–18× for `analyze_with_gradients()`.

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
`analyze_with_scalar_gradients`) are infallible and return
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
```

The batched path pushes blocks of `BATCH_BLOCK_SIZE` (8) network evaluations
through each weight matrix together, with many independent fused multiply-add
chains and a vectorized activation function, instead of streaming the weights
once per point. Measured on an Apple M1 Pro (single thread), per-point time of
`analyze_batch` versus a loop over `analyze`, for batches of 8 or more points:

| Model | `analyze` loop | `analyze_batch` | Speedup |
|-------|----------------|-----------------|---------|
| `XXSmall` | 6.6 µs | 2.6 µs | 2.6x |
| `XSmall` | 8.8 µs | 2.9 µs | 3.0x |
| `Small` | 13.3 µs | 3.8 µs | 3.5x |
| `Medium` | 16.6 µs | 4.4 µs | 3.8x |
| `Large` | 45.6 µs | 10.0 µs | 4.6x |
| `XLarge` | 57.3 µs | 12.0 µs | 4.8x |
| `XXLarge` | 183 µs | 34.5 µs | 5.3x |
| `XXXLarge` | 860 µs | 147 µs | 5.8x |

Batches of two or three points are about 1.5–4x faster per point; a single
point costs the same as `analyze`. Results agree with `analyze` to within a
few ulps of the network outputs (the batched kernel sums in a different order
and uses its own vectorizable `exp`), and a point's result does not depend on
its position in the batch or on the other points.

The underlying kernel, `Network::evaluate_batch`, is public: it maps an
`input_size() x n_points` matrix to an `output_size() x n_points` matrix, both
row-major with one column per point.
