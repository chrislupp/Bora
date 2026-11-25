---
sidebar_position: 1
title: "API overview"
description: "A map of Bora's public API: the analysis model, data types, constants, utilities and errors."
---

# API overview

Everything public is re-exported from the crate root, so one
`use bora::{...}` reaches all of it. This page is a map. The complete
reference, generated from the source, is the rustdoc:

```bash
cargo doc --open --all-features
```

## The analysis model

### `NeuralFoil`

The model. Construct it once; every analysis call takes `&self`, cannot fail
and is safe to call from several threads.

| Item | Signature | Notes |
|------|-----------|-------|
| `new` | `(size: ModelSize) -> Result<NeuralFoil>` | Finds the NeuralFoil weights through `BORA_DATA_DIR` or the source tree |
| `with_data_dir` | `(size, dir) -> Result<NeuralFoil>` | Loads from an explicit directory |
| `from_parts` | `(Network, InputDistribution) -> Result<NeuralFoil>` | From pre-loaded or embedded weights |
| `analyze` | `(&AeroInput) -> AeroOutput` | All outputs |
| `analyze_with_gradients` | `(&AeroInput) -> AeroResult` | All outputs and their gradients, forward mode |
| `analyze_scalars` | `(&AeroInput) -> AeroScalars` | The six scalar outputs |
| `analyze_scalars_with_gradients` | `(&AeroInput) -> AeroScalarResult` | The six scalars and their gradients, reverse mode |
| `analyze_with_tangent` | `(&AeroInput, &Gradient) -> AeroTangent` | All outputs and their derivative along one direction |
| `analyze_batch` | `(&[AeroInput]) -> Vec<AeroOutput>` | Many inputs at once, all outputs |
| `analyze_batch_into` | `(&[AeroInput], &mut [AeroOutput], &mut BatchWorkspace)` | The same, without allocating |
| `analyze_scalars_batch` | `(&[AeroInput]) -> Vec<AeroScalars>` | Many inputs at once, the six scalars |
| `analyze_scalars_batch_into` | `(&[AeroInput], &mut [AeroScalars], &mut BatchWorkspace)` | The same, without allocating |
| `analyze_corrected` | `(&CorrectedInput) -> CorrectedOutput` | With Mach number, post-stall and control-surface corrections |
| `analyze_corrected_with_gradients` | `(&CorrectedInput) -> CorrectedResult` | The same, with gradients including Mach and deflections |
| `analyze_corrected_batch` | `(&[CorrectedInput]) -> Vec<CorrectedOutput>` | Many corrected inputs at once |
| `analyze_airfoil` | `(&FittedAirfoil, &OperatingPoint) -> AeroOutput` | An airfoil given as coordinates, in its own frame |
| `model_size` | `() -> ModelSize` | The size the model was constructed with |

The `_into` forms panic if the output slice and the input slice differ in
length. See [Inputs and outputs](../guide/analysis.md),
[Gradients](../guide/gradients.md),
[Mach number, stall and flaps](../guide/corrections.md) and
[Weights and deployment](../guide/deployment.md).

### `ModelSize`

`XXSmall`, `XSmall`, `Small`, `Medium`, `Large`, `XLarge` (the default),
`XXLarge`, `XXXLarge`. `ModelSize::ALL` lists them in order; `as_str()`,
`Display` and `FromStr` convert to and from names such as `"xlarge"`. See
[Choosing a network size](../guide/model-sizes.md).

## Data types

| Type | Contents |
|------|----------|
| `KulfanParameters` | `upper_weights`, `lower_weights` (`[f64; 8]` each), `leading_edge_weight`, `te_thickness`; `max_thickness()` |
| `AeroInput` | `kulfan`, `alpha`, `re`, `n_crit`, `xtr_upper`, `xtr_lower` |
| `AeroOutput` | `cl`, `cd`, `cm`, `top_xtr`, `bot_xtr`, `analysis_confidence`, and `upper`, `lower` boundary layers |
| `BoundaryLayerData` | `theta`, `h`, `ue_over_vinf`, each `[f64; NUM_BL_POINTS]` |
| `AeroScalars` | The six scalar fields of `AeroOutput` |
| `Gradient` | `[f64; NUM_INPUTS]`, indexed by the constants in `input_index` |
| `AeroGradients` | A `Gradient` for every field of `AeroOutput`, same layout |
| `BoundaryLayerGradients` | A `Gradient` for every boundary-layer value |
| `AeroScalarGradients` | A `Gradient` for each of the six scalars |
| `AeroResult` | `output: AeroOutput`, `gradient: AeroGradients` |
| `AeroScalarResult` | `output: AeroScalars`, `gradient: AeroScalarGradients` |
| `AeroTangent` | `output: AeroOutput`, `tangent: AeroOutput`, the directional derivative of every field |
| `OperatingPoint` | The fields of `AeroInput` without the shape: `alpha`, `re`, `n_crit`, `xtr_upper`, `xtr_lower` |
| `FittedAirfoil` | `kulfan` of the normalized airfoil, and the `x_translation`, `y_translation`, `scale_factor`, `rotation_angle` that normalized it |
| `NormalizedCoordinates` | `coordinates` and the same four transform values |
| `ControlSurface` | `deflection` (degrees, trailing edge down), `hinge_point` |
| `CorrectedInput` | `aero: AeroInput`, `mach`, `control_surfaces`, `include_360_deg_effects` |
| `CorrectedOutput` | The fields of `AeroOutput`, and `cpmin`, `cpmin_0`, `mach_crit`, `mach_dd` |
| `CorrectedGradient` | `[f64; NUM_CORRECTED_INPUTS]`: a `Gradient` followed by the Mach number |
| `CorrectedGradients` | A `CorrectedGradient` for every scalar of `CorrectedOutput`, the boundary-layer gradients, and `deflection`, one `CorrectedOutput` of derivatives per control surface |
| `CorrectedResult` | `output: CorrectedOutput`, `gradient: CorrectedGradients` |
| `BatchWorkspace` | Reusable scratch space for the batched calls; grows on first use |
| `Point2` | `[f64; 2]`, an airfoil coordinate |

The `input_index` module names the entries of a `Gradient`: `UPPER_WEIGHTS`,
`LOWER_WEIGHTS`, `LEADING_EDGE_WEIGHT`, `TE_THICKNESS`, `ALPHA`, `RE`,
`N_CRIT`, `XTR_UPPER`, `XTR_LOWER`. The first two are the start of
eight-entry blocks. `MACH` is the last entry of a `CorrectedGradient` only.

## Constants

| Constant | Value | Meaning |
|----------|------:|---------|
| `NUM_INPUTS` | 23 | Differentiable user inputs |
| `NUM_CORRECTED_INPUTS` | 24 | The same, and the Mach number |
| `NUM_CST_WEIGHTS_PER_SIDE` | 8 | Kulfan weights per surface |
| `NUM_BL_POINTS` | 32 | Boundary-layer stations per surface |
| `NUM_LATENT_INPUTS` | 25 | Network input dimension |
| `NUM_LATENT_OUTPUTS` | 198 | Network output dimension |
| `BATCH_BLOCK_SIZE` | 8 | Network evaluations the batched kernel processes together |

## Utilities

| Function | Signature | Purpose |
|----------|-----------|---------|
| `parse_dat` | `(&str) -> Result<Vec<Point2>>` | Points from the text of a coordinate file |
| `read_dat_file` | `(path) -> Result<Vec<Point2>>` | Points from a coordinate file |
| `normalize_coordinates` | `(&[Point2]) -> Result<NormalizedCoordinates>` | Leading edge to the origin, unit chord on the x axis |
| `FittedAirfoil::from_coordinates`, `from_dat_file` | `-> Result<FittedAirfoil>` | Normalize and fit in one step |
| `fit_kulfan_parameters` | `(&[Point2]) -> Result<KulfanParameters>` | Least-squares fit to normalized Selig-format coordinates |
| `generate_kulfan_coordinates` | `(&KulfanParameters, n_points_per_side) -> Result<Vec<Point2>>` | Selig-format coordinates from parameters |
| `bl_x_points` | `() -> [f64; NUM_BL_POINTS]` | Chordwise stations of the boundary-layer outputs |
| `version` | `() -> &'static str` | The crate version |

See [Airfoil geometry](../guide/geometry.mdx).

## Errors

`bora::Result<T>` is `Result<T, bora::Error>`. Constructors, loaders and the
geometry utilities return it; analysis calls do not. The variants of `Error`
are `Open`, `Io`, `Format`, `DataDirNotFound`, `DimensionMismatch`,
`InvalidArgument` and `SingularFit`, described in
[Weights and deployment](../guide/deployment.md#errors).

## Lower-level access

Most programs need nothing below `NeuralFoil`. The pieces it is built from are
public for programs that want to embed weights or evaluate the raw network.

### `Network`

A fully connected network with Swish activations.

| Item | Purpose |
|------|---------|
| `load(path)`, `from_bytes(data)` | Read a weight file from disk or memory |
| `evaluate(x, y)` | Outputs for one input |
| `evaluate_pair([x0, x1], [y0, y1])` | Two inputs in one pass over the weights; bitwise identical to two `evaluate` calls |
| `evaluate_with_jacobian(x, jx, n_seed, y, jy)` | Outputs with a forward-mode Jacobian for `n_seed` seed directions |
| `evaluate_with_tangent(x, dx, y, dy)` | Outputs with the tangent along one direction |
| `evaluate_batch(x, y, n_points, workspace)` | Many inputs at once; `x` is `input_size()` by `n_points` and `y` is `output_size()` by `n_points`, row-major with one column per point |
| `evaluate_first(x, n, y)` | Only the first `n` outputs |
| `evaluate_first_with_gradients(x, n, y, grad)` | The first `n` outputs and their reverse-mode gradients, row-major `n` by `input_size()` |
| `input_size()`, `output_size()`, `layers()` | Dimensions and the `Layer` list |

### `InputDistribution`

The mean and inverse covariance of NeuralFoil's training inputs, used for the
confidence correction. `load(path)` and `from_bytes(data)` read it;
`squared_mahalanobis(x)` and `squared_mahalanobis_gradient(x, grad)` evaluate
the distance and its gradient; `size()` is its dimension.

## Philote discipline

With the `philote` feature, `bora::philote::BoraDiscipline` wraps the model as
an explicit discipline for the Philote MDO standard, and the `bora_server`
binary serves it over gRPC. See [Philote discipline](../guide/philote.md).
