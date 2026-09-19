---
sidebar_position: 1
title: "API Overview"
---

# API Overview

Bora is a Rust crate. The full public API is re-exported from the crate root,
so a single `use bora::{...}` covers everything. This page is an overview; for
the complete rustdoc reference (including the Philote discipline), run:

```bash
cargo doc --open --all-features
```

## Core Types

- **`NeuralFoil`** — The analysis model. Construct with
  `NeuralFoil::new(size)`, `NeuralFoil::with_data_dir(size, dir)`, or
  `NeuralFoil::from_parts(network, distribution)` (all return
  `bora::Result<NeuralFoil>`), then call `analyze(&input)` or
  `analyze_with_gradients(&input)`. `model_size()` reports the loaded size.
  Immutable and safe to share across threads after construction.
- **`Network`** / **`Layer`** — Low-level MLP container: loads HDF5 (`.h5`)
  weight files with `Network::load(path)` or `Network::from_bytes(data)` and
  evaluates the raw network (`evaluate`) with optional forward-mode Jacobian
  propagation (`evaluate_with_jacobian`).
- **`InputDistribution`** — Training-distribution statistics used for the
  Mahalanobis-based analysis confidence; `InputDistribution::load(path)` or
  `InputDistribution::from_bytes(data)`.

The `from_bytes` loaders make it possible to embed weights in the binary so no
data directory is needed at runtime. Copy the needed files from Bora's `data/`
directory (`nn-<size>.h5` and `scaled_input_distribution.h5`) into your
project; `include_bytes!` paths are relative to the source file:

```rust
use bora::{InputDistribution, NeuralFoil, Network};

fn embedded_model() -> bora::Result<NeuralFoil> {
    let network = Network::from_bytes(include_bytes!("../data/nn-xlarge.h5"))?;
    let distribution =
        InputDistribution::from_bytes(include_bytes!("../data/scaled_input_distribution.h5"))?;
    NeuralFoil::from_parts(network, distribution)
}
```

## Data Types

- **`KulfanParameters`** — CST shape description: `upper_weights`,
  `lower_weights` (`[f64; 8]` each), `leading_edge_weight`, `te_thickness`.
- **`AeroInput`** — Shape plus operating condition: `kulfan`, `alpha`, `re`,
  `n_crit`, `xtr_upper`, `xtr_lower`. `Default` gives `re = 1e6`,
  `n_crit = 9.0`, and free transition (`xtr_upper = xtr_lower = 1.0`).
- **`AeroOutput`** — `cl`, `cd`, `cm`, `top_xtr`, `bot_xtr`,
  `analysis_confidence`, and boundary layer distributions `upper` / `lower`.
- **`BoundaryLayerData`** — `theta`, `h`, `ue_over_vinf` at `NUM_BL_POINTS`
  stations.
- **`AeroGradients`** / **`BoundaryLayerGradients`** — Gradients of every
  output with respect to the 23 inputs, mirroring the output layout.
- **`AeroResult`** — Bundles `output: AeroOutput` and
  `gradient: AeroGradients`.
- **`Gradient`** — `[f64; NUM_INPUTS]`; indices named in `bora::input_index`
  (`UPPER_WEIGHTS`, `LOWER_WEIGHTS`, `LEADING_EDGE_WEIGHT`, `TE_THICKNESS`,
  `ALPHA`, `RE`, `N_CRIT`, `XTR_UPPER`, `XTR_LOWER`).
- **`ModelSize`** — `XXSmall`, `XSmall`, `Small`, `Medium`, `Large`, `XLarge`
  (default), `XXLarge`, `XXXLarge`. `ModelSize::ALL` lists all sizes;
  `as_str()`, `Display`, and `FromStr` convert to and from names like
  `"xlarge"`.
- **`Point2`** — `[f64; 2]` airfoil coordinate.

## Constants

- **`NUM_INPUTS`** (23) — Differentiable user-facing inputs.
- **`NUM_CST_WEIGHTS_PER_SIDE`** (8) — CST weights per surface.
- **`NUM_BL_POINTS`** (32) — Boundary layer stations per surface.
- **`NUM_LATENT_INPUTS`** (25), **`NUM_LATENT_OUTPUTS`** (198) — Network
  input and output dimensions.

## Utilities

- **`generate_kulfan_coordinates(&params, n_points_per_side)`** — CST
  parameters → Selig-format coordinates (`Result<Vec<Point2>>`).
- **`fit_kulfan_parameters(&coordinates)`** — Least-squares CST fit from
  Selig-format coordinates (`Result<KulfanParameters>`).
- **`bl_x_points()`** — Chordwise stations of the boundary layer outputs.
- **`version()`** — Crate version string.

## Errors

- **`Error`** — Enum of load and argument errors: `Open`, `Format`,
  `DataDirNotFound`, `DimensionMismatch`, `InvalidArgument`, `SingularFit`.
  Implements `std::error::Error`.
- **`Result<T>`** — Alias for `std::result::Result<T, bora::Error>`.

Constructors, loaders, and the Kulfan utilities return `Result`; analysis calls
are infallible.

## Philote Discipline

With the `philote` feature enabled, `bora::philote::BoraDiscipline` wraps
`NeuralFoil` as an explicit discipline for the Philote MDO standard (built on
the [`philote-mdo`](https://github.com/MDO-Standards/Philote-Rust) crate), so
any Philote client can evaluate airfoil aerodynamics and analytic partials over
gRPC. Variable names follow upstream NeuralFoil's capitalization (`CL`, `Re`,
...). Partials are row-major `[output_size x input_size]` blocks with singleton
dimensions dropped.

- **Inputs:** `upper_weights` \[8\], `lower_weights` \[8\],
  `leading_edge_weight`, `TE_thickness`, `alpha`, `Re`, `n_crit`,
  `xtr_upper`, `xtr_lower`.
- **Outputs:** `CL`, `CD`, `CM`, `Top_Xtr`, `Bot_Xtr`,
  `analysis_confidence`, and optionally the boundary layer distributions
  `upper_theta`, `upper_H`, `upper_ue_over_vinf`, `lower_theta`, `lower_H`,
  `lower_ue_over_vinf`.
- **Options:**

| Option | Type | Description |
|--------|------|-------------|
| `model_size` | str | Network size name, e.g. `"xlarge"` (default) |
| `data_directory` | str | Directory with the `.h5` weight files; empty uses the standard resolution order |
| `include_boundary_layer` | bool | Expose boundary layer outputs and their partials (default false) |

The `bora_server` binary serves the discipline:

```bash
# Default address is localhost:50051
cargo run --release --features philote --bin bora_server [address]
```
