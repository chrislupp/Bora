# Bora

[![Build and Tests](https://github.com/chrislupp/Bora/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/chrislupp/Bora/actions/workflows/build.yml)

A Rust port of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil), the
physics-informed neural network surrogate of XFoil, with **analytic gradients**
for gradient-based airfoil design optimization.

Given an airfoil shape in Kulfan (CST) parameterization and an operating
condition (alpha, Re, n_crit, forced transition), Bora predicts:

- Lift, drag, and moment coefficients (CL, CD, CM)
- Transition locations on both surfaces
- Boundary layer distributions (momentum thickness, shape factor, edge
  velocity) at 32 stations per surface
- An analysis confidence measure (distance from the training distribution)

Every output can be differentiated analytically with respect to all 23 inputs
(16 CST weights, leading edge weight, TE thickness, alpha, Re, n_crit, and
forced transition locations), making Bora suitable both as a standalone
optimization kernel and as a differentiable airfoil-analysis module inside
larger aero tools. Outputs match Python NeuralFoil 0.3.3 to floating-point
precision (verified by reference-value tests) and all eight upstream network
sizes (`xxsmall` ... `xxxlarge`) are supported.

## Quick Start

```rust
use bora::{AeroInput, ModelSize, NeuralFoil, input_index};

fn main() -> bora::Result<()> {
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.te_thickness = 0.0025;
    input.alpha = 3.0; // deg
    input.re = 1e6;

    let model = NeuralFoil::new(ModelSize::XLarge)?;

    // Analysis only:
    let aero = model.analyze(&input);

    // Analysis with gradients of every output w.r.t. all inputs:
    let result = model.analyze_with_gradients(&input);
    let dcl_dalpha = result.gradient.cl[input_index::ALPHA];
    Ok(())
}
```

Add Bora to your project with:

```toml
[dependencies]
bora = { git = "https://github.com/chrislupp/Bora" }
```

See `examples/example.rs` for basic usage and `examples/optimize_airfoil.rs`
for a full gradient-based drag minimization at fixed lift.

## Requirements

- Rust 1.89+ (install via [rustup](https://rustup.rs))

That's it: the HDF5 weight files are read with the pure-Rust
[`hdf5-pure`](https://crates.io/crates/hdf5-pure) crate, so no system HDF5
library is needed, and the network evaluation and its Jacobian are implemented
directly in the crate with no linear algebra or ML runtime dependencies.

## Building

```bash
# Build
cargo build --release

# Run tests (--all-features includes the Philote discipline)
cargo test --all-features

# Run examples
cargo run --release --example example
cargo run --release --example optimize_airfoil

# API documentation
cargo doc --open --all-features
```

For the best performance on x86-64, build with
`RUSTFLAGS="-C target-cpu=native"`, which enables the fused multiply-add
instructions used by the network kernels (always available on aarch64).

### Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `philote` | off | Philote discipline (`bora::philote::BoraDiscipline`) and the `bora_server` gRPC executable |

## Philote Discipline

With the `philote` feature, Bora exposes NeuralFoil function and analytic
gradient evaluations to any [Philote](https://github.com/MDO-Standards/Philote-MDO)
client (e.g. OpenMDAO via Philote-Python) over gRPC:

```bash
cargo run --release --features philote --bin bora_server [address]  # default localhost:50051
```

Options: `model_size` (str, default `"xlarge"`), `data_directory` (str), and
`include_boundary_layer` (bool, default `false`).

## Network Weights

The neural network weights live in `data/` as HDF5 files converted from the
upstream NeuralFoil `.npz` files (see `scripts/convert_weights.py`). At
runtime the weight directory is resolved in this order:

1. Explicit directory (`NeuralFoil::with_data_dir`)
2. `BORA_DATA_DIR` environment variable
3. The crate's source-tree `data/` directory (baked in at build time)

`Network::from_bytes` / `InputDistribution::from_bytes` are available if you
prefer to embed the `.h5` file images into your own binary with
`include_bytes!`.

## Attribution

The neural network weights and model architecture come from
[NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) by Peter Sharpe (MIT
license). If you use Bora in academic work, please cite NeuralFoil.
