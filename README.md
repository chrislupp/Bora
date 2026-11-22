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

Bora is derived from [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil)
by Peter Sharpe: the network architecture, input/output processing, and
analysis model are ported from NeuralFoil, and the network weights in `data/`
are converted from its weight files. The Kulfan (CST) geometry utilities follow
[AeroSandbox](https://github.com/peterdsharpe/AeroSandbox), also by Peter
Sharpe. Both projects are MIT-licensed; their license notices are reproduced in
[NOTICE](NOTICE).

## Citation

If you use Bora in academic work, please cite the NeuralFoil paper

```bibtex
@misc{neuralfoil_paper,
  title = {{NeuralFoil}: An Airfoil Aerodynamics Analysis Tool Using Physics-Informed Machine Learning},
  author = {Peter Sharpe and R. John Hansman},
  year = {2025},
  eprint = {2503.16323},
  archivePrefix = {arXiv},
  primaryClass = {physics.flu-dyn},
  url = {https://arxiv.org/abs/2503.16323},
}
```

and [Peter Sharpe's PhD thesis](https://dspace.mit.edu/handle/1721.1/157809), whose
extended chapter on NeuralFoil is its primary long-form documentation:

```bibtex
@phdthesis{aerosandbox_phd_thesis,
   title = {Accelerating Practical Engineering Design Optimization with Computational Graph Transformations},
   author = {Sharpe, Peter D.},
   school = {Massachusetts Institute of Technology},
   year = {2024},
}
```

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)). See
[NOTICE](NOTICE) for third-party attributions.

## Public Release

This work has been cleared for public release, distribution unlimited, case
number: AFRL-2026-2087.

The views expressed are those of the authors and do not reflect the official
guidance or position of the United States Government, the Department of
Defense or of the United States Air Force.

Statement from DoD: The Appearance of external hyperlinks does not constitute
endorsement by the United States Department of Defense (DoD) of the linked
websites, of the information, products, or services contained therein. The DoD
does not exercise any editorial, security, or other control over the
information you may find at these locations.
