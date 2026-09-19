---
sidebar_position: 1
title: "Installation"
---

# Installation

Bora is a Rust crate built with Cargo. The network weights are stored in HDF5
files that are read with the pure-Rust [`hdf5-pure`](https://crates.io/crates/hdf5-pure)
crate, so no system HDF5 library (or any other C dependency) is required.

## Prerequisites

| Dependency | Minimum Version | Notes |
|------------|----------------|-------|
| Rust toolchain | 1.89+ | Edition 2024; install via [rustup](https://rustup.rs) |

```bash
# Install or update the toolchain
rustup update stable
rustc --version
```

## Using Bora as a Dependency

Bora is not published on crates.io. Add it to your project's `Cargo.toml` as a
git dependency:

```toml
[dependencies]
bora = { git = "https://github.com/chrislupp/Bora" }
```

To also get the Philote discipline (`bora::philote`), enable the `philote`
feature:

```toml
[dependencies]
bora = { git = "https://github.com/chrislupp/Bora", features = ["philote"] }
```

## Building from Source

Clone the repository and build with Cargo:

```bash
git clone https://github.com/chrislupp/Bora
cd Bora

# Release build (recommended; the networks are slow in unoptimized builds)
cargo build --release
```

For a debug build:

```bash
cargo build
```

### Performance

For the fastest evaluation, compile for the host CPU. On x86-64 this enables
fused multiply-add (FMA) instructions in the network's inner loops; aarch64
targets already use them by default.

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `philote` | off | Philote discipline (`bora::philote::BoraDiscipline`) and the `bora_server` gRPC executable |

Build the discipline server with:

```bash
cargo build --release --features philote --bin bora_server
```

## Running Tests

The test suite consists of Cargo integration tests under `tests/`, including
validation against reference outputs and gradients stored in
`tests/data/*.h5`:

```bash
cargo test
```

To include the Philote discipline tests:

```bash
cargo test --all-features
```

To run a single test file or filter by test name:

```bash
cargo test --test neuralfoil
cargo test gradient
```

## Weight Data

The NeuralFoil weights ship in the repository's `data/` directory. At runtime,
`NeuralFoil` locates them in this order:

1. An explicit directory passed to `NeuralFoil::with_data_dir`.
2. The `BORA_DATA_DIR` environment variable.
3. The crate's source-tree `data/` directory (baked in at build time).

If you move the compiled binary away from the source tree, set `BORA_DATA_DIR`
or pass the directory explicitly.

## Verifying the Installation

A successful build, test run, and example run confirms the installation:

```bash
cargo build --release
cargo test
cargo run --release --example example
```

All tests should pass, and the example prints the aerodynamic coefficients of
a sample airfoil.
