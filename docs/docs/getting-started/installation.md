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

The network's inner loops use fused multiply-add (FMA) instructions. aarch64
targets always have them. On x86-64, a default build checks once whether the
CPU supports AVX2 and FMA and, if so, runs the network with them (256-bit
vectors and FMA); older CPUs use the SSE2 baseline. No build flags are needed
for this.

Compiling for the host CPU builds the AVX2/FMA code in statically, which
skips the run-time check (and lets the rest of the crate use the host's
instruction set), at the cost of a binary that only runs on CPUs like the
build machine:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Results with and without FMA differ in the last bits, so the same input can
give slightly different outputs on different x86-64 machines.

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
