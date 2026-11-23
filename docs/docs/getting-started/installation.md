---
sidebar_position: 1
title: "Installation"
description: "Add Bora to a Cargo project, build it from source, and run its tests."
---

# Installation

Bora is a Rust crate built with Cargo. It needs a Rust toolchain and nothing
else: the NeuralFoil network weights ship as HDF5 files read with the pure-Rust
[`hdf5-pure`](https://crates.io/crates/hdf5-pure) crate, so there is no system
HDF5 library, no BLAS and no machine-learning runtime to install.

| Requirement | Version | Notes |
|-------------|---------|-------|
| Rust toolchain | 1.89 or newer | Edition 2024; install with [rustup](https://rustup.rs) |

```bash
rustup update stable
rustc --version
```

## Add Bora to a project

Bora is published on crates.io as `bora-foil` (the name `bora` belongs to an
unrelated crate). Add it to your `Cargo.toml`:

```toml
[dependencies]
bora-foil = "0.1"
```

The library keeps the name `bora`, so your code still writes `use bora::...`.

The [Philote discipline](../guide/philote.md) is behind a Cargo feature:

```toml
[dependencies]
bora-foil = { version = "0.1", features = ["philote"] }
```

| Feature | Default | Adds |
|---------|---------|------|
| `philote` | off | `bora::philote::BoraDiscipline` and the `bora_server` gRPC executable |

The weight files ship inside the crate, and Bora finds them in Cargo's
cache without any configuration. If you later ship a binary to another
machine, read [Weights and deployment](../guide/deployment.md).

## Build from source

```bash
git clone https://github.com/chrislupp/Bora
cd Bora
cargo build --release
```

Always measure and run real work with `--release`. The networks are several
times slower in an unoptimized build.

To check the build, run the test suite and the basic example:

```bash
cargo test --all-features
cargo run --release --example example
```

The tests compare Bora against outputs and Jacobians computed with Python
NeuralFoil (stored in `tests/data/*.h5`), and against finite differences.
`--all-features` includes the Philote discipline tests. To run one test file
or filter by name:

```bash
cargo test --test neuralfoil
cargo test --all-features -- matches_python
```

## Processor features

The network's inner loops use fused multiply-add (FMA) instructions.

- **aarch64** (Apple silicon, ARM servers) always has them.
- **x86-64**: a default build checks once per process whether the CPU has AVX2
  and FMA and uses them if so. Older CPUs fall back to the SSE2 baseline. No
  build flags are needed.

Compiling for the host CPU builds the AVX2 and FMA code in statically and skips
the run-time check, at the cost of a binary that only runs on similar CPUs:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Results computed with and without FMA differ in the last bits, so the same
input can give very slightly different outputs on different x86-64 machines.

## API documentation

The rustdoc reference for every public item, including the Philote discipline,
is built locally:

```bash
cargo doc --open --all-features
```
