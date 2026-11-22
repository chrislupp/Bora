# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Apache-2.0 license header with the public release statement (case number
  AFRL-2026-2087) and DoD disclaimers in every Rust source file and the README
- README citation section with BibTeX entries for the NeuralFoil paper and Peter Sharpe's
  PhD thesis

## [0.1.0] - 2026-09-19

### Added

- Rust port of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) 0.3.3: `bora::NeuralFoil`
  predicts CL, CD, CM, transition locations, analysis confidence, and boundary layer
  distributions (theta, H, ue/vinf at 32 stations per surface) from Kulfan (CST) parameters
  and operating conditions, for all eight upstream model sizes
- Analytic gradients of every output with respect to all 23 inputs (16 CST weights, leading
  edge weight, TE thickness, alpha, Re, n_crit, xtr_upper, xtr_lower) via forward-mode
  differentiation through the network, for gradient-based optimization
- Kulfan (CST) geometry utilities: coordinate generation and least-squares parameter fitting
  from coordinates (`bora::generate_kulfan_coordinates`, `bora::fit_kulfan_parameters`)
- HDF5 weight format (`data/nn-*.h5`, `data/scaled_input_distribution.h5`) with converter
  script (`scripts/convert_weights.py`); weight directory resolution via explicit path,
  `BORA_DATA_DIR`, or the source-tree `data/` directory; `Network::from_bytes` and
  `InputDistribution::from_bytes` for embedded weights
- Reference-value test suite (64 cases generated from Python NeuralFoil via
  `scripts/generate_output_ref_data.py`) and finite-difference verification of all gradients
- Reference-gradient test suite: analytic gradients verified directly against Python
  NeuralFoil jacobians computed with CasADi (`scripts/generate_gradient_ref_data.py`)
- Test reference data is committed as HDF5 files (`tests/data/output_ref_data.h5`,
  `tests/data/gradient_ref_data.h5`) loaded at test runtime via a shared loader
  (`tests/common/mod.rs`); regenerate only when updating the reference data
- Examples: basic analysis with gradients (`example`) and gradient-based airfoil drag
  optimization at fixed lift with Adam (`optimize_airfoil`)
- Philote discipline (`bora::philote::BoraDiscipline`, behind the `philote` Cargo feature):
  exposes NeuralFoil function and analytic gradient evaluations over gRPC via `philote-mdo`,
  with `model_size`, `data_directory`, and `include_boundary_layer` options, plus a standalone
  `bora_server` executable
- Docusaurus documentation site with custom theme, KaTeX math support, and versioned docs
- GitHub Actions workflow for automatic documentation deployment to GitHub Pages
- Docs versioning integrated into the release workflow
- Apache-2.0 license (`LICENSE`) and a `NOTICE` file reproducing the MIT license notices of
  NeuralFoil (ported code and network weights) and AeroSandbox (Kulfan geometry)

### Changed

- Ported the library from C++20/CMake to a Rust crate (edition 2024, MSRV 1.89) with an
  idiomatic API (`snake_case` fields, `Result`-returning constructors, `ModelSize` parsing
  via `FromStr`); outputs are unchanged and the network kernels are as fast as or faster than
  the C++ implementation
- HDF5 files are read with the pure-Rust `hdf5-pure` crate: the system HDF5 library is no
  longer required, and the Philote build no longer needs a system gRPC/protobuf install
- CI now checks `cargo fmt` and `cargo clippy`, builds and tests on Ubuntu and macOS with
  stable Rust and the MSRV, and runs the Philote tests; the release workflow bumps the
  version in `Cargo.toml`
- CI workflows use the current Node 24-based major versions of all GitHub Actions (dropping
  the deprecated Node 20 runtime and the `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24` workaround),
  and the documentation builds on Node 24 LTS
- Documentation site upgraded to Docusaurus 3.10.2 with the `@docusaurus/faster` (Rspack)
  bundler, fixing production builds broken by webpack 5.106+; `docs/package-lock.json` is now
  committed so the docs workflow's `npm ci` works

### Deprecated

### Removed

### Fixed

### Security

- Documentation site: npm `overrides` pin patched transitive dependencies of Docusaurus —
  `serialize-javascript` ^7.0.5 (GHSA-5c6j-r48x-rmvq, GHSA-qj8w-gfj5-8c6v) and `uuid`
  ^11.1.1 (GHSA-w5hq-g745-h8pq); `npm audit` reports no vulnerabilities
[Unreleased]: https://github.com/chrislupp/Bora/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/chrislupp/Bora/releases/tag/v0.1.0
