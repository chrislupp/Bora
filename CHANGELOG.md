# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- C++20 port of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) 0.3.3: `bora::NeuralFoil`
  predicts CL, CD, CM, transition locations, analysis confidence, and boundary layer
  distributions (theta, H, ue/vinf at 32 stations per surface) from Kulfan (CST) parameters
  and operating conditions, for all eight upstream model sizes
- Analytic gradients of every output with respect to all 23 inputs (16 CST weights, leading
  edge weight, TE thickness, alpha, Re, n_crit, xtr_upper, xtr_lower) via forward-mode
  differentiation through the network, for gradient-based optimization
- Kulfan (CST) geometry utilities: coordinate generation and least-squares parameter fitting
  from coordinates (`bora::GenerateKulfanCoordinates`, `bora::FitKulfanParameters`)
- Binary weight format (`data/*.bnn`, `data/*.bid`) with converter script
  (`scripts/convert_weights.py`) and installed data files; weight directory resolution via
  explicit path, `BORA_DATA_DIR`, or baked-in install/source locations
- Reference-value test suite (64 cases generated from Python NeuralFoil via
  `scripts/generate_output_ref_data.py`) and finite-difference verification of all gradients
- Reference-gradient test suite: analytic gradients verified directly against Python
  NeuralFoil jacobians computed with CasADi (`scripts/generate_gradient_ref_data.py`)
- Test reference data is committed as HDF5 files (`test/data/output_ref_data.h5`,
  `test/data/gradient_ref_data.h5`) loaded at test runtime via a shared loader
  (`test/ref_data.h`); regenerate only when updating the reference data (tests
  require HDF5)
- Examples: basic analysis with gradients (`example`) and gradient-based airfoil drag
  optimization at fixed lift with Adam (`optimize_airfoil`)
- Philote discipline (`bora::BoraDiscipline`, gated behind `BORA_BUILD_PHILOTE`): exposes
  NeuralFoil function and analytic gradient evaluations over gRPC via Philote-Cpp, with
  `model_size`, `data_directory`, and `include_boundary_layer` options, plus a standalone
  `bora_server` executable
- Docusaurus documentation site with custom theme, KaTeX math support, and versioned docs
- GitHub Actions workflow for automatic documentation deployment to GitHub Pages
- Docs versioning integrated into the release workflow

### Changed

### Deprecated

### Removed

### Fixed

### Security
