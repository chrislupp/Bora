---
sidebar_position: 1
title: "API Overview"
---

# API Overview

Bora is a C++20 library. Everything lives in the `bora` namespace; include
`bora.h` for the full API.

## Core Classes

- **`NeuralFoil`** — The analysis model. Construct with a `ModelSize` (and
  optionally a weight directory), then call `Analyze()` or
  `AnalyzeWithGradients()`. Immutable and thread-safe after construction.
- **`Network`** — Low-level MLP container: loads HDF5 (`.h5`) weight files
  (from disk or memory) and evaluates the raw network with optional
  forward-mode Jacobian propagation.
- **`InputDistribution`** — Training-distribution statistics used for the
  Mahalanobis-based analysis confidence.

## Core Types

- **`KulfanParameters`** — CST shape description: 8 weights per surface,
  leading edge weight, TE thickness.
- **`AeroInput`** — Shape plus operating condition (alpha, Re, n_crit,
  xtr_upper, xtr_lower).
- **`AeroOutput`** — Coefficients, transition locations, confidence, and
  boundary layer distributions.
- **`AeroGradients` / `AeroResult`** — Gradients of every output with respect
  to the 23 inputs; `AeroResult` bundles output and gradients.
- **`Gradient`** — `std::array<double, kNumInputs>`; indices named in
  `bora::input_index`.
- **`ModelSize`** — `kXXSmall` … `kXXXLarge` network size selector.

## Utilities

- **`GenerateKulfanCoordinates()`** — CST parameters → Selig-format
  coordinates.
- **`FitKulfanParameters()`** — Least-squares CST fit from coordinates.
- **`BlXPoints()`** — Chordwise stations of the boundary layer outputs.
- **`ToString(ModelSize)`**, **`GetVersion()`** — Naming and version helpers.
