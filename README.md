# Bora

[![Build and Tests](https://github.com/clupp/Bora/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/clupp/Bora/actions/workflows/build.yml)

A C++20 port of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil), the
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
precision (verified by golden-value tests) and all eight upstream network
sizes (`xxsmall` ... `xxxlarge`) are supported.

## Quick Start

```cpp
#include "bora.h"

bora::AeroInput input;
input.kulfan.upper_weights = {0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32};
input.kulfan.lower_weights = {-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06};
input.kulfan.TE_thickness = 0.0025;
input.alpha = 3.0;   // deg
input.Re = 1e6;

const bora::NeuralFoil model(bora::ModelSize::kXLarge);

// Analysis only:
const bora::AeroOutput aero = model.Analyze(input);

// Analysis with gradients of every output w.r.t. all inputs:
const bora::AeroResult result = model.AnalyzeWithGradients(input);
double dCL_dalpha = result.gradient.CL[bora::input_index::kAlpha];
```

See `examples/example.cpp` for basic usage and
`examples/optimize_airfoil.cpp` for a full gradient-based drag minimization at
fixed lift.

## Requirements

- CMake 3.20+
- C++20 compatible compiler (e.g., GCC 12+, Clang 16+)
- GTest (for unit tests)

No external linear algebra or ML runtime dependencies; the network evaluation
and its Jacobian are implemented directly in the library.

## Building

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build

# Run tests
ctest --test-dir build

# Run examples
./build/examples/example
./build/examples/optimize_airfoil
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | `ON` | Enable/disable test building |
| `BUILD_EXAMPLES` | `ON` | Enable/disable example programs |
| `BORA_NATIVE_ARCH` | `ON` | Enable native architecture optimizations |

## Network Weights

The neural network weights live in `data/` in a simple binary format converted
from the upstream NeuralFoil `.npz` files (see `scripts/convert_weights.py`).
At runtime the weight directory is resolved in this order:

1. Explicit `data_dir` constructor argument
2. `BORA_DATA_DIR` environment variable
3. Installed data directory (`<prefix>/share/bora`)
4. Source-tree `data/` directory (for build-tree use)

`bora::Network::FromMemory` / `bora::InputDistribution::FromMemory` are
available if you prefer to embed the weight blobs into your own binary.

## Attribution

The neural network weights and model architecture come from
[NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) by Peter Sharpe (MIT
license). If you use Bora in academic work, please cite NeuralFoil.
