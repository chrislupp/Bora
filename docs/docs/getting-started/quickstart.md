---
sidebar_position: 2
title: "Quickstart"
---

# Quickstart

This guide walks through a minimal Bora program: analyzing an airfoil and
reading off aerodynamic coefficients and their gradients.

## Complete Example

```cpp
#include <bora.h>
#include <cstdio>

int main() {
    // Airfoil shape in Kulfan (CST) parameterization, 8 weights per side.
    bora::AeroInput input;
    input.kulfan.upper_weights = {0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32};
    input.kulfan.lower_weights = {-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06};
    input.kulfan.TE_thickness = 0.0025;

    // Operating condition.
    input.alpha = 3.0;  // degrees
    input.Re = 1e6;

    // Load the network (weights are found automatically; see below).
    const bora::NeuralFoil model(bora::ModelSize::kXLarge);

    // Analyze, with analytic gradients.
    const bora::AeroResult result = model.AnalyzeWithGradients(input);

    std::printf("CL = %.4f, CD = %.5f, CM = %.4f\n",
                result.output.CL, result.output.CD, result.output.CM);
    std::printf("dCL/dalpha = %.4f per degree\n",
                result.gradient.CL[bora::input_index::kAlpha]);
    return 0;
}
```

## Step-by-Step Explanation

### 1. Describe the airfoil

Bora uses the Kulfan (CST) parameterization: 8 Bernstein weights per surface,
a leading-edge modification weight, and a trailing-edge thickness. If you have
coordinates instead, fit parameters with `bora::FitKulfanParameters()`.

### 2. Set the operating condition

`bora::AeroInput` also carries alpha (degrees), Reynolds number, `n_crit`
(default 9.0), and forced transition locations (`xtr_upper`, `xtr_lower`,
default 1.0 = free transition).

### 3. Construct the model

`bora::NeuralFoil` loads the network weights once and is immutable and
thread-safe afterwards. Model sizes range from `kXXSmall` (fastest) to
`kXXXLarge` (most accurate); `kXLarge` is the upstream default. The weight
directory is resolved from the constructor argument, the `BORA_DATA_DIR`
environment variable, the install location, or the source tree, in that order.

### 4. Analyze

Use `Analyze()` for outputs only, or `AnalyzeWithGradients()` to also get the
derivative of every output with respect to all 23 inputs. Gradients are exact
(forward-mode differentiation through the network), not finite differences.

### 5. Build and Run

Add Bora to your CMakeLists.txt:

```cmake
find_package(Bora REQUIRED)
target_link_libraries(my_app PRIVATE Bora::Bora)
```

## Next Steps

- [Core Concepts](../guide/concepts.md) — inputs, outputs, and gradient layout
- [Basic Usage](../examples/basic-usage.md) — worked examples, including
  gradient-based optimization
