---
sidebar_position: 2
title: "Quickstart"
---

# Quickstart

This guide walks through a minimal Bora program: analyzing an airfoil and
reading off aerodynamic coefficients and their gradients.

## Add the Dependency

Bora is distributed from GitHub. In your project's `Cargo.toml`:

```toml
[dependencies]
bora = { git = "https://github.com/chrislupp/Bora" }
```

## Complete Example

```rust
use bora::{AeroInput, ModelSize, NeuralFoil, input_index};

fn main() -> bora::Result<()> {
    // Airfoil shape in Kulfan (CST) parameterization, 8 weights per side.
    let mut input = AeroInput::default();
    input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
    input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
    input.kulfan.te_thickness = 0.0025;

    // Operating condition.
    input.alpha = 3.0; // degrees
    input.re = 1e6;

    // Load the network (weights are found automatically; see below).
    let model = NeuralFoil::new(ModelSize::XLarge)?;

    // Analyze, with analytic gradients.
    let result = model.analyze_with_gradients(&input);

    println!(
        "CL = {:.4}, CD = {:.5}, CM = {:.4}",
        result.output.cl, result.output.cd, result.output.cm
    );
    println!(
        "dCL/dalpha = {:.4} per degree",
        result.gradient.cl[input_index::ALPHA]
    );
    Ok(())
}
```

## Step-by-Step Explanation

### 1. Describe the airfoil

Bora uses the Kulfan (CST) parameterization: 8 Bernstein weights per surface,
a leading-edge modification weight, and a trailing-edge thickness
(`KulfanParameters`). If you have coordinates instead, fit parameters with
`bora::fit_kulfan_parameters()`.

### 2. Set the operating condition

`AeroInput` also carries `alpha` (degrees), `re` (Reynolds number), `n_crit`,
and forced transition locations (`xtr_upper`, `xtr_lower`). Its `Default`
implementation gives `re = 1e6`, `n_crit = 9.0`, and `xtr_upper = xtr_lower =
1.0` (free transition), so you only need to set what differs.

### 3. Construct the model

`NeuralFoil::new` loads the network weights once; the model is immutable and
safe to share across threads afterwards. Model sizes range from
`ModelSize::XXSmall` (fastest) to `ModelSize::XXXLarge` (most accurate);
`ModelSize::XLarge` is the upstream default. The weight directory is taken from
the `BORA_DATA_DIR` environment variable or, failing that, the crate's
source-tree `data/` directory. Use `NeuralFoil::with_data_dir(size, dir)` to
pass a directory explicitly.

Loading returns a `bora::Result`, so a missing or malformed weight file is
reported as a `bora::Error` rather than a panic.

### 4. Analyze

Use `analyze()` for outputs only (`AeroOutput`), or `analyze_with_gradients()`
to also get the derivative of every output with respect to all 23 inputs
(`AeroResult`). Gradients are exact (algorithmic differentiation through the
network), not finite differences. Analysis calls cannot fail.

If you only need the scalar outputs (coefficients, transition locations, and
confidence), use `analyze_scalars()` and `analyze_scalars_with_gradients()`.
They return the same scalar values, and the gradient version is about 4-8x
faster than `analyze_with_gradients()`. See
[Scalar-only analysis](../guide/concepts.md#scalar-only-analysis).

### 5. Build and Run

```bash
cargo run --release
```

Always use `--release` for real work: unoptimized builds of the network
evaluation are much slower.

## Next Steps

- [Core Concepts](../guide/concepts.md) — inputs, outputs, and gradient layout
- [Basic Usage](../examples/basic-usage.md) — worked examples, including
  gradient-based optimization
