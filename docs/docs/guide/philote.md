---
sidebar_position: 7
title: "Philote discipline"
description: "Serve Bora to OpenMDAO and other MDO frameworks over gRPC as a Philote explicit discipline."
---

# Philote discipline

[Philote](https://github.com/MDO-Standards/Philote-MDO) is a standard for
calling analysis disciplines over gRPC, so that a framework written in one
language can use a discipline written in another. With the `philote` Cargo
feature, Bora provides an explicit discipline that serves both function values
and analytic partial derivatives. Any Philote client can then use Bora without
linking Rust code, for example an OpenMDAO model through Philote-Python.

## Run the server

```bash
cargo run --release --features philote --bin bora_server
```

The server listens on `localhost:50051`. Pass a different address as the only
argument:

```bash
cargo run --release --features philote --bin bora_server 0.0.0.0:50100
```

The build needs no system gRPC or protobuf installation; the
[`philote-mdo`](https://github.com/MDO-Standards/Philote-Rust) crate that Bora
builds on brings its own.

## Variables

Variable names keep NeuralFoil's capitalization (`CL`, `Re`, `Top_Xtr`), so a
model written against Python NeuralFoil's output names needs no translation.

<TableCaption>Inputs of the Philote discipline.</TableCaption>

| Input | Shape | Units | Corresponds to |
|-------|------:|-------|----------------|
| `upper_weights` | 8 | | `kulfan.upper_weights` |
| `lower_weights` | 8 | | `kulfan.lower_weights` |
| `leading_edge_weight` | 1 | | `kulfan.leading_edge_weight` |
| `TE_thickness` | 1 | | `kulfan.te_thickness` |
| `alpha` | 1 | deg | `alpha` |
| `Re` | 1 | | `re` |
| `n_crit` | 1 | | `n_crit` |
| `xtr_upper` | 1 | | `xtr_upper` |
| `xtr_lower` | 1 | | `xtr_lower` |
| `mach` | 1 | | `mach`, with `include_corrections` |

<TableCaption>Outputs of the Philote discipline.</TableCaption>

| Output | Shape | Present |
|--------|------:|---------|
| `CL`, `CD`, `CM` | 1 | Always |
| `Top_Xtr`, `Bot_Xtr` | 1 | Always |
| `analysis_confidence` | 1 | Always |
| `upper_theta`, `upper_H`, `upper_ue_over_vinf` | 32 | With `include_boundary_layer` |
| `lower_theta`, `lower_H`, `lower_ue_over_vinf` | 32 | With `include_boundary_layer` |
| `Cpmin`, `Cpmin_0`, `mach_crit`, `mach_dd` | 1 | With `include_corrections` |

Partials are declared for every output with respect to every input. Each is a
row-major block of output size by input size with singleton dimensions
dropped: the partial of `CL` with respect to `upper_weights` has shape 8, and
that of `upper_theta` with respect to `upper_weights` has shape 32 by 8. The
boundary layer does not depend on `mach`, so that pair is not declared.

## Options

<TableCaption>Options of the Philote discipline.</TableCaption>

| Option | Type | Default | Meaning |
|--------|------|---------|---------|
| `model_size` | str | `"xlarge"` | Network size name, see [Choosing a network size](./model-sizes.md#names) |
| `data_directory` | str | empty | Directory holding the weight files; empty uses the [standard search](./deployment.md#where-bora-looks) |
| `include_boundary_layer` | bool | `false` | Also expose the boundary-layer outputs and their partials |
| `include_corrections` | bool | `false` | Add the `mach` input and apply the [corrections](./corrections.md) to the coefficients and transition locations |
| `include_360_deg_effects` | bool | `true` | With `include_corrections`, blend into the post-stall model at large angles |

Options are applied when the client sets them, and the model is reloaded at
that point, so a bad size name or directory fails the options call rather than
the first evaluation.

Leave `include_boundary_layer` off unless you need those outputs. Without it
the discipline computes values with `analyze_scalars` and partials with the
reverse-mode `analyze_scalars_with_gradients`, which is several times faster
than the full calls it must use otherwise (see [Gradients](./gradients.md)).
The same holds for `include_corrections`, which needs the edge velocities and
so always takes the full forward-mode path. The discipline has no
control-surface inputs; use the Rust API for those.

For a complete OpenMDAO model that drives the server with IPOPT, see
[Multipoint optimization](../examples/openmdao.mdx).

## Embedding the discipline

To serve the discipline from your own executable, or to combine it with
others, construct `bora::philote::BoraDiscipline` and hand it to
`philote-mdo`'s `ExplicitServer`. The `bora_server` source
(`src/bin/bora_server.rs`) is a complete example in about forty lines.
