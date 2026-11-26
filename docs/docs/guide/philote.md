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

## Run it in Docker

The server is also published as a Docker image,
[`chrislupp/bora`](https://hub.docker.com/r/chrislupp/bora), so you can use
it without a Rust toolchain:

<DockerRun />

The container serves on port 50051 like a local server, so clients connect
to `localhost:50051` unchanged. Stop it with `docker stop bora`.

<TableCaption>Tags of the Docker image.</TableCaption>

| Tag | Tracks | Updated |
|-----|--------|---------|
| `<version>`, for example `0.4.0` | That release | Never |
| `<major>.<minor>`, `<major>`, `latest` | The newest stable release in that series | On each stable release |
| `next` | The `develop` branch | On each push to `develop` whose image passes the smoke test |

Pin a version tag rather than `latest`, so your results do not change under
you. The command above uses the tag that matches the documentation you are
reading. `next` moves with `develop`, so pull it again before each run
(`docker pull chrislupp/bora:next`).

The image is built for `linux/amd64` and `linux/arm64`, so it runs natively on
Apple silicon as well as on x86-64, where the server picks its AVX2 and FMA
kernels at run time. It holds every network size, runs as a non-root user and
reads the weights from the directory named by `BORA_DATA_DIR`, so the
`data_directory` option is only needed for weights you mount into the
container yourself.

To build the image from a checkout:

```bash
docker build -f docker/Dockerfile -t bora .
docker run --rm -p 50051:50051 bora
```

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
| `control_surface_deflections` | n | deg | `deflection` of each control surface, with `control_surface_hinge_points` |

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
boundary layer does not depend on `mach`, so that pair is not declared. It
does depend on the deflections, which shift the angle of attack, so the
partial of `upper_theta` with respect to two deflections has shape 32 by 2.

## Options

<TableCaption>Options of the Philote discipline.</TableCaption>

| Option | Type | Default | Meaning |
|--------|------|---------|---------|
| `model_size` | str | `"xlarge"` | Network size name, see [Choosing a network size](./model-sizes.md#names) |
| `data_directory` | str | empty | Directory holding the weight files; empty uses the [standard search](./deployment.md#where-bora-looks) |
| `include_boundary_layer` | bool | `false` | Also expose the boundary-layer outputs and their partials |
| `include_corrections` | bool | `false` | Add the `mach` input and apply the [corrections](./corrections.md) to the coefficients and transition locations |
| `include_360_deg_effects` | bool | `true` | With `include_corrections`, blend into the post-stall model at large angles |
| `control_surface_hinge_points` | str | empty | With `include_corrections`, the hinge locations (x/c) of the [control surfaces](./corrections.md), separated by commas or spaces |

Options are applied when the client sets them, and the model is reloaded at
that point, so a bad size name, directory or hinge point fails the options
call rather than the first evaluation.

## Control surfaces

Each hinge point in `control_surface_hinge_points` adds a control surface to
the section, and the `control_surface_deflections` input carries one
deflection per surface, in degrees with the trailing edge down, in the order
of the hinge points. Control surfaces are part of the corrected analysis, so
the option requires `include_corrections`. The hinge points are fixed by the
option; the deflections are inputs with analytic partials, so they can be
design variables. In OpenMDAO, a plain flap at 75% chord and an aileron at 85%:

```python
comp = pmom.RemoteExplicitComponent(
    channel=channel,
    include_corrections=True,
    control_surface_hinge_points="0.75, 0.85",
)
```

The option is a string because OpenMDAO only passes Philote options of a
scalar type; a client that talks to the server directly may send a list of
numbers or a single number instead. An empty string removes the control
surfaces again. [Analysis over Philote](../examples/philote.mdx#step-6-hold-the-lift-with-a-flap)
works through an example: a flap that holds the lift at a fixed angle.

Leave `include_boundary_layer` off unless you need those outputs. Without it
the discipline computes values with `analyze_scalars` and partials with the
reverse-mode `analyze_scalars_with_gradients`, which is several times faster
than the full calls it must use otherwise (see [Gradients](./gradients.md)).
The same holds for `include_corrections`, which needs the edge velocities and
so always takes the full forward-mode path.

For a walk-through that calls the server from Python, both directly and from
an OpenMDAO model, see [Analysis over Philote](../examples/philote.mdx). For a
complete OpenMDAO model that drives the server with IPOPT, see
[Multipoint optimization](../examples/openmdao.mdx).

## Embedding the discipline

To serve the discipline from your own executable, or to combine it with
others, construct `bora::philote::BoraDiscipline` and hand it to
`philote-mdo`'s `ExplicitServer`. The `bora_server` source
(`src/bin/bora_server.rs`) is a complete example in about forty lines.
