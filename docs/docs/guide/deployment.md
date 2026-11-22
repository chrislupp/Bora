---
sidebar_position: 6
title: "Weights and deployment"
description: "Where Bora finds the NeuralFoil network weights, how to embed them in a binary, and how to use a model from several threads."
---

# Weights and deployment

The NeuralFoil network weights are data files, not code. This page covers
where Bora looks for them, how to ship them with your program, and how to share a loaded
model between threads.

## The weight files

The repository's `data/` directory holds one file per network size and one
file of training-distribution statistics used by the
[confidence output](./confidence.md):

```text
data/
  nn-xxsmall.h5  ...  nn-xxxlarge.h5
  scaled_input_distribution.h5
```

They are HDF5 files converted from NeuralFoil's own weight files by
`scripts/convert_weights.py`. The weights themselves are unchanged; they are
the product of NeuralFoil's training and are distributed under its MIT license
(see `NOTICE` in the repository).

## Where Bora looks

`NeuralFoil::new(size)` resolves the directory in this order:

1. The `BORA_DATA_DIR` environment variable, if set.
2. The `data/` directory of the Bora source tree, whose path is recorded when
   the crate is compiled.

`NeuralFoil::with_data_dir(size, dir)` skips the search and uses the directory
you pass.

The second rule is what makes a plain git dependency work with no
configuration: the path points into Cargo's checkout. It also means a binary
copied to another machine will not find its weights, because that path does
not exist there. For anything you deploy, choose one of the following.

### Ship the directory

Copy `data/` alongside your executable and point Bora at it, either with the
environment variable or explicitly:

```rust
use bora::{ModelSize, NeuralFoil};

let model = NeuralFoil::with_data_dir(ModelSize::XLarge, "/opt/myapp/bora-data")?;
```

You only need the file for the size you use plus
`scaled_input_distribution.h5`.

### Embed the weights

`Network::from_bytes` and `InputDistribution::from_bytes` read the same files
from memory, so `include_bytes!` can compile them into the executable. Copy
the two files into your project; `include_bytes!` paths are relative to the
source file.

```rust
use bora::{InputDistribution, NeuralFoil, Network};

fn embedded_model() -> bora::Result<NeuralFoil> {
    let network = Network::from_bytes(include_bytes!("../data/nn-xlarge.h5"))?;
    let distribution =
        InputDistribution::from_bytes(include_bytes!("../data/scaled_input_distribution.h5"))?;
    NeuralFoil::from_parts(network, distribution)
}
```

The result is a single self-contained binary. A model built this way reports
`ModelSize::XLarge` from `model_size()` whatever network it was given, since
the size is not recorded in the file.

## Errors

Loading is the only fallible part of the API. Every loader returns
`bora::Result<T>`, and `bora::Error` says what went wrong:

| Variant | Cause |
|---------|-------|
| `Open` | An HDF5 file is missing or unreadable |
| `Format` | A file is missing a dataset or has the wrong layout |
| `DataDirNotFound` | No weight directory could be resolved |
| `DimensionMismatch` | The network or distribution does not have NeuralFoil's dimensions |
| `InvalidArgument` | An argument is out of range, such as an unknown size name |
| `SingularFit` | A Kulfan fit problem is degenerate |

`Error` implements `std::error::Error`, so it works with `?` in functions
returning `Box<dyn std::error::Error>` and with error-handling crates.

## Threads

A `NeuralFoil` is immutable after construction and is `Send` and `Sync`. Load
it once and share a reference; the analysis calls take `&self`.

Bora does not spawn threads itself. For many inputs at once, such as every
spanwise station of a wing or every candidate of a population-based optimizer,
first use the [batched calls](./analysis.md#many-inputs-at-once), which are
about twice as fast per input on one thread. For more, split the inputs across
threads yourself. With the standard library alone:

```rust
use bora::{AeroInput, AeroScalars, BatchWorkspace, NeuralFoil};

fn analyze_all(model: &NeuralFoil, inputs: &[AeroInput]) -> Vec<AeroScalars> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = inputs.len().div_ceil(threads).max(1);
    let mut results = vec![AeroScalars::default(); inputs.len()];

    std::thread::scope(|scope| {
        for (inputs, results) in inputs.chunks(chunk).zip(results.chunks_mut(chunk)) {
            scope.spawn(move || {
                let mut workspace = BatchWorkspace::default();
                model.analyze_scalars_batch_into(inputs, results, &mut workspace);
            });
        }
    });
    results
}
```

Each thread analyzes its share as one batch. A single analysis takes
microseconds, so threads only pay off for at least a few hundred inputs.
