---
sidebar_position: 5
title: "Choosing a network size"
sidebar_label: "Network sizes"
description: "The eight NeuralFoil network sizes: their accuracy against XFoil and their cost in Bora."
---

# Choosing a network size

NeuralFoil comes in eight sizes, all trained on the same data. They differ
only in the depth and width of the network, which trades accuracy for speed.
The Bora repository includes the weights of all eight, converted from
NeuralFoil's weight files.

```rust
use bora::{ModelSize, NeuralFoil};

let model = NeuralFoil::new(ModelSize::Large)?;
```

`ModelSize::default()` is `XLarge`, the default of NeuralFoil itself, where
its authors place the knee of the speed-accuracy curve
([Sharpe and Hansman, 2025](../references.md#neuralfoil-paper)).

## Accuracy

The table reproduces the accuracy NeuralFoil's authors measured for each
network against XFoil on a held-out test set of 395,665 cases that the
networks never saw in training. Bora evaluates the same networks and matches
Python NeuralFoil on reference cases, so these figures carry over unchanged.

<TableCaption>
Network sizes and mean absolute error against XFoil on NeuralFoil's test set.
Data from [Sharpe and Hansman (2025)](../references.md#neuralfoil-paper).
</TableCaption>

| `ModelSize` | Hidden layers | Width | $C_L$ | $\ln C_D$ | $C_M$ | $x_{tr}/c$ |
|-------------|--------------:|------:|------:|----------:|------:|-----------:|
| `XXSmall` | 2 | 48 | 0.040 | 0.078 | 0.007 | 0.044 |
| `XSmall` | 3 | 48 | 0.030 | 0.057 | 0.005 | 0.033 |
| `Small` | 3 | 64 | 0.027 | 0.050 | 0.005 | 0.027 |
| `Medium` | 4 | 64 | 0.020 | 0.039 | 0.003 | 0.020 |
| `Large` | 4 | 128 | 0.016 | 0.030 | 0.003 | 0.014 |
| `XLarge` | 5 | 128 | 0.013 | 0.024 | 0.002 | 0.010 |
| `XXLarge` | 5 | 256 | 0.012 | 0.022 | 0.002 | 0.009 |
| `XXXLarge` | 6 | 512 | 0.012 | 0.020 | 0.002 | 0.007 |

An error in $\ln C_D$ reads as a relative error in drag: 0.024 means the drag
is typically within about 2.4% of XFoil's.

Two cautions from the same source. The test set is deliberately hard, with
many post-stall and transition-sensitive cases; on ordinary attached-flow
cases the paper reports a mean drag error of 0.37% for the largest network.
And all of these are errors against XFoil, not against experiment.

## Speed

Time per call in Bora, in microseconds, measured with `cargo bench` on an
Apple M1 Pro. Run `cargo bench` to time your own machine.

<TableCaption>
Time per call in microseconds for each network size (Apple M1 Pro, one
thread). The columns are, in order, `analyze`, `analyze_scalars`,
`analyze_with_gradients` and `analyze_scalars_with_gradients`.
</TableCaption>

| `ModelSize` | All outputs | Scalars | All outputs with gradients | Scalars with gradients |
|-------------|------------:|--------:|---------------------------:|-----------------------:|
| `XXSmall` | 5.0 | 2.3 | 57 | 7.0 |
| `XSmall` | 6.0 | 3.1 | 66 | 9.8 |
| `Small` | 7.8 | 4.4 | 96 | 14 |
| `Medium` | 9.3 | 5.9 | 107 | 19 |
| `Large` | 23 | 17 | 267 | 57 |
| `XLarge` | 27 | 22 | 322 | 72 |
| `XXLarge` | 84 | 73 | 1,020 | 247 |
| `XXXLarge` | 356 | 331 | 4,480 | 1,130 |

Cost rises steeply with width, because the work is dominated by products
with width-by-width weight matrices: doubling the width of the five-layer
network from `XLarge` to `XXLarge` triples the time.

## Which one

**Start with `XLarge`.** It is within a small margin of the largest network's
accuracy at a thirteenth of its cost.

**Go smaller** when the analysis sits in a tight inner loop and a few percent
of drag accuracy is not the limiting error: early design-space exploration,
or sectional data inside a low-order wing method. `Medium` is three times
faster than `XLarge`.

**Go larger with care.** `XXLarge` and `XXXLarge` buy little accuracy against
XFoil for a large cost. NeuralFoil's documentation also observes that larger
networks follow XFoil's small-scale irregularities more closely, which can
make their outputs less smooth for a gradient-based optimizer, while being
better able to represent the odd shapes an optimizer passes through
([Sharpe, 2023](../references.md#neuralfoil-repository)). If an optimization
behaves poorly at one size, trying a neighboring size is a cheap experiment.

A model is immutable once loaded, so nothing stops you holding two: a small
one for exploration and a larger one to confirm the result.

## Names

Each size has a lower-case name matching NeuralFoil's (`"xlarge"`), used in
weight file names and in the Philote discipline's options. `as_str()`,
`Display` and `FromStr` convert between the two, and `ModelSize::ALL` lists
every size from smallest to largest:

```rust
let size: bora::ModelSize = "large".parse()?;
assert_eq!(size.to_string(), "large");

for size in bora::ModelSize::ALL {
    println!("{size}");
}
```

Parsing an unknown name returns `Error::InvalidArgument`.
