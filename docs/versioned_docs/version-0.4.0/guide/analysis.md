---
sidebar_position: 2
title: "Inputs and outputs"
description: "Every field of AeroInput and AeroOutput, with units, defaults, and the ranges the model was trained on."
---

# Inputs and outputs

An analysis maps one `AeroInput` to one `AeroOutput`. This page lists every
field, its units and its sensible range.

## Inputs

```rust
pub struct AeroInput {
    pub kulfan: KulfanParameters,
    pub alpha: f64,
    pub re: f64,
    pub n_crit: f64,
    pub xtr_upper: f64,
    pub xtr_lower: f64,
}
```

| Field | Meaning | Units | Default |
|-------|---------|-------|--------:|
| `kulfan` | Airfoil shape, see [Airfoil geometry](./geometry.mdx) | chord lengths | all zero |
| `alpha` | Angle of attack | degrees | 0 |
| `re` | Reynolds number based on chord | none | 10⁶ |
| `n_crit` | Critical amplification factor of the $e^N$ transition model | none | 9 |
| `xtr_upper` | Forced transition location, upper surface | fraction of chord | 1 |
| `xtr_lower` | Forced transition location, lower surface | fraction of chord | 1 |

`AeroInput::default()` gives the defaults in the last column. The default
shape is a flat plate of zero thickness, which is not a useful airfoil, so
always set `kulfan`.

**Transition.** `n_crit` has the meaning it has in XFoil: 9 is a clean wind
tunnel or free flight, lower values model a more turbulent freestream and trip
the boundary layer earlier. A forced transition location of 1 means free
transition; any smaller value trips the boundary layer no later than that
station, as a transition strip would.

**Mach number.** `AeroInput` has no Mach input: the network is incompressible.
Mach number, post-stall behaviour and control surfaces are added by the
corrected analysis, see [Mach number, stall and flaps](./corrections.md).

### The range the model knows

NeuralFoil was trained on XFoil runs drawn from the distributions below
([Sharpe and Hansman, 2025](../references.md#neuralfoil-paper)).
Inside them the network interpolates; outside them it extrapolates, and the
[analysis confidence](./confidence.md) falls.

| Input | Training distribution |
|-------|-----------------------|
| Angle of attack | Middle 95% between −17° and +18°; extremes near ±28° |
| Reynolds number | Middle 95% between about 2 × 10³ and 3 × 10⁸; median 3 × 10⁵ |
| `n_crit` | Uniform between 0 and 18 |
| Forced transition | Free in 80% of cases; otherwise uniform between 0 and 1 |
| Shape | Blends of about 2,000 existing airfoils, with thickness scaling and random perturbation |

## Outputs

```rust
#[non_exhaustive]
pub struct AeroOutput {
    pub analysis_confidence: f64,
    pub cl: f64,
    pub cd: f64,
    pub cm: f64,
    pub top_xtr: f64,
    pub bot_xtr: f64,
    pub upper: BoundaryLayerData,
    pub lower: BoundaryLayerData,
}
```

| Field | Meaning | Range |
|-------|---------|-------|
| `cl` | Lift coefficient | |
| `cd` | Drag coefficient | Always positive |
| `cm` | Moment coefficient about the quarter chord, nose-up positive | |
| `top_xtr`, `bot_xtr` | Transition location on the upper and lower surface | 0 to 1; 1 means laminar to the trailing edge |
| `analysis_confidence` | The model's confidence in this result | 0 to 1 |
| `upper`, `lower` | Boundary-layer distributions | See below |

### Boundary layer

Each surface carries three distributions sampled at `NUM_BL_POINTS` = 32
stations:

```rust
#[non_exhaustive]
pub struct BoundaryLayerData {
    pub theta: [f64; 32],        // momentum thickness / chord
    pub h: [f64; 32],            // shape factor
    pub ue_over_vinf: [f64; 32], // edge velocity / freestream velocity
}
```

The stations are the midpoints of 32 equal intervals of chord, so the first is
at $x/c = 1/64$ and the last at $63/64$. `bl_x_points()` returns them:

```rust
use bora::{NUM_BL_POINTS, bl_x_points};

let aero = model.analyze(&input);
let x = bl_x_points();
for i in (0..NUM_BL_POINTS).step_by(8) {
    println!(
        "x/c = {:.3}  theta = {:.6}  H = {:.3}  ue/Vinf = {:.3}",
        x[i], aero.upper.theta[i], aero.upper.h[i], aero.upper.ue_over_vinf[i]
    );
}
```

For the quickstart airfoil at 3° and a Reynolds number of 10⁶ this prints

```text
x/c = 0.016  theta = 0.000077  H = 2.400  ue/Vinf = 1.315
x/c = 0.266  theta = 0.000312  H = 2.792  ue/Vinf = 1.471
x/c = 0.516  theta = 0.000541  H = 2.427  ue/Vinf = 1.340
x/c = 0.766  theta = 0.001387  H = 1.475  ue/Vinf = 1.179
```

The shape factor falls from laminar values above 2.4 to a turbulent value
near 1.5 between the third and fourth rows, just behind the reported
upper-surface transition at 49% of chord.

The edge velocity gives the surface pressure coefficient through Bernoulli's
relation for incompressible flow, $C_p = 1 - (u_e/V_\infty)^2$.

## The analysis calls

| Call | Returns | Cost relative to `analyze` (xlarge) |
|------|---------|------------------------------------:|
| `analyze(&input)` | `AeroOutput` | 1 |
| `analyze_scalars(&input)` | `AeroScalars` | 0.8 |
| `analyze_with_tangent(&input, &direction)` | `AeroTangent` | 2.0 |
| `analyze_scalars_with_gradients(&input)` | `AeroScalarResult` | 2.6 |
| `analyze_with_gradients(&input)` | `AeroResult` | 12 |

`AeroScalars` holds the six scalar fields of `AeroOutput` under the same
names, without the boundary layer. Its values are bitwise identical to those
from `analyze`. The derivative calls are described in [Gradients](./gradients.md),
and the batched forms of `analyze` and `analyze_scalars` under
[Many inputs at once](#many-inputs-at-once).

None of these calls can fail or mutate the model. Out-of-range inputs produce
a low confidence, not an error. One input is worth guarding in your own code:
`re` must be positive, because the model takes its logarithm.

## Many inputs at once

Python NeuralFoil evaluates arrays of operating points in one vectorized call.
Bora's equivalent is `analyze_batch`, which takes a slice of `AeroInput`s, any
mix of airfoils and operating points, and returns one `AeroOutput` per input,
in order:

```rust
use bora::{AeroInput, AeroOutput, BatchWorkspace, ModelSize, NeuralFoil};

let model = NeuralFoil::new(ModelSize::XLarge)?;
let inputs: Vec<AeroInput> = (0..64)
    .map(|i| AeroInput { alpha: -4.0 + 0.25 * i as f64, ..input })
    .collect();
let outputs = model.analyze_batch(&inputs);

// In a solver loop, keep the output buffer and the workspace between
// iterations so that repeated calls do not allocate.
let mut results = vec![AeroOutput::default(); inputs.len()];
let mut workspace = BatchWorkspace::default();
model.analyze_batch_into(&inputs, &mut results, &mut workspace);

// Scalar outputs only, as `AeroScalars`.
let scalars = model.analyze_scalars_batch(&inputs);
```

`analyze_scalars_batch_into` is the allocation-free form of the last call. A
`BatchWorkspace` grows on first use and can be shared between network sizes.

The batched calls push blocks of eight network evaluations (four inputs and
their mirror images) through each weight matrix together, instead of reading
the weights once per input. A batch of 64 is about twice as fast per input as
a loop over the single-input calls.

<TableCaption>
Time per input in microseconds for a batch of 64 (Apple M1 Pro, one thread).
</TableCaption>

| `ModelSize` | `analyze` | `analyze_batch` | Speedup | `analyze_scalars` | `analyze_scalars_batch` | Speedup |
|-------------|----------:|----------------:|--------:|------------------:|------------------------:|--------:|
| `XXSmall` | 4.9 | 2.5 | 1.9× | 2.2 | 1.1 | 2.1× |
| `XSmall` | 5.9 | 2.9 | 2.0× | 3.1 | 1.5 | 2.1× |
| `Small` | 7.6 | 3.8 | 2.0× | 4.3 | 2.0 | 2.1× |
| `Medium` | 9.2 | 4.4 | 2.1× | 5.9 | 2.6 | 2.3× |
| `Large` | 22.9 | 9.9 | 2.3× | 16.8 | 7.0 | 2.4× |
| `XLarge` | 27.2 | 11.8 | 2.3× | 21.6 | 8.9 | 2.4× |
| `XXLarge` | 84 | 34 | 2.5× | 72.5 | 29.0 | 2.5× |
| `XXXLarge` | 357 | 147 | 2.4× | 331 | 134 | 2.5× |

Small batches gain less: two inputs are about 1.1 to 1.2 times faster per
input, three about 1.5 to 1.9 times, and a single input costs the same as
`analyze`. Results agree with the single-input calls to within a few units in
the last place, because the batched kernel sums in a different order and uses
its own vectorizable exponential, and an input's result does not depend on its
position in the batch or on the other inputs. There are no batched gradient
calls.

## Symmetry

Mirroring an airfoil top to bottom and negating the angle of attack must
negate the lift and moment and leave the drag unchanged. Bora's results obey
this exactly, to the last bit, because of how the network is evaluated (see
[How NeuralFoil works](../background/how-it-works.mdx#symmetry)). A symmetric
section at zero angle of attack returns a lift and moment coefficient of
exactly zero and equal transition locations on both surfaces.
