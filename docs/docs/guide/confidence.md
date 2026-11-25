---
sidebar_position: 4
title: "Analysis confidence"
description: "What the analysis_confidence output measures, how to read it, and how to use it as an optimization constraint."
---

# Analysis confidence

Every analysis returns `analysis_confidence`, a number between 0 and 1. It is
the model's own estimate of whether the other outputs can be trusted. No
conventional airfoil code has an equivalent, and it is one of the more useful
ideas in NeuralFoil ([Sharpe and Hansman, 2025](../references.md#neuralfoil-paper)).

## What it measures

During training, each case was labelled by whether XFoil converged on it. The
network learned to predict that label, so the raw confidence is an estimate of
the probability that XFoil would have converged for this airfoil at this
operating point. Cases where XFoil fails are mostly those with large
separation or very delicate transition behavior, which are also the cases
where its answers, and a surrogate's, are least reliable.

A learned classifier is only meaningful near its training data. To make the
confidence fall to zero far from that data, NeuralFoil subtracts a term that
grows with the distance of the query from the training distribution before
converting to a probability. The details are in
[How NeuralFoil works](../background/how-it-works.mdx#confidence).

So a low confidence means one or both of:

- the flow is one XFoil itself would struggle with, or
- the inputs are unlike anything the network was trained on.

## Reading the number

The confidence is a continuous score, not a calibrated error bar. It does not
say how large the error is. As a working rule:

| Confidence | Reading |
|-----------:|---------|
| Above 0.9 | Ordinary attached-flow conditions; results are as accurate as the network gets |
| 0.5 to 0.9 | Sensitive or unusual conditions; cross-check anything that matters |
| Below 0.5 | Extrapolation; the numbers still vary smoothly but should not be believed |

These thresholds are a convention of this documentation, not a property of the
model. The quickstart airfoil illustrates the range, using NeuralFoil's xlarge network:

| Case | Confidence |
|------|-----------:|
| α = 3°, Re = 10⁶ | 0.964 |
| α = −8°, Re = 2 × 10⁵ | 0.363 |
| α = 60°, Re = 10⁶ | 0.000 |
| All upper weights set to 1.5 (not an airfoil) | 0.002 |

The second row is a cambered section at a negative angle and low Reynolds
number. The predicted drag coefficient there is 0.08, the mark of a largely
separated flow: a real condition, but a hard one. The last two rows are far
outside the training data.

## As an optimization constraint

A gradient-based optimizer will exploit any weakness in the model it is
given. Without a guard it can walk a design into a region where the surrogate
is confidently wrong, or to a shape that performs well at exactly the design
point and badly beside it, a long-known hazard of airfoil optimization
([Drela, 1998](../references.md#drela-1998)).

Constraining or penalizing the confidence keeps the search inside the region
the model knows. Because the confidence has an analytic gradient like any
other output, this costs nothing extra:

```rust
const CONFIDENCE_FLOOR: f64 = 0.95;
const PENALTY: f64 = 0.1;

let result = model.analyze_scalars_with_gradients(&input);
let deficit = (CONFIDENCE_FLOOR - result.output.analysis_confidence).max(0.0);

let objective = result.output.cd + PENALTY * deficit * deficit;
let gradient: [f64; bora::NUM_INPUTS] = std::array::from_fn(|k| {
    result.gradient.cd[k] - 2.0 * PENALTY * deficit * result.gradient.analysis_confidence[k]
});
```

The distance term also has a convenient shape for this purpose: it is a
positive-definite quadratic in the encoded inputs, which makes the region of
high confidence close to convex and well behaved as a constraint.

With a constrained optimizer, impose it directly instead:
[Multipoint optimization](../examples/openmdao.mdx) holds the
confidence at or above 0.90 at every operating point as an IPOPT constraint.
