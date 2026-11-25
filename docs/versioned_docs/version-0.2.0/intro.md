---
sidebar_position: 1
slug: /intro
title: "Overview"
description: "What Bora is, what it computes, how it relates to NeuralFoil and XFoil, and what it leaves out."
---

# Overview

Bora is a Rust port of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil),
Peter Sharpe's neural-network surrogate of XFoil
([Sharpe and Hansman, 2025](./references.md#neuralfoil-paper)). You describe an
airfoil and an operating point; one function call returns its aerodynamic
coefficients, transition locations and boundary-layer state, and optionally the
exact derivative of each of those with respect to every input.

The model, its training data and its weights are Sharpe's work. Bora
reimplements the evaluation in Rust, adds hand-derived forward- and
reverse-mode derivatives, and checks its output against Python NeuralFoil 0.3.3
on committed reference cases.

## Why a Rust port

Bora was written for [Zephyr](https://github.com/chrislupp/Zephyr), a vortex-lattice code that needs sectional
airfoil data, with derivatives, at many spanwise stations in every iteration of
its viscous coupling. For most users NeuralFoil itself, in Python, will do the
job; reach for Bora when you need the analysis inside a Rust program, or need
its gradients without an automatic-differentiation framework.

## What one analysis returns

| Output | Field | Notes |
|--------|-------|-------|
| Lift coefficient | `cl` | |
| Drag coefficient | `cd` | |
| Moment coefficient | `cm` | About the quarter chord |
| Transition location, upper and lower | `top_xtr`, `bot_xtr` | Fraction of chord |
| Momentum thickness $\theta$ | `upper.theta`, `lower.theta` | 32 stations per surface, chord-normalized |
| Shape factor $H$ | `upper.h`, `lower.h` | 32 stations per surface |
| Edge velocity $u_e/V_\infty$ | `upper.ue_over_vinf`, `lower.ue_over_vinf` | 32 stations per surface |
| Analysis confidence | `analysis_confidence` | Between 0 and 1; see [Analysis confidence](./guide/confidence.md) |

The inputs are 18 shape parameters (a [Kulfan parameterization](./guide/geometry.mdx)),
the angle of attack, the Reynolds number, the critical amplification factor
$N_\mathrm{crit}$ and a forced transition location for each surface: 23 numbers
in all.

## Why use a surrogate of XFoil

XFoil is the reference tool for subsonic airfoil analysis, and within its
assumptions it is hard to beat for accuracy. It is less comfortable as a
component inside another program. The NeuralFoil documentation makes the case
in detail ([Sharpe, 2023](./references.md#neuralfoil-repository)); the points
that matter most for a Rust library are these.

**It always answers.** XFoil iterates and sometimes fails to converge. A Bora
analysis is a fixed sequence of matrix products, so it cannot fail, and it
costs the same every time: about 30 µs for the default network on an Apple M1
Pro, against tens of milliseconds for an XFoil run in the paper's measurements.
An outer loop that calls it thousands of times, such as the viscous coupling of
a vortex-lattice code, never has to handle a missing section polar.

**It is smooth.** XFoil's results are not continuously differentiable in the
inputs, chiefly because of how it treats the transition point, which makes
gradients of its outputs ragged at small scales
([Sharpe, 2023](./references.md#neuralfoil-repository)). NeuralFoil uses
smooth activation functions throughout, so its outputs are infinitely
differentiable, and Bora returns those derivatives analytically.

**It reports when to doubt it.** Each analysis carries a confidence that falls
as the query moves away from the conditions the network was trained on. An
optimizer can be kept inside the trustworthy region by constraining it.

## What Bora leaves out

Bora ports NeuralFoil's standalone model: the neural network, its input and
output encodings, the angle-of-attack symmetry and the confidence correction.
The NeuralFoil paper also describes features that are implemented in
[AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) around the network.
These are **not** part of Bora:

- **Compressibility.** The network models incompressible flow. There is no
  Mach number input and no transonic correction.
- **Post-stall model fusion.** AeroSandbox blends the network with an
  empirical model for massively separated flow, giving sensible results over
  all 360° of angle of attack. Bora returns the network's own output, which is
  only meaningful where the training data lies, roughly between −20° and +20°.
- **Control surface deflections.** Deflect the surface in your own geometry
  and fit Kulfan parameters to the result if you need this.

Two more limits follow from what the model is. Its accuracy is measured against
XFoil, not against experiment, so XFoil's modeling assumptions are a ceiling.
And the 18-parameter shape description cannot represent every airfoil: see
[Accuracy and limits](./background/accuracy-and-limits.md) before relying on
results for unusual shapes.

## Where to go next

- [Installation](./getting-started/installation.md) and the
  [Quickstart](./getting-started/quickstart.md) get a first analysis running.
- The guide covers [geometry](./guide/geometry.mdx),
  [inputs and outputs](./guide/analysis.md), [gradients](./guide/gradients.md),
  the [confidence output](./guide/confidence.md) and
  [choosing a network size](./guide/model-sizes.md).
- [How NeuralFoil works](./background/how-it-works.mdx) explains the model, with
  the exact encodings Bora implements.
- [References](./references.md) lists what to cite.
