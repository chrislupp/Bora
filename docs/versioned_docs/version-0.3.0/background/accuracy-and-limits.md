---
sidebar_position: 2
title: "Accuracy and limits"
description: "How accurate Bora is, relative to what, and the conditions under which its results should not be used."
---

# Accuracy and limits

Bora's accuracy has two parts that are worth keeping separate: how closely it
reproduces NeuralFoil, and how closely NeuralFoil reproduces the physics.

## Bora against NeuralFoil

Bora evaluates the same networks with the same transformations as Python
NeuralFoil 0.3.3, and the same corrections as AeroSandbox 4.2.10. The test suite holds it to that.

| Check | Reference | Tolerance |
|-------|-----------|-----------|
| All 198 outputs, 64 cases across model sizes | Python NeuralFoil | 10⁻⁶ relative |
| Analytic Jacobians | CasADi derivatives of Python NeuralFoil | 10⁻⁶ relative |
| Analytic Jacobians | Central finite differences of Bora | 5 × 10⁻⁵ relative |
| Mirror symmetry of lift, moment and transition | Exact property | 10⁻¹² absolute |
| Corrected analysis, 88 cases from Mach 0 to 1.6 and around the circle | AeroSandbox 4.2.10 `get_aero_from_neuralfoil` | 10⁻⁶ relative |
| Gradients of the corrected analysis | Central finite differences of Bora | 10⁻⁴ relative |
| Analysis from rotated, scaled and shifted coordinates | Python NeuralFoil `get_aero_from_coordinates` | 10⁻⁶ relative |

The remaining differences are floating-point rounding. Bora sums in a
different order from NumPy and uses fused multiply-add instructions where the
processor has them, so the last few bits differ. For the same reason, results
can differ in the last bits between processors and between Bora versions.
These differences are many orders of magnitude below the model's error
against XFoil.

## NeuralFoil against XFoil

This is the error that matters in practice, and Bora inherits it unchanged.
NeuralFoil's authors measured it on 395,665 held-out cases
([Sharpe and Hansman, 2025](../references.md#neuralfoil-paper)); the full
table is in [Choosing a network size](../guide/model-sizes.md#accuracy). For
the default `XLarge` network the mean absolute errors are 0.013 in $C_L$,
about 2.4% in $C_D$, 0.002 in $C_M$ and 0.010 chord in transition location.

Those averages are taken over a test set weighted toward difficult cases.
On a set of ordinary NACA four-digit sections at 5° the paper reports a mean
drag error of 0.37% for the largest network. Expect errors between those two
figures depending on how benign the flow is, and larger ones where the
[analysis confidence](../guide/confidence.md) is low.

## XFoil against reality

A surrogate cannot be better than what it imitates. XFoil couples a panel
method to an integral boundary-layer formulation with an $e^N$ transition
model. It is very accurate for attached and mildly separated subsonic flow
and loses accuracy with large separation. NeuralFoil's documentation makes
the point that pushing the surrogate's error much below its present level
would not make its predictions of real airfoils any better, because the
remaining error would be XFoil's own
([Sharpe, 2023](../references.md#neuralfoil-repository)).

## Conditions to avoid

**Compressible flow, without the corrections.** `analyze` and its variants
have no Mach number input and represent incompressible flow. The
[corrected analysis](../guide/corrections.md) adds a Mach number.

**Transonic flow, as a prediction.** The compressibility corrections are
empirical fits layered on an incompressible analysis. They place the drag
rise roughly where it belongs and keep the model smooth through Mach 1, which
is what a conceptual-design optimizer needs. They do not compute a shock, so
do not use them to judge a transonic section.

**Deep stall and large angles.** The training data ends near ±28° and thins
well before that. Beyond about ±20° treat the network's output as
extrapolation; the confidence output will say so. The corrected analysis
replaces it there with an empirical post-stall model that is the same for
every section, which keeps a solver running but says little about your
airfoil.

**Deflected control surfaces.** The corrected analysis treats a deflection as
a change in angle of attack and a drag penalty. The pressure distribution,
transition and moment of a real flapped section are not modelled.

**Shapes the parameterization cannot represent.** Eighteen parameters describe
smooth, conventional sections well. They cannot reproduce sharp corners,
steps, gaps, multi-element sections or small surface features. If you fit
such a shape, Bora analyzes the smoothed fit. Check the fit as described in
[Airfoil geometry](../guide/geometry.mdx#from-coordinates-to-parameters).

There is a second side to this. Mark Drela, the author of XFoil, has observed
of NeuralFoil that the limited shape space is likely an advantage in
optimization: the small bumps that let an optimizer tune a separation bubble
at a single operating point cannot appear in it, which is the mechanism
behind the classic single-point optimization failure
([Sharpe, 2023](../references.md#neuralfoil-repository);
[Drela, 1998](../references.md#drela-1998)).

**Very low and very high Reynolds numbers.** The middle 95% of the training
data spans roughly 2 × 10³ to 3 × 10⁸, and half of it lies between 3 × 10⁴
and 3.5 × 10⁶. Results toward the edges deserve a look at the confidence.

**Unsteady or three-dimensional effects.** The model is a steady
two-dimensional section analysis. It has no notion of dynamic stall, sweep or
finite span.

## A practical checklist

1. Read `analysis_confidence` for every result you report.
2. If you fitted the shape from coordinates, compare the fit with the
   original.
3. For a design you intend to build, confirm the final section with XFoil or
   a higher-fidelity method. A surrogate is the right tool for the thousands
   of evaluations on the way to a design, not for the last one.
