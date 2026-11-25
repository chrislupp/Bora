"""Kulfan (CST) geometry for the OpenMDAO example.

The parameterization matches Bora's `generate_kulfan_coordinates` and
AeroSandbox's `KulfanAirfoil`: class function x^0.5 (1 - x), an order-7
Bernstein shape function per surface, a linear trailing-edge thickness term and
the leading-edge modification mode.
"""

from math import comb

import numpy as np
import openmdao.api as om

NUM_WEIGHTS = 8
DEG = 180.0 / np.pi


def bernstein_basis(x):
    """Order-7 Bernstein basis at stations x, shape (len(x), 8)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = NUM_WEIGHTS - 1
    return np.stack(
        [comb(n, k) * x**k * (1.0 - x) ** (n - k) for k in range(NUM_WEIGHTS)], axis=-1
    )


def class_function(x):
    return np.sqrt(x) * (1.0 - x)


def leading_edge_mode(x):
    return x * np.maximum(1.0 - x, 0.0) ** (NUM_WEIGHTS + 0.5)


def surface(x, weights, leading_edge_weight, te_thickness, te_sign):
    """y/c of one surface at stations x (te_sign is +1 upper, -1 lower)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return (
        class_function(x) * (bernstein_basis(x) @ weights)
        + te_sign * x * te_thickness / 2.0
        + leading_edge_weight * leading_edge_mode(x)
    )


def selig_coordinates(upper_weights, lower_weights, leading_edge_weight, te_thickness, n=101):
    """Selig-format coordinates (upper TE -> LE -> lower TE) with cosine spacing."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    yu = surface(x, upper_weights, leading_edge_weight, te_thickness, 1.0)
    yl = surface(x, lower_weights, leading_edge_weight, te_thickness, -1.0)
    upper = np.column_stack([x[::-1], yu[::-1]])
    lower = np.column_stack([x[1:], yl[1:]])
    return np.vstack([upper, lower])


def wiggliness(upper_weights, lower_weights):
    """Sum of squared second differences of both weight vectors."""
    return np.sum(np.diff(upper_weights, 2) ** 2) + np.sum(np.diff(lower_weights, 2) ** 2)


class KulfanGeometry(om.ExplicitComponent):
    """Geometric quantities the optimization constrains.

    Outputs
    -------
    thickness : t/c at the `stations` option. Upper minus lower surface, so the
        leading-edge mode cancels and t/c is linear in the weights.
    te_angle : trailing-edge wedge angle in degrees, defined as in AeroSandbox's
        `KulfanAirfoil.TE_angle()`. At x = 1 the class function has slope -1 and
        only the last Bernstein mode is nonzero, so each surface's slope there
        is set by its last weight.
    wiggliness : sum of squared second differences of the weights, a discrete
        measure of the shape's curvature variation.
    """

    def initialize(self):
        self.options.declare("stations", types=(list, tuple, np.ndarray))

    def setup(self):
        x = np.asarray(self.options["stations"], dtype=float)
        self._x = x
        # t(x) = C(x) B(x) . (w_u - w_l) + x * TE_thickness
        self._rows = class_function(x)[:, None] * bernstein_basis(x)

        self.add_input("upper_weights", shape=NUM_WEIGHTS)
        self.add_input("lower_weights", shape=NUM_WEIGHTS)
        self.add_input("TE_thickness", val=0.0)
        self.add_output("thickness", shape=len(x))
        self.add_output("te_angle", val=0.0, units="deg")
        self.add_output("wiggliness", val=0.0)

        last = [NUM_WEIGHTS - 1]
        self.declare_partials("thickness", "upper_weights", val=self._rows)
        self.declare_partials("thickness", "lower_weights", val=-self._rows)
        self.declare_partials("thickness", "TE_thickness", val=x)
        self.declare_partials("te_angle", "upper_weights", rows=[0], cols=last)
        self.declare_partials("te_angle", "lower_weights", rows=[0], cols=last)
        self.declare_partials("te_angle", "TE_thickness")
        self.declare_partials("wiggliness", ["upper_weights", "lower_weights"])

    def compute(self, inputs, outputs):
        wu, wl, te = inputs["upper_weights"], inputs["lower_weights"], inputs["TE_thickness"]
        outputs["thickness"] = self._rows @ (wu - wl) + self._x * te
        outputs["te_angle"] = DEG * (np.arctan(wu[-1]) - np.arctan(wl[-1]) + np.arctan(te))
        outputs["wiggliness"] = wiggliness(wu, wl)

    def compute_partials(self, inputs, partials):
        wu, wl, te = inputs["upper_weights"], inputs["lower_weights"], inputs["TE_thickness"]
        partials["te_angle", "upper_weights"] = DEG * (1.0 / (1.0 + wu[-1] ** 2))
        partials["te_angle", "lower_weights"] = -DEG * (1.0 / (1.0 + wl[-1] ** 2))
        partials["te_angle", "TE_thickness"] = DEG * (1.0 / (1.0 + te**2))
        # d/dw sum((D w)^2) = 2 D^T D w, with D the second-difference operator.
        d = np.diff(np.eye(NUM_WEIGHTS), 2, axis=0)
        partials["wiggliness", "upper_weights"] = 2.0 * d.T @ (d @ wu)
        partials["wiggliness", "lower_weights"] = 2.0 * d.T @ (d @ wl)
