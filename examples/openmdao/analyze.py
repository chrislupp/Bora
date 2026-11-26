#!/usr/bin/env python3
"""Airfoil analysis with Bora over Philote: a direct client call, then OpenMDAO.

Part 1 talks to bora_server with Philote-Python's bare ExplicitClient: it sets
the options, reads the variable definitions and asks for values and partials at
one operating point, the same one examples/example.rs analyzes in Rust.

Part 2 puts the same discipline in an OpenMDAO model and flies the airfoil at a
prescribed lift coefficient instead of a prescribed angle: a BalanceComp and a
Newton solver find the angle, using Bora's partials for the Newton Jacobian.
The model then sweeps the lift coefficient, and computes the derivatives of
drag with respect to the shape at fixed lift, which it checks against central
differences of the solved model.

Start the server first, in a second terminal, from the repository root:
    cargo run --release --features philote --bin bora_server

Then run, also from the repository root:
    pip install -r examples/openmdao/requirements.txt
    python3 examples/openmdao/analyze.py
"""

import argparse
import sys

import grpc
import numpy as np
import openmdao.api as om
import philote_mdo.general as pmg
import philote_mdo.generated.data_pb2 as data
import philote_mdo.openmdao as pmom

# The airfoil of the quickstart and of examples/example.rs.
UPPER_WEIGHTS = np.array([0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32])
LOWER_WEIGHTS = np.array([-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06])
LEADING_EDGE_WEIGHT = 0.02
TE_THICKNESS = 0.0025

REYNOLDS = 1e6
N_CRIT = 9.0

CL_TARGET = 1.0
CL_SWEEP = np.arange(0.2, 1.61, 0.2)


def connect(address, timeout=10.0):
    """Open a channel to bora_server, failing with a hint if nothing answers."""
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        sys.exit(f"no bora_server at {address}; start one with\n"
                 "    cargo run --release --features philote --bin bora_server")
    return channel


def flow_inputs(alpha=None):
    """The airfoil and flow condition as Philote inputs: name -> array."""
    inputs = {
        "upper_weights": UPPER_WEIGHTS,
        "lower_weights": LOWER_WEIGHTS,
        "leading_edge_weight": LEADING_EDGE_WEIGHT,
        "TE_thickness": TE_THICKNESS,
        "Re": REYNOLDS,
        "n_crit": N_CRIT,
        "xtr_upper": 1.0,
        "xtr_lower": 1.0,
    }
    if alpha is not None:
        inputs["alpha"] = alpha
    return {name: np.atleast_1d(np.asarray(value, dtype=float))
            for name, value in inputs.items()}


def open_client(channel, model_size):
    """A bare Philote client, set up the way OpenMDAO's component sets up its own."""
    client = pmg.ExplicitClient(channel=channel)
    client.get_discipline_info()
    client.send_options({"model_size": model_size})
    client.run_setup()  # the server reloads the network here
    client.get_variable_definitions()
    client.get_partials_definitions()
    return client


def direct_call(channel, model_size):
    """Part 1: one operating point through the bare Philote client."""
    client = open_client(channel, model_size)
    print(f"Discipline '{client._name}' {client._version}, model size '{model_size}'")
    print("  inputs: ", " ".join(v.name for v in client._var_meta if v.type == data.kInput))
    print("  outputs:", " ".join(v.name for v in client._var_meta if v.type == data.kOutput))

    inputs = flow_inputs(alpha=3.0)
    outputs = client.run_compute(inputs)
    partials = client.run_compute_partials(inputs)

    cl, cd, cm = (outputs[k][0] for k in ("CL", "CD", "CM"))
    print(f"\nalpha = 3.00 deg, Re = {REYNOLDS:.2e}")
    print(f"  CL = {cl:9.5f}   CD = {cd:9.5f}   CM = {cm:9.5f}   L/D = {cl / cd:8.3f}")
    print(f"  dCL/dalpha = {partials['CL', 'alpha'][0]:.5f} per deg")
    print("  dCL/d(upper_weights):", fmt(partials["CL", "upper_weights"]))


class FixedLift(om.Group):
    """Bora at a prescribed lift coefficient: Newton drives alpha until CL = CL_target."""

    def initialize(self):
        self.options.declare("channel")
        self.options.declare("model_size", types=str)

    def setup(self):
        self.add_subsystem(
            "bora",
            pmom.RemoteExplicitComponent(
                channel=self.options["channel"], model_size=self.options["model_size"]
            ),
            promotes=["*"],
        )
        balance = om.BalanceComp()
        balance.add_balance("alpha", units="deg", val=2.0, lhs_name="CL", rhs_name="CL_target")
        self.add_subsystem("balance", balance, promotes=["*"])

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        newton.options["atol"] = 1e-12
        newton.options["rtol"] = 1e-12
        newton.options["maxiter"] = 20
        newton.options["iprint"] = 0
        self.linear_solver = om.DirectSolver()


def fixed_lift(channel, model_size):
    """Part 2: the airfoil at fixed lift in OpenMDAO."""
    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        "wing", FixedLift(channel=channel, model_size=model_size), promotes=["*"]
    )
    prob.setup()
    for name, value in flow_inputs().items():
        prob.set_val(name, value)
    newton = prob.model.wing.nonlinear_solver

    print(f"\nFixed lift, Re = {REYNOLDS:.2e}")
    print(f"{'CL':>6}  {'alpha':>7}  {'CD':>8}  {'CM':>8}  {'L/D':>6}  {'confidence':>10}"
          f"  {'Newton':>6}")
    for target in CL_SWEEP:
        prob.set_val("CL_target", target)
        prob.run_model()
        cl, cd, cm, conf = (prob.get_val(k)[0] for k in ("CL", "CD", "CM",
                                                         "analysis_confidence"))
        print(f"{cl:6.3f}  {prob.get_val('alpha')[0]:7.3f}  {cd:8.5f}  {cm:8.4f}  "
              f"{cl / cd:6.1f}  {conf:10.4f}  {newton._iter_count:6d}")

    # Derivatives of the solved model: OpenMDAO differentiates through the
    # Newton solve, so alpha adjusts to hold CL at the target.
    prob.set_val("CL_target", CL_TARGET)
    prob.run_model()
    totals = prob.compute_totals(of=["CD", "alpha"], wrt=["upper_weights"])
    dcd_fixed_lift = totals["CD", "upper_weights"].ravel()

    print(f"\nShape derivatives at CL = {CL_TARGET} "
          f"(alpha = {prob.get_val('alpha')[0]:.3f} deg)")
    print("  dalpha/d(upper_weights):          ", fmt(totals["alpha", "upper_weights"].ravel()))
    print("  dCD/d(upper_weights), fixed CL:   ", fmt(dcd_fixed_lift, 5))

    # The same through the bare client, at this angle: fixed alpha, CL free.
    client = open_client(channel, model_size)
    inputs = flow_inputs(alpha=prob.get_val("alpha")[0])
    dcd_fixed_alpha = client.run_compute_partials(inputs)["CD", "upper_weights"]
    print("  dCD/d(upper_weights), fixed alpha:", fmt(dcd_fixed_alpha, 5))

    # Check the fixed-lift derivative of one weight by central differences,
    # re-solving the model on each side.
    k, step = 3, 1e-5
    drag = []
    for sign in (1.0, -1.0):
        weights = UPPER_WEIGHTS.copy()
        weights[k] += sign * step
        prob.set_val("upper_weights", weights)
        prob.run_model()
        drag.append(prob.get_val("CD")[0])
    fd = (drag[0] - drag[1]) / (2.0 * step)
    print(f"\n  dCD/d(upper_weights[{k}]) at fixed CL: analytic {dcd_fixed_lift[k]: .6e}, "
          f"central difference {fd: .6e}")


def fmt(values, digits=4):
    return " ".join(f"{v:{digits + 4}.{digits}f}" for v in np.ravel(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="localhost:50051",
                        help="address of a running bora_server (default: localhost:50051)")
    parser.add_argument("--model-size", default="xlarge",
                        help="NeuralFoil network size (default: xlarge)")
    args = parser.parse_args()

    channel = connect(args.server)
    direct_call(channel, args.model_size)
    fixed_lift(channel, args.model_size)


if __name__ == "__main__":
    main()
