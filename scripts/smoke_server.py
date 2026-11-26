#!/usr/bin/env python3
"""Smoke-test a running bora_server through Philote-Python.

Checks that a bad option is refused and that the server keeps serving after it,
then analyzes the quickstart airfoil (examples/example.rs) and compares the
values and one partial with the known results of the xlarge network.

Usage: python3 scripts/smoke_server.py --address localhost:50051
"""

import argparse
import math
import sys

import grpc
import numpy as np
import philote_mdo.general as pmg

# examples/example.rs at alpha = 3 deg, Re = 1e6, xlarge network.
INPUTS = {
    "upper_weights": [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32],
    "lower_weights": [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06],
    "leading_edge_weight": [0.02],
    "TE_thickness": [0.0025],
    "alpha": [3.0],
    "Re": [1e6],
    "n_crit": [9.0],
    "xtr_upper": [1.0],
    "xtr_lower": [1.0],
}
EXPECTED = {"CL": 0.95162, "CD": 0.00729, "CM": -0.13014}
EXPECTED_DCL_DALPHA = 0.10185
TOLERANCE = 1e-4


def client_for(channel, model_size):
    client = pmg.ExplicitClient(channel=channel)
    client.send_options({"model_size": model_size})
    client.run_setup()
    client.get_variable_definitions()
    client.get_partials_definitions()
    return client


def smoke(address):
    channel = grpc.insecure_channel(address)
    grpc.channel_ready_future(channel).result(timeout=30)

    try:
        client_for(channel, "no-such-size")
    except Exception:  # Philote-Python wraps the gRPC error differently across versions
        pass
    else:
        raise RuntimeError("an unknown model_size was accepted")

    client = client_for(channel, "xlarge")
    inputs = {name: np.array(value, dtype=float) for name, value in INPUTS.items()}
    outputs = client.run_compute(inputs)
    partials = client.run_compute_partials(inputs)

    for name, expected in EXPECTED.items():
        value = float(outputs[name][0])
        if not math.isclose(value, expected, abs_tol=TOLERANCE):
            raise RuntimeError(f"{name} = {value}, expected {expected}")
    slope = float(partials["CL", "alpha"][0])
    if not math.isclose(slope, EXPECTED_DCL_DALPHA, abs_tol=TOLERANCE):
        raise RuntimeError(f"dCL/dalpha = {slope}, expected {EXPECTED_DCL_DALPHA}")

    print(f"bora_server at {address}: bad option refused; "
          f"CL = {float(outputs['CL'][0]):.5f}, dCL/dalpha = {slope:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--address", default="localhost:50051")
    try:
        smoke(parser.parse_args().address)
    except Exception as err:
        sys.exit(f"smoke test failed: {err}")
