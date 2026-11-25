#!/usr/bin/env python3
"""Multipoint airfoil optimization with OpenMDAO, IPOPT and Bora over Philote.

Reproduces the human-powered-aircraft case from AeroSandbox's NeuralFoil
optimization tutorial, which follows Drela's "Pros and Cons of Airfoil
Optimization" (1998), with Bora as the aerodynamic analysis:

    minimize    mean_i(w_i * CD_i)  over six lift coefficients CL_i
    w.r.t.      16 Kulfan weights, the leading-edge weight, one alpha per point
    subject to  CL_i = target_i,  CM_i >= -0.133,  confidence_i >= 0.90,
                alpha increasing with CL,
                t/c >= 0.128 at x/c = 0.33,  t/c >= 0.014 at x/c = 0.90,
                trailing-edge angle >= 6.03 deg,  t/c > 0 everywhere,
                wiggliness <= 2x that of the starting NACA 0012

Each operating point is a `RemoteExplicitComponent` (Philote-Python) talking to
one `bora_server` over gRPC; Bora supplies the analytic partials. The geometric
constraints are evaluated locally in `kulfan.py`.

The script starts `bora_server` itself (building it on first use through cargo)
unless `--server` names one that is already running.

Usage, from the repository root:
    pip install -r examples/openmdao/requirements.txt   # plus pyOptSparse with IPOPT
    python3 examples/openmdao/optimize_daedalus.py
"""

import argparse
import atexit
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import grpc
import numpy as np
import openmdao.api as om
import philote_mdo.openmdao as pmom

from kulfan import NUM_WEIGHTS, KulfanGeometry, selig_coordinates, wiggliness

REPO_ROOT = Path(__file__).resolve().parents[2]

# Operating points: lift coefficients, their weights in the objective, and a
# Reynolds number that keeps Re * sqrt(CL) fixed (constant wing loading).
CL_TARGETS = np.array([0.8, 1.0, 1.2, 1.4, 1.5, 1.6])
CL_WEIGHTS = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
REYNOLDS = 500e3 * (CL_TARGETS / 1.25) ** -0.5
N_POINTS = len(CL_TARGETS)

OBJECTIVE_REF = 0.01
CM_MIN = -0.133
CONFIDENCE_MIN = 0.90
TE_ANGLE_MIN = 6.03  # deg
SPAR_STATIONS = [0.33, 0.90]
SPAR_THICKNESS_MIN = [0.128, 0.014]
# Positive thickness is checked at interior stations; it is identically zero at
# the leading edge, and at the trailing edge with zero trailing-edge thickness.
POSITIVE_STATIONS = list(np.linspace(0.0, 1.0, 101)[1:-1])

# Starting airfoil: NACA 0012 (Kulfan weights fitted to its coordinates).
NACA0012_UPPER = np.array(
    [0.172884, 0.151563, 0.173763, 0.127681, 0.164818, 0.126376, 0.145898, 0.1392]
)
NACA0012_LOWER = -NACA0012_UPPER
WIGGLINESS_MAX = 2.0 * wiggliness(NACA0012_UPPER, NACA0012_LOWER)


def free_port():
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def start_server():
    """Start bora_server on a free local port; returns its address."""
    address = f"localhost:{free_port()}"
    command = ["cargo", "run", "--release", "--quiet", "--features", "philote",
               "--bin", "bora_server", "--", address]
    print(f"Starting bora_server on {address} (first run builds it)...", flush=True)
    server = subprocess.Popen(command, cwd=REPO_ROOT)
    atexit.register(server.terminate)
    return address, server


def wait_for(channel, server, timeout=600.0):
    """Block until the channel is ready, failing early if the server exits."""
    future = grpc.channel_ready_future(channel)
    waited = 0.0
    while True:
        try:
            future.result(timeout=1.0)
            return
        except grpc.FutureTimeoutError:
            waited += 1.0
            if server is not None and server.poll() is not None:
                sys.exit("bora_server exited before accepting connections")
            if waited >= timeout:
                sys.exit("timed out waiting for bora_server")


class AlphaSchedule(om.ExplicitComponent):
    """Differences of consecutive alphas, constrained positive."""

    def setup(self):
        self.add_input("alpha", shape=N_POINTS, units="deg")
        self.add_output("alpha_steps", shape=N_POINTS - 1, units="deg")
        rows = np.repeat(np.arange(N_POINTS - 1), 2)
        cols = np.column_stack([np.arange(N_POINTS - 1), np.arange(1, N_POINTS)]).ravel()
        vals = np.tile([-1.0, 1.0], N_POINTS - 1)
        self.declare_partials("alpha_steps", "alpha", rows=rows, cols=cols, val=vals)

    def compute(self, inputs, outputs):
        outputs["alpha_steps"] = np.diff(inputs["alpha"])


def build_problem(channel, model_size):
    prob = om.Problem(reports=False)
    model = prob.model

    design = model.add_subsystem("design", om.IndepVarComp(), promotes=["*"])
    design.add_output("upper_weights", NACA0012_UPPER)
    design.add_output("lower_weights", NACA0012_LOWER)
    design.add_output("leading_edge_weight", 0.0)
    design.add_output("TE_thickness", 0.0)
    design.add_output("alpha", np.degrees(CL_TARGETS / (2.0 * np.pi)), units="deg")

    shape_inputs = ["upper_weights", "lower_weights", "leading_edge_weight", "TE_thickness"]
    for i in range(N_POINTS):
        name = f"point{i}"
        model.add_subsystem(
            name,
            pmom.RemoteExplicitComponent(channel=channel, model_size=model_size),
            promotes_inputs=shape_inputs,
        )
        model.connect("alpha", f"{name}.alpha", src_indices=[i])

    model.add_subsystem(
        "geometry",
        KulfanGeometry(stations=SPAR_STATIONS + POSITIVE_STATIONS),
        promotes_inputs=["upper_weights", "lower_weights", "TE_thickness"],
    )
    model.add_subsystem("schedule", AlphaSchedule(), promotes_inputs=["alpha"])

    # Objective: the weighted drag, averaged over the points (AeroSandbox's
    # normalization, mean(w * CD), so the optimum compares directly).
    terms = " + ".join(f"{w:g} * CD{i}" for i, w in enumerate(CL_WEIGHTS))
    model.add_subsystem("objective", om.ExecComp(f"drag = ({terms}) / {N_POINTS}"))
    for i in range(N_POINTS):
        model.connect(f"point{i}.CD", f"objective.CD{i}")

    # Design variables and bounds. The first weight of each side is kept away
    # from zero (a round leading edge), as in the reference case.
    upper_lower = np.full(NUM_WEIGHTS, -0.25)
    upper_lower[0] = 0.05
    lower_upper = np.full(NUM_WEIGHTS, 0.25)
    lower_upper[0] = -0.05
    model.add_design_var("upper_weights", lower=upper_lower, upper=0.5)
    model.add_design_var("lower_weights", lower=-0.5, upper=lower_upper)
    model.add_design_var("leading_edge_weight", lower=-1.0, upper=1.0)
    model.add_design_var("alpha", lower=-5.0, upper=18.0, units="deg")

    # ref scales the objective (about 0.08) towards one for IPOPT.
    model.add_objective("objective.drag", ref=OBJECTIVE_REF)

    for i, cl in enumerate(CL_TARGETS):
        model.add_constraint(f"point{i}.CL", equals=cl)
        model.add_constraint(f"point{i}.CM", lower=CM_MIN)
        model.add_constraint(f"point{i}.analysis_confidence", lower=CONFIDENCE_MIN)
    model.add_constraint("schedule.alpha_steps", lower=0.0, linear=True)

    n_spar = len(SPAR_STATIONS)
    model.add_constraint(
        "geometry.thickness", lower=SPAR_THICKNESS_MIN, indices=list(range(n_spar)),
        alias="spar_thickness", linear=True,
    )
    model.add_constraint(
        "geometry.thickness", lower=0.0,
        indices=list(range(n_spar, n_spar + len(POSITIVE_STATIONS))),
        alias="positive_thickness", linear=True,
    )
    model.add_constraint("geometry.te_angle", lower=TE_ANGLE_MIN, units="deg")
    model.add_constraint("geometry.wiggliness", upper=WIGGLINESS_MAX)

    # pyOptSparse runs IPOPT with a limited-memory (L-BFGS) Hessian, where the
    # reference used an exact one; a longer memory than the default 6 pairs
    # roughly halves the iteration count. The reference's start_with_resto is
    # left off: from the NACA 0012 the restoration phase stalls without an
    # exact Hessian.
    driver = prob.driver = om.pyOptSparseDriver(optimizer="IPOPT", print_results=False)
    driver.opt_settings["mu_strategy"] = "monotone"
    driver.opt_settings["limited_memory_max_history"] = 50
    driver.opt_settings["print_level"] = 5

    prob.setup()
    for i, reynolds in enumerate(REYNOLDS):
        prob.set_val(f"point{i}.Re", reynolds)
        prob.set_val(f"point{i}.n_crit", 9.0)
        prob.set_val(f"point{i}.xtr_upper", 1.0)
        prob.set_val(f"point{i}.xtr_lower", 1.0)
    return prob


def report(prob, title):
    print(f"\n{title}")
    print(f"{'CL target':>9}  {'Re':>9}  {'alpha':>7}  {'CL':>7}  {'CD':>8}  "
          f"{'CM':>8}  {'L/D':>6}  {'confidence':>10}")
    alpha = prob.get_val("alpha")
    for i, target in enumerate(CL_TARGETS):
        cl, cd, cm = (prob.get_val(f"point{i}.{k}")[0] for k in ("CL", "CD", "CM"))
        conf = prob.get_val(f"point{i}.analysis_confidence")[0]
        print(f"{target:9.2f}  {REYNOLDS[i]:9.3g}  {alpha[i]:7.3f}  {cl:7.4f}  {cd:8.5f}  "
              f"{cm:8.4f}  {cl / cd:6.1f}  {conf:10.4f}")
    thickness = prob.get_val("geometry.thickness")
    print(f"\nobjective mean(w * CD) = {prob.get_val('objective.drag')[0]:.6f}")
    print(f"t/c at x/c = 0.33: {thickness[0]:.4f}   at x/c = 0.90: {thickness[1]:.4f}")
    print(f"trailing-edge angle = {prob.get_val('geometry.te_angle')[0]:.3f} deg")
    print(f"wiggliness = {prob.get_val('geometry.wiggliness')[0]:.5f} "
          f"(limit {WIGGLINESS_MAX:.5f})")

    np.set_printoptions(precision=6, floatmode="fixed", suppress=True, linewidth=100)
    print("\nKulfan parameters:")
    print(f"  upper_weights = {prob.get_val('upper_weights')}")
    print(f"  lower_weights = {prob.get_val('lower_weights')}")
    print(f"  leading_edge_weight = {prob.get_val('leading_edge_weight')[0]:.6f}")


def state(prob):
    """The design and its operating points, for --json."""
    alpha = prob.get_val("alpha")
    points = []
    for i, target in enumerate(CL_TARGETS):
        values = {k: float(prob.get_val(f"point{i}.{k}")[0])
                  for k in ("CL", "CD", "CM", "analysis_confidence")}
        points.append({"cl_target": float(target), "re": float(REYNOLDS[i]),
                       "alpha": float(alpha[i]), "cl": values["CL"], "cd": values["CD"],
                       "cm": values["CM"], "confidence": values["analysis_confidence"]})
    return {
        "kulfan": {
            "upper": prob.get_val("upper_weights").tolist(),
            "lower": prob.get_val("lower_weights").tolist(),
            "leading_edge": float(prob.get_val("leading_edge_weight")[0]),
            "te_thickness": float(prob.get_val("TE_thickness")[0]),
        },
        "objective": float(prob.get_val("objective.drag")[0]),
        "spar_thickness": prob.get_val("geometry.thickness")[:len(SPAR_STATIONS)].tolist(),
        "te_angle": float(prob.get_val("geometry.te_angle")[0]),
        "points": points,
    }


def schedule_polar(channel, model_size, kulfan):
    """Sweep alpha along the flight schedule, Re = 500k (CL / 1.25)^-0.5.

    The Reynolds number depends on the lift it produces, so each angle is
    solved by fixed-point iteration on Re, starting from the previous angle.
    """
    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        "nf", pmom.RemoteExplicitComponent(channel=channel, model_size=model_size),
        promotes=["*"])
    prob.setup()
    prob.set_val("upper_weights", kulfan["upper"])
    prob.set_val("lower_weights", kulfan["lower"])
    prob.set_val("leading_edge_weight", kulfan["leading_edge"])
    prob.set_val("TE_thickness", kulfan["te_thickness"])
    prob.set_val("n_crit", 9.0)
    prob.set_val("xtr_upper", 1.0)
    prob.set_val("xtr_lower", 1.0)

    rows, reynolds = [], 500e3
    for alpha in np.arange(-4.0, 18.0 + 1e-9, 0.25):
        prob.set_val("alpha", alpha)
        cl = None
        for _ in range(20):
            prob.set_val("Re", reynolds)
            prob.run_model()
            cl = float(prob.get_val("CL")[0])
            if cl <= 0.2:
                break
            updated = 500e3 * (cl / 1.25) ** -0.5
            if abs(updated - reynolds) < 1.0:
                break
            reynolds = updated
        if cl <= 0.2:
            reynolds = 500e3
            continue
        rows.append({"alpha": float(alpha), "re": float(reynolds), "cl": cl,
                     "cd": float(prob.get_val("CD")[0]), "cm": float(prob.get_val("CM")[0]),
                     "confidence": float(prob.get_val("analysis_confidence")[0])})
    return rows


def ipopt_history(prob, objective_ref):
    """Objective, feasibility (inf_pr) and optimality (inf_du) per iteration, from IPOPT's
    output file."""
    path = Path(prob.get_outputs_dir()) / "IPOPT.out"
    pattern = re.compile(r"^\s*(\d+)(r?)\s+(\S+)\s+(\S+)\s+(\S+)\s")
    history = []
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            history.append({"iter": int(match[1]), "restoration": bool(match[2]),
                            "objective": float(match[3]) * objective_ref,
                            "feasibility": float(match[4]),
                            "optimality": float(match[5])})
    return history


def write_json(path, channel, model_size, start, optimized, history):
    data = {
        "model": model_size,
        "cl_targets": CL_TARGETS.tolist(),
        "cl_weights": CL_WEIGHTS.tolist(),
        "constraints": {
            "cm_min": CM_MIN, "confidence_min": CONFIDENCE_MIN, "te_angle_min": TE_ANGLE_MIN,
            "spar_stations": SPAR_STATIONS, "spar_thickness_min": SPAR_THICKNESS_MIN,
        },
        "start": start,
        "optimized": optimized,
        "history": history,
    }
    for key in ("start", "optimized"):
        data[key]["schedule"] = schedule_polar(channel, model_size, data[key]["kulfan"])
    Path(path).write_text(json.dumps(data, indent=1) + "\n")
    print(f"Wrote {path}")


def write_dat(prob, path):
    xy = selig_coordinates(
        prob.get_val("upper_weights"), prob.get_val("lower_weights"),
        prob.get_val("leading_edge_weight")[0], prob.get_val("TE_thickness")[0],
    )
    np.savetxt(path, xy, fmt="%.6f", header="Bora OpenMDAO optimized", comments="")
    print(f"\nWrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", help="address of a running bora_server "
                        "(default: start one on a free port)")
    parser.add_argument("--model-size", default="large",
                        help="NeuralFoil network size (default: large, as in the reference)")
    parser.add_argument("--output", default="optimized_daedalus.dat",
                        help="Selig-format file for the optimized airfoil")
    parser.add_argument("--json", metavar="PATH",
                        help="also write the figure data of the docs page to PATH "
                        "(docs/src/data/optimization.json)")
    args = parser.parse_args()

    server = None
    address = args.server
    if address is None:
        address, server = start_server()
    channel = grpc.insecure_channel(address)
    wait_for(channel, server)

    # OpenMDAO writes its output directory (with IPOPT.out) under target/.
    os.environ.setdefault("OPENMDAO_WORKDIR", str(REPO_ROOT / "target"))
    (REPO_ROOT / "target").mkdir(exist_ok=True)
    prob = build_problem(channel, args.model_size)
    prob.run_model()
    report(prob, "Starting point (NACA 0012):")
    start = state(prob)

    result = prob.run_driver()
    report(prob, "Optimized:" if result.success else "IPOPT did not converge; last iterate:")
    write_dat(prob, args.output)
    if args.json:
        history = ipopt_history(prob, OBJECTIVE_REF)
        write_json(args.json, channel, args.model_size, start, state(prob), history)


if __name__ == "__main__":
    main()
