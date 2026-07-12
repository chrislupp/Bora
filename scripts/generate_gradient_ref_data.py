#!/usr/bin/env python3
"""Generate gradient reference data for Bora's tests from Python NeuralFoil.

Differentiates `neuralfoil.get_aero_from_kulfan_parameters` with CasADi over
the same case matrix as generate_output_ref_data.py and writes the jacobians of
the scalar outputs plus sampled boundary layer stations to an HDF5 file
consumed by the GTest suite. The file is committed to the repository, so this
script only needs to be re-run when updating the reference data.

Usage (requires `pip install neuralfoil aerosandbox casadi h5py`):
    python3 scripts/generate_gradient_ref_data.py test/data/gradient_ref_data.h5
"""

import sys

import aerosandbox as asb
import casadi as cs
import h5py
import neuralfoil as nf
import numpy as np

N = 32  # Boundary layer points per side

# Boundary layer stations whose gradients are recorded (must match the test).
BL_STATIONS = [0, 15, 31]

AIRFOILS = ["naca0012", "naca4412", "dae11", "ag36"]

# (alpha [deg], Re, n_crit, xtr_upper, xtr_lower)
CONDITIONS = [
    (3.0, 1e6, 9.0, 1.0, 1.0),
    (-5.0, 2e5, 9.0, 1.0, 1.0),
    (12.0, 5e6, 9.0, 1.0, 1.0),
    (0.0, 1e6, 6.0, 0.3, 0.7),
]

MODEL_SIZES = ["xxsmall", "medium", "xlarge", "xxxlarge"]


def build_jacobian_function(model_size: str) -> cs.Function:
    """Returns a CasADi function mapping the 23 inputs to the row-major
    jacobian of the recorded outputs."""
    x = cs.MX.sym("x", 23)
    kulfan = {
        "upper_weights": x[0:8],
        "lower_weights": x[8:16],
        "leading_edge_weight": x[16],
        "TE_thickness": x[17],
    }
    aero = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=kulfan,
        alpha=x[18],
        Re=x[19],
        n_crit=x[20],
        xtr_upper=x[21],
        xtr_lower=x[22],
        model_size=model_size,
    )
    rows = [
        aero["analysis_confidence"],
        aero["CL"],
        aero["CD"],
        aero["CM"],
        aero["Top_Xtr"],
        aero["Bot_Xtr"],
    ]
    for surface in ("upper", "lower"):
        for quantity in ("theta", "H", "ue/vinf"):
            rows += [aero[f"{surface}_bl_{quantity}_{i}"] for i in BL_STATIONS]
    jacobian = cs.jacobian(cs.vertcat(*rows), x)
    return cs.Function("jacobian_fn", [x], [jacobian])


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "test/data/gradient_ref_data.h5"

    jacobian_functions = {size: build_jacobian_function(size) for size in MODEL_SIZES}

    names = []
    model_sizes = []
    inputs = []
    jacobians = []
    for airfoil_name in AIRFOILS:
        kulfan = asb.Airfoil(airfoil_name).to_kulfan_airfoil().kulfan_parameters
        for alpha, re, n_crit, xtr_upper, xtr_lower in CONDITIONS:
            for model_size in MODEL_SIZES:
                case_inputs = np.concatenate(
                    [
                        np.asarray(kulfan["upper_weights"], dtype=float),
                        np.asarray(kulfan["lower_weights"], dtype=float),
                        [
                            float(kulfan["leading_edge_weight"]),
                            float(kulfan["TE_thickness"]),
                            alpha,
                            re,
                            n_crit,
                            xtr_upper,
                            xtr_lower,
                        ],
                    ]
                )
                names.append(f"{airfoil_name}_a{alpha}_re{re:.0e}_{model_size}")
                model_sizes.append(model_size)
                inputs.append(case_inputs)
                jacobians.append(np.asarray(jacobian_functions[model_size](case_inputs)))

    n_rows = 6 + 2 * 3 * len(BL_STATIONS)

    with h5py.File(out_path, "w") as f:
        f.attrs["neuralfoil_version"] = nf.__version__
        f.attrs["description"] = (
            "Jacobians of Python NeuralFoil outputs w.r.t. the 23 Bora inputs "
            "(upper_weights[8], lower_weights[8], LE weight, TE thickness, "
            "alpha, Re, n_crit, xtr_upper, xtr_lower), computed with CasADi. "
            "Row order: analysis_confidence, CL, CD, CM, Top_Xtr, Bot_Xtr, "
            "then for each surface (upper, lower) and quantity (theta, H, "
            "ue/vinf) the stations in bl_stations."
        )
        f.create_dataset("names", data=np.array(names, dtype="S64"))
        f.create_dataset("model_sizes", data=np.array(model_sizes, dtype="S16"))
        f.create_dataset("bl_stations", data=np.array(BL_STATIONS, dtype=np.int64))
        f.create_dataset("inputs", data=np.array(inputs, dtype=np.float64))
        f.create_dataset("jacobians", data=np.array(jacobians, dtype=np.float64))

    print(f"Wrote {len(names)} gradient reference cases ({n_rows} rows each) to {out_path}")


if __name__ == "__main__":
    main()
