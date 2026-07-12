#!/usr/bin/env python3
"""Generate output reference data for Bora's tests from Python NeuralFoil.

Runs `neuralfoil.get_aero_from_kulfan_parameters` over a matrix of airfoils,
operating conditions, and model sizes, and writes the results to an HDF5 file
consumed by the GTest suite. The file is committed to the repository, so this
script only needs to be re-run when updating the reference data.

Usage (requires `pip install neuralfoil aerosandbox h5py`):
    python3 scripts/generate_output_ref_data.py test/data/output_ref_data.h5
"""

import sys

import aerosandbox as asb
import h5py
import neuralfoil as nf
import numpy as np

N = 32  # Boundary layer points per side

AIRFOILS = ["naca0012", "naca4412", "dae11", "ag36"]

# (alpha [deg], Re, n_crit, xtr_upper, xtr_lower)
CONDITIONS = [
    (3.0, 1e6, 9.0, 1.0, 1.0),
    (-5.0, 2e5, 9.0, 1.0, 1.0),
    (12.0, 5e6, 9.0, 1.0, 1.0),
    (0.0, 1e6, 6.0, 0.3, 0.7),
]

MODEL_SIZES = ["xxsmall", "medium", "xlarge", "xxxlarge"]


def output_vector(aero: dict) -> list[float]:
    """Flattens a NeuralFoil result dict into Bora's canonical output order."""
    values = [
        float(aero["analysis_confidence"][0]),
        float(aero["CL"][0]),
        float(aero["CD"][0]),
        float(aero["CM"][0]),
        float(aero["Top_Xtr"][0]),
        float(aero["Bot_Xtr"][0]),
    ]
    for surface in ("upper", "lower"):
        for quantity in ("theta", "H", "ue/vinf"):
            values += [float(aero[f"{surface}_bl_{quantity}_{i}"][0]) for i in range(N)]
    return values


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "test/data/output_ref_data.h5"

    names = []
    model_sizes = []
    inputs = []
    outputs = []
    for airfoil_name in AIRFOILS:
        kulfan = asb.Airfoil(airfoil_name).to_kulfan_airfoil().kulfan_parameters
        for alpha, re, n_crit, xtr_upper, xtr_lower in CONDITIONS:
            for model_size in MODEL_SIZES:
                aero = nf.get_aero_from_kulfan_parameters(
                    kulfan_parameters=kulfan,
                    alpha=alpha,
                    Re=re,
                    n_crit=n_crit,
                    xtr_upper=xtr_upper,
                    xtr_lower=xtr_lower,
                    model_size=model_size,
                )
                names.append(f"{airfoil_name}_a{alpha}_re{re:.0e}_{model_size}")
                model_sizes.append(model_size)
                inputs.append(
                    [float(w) for w in kulfan["upper_weights"]]
                    + [float(w) for w in kulfan["lower_weights"]]
                    + [
                        float(kulfan["leading_edge_weight"]),
                        float(kulfan["TE_thickness"]),
                        alpha,
                        re,
                        n_crit,
                        xtr_upper,
                        xtr_lower,
                    ]
                )
                outputs.append(output_vector(aero))

    with h5py.File(out_path, "w") as f:
        f.attrs["neuralfoil_version"] = nf.__version__
        f.attrs["description"] = (
            "Python NeuralFoil outputs over a matrix of airfoils, operating "
            "conditions, and model sizes. Inputs are the 23 Bora inputs "
            "(upper_weights[8], lower_weights[8], LE weight, TE thickness, "
            "alpha, Re, n_crit, xtr_upper, xtr_lower). Output order: "
            "analysis_confidence, CL, CD, CM, Top_Xtr, Bot_Xtr, then upper "
            "theta[32], H[32], ue/vinf[32], lower theta[32], H[32], ue/vinf[32]."
        )
        f.create_dataset("names", data=np.array(names, dtype="S64"))
        f.create_dataset("model_sizes", data=np.array(model_sizes, dtype="S16"))
        f.create_dataset("inputs", data=np.array(inputs, dtype=np.float64))
        f.create_dataset("outputs", data=np.array(outputs, dtype=np.float64))

    print(f"Wrote {len(names)} reference cases to {out_path}")


if __name__ == "__main__":
    main()
