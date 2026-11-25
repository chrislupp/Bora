#!/usr/bin/env python3
"""Generates reference data for the corrected and coordinate-based analyses.

Writes tests/data/corrected_ref_data.h5 with two groups of cases:

- "corrected": AeroSandbox's `KulfanAirfoil.get_aero_from_neuralfoil`
  (compressibility, post-stall and control surface corrections) over a matrix
  of airfoils, angles of attack, Mach numbers and control surface settings.
- "airfoil": NeuralFoil's `get_aero_from_coordinates` for coordinates that
  are rotated, scaled and translated away from the normalized frame.

Requires: pip install neuralfoil aerosandbox h5py

Usage: python scripts/generate_corrected_ref_data.py [output.h5]
"""

import sys

import aerosandbox as asb
import h5py
import neuralfoil as nf
import numpy as np
from aerosandbox.geometry.wing import ControlSurface

MODEL_SIZE = "medium"
AIRFOILS = ["naca0012", "naca4412", "dae11", "rae2822"]

# (alpha [deg], Re, mach, include_360_deg_effects,
#  control surfaces as (deflection [deg], hinge_point))
CONDITIONS = [
    (3.0, 1e6, 0.0, True, []),
    (3.0, 1e6, 0.0, False, []),
    (2.0, 5e6, 0.3, True, []),
    (2.0, 5e6, 0.6, False, []),
    (1.0, 8e6, 0.72, True, []),
    (1.0, 8e6, 0.78, True, []),
    (0.5, 8e6, 0.85, True, []),
    (0.5, 8e6, 0.95, True, []),
    (1.0, 1e7, 1.05, True, []),
    (1.0, 1e7, 1.15, True, []),
    (2.0, 1e7, 1.6, False, []),
    (17.0, 5e5, 0.1, True, []),
    (24.0, 5e5, 0.1, True, []),
    (-27.0, 5e5, 0.2, True, []),
    (60.0, 1e6, 0.0, True, []),
    (170.0, 1e6, 0.05, True, []),
    (200.0, 1e6, 0.05, True, []),
    (-330.0, 2e6, 0.3, True, []),
    (24.0, 5e5, 0.1, False, []),
    (2.0, 2e6, 0.2, True, [(10.0, 0.75)]),
    (-1.0, 2e6, 0.4, True, [(-15.0, 0.7), (5.0, 0.9)]),
    (4.0, 3e6, 0.74, False, [(3.0, 0.8)]),
]

CORRECTED_KEYS = [
    "analysis_confidence",
    "CL",
    "CD",
    "CM",
    "Cpmin",
    "Top_Xtr",
    "Bot_Xtr",
    "mach_crit",
    "mach_dd",
    "Cpmin_0",
]
SCALAR_KEYS = ["analysis_confidence", "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]

# (rotation [deg], scale, x translation, y translation) applied to the
# normalized coordinates, then (alpha [deg], Re).
TRANSFORMS = [
    (0.0, 1.0, 0.0, 0.0, 3.0, 1e6),
    (4.0, 1.0, 0.0, 0.0, 3.0, 1e6),
    (-6.0, 2.5, 0.3, -0.2, 1.0, 4e5),
    (2.0, 0.4, -1.0, 0.5, 6.0, 3e6),
]


def scalar(value) -> float:
    return float(np.reshape(value, -1)[0])


def kulfan_vector(kulfan: dict) -> list[float]:
    return (
        [float(w) for w in kulfan["upper_weights"]]
        + [float(w) for w in kulfan["lower_weights"]]
        + [float(kulfan["leading_edge_weight"]), float(kulfan["TE_thickness"])]
    )


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "tests/data/corrected_ref_data.h5"

    names, inputs, outputs, surfaces = [], [], [], []
    for airfoil_name in AIRFOILS:
        airfoil = asb.Airfoil(airfoil_name).to_kulfan_airfoil()
        for alpha, re, mach, include_360, control_surfaces in CONDITIONS:
            aero = airfoil.get_aero_from_neuralfoil(
                alpha=alpha,
                Re=re,
                mach=mach,
                model_size=MODEL_SIZE,
                control_surfaces=[
                    ControlSurface(deflection=d, hinge_point=h)
                    for d, h in control_surfaces
                ],
                include_360_deg_effects=include_360,
            )
            names.append(f"{airfoil_name}_a{alpha}_m{mach}_case{len(names)}")
            inputs.append(
                kulfan_vector(airfoil.kulfan_parameters)
                + [alpha, re, mach, float(include_360), float(len(control_surfaces))]
            )
            # Up to two control surfaces per case, padded with zeros.
            padded = control_surfaces + [(0.0, 0.0)] * (2 - len(control_surfaces))
            surfaces.append([value for surface in padded for value in surface])
            outputs.append(
                [scalar(aero[key]) for key in CORRECTED_KEYS]
                + [scalar(airfoil.max_thickness())]
            )

    airfoil_names, coordinates, counts, conditions, airfoil_outputs = [], [], [], [], []
    for airfoil_name in AIRFOILS:
        base = asb.Airfoil(airfoil_name).normalize()
        for rotation, scale, dx, dy, alpha, re in TRANSFORMS:
            moved = (
                base.rotate(np.radians(rotation))
                .scale(scale, scale)
                .translate(dx, dy)
            )
            aero = nf.get_aero_from_coordinates(
                moved.coordinates, alpha=alpha, Re=re, model_size=MODEL_SIZE
            )
            normalization = moved.normalize(return_dict=True)
            airfoil_names.append(f"{airfoil_name}_r{rotation}_s{scale}")
            counts.append(len(moved.coordinates))
            coordinates.append(moved.coordinates)
            conditions.append([alpha, re])
            airfoil_outputs.append(
                [scalar(aero[key]) for key in SCALAR_KEYS]
                + [
                    float(normalization[key])
                    for key in (
                        "x_translation",
                        "y_translation",
                        "scale_factor",
                        "rotation_angle",
                    )
                ]
            )

    with h5py.File(out_path, "w") as f:
        f.attrs["neuralfoil_version"] = nf.__version__
        f.attrs["aerosandbox_version"] = asb.__version__
        f.attrs["model_size"] = MODEL_SIZE

        g = f.create_group("corrected")
        g.attrs["description"] = (
            "KulfanAirfoil.get_aero_from_neuralfoil. Inputs: upper_weights[8], "
            "lower_weights[8], LE weight, TE thickness, alpha, Re, mach, "
            "include_360_deg_effects, number of control surfaces. "
            "control_surfaces: (deflection, hinge_point) x 2, zero padded. "
            "Outputs: " + ", ".join(CORRECTED_KEYS) + ", max_thickness."
        )
        g.create_dataset("names", data=np.array(names, dtype="S64"))
        g.create_dataset("inputs", data=np.array(inputs, dtype=np.float64))
        g.create_dataset("control_surfaces", data=np.array(surfaces, dtype=np.float64))
        g.create_dataset("outputs", data=np.array(outputs, dtype=np.float64))

        g = f.create_group("airfoil")
        g.attrs["description"] = (
            "neuralfoil.get_aero_from_coordinates. coordinates: all cases' "
            "(x, y) points concatenated, counts[i] points per case. "
            "conditions: alpha, Re. Outputs: " + ", ".join(SCALAR_KEYS) + ", "
            "then the normalization x_translation, y_translation, "
            "scale_factor, rotation_angle [deg]."
        )
        g.create_dataset("names", data=np.array(airfoil_names, dtype="S64"))
        g.create_dataset(
            "coordinates", data=np.concatenate(coordinates).astype(np.float64)
        )
        g.create_dataset("counts", data=np.array(counts, dtype=np.float64))
        g.create_dataset("conditions", data=np.array(conditions, dtype=np.float64))
        g.create_dataset("outputs", data=np.array(airfoil_outputs, dtype=np.float64))

    print(
        f"Wrote {len(names)} corrected and {len(airfoil_names)} airfoil "
        f"reference cases to {out_path}"
    )


if __name__ == "__main__":
    main()
