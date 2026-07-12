#!/usr/bin/env python3
"""Generate golden reference data for Bora's tests from Python NeuralFoil.

Runs `neuralfoil.get_aero_from_kulfan_parameters` over a matrix of airfoils,
operating conditions, and model sizes, and writes the results as a C++ header
consumed by the GTest suite.

Usage (requires `pip install neuralfoil aerosandbox`):
    python3 scripts/generate_golden_data.py test/golden_data.h
"""

import sys

import aerosandbox as asb
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
    out_path = sys.argv[1] if len(sys.argv) > 1 else "test/golden_data.h"

    cases = []
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
                inputs = (
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
                cases.append(
                    {
                        "name": f"{airfoil_name}_a{alpha}_re{re:.0e}_{model_size}",
                        "model_size": model_size,
                        "inputs": inputs,
                        "expected": output_vector(aero),
                    }
                )

    def fmt(values: list[float]) -> str:
        return ", ".join(np.format_float_scientific(v, precision=17) for v in values)

    with open(out_path, "w") as f:
        f.write(
            "// Golden reference data generated from Python NeuralFoil "
            f"{nf.__version__}.\n"
            "// Do not edit by hand; regenerate with scripts/generate_golden_data.py.\n"
            "#pragma once\n\n"
            "#include <array>\n\n"
            "namespace bora::testing {\n\n"
            "struct GoldenCase {\n"
            "    const char* name;\n"
            "    const char* model_size;\n"
            "    // upper_weights[8], lower_weights[8], LE weight, TE thickness,\n"
            "    // alpha, Re, n_crit, xtr_upper, xtr_lower\n"
            "    std::array<double, 23> inputs;\n"
            "    // analysis_confidence, CL, CD, CM, Top_Xtr, Bot_Xtr,\n"
            "    // upper theta[32], H[32], ue/vinf[32], lower theta[32], H[32], ue/vinf[32]\n"
            "    std::array<double, 198> expected;\n"
            "};\n\n"
            f"inline constexpr std::array<GoldenCase, {len(cases)}> kGoldenCases = {{{{\n"
        )
        for case in cases:
            f.write(
                f'    {{"{case["name"]}", "{case["model_size"]}",\n'
                f'     {{{fmt(case["inputs"])}}},\n'
                f'     {{{fmt(case["expected"])}}}}},\n'
            )
        f.write("}};\n\n}  // namespace bora::testing\n")

    print(f"Wrote {len(cases)} golden cases to {out_path}")


if __name__ == "__main__":
    main()
