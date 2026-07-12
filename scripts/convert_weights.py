#!/usr/bin/env python3
"""Convert NeuralFoil .npz weight files into Bora's flat binary format.

Usage:
    python3 scripts/convert_weights.py <neuralfoil_weights_dir> <output_dir>

The input directory is `neuralfoil/nn_weights_and_biases` from the NeuralFoil
repository (https://github.com/peterdsharpe/NeuralFoil) or from an installed
`neuralfoil` package.

Output formats (all little-endian):

nn-<size>.bnn (network weights):
    magic      char[8]  "BORANN01"
    n_layers   uint32
    per layer:
        rows   uint32
        cols   uint32
        weight float32[rows*cols]   row-major
        bias   float32[rows]

scaled_input_distribution.bid (input distribution statistics):
    magic      char[8]  "BORAID01"
    n          uint32
    mean       float64[n]
    inv_cov    float64[n*n]         row-major
"""

import struct
import sys
from pathlib import Path

import numpy as np

NN_MAGIC = b"BORANN01"
DIST_MAGIC = b"BORAID01"


def convert_network(npz_path: Path, out_path: Path) -> None:
    params = dict(np.load(npz_path))
    layer_indices = sorted({int(key.split(".")[1]) for key in params.keys()})

    with open(out_path, "wb") as f:
        f.write(NN_MAGIC)
        f.write(struct.pack("<I", len(layer_indices)))
        for i in layer_indices:
            w = np.ascontiguousarray(params[f"net.{i}.weight"], dtype="<f4")
            b = np.ascontiguousarray(params[f"net.{i}.bias"], dtype="<f4")
            assert w.ndim == 2 and b.ndim == 1 and w.shape[0] == b.shape[0]
            f.write(struct.pack("<II", w.shape[0], w.shape[1]))
            f.write(w.tobytes())
            f.write(b.tobytes())
    print(f"  {npz_path.name} -> {out_path.name} "
          f"({len(layer_indices)} layers, {out_path.stat().st_size} bytes)")


def convert_distribution(npz_path: Path, out_path: Path) -> None:
    dist = dict(np.load(npz_path))
    mean = np.ascontiguousarray(dist["mean_inputs_scaled"], dtype="<f8")
    inv_cov = np.ascontiguousarray(dist["inv_cov_inputs_scaled"], dtype="<f8")
    n = mean.shape[0]
    assert inv_cov.shape == (n, n)

    with open(out_path, "wb") as f:
        f.write(DIST_MAGIC)
        f.write(struct.pack("<I", n))
        f.write(mean.tobytes())
        f.write(inv_cov.tobytes())
    print(f"  {npz_path.name} -> {out_path.name} ({n} inputs)")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    nn_files = sorted(in_dir.glob("nn-*.npz"))
    if not nn_files:
        print(f"error: no nn-*.npz files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    for npz_path in nn_files:
        convert_network(npz_path, out_dir / f"{npz_path.stem}.bnn")

    dist_path = in_dir / "scaled_input_distribution.npz"
    convert_distribution(dist_path, out_dir / "scaled_input_distribution.bid")


if __name__ == "__main__":
    main()
