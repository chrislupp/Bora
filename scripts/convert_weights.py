#!/usr/bin/env python3
"""Convert NeuralFoil .npz weight files into Bora's HDF5 weight format.

Usage:
    python3 scripts/convert_weights.py <neuralfoil_weights_dir> <output_dir>

The input directory is `neuralfoil/nn_weights_and_biases` from the NeuralFoil
repository (https://github.com/peterdsharpe/NeuralFoil) or from an installed
`neuralfoil` package.

Output layouts:

nn-<size>.h5 (network weights):
    /layers/<i>/weights  float32, shape (rows, cols), row-major
    /layers/<i>/biases   float32, shape (rows,)

    Layer groups are named "0", "1", ... in evaluation order; the layer count
    is the number of groups under /layers.

scaled_input_distribution.h5 (input distribution statistics):
    /mean     float64, shape (n,)
    /inv_cov  float64, shape (n, n), row-major
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def convert_network(npz_path: Path, out_path: Path) -> None:
    params = dict(np.load(npz_path))
    layer_indices = sorted({int(key.split(".")[1]) for key in params.keys()})

    with h5py.File(out_path, "w") as f:
        layers = f.create_group("layers")
        for out_index, i in enumerate(layer_indices):
            w = np.asarray(params[f"net.{i}.weight"], dtype=np.float32)
            b = np.asarray(params[f"net.{i}.bias"], dtype=np.float32)
            assert w.ndim == 2 and b.ndim == 1 and w.shape[0] == b.shape[0]
            group = layers.create_group(str(out_index))
            group.create_dataset("weights", data=w)
            group.create_dataset("biases", data=b)
    print(f"  {npz_path.name} -> {out_path.name} "
          f"({len(layer_indices)} layers, {out_path.stat().st_size} bytes)")


def convert_distribution(npz_path: Path, out_path: Path) -> None:
    dist = dict(np.load(npz_path))
    mean = np.asarray(dist["mean_inputs_scaled"], dtype=np.float64)
    inv_cov = np.asarray(dist["inv_cov_inputs_scaled"], dtype=np.float64)
    n = mean.shape[0]
    assert inv_cov.shape == (n, n)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("mean", data=mean)
        f.create_dataset("inv_cov", data=inv_cov)
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
        convert_network(npz_path, out_dir / f"{npz_path.stem}.h5")

    dist_path = in_dir / "scaled_input_distribution.npz"
    convert_distribution(dist_path, out_dir / "scaled_input_distribution.h5")


if __name__ == "__main__":
    main()
