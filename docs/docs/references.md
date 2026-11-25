---
sidebar_position: 99
title: "References"
description: "How to cite Bora and NeuralFoil, and the sources this documentation draws on."
---

# References

## Citing

Bora is an implementation. The model it evaluates, and every claim about that
model's accuracy, comes from NeuralFoil. If Bora contributes to published
work, cite the NeuralFoil paper:

```bibtex
@misc{neuralfoil_paper,
  title = {{NeuralFoil}: An Airfoil Aerodynamics Analysis Tool Using Physics-Informed Machine Learning},
  author = {Peter Sharpe and R. John Hansman},
  year = {2025},
  eprint = {2503.16323},
  archivePrefix = {arXiv},
  primaryClass = {physics.flu-dyn},
  url = {https://arxiv.org/abs/2503.16323},
}
```

and Peter Sharpe's thesis, whose chapter on NeuralFoil is its long-form
documentation:

```bibtex
@phdthesis{aerosandbox_phd_thesis,
  title = {Accelerating Practical Engineering Design Optimization with Computational Graph Transformations},
  author = {Sharpe, Peter D.},
  school = {Massachusetts Institute of Technology},
  year = {2024},
}
```

To identify the software you ran, cite the Bora repository,
[github.com/chrislupp/Bora](https://github.com/chrislupp/Bora), with the
version from `bora::version()`.

## Attribution and licenses

Bora is derived from two MIT-licensed projects by Peter Sharpe:

- **NeuralFoil**: the network architecture, the input and output processing,
  the analysis model, and the network weights in `data/`, which are converted
  from NeuralFoil's weight files.
- **AeroSandbox**: the Kulfan parameterization, coordinate generation,
  normalization and parameter fitting, and the compressibility, post-stall
  and control-surface corrections.

Bora itself is licensed under Apache-2.0. The license notices of both
upstream projects are reproduced in
[`NOTICE`](https://github.com/chrislupp/Bora/blob/main/NOTICE).

This documentation is written for Bora. Where it describes how NeuralFoil
works, how it was trained or how accurate it is, it summarizes the sources
below in its own words and cites them at the point of use. Numerical results
quoted from the NeuralFoil paper are identified as such where they appear.

## Sources

### The model

#### Sharpe and Hansman, 2025 {/* #neuralfoil-paper */}

P. Sharpe and R. J. Hansman, "NeuralFoil: An Airfoil Aerodynamics Analysis
Tool Using Physics-Informed Machine Learning," arXiv:2503.16323, 2025.
[arxiv.org/abs/2503.16323](https://arxiv.org/abs/2503.16323)

The primary reference for the model: encodings, architecture, symmetry
embedding, the confidence output, training data, accuracy against XFoil, and
an airfoil optimization case study.

#### Sharpe, 2024 {/* #sharpe-thesis */}

P. D. Sharpe, "Accelerating Practical Engineering Design Optimization with
Computational Graph Transformations," PhD thesis, Massachusetts Institute of
Technology, 2024.
[dspace.mit.edu/handle/1721.1/157809](https://dspace.mit.edu/handle/1721.1/157809)

Contains an extended chapter on NeuralFoil.

#### Sharpe, 2023 {/* #neuralfoil-repository */}

P. Sharpe, *NeuralFoil* (software and documentation), 2023.
[github.com/peterdsharpe/NeuralFoil](https://github.com/peterdsharpe/NeuralFoil)

The reference implementation that Bora ports, and its documentation, which
includes the comparison with XFoil and the guidance on choosing a network
size that these pages draw on.

#### Sharpe, 2021 {/* #aerosandbox */}

P. Sharpe, *AeroSandbox* (software).
[github.com/peterdsharpe/AeroSandbox](https://github.com/peterdsharpe/AeroSandbox)

The source of the Kulfan geometry routines, and the home of NeuralFoil's
compressibility, post-stall and control-surface extensions, which Bora ports
as its [corrected analysis](./guide/corrections.md).

### Background

#### Drela, 1989 {/* #xfoil */}

M. Drela, "XFOIL: An Analysis and Design System for Low Reynolds Number
Airfoils," in *Low Reynolds Number Aerodynamics*, Lecture Notes in
Engineering, vol. 54, Springer, 1989.
[web.mit.edu/drela/Public/web/xfoil](https://web.mit.edu/drela/Public/web/xfoil/)

The code NeuralFoil is trained to reproduce.

#### Kulfan, 2008 {/* #kulfan-2008 */}

B. M. Kulfan, "Universal Parametric Geometry Representation Method,"
*Journal of Aircraft*, vol. 45, no. 1, 2008.
[doi.org/10.2514/1.29958](https://doi.org/10.2514/1.29958)

The class-shape transformation behind the 18 shape parameters.

#### Masters et al., 2017 {/* #masters-2017 */}

D. A. Masters, N. J. Taylor, T. C. S. Rendall, C. B. Allen and D. J. Poole,
"Geometric Comparison of Aerofoil Shape Parameterization Methods,"
*AIAA Journal*, vol. 55, no. 5, 2017.
[doi.org/10.2514/1.J054943](https://doi.org/10.2514/1.J054943)

Evidence for how many shape parameters an airfoil description needs.

#### Truong, 2020 {/* #truong-2020 */}

V. K. Truong, "An analytical model for airfoil aerodynamic characteristics
over the entire 360° angle of attack range," *Journal of Renewable and
Sustainable Energy*, vol. 12, 2020.
[doi.org/10.1063/1.5126055](https://doi.org/10.1063/1.5126055)

The separated-flow model that the corrected analysis blends into beyond
stall.

#### Drela, 1998 {/* #drela-1998 */}

M. Drela, "Pros and Cons of Airfoil Optimization," in *Frontiers of
Computational Fluid Dynamics 1998*, World Scientific, 1998.

How a single-point optimization exploits a flow model, and why robustness
constraints matter.
