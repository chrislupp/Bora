/*
    Bora: Rust Port of NeuralFoil

    Copyright 2026 Christopher A. Lupp

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    This work has been cleared for public release, distribution unlimited, case
    number: AFRL-2026-2087.

    The views expressed are those of the authors and do not reflect the
    official guidance or position of the United States Government, the
    Department of Defense or of the United States Air Force.

    Statement from DoD: The Appearance of external hyperlinks does not
    constitute endorsement by the United States Department of Defense (DoD) of
    the linked websites, of the information, products, or services contained
    therein. The DoD does not exercise any editorial, security, or other
    control over the information you may find at these locations.
*/

//! A Rust port of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil), the
//! physics-informed neural network surrogate of XFoil, with analytic gradients
//! for gradient-based airfoil design optimization.
//!
//! Given an airfoil shape in Kulfan (CST) parameterization and an operating
//! condition, [`NeuralFoil`] predicts lift, drag, and moment coefficients,
//! transition locations, boundary layer distributions, and an analysis
//! confidence — optionally with the gradient of every output with respect to
//! all [`NUM_INPUTS`] inputs.
//!
//! ```no_run
//! use bora::{AeroInput, ModelSize, NeuralFoil, input_index};
//!
//! let mut input = AeroInput::default();
//! input.kulfan.upper_weights = [0.24, 0.31, 0.25, 0.35, 0.24, 0.35, 0.28, 0.32];
//! input.kulfan.lower_weights = [-0.12, -0.05, -0.13, 0.02, -0.08, 0.06, 0.02, 0.06];
//! input.kulfan.te_thickness = 0.0025;
//! input.alpha = 3.0;
//! input.re = 1e6;
//!
//! let model = NeuralFoil::new(ModelSize::XLarge)?;
//! let aero = model.analyze(&input);
//! let result = model.analyze_with_gradients(&input);
//! let dcl_dalpha = result.gradient.cl[input_index::ALPHA];
//! # Ok::<(), bora::Error>(())
//! ```

mod error;
mod kulfan;
mod network;
mod neuralfoil;
#[cfg(feature = "philote")]
pub mod philote;
mod types;

pub use error::{Error, Result};
pub use kulfan::{Point2, fit_kulfan_parameters, generate_kulfan_coordinates};
pub use network::{InputDistribution, Layer, Network};
pub use neuralfoil::NeuralFoil;
pub use types::*;

/// Returns the crate version string in "major.minor.patch" format.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
