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
//! When only the scalar outputs (coefficients, transition locations and
//! confidence) are needed, [`NeuralFoil::analyze_scalars`] and
//! [`NeuralFoil::analyze_scalars_with_gradients`] skip the boundary layer; the
//! latter differentiates in reverse mode and is several times faster than
//! [`NeuralFoil::analyze_with_gradients`].
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
//!
//! // Scalar outputs only, with gradients:
//! let scalars = model.analyze_scalars_with_gradients(&input);
//! let dcd_dalpha = scalars.gradient.cd[input_index::ALPHA];
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
