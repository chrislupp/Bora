use std::fmt;
use std::str::FromStr;

use crate::Error;

/// Number of CST (Kulfan) shape weights per airfoil surface.
pub const NUM_CST_WEIGHTS_PER_SIDE: usize = 8;

/// Number of boundary layer evaluation points per surface (NeuralFoil's
/// `Data.N`).
pub const NUM_BL_POINTS: usize = 32;

/// Dimension of the network input latent space.
pub const NUM_LATENT_INPUTS: usize = 25;

/// Dimension of the network output latent space (6 scalars + 6 boundary layer
/// distributions of [`NUM_BL_POINTS`] each).
pub const NUM_LATENT_OUTPUTS: usize = 6 + 6 * NUM_BL_POINTS;

/// Number of differentiable user-facing inputs: 16 CST weights, leading edge
/// weight, trailing edge thickness, alpha, Re, n_crit, xtr_upper, xtr_lower.
pub const NUM_INPUTS: usize = 23;

/// Indices of the differentiable inputs within a [`Gradient`] array.
pub mod input_index {
    /// 8 entries: 0..=7.
    pub const UPPER_WEIGHTS: usize = 0;
    /// 8 entries: 8..=15.
    pub const LOWER_WEIGHTS: usize = 8;
    pub const LEADING_EDGE_WEIGHT: usize = 16;
    pub const TE_THICKNESS: usize = 17;
    /// Angle of attack (deg).
    pub const ALPHA: usize = 18;
    /// Reynolds number.
    pub const RE: usize = 19;
    pub const N_CRIT: usize = 20;
    pub const XTR_UPPER: usize = 21;
    pub const XTR_LOWER: usize = 22;
}

/// Gradient of a scalar output with respect to the [`NUM_INPUTS`] user inputs.
/// Component ordering is given by the constants in [`input_index`].
pub type Gradient = [f64; NUM_INPUTS];

/// Kulfan (CST) parameterization of an airfoil shape.
///
/// Uses the "CST with leading-edge modification" flavor with 8 weights per
/// side, class function exponents N1=0.5, N2=1.0. Weight ordering is from the
/// leading edge to the trailing edge.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KulfanParameters {
    pub upper_weights: [f64; NUM_CST_WEIGHTS_PER_SIDE],
    pub lower_weights: [f64; NUM_CST_WEIGHTS_PER_SIDE],
    pub leading_edge_weight: f64,
    /// Trailing edge thickness, in chord units.
    pub te_thickness: f64,
}

/// Inputs describing a single airfoil analysis case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AeroInput {
    pub kulfan: KulfanParameters,
    /// Angle of attack (deg).
    pub alpha: f64,
    /// Chord-based Reynolds number.
    pub re: f64,
    /// Critical amplification factor for transition.
    pub n_crit: f64,
    /// Forced transition location, upper surface (x/c).
    pub xtr_upper: f64,
    /// Forced transition location, lower surface (x/c).
    pub xtr_lower: f64,
}

impl Default for AeroInput {
    fn default() -> Self {
        Self {
            kulfan: KulfanParameters::default(),
            alpha: 0.0,
            re: 1e6,
            n_crit: 9.0,
            xtr_upper: 1.0,
            xtr_lower: 1.0,
        }
    }
}

/// Boundary layer quantities sampled at [`NUM_BL_POINTS`] stations along one
/// surface. Station x/c locations are given by [`bl_x_points`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryLayerData {
    /// Momentum thickness (chord-normalized).
    pub theta: [f64; NUM_BL_POINTS],
    /// Shape factor.
    pub h: [f64; NUM_BL_POINTS],
    /// Edge velocity ratio.
    pub ue_over_vinf: [f64; NUM_BL_POINTS],
}

impl Default for BoundaryLayerData {
    fn default() -> Self {
        Self {
            theta: [0.0; NUM_BL_POINTS],
            h: [0.0; NUM_BL_POINTS],
            ue_over_vinf: [0.0; NUM_BL_POINTS],
        }
    }
}

/// Aerodynamic analysis outputs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AeroOutput {
    /// Network confidence in [0, 1].
    pub analysis_confidence: f64,
    /// Lift coefficient.
    pub cl: f64,
    /// Drag coefficient.
    pub cd: f64,
    /// Quarter-chord moment coefficient.
    pub cm: f64,
    /// Upper surface transition location (x/c).
    pub top_xtr: f64,
    /// Lower surface transition location (x/c).
    pub bot_xtr: f64,
    pub upper: BoundaryLayerData,
    pub lower: BoundaryLayerData,
}

/// The scalar analysis outputs: the fields of [`AeroOutput`] without the
/// boundary layer distributions, as returned by
/// [`NeuralFoil::analyze_scalars`](crate::NeuralFoil::analyze_scalars).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AeroScalars {
    /// Network confidence in [0, 1].
    pub analysis_confidence: f64,
    /// Lift coefficient.
    pub cl: f64,
    /// Drag coefficient.
    pub cd: f64,
    /// Quarter-chord moment coefficient.
    pub cm: f64,
    /// Upper surface transition location (x/c).
    pub top_xtr: f64,
    /// Lower surface transition location (x/c).
    pub bot_xtr: f64,
}

/// Gradients of the boundary layer quantities on one surface.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryLayerGradients {
    pub theta: [Gradient; NUM_BL_POINTS],
    pub h: [Gradient; NUM_BL_POINTS],
    pub ue_over_vinf: [Gradient; NUM_BL_POINTS],
}

impl Default for BoundaryLayerGradients {
    fn default() -> Self {
        Self {
            theta: [[0.0; NUM_INPUTS]; NUM_BL_POINTS],
            h: [[0.0; NUM_INPUTS]; NUM_BL_POINTS],
            ue_over_vinf: [[0.0; NUM_INPUTS]; NUM_BL_POINTS],
        }
    }
}

/// Gradients of all analysis outputs with respect to the user inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct AeroGradients {
    pub analysis_confidence: Gradient,
    pub cl: Gradient,
    pub cd: Gradient,
    pub cm: Gradient,
    pub top_xtr: Gradient,
    pub bot_xtr: Gradient,
    pub upper: BoundaryLayerGradients,
    pub lower: BoundaryLayerGradients,
}

impl Default for AeroGradients {
    fn default() -> Self {
        Self {
            analysis_confidence: [0.0; NUM_INPUTS],
            cl: [0.0; NUM_INPUTS],
            cd: [0.0; NUM_INPUTS],
            cm: [0.0; NUM_INPUTS],
            top_xtr: [0.0; NUM_INPUTS],
            bot_xtr: [0.0; NUM_INPUTS],
            upper: BoundaryLayerGradients::default(),
            lower: BoundaryLayerGradients::default(),
        }
    }
}

/// Analysis outputs bundled with their gradients.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AeroResult {
    pub output: AeroOutput,
    pub gradient: AeroGradients,
}

/// Gradients of the scalar analysis outputs with respect to the user inputs:
/// the fields of [`AeroGradients`] without the boundary layer gradients.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AeroScalarGradients {
    pub analysis_confidence: Gradient,
    pub cl: Gradient,
    pub cd: Gradient,
    pub cm: Gradient,
    pub top_xtr: Gradient,
    pub bot_xtr: Gradient,
}

/// Scalar analysis outputs bundled with their gradients, as returned by
/// [`NeuralFoil::analyze_scalars_with_gradients`](crate::NeuralFoil::analyze_scalars_with_gradients).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AeroScalarResult {
    pub output: AeroScalars,
    pub gradient: AeroScalarGradients,
}

/// Available NeuralFoil network sizes (speed/accuracy tradeoff).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ModelSize {
    XXSmall,
    XSmall,
    Small,
    Medium,
    Large,
    /// Default in upstream NeuralFoil.
    #[default]
    XLarge,
    XXLarge,
    XXXLarge,
}

impl ModelSize {
    /// All model sizes, from smallest to largest.
    pub const ALL: [ModelSize; 8] = [
        ModelSize::XXSmall,
        ModelSize::XSmall,
        ModelSize::Small,
        ModelSize::Medium,
        ModelSize::Large,
        ModelSize::XLarge,
        ModelSize::XXLarge,
        ModelSize::XXXLarge,
    ];

    /// Returns the canonical lower-case name (e.g. "xlarge"), matching upstream
    /// NeuralFoil file naming.
    pub fn as_str(self) -> &'static str {
        match self {
            ModelSize::XXSmall => "xxsmall",
            ModelSize::XSmall => "xsmall",
            ModelSize::Small => "small",
            ModelSize::Medium => "medium",
            ModelSize::Large => "large",
            ModelSize::XLarge => "xlarge",
            ModelSize::XXLarge => "xxlarge",
            ModelSize::XXXLarge => "xxxlarge",
        }
    }
}

impl fmt::Display for ModelSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ModelSize {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        ModelSize::ALL
            .into_iter()
            .find(|size| size.as_str() == name)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "unknown model size '{name}' (expected one of xxsmall, xsmall, small, \
                     medium, large, xlarge, xxlarge, xxxlarge)"
                ))
            })
    }
}

/// Returns the x/c locations of the boundary layer evaluation points: the
/// midpoints of a uniform partition of [0, 1] into [`NUM_BL_POINTS`] cells.
pub fn bl_x_points() -> [f64; NUM_BL_POINTS] {
    std::array::from_fn(|i| (2.0 * i as f64 + 1.0) / (2.0 * NUM_BL_POINTS as f64))
}
