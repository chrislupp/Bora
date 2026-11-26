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

//! Philote explicit discipline exposing NeuralFoil analyses over gRPC.
//!
//! Enabled with the `philote` Cargo feature.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

use async_trait::async_trait;
use ndarray::{ArrayD, IxDyn};
use philote_mdo::{
    ArrayMap, Discipline, ExplicitDiscipline, PartialMap, PhiloteError, Result, VariableRegistry,
    impl_registry,
};

use crate::{
    AeroInput, AeroScalars, BoundaryLayerData, BoundaryLayerGradients, ControlSurface,
    CorrectedInput, CorrectedOutput, Gradient, ModelSize, NUM_BL_POINTS, NUM_CST_WEIGHTS_PER_SIDE,
    NeuralFoil, input_index,
};

/// One named discipline input and its slice of the gradient vector.
struct InputBlock {
    name: &'static str,
    offset: usize,
    size: usize,
}

const INPUT_BLOCKS: [InputBlock; 9] = [
    InputBlock {
        name: "upper_weights",
        offset: input_index::UPPER_WEIGHTS,
        size: NUM_CST_WEIGHTS_PER_SIDE,
    },
    InputBlock {
        name: "lower_weights",
        offset: input_index::LOWER_WEIGHTS,
        size: NUM_CST_WEIGHTS_PER_SIDE,
    },
    InputBlock {
        name: "leading_edge_weight",
        offset: input_index::LEADING_EDGE_WEIGHT,
        size: 1,
    },
    InputBlock {
        name: "TE_thickness",
        offset: input_index::TE_THICKNESS,
        size: 1,
    },
    InputBlock {
        name: "alpha",
        offset: input_index::ALPHA,
        size: 1,
    },
    InputBlock {
        name: "Re",
        offset: input_index::RE,
        size: 1,
    },
    InputBlock {
        name: "n_crit",
        offset: input_index::N_CRIT,
        size: 1,
    },
    InputBlock {
        name: "xtr_upper",
        offset: input_index::XTR_UPPER,
        size: 1,
    },
    InputBlock {
        name: "xtr_lower",
        offset: input_index::XTR_LOWER,
        size: 1,
    },
];

/// The Mach number input, present with the `include_corrections` option.
const MACH_BLOCK: InputBlock = InputBlock {
    name: "mach",
    offset: input_index::MACH,
    size: 1,
};

/// The option holding the hinge locations of the control surfaces.
const HINGE_POINTS_OPTION: &str = "control_surface_hinge_points";

/// The input holding the control surface deflections, present when the
/// `control_surface_hinge_points` option names at least one surface.
const DEFLECTIONS: &str = "control_surface_deflections";

/// Outputs that exist only with the `include_corrections` option.
const CORRECTION_OUTPUTS: [&str; 4] = ["Cpmin", "Cpmin_0", "mach_crit", "mach_dd"];

const SCALAR_OUTPUTS: [&str; 6] = [
    "CL",
    "CD",
    "CM",
    "Top_Xtr",
    "Bot_Xtr",
    "analysis_confidence",
];

/// Reads one scalar field of a corrected output.
type CorrectedField = fn(&CorrectedOutput) -> f64;

/// The scalar outputs of a corrected analysis, by variable name.
const CORRECTED_SCALARS: [(&str, CorrectedField); 10] = [
    ("CL", |o| o.cl),
    ("CD", |o| o.cd),
    ("CM", |o| o.cm),
    ("Top_Xtr", |o| o.top_xtr),
    ("Bot_Xtr", |o| o.bot_xtr),
    ("analysis_confidence", |o| o.analysis_confidence),
    ("Cpmin", |o| o.cpmin),
    ("Cpmin_0", |o| o.cpmin_0),
    ("mach_crit", |o| o.mach_crit),
    ("mach_dd", |o| o.mach_dd),
];

const BL_OUTPUTS: [&str; 6] = [
    "upper_theta",
    "upper_H",
    "upper_ue_over_vinf",
    "lower_theta",
    "lower_H",
    "lower_ue_over_vinf",
];

/// Philote explicit discipline exposing NeuralFoil analyses over gRPC.
///
/// Wraps [`NeuralFoil`] as a Philote [`ExplicitDiscipline`] so that any Philote
/// client (e.g. an OpenMDAO/Philote-Python component) can evaluate airfoil
/// aerodynamics and their analytic gradients remotely.
///
/// Inputs (all differentiable):
///   - `upper_weights` \[8\], `lower_weights` \[8\]: CST weights, LE to TE
///   - `leading_edge_weight`, `TE_thickness`: Kulfan shape modifiers
///   - `alpha` \[deg\], `Re`, `n_crit`, `xtr_upper`, `xtr_lower`: operating
///     condition
///   - `mach` (with the `include_corrections` option): Mach number
///   - `control_surface_deflections` \[n\] \[deg\] (with the
///     `control_surface_hinge_points` option): deflection of each control
///     surface, positive trailing edge down
///
/// Outputs: `CL`, `CD`, `CM`, `Top_Xtr`, `Bot_Xtr`, `analysis_confidence`, and
/// (when enabled via the `include_boundary_layer` option) the boundary layer
/// distributions `upper_theta`, `upper_H`, `upper_ue_over_vinf`,
/// `lower_theta`, `lower_H`, `lower_ue_over_vinf` at [`NUM_BL_POINTS`]
/// stations each. With `include_corrections`, the coefficients and transition
/// locations are those of
/// [`NeuralFoil::analyze_corrected`], and `Cpmin`, `Cpmin_0`, `mach_crit` and
/// `mach_dd` are added.
///
/// Options:
///   - `model_size` (str): network size, one of "xxsmall", "xsmall",
///     "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge" (default
///     "xlarge")
///   - `data_directory` (str): directory containing the HDF5 (`.h5`) weight
///     files; empty uses the standard resolution order (`BORA_DATA_DIR`, then
///     the source-tree `data/` directory)
///   - `include_boundary_layer` (bool): expose boundary layer distribution
///     outputs and their partials (default false)
///   - `include_corrections` (bool): add the `mach` input and apply the
///     compressibility corrections (default false)
///   - `include_360_deg_effects` (bool): with `include_corrections`, also
///     blend into the post-stall model at large angles of attack (default
///     true)
///   - `control_surface_hinge_points` (str): with `include_corrections`, the
///     hinge locations (x/c, each in \[0, 1\]) of the control surfaces on the
///     section, separated by commas or spaces, e.g. `"0.75"` or `"0.7, 0.85"`;
///     one `control_surface_deflections` entry per hinge point, in this order.
///     A list of numbers is accepted as well. Empty (the default) means no
///     control surfaces
///
/// Partials are row-major `[output_size x input_size]` blocks, with singleton
/// dimensions dropped. The boundary layer does not depend on `mach`, so no
/// partials are declared for that pair; it does depend on the deflections,
/// through the angle of attack shift.
pub struct BoraDiscipline {
    registry: VariableRegistry,
    model: OnceLock<NeuralFoil>,
    model_size: ModelSize,
    data_directory: Option<PathBuf>,
    include_boundary_layer: bool,
    include_corrections: bool,
    include_360_deg_effects: bool,
    hinge_points: Vec<f64>,
}

impl Default for BoraDiscipline {
    fn default() -> Self {
        Self {
            registry: VariableRegistry::default(),
            model: OnceLock::new(),
            model_size: ModelSize::default(),
            data_directory: None,
            include_boundary_layer: false,
            include_corrections: false,
            include_360_deg_effects: true,
            hinge_points: Vec::new(),
        }
    }
}

fn config_error(err: crate::Error) -> PhiloteError {
    PhiloteError::config_error(err.to_string())
}

impl BoraDiscipline {
    /// Creates a discipline with default options.
    pub fn new() -> Self {
        Self::default()
    }

    fn load_model(&self) -> Result<NeuralFoil> {
        match &self.data_directory {
            Some(dir) => NeuralFoil::with_data_dir(self.model_size, dir),
            None => NeuralFoil::new(self.model_size),
        }
        .map_err(config_error)
    }

    /// Returns the analysis model, constructing it with the current option
    /// values on first use.
    fn model(&self) -> Result<&NeuralFoil> {
        if let Some(model) = self.model.get() {
            return Ok(model);
        }
        let model = self.load_model()?;
        Ok(self.model.get_or_init(|| model))
    }

    /// The scalar outputs present with the current options.
    fn scalar_outputs(&self) -> impl Iterator<Item = &'static str> + use<> {
        let corrections: &[&str] = if self.include_corrections {
            &CORRECTION_OUTPUTS
        } else {
            &[]
        };
        SCALAR_OUTPUTS
            .into_iter()
            .chain(corrections.iter().copied())
    }

    /// The inputs that scalar outputs are differentiated with respect to.
    fn input_blocks(&self) -> impl Iterator<Item = &'static InputBlock> + use<> {
        INPUT_BLOCKS
            .iter()
            .chain(self.include_corrections.then_some(&MACH_BLOCK))
    }

    /// Whether the `control_surface_deflections` input exists.
    fn has_control_surfaces(&self) -> bool {
        !self.hinge_points.is_empty()
    }

    fn read_corrected_input(&self, inputs: &ArrayMap) -> Result<CorrectedInput> {
        let control_surfaces = if self.has_control_surfaces() {
            let deflections = read_input(inputs, DEFLECTIONS, self.hinge_points.len())?;
            self.hinge_points
                .iter()
                .zip(deflections)
                .map(|(&hinge_point, deflection)| ControlSurface {
                    deflection,
                    hinge_point,
                })
                .collect()
        } else {
            Vec::new()
        };
        Ok(CorrectedInput {
            aero: read_aero_input(inputs)?,
            mach: read_scalar(inputs, "mach")?,
            control_surfaces,
            include_360_deg_effects: self.include_360_deg_effects,
        })
    }
}

/// Reads the `control_surface_hinge_points` option: a string of numbers
/// separated by commas or whitespace, a list of numbers, or one number.
fn parse_hinge_points(value: &serde_json::Value) -> Result<Vec<f64>> {
    let invalid = || {
        PhiloteError::config_error(format!(
            "{HINGE_POINTS_OPTION}: expected hinge locations x/c in [0, 1], separated by \
             commas or spaces, got {value}"
        ))
    };
    let points = match value {
        serde_json::Value::Null => Vec::new(),
        serde_json::Value::Number(number) => vec![number.as_f64().ok_or_else(invalid)?],
        serde_json::Value::String(text) => text
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|token| !token.is_empty())
            .map(|token| token.parse::<f64>().map_err(|_| invalid()))
            .collect::<Result<_>>()?,
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| item.as_f64().ok_or_else(invalid))
            .collect::<Result<_>>()?,
        _ => return Err(invalid()),
    };
    if points.iter().any(|h| !(0.0..=1.0).contains(h)) {
        return Err(invalid());
    }
    Ok(points)
}

fn read_input(inputs: &ArrayMap, name: &str, size: usize) -> Result<Vec<f64>> {
    let array = inputs
        .get(name)
        .ok_or_else(|| PhiloteError::validation(name, "missing input"))?;
    if array.len() != size {
        return Err(PhiloteError::validation(
            name,
            format!("expected {size} entries, got {}", array.len()),
        ));
    }
    Ok(array.iter().copied().collect())
}

fn read_scalar(inputs: &ArrayMap, name: &str) -> Result<f64> {
    Ok(read_input(inputs, name, 1)?[0])
}

fn read_aero_input(inputs: &ArrayMap) -> Result<AeroInput> {
    let mut input = AeroInput::default();
    let upper = read_input(inputs, "upper_weights", NUM_CST_WEIGHTS_PER_SIDE)?;
    let lower = read_input(inputs, "lower_weights", NUM_CST_WEIGHTS_PER_SIDE)?;
    input.kulfan.upper_weights.copy_from_slice(&upper);
    input.kulfan.lower_weights.copy_from_slice(&lower);
    input.kulfan.leading_edge_weight = read_scalar(inputs, "leading_edge_weight")?;
    input.kulfan.te_thickness = read_scalar(inputs, "TE_thickness")?;
    input.alpha = read_scalar(inputs, "alpha")?;
    input.re = read_scalar(inputs, "Re")?;
    input.n_crit = read_scalar(inputs, "n_crit")?;
    input.xtr_upper = read_scalar(inputs, "xtr_upper")?;
    input.xtr_lower = read_scalar(inputs, "xtr_lower")?;
    Ok(input)
}

fn array(shape: &[usize], values: Vec<f64>) -> ArrayD<f64> {
    ArrayD::from_shape_vec(IxDyn(shape), values).expect("shape matches value count")
}

/// Adds the boundary layer distributions of both surfaces to the outputs.
fn insert_boundary_layer(
    outputs: &mut ArrayMap,
    upper: &BoundaryLayerData,
    lower: &BoundaryLayerData,
) {
    let distributions = [
        ("upper_theta", &upper.theta),
        ("upper_H", &upper.h),
        ("upper_ue_over_vinf", &upper.ue_over_vinf),
        ("lower_theta", &lower.theta),
        ("lower_H", &lower.h),
        ("lower_ue_over_vinf", &lower.ue_over_vinf),
    ];
    for (name, values) in distributions {
        outputs.insert(name.to_string(), array(&[NUM_BL_POINTS], values.to_vec()));
    }
}

/// Adds the partials of all six boundary layer distributions.
fn insert_all_bl_partials(
    partials: &mut PartialMap,
    upper: &BoundaryLayerGradients,
    lower: &BoundaryLayerGradients,
) {
    insert_bl_partials(partials, "upper_theta", &upper.theta);
    insert_bl_partials(partials, "upper_H", &upper.h);
    insert_bl_partials(partials, "upper_ue_over_vinf", &upper.ue_over_vinf);
    insert_bl_partials(partials, "lower_theta", &lower.theta);
    insert_bl_partials(partials, "lower_H", &lower.h);
    insert_bl_partials(partials, "lower_ue_over_vinf", &lower.ue_over_vinf);
}

/// Copies one scalar output's gradient into its per-input jacobian blocks.
fn insert_scalar_partials(
    partials: &mut PartialMap,
    output: &str,
    gradient: &[f64],
    blocks: impl Iterator<Item = &'static InputBlock>,
) {
    for block in blocks {
        let values = gradient[block.offset..block.offset + block.size].to_vec();
        partials.insert(
            (output.to_string(), block.name.to_string()),
            array(&[block.size], values),
        );
    }
}

/// Copies a boundary layer distribution's gradients into its jacobian blocks,
/// each row-major `[NUM_BL_POINTS x block.size]`.
fn insert_bl_partials(
    partials: &mut PartialMap,
    output: &str,
    gradients: &[Gradient; NUM_BL_POINTS],
) {
    for block in &INPUT_BLOCKS {
        let values = gradients
            .iter()
            .flat_map(|row| row[block.offset..block.offset + block.size].iter().copied())
            .collect();
        let shape: &[usize] = if block.size == 1 {
            &[NUM_BL_POINTS]
        } else {
            &[NUM_BL_POINTS, block.size]
        };
        partials.insert(
            (output.to_string(), block.name.to_string()),
            array(shape, values),
        );
    }
}

/// The boundary layer distribution of one output, by variable name.
fn distribution<'a>(output: &'a CorrectedOutput, name: &str) -> &'a [f64; NUM_BL_POINTS] {
    match name {
        "upper_theta" => &output.upper.theta,
        "upper_H" => &output.upper.h,
        "upper_ue_over_vinf" => &output.upper.ue_over_vinf,
        "lower_theta" => &output.lower.theta,
        "lower_H" => &output.lower.h,
        "lower_ue_over_vinf" => &output.lower.ue_over_vinf,
        _ => unreachable!("not a boundary layer output: {name}"),
    }
}

/// Adds the partials with respect to the control surface deflections, given
/// the derivatives of every output with respect to each deflection: shape
/// `[n]` for a scalar output and `[NUM_BL_POINTS x n]` for a boundary layer
/// distribution, with singleton dimensions dropped.
fn insert_deflection_partials(
    partials: &mut PartialMap,
    deflection: &[CorrectedOutput],
    include_boundary_layer: bool,
) {
    let n = deflection.len();
    for (name, value) in CORRECTED_SCALARS {
        partials.insert(
            (name.to_string(), DEFLECTIONS.to_string()),
            array(&[n], deflection.iter().map(value).collect()),
        );
    }
    if !include_boundary_layer {
        return;
    }
    let shape: &[usize] = if n == 1 {
        &[NUM_BL_POINTS]
    } else {
        &[NUM_BL_POINTS, n]
    };
    for name in BL_OUTPUTS {
        let values = (0..NUM_BL_POINTS)
            .flat_map(|i| deflection.iter().map(move |d| distribution(d, name)[i]))
            .collect();
        partials.insert(
            (name.to_string(), DEFLECTIONS.to_string()),
            array(shape, values),
        );
    }
}

impl Discipline for BoraDiscipline {
    impl_registry!(registry);

    fn name(&self) -> &str {
        "Bora"
    }

    fn version(&self) -> &str {
        crate::version()
    }

    fn is_differentiable(&self) -> bool {
        true
    }

    fn provides_gradients(&self) -> bool {
        true
    }

    fn initialize(&mut self) -> Result<()> {
        self.add_option("model_size", "str")?;
        self.add_option("data_directory", "str")?;
        self.add_option("include_boundary_layer", "bool")?;
        self.add_option("include_corrections", "bool")?;
        self.add_option("include_360_deg_effects", "bool")?;
        self.add_option(HINGE_POINTS_OPTION, "str")
    }

    fn set_options(&mut self, options: &HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(name) = options.get("model_size").and_then(|v| v.as_str()) {
            self.model_size = name.parse().map_err(config_error)?;
        }
        if let Some(dir) = options.get("data_directory").and_then(|v| v.as_str()) {
            self.data_directory = (!dir.is_empty()).then(|| PathBuf::from(dir));
        }
        if let Some(value) = options.get("include_boundary_layer") {
            self.include_boundary_layer = value.as_bool().unwrap_or(false);
        }
        if let Some(value) = options.get("include_corrections") {
            self.include_corrections = value.as_bool().unwrap_or(false);
        }
        if let Some(value) = options.get("include_360_deg_effects") {
            self.include_360_deg_effects = value.as_bool().unwrap_or(true);
        }
        let hinge_points = match options.get(HINGE_POINTS_OPTION) {
            Some(value) => parse_hinge_points(value)?,
            None => self.hinge_points.clone(),
        };
        if !hinge_points.is_empty() && !self.include_corrections {
            return Err(PhiloteError::config_error(format!(
                "{HINGE_POINTS_OPTION} requires include_corrections"
            )));
        }
        self.hinge_points = hinge_points;

        // Rebuild the model eagerly so bad options fail the SetOptions RPC
        // rather than the first compute.
        self.model = OnceLock::from(self.load_model()?);
        Ok(())
    }

    fn setup(&mut self) -> Result<()> {
        self.add_input("upper_weights", &[NUM_CST_WEIGHTS_PER_SIDE], "")?;
        self.add_input("lower_weights", &[NUM_CST_WEIGHTS_PER_SIDE], "")?;
        self.add_input("leading_edge_weight", &[1], "")?;
        self.add_input("TE_thickness", &[1], "")?;
        self.add_input("alpha", &[1], "deg")?;
        self.add_input("Re", &[1], "")?;
        self.add_input("n_crit", &[1], "")?;
        self.add_input("xtr_upper", &[1], "")?;
        self.add_input("xtr_lower", &[1], "")?;
        if self.include_corrections {
            self.add_input(MACH_BLOCK.name, &[1], "")?;
        }
        if self.has_control_surfaces() {
            self.add_input(DEFLECTIONS, &[self.hinge_points.len()], "deg")?;
        }

        for name in self.scalar_outputs() {
            self.add_output(name, &[1], "")?;
        }
        if self.include_boundary_layer {
            for name in BL_OUTPUTS {
                self.add_output(name, &[NUM_BL_POINTS], "")?;
            }
        }
        Ok(())
    }

    fn setup_partials(&mut self) -> Result<()> {
        let bl_outputs: &[&str] = if self.include_boundary_layer {
            &BL_OUTPUTS
        } else {
            &[]
        };
        for output in self.scalar_outputs() {
            for block in self.input_blocks() {
                self.declare_partials(output, block.name)?;
            }
        }
        for output in bl_outputs {
            for block in &INPUT_BLOCKS {
                self.declare_partials(output, block.name)?;
            }
        }
        if self.has_control_surfaces() {
            let outputs: Vec<_> = self
                .scalar_outputs()
                .chain(bl_outputs.iter().copied())
                .collect();
            for output in outputs {
                self.declare_partials(output, DEFLECTIONS)?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl ExplicitDiscipline for BoraDiscipline {
    async fn compute(&self, inputs: &ArrayMap) -> Result<ArrayMap> {
        let model = self.model()?;
        let mut outputs = HashMap::new();
        if self.include_corrections {
            let aero = model.analyze_corrected(&self.read_corrected_input(inputs)?);
            for (name, value) in CORRECTED_SCALARS {
                outputs.insert(name.to_string(), array(&[1], vec![value(&aero)]));
            }
            if self.include_boundary_layer {
                insert_boundary_layer(&mut outputs, &aero.upper, &aero.lower);
            }
            return Ok(outputs);
        }

        let input = read_aero_input(inputs)?;

        // Without boundary layer outputs, the scalar path skips decoding them.
        let (scalars, boundary_layer) = if self.include_boundary_layer {
            let aero = model.analyze(&input);
            let scalars = AeroScalars {
                analysis_confidence: aero.analysis_confidence,
                cl: aero.cl,
                cd: aero.cd,
                cm: aero.cm,
                top_xtr: aero.top_xtr,
                bot_xtr: aero.bot_xtr,
            };
            (scalars, Some((aero.upper, aero.lower)))
        } else {
            (model.analyze_scalars(&input), None)
        };

        let values = [
            ("CL", scalars.cl),
            ("CD", scalars.cd),
            ("CM", scalars.cm),
            ("Top_Xtr", scalars.top_xtr),
            ("Bot_Xtr", scalars.bot_xtr),
            ("analysis_confidence", scalars.analysis_confidence),
        ];
        for (name, value) in values {
            outputs.insert(name.to_string(), array(&[1], vec![value]));
        }

        if let Some((upper, lower)) = boundary_layer {
            insert_boundary_layer(&mut outputs, &upper, &lower);
        }
        Ok(outputs)
    }

    async fn compute_partials(&self, inputs: &ArrayMap) -> Result<PartialMap> {
        let model = self.model()?;
        let mut partials = HashMap::new();

        if self.include_corrections {
            let input = self.read_corrected_input(inputs)?;
            let grad = model.analyze_corrected_with_gradients(&input).gradient;
            let scalars = [
                ("CL", &grad.cl),
                ("CD", &grad.cd),
                ("CM", &grad.cm),
                ("Top_Xtr", &grad.top_xtr),
                ("Bot_Xtr", &grad.bot_xtr),
                ("analysis_confidence", &grad.analysis_confidence),
                ("Cpmin", &grad.cpmin),
                ("Cpmin_0", &grad.cpmin_0),
                ("mach_crit", &grad.mach_crit),
                ("mach_dd", &grad.mach_dd),
            ];
            for (name, gradient) in scalars {
                insert_scalar_partials(&mut partials, name, gradient, self.input_blocks());
            }
            if self.include_boundary_layer {
                insert_all_bl_partials(&mut partials, &grad.upper, &grad.lower);
            }
            if self.has_control_surfaces() {
                insert_deflection_partials(
                    &mut partials,
                    &grad.deflection,
                    self.include_boundary_layer,
                );
            }
            return Ok(partials);
        }

        let input = read_aero_input(inputs)?;

        // Without boundary layer outputs only the six scalar gradients are
        // needed, which the reverse-mode path computes several times faster.
        let (scalars, boundary_layer) = if self.include_boundary_layer {
            let grad = model.analyze_with_gradients(&input).gradient;
            let scalars = [
                grad.cl,
                grad.cd,
                grad.cm,
                grad.top_xtr,
                grad.bot_xtr,
                grad.analysis_confidence,
            ];
            (scalars, Some((grad.upper, grad.lower)))
        } else {
            let grad = model.analyze_scalars_with_gradients(&input).gradient;
            let scalars = [
                grad.cl,
                grad.cd,
                grad.cm,
                grad.top_xtr,
                grad.bot_xtr,
                grad.analysis_confidence,
            ];
            (scalars, None)
        };
        for (name, gradient) in SCALAR_OUTPUTS.iter().zip(&scalars) {
            insert_scalar_partials(&mut partials, name, gradient, self.input_blocks());
        }
        if let Some((upper, lower)) = boundary_layer {
            insert_all_bl_partials(&mut partials, &upper, &lower);
        }
        Ok(partials)
    }
}
