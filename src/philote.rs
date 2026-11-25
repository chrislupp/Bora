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
    AeroInput, AeroScalars, BoundaryLayerData, BoundaryLayerGradients, CorrectedInput, Gradient,
    ModelSize, NUM_BL_POINTS, NUM_CST_WEIGHTS_PER_SIDE, NeuralFoil, input_index,
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
///
/// Partials are row-major `[output_size x input_size]` blocks, with singleton
/// dimensions dropped. The boundary layer does not depend on `mach`, so no
/// partials are declared for that pair.
pub struct BoraDiscipline {
    registry: VariableRegistry,
    model: OnceLock<NeuralFoil>,
    model_size: ModelSize,
    data_directory: Option<PathBuf>,
    include_boundary_layer: bool,
    include_corrections: bool,
    include_360_deg_effects: bool,
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

    fn read_corrected_input(&self, inputs: &ArrayMap) -> Result<CorrectedInput> {
        Ok(CorrectedInput {
            aero: read_aero_input(inputs)?,
            mach: read_scalar(inputs, "mach")?,
            control_surfaces: Vec::new(),
            include_360_deg_effects: self.include_360_deg_effects,
        })
    }
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
        self.add_option("include_360_deg_effects", "bool")
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
            let values = [
                ("CL", aero.cl),
                ("CD", aero.cd),
                ("CM", aero.cm),
                ("Top_Xtr", aero.top_xtr),
                ("Bot_Xtr", aero.bot_xtr),
                ("analysis_confidence", aero.analysis_confidence),
                ("Cpmin", aero.cpmin),
                ("Cpmin_0", aero.cpmin_0),
                ("mach_crit", aero.mach_crit),
                ("mach_dd", aero.mach_dd),
            ];
            for (name, value) in values {
                outputs.insert(name.to_string(), array(&[1], vec![value]));
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
