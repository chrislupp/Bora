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

//! Shared loaders for the HDF5 reference data in tests/data/ (generated from
//! Python NeuralFoil; see scripts/generate_output_ref_data.py and
//! scripts/generate_gradient_ref_data.py). The files are committed to the
//! repository and only regenerated when the reference data is updated.

#![allow(dead_code)]

use std::fmt::Write as _;
use std::sync::OnceLock;

use bora::{AeroInput, ModelSize, NUM_CST_WEIGHTS_PER_SIDE, NUM_INPUTS, NeuralFoil, input_index};
use hdf5_pure::File;

pub const TEST_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data");
pub const WEIGHTS_DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data");

/// Reference outputs loaded from output_ref_data.h5.
pub struct OutputRefData {
    pub names: Vec<String>,
    pub model_sizes: Vec<String>,
    /// Row-major [n_cases x NUM_INPUTS].
    pub inputs: Vec<f64>,
    /// Row-major [n_cases x n_outputs].
    pub outputs: Vec<f64>,
    pub n_cases: usize,
    pub n_outputs: usize,
}

/// Reference jacobians loaded from gradient_ref_data.h5.
pub struct GradientRefData {
    pub names: Vec<String>,
    pub model_sizes: Vec<String>,
    pub bl_stations: Vec<usize>,
    /// Row-major [n_cases x NUM_INPUTS].
    pub inputs: Vec<f64>,
    /// Row-major [n_cases x n_rows x NUM_INPUTS].
    pub jacobians: Vec<f64>,
    pub n_cases: usize,
    pub n_rows: usize,
}

fn open(name: &str) -> File {
    let path = format!("{TEST_DATA_DIR}/{name}");
    File::open(&path).unwrap_or_else(|err| panic!("cannot open {path}: {err}"))
}

fn read_strings(file: &File, name: &str) -> Vec<String> {
    file.dataset(name)
        .and_then(|dataset| dataset.read_string())
        .unwrap_or_else(|err| panic!("cannot read dataset {name}: {err}"))
}

fn read_doubles(file: &File, name: &str) -> (Vec<f64>, Vec<usize>) {
    let dataset = file
        .dataset(name)
        .unwrap_or_else(|err| panic!("missing dataset {name}: {err}"));
    let shape = dataset.shape().expect("dataset shape");
    let values = dataset.read_f64().expect("dataset values");
    (values, shape.into_iter().map(|d| d as usize).collect())
}

pub fn output_ref_data() -> &'static OutputRefData {
    static DATA: OnceLock<OutputRefData> = OnceLock::new();
    DATA.get_or_init(|| {
        let file = open("output_ref_data.h5");
        let (inputs, input_dims) = read_doubles(&file, "inputs");
        assert_eq!(input_dims.len(), 2);
        assert_eq!(input_dims[1], NUM_INPUTS);
        let (outputs, output_dims) = read_doubles(&file, "outputs");
        assert_eq!(output_dims.len(), 2);
        assert_eq!(output_dims[0], input_dims[0]);

        let data = OutputRefData {
            names: read_strings(&file, "names"),
            model_sizes: read_strings(&file, "model_sizes"),
            inputs,
            outputs,
            n_cases: input_dims[0],
            n_outputs: output_dims[1],
        };
        assert_eq!(data.names.len(), data.n_cases);
        assert_eq!(data.model_sizes.len(), data.n_cases);
        data
    })
}

pub fn gradient_ref_data() -> GradientRefData {
    let file = open("gradient_ref_data.h5");
    let (stations, _) = read_doubles(&file, "bl_stations");
    let (inputs, input_dims) = read_doubles(&file, "inputs");
    assert_eq!(input_dims.len(), 2);
    assert_eq!(input_dims[1], NUM_INPUTS);
    let (jacobians, jac_dims) = read_doubles(&file, "jacobians");
    assert_eq!(jac_dims.len(), 3);
    assert_eq!(jac_dims[0], input_dims[0]);
    assert_eq!(jac_dims[2], NUM_INPUTS);

    let data = GradientRefData {
        names: read_strings(&file, "names"),
        model_sizes: read_strings(&file, "model_sizes"),
        bl_stations: stations.into_iter().map(|s| s as usize).collect(),
        inputs,
        jacobians,
        n_cases: input_dims[0],
        n_rows: jac_dims[1],
    };
    assert_eq!(data.names.len(), data.n_cases);
    assert_eq!(data.model_sizes.len(), data.n_cases);
    assert_eq!(data.n_rows, 6 + 2 * 3 * data.bl_stations.len());
    data
}

/// Returns a lazily constructed, cached model for a size name ("xlarge", ...).
pub fn model(size_name: &str) -> &'static NeuralFoil {
    static MODELS: [OnceLock<NeuralFoil>; 8] = [const { OnceLock::new() }; 8];
    let size: ModelSize = size_name.parse().expect("known model size");
    let slot = ModelSize::ALL.iter().position(|s| *s == size).unwrap();
    MODELS[slot].get_or_init(|| NeuralFoil::new(size).expect("model weights load"))
}

/// Builds an AeroInput from a NUM_INPUTS-long row of a reference inputs array.
pub fn input_from_array(values: &[f64]) -> AeroInput {
    let w = NUM_CST_WEIGHTS_PER_SIDE;
    let mut input = AeroInput::default();
    input
        .kulfan
        .upper_weights
        .copy_from_slice(&values[input_index::UPPER_WEIGHTS..][..w]);
    input
        .kulfan
        .lower_weights
        .copy_from_slice(&values[input_index::LOWER_WEIGHTS..][..w]);
    input.kulfan.leading_edge_weight = values[input_index::LEADING_EDGE_WEIGHT];
    input.kulfan.te_thickness = values[input_index::TE_THICKNESS];
    input.alpha = values[input_index::ALPHA];
    input.re = values[input_index::RE];
    input.n_crit = values[input_index::N_CRIT];
    input.xtr_upper = values[input_index::XTR_UPPER];
    input.xtr_lower = values[input_index::XTR_LOWER];
    input
}

/// Collects non-fatal comparison failures so a test reports every mismatch
/// rather than stopping at the first.
#[derive(Default)]
pub struct Failures {
    messages: Vec<String>,
}

impl Failures {
    pub fn expect_near(
        &mut self,
        actual: f64,
        expected: f64,
        tolerance: f64,
        context: impl FnOnce() -> String,
    ) {
        let within = (actual - expected).abs() <= tolerance; // false for NaN
        if !within {
            self.messages.push(format!(
                "{}: actual {actual:e}, expected {expected:e} (tolerance {tolerance:e})",
                context()
            ));
        }
    }

    pub fn assert_none(self) {
        if self.messages.is_empty() {
            return;
        }
        let mut report = format!("{} mismatches:\n", self.messages.len());
        for message in self.messages.iter().take(50) {
            let _ = writeln!(report, "  {message}");
        }
        panic!("{report}");
    }
}
