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

use std::path::PathBuf;

/// Errors raised while loading model data or preparing an analysis.
///
/// Analysis calls themselves never fail.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An HDF5 file could not be opened or parsed.
    #[error("cannot open {what} HDF5 file {path}: {source}")]
    Open {
        what: &'static str,
        path: PathBuf,
        #[source]
        source: hdf5_pure::Error,
    },

    /// A weight or distribution file is missing data or has the wrong layout.
    #[error("{0}")]
    Format(String),

    /// The weight data directory could not be resolved.
    #[error(
        "cannot locate the Bora weight data directory; pass it explicitly or set the \
         BORA_DATA_DIR environment variable"
    )]
    DataDirNotFound,

    /// Loaded components have inconsistent dimensions.
    #[error("{0} dimension mismatch")]
    DimensionMismatch(&'static str),

    /// An argument is outside its valid range.
    #[error("{0}")]
    InvalidArgument(String),

    /// The Kulfan least-squares fit problem is singular.
    #[error("Kulfan fit problem is singular")]
    SingularFit,
}

/// Result alias for fallible Bora operations.
pub type Result<T> = std::result::Result<T, Error>;
