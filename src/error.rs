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
