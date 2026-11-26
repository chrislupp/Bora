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

//! Airfoils given as coordinates: `.dat` files, normalization, and the
//! analysis of an airfoil that is not in the normalized frame the network
//! expects (NeuralFoil's `get_aero_from_airfoil`, `get_aero_from_coordinates`
//! and `get_aero_from_dat_file`).

use std::path::Path;

use crate::{
    AeroInput, AeroOutput, Error, KulfanParameters, NeuralFoil, OperatingPoint, Point2, Result,
    fit_kulfan_parameters,
};

/// Parses the text of an airfoil coordinate (`.dat`) file.
///
/// Every line that consists of exactly two numbers, separated by whitespace,
/// commas, semicolons or `|`, is a point; all other lines (the name, blank
/// lines, comments) are skipped. This is the rule of AeroSandbox's
/// `get_coordinates_from_raw_dat`. The points are returned in file order,
/// which should be the Selig format (upper surface trailing edge, around the
/// leading edge, to the lower surface trailing edge).
///
/// # Errors
///
/// [`Error::InvalidArgument`] if no line holds a point.
pub fn parse_dat(text: &str) -> Result<Vec<Point2>> {
    let is_separator = |c: char| c.is_whitespace() || matches!(c, ';' | ',' | '|');
    let parse_line = |line: &str| -> Option<Point2> {
        let mut items = line.split(is_separator).filter(|item| !item.is_empty());
        let x = items.next()?.parse().ok()?;
        let y = items.next()?.parse().ok()?;
        items.next().is_none().then_some([x, y])
    };
    let coordinates: Vec<Point2> = text.lines().filter_map(parse_line).collect();
    if coordinates.is_empty() {
        return Err(Error::InvalidArgument(
            "no coordinates found in the airfoil data".into(),
        ));
    }
    Ok(coordinates)
}

/// Reads an airfoil coordinate (`.dat`) file; see [`parse_dat`] for the
/// format.
///
/// # Errors
///
/// [`Error::Io`] if the file cannot be read, or [`Error::InvalidArgument`] if
/// it holds no coordinates.
pub fn read_dat_file(path: impl AsRef<Path>) -> Result<Vec<Point2>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    parse_dat(&text)
}

/// Airfoil coordinates brought to the normalized frame, with the transform
/// that was applied (translation, then scaling, then rotation about the
/// leading edge).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct NormalizedCoordinates {
    /// The coordinates with the leading edge at (0, 0) and the trailing edge
    /// at (1, 0).
    pub coordinates: Vec<Point2>,
    /// Translation in x that moved the leading edge to the origin.
    pub x_translation: f64,
    /// Translation in y that moved the leading edge to the origin.
    pub y_translation: f64,
    /// Scale factor that made the chord 1 (the reciprocal of the chord).
    pub scale_factor: f64,
    /// Counter-clockwise rotation (deg) that put the trailing edge on the
    /// x axis.
    pub rotation_angle: f64,
}

/// Normalizes Selig-format airfoil coordinates as AeroSandbox's
/// `Airfoil.normalize()` does: leading edge at (0, 0), trailing edge at
/// (1, 0), unit chord.
///
/// The trailing edge is the midpoint of the first and last points; the
/// leading edge is the point farthest from it.
///
/// # Errors
///
/// [`Error::InvalidArgument`] if there are fewer than three points or the
/// chord has no length.
pub fn normalize_coordinates(coordinates: &[Point2]) -> Result<NormalizedCoordinates> {
    let (Some(first), Some(last)) = (coordinates.first(), coordinates.last()) else {
        return Err(Error::InvalidArgument("no airfoil coordinates".into()));
    };
    if coordinates.len() < 3 {
        return Err(Error::InvalidArgument(
            "too few coordinates to normalize an airfoil".into(),
        ));
    }
    let trailing_edge = [(first[0] + last[0]) / 2.0, (first[1] + last[1]) / 2.0];
    let distance = |p: &Point2| (p[0] - trailing_edge[0]).hypot(p[1] - trailing_edge[1]);
    let mut leading_edge = coordinates[0];
    for point in coordinates {
        if distance(point) > distance(&leading_edge) {
            leading_edge = *point;
        }
    }
    let chord = distance(&leading_edge);
    if !(chord > 0.0 && chord.is_finite()) {
        return Err(Error::InvalidArgument(
            "airfoil coordinates have no chord".into(),
        ));
    }

    let scale_factor = 1.0 / chord;
    let rotation = -(trailing_edge[1] - leading_edge[1]).atan2(trailing_edge[0] - leading_edge[0]);
    let (sin, cos) = rotation.sin_cos();
    let coordinates = coordinates
        .iter()
        .map(|p| {
            let x = (p[0] - leading_edge[0]) * scale_factor;
            let y = (p[1] - leading_edge[1]) * scale_factor;
            [cos * x - sin * y, sin * x + cos * y]
        })
        .collect();
    Ok(NormalizedCoordinates {
        coordinates,
        x_translation: -leading_edge[0],
        y_translation: -leading_edge[1],
        scale_factor,
        rotation_angle: rotation.to_degrees(),
    })
}

/// An airfoil given as coordinates, fitted with Kulfan parameters in the
/// normalized frame, together with the transform back to its own frame.
///
/// NeuralFoil analyzes normalized airfoils (unit chord along the x axis). An
/// airfoil given with incidence, another chord or an offset is normalized
/// first; the analysis then runs at a shifted angle of attack and a scaled
/// Reynolds number, and the moment is moved back to the quarter-chord point
/// of the original airfoil, as in NeuralFoil's `get_aero_from_airfoil`.
///
/// ```no_run
/// use bora::{FittedAirfoil, ModelSize, NeuralFoil, OperatingPoint};
///
/// let airfoil = FittedAirfoil::from_dat_file("naca4412.dat")?;
/// let model = NeuralFoil::new(ModelSize::XLarge)?;
/// let aero = model.analyze_airfoil(
///     &airfoil,
///     &OperatingPoint {
///         alpha: 3.0,
///         re: 1e6,
///         ..OperatingPoint::default()
///     },
/// );
/// # Ok::<(), bora::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct FittedAirfoil {
    /// Kulfan parameters of the normalized airfoil.
    pub kulfan: KulfanParameters,
    /// See [`NormalizedCoordinates::x_translation`].
    pub x_translation: f64,
    /// See [`NormalizedCoordinates::y_translation`].
    pub y_translation: f64,
    /// See [`NormalizedCoordinates::scale_factor`].
    pub scale_factor: f64,
    /// See [`NormalizedCoordinates::rotation_angle`].
    pub rotation_angle: f64,
}

impl FittedAirfoil {
    /// Normalizes Selig-format coordinates ([`normalize_coordinates`]) and
    /// fits Kulfan parameters to them ([`fit_kulfan_parameters`]).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if there are too few coordinates or they
    /// have no chord, or [`Error::SingularFit`] if the fit is singular.
    pub fn from_coordinates(coordinates: &[Point2]) -> Result<Self> {
        let normalized = normalize_coordinates(coordinates)?;
        Ok(Self {
            kulfan: fit_kulfan_parameters(&normalized.coordinates)?,
            x_translation: normalized.x_translation,
            y_translation: normalized.y_translation,
            scale_factor: normalized.scale_factor,
            rotation_angle: normalized.rotation_angle,
        })
    }

    /// Reads a coordinate (`.dat`) file ([`read_dat_file`]) and fits it.
    ///
    /// # Errors
    ///
    /// Those of [`read_dat_file`] and [`from_coordinates`](Self::from_coordinates).
    pub fn from_dat_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_coordinates(&read_dat_file(path)?)
    }

    /// The network input for an operating point of the original airfoil: the
    /// normalized shape, the angle of attack shifted by the rotation, and the
    /// Reynolds number rescaled to the chord. `point.re` is based on the unit
    /// length of the coordinates (the chord Reynolds number if they have unit
    /// chord).
    pub fn input(&self, point: &OperatingPoint) -> AeroInput {
        AeroInput {
            kulfan: self.kulfan,
            alpha: point.alpha + self.rotation_angle,
            re: point.re / self.scale_factor,
            n_crit: point.n_crit,
            xtr_upper: point.xtr_upper,
            xtr_lower: point.xtr_lower,
        }
    }

    /// Offset (x, y) between the quarter-chord points of the original and the
    /// normalized airfoil, which is the arm of the moment transfer.
    pub fn quarter_chord_offset(&self) -> (f64, f64) {
        let (sin, cos) = self.rotation_angle.to_radians().sin_cos();
        (
            -self.x_translation + 0.25 * cos / self.scale_factor - 0.25,
            -self.y_translation - 0.25 * sin / self.scale_factor,
        )
    }

    /// Moves a moment coefficient computed for the normalized airfoil (at
    /// [`input`](Self::input)) back to the original airfoil, given the lift
    /// and drag coefficients at the same point.
    pub fn transfer_cm(&self, cl: f64, cd: f64, cm: f64) -> f64 {
        let (x, y) = self.quarter_chord_offset();
        cm - cl * x + cd * y
    }
}

impl NeuralFoil {
    /// Analyzes an airfoil given as coordinates at an operating point stated
    /// in the airfoil's own frame: [`analyze`](Self::analyze) at
    /// [`FittedAirfoil::input`], with the moment coefficient moved back by
    /// [`FittedAirfoil::transfer_cm`]. For a normalized airfoil this is
    /// `analyze` of the fitted shape.
    pub fn analyze_airfoil(&self, airfoil: &FittedAirfoil, point: &OperatingPoint) -> AeroOutput {
        let mut output = self.analyze(&airfoil.input(point));
        output.cm = airfoil.transfer_cm(output.cl, output.cd, output.cm);
        output
    }
}
