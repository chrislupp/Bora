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

//! Compressibility, post-stall and control surface corrections on top of the
//! network: a port of AeroSandbox's `KulfanAirfoil.get_aero_from_neuralfoil`.
//!
//! The correction formulas are written once, generic over [`Real`], and run
//! either on `f64` (values only) or on [`Dual`] numbers that carry the
//! derivatives with respect to every input (forward-mode differentiation).

use std::f64::consts::PI;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::input_index as idx;
use crate::kulfan::{max_thickness_station, thickness_weight_partials};
use crate::{
    AeroInput, AeroOutput, BoundaryLayerData, ControlSurface, CorrectedGradient,
    CorrectedGradients, CorrectedInput, CorrectedOutput, CorrectedResult, Gradient, NUM_BL_POINTS,
    NUM_CORRECTED_INPUTS, NUM_CST_WEIGHTS_PER_SIDE, NUM_INPUTS, NeuralFoil,
};

/// Number of derivative components carried through the corrections: the
/// [`NUM_CORRECTED_INPUTS`] inputs and the two quantities through which the
/// control surfaces act.
const NUM_SEEDS: usize = NUM_CORRECTED_INPUTS + 2;
/// Seed of the angle of attack shift caused by the control surfaces.
const SEED_ALPHA_SHIFT: usize = NUM_CORRECTED_INPUTS;
/// Seed of the drag multiplier caused by the control surfaces.
const SEED_CD_MULTIPLIER: usize = NUM_CORRECTED_INPUTS + 1;

/// The scalar type of the correction formulas: `f64`, or a [`Dual`] number.
trait Real:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    fn constant(value: f64) -> Self;
    fn value(self) -> f64;
    /// The result of a function of `self` whose value is `f` and whose
    /// derivative with respect to `self` is `df`.
    fn chain(self, f: f64, df: f64) -> Self;

    fn exp(self) -> Self {
        let e = self.value().exp();
        self.chain(e, e)
    }
    fn ln(self) -> Self {
        self.chain(self.value().ln(), 1.0 / self.value())
    }
    fn tanh(self) -> Self {
        let t = self.value().tanh();
        self.chain(t, 1.0 - t * t)
    }
    fn powf(self, p: f64) -> Self {
        let v = self.value();
        self.chain(v.powf(p), p * v.powf(p - 1.0))
    }
    fn sqrt(self) -> Self {
        let r = self.value().sqrt();
        self.chain(r, 0.5 / r)
    }
    /// Sine and cosine of an angle in degrees.
    fn sin_cos_deg(self) -> (Self, Self) {
        let (s, c) = self.value().to_radians().sin_cos();
        let k = PI / 180.0;
        (self.chain(s, k * c), self.chain(c, -k * s))
    }
}

impl Real for f64 {
    fn constant(value: f64) -> Self {
        value
    }
    fn value(self) -> f64 {
        self
    }
    fn chain(self, f: f64, _df: f64) -> Self {
        f
    }
}

/// A value with its derivatives with respect to the [`NUM_SEEDS`] seeds.
#[derive(Clone, Copy)]
struct Dual {
    v: f64,
    d: [f64; NUM_SEEDS],
}

impl Dual {
    /// An independent variable: unit derivative with respect to `seed`.
    fn variable(v: f64, seed: usize) -> Self {
        let mut d = [0.0; NUM_SEEDS];
        d[seed] = 1.0;
        Self { v, d }
    }
}

impl Real for Dual {
    fn constant(v: f64) -> Self {
        Self {
            v,
            d: [0.0; NUM_SEEDS],
        }
    }
    fn value(self) -> f64 {
        self.v
    }
    fn chain(self, f: f64, df: f64) -> Self {
        Self {
            v: f,
            d: self.d.map(|d| df * d),
        }
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            v: self.v + rhs.v,
            d: std::array::from_fn(|i| self.d[i] + rhs.d[i]),
        }
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            v: self.v - rhs.v,
            d: std::array::from_fn(|i| self.d[i] - rhs.d[i]),
        }
    }
}

impl Mul for Dual {
    type Output = Self;
    // Product rule.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        Self {
            v: self.v * rhs.v,
            d: std::array::from_fn(|i| self.d[i] * rhs.v + self.v * rhs.d[i]),
        }
    }
}

impl Div for Dual {
    type Output = Self;
    // Quotient rule.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let v = self.v / rhs.v;
        Self {
            v,
            d: std::array::from_fn(|i| (self.d[i] - v * rhs.d[i]) / rhs.v),
        }
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        self.chain(-self.v, -1.0)
    }
}

impl Add<f64> for Dual {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Self {
            v: self.v + rhs,
            d: self.d,
        }
    }
}

impl Sub<f64> for Dual {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Self {
            v: self.v - rhs,
            d: self.d,
        }
    }
}

impl Mul<f64> for Dual {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.chain(self.v * rhs, rhs)
    }
}

impl Div<f64> for Dual {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        self.chain(self.v / rhs, 1.0 / rhs)
    }
}

/// AeroSandbox's `softmax`: a smooth maximum (log-sum-exp) of `args`, which
/// approaches the true maximum as `softness` goes to zero.
fn softmax<T: Real>(args: &[T], softness: f64) -> T {
    let scaled = |a: T| a / softness;
    let mut max = scaled(args[0]);
    for &a in &args[1..] {
        if a.value() / softness > max.value() {
            max = scaled(a);
        }
    }
    let mut sum = T::constant(0.0);
    for &a in args {
        let below_max = scaled(a) - max;
        // Upstream clips the exponent at -500, where the term is constant.
        sum = sum
            + if below_max.value() < -500.0 {
                T::constant((-500.0_f64).exp())
            } else {
                below_max.exp()
            };
    }
    (max + sum.ln()) * softness
}

/// AeroSandbox's `softmin`: the smooth minimum matching [`softmax`].
fn softmin<T: Real>(args: &[T], softness: f64) -> T {
    let negated: Vec<T> = args.iter().map(|&a| -a).collect();
    -softmax(&negated, softness)
}

/// AeroSandbox's `blend`: `high` where `switch` is large, `low` where it is
/// very negative, with a tanh transition around zero.
fn blend<T: Real>(switch: T, high: T, low: T) -> T {
    let weight = switch.tanh() * 0.5 + 0.5;
    high * weight + low * (-weight + 1.0)
}

/// What the corrections take from the network, with the drag already
/// multiplied by the control surface factor.
struct Section<T> {
    cl: T,
    cd: T,
    cm: T,
    top_xtr: T,
    bot_xtr: T,
    /// Edge velocity ratios of both surfaces, upper surface first.
    ue_over_vinf: [T; 2 * NUM_BL_POINTS],
}

/// The corrected scalar outputs.
struct Corrected<T> {
    cl: T,
    cd: T,
    cm: T,
    cpmin: T,
    top_xtr: T,
    bot_xtr: T,
    mach_crit: T,
    mach_dd: T,
    cpmin_0: T,
}

/// Applies the post-stall and compressibility models to the network outputs.
///
/// `alpha` is the angle of attack (deg) wrapped to [-180, 180) and without
/// the shift of the control surfaces; `thickness` is the maximum thickness
/// of the airfoil.
fn correct<T: Real>(
    section: Section<T>,
    alpha: T,
    re: T,
    mach: T,
    thickness: T,
    include_360_deg_effects: bool,
) -> Corrected<T> {
    let constant = T::constant;
    let Section {
        mut cl,
        mut cd,
        mut cm,
        mut top_xtr,
        mut bot_xtr,
        ue_over_vinf,
    } = section;
    let (sin_alpha, cos_alpha) = alpha.sin_cos_deg();

    // Incompressible minimum pressure coefficient, from the largest edge
    // velocity (Bernoulli).
    let cp: Vec<T> = ue_over_vinf.iter().map(|&ue| -(ue * ue) + 1.0).collect();
    let mut cpmin_0 = softmin(&cp, 0.01);

    // Post-stall: blend towards a model of fully separated flow fitted to a
    // NACA 0012 over the full 360 deg range (Truong, 2020).
    let mut is_separated = None;
    if include_360_deg_effects {
        let cd90 = cos_alpha * 8.36e-2 + cos_alpha * cos_alpha * 4.06e-1 + 2.08;
        let cn = cd90 * sin_alpha;
        let ct = (cos_alpha * -1.78e-1 + cos_alpha * cos_alpha * cos_alpha * -2.98e-1 + 9.00e-2)
            * sin_alpha
            * sin_alpha;
        let cl_separated = cn * cos_alpha + ct * sin_alpha;
        let cd_separated = cn * sin_alpha - ct * cos_alpha;
        // Turbulent flat-plate skin friction, which keeps the drag positive.
        let cf = re.powf(-0.2) * 0.074;

        // About -1 where the flow is ~90% attached and 1 where it is ~90%
        // separated; stall is placed at +-20 deg since the network
        // extrapolates well past stall.
        let separated = softmax(&[alpha - 20.0, -alpha - 20.0], 1.0) / 3.0;

        cl = blend(separated, cl_separated, cl);
        cd = blend(separated, (cd_separated + cf).ln(), cd.ln()).exp();
        cm = blend(separated, constant(0.0), cm);
        cpmin_0 = blend(separated, sin_alpha * sin_alpha * -0.5 - 1.0, cpmin_0);
        let tripped = (sin_alpha * 10.0).tanh() * 0.5;
        top_xtr = blend(separated, -tripped + 0.5, top_xtr);
        bot_xtr = blend(separated, tripped + 0.5, bot_xtr);
        is_separated = Some(separated);
    }

    // Critical Mach number: a fit of the Mach number at which the minimum
    // pressure reaches the sonic value under Laitone's rule.
    let cpmin_0 = softmin(&[cpmin_0, constant(0.0)], 0.001);
    let mach_crit =
        (-cpmin_0 + (-cpmin_0).powf(0.6724789439840343) * 0.6582431351007195 + 1.011571026701678)
            .powf(-0.5504677038358711);
    // Drag divergence, from the Korn equation.
    let mach_dd = mach_crit + (0.1_f64 / 320.0).cbrt();

    // Prandtl-Glauert factor, kept real and positive through Mach 1.
    let beta_squared = -(mach * mach) + 1.0;
    let beta = softmax(&[beta_squared, -beta_squared], 0.5).sqrt();
    let mut cl = cl / beta;
    let mut cm = cm / beta;
    let cpmin = cpmin_0 / beta;

    // Lift loss from buffet, and the drop of the lift slope from 2 pi to 4
    // going supersonic.
    let supersonic = (mach - 1.0) / 0.1;
    let buffet_factor = blend(
        (mach - (mach_dd + 0.04)) * 50.0,
        blend(supersonic, constant(1.0), constant(0.5)),
        constant(1.0),
    );
    let lift_slope_factor = blend(supersonic, constant(4.0 / (2.0 * PI)), constant(1.0));
    cl = cl * buffet_factor * lift_slope_factor;

    // Wave drag.
    let m = mach.value();
    let cd_wave = if m < mach_crit.value() {
        constant(0.0)
    } else if m < mach_dd.value() {
        (mach - mach_crit).powf(4.0) * 80.0
    } else if m < 1.1 {
        // Cosine-blended Hermite patch from drag divergence to Mach 1.1.
        let rise = (mach - mach_dd) * 0.1 + 80.0 * (0.1_f64 / 320.0).powf(4.0 / 3.0);
        let peak = (mach - 1.1) * (thickness * (-0.8 * 8.0)) + thickness * 0.8;
        let t = (mach - mach_dd) / (-mach_dd + 1.1);
        let (_, cos_t) = (t * 180.0).sin_cos_deg();
        let weight = cos_t * 0.5 + 0.5;
        weight * rise + (-weight + 1.0) * peak
    } else {
        blend(
            (mach - 1.1) * (8.0 * 2.0 / (1.2 - 0.8)),
            thickness * (0.8 * 0.8),
            thickness * (1.2 * 0.8),
        )
    };
    let cd = cd + cd_wave;

    // Beyond drag divergence, or when separated, the aerodynamic center moves
    // back to mid-chord (Mach tuck).
    let mut center_shift = (mach - (mach_dd + 0.06)) / 0.06;
    if let Some(separated) = is_separated {
        center_shift = softmax(&[separated, center_shift], 0.1);
    }
    cm = cm
        + blend(
            center_shift,
            cos_alpha * cl * -0.25 - sin_alpha * cd * 0.25,
            constant(0.0),
        );

    Corrected {
        cl,
        cd,
        cm,
        cpmin,
        top_xtr,
        bot_xtr,
        mach_crit,
        mach_dd,
        cpmin_0,
    }
}

/// One control surface's contribution, with the derivatives with respect to
/// its deflection.
struct SurfaceEffect {
    /// Angle of attack shift per degree of deflection.
    effectiveness: f64,
    /// Factor on the drag coefficient.
    cd_factor: f64,
    /// d(`cd_factor`)/d(deflection).
    cd_factor_slope: f64,
}

impl SurfaceEffect {
    fn new(surface: &ControlSurface) -> Self {
        // Fit to an XFoil study of flap effectiveness (AeroSandbox).
        let effectiveness = 1.0
            - (surface.hinge_point + 1e-16)
                .max(0.0)
                .powf(2.751428551177291);
        // Fit to wind tunnel data from Hoerner, "Fluid-Dynamic Drag" (1965).
        let u = surface.deflection / 11.5;
        let root = (1.0 + u * u).sqrt();
        Self {
            effectiveness,
            cd_factor: 2.0 + u * u - root,
            cd_factor_slope: (2.0 * u - u / root) / 11.5,
        }
    }
}

/// Wraps an angle of attack (deg) to [-180, 180).
fn wrap_alpha(alpha: f64) -> f64 {
    (alpha + 180.0).rem_euclid(360.0) - 180.0
}

/// The network evaluation behind a corrected analysis: the wrapped angle of
/// attack, the control surface effects, and the network input, which sees the
/// angle of attack shifted by the control surfaces.
struct Setup {
    alpha: f64,
    surfaces: Vec<SurfaceEffect>,
    cd_multiplier: f64,
    network_input: AeroInput,
}

impl Setup {
    fn new(input: &CorrectedInput) -> Self {
        let alpha = wrap_alpha(input.aero.alpha);
        let surfaces: Vec<SurfaceEffect> = input
            .control_surfaces
            .iter()
            .map(SurfaceEffect::new)
            .collect();
        let alpha_shift: f64 = input
            .control_surfaces
            .iter()
            .zip(&surfaces)
            .map(|(surface, effect)| surface.deflection * effect.effectiveness)
            .sum();
        Self {
            alpha,
            cd_multiplier: surfaces.iter().map(|effect| effect.cd_factor).product(),
            surfaces,
            network_input: AeroInput {
                alpha: alpha + alpha_shift,
                ..input.aero
            },
        }
    }
}

fn edge_velocities<T>(
    upper: [T; NUM_BL_POINTS],
    lower: [T; NUM_BL_POINTS],
) -> [T; 2 * NUM_BL_POINTS]
where
    T: Copy,
{
    std::array::from_fn(|i| {
        if i < NUM_BL_POINTS {
            upper[i]
        } else {
            lower[i - NUM_BL_POINTS]
        }
    })
}

/// The corrected analysis without derivatives, given the network outputs.
fn corrected_output(input: &CorrectedInput, setup: &Setup, base: &AeroOutput) -> CorrectedOutput {
    let section = Section {
        cl: base.cl,
        cd: base.cd * setup.cd_multiplier,
        cm: base.cm,
        top_xtr: base.top_xtr,
        bot_xtr: base.bot_xtr,
        ue_over_vinf: edge_velocities(base.upper.ue_over_vinf, base.lower.ue_over_vinf),
    };
    let corrected = correct(
        section,
        setup.alpha,
        input.aero.re,
        input.mach,
        input.aero.kulfan.max_thickness(),
        input.include_360_deg_effects,
    );
    CorrectedOutput {
        analysis_confidence: base.analysis_confidence,
        cl: corrected.cl,
        cd: corrected.cd,
        cm: corrected.cm,
        cpmin: corrected.cpmin,
        top_xtr: corrected.top_xtr,
        bot_xtr: corrected.bot_xtr,
        mach_crit: corrected.mach_crit,
        mach_dd: corrected.mach_dd,
        cpmin_0: corrected.cpmin_0,
        upper: base.upper,
        lower: base.lower,
    }
}

impl NeuralFoil {
    /// Analyzes an airfoil with the corrections that AeroSandbox's
    /// `KulfanAirfoil.get_aero_from_neuralfoil` applies on top of the
    /// network, which is trained on incompressible XFoil data:
    ///
    /// - **control surfaces** shift the angle of attack seen by the network
    ///   and multiply the drag;
    /// - **post-stall** (`include_360_deg_effects`): beyond about 20 deg the
    ///   coefficients blend into a model of fully separated flow, and the
    ///   angle of attack is periodic over 360 deg;
    /// - **compressibility**: a Prandtl-Glauert scaling of lift, moment and
    ///   minimum pressure, the critical and drag-divergence Mach numbers,
    ///   wave drag, the lift loss from buffet, the supersonic lift slope, and
    ///   the aft shift of the aerodynamic center.
    ///
    /// The boundary layer outputs and the analysis confidence are the
    /// network's, at the angle of attack shifted by the control surfaces.
    ///
    /// The corrections are smooth blends that are never exactly inactive: at
    /// Mach 0, without control surfaces and with `include_360_deg_effects`
    /// off, the lift and moment coefficients are still about 0.4% below those
    /// of [`analyze`](Self::analyze), exactly as upstream.
    pub fn analyze_corrected(&self, input: &CorrectedInput) -> CorrectedOutput {
        let setup = Setup::new(input);
        let base = self.analyze(&setup.network_input);
        corrected_output(input, &setup, &base)
    }

    /// Batched [`analyze_corrected`](Self::analyze_corrected): one output per
    /// input, in order, with the network evaluated by
    /// [`analyze_batch`](Self::analyze_batch) (so the results match the
    /// single-point call to within a few ulps of the network outputs, not bit
    /// for bit).
    pub fn analyze_corrected_batch(&self, inputs: &[CorrectedInput]) -> Vec<CorrectedOutput> {
        let setups: Vec<Setup> = inputs.iter().map(Setup::new).collect();
        let network_inputs: Vec<AeroInput> = setups.iter().map(|s| s.network_input).collect();
        let base = self.analyze_batch(&network_inputs);
        inputs
            .iter()
            .zip(&setups)
            .zip(&base)
            .map(|((input, setup), base)| corrected_output(input, setup, base))
            .collect()
    }

    /// Like [`analyze_corrected`](Self::analyze_corrected), with analytic
    /// gradients of every output with respect to the
    /// [`NUM_CORRECTED_INPUTS`] inputs (the [`NUM_INPUTS`] inputs of
    /// [`analyze_with_gradients`](Self::analyze_with_gradients), then Mach)
    /// and with respect to each control surface deflection.
    ///
    /// The outputs are those of `analyze_corrected` up to rounding. The cost
    /// is that of `analyze_with_gradients`: the corrections need the edge
    /// velocities and their gradients, so there is no scalar-only shortcut.
    ///
    /// The maximum thickness, which sets the supersonic wave drag, is taken
    /// at a sampled station; its gradient is that of the thickness at this
    /// station, and jumps when the thickest station changes.
    pub fn analyze_corrected_with_gradients(&self, input: &CorrectedInput) -> CorrectedResult {
        let setup = Setup::new(input);
        let base = self.analyze_with_gradients(&setup.network_input);
        let (out, grad) = (&base.output, &base.gradient);

        // The network sees alpha plus the control surface shift, so its
        // alpha derivative is also its derivative with respect to the shift.
        let lift = |v: f64, gradient: &Gradient| {
            let mut d = [0.0; NUM_SEEDS];
            d[..NUM_INPUTS].copy_from_slice(gradient);
            d[SEED_ALPHA_SHIFT] = gradient[idx::ALPHA];
            Dual { v, d }
        };

        let (thickness, station) = max_thickness_station(&input.aero.kulfan);
        let mut thickness = Dual::constant(thickness);
        for (k, partial) in thickness_weight_partials(station).into_iter().enumerate() {
            debug_assert!(k < NUM_CST_WEIGHTS_PER_SIDE);
            thickness.d[idx::UPPER_WEIGHTS + k] = partial;
            thickness.d[idx::LOWER_WEIGHTS + k] = -partial;
        }
        thickness.d[idx::TE_THICKNESS] = station;

        let section = Section {
            cl: lift(out.cl, &grad.cl),
            cd: lift(out.cd, &grad.cd) * Dual::variable(setup.cd_multiplier, SEED_CD_MULTIPLIER),
            cm: lift(out.cm, &grad.cm),
            top_xtr: lift(out.top_xtr, &grad.top_xtr),
            bot_xtr: lift(out.bot_xtr, &grad.bot_xtr),
            ue_over_vinf: edge_velocities(
                std::array::from_fn(|i| {
                    lift(out.upper.ue_over_vinf[i], &grad.upper.ue_over_vinf[i])
                }),
                std::array::from_fn(|i| {
                    lift(out.lower.ue_over_vinf[i], &grad.lower.ue_over_vinf[i])
                }),
            ),
        };
        let corrected = correct(
            section,
            Dual::variable(setup.alpha, idx::ALPHA),
            Dual::variable(input.aero.re, idx::RE),
            Dual::variable(input.mach, idx::MACH),
            thickness,
            input.include_360_deg_effects,
        );
        let confidence = lift(out.analysis_confidence, &grad.analysis_confidence);

        let inputs_gradient =
            |dual: &Dual| -> CorrectedGradient { std::array::from_fn(|k| dual.d[k]) };
        let output = CorrectedOutput {
            analysis_confidence: confidence.v,
            cl: corrected.cl.v,
            cd: corrected.cd.v,
            cm: corrected.cm.v,
            cpmin: corrected.cpmin.v,
            top_xtr: corrected.top_xtr.v,
            bot_xtr: corrected.bot_xtr.v,
            mach_crit: corrected.mach_crit.v,
            mach_dd: corrected.mach_dd.v,
            cpmin_0: corrected.cpmin_0.v,
            upper: out.upper,
            lower: out.lower,
        };

        // A deflection acts through the alpha shift and the drag multiplier.
        let deflection = setup
            .surfaces
            .iter()
            .map(|effect| {
                let multiplier_slope =
                    setup.cd_multiplier / effect.cd_factor * effect.cd_factor_slope;
                let scalar = |dual: &Dual| {
                    dual.d[SEED_ALPHA_SHIFT] * effect.effectiveness
                        + dual.d[SEED_CD_MULTIPLIER] * multiplier_slope
                };
                let distribution = |gradients: &[Gradient; NUM_BL_POINTS]| {
                    std::array::from_fn(|i| gradients[i][idx::ALPHA] * effect.effectiveness)
                };
                let surface = |g: &crate::BoundaryLayerGradients| BoundaryLayerData {
                    theta: distribution(&g.theta),
                    h: distribution(&g.h),
                    ue_over_vinf: distribution(&g.ue_over_vinf),
                };
                CorrectedOutput {
                    analysis_confidence: scalar(&confidence),
                    cl: scalar(&corrected.cl),
                    cd: scalar(&corrected.cd),
                    cm: scalar(&corrected.cm),
                    cpmin: scalar(&corrected.cpmin),
                    top_xtr: scalar(&corrected.top_xtr),
                    bot_xtr: scalar(&corrected.bot_xtr),
                    mach_crit: scalar(&corrected.mach_crit),
                    mach_dd: scalar(&corrected.mach_dd),
                    cpmin_0: scalar(&corrected.cpmin_0),
                    upper: surface(&grad.upper),
                    lower: surface(&grad.lower),
                }
            })
            .collect();

        let gradient = CorrectedGradients {
            analysis_confidence: inputs_gradient(&confidence),
            cl: inputs_gradient(&corrected.cl),
            cd: inputs_gradient(&corrected.cd),
            cm: inputs_gradient(&corrected.cm),
            cpmin: inputs_gradient(&corrected.cpmin),
            top_xtr: inputs_gradient(&corrected.top_xtr),
            bot_xtr: inputs_gradient(&corrected.bot_xtr),
            mach_crit: inputs_gradient(&corrected.mach_crit),
            mach_dd: inputs_gradient(&corrected.mach_dd),
            cpmin_0: inputs_gradient(&corrected.cpmin_0),
            upper: base.gradient.upper,
            lower: base.gradient.lower,
            deflection,
        };
        CorrectedResult { output, gradient }
    }
}
