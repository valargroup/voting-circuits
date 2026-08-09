//! Unsigned 30-bit fixed-base multiplication for El Gamal share values.
//!
//! This specializes the Orchard three-bit fixed-base construction to the ten
//! windows required by the vote proof's `[0, 2^30)` share range. It preserves
//! the offset-point construction used by `halo2_gadgets`, so every incomplete
//! addition operates on non-identity, non-exceptional points. The final two
//! complete additions recover `[share] SpendAuthG` and add `[r] ea_pk`.

use ff::{Field, PrimeField};
use group::Curve;
use halo2_gadgets::{
    ecc::chip::{compute_lagrange_coeffs, H},
    utilities::decompose_running_sum::RunningSumConfig,
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Region, Value},
    plonk::{Advice, Column, ConstraintSystem, Constraints, Error, Expression, Fixed, Selector},
    poly::Rotation,
};
use lazy_static::lazy_static;
use orchard::constants::fixed_bases::spend_auth_g;
use pasta_curves::{arithmetic::CurveAffine, pallas};

use crate::params::{RANGE_CHECK_WORD_BITS, SHARE_VALUE_BITS};

const WINDOW_BITS: usize = 3;
const NUM_WINDOWS: usize = SHARE_VALUE_BITS / WINDOW_BITS;
const PRECOMPUTED_NUM_WINDOWS: usize = 10;

const _: () = assert!(SHARE_VALUE_BITS % WINDOW_BITS == 0);
const _: () = assert!(SHARE_VALUE_BITS % RANGE_CHECK_WORD_BITS == 0);
// `FINAL_WINDOW_Z` and `FINAL_WINDOW_U` are precomputed for this window count.
const _: () = assert!(NUM_WINDOWS == PRECOMPUTED_NUM_WINDOWS);

// The first nine windows use the full SpendAuthG table. The final window is
// different because it cancels the offset accumulated by those nine windows.
// These final-window constants are the last entry produced by
// `find_zs_and_us(spend_auth_g::generator(), NUM_WINDOWS)`.
const FINAL_WINDOW_Z: u64 = 149_621;
const FINAL_WINDOW_U: [[u8; 32]; H] = [
    [
        128, 37, 143, 180, 247, 179, 153, 160, 208, 21, 176, 71, 24, 133, 244, 85, 228, 223, 33,
        134, 230, 147, 157, 101, 133, 169, 137, 177, 53, 232, 183, 3,
    ],
    [
        23, 65, 71, 143, 97, 98, 147, 179, 149, 106, 120, 127, 190, 158, 135, 183, 26, 127, 89,
        253, 40, 27, 179, 52, 7, 215, 107, 37, 176, 156, 96, 39,
    ],
    [
        111, 58, 214, 249, 96, 178, 240, 72, 9, 75, 218, 206, 11, 87, 3, 154, 73, 71, 247, 66, 106,
        5, 17, 225, 220, 219, 81, 8, 180, 150, 74, 16,
    ],
    [
        185, 185, 45, 234, 66, 210, 75, 67, 178, 5, 73, 183, 159, 242, 100, 80, 29, 187, 103, 255,
        131, 141, 165, 164, 60, 88, 174, 124, 168, 131, 9, 12,
    ],
    [
        99, 225, 26, 225, 215, 57, 169, 0, 221, 232, 167, 206, 255, 128, 72, 215, 131, 238, 41,
        247, 229, 172, 211, 253, 223, 49, 108, 243, 210, 216, 43, 21,
    ],
    [
        79, 70, 131, 151, 213, 28, 224, 52, 13, 72, 138, 174, 117, 183, 81, 238, 121, 246, 132,
        123, 255, 194, 16, 51, 116, 120, 219, 102, 125, 202, 68, 6,
    ],
    [
        39, 123, 235, 30, 157, 64, 134, 101, 178, 96, 88, 88, 190, 57, 59, 210, 191, 196, 196, 148,
        180, 64, 133, 224, 56, 209, 7, 119, 61, 237, 200, 62,
    ],
    [
        185, 93, 140, 245, 242, 235, 178, 173, 215, 139, 5, 219, 227, 68, 156, 73, 167, 228, 138,
        20, 22, 149, 78, 240, 159, 215, 39, 171, 131, 70, 93, 47,
    ],
];

lazy_static! {
    static ref LAGRANGE_COEFFS: Vec<[pallas::Base; H]> =
        compute_lagrange_coeffs(spend_auth_g::generator(), NUM_WINDOWS);
}

#[derive(Clone, Debug)]
struct AssignedPoint {
    x: AssignedCell<pallas::Base, pallas::Base>,
    y: AssignedCell<pallas::Base, pallas::Base>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Tamper {
    SelectedY,
    SelectedU,
    IncompleteAddX,
    CompleteAddLambda,
    CompleteAddAlpha,
    CompleteAddBeta,
    CompleteAddGamma,
    CompleteAddDelta,
    RunningSumWord,
}

/// Configuration for unsigned 30-bit multiplication by SpendAuthG followed by
/// complete addition of another constrained point.
#[derive(Clone, Debug)]
pub(crate) struct SpendAuthGFixedBase30Config {
    advices: [Column<Advice>; 10],
    lagrange_coeffs: [Column<Fixed>; H],
    fixed_z: Column<Fixed>,
    running_sum: RunningSumConfig<pallas::Base, WINDOW_BITS>,
    q_coords: Selector,
    q_add_incomplete: Selector,
    q_add_complete: Selector,
    #[cfg(test)]
    q_range_check: Selector,
    #[cfg(test)]
    tamper: Option<Tamper>,
}

impl SpendAuthGFixedBase30Config {
    /// Configures the ten-window fixed-base multiplication gates.
    pub(crate) fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        advices: [Column<Advice>; 10],
        lagrange_coeffs: [Column<Fixed>; H],
    ) -> Self {
        let q_range_check = meta.selector();
        let config = Self {
            advices,
            lagrange_coeffs,
            fixed_z: meta.fixed_column(),
            running_sum: RunningSumConfig::configure(meta, q_range_check, advices[4]),
            q_coords: meta.selector(),
            q_add_incomplete: meta.selector(),
            q_add_complete: meta.selector(),
            #[cfg(test)]
            q_range_check,
            #[cfg(test)]
            tamper: None,
        };

        config.create_window_gates(meta);
        config.create_complete_add_gate(meta);
        config
    }

    fn create_window_gates(&self, meta: &mut ConstraintSystem<pallas::Base>) {
        // https://p.z.cash/halo2-0.1:ecc-fixed-mul-coordinates
        meta.create_gate("30-bit fixed-base window coordinates", |meta| {
            let q = meta.query_selector(self.q_coords);
            let z_cur = meta.query_advice(self.advices[4], Rotation::cur());
            let z_next = meta.query_advice(self.advices[4], Rotation::next());
            let word = z_cur - z_next * pallas::Base::from(1 << WINDOW_BITS);

            let x = meta.query_advice(self.advices[0], Rotation::cur());
            let y = meta.query_advice(self.advices[1], Rotation::cur());
            let u = meta.query_advice(self.advices[5], Rotation::cur());
            let fixed_z = meta.query_fixed(self.fixed_z);

            let mut word_power = Expression::Constant(pallas::Base::one());
            let mut interpolated_x = Expression::Constant(pallas::Base::zero());
            for coeff in self.lagrange_coeffs {
                interpolated_x = interpolated_x + word_power.clone() * meta.query_fixed(coeff);
                word_power = word_power * word.clone();
            }

            Constraints::with_selector(
                q,
                [
                    ("interpolated x", interpolated_x - x.clone()),
                    ("selected y", u.square() - y.clone() - fixed_z),
                    (
                        "selected point on curve",
                        y.square()
                            - x.clone().square() * x
                            - Expression::Constant(pallas::Affine::b()),
                    ),
                ],
            )
        });

        // https://p.z.cash/halo2-0.1:ecc-incomplete-addition
        meta.create_gate("30-bit fixed-base incomplete addition", |meta| {
            let q = meta.query_selector(self.q_add_incomplete);
            let x_p = meta.query_advice(self.advices[0], Rotation::cur());
            let y_p = meta.query_advice(self.advices[1], Rotation::cur());
            let x_q = meta.query_advice(self.advices[2], Rotation::cur());
            let y_q = meta.query_advice(self.advices[3], Rotation::cur());
            let x_r = meta.query_advice(self.advices[2], Rotation::next());
            let y_r = meta.query_advice(self.advices[3], Rotation::next());

            let x_difference = x_p.clone() - x_q.clone();
            let y_difference = y_p.clone() - y_q.clone();
            let x_check = (x_r.clone() + x_q.clone() + x_p.clone()) * x_difference.clone().square()
                - y_difference.clone().square();
            let y_check = (y_r + y_q) * x_difference - y_difference * (x_q - x_r);

            Constraints::with_selector(q, [("x coordinate", x_check), ("y coordinate", y_check)])
        });
    }

    fn create_complete_add_gate(&self, meta: &mut ConstraintSystem<pallas::Base>) {
        // https://p.z.cash/halo2-0.1:ecc-complete-addition
        meta.create_gate("30-bit fixed-base complete addition", |meta| {
            let q = meta.query_selector(self.q_add_complete);
            let x_p = meta.query_advice(self.advices[0], Rotation::cur());
            let y_p = meta.query_advice(self.advices[1], Rotation::cur());
            let x_q = meta.query_advice(self.advices[2], Rotation::cur());
            let y_q = meta.query_advice(self.advices[3], Rotation::cur());
            let x_r = meta.query_advice(self.advices[2], Rotation::next());
            let y_r = meta.query_advice(self.advices[3], Rotation::next());
            let lambda = meta.query_advice(self.advices[4], Rotation::cur());
            let alpha = meta.query_advice(self.advices[5], Rotation::cur());
            let beta = meta.query_advice(self.advices[6], Rotation::cur());
            let gamma = meta.query_advice(self.advices[7], Rotation::cur());
            let delta = meta.query_advice(self.advices[8], Rotation::cur());

            let one = Expression::Constant(pallas::Base::one());
            let x_difference = x_q.clone() - x_p.clone();
            let y_sum = y_q.clone() + y_p.clone();
            let if_alpha = x_difference.clone() * alpha;
            let if_beta = x_p.clone() * beta;
            let if_gamma = x_q.clone() * gamma;
            let if_delta = y_sum.clone() * delta;

            let slope_check = x_difference.clone()
                * (x_difference.clone() * lambda.clone() - (y_q.clone() - y_p.clone()));
            let two = Expression::Constant(pallas::Base::from(2));
            let three = Expression::Constant(pallas::Base::from(3));
            let tangent_check = (one.clone() - if_alpha.clone())
                * (two * y_p.clone() * lambda.clone() - three * x_p.clone().square());
            let x_result = lambda.clone().square() - x_p.clone() - x_q.clone() - x_r.clone();
            let y_result = lambda * (x_p.clone() - x_r.clone()) - y_p.clone() - y_r.clone();

            Constraints::with_selector(
                q,
                [
                    ("slope", slope_check),
                    ("tangent", tangent_check),
                    (
                        "nonexceptional x by x difference",
                        x_p.clone() * x_q.clone() * x_difference.clone() * x_result.clone(),
                    ),
                    (
                        "nonexceptional y by x difference",
                        x_p.clone() * x_q.clone() * x_difference * y_result.clone(),
                    ),
                    (
                        "nonexceptional x by y sum",
                        x_p.clone() * x_q.clone() * y_sum.clone() * x_result,
                    ),
                    (
                        "nonexceptional y by y sum",
                        x_p.clone() * x_q.clone() * y_sum * y_result,
                    ),
                    (
                        "left identity x",
                        (one.clone() - if_beta.clone()) * (x_r.clone() - x_q.clone()),
                    ),
                    (
                        "left identity y",
                        (one.clone() - if_beta) * (y_r.clone() - y_q.clone()),
                    ),
                    (
                        "right identity x",
                        (one.clone() - if_gamma.clone()) * (x_r.clone() - x_p.clone()),
                    ),
                    (
                        "right identity y",
                        (one.clone() - if_gamma) * (y_r.clone() - y_p.clone()),
                    ),
                    (
                        "inverse points x",
                        (one.clone() - if_alpha.clone() - if_delta.clone()) * x_r,
                    ),
                    ("inverse points y", (one - if_alpha - if_delta) * y_r),
                ],
            )
        });
    }

    /// Constrains `result = [share] SpendAuthG + addend` for an unsigned
    /// 30-bit `share`. The addend must already be constrained as a curve point.
    pub(crate) fn mul_add(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        share: &AssignedCell<pallas::Base, pallas::Base>,
        addend_x: &AssignedCell<pallas::Base, pallas::Base>,
        addend_y: &AssignedCell<pallas::Base, pallas::Base>,
    ) -> Result<
        (
            AssignedCell<pallas::Base, pallas::Base>,
            AssignedCell<pallas::Base, pallas::Base>,
        ),
        Error,
    > {
        let (accumulator, most_significant) = layouter.assign_region(
            || "Unsigned 30-bit fixed-base mul",
            |mut region| self.assign_windows(&mut region, share),
        )?;
        let magnitude = self.complete_add(
            layouter.namespace(|| "Unsigned 30-bit fixed-base final window"),
            &accumulator,
            &most_significant,
        )?;
        let addend = AssignedPoint {
            x: addend_x.clone(),
            y: addend_y.clone(),
        };
        let result = self.complete_add(
            layouter.namespace(|| "Unsigned 30-bit fixed-base plus addend"),
            &magnitude,
            &addend,
        )?;

        Ok((result.x, result.y))
    }

    #[cfg(test)]
    fn assign_tampered_running_sum(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || "tampered running-sum word",
            |mut region| {
                self.q_range_check.enable(&mut region, 0)?;
                region.assign_advice(
                    || "z_cur",
                    self.advices[4],
                    0,
                    || Value::known(pallas::Base::from(1 << WINDOW_BITS)),
                )?;
                region.assign_advice(
                    || "z_next",
                    self.advices[4],
                    1,
                    || Value::known(pallas::Base::zero()),
                )?;
                Ok(())
            },
        )
    }

    #[cfg(test)]
    fn corrupt_value(&self, target: Tamper, value: Value<pallas::Base>) -> Value<pallas::Base> {
        if self.tamper == Some(target) {
            value + Value::known(pallas::Base::one())
        } else {
            value
        }
    }

    fn assign_windows(
        &self,
        region: &mut Region<'_, pallas::Base>,
        share: &AssignedCell<pallas::Base, pallas::Base>,
    ) -> Result<(AssignedPoint, AssignedPoint), Error> {
        let running_sum = self.running_sum.copy_decompose(
            region,
            0,
            share.clone(),
            true,
            SHARE_VALUE_BITS,
            NUM_WINDOWS,
        )?;

        let mut accumulator = None;
        let mut most_significant = None;
        for window in 0..NUM_WINDOWS {
            self.q_coords.enable(region, window)?;
            self.assign_fixed_constants(region, window)?;

            let digit = running_sum[window]
                .value()
                .zip(running_sum[window + 1].value())
                .map(|(current, next)| {
                    let word = *current - *next * pallas::Base::from(1 << WINDOW_BITS);
                    word.to_repr().as_ref()[0] as usize
                });
            let selected = self.assign_selected_point(region, window, digit)?;

            if window == 0 {
                let x = selected.x.copy_advice(
                    || "initial accumulator x",
                    region,
                    self.advices[2],
                    1,
                )?;
                let y = selected.y.copy_advice(
                    || "initial accumulator y",
                    region,
                    self.advices[3],
                    1,
                )?;
                accumulator = Some(AssignedPoint { x, y });
            } else if window < NUM_WINDOWS - 1 {
                let current = accumulator.as_ref().expect("initialized at window zero");
                let next = self.assign_incomplete_add(region, window, &selected, current)?;
                accumulator = Some(next);
            } else {
                most_significant = Some(selected);
            }
        }

        Ok((
            accumulator.expect("at least two windows"),
            most_significant.expect("at least one window"),
        ))
    }

    fn assign_fixed_constants(
        &self,
        region: &mut Region<'_, pallas::Base>,
        window: usize,
    ) -> Result<(), Error> {
        for (column, coefficient) in self
            .lagrange_coeffs
            .iter()
            .zip(LAGRANGE_COEFFS[window].iter())
        {
            region.assign_fixed(
                || format!("window {window} Lagrange coefficient"),
                *column,
                window,
                || Value::known(*coefficient),
            )?;
        }
        let z = if window == NUM_WINDOWS - 1 {
            FINAL_WINDOW_Z
        } else {
            spend_auth_g::Z[window]
        };
        region.assign_fixed(
            || format!("window {window} z"),
            self.fixed_z,
            window,
            || Value::known(pallas::Base::from(z)),
        )?;
        Ok(())
    }

    fn assign_selected_point(
        &self,
        region: &mut Region<'_, pallas::Base>,
        window: usize,
        digit: Value<usize>,
    ) -> Result<AssignedPoint, Error> {
        let scalar = digit.map(|digit| window_scalar(window, digit));
        let coordinates = scalar.map(|scalar| {
            let point = (spend_auth_g::generator() * scalar).to_affine();
            let coordinates = point.coordinates().unwrap();
            (*coordinates.x(), *coordinates.y())
        });
        let x = region.assign_advice(
            || format!("window {window} selected x"),
            self.advices[0],
            window,
            || coordinates.map(|coordinates| coordinates.0),
        )?;
        let selected_y = coordinates.map(|coordinates| coordinates.1);
        #[cfg(test)]
        let selected_y = if self.tamper == Some(Tamper::SelectedY) && window == 0 {
            selected_y.map(|y| -y)
        } else {
            selected_y
        };
        let y = region.assign_advice(
            || format!("window {window} selected y"),
            self.advices[1],
            window,
            || selected_y,
        )?;

        let u = digit.map(|digit| {
            let repr = if window == NUM_WINDOWS - 1 {
                FINAL_WINDOW_U[digit]
            } else {
                spend_auth_g::U[window][digit]
            };
            pallas::Base::from_repr(repr).unwrap()
        });
        #[cfg(test)]
        let u = if window == 0 {
            self.corrupt_value(Tamper::SelectedU, u)
        } else {
            u
        };
        region.assign_advice(
            || format!("window {window} u"),
            self.advices[5],
            window,
            || u,
        )?;

        Ok(AssignedPoint { x, y })
    }

    fn assign_incomplete_add(
        &self,
        region: &mut Region<'_, pallas::Base>,
        row: usize,
        selected: &AssignedPoint,
        accumulator: &AssignedPoint,
    ) -> Result<AssignedPoint, Error> {
        self.q_add_incomplete.enable(region, row)?;
        selected
            .x
            .value()
            .zip(accumulator.x.value())
            .error_if_known_and(|(selected, accumulator)| selected == accumulator)?;

        let result = selected
            .x
            .value()
            .zip(selected.y.value())
            .zip(accumulator.x.value())
            .zip(accumulator.y.value())
            .map(|(((x_p, y_p), x_q), y_q)| {
                let lambda = (*y_q - *y_p) * (*x_q - *x_p).invert().unwrap();
                let x_r = lambda.square() - x_p - x_q;
                let y_r = lambda * (*x_p - x_r) - y_p;
                (x_r, y_r)
            });
        let result_x = result.map(|result| result.0);
        #[cfg(test)]
        let result_x = if row == 1 {
            self.corrupt_value(Tamper::IncompleteAddX, result_x)
        } else {
            result_x
        };
        let x =
            region.assign_advice(|| "incomplete sum x", self.advices[2], row + 1, || result_x)?;
        let y = region.assign_advice(
            || "incomplete sum y",
            self.advices[3],
            row + 1,
            || result.map(|result| result.1),
        )?;
        Ok(AssignedPoint { x, y })
    }

    fn complete_add(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        p: &AssignedPoint,
        q: &AssignedPoint,
    ) -> Result<AssignedPoint, Error> {
        layouter.assign_region(
            || "complete point addition",
            |mut region| {
                self.q_add_complete.enable(&mut region, 0)?;
                p.x.copy_advice(|| "x_p", &mut region, self.advices[0], 0)?;
                p.y.copy_advice(|| "y_p", &mut region, self.advices[1], 0)?;
                q.x.copy_advice(|| "x_q", &mut region, self.advices[2], 0)?;
                q.y.copy_advice(|| "y_q", &mut region, self.advices[3], 0)?;

                let x_p = p.x.value().copied();
                let y_p = p.y.value().copied();
                let x_q = q.x.value().copied();
                let y_q = q.y.value().copied();
                let alpha = (x_q - x_p).map(invert_or_zero);
                let beta = x_p.map(invert_or_zero);
                let gamma = x_q.map(invert_or_zero);
                let delta = x_p
                    .zip(x_q)
                    .zip(y_p)
                    .zip(y_q)
                    .map(|(((x_p, x_q), y_p), y_q)| {
                        if x_q == x_p {
                            invert_or_zero(y_q + y_p)
                        } else {
                            pallas::Base::zero()
                        }
                    });
                let lambda = x_p.zip(y_p).zip(x_q).zip(y_q).zip(alpha).map(
                    |((((x_p, y_p), x_q), y_q), alpha)| {
                        if x_q != x_p {
                            (y_q - y_p) * alpha
                        } else if !bool::from(y_p.is_zero()) {
                            pallas::Base::from(3)
                                * x_p.square()
                                * invert_or_zero(pallas::Base::from(2) * y_p)
                        } else {
                            pallas::Base::zero()
                        }
                    },
                );

                #[cfg(test)]
                let lambda = self.corrupt_value(Tamper::CompleteAddLambda, lambda);
                #[cfg(test)]
                let alpha = self.corrupt_value(Tamper::CompleteAddAlpha, alpha);
                #[cfg(test)]
                let beta = self.corrupt_value(Tamper::CompleteAddBeta, beta);
                #[cfg(test)]
                let gamma = self.corrupt_value(Tamper::CompleteAddGamma, gamma);
                #[cfg(test)]
                let delta = self.corrupt_value(Tamper::CompleteAddDelta, delta);

                for (name, column, value) in [
                    ("lambda", self.advices[4], lambda),
                    ("alpha", self.advices[5], alpha),
                    ("beta", self.advices[6], beta),
                    ("gamma", self.advices[7], gamma),
                    ("delta", self.advices[8], delta),
                ] {
                    region.assign_advice(|| name, column, 0, || value)?;
                }

                let result = x_p.zip(y_p).zip(x_q).zip(y_q).zip(lambda).map(
                    |((((x_p, y_p), x_q), y_q), lambda)| {
                        if bool::from(x_p.is_zero()) {
                            (x_q, y_q)
                        } else if bool::from(x_q.is_zero()) {
                            (x_p, y_p)
                        } else if x_q == x_p && y_q == -y_p {
                            (pallas::Base::zero(), pallas::Base::zero())
                        } else {
                            let x_r = lambda.square() - x_p - x_q;
                            let y_r = lambda * (x_p - x_r) - y_p;
                            (x_r, y_r)
                        }
                    },
                );
                let x = region.assign_advice(
                    || "x_r",
                    self.advices[2],
                    1,
                    || result.map(|result| result.0),
                )?;
                let y = region.assign_advice(
                    || "y_r",
                    self.advices[3],
                    1,
                    || result.map(|result| result.1),
                )?;
                Ok(AssignedPoint { x, y })
            },
        )
    }
}

fn invert_or_zero(value: pallas::Base) -> pallas::Base {
    Option::<pallas::Base>::from(value.invert()).unwrap_or(pallas::Base::zero())
}

fn window_scalar(window: usize, digit: usize) -> pallas::Scalar {
    let radix = pallas::Scalar::from(H as u64);
    if window < NUM_WINDOWS - 1 {
        pallas::Scalar::from(digit as u64 + 2) * radix.pow([window as u64, 0, 0, 0])
    } else {
        let offset = (0..(NUM_WINDOWS - 1)).fold(pallas::Scalar::zero(), |acc, window| {
            acc + pallas::Scalar::from(2).pow([(WINDOW_BITS * window + 1) as u64, 0, 0, 0])
        });
        pallas::Scalar::from(digit as u64) * radix.pow([(NUM_WINDOWS - 1) as u64, 0, 0, 0]) - offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SHARE_VALUE_LIMIT;
    use group::Group;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error, Instance},
    };

    const TEST_K: u32 = 6;
    const TEST_SHARE: u64 = 625;

    #[derive(Clone, Debug)]
    struct TestConfig {
        fixed_base_30: SpendAuthGFixedBase30Config,
        witness: Column<Advice>,
        instance: Column<Instance>,
    }

    #[derive(Clone, Debug, Default)]
    struct TestCircuit {
        share: Value<pallas::Base>,
        addend: Value<(pallas::Base, pallas::Base)>,
        tamper: Option<Tamper>,
    }

    impl Circuit<pallas::Base> for TestCircuit {
        type Config = TestConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
            let advices = core::array::from_fn(|_| meta.advice_column());
            for advice in advices {
                meta.enable_equality(advice);
            }
            let lagrange_coeffs = core::array::from_fn(|_| meta.fixed_column());
            let constants = meta.fixed_column();
            meta.enable_constant(constants);
            let instance = meta.instance_column();
            meta.enable_equality(instance);

            TestConfig {
                fixed_base_30: SpendAuthGFixedBase30Config::configure(
                    meta,
                    advices,
                    lagrange_coeffs,
                ),
                witness: advices[0],
                instance,
            }
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<pallas::Base>,
        ) -> Result<(), Error> {
            if self.tamper == Some(Tamper::RunningSumWord) {
                return config.fixed_base_30.assign_tampered_running_sum(layouter);
            }
            config.fixed_base_30.tamper = self.tamper;

            let (share, addend_x, addend_y) = layouter.assign_region(
                || "witness inputs",
                |mut region| {
                    let share =
                        region.assign_advice(|| "share", config.witness, 0, || self.share)?;
                    let addend_x = region.assign_advice(
                        || "addend x",
                        config.witness,
                        1,
                        || self.addend.map(|coordinates| coordinates.0),
                    )?;
                    let addend_y = region.assign_advice(
                        || "addend y",
                        config.witness,
                        2,
                        || self.addend.map(|coordinates| coordinates.1),
                    )?;
                    Ok((share, addend_x, addend_y))
                },
            )?;

            let (result_x, result_y) = config.fixed_base_30.mul_add(
                layouter.namespace(|| "test mul-add"),
                &share,
                &addend_x,
                &addend_y,
            )?;
            layouter.constrain_instance(result_x.cell(), config.instance, 0)?;
            layouter.constrain_instance(result_y.cell(), config.instance, 1)
        }
    }

    fn point_coordinates(point: pallas::Affine) -> (pallas::Base, pallas::Base) {
        let coordinates = point.coordinates();
        if bool::from(coordinates.is_some()) {
            let coordinates = coordinates.unwrap();
            (*coordinates.x(), *coordinates.y())
        } else {
            (pallas::Base::zero(), pallas::Base::zero())
        }
    }

    fn run_mul_add(
        share: u64,
        addend: pallas::Affine,
        expected: pallas::Affine,
        tamper: Option<Tamper>,
    ) -> MockProver<pallas::Base> {
        let expected = point_coordinates(expected);
        let circuit = TestCircuit {
            share: Value::known(pallas::Base::from(share)),
            addend: Value::known(point_coordinates(addend)),
            tamper,
        };
        MockProver::run(TEST_K, &circuit, vec![vec![expected.0, expected.1]]).unwrap()
    }

    fn test_mul_add(share: u64, should_succeed: bool) {
        let generator = spend_auth_g::generator();
        let addend = (generator * pallas::Scalar::from(13)).to_affine();
        let expected = (generator * pallas::Scalar::from(share) + addend).to_affine();
        let prover = run_mul_add(share, addend, expected, None);

        assert_eq!(prover.verify().is_ok(), should_succeed);
    }

    fn assert_tamper_fails_at_gate(tamper: Tamper, expected_gate: &str) {
        let prover = if tamper == Tamper::RunningSumWord {
            let circuit = TestCircuit {
                tamper: Some(tamper),
                ..TestCircuit::default()
            };
            MockProver::run(TEST_K, &circuit, vec![vec![]]).unwrap()
        } else {
            let generator = spend_auth_g::generator();
            let addend = (generator * pallas::Scalar::from(13)).to_affine();
            let expected = (generator * pallas::Scalar::from(TEST_SHARE) + addend).to_affine();
            run_mul_add(TEST_SHARE, addend, expected, Some(tamper))
        };

        let failures = prover.verify().expect_err("tampered witness must fail");
        let failures = failures
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            failures.contains(expected_gate),
            "expected failure in gate '{expected_gate}', got:\n{failures}"
        );
    }

    #[test]
    fn window_scalars_reconstruct_share() {
        for value in [0, 1, 7, 8, 625, SHARE_VALUE_LIMIT - 1] {
            let reconstructed =
                (0..NUM_WINDOWS).fold(pallas::Scalar::zero(), |accumulator, window| {
                    let digit = ((value >> (WINDOW_BITS * window)) & (H as u64 - 1)) as usize;
                    accumulator + window_scalar(window, digit)
                });

            assert_eq!(reconstructed, pallas::Scalar::from(value));
        }
    }

    #[test]
    fn final_window_constants_select_positive_y() {
        for (digit, u_repr) in FINAL_WINDOW_U.iter().enumerate() {
            let point =
                (spend_auth_g::generator() * window_scalar(NUM_WINDOWS - 1, digit)).to_affine();
            let y = *point.coordinates().unwrap().y();
            let u = pallas::Base::from_repr(*u_repr).unwrap();
            let z = pallas::Base::from(FINAL_WINDOW_Z);

            assert_eq!(u.square(), y + z);
            assert!(bool::from((z - y).sqrt().is_none()));
        }
    }

    #[test]
    fn mul_add_accepts_zero_and_maximum_share() {
        test_mul_add(0, true);
        test_mul_add(SHARE_VALUE_LIMIT - 1, true);
    }

    #[test]
    fn mul_add_rejects_share_at_limit() {
        test_mul_add(SHARE_VALUE_LIMIT, false);
    }

    #[test]
    fn mul_add_accepts_doubling() {
        let generator = spend_auth_g::generator();
        let magnitude = generator * pallas::Scalar::from(TEST_SHARE);
        let prover = run_mul_add(
            TEST_SHARE,
            magnitude.to_affine(),
            (magnitude + magnitude).to_affine(),
            None,
        );

        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn mul_add_accepts_inverse_points() {
        let generator = spend_auth_g::generator();
        let magnitude = generator * pallas::Scalar::from(TEST_SHARE);
        let prover = run_mul_add(
            TEST_SHARE,
            (-magnitude).to_affine(),
            pallas::Point::identity().to_affine(),
            None,
        );

        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn mul_add_accepts_identity_addend() {
        let generator = spend_auth_g::generator();
        let magnitude = generator * pallas::Scalar::from(TEST_SHARE);
        let prover = run_mul_add(
            TEST_SHARE,
            pallas::Point::identity().to_affine(),
            magnitude.to_affine(),
            None,
        );

        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn rejects_selected_y_sign_flip() {
        assert_tamper_fails_at_gate(Tamper::SelectedY, "30-bit fixed-base window coordinates");
    }

    #[test]
    fn rejects_selected_u_corruption() {
        assert_tamper_fails_at_gate(Tamper::SelectedU, "30-bit fixed-base window coordinates");
    }

    #[test]
    fn rejects_incomplete_add_output_corruption() {
        assert_tamper_fails_at_gate(
            Tamper::IncompleteAddX,
            "30-bit fixed-base incomplete addition",
        );
    }

    #[test]
    fn rejects_complete_add_lambda_corruption() {
        assert_tamper_fails_at_gate(
            Tamper::CompleteAddLambda,
            "30-bit fixed-base complete addition",
        );
    }

    #[test]
    fn rejects_complete_add_alpha_corruption() {
        assert_tamper_fails_at_gate(
            Tamper::CompleteAddAlpha,
            "30-bit fixed-base complete addition",
        );
    }

    #[test]
    fn rejects_complete_add_beta_corruption() {
        assert_tamper_fails_at_gate(
            Tamper::CompleteAddBeta,
            "30-bit fixed-base complete addition",
        );
    }

    #[test]
    fn rejects_complete_add_gamma_corruption() {
        assert_tamper_fails_at_gate(
            Tamper::CompleteAddGamma,
            "30-bit fixed-base complete addition",
        );
    }

    #[test]
    fn rejects_complete_add_delta_corruption() {
        assert_tamper_fails_at_gate(
            Tamper::CompleteAddDelta,
            "30-bit fixed-base complete addition",
        );
    }

    #[test]
    fn rejects_out_of_range_running_sum_word() {
        assert_tamper_fails_at_gate(Tamper::RunningSumWord, "range check");
    }
}
