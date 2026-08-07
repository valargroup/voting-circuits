//! El Gamal encryption integrity gadget for vote proof (ZKP #2).
//!
//! Proves that sixteen ciphertext pairs
//! (enc_share_c1_x/y[i], enc_share_c2_x/y[i]) are valid El Gamal encryptions of
//! the corresponding plaintext shares under the election authority public key:
//! C1_i = [r_i]*G, C2_i = [v_i]*G + [r_i]*ea_pk.
//!
//! Used by the vote proof circuit (Condition 11: Encryption Integrity). The
//! caller passes share cells, randomness cells, and enc_share coordinate cells;
//! this gadget owns all ea_pk scaffolding (witnesses ea_pk once as a
//! `NonIdentityPoint` and pins it to the instance column via `constrain_instance`)
//! and handles G for C1 via `FixedPointBaseField`. C2's [v_i]*G term uses a
//! custom unsigned ten-window multiplication specialized to 30-bit shares.
//!
//! Also provides the public `spend_auth_g_affine` helper for downstream
//! consumers and internal scalar/encryption helpers for the builder and tests.

#[cfg(test)]
use ff::Field;
use halo2_gadgets::ecc::{chip::EccChip, FixedPointBaseField, NonIdentityPoint, ScalarVar};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::{Column, Error, Instance as InstanceColumn},
};
use orchard::constants::{OrchardBaseFieldBases, OrchardFixedBases};
#[cfg(test)]
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::pallas;

use super::nonzero::NonZeroConfig;

mod fixed_base_30;

pub(crate) use fixed_base_30::SpendAuthGFixedBase30Config;

// ================================================================
// Instance-location descriptor
// ================================================================

/// Describes where ea_pk lives in the public-input (instance) column.
///
/// The gadget uses this to call `layouter.constrain_instance` directly
/// on the witnessed NonIdentityPoint cells, removing the need for the
/// caller to pre-allocate advice-from-instance cells and pass them down.
pub(crate) struct EaPkInstanceLoc {
    /// The instance column that holds the public inputs.
    pub instance: Column<InstanceColumn>,
    /// Row of the ea_pk x-coordinate in the instance column.
    pub x_row: usize,
    /// Row of the ea_pk y-coordinate in the instance column.
    pub y_row: usize,
}

// ================================================================
// Out-of-circuit helpers
// ================================================================

/// Returns the SpendAuthG generator point (used as G in El Gamal).
///
/// Why SpendAuthG? El Gamal requires a prime-order generator with an unknown
/// discrete log. SpendAuthG is derived via `GroupPHash("z.cash:Orchard", "G")`
/// — a nothing-up-my-sleeve point. Using it for El Gamal (Condition 11) avoids
/// introducing a second generator point; the custom 30-bit table shares the
/// same generator as the full-scalar SpendAuthG.
pub fn spend_auth_g_affine() -> pallas::Affine {
    orchard::constants::fixed_bases::spend_auth_g::generator()
}

/// Converts a `pallas::Base` field element to a `pallas::Scalar`.
///
/// Pallas's base field modulus is smaller than its scalar field modulus, so
/// every canonical `pallas::Base` element is representable as a scalar. The
/// `Option` return keeps that invariant explicit at the call sites.
pub(crate) fn base_to_scalar(b: pallas::Base) -> Option<pallas::Scalar> {
    use ff::PrimeField;
    pallas::Scalar::from_repr(b.to_repr()).into()
}

/// Errors from out-of-circuit El Gamal encryption helpers.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ElGamalEncryptError {
    /// The randomness is zero, which would make C1 the identity.
    ZeroRandomness,
    /// The share value could not be represented as a scalar.
    InvalidShareValue,
    /// The randomness could not be represented as a scalar.
    InvalidRandomness,
    /// A derived ciphertext component was the identity point.
    IdentityCiphertext { component: &'static str },
}

#[cfg(test)]
impl core::fmt::Display for ElGamalEncryptError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ElGamalEncryptError::ZeroRandomness => {
                write!(f, "El Gamal randomness must be non-zero")
            }
            ElGamalEncryptError::InvalidShareValue => {
                write!(f, "share value is not representable as a scalar")
            }
            ElGamalEncryptError::InvalidRandomness => {
                write!(f, "randomness is not representable as a scalar")
            }
            ElGamalEncryptError::IdentityCiphertext { component } => {
                write!(f, "El Gamal {component} point is the identity")
            }
        }
    }
}

#[cfg(test)]
impl std::error::Error for ElGamalEncryptError {}

/// Test-only out-of-circuit El Gamal encryption under SpendAuthG.
///
/// Computes C1 = [r]*SpendAuthG, C2 = [v]*SpendAuthG + [r]*ea_pk.
/// Returns (c1_x, c2_x, c1_y, c2_y). Used by tests as a circuit oracle.
///
/// Both coordinates are returned so that share commitments can bind to the
/// full curve point, preventing ciphertext sign-malleability attacks.
#[cfg(test)]
pub(crate) fn elgamal_encrypt(
    share_value: pallas::Base,
    randomness: pallas::Base,
    ea_pk: pallas::Affine,
) -> Result<(pallas::Base, pallas::Base, pallas::Base, pallas::Base), ElGamalEncryptError> {
    use group::Curve;

    let g = spend_auth_g_affine();
    if bool::from(randomness.is_zero()) {
        return Err(ElGamalEncryptError::ZeroRandomness);
    }
    let r_scalar = base_to_scalar(randomness).ok_or(ElGamalEncryptError::InvalidRandomness)?;
    let v_scalar = base_to_scalar(share_value).ok_or(ElGamalEncryptError::InvalidShareValue)?;

    let c1 = g * r_scalar;
    let c2 = g * v_scalar + ea_pk * r_scalar;

    let c1_affine = c1.to_affine();
    let c2_affine = c2.to_affine();
    let c1_coords = c1_affine
        .coordinates()
        .into_option()
        .ok_or(ElGamalEncryptError::IdentityCiphertext { component: "C1" })?;
    let c2_coords = c2_affine
        .coordinates()
        .into_option()
        .ok_or(ElGamalEncryptError::IdentityCiphertext { component: "C2" })?;
    Ok((
        *c1_coords.x(),
        *c2_coords.x(),
        *c1_coords.y(),
        *c2_coords.y(),
    ))
}

// ================================================================
// In-circuit gadget
// ================================================================

/// Proves that for each share i, (enc_c1_x/y[i], enc_c2_x/y[i]) is a
/// valid El Gamal encryption of share_cells[i] under randomness r_cells[i]
/// and public key ea_pk: C1_i = [r_i]*G, C2_i = [v_i]*G + [r_i]*ea_pk.
///
/// ## Generator handling
///
/// - **C1 = [r_i]*G**: uses `FixedPointBaseField::mul` (85-window, full-field scalar).
///   `r_i` is a 255-bit scalar, so the full decomposition is required. This
///   crate rejects `r_i = 0` with a small inverse-witness constraint to prevent
///   the exact degenerate self-leaking ciphertext.
/// - **C2's [v_i]*G**: uses a custom ten-window, unsigned fixed-base
///   multiplication for the exact 30-bit share range. Its final complete
///   addition is fused with the addition of `[r_i]ea_pk`.
///
/// ## Soundness
///
/// The custom fixed-base gadget copies the caller-supplied `share_cells[i]`
/// directly into a strict 30-bit decomposition. Because conditions 8 and 9
/// constrain the same source cell, the encryption scalar is identical to the
/// summed and range-checked share value. The unsigned decomposition eliminates
/// the separate sign witness and prevents negation.
///
/// ## Gadget ownership
///
/// The gadget witnesses ea_pk internally as a `NonIdentityPoint` and pins both
/// coordinates to the public instance column. This proves encryption under the
/// caller-supplied key, but the caller must authenticate that key against the
/// active round's governance announcement. The caller need only supply the
/// share, randomness, and ciphertext coordinate arrays plus the ea_pk value.
pub(crate) fn prove_elgamal_encryptions(
    ecc_chip: EccChip<OrchardFixedBases>,
    nonzero: NonZeroConfig,
    fixed_base_30: &SpendAuthGFixedBase30Config,
    mut layouter: impl Layouter<pallas::Base>,
    namespace: &str,
    ea_pk: halo2_proofs::circuit::Value<pallas::Affine>,
    ea_pk_loc: EaPkInstanceLoc,
    share_cells: [AssignedCell<pallas::Base, pallas::Base>; 16],
    r_cells: [AssignedCell<pallas::Base, pallas::Base>; 16],
    enc_c1_cells: [AssignedCell<pallas::Base, pallas::Base>; 16],
    enc_c2_cells: [AssignedCell<pallas::Base, pallas::Base>; 16],
    enc_c1_y_cells: [AssignedCell<pallas::Base, pallas::Base>; 16],
    enc_c2_y_cells: [AssignedCell<pallas::Base, pallas::Base>; 16],
) -> Result<(), Error> {
    // Election Authority's public key as a Pallas curve point, wrapped in Value.
    // ea_pk must be witnessed into advice cells to compute [r_i] * ea_pk.
    // The constrain_instance calls bind those advice cells to the public instance
    // column, giving the verifier a guarantee that the prover used the specific
    // EA key declared publicly. The caller is still responsible for checking
    // that those public coordinates are the governance-announced EA key.

    // Witness ea_pk once. NonIdentityPoint is Copy, so the value is cheaply
    // copied into each iteration's mul() call without re-witnessing.
    let ea_pk_point = NonIdentityPoint::new(
        ecc_chip.clone(),
        layouter.namespace(|| format!("{namespace} ea_pk witness")),
        ea_pk,
    )?;
    // Pin the witness directly to the public-input column.
    layouter.constrain_instance(
        ea_pk_point.inner().x().cell(),
        ea_pk_loc.instance,
        ea_pk_loc.x_row,
    )?;
    layouter.constrain_instance(
        ea_pk_point.inner().y().cell(),
        ea_pk_loc.instance,
        ea_pk_loc.y_row,
    )?;

    // SpendAuthG fixed-base descriptor for C1's [r_i]*G (full 85-window path).
    // r_i is a 255-bit base-field scalar, requiring the full decomposition.
    let spend_auth_g_base =
        FixedPointBaseField::from_inner(ecc_chip.clone(), OrchardBaseFieldBases::SpendAuthGBase);

    for i in 0..16 {
        nonzero.constrain_nonzero(
            layouter.namespace(|| format!("{namespace} r[{i}] != 0")),
            "El Gamal randomness != 0",
            &r_cells[i],
        )?;

        // --- C1_i = [r_i] * G ---
        //
        // G is baked into the fixed-base lookup table; no NonIdentityPoint
        // witness or constrain_equal needed for the base point.
        let c1_point = spend_auth_g_base.clone().mul(
            layouter.namespace(|| format!("{namespace} [r_{i}] * G")),
            r_cells[i].clone(),
        )?;

        // Both coordinates of C1 are constrained: x via extract_p, y via
        // the inner point. The y-coordinate binding prevents ciphertext
        // sign-malleability (negating a point preserves x but flips y).
        let c1_x = c1_point.extract_p().inner().clone();
        layouter.assign_region(
            || format!("{namespace} C1[{i}] x == enc_c1_x[{i}]"),
            |mut region| region.constrain_equal(c1_x.cell(), enc_c1_cells[i].cell()),
        )?;
        let c1_y = c1_point.inner().y();
        layouter.assign_region(
            || format!("{namespace} C1[{i}] y == enc_c1_y[{i}]"),
            |mut region| region.constrain_equal(c1_y.cell(), enc_c1_y_cells[i].cell()),
        )?;

        // --- C2_i = [v_i] * G + [r_i] * ea_pk ---
        //
        let r_i_scalar = ScalarVar::from_base(
            ecc_chip.clone(),
            layouter.namespace(|| format!("{namespace} r[{i}] to ScalarVar")),
            &r_cells[i],
        )?;
        // ea_pk_point is Copy: no new witness cells, just copies the AssignedCell
        // references for this mul.
        let (r_ea_pk_point, _) = ea_pk_point.mul(
            layouter.namespace(|| format!("{namespace} [r_{i}] * ea_pk")),
            r_i_scalar,
        )?;

        let addend_x = r_ea_pk_point.inner().x();
        let addend_y = r_ea_pk_point.inner().y();
        let (c2_x, c2_y) = fixed_base_30.mul_add(
            layouter.namespace(|| format!("{namespace} C2[{i}] = vG + rP")),
            &share_cells[i],
            &addend_x,
            &addend_y,
        )?;

        layouter.assign_region(
            || format!("{namespace} C2[{i}] x == enc_c2_x[{i}]"),
            |mut region| region.constrain_equal(c2_x.cell(), enc_c2_cells[i].cell()),
        )?;
        layouter.assign_region(
            || format!("{namespace} C2[{i}] y == enc_c2_y[{i}]"),
            |mut region| region.constrain_equal(c2_y.cell(), enc_c2_y_cells[i].cell()),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use group::{prime::PrimeCurveAffine, Curve};

    #[test]
    fn elgamal_encrypt_rejects_zero_randomness() {
        let ea_pk = spend_auth_g_affine();
        let err = elgamal_encrypt(pallas::Base::from(1), pallas::Base::zero(), ea_pk)
            .expect_err("zero randomness should be rejected without panicking");

        assert_eq!(err, ElGamalEncryptError::ZeroRandomness);
    }

    #[test]
    fn elgamal_encrypt_returns_slots_in_documented_order() {
        let share_value = pallas::Base::from(7u64);
        let randomness = pallas::Base::from(11u64);
        let g = spend_auth_g_affine();
        let ea_pk = (g * pallas::Scalar::from(13u64)).to_affine();

        let (c1_x, c2_x, c1_y, c2_y) = elgamal_encrypt(share_value, randomness, ea_pk)
            .expect("test encryption inputs should produce non-identity ciphertext points");

        let r_scalar = base_to_scalar(randomness).expect("test randomness should fit scalar field");
        let v_scalar = base_to_scalar(share_value).expect("test share should fit scalar field");
        let expected_c1 = (g * r_scalar).to_affine();
        let expected_c2 = (g * v_scalar + ea_pk * r_scalar).to_affine();
        let expected_c1_coords = expected_c1
            .coordinates()
            .into_option()
            .expect("non-zero randomness should produce non-identity C1");
        let expected_c2_coords = expected_c2
            .coordinates()
            .into_option()
            .expect("chosen test inputs should produce non-identity C2");

        assert_eq!(c1_x, *expected_c1_coords.x(), "slot 0 must be C1.x");
        assert_eq!(c2_x, *expected_c2_coords.x(), "slot 1 must be C2.x");
        assert_eq!(c1_y, *expected_c1_coords.y(), "slot 2 must be C1.y");
        assert_eq!(c2_y, *expected_c2_coords.y(), "slot 3 must be C2.y");
    }

    #[test]
    fn elgamal_encrypt_rejects_identity_c2() {
        let err = elgamal_encrypt(
            pallas::Base::zero(),
            pallas::Base::from(1),
            pallas::Affine::identity(),
        )
        .expect_err("zero share under identity key should produce identity C2");

        assert_eq!(
            err,
            ElGamalEncryptError::IdentityCiphertext { component: "C2" }
        );
    }
}
