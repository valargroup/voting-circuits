//! El Gamal helpers and the 30-bit fixed-base gadget shared by the vote
//! circuits.
//!
//! The production encryption-integrity constraints live in the encrypt-choice
//! circuit (ZKP 1.5), which consumes [`SpendAuthGFixedBase30Config`] from this
//! module.
//!
//! Also provides the public `spend_auth_g_affine` helper for downstream
//! consumers and internal scalar/encryption helpers for the builders and
//! tests.

#[cfg(test)]
use crate::ff::Field;
#[cfg(test)]
use voting_crypto_deps::pasta_curves::arithmetic::CurveAffine;
use voting_crypto_deps::pasta_curves::pallas;

mod fixed_base_30;

pub(crate) use fixed_base_30::SpendAuthGFixedBase30Config;

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
    voting_crypto_deps::orchard::constants::fixed_bases::spend_auth_g::generator()
}

/// Converts a `pallas::Base` field element to a `pallas::Scalar`.
///
/// Pallas's base field modulus is smaller than its scalar field modulus, so
/// every canonical `pallas::Base` element is representable as a scalar. The
/// `Option` return keeps that invariant explicit at the call sites.
pub(crate) fn base_to_scalar(b: pallas::Base) -> Option<pallas::Scalar> {
    use crate::ff::PrimeField;
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
    use crate::group::Curve;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::Curve;

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
            pallas::Affine::default(),
        )
        .expect_err("zero share under identity key should produce identity C2");

        assert_eq!(
            err,
            ElGamalEncryptError::IdentityCiphertext { component: "C2" }
        );
    }
}
