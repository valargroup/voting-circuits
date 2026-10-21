//! Canonical Orchard encodings at the governance circuit boundary.
#[cfg(feature = "lrz-orchard")]
use crate::pasta_curves::group::Curve;
use crate::pasta_curves::group::{ff::PrimeField, GroupEncoding};
use crate::{orchard, pasta_curves::pallas};

/// Converts a typed protocol value into its mathematical circuit representation.
pub trait CircuitValue {
    /// The field or group used by the selected circuit backend.
    type Value;
    /// Decodes the protocol owner's canonical encoding for use in the circuit.
    fn circuit_value(&self) -> Self::Value;
}

macro_rules! field_value {
    ($ty:ty, $field:ty) => {
        impl CircuitValue for $ty {
            type Value = $field;
            fn circuit_value(&self) -> Self::Value {
                #[cfg(feature = "orchard")]
                {
                    <$field>::from_repr(self.to_bytes()).unwrap()
                }
                #[cfg(feature = "lrz-orchard")]
                {
                    self.inner()
                }
            }
        }
    };
}
field_value!(crate::NullifierDerivingKey, pallas::Base);
field_value!(crate::CommitIvkRandomness, pallas::Scalar);
field_value!(crate::NoteTrapdoor, pallas::Scalar);
field_value!(orchard::note::ExtractedNoteCommitment, pallas::Base);
field_value!(orchard::note::Nullifier, pallas::Base);
field_value!(orchard::tree::MerkleHashOrchard, pallas::Base);

impl CircuitValue for orchard::note::Rho {
    type Value = pallas::Base;
    fn circuit_value(&self) -> Self::Value {
        pallas::Base::from_repr(self.to_bytes()).unwrap()
    }
}
impl CircuitValue for orchard::note::NoteCommitment {
    type Value = pallas::Point;
    fn circuit_value(&self) -> Self::Value {
        #[cfg(feature = "orchard")]
        {
            pallas::Point::from_bytes(
                &orchard::circuit::composition::witness::NoteCommitment::from(self).to_bytes(),
            )
            .unwrap()
        }
        #[cfg(feature = "lrz-orchard")]
        {
            self.inner()
        }
    }
}
impl CircuitValue for orchard::keys::SpendValidatingKey {
    type Value = pallas::Point;
    fn circuit_value(&self) -> Self::Value {
        #[cfg(feature = "orchard")]
        {
            pallas::Point::from_bytes(
                &orchard::circuit::composition::witness::SpendValidatingKey::from(self).to_bytes(),
            )
            .unwrap()
        }
        #[cfg(feature = "lrz-orchard")]
        {
            self.into()
        }
    }
}

/// Returns the note-randomness witness through the selected protocol API.
pub fn note_psi(seed: &orchard::note::RandomSeed, rho: &orchard::note::Rho) -> pallas::Base {
    #[cfg(feature = "orchard")]
    {
        pallas::Base::from_repr(
            orchard::circuit::composition::witness::NotePsi::derive(seed, rho).to_bytes(),
        )
        .unwrap()
    }
    #[cfg(feature = "lrz-orchard")]
    {
        seed.psi(rho)
    }
}

/// Derives the same commitment trapdoor for standard and synthetic governance notes.
pub fn note_trapdoor(
    version: orchard::note::NoteVersion,
    seed: &orchard::note::RandomSeed,
    rho: &orchard::note::Rho,
    g_d: &crate::composition::NonIdentityPallasPoint,
    pk_d: &crate::composition::NonIdentityPallasPoint,
    value: orchard::value::NoteValue,
    psi: pallas::Base,
) -> pallas::Scalar {
    match version {
        orchard::note::NoteVersion::V2 => {
            #[cfg(feature = "orchard")]
            {
                orchard::circuit::composition::witness::NoteCommitTrapdoor::derive_v2(seed, rho)
                    .circuit_value()
            }
            #[cfg(feature = "lrz-orchard")]
            {
                seed.rcm_v2(rho).inner()
            }
        }
        orchard::note::NoteVersion::V3 => {
            #[cfg(feature = "orchard")]
            {
                let psi =
                    orchard::circuit::composition::witness::NotePsi::from_bytes(psi.to_repr())
                        .unwrap();
                orchard::circuit::composition::witness::NoteCommitTrapdoor::derive_v3(
                    seed,
                    rho,
                    &orchard::circuit::composition::witness::NonIdentityPallasPoint::from_bytes(
                        &g_d.to_bytes(),
                    )
                    .unwrap(),
                    &orchard::circuit::composition::witness::NonIdentityPallasPoint::from_bytes(
                        &pk_d.to_bytes(),
                    )
                    .unwrap(),
                    value,
                    &psi,
                )
                .circuit_value()
            }
            #[cfg(feature = "lrz-orchard")]
            {
                seed.rcm_v3(rho, g_d, pk_d, value.inner(), &psi).inner()
            }
        }
    }
}

/// Returns the sign-corrected spending scalar used by the governance witness.
///
/// This scalar permits spending. Keep it in the trusted signer; never send it to
/// an untrusted proving service or include it in public inputs.
pub fn spend_authorizing_scalar(sk: &orchard::keys::SpendingKey) -> pallas::Scalar {
    #[cfg(feature = "orchard")]
    {
        let ask = orchard::keys::SpendAuthorizingKey::from(sk);
        pallas::Scalar::from_repr(
            orchard::circuit::composition::witness::SpendAuthorizingKey::from(&ask).to_bytes(),
        )
        .unwrap()
    }
    #[cfg(feature = "lrz-orchard")]
    {
        let ask = orchard::keys::SpendAuthorizingKey::derive_inner(sk);
        let generator = orchard::constants::fixed_bases::spend_auth_g::generator();
        let encoded = (generator * ask).to_affine().to_bytes();
        if encoded[31] >> 7 == 1 {
            -ask
        } else {
            ask
        }
    }
}

/// Mathematical point at the circuit boundary, decoded through Orchard's canonical adapter.
#[cfg(feature = "orchard")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NonIdentityPallasPoint(pallas::Point);
#[cfg(feature = "orchard")]
impl NonIdentityPallasPoint {
    pub(crate) fn from_witness(
        point: orchard::circuit::composition::witness::NonIdentityPallasPoint,
    ) -> Self {
        Self(pallas::Point::from_bytes(&point.to_bytes()).unwrap())
    }
    pub fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
        orchard::circuit::composition::witness::NonIdentityPallasPoint::from_bytes(bytes)
            .into_option()
            .map(Self::from_witness)
    }
}
#[cfg(feature = "orchard")]
impl core::ops::Deref for NonIdentityPallasPoint {
    type Target = pallas::Point;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
