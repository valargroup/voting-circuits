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
field_value!(orchard::keys::NullifierDerivingKey, pallas::Base);
field_value!(orchard::keys::CommitIvkRandomness, pallas::Scalar);
field_value!(crate::VotingNoteTrapdoor, pallas::Scalar);
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
            pallas::Point::from_bytes(&self.to_bytes()).unwrap()
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
            pallas::Point::from_bytes(&self.to_bytes()).unwrap()
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
        pallas::Base::from_repr(seed.voting_psi(rho).to_bytes()).unwrap()
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
    g_d: &crate::voting::NonIdentityPallasPoint,
    pk_d: &crate::voting::NonIdentityPallasPoint,
    value: orchard::value::NoteValue,
    psi: pallas::Base,
) -> pallas::Scalar {
    match version {
        orchard::note::NoteVersion::V2 => seed.rcm_v2(rho).circuit_value(),
        orchard::note::NoteVersion::V3 => {
            #[cfg(feature = "orchard")]
            {
                let psi = orchard::note::NotePsi::from_bytes(psi.to_repr()).unwrap();
                seed.voting_rcm_v3(rho, g_d, pk_d, value, &psi)
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
pub fn spend_authorizing_scalar(sk: &orchard::keys::SpendingKey) -> pallas::Scalar {
    #[cfg(feature = "orchard")]
    {
        let ask = orchard::keys::SpendAuthorizingKey::from(sk);
        pallas::Scalar::from_repr(ask.to_bytes()).unwrap()
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
