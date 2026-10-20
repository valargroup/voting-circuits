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
#[cfg(feature = "lrz-orchard")]
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
impl CircuitValue for orchard::keys::SpendValidatingKey {
    type Value = pallas::Point;
    fn circuit_value(&self) -> Self::Value {
        #[cfg(feature = "orchard")]
        {
            let zero = crate::spend_auth_randomizer(&pallas::Scalar::from(0));
            let bytes: [u8; 32] = self.randomize(&zero).into();
            pallas::Point::from_bytes(&bytes).unwrap()
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
        crate::legacy::witness::note_psi(seed, rho)
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
    #[cfg(feature = "orchard")]
    {
        crate::legacy::witness::note_trapdoor(version, seed, rho, g_d, pk_d, value, psi)
    }
    #[cfg(feature = "lrz-orchard")]
    {
        match version {
            orchard::note::NoteVersion::V2 => seed.rcm_v2(rho).inner(),
            orchard::note::NoteVersion::V3 => {
                seed.rcm_v3(rho, g_d, pk_d, value.inner(), &psi).inner()
            }
        }
    }
}

/// Returns the sign-corrected spending scalar used by the governance witness.
///
/// This is spend-capable secret material. Keep it in the trusted signing/proving
/// boundary; never log it or send it to an untrusted helper or public instance.
pub fn spend_authorizing_scalar(sk: &orchard::keys::SpendingKey) -> pallas::Scalar {
    #[cfg(feature = "orchard")]
    {
        let ask = orchard::keys::SpendAuthorizingKey::from(sk);
        let zero = crate::spend_auth_randomizer(&pallas::Scalar::from(0));
        let bytes: [u8; 32] = (&ask.randomize(&zero)).into();
        pallas::Scalar::from_repr(bytes).unwrap()
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
    /// Rejects invalid, noncanonical, and identity point encodings.
    pub fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
        use crate::pasta_curves::group::Group;
        pallas::Point::from_bytes(bytes)
            .into_option()
            .filter(|p| !bool::from(p.is_identity()))
            .map(Self)
    }
}
#[cfg(feature = "orchard")]
impl core::ops::Deref for NonIdentityPallasPoint {
    type Target = pallas::Point;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Complete commitment point for a validated protocol note at the circuit boundary.
pub fn note_commitment(note: &orchard::note::Note) -> pallas::Point {
    #[cfg(feature = "orchard")]
    {
        crate::legacy::witness::note_commitment(note)
    }
    #[cfg(feature = "lrz-orchard")]
    {
        note.commitment().inner()
    }
}

/// Native NoteCommit at the circuit boundary, including synthetic governance components.
///
/// Inherited from Orchard NoteCommit and the existing voting padding derivation.
/// Caller must enforce nonidentity components, ownership, and membership where
/// required. This operation alone establishes none of those properties.
pub fn note_commitment_point(
    g_d: pallas::Point,
    pk_d: pallas::Point,
    value: orchard::value::NoteValue,
    rho: pallas::Base,
    psi: pallas::Base,
    rcm: pallas::Scalar,
) -> Option<pallas::Point> {
    use crate::composition::{L_ORCHARD_BASE, L_VALUE, NOTE_COMMITMENT_PERSONALIZATION};
    use crate::pasta_curves::group::ff::PrimeFieldBits;
    let bits = |bytes: [u8; 32]| {
        bytes
            .into_iter()
            .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1 == 1))
    };
    crate::sinsemilla::CommitDomain::new(NOTE_COMMITMENT_PERSONALIZATION)
        .commit(
            bits(g_d.to_bytes())
                .chain(bits(pk_d.to_bytes()))
                .chain((0..L_VALUE).map(|i| (value.inner() >> i) & 1 == 1))
                .chain(rho.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE))
                .chain(psi.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE)),
            &rcm,
        )
        .into_option()
}
