//! Selects one coherent cryptography dependency family for shielded voting.
//!
//! The default features reexport the complete Zakura package family. Consumers
//! that need a smaller graph can disable default features and select
//! fine-grained features such as `vct`. LRZ consumers use the corresponding
//! `lrz-*` features. Mixing the two package families, or selecting neither
//! family, is an error.
//!
//! Both dependency families require Rust 1.97.

#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

#[cfg(all(
    any(
        feature = "pasta",
        feature = "gadgets",
        feature = "poseidon",
        feature = "proofs",
        feature = "orchard",
        feature = "rand",
        feature = "sinsemilla",
        feature = "validator",
    ),
    any(
        feature = "lrz-pasta",
        feature = "lrz-gadgets",
        feature = "lrz-poseidon",
        feature = "lrz-proofs",
        feature = "lrz-orchard",
        feature = "lrz-rand",
        feature = "lrz-sinsemilla",
        feature = "lrz-validator",
    )
))]
compile_error!("Zakura and LRZ dependency features cannot be enabled together");

#[cfg(not(any(
    feature = "pasta",
    feature = "gadgets",
    feature = "poseidon",
    feature = "proofs",
    feature = "orchard",
    feature = "rand",
    feature = "sinsemilla",
    feature = "validator",
    feature = "lrz-pasta",
    feature = "lrz-gadgets",
    feature = "lrz-poseidon",
    feature = "lrz-proofs",
    feature = "lrz-orchard",
    feature = "lrz-rand",
    feature = "lrz-sinsemilla",
    feature = "lrz-validator",
)))]
compile_error!("enable at least one Zakura or LRZ dependency feature");

#[cfg(feature = "orchard")]
pub use ::incrementalmerkletree;
#[cfg(feature = "orchard")]
pub use ::zcash_note_encryption;
#[cfg(feature = "orchard")]
pub use ::zip32;
#[cfg(feature = "lrz-orchard")]
pub use lrz_incrementalmerkletree as incrementalmerkletree;
#[cfg(feature = "lrz-orchard")]
pub use lrz_zcash_note_encryption as zcash_note_encryption;
#[cfg(feature = "lrz-orchard")]
pub use lrz_zip32 as zip32;

#[cfg(feature = "gadgets")]
pub use ::halo2_gadgets;
#[cfg(feature = "poseidon")]
pub use ::halo2_poseidon;
#[cfg(feature = "proofs")]
pub use ::halo2_proofs;
#[cfg(feature = "orchard")]
pub use ::orchard;
#[cfg(feature = "pasta")]
pub use ::pasta_curves;
#[cfg(any(feature = "rand", feature = "lrz-rand"))]
pub mod rand {
    use ::rand as backend;

    pub use backend::{CryptoRng, Rng};

    pub mod rngs {
        use super::backend::{
            rand_core::{TryCryptoRng, TryRng, UnwrapErr},
            rngs::SysRng,
        };
        use core::convert::Infallible;

        #[derive(Clone, Copy, Debug, Default)]
        pub struct OsRng;

        impl TryRng for OsRng {
            type Error = Infallible;

            fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
                UnwrapErr(SysRng).try_next_u32()
            }

            fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
                UnwrapErr(SysRng).try_next_u64()
            }

            fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
                UnwrapErr(SysRng).try_fill_bytes(dst)
            }
        }

        impl TryCryptoRng for OsRng {}
    }
}
#[cfg(feature = "validator")]
pub use ::reddsa;
#[cfg(feature = "sinsemilla")]
pub use ::sinsemilla;
#[cfg(feature = "validator")]
pub use ::zcash_primitives;

#[cfg(feature = "lrz-gadgets")]
pub use lrz_halo2_gadgets as halo2_gadgets;
#[cfg(feature = "lrz-poseidon")]
pub use lrz_halo2_poseidon as halo2_poseidon;
#[cfg(feature = "lrz-proofs")]
pub use lrz_halo2_proofs as halo2_proofs;
#[cfg(feature = "lrz-orchard")]
pub use lrz_orchard as orchard;
#[cfg(feature = "lrz-pasta")]
pub use lrz_pasta_curves as pasta_curves;
#[cfg(feature = "lrz-validator")]
pub use lrz_reddsa as reddsa;
#[cfg(feature = "lrz-sinsemilla")]
pub use lrz_sinsemilla as sinsemilla;
#[cfg(feature = "lrz-validator")]
pub use lrz_zcash_primitives as zcash_primitives;

#[cfg(all(test, any(feature = "validator", feature = "lrz-validator")))]
mod tests {
    use super::{orchard, zcash_primitives};

    #[test]
    fn transaction_primitives_use_the_selected_orchard_type() {
        let parsed =
            zcash_primitives::transaction::components::orchard::read_action_without_auth(&[][..]);
        let _: Option<orchard::Action<()>> = parsed.ok();
    }
}

/// Canonical spend-authorization randomizer for the selected Orchard backend.
#[cfg(feature = "orchard")]
pub use orchard::primitives::redpallas::SpendAuthRandomizer;
#[cfg(feature = "lrz-orchard")]
pub type SpendAuthRandomizer = pasta_curves::pallas::Scalar;

/// Converts a circuit witness scalar to the selected backend's signing randomizer.
#[cfg(any(feature = "orchard", feature = "lrz-orchard"))]
pub fn spend_auth_randomizer(scalar: &pasta_curves::pallas::Scalar) -> SpendAuthRandomizer {
    #[cfg(feature = "orchard")]
    {
        use pasta_curves::group::ff::PrimeField;
        SpendAuthRandomizer::from_bytes(scalar.to_repr()).unwrap()
    }
    #[cfg(feature = "lrz-orchard")]
    {
        *scalar
    }
}

/// Value representation consumed by the selected Orchard commitment gadget.
#[cfg(feature = "orchard")]
pub type NoteValueWitness = pasta_curves::pallas::Base;
#[cfg(feature = "lrz-orchard")]
pub type NoteValueWitness = orchard::value::NoteValue;

/// Preserves the note value while adapting the selected gadget's witness type.
#[cfg(any(feature = "orchard", feature = "lrz-orchard"))]
pub fn note_value_witness(value: orchard::value::NoteValue) -> NoteValueWitness {
    #[cfg(feature = "orchard")]
    {
        pasta_curves::pallas::Base::from(value.inner())
    }
    #[cfg(feature = "lrz-orchard")]
    {
        value
    }
}

#[cfg(any(feature = "orchard", feature = "lrz-orchard"))]
mod composition_witness;
#[cfg(any(feature = "orchard", feature = "lrz-orchard"))]
pub use composition_witness::{note_psi, note_trapdoor, spend_authorizing_scalar, CircuitValue};

#[cfg(feature = "orchard")]
pub mod composition {
    pub use super::orchard::circuit::composition::*;
    pub use super::NonIdentityPallasPoint;
}
#[cfg(feature = "orchard")]
pub use orchard::circuit::composition::witness::NoteCommitTrapdoor as NoteTrapdoor;
#[cfg(feature = "lrz-orchard")]
pub use orchard::note::commitment::NoteCommitTrapdoor as NoteTrapdoor;
#[cfg(feature = "lrz-orchard")]
pub mod composition {
    pub use super::orchard::{
        circuit::{
            commit_ivk::{gadgets::commit_ivk, CommitIvkChip, CommitIvkConfig},
            gadget::{
                add_chip::{AddChip, AddConfig},
                assign_free_advice, derive_nullifier, AddInstruction,
            },
            note_commit::{gadgets::note_commit, NoteCommitChip, NoteCommitConfig},
        },
        constants::{
            fixed_bases::{
                spend_auth_g::{
                    generator as spend_auth_generator, U as SPEND_AUTH_U, Z as SPEND_AUTH_Z,
                },
                COMMIT_IVK_PERSONALIZATION, NOTE_COMMITMENT_PERSONALIZATION,
            },
            OrchardBaseFieldBases, OrchardCommitDomains, OrchardFixedBases, OrchardFixedBasesFull,
            OrchardHashDomains, L_ORCHARD_BASE, L_VALUE, MERKLE_DEPTH_ORCHARD,
        },
        spec::NonIdentityPallasPoint,
    };
}

pub fn transmission_point(key: &DiversifiedTransmissionKey) -> composition::NonIdentityPallasPoint {
    #[cfg(feature = "orchard")]
    {
        *key
    }
    #[cfg(feature = "lrz-orchard")]
    {
        key.inner()
    }
}
pub fn rho_from_nullifier(nf: orchard::note::Nullifier) -> orchard::note::Rho {
    orchard::note::Rho::from_bytes(&nf.to_bytes()).unwrap()
}
pub fn random_spending_key(rng: &mut impl rand::Rng) -> orchard::keys::SpendingKey {
    loop {
        let mut bytes = [0; 32];
        rng.fill_bytes(&mut bytes);
        if let Some(key) = orchard::keys::SpendingKey::from_bytes(bytes).into_option() {
            return key;
        }
    }
}
pub fn dummy_merkle_path(rng: &mut impl rand::Rng) -> orchard::tree::MerklePath {
    use pasta_curves::{
        group::ff::{Field, PrimeField},
        pallas,
    };
    orchard::tree::MerklePath::from_parts(
        rng.next_u32(),
        core::array::from_fn(|_| {
            orchard::tree::MerkleHashOrchard::from_bytes(&pallas::Base::random(&mut *rng).to_repr())
                .unwrap()
        }),
    )
}
pub fn dummy_note(
    rng: &mut impl rand::Rng,
    rho: Option<orchard::note::Rho>,
    version: orchard::note::NoteVersion,
) -> (
    orchard::keys::SpendingKey,
    orchard::keys::FullViewingKey,
    orchard::note::Note,
) {
    use pasta_curves::{
        group::ff::{Field, PrimeField},
        pallas,
    };
    let sk = random_spending_key(rng);
    let fvk = orchard::keys::FullViewingKey::from(&sk);
    let recipient = fvk.address_at(0u32, orchard::keys::Scope::External);
    let rho = rho.unwrap_or_else(|| {
        orchard::note::Rho::from_bytes(&pallas::Base::random(&mut *rng).to_repr()).unwrap()
    });
    loop {
        let mut bytes = [0; 32];
        rng.fill_bytes(&mut bytes);
        if let Some(seed) = orchard::note::RandomSeed::from_bytes(bytes, &rho).into_option() {
            if let Some(note) = orchard::note::Note::from_parts(
                recipient,
                orchard::value::NoteValue::ZERO,
                rho,
                seed,
                version,
            )
            .into_option()
            {
                return (sk, fvk, note);
            }
        }
    }
}

pub fn new_note(
    recipient: orchard::Address,
    value: orchard::value::NoteValue,
    rho: orchard::note::Rho,
    version: orchard::note::NoteVersion,
    rng: &mut impl rand::Rng,
) -> orchard::note::Note {
    loop {
        let mut bytes = [0; 32];
        rng.fill_bytes(&mut bytes);
        if let Some(seed) = orchard::note::RandomSeed::from_bytes(bytes, &rho).into_option() {
            if let Some(note) =
                orchard::note::Note::from_parts(recipient, value, rho, seed, version).into_option()
            {
                return note;
            }
        }
    }
}

#[cfg(feature = "orchard")]
pub use composition_witness::NonIdentityPallasPoint;
#[cfg(feature = "orchard")]
pub use orchard::circuit::composition::witness::{CommitIvkRandomness, NullifierDerivingKey};
#[cfg(feature = "lrz-orchard")]
pub use orchard::keys::{CommitIvkRandomness, DiversifiedTransmissionKey, NullifierDerivingKey};
#[cfg(feature = "orchard")]
pub type DiversifiedTransmissionKey = NonIdentityPallasPoint;

pub fn nullifier_key(key: &orchard::keys::FullViewingKey) -> NullifierDerivingKey {
    #[cfg(feature = "orchard")]
    {
        NullifierDerivingKey::from(key)
    }
    #[cfg(feature = "lrz-orchard")]
    {
        *key.nk()
    }
}
pub fn viewing_randomness(
    key: &orchard::keys::FullViewingKey,
    scope: orchard::keys::Scope,
) -> CommitIvkRandomness {
    #[cfg(feature = "orchard")]
    {
        CommitIvkRandomness::from_viewing_key(key, scope)
    }
    #[cfg(feature = "lrz-orchard")]
    {
        key.rivk(scope)
    }
}
pub fn diversified_base(address: &orchard::Address) -> composition::NonIdentityPallasPoint {
    #[cfg(feature = "orchard")]
    {
        NonIdentityPallasPoint::from_witness(
            orchard::circuit::composition::witness::NonIdentityPallasPoint::diversified_base(
                address,
            ),
        )
    }
    #[cfg(feature = "lrz-orchard")]
    {
        address.g_d()
    }
}
pub fn transmission_key(address: &orchard::Address) -> DiversifiedTransmissionKey {
    #[cfg(feature = "orchard")]
    {
        NonIdentityPallasPoint::from_witness(
            orchard::circuit::composition::witness::NonIdentityPallasPoint::transmission_key(
                address,
            ),
        )
    }
    #[cfg(feature = "lrz-orchard")]
    {
        *address.pk_d()
    }
}
