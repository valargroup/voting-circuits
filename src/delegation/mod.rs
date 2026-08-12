//! Delegation ZKP circuit.
//!
//! A single circuit proving all 14 conditions of the delegation ZKP,
//! including 5 per-note slots.
//! The builder layer creates padded notes for unused slots and
//! produces a single proof.

mod builder;
mod circuit;
mod gadgets;
mod imt;
mod prove;

/// Signed-note value used for hardware-wallet display.
const KEYSTONE_NOTE_VALUE: u64 = 1;

pub use builder::{
    build_delegation_bundle, synthetic_padding_note_parts, DelegationBuildError, DelegationBundle,
    PaddedNoteData, PrecomputedRandomness, RealNoteInput,
};
pub use circuit::{rho_binding_hash, van_commitment_hash, Circuit, Instance, K};
pub use imt::{
    build_sentinel_list, derive_nullifier_domain, gov_null_hash, ImtError, ImtProofData,
    ImtProvider, SpacedLeafImtProvider, IMT_DEPTH,
};
pub use prove::{
    create_delegation_proof, delegation_cached_keys, delegation_params, delegation_proving_key,
    verify_delegation_proof, warm_delegation_keys,
};

#[cfg(feature = "unstable-internal-api")]
pub use imt::build_nullifier_list;
