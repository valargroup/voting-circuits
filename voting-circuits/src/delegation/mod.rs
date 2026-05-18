//! Delegation ZKP circuit.
//!
//! A single circuit proving all 15 conditions of the delegation ZKP,
//! including 5 per-note slots.
//! The builder layer creates padded notes for unused slots and
//! produces a single proof.

pub(crate) mod builder;
pub(crate) mod circuit;
pub(crate) mod imt;
pub(crate) mod imt_circuit;
pub(crate) mod prove;

pub use builder::{
    build_delegation_bundle, DelegationBuildError, DelegationBundle, PaddedNoteData,
    PrecomputedRandomness, RealNoteInput,
};
pub use circuit::{Circuit, Instance, K};
pub use imt::{
    build_sentinel_list, derive_nullifier_domain, ImtError, ImtProofData, ImtProvider,
    SpacedLeafImtProvider, IMT_DEPTH,
};
pub use prove::{
    create_delegation_proof, delegation_cached_keys, delegation_params, delegation_proving_key,
    verify_delegation_proof, warm_delegation_keys,
};
