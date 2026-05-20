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
    build_delegation_bundle, synthetic_padding_note_parts, DelegationBuildError, DelegationBundle,
    PaddedNoteData, PrecomputedRandomness, RealNoteInput, SyntheticPaddingNoteParts,
};
pub use circuit::{
    Circuit, Instance, CMX_NEW_PUBLIC_OFFSET, DOM_PUBLIC_OFFSET, GOV_NULL_1_PUBLIC_OFFSET,
    GOV_NULL_2_PUBLIC_OFFSET, GOV_NULL_3_PUBLIC_OFFSET, GOV_NULL_4_PUBLIC_OFFSET,
    GOV_NULL_5_PUBLIC_OFFSET, GOV_NULL_PUBLIC_OFFSETS, K, NC_ROOT_PUBLIC_OFFSET,
    NF_IMT_ROOT_PUBLIC_OFFSET, NF_SIGNED_PUBLIC_OFFSET, RK_X_PUBLIC_OFFSET, RK_Y_PUBLIC_OFFSET,
    VAN_COMM_PUBLIC_OFFSET, VOTE_ROUND_ID_PUBLIC_OFFSET,
};
pub use imt::{
    build_nullifier_list, build_sentinel_list, derive_nullifier_domain, ImtError, ImtProofData,
    ImtProvider, SpacedLeafImtProvider, IMT_DEPTH,
};
pub use prove::{
    create_delegation_proof, delegation_cached_keys, delegation_params, delegation_proving_key,
    verify_delegation_proof, warm_delegation_keys,
};
