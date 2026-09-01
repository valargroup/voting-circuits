//! Vote proof ZKP circuit (ZKP #2, compact cast circuit).
//!
//! Proves that a vote is well-formed and authorized with respect to
//! delegation and the vote commitment tree. The ElGamal encryption work lives
//! in the encrypt-choice circuit (ZKP 1.5); the two proofs are submitted as
//! one [`VoteBundle`], bound by a shared public bridge commitment. The
//! circuit verifies 12 conditions; all are fully constrained.
//!
//! - **Condition 1**: VAN Membership (Poseidon Merkle path, `constrain_instance`).
//! - **Condition 2**: VAN Integrity (Poseidon hash).
//! - **Condition 3**: Diversified Address Integrity (CommitIvk chain, `constrain_equal`).
//! - **Condition 4**: Spend Authority (fixed-base mul + point add, `constrain_instance`).
//! - **Condition 5**: VAN Nullifier Integrity (nested Poseidon, `constrain_instance`).
//! - **Condition 6**: Proposal Authority Decrement (custom bit-decomposition chip with a `(proposal_id, 2^proposal_id)` lookup; see `gadgets/authority_decrement.rs`).
//! - **Condition 7**: New VAN Integrity (Poseidon hash, `constrain_instance`).
//! - **Condition 8**: Shares Sum Correctness (AddChip, `constrain_equal`).
//! - **Condition 9**: Shares Range (LookupRangeCheck, `[0, 2^30)`).
//! - **Condition 10′**: Bridge Re-Opening (36-input Poseidon over the shares and witnessed selected commitments, `constrain_instance`; see `crate::bridge`).
//! - **Condition 11′**: Shares Hash Integrity (Poseidon `ConstantLength<16>` over the 16 selected commitments; output flows to condition 12′).
//! - **Condition 12′**: Vote Commitment Integrity v2 (Poseidon `ConstantLength<5>` over round, shares hash, proposal, and bucket count, `constrain_instance`).

mod builder;
mod circuit;
mod gadgets;
mod prove;

pub use builder::{
    build_vote_proof_from_delegation, check_vote_bundle_consistency,
    derive_vote_authority_transition, verify_vote_bundle, VoteAuthorityTransition, VoteBundle,
    VoteBundleError, VoteProofBundle,
};
pub use circuit::{Circuit, Instance, K};
pub use prove::{
    verify_vote_proof, vote_proof_cached_keys, vote_proof_params, vote_proof_proving_key,
    warm_vote_proof_keys,
};
