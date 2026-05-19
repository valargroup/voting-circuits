//! Vote proof ZKP circuit (ZKP #2).
//!
//! Proves that a vote is well-formed and authorized with respect to
//! delegation and the vote commitment tree. The circuit verifies 12
//! conditions; all are fully constrained.
//!
//! - **Condition 1**: VAN Membership (Poseidon Merkle path, `constrain_instance`).
//! - **Condition 2**: VAN Integrity (Poseidon hash).
//! - **Condition 3**: Diversified Address Integrity (CommitIvk chain, `constrain_equal`).
//! - **Condition 4**: Spend Authority (fixed-base mul + point add, `constrain_instance`).
//! - **Condition 5**: VAN Nullifier Integrity (nested Poseidon, `constrain_instance`).
//! - **Condition 6**: Proposal Authority Decrement (custom bit-decomposition chip with a `(proposal_id, 2^proposal_id)` lookup; see `authority_decrement.rs`).
//! - **Condition 7**: New VAN Integrity (Poseidon hash, `constrain_instance`).
//! - **Condition 8**: Shares Sum Correctness (AddChip, `constrain_equal`).
//! - **Condition 9**: Shares Range (LookupRangeCheck, `[0, 2^30)`).
//! - **Condition 10**: Shares Hash Integrity (Poseidon `ConstantLength<16>` over 16 blinded share commitments; output flows to condition 12).
//! - **Condition 11**: Encryption Integrity (ECC variable-base mul, `constrain_equal`).
//! - **Condition 12**: Vote Commitment Integrity (Poseidon `ConstantLength<5>`, `constrain_instance`).

pub(crate) mod authority_decrement;
pub(crate) mod builder;
pub(crate) mod circuit;
pub(crate) mod prove;

pub use crate::circuit::elgamal::spend_auth_g_affine;
pub use builder::{
    build_vote_proof_from_delegation, EncryptedShareOutput, VoteProofBuildError, VoteProofBundle,
};
pub use circuit::{
    poseidon_hash_2, share_commitment, shares_hash, Circuit, Config, Instance,
    EA_PK_X_PUBLIC_OFFSET, EA_PK_Y_PUBLIC_OFFSET, K, PROPOSAL_ID_PUBLIC_OFFSET,
    R_VPK_X_PUBLIC_OFFSET, R_VPK_Y_PUBLIC_OFFSET, VAN_NULLIFIER_PUBLIC_OFFSET,
    VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET, VOTE_COMMITMENT_PUBLIC_OFFSET,
    VOTE_COMM_TREE_ANCHOR_HEIGHT_PUBLIC_OFFSET, VOTE_COMM_TREE_DEPTH,
    VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET, VOTING_ROUND_ID_PUBLIC_OFFSET,
};
pub use prove::{
    create_vote_proof, verify_vote_proof, vote_proof_cached_keys, vote_proof_params,
    vote_proof_proving_key, warm_vote_proof_keys,
};
