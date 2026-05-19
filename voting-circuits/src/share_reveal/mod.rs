//! Share Reveal ZKP circuit (ZKP #3).
//!
//! Proves that a publicly-revealed encrypted share came from a valid,
//! registered vote commitment — without revealing which one.
//!
//! The circuit verifies 5 conditions:
//! - **Condition 1**: VC Membership (Poseidon Merkle path).
//! - **Condition 2**: Vote Commitment Integrity (ConstantLength<5> Poseidon).
//! - **Condition 3**: Shares Hash Integrity (blinded per-share commitments,
//!   then ConstantLength<16> Poseidon over the 16 commitments).
//! - **Condition 4**: Share Membership (custom mux gate).
//! - **Condition 5**: Share Nullifier Integrity (four-input Poseidon hash with
//!   round binding through `vote_commitment`).

pub(crate) mod builder;
pub(crate) mod circuit;
pub(crate) mod prove;

pub use builder::{build_share_reveal, ShareRevealBundle};
pub use circuit::{
    domain_tag_share_spend, share_nullifier_hash, Circuit, Config, Instance,
    ENC_SHARE_C1_X_PUBLIC_OFFSET, ENC_SHARE_C1_Y_PUBLIC_OFFSET, ENC_SHARE_C2_X_PUBLIC_OFFSET,
    ENC_SHARE_C2_Y_PUBLIC_OFFSET, K, PROPOSAL_ID_PUBLIC_OFFSET, SHARE_NULLIFIER_PUBLIC_OFFSET,
    VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET, VOTE_DECISION_PUBLIC_OFFSET, VOTING_ROUND_ID_PUBLIC_OFFSET,
};
pub use prove::share_reveal_cached_keys;
pub use prove::{
    create_share_reveal_proof, share_reveal_params, share_reveal_proving_key,
    verify_share_reveal_proof, warm_share_reveal_keys,
};
