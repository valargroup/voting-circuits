//! Share Reveal ZKP circuit (ZKP #3).
//!
//! Proves that a publicly-revealed encrypted share vector — all 8 bucket
//! ciphertexts of one weight share — came from a valid, registered vote
//! commitment, without revealing which one or which bucket carries the
//! weight.
//!
//! The circuit verifies 5 conditions:
//! - **Condition 1**: VC Membership (Poseidon Merkle path).
//! - **Condition 2**: Vote Commitment Integrity v2 (ConstantLength<5>
//!   Poseidon over round, shares hash, proposal, and bucket count).
//! - **Condition 3**: Shares Hash Integrity (ConstantLength<16> Poseidon
//!   over the 16 private selected commitments).
//! - **Condition 4**: Share Membership (custom mux gate plus the 34-input
//!   weighted selected-commitment hash over the revealed ciphertext vector;
//!   see `crate::bridge`).
//! - **Condition 5**: Share Nullifier Integrity (four-input Poseidon hash with
//!   round binding through `vote_commitment`).

mod builder;
mod circuit;
mod prove;

pub use builder::{build_share_reveal, ShareRevealBundle};
pub use circuit::{domain_tag_share_spend, share_nullifier_hash, Circuit, Instance, K};
pub use prove::{
    create_share_reveal_proof, share_reveal_cached_keys, share_reveal_params,
    share_reveal_proving_key, verify_share_reveal_proof, warm_share_reveal_keys,
};
