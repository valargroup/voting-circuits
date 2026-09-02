//! Governance ZKP circuits for the Zally voting protocol.
//!
//! Contains four circuits:
//! - **Delegation** (ZKP #1): Proves delegation of voting rights.
//! - **Encrypt Choice** (ZKP 1.5): Proves the decision-bound ElGamal
//!   encryption of all weight shares, produced in the background once the
//!   voter selects a choice.
//! - **Vote Proof** (ZKP #2): Proves a valid, authorized vote; bound to the
//!   encrypt-choice proof by a shared bridge commitment.
//! - **Share Reveal** (ZKP #3): Proves a revealed encrypted share vector
//!   belongs to a registered vote commitment.

#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

pub use voting_crypto_deps::{
    pasta_curves::group::{self, ff},
    rand,
};

mod bridge;
mod domain_tags;
mod gadgets;
mod params;
mod protocol_hash;
mod prove_error;
mod shares_hash;
mod vote_prf;

pub use bridge::{
    bridge_commitment, selected_share_commitment, CiphertextCoordinates, WeightedShareCiphertexts,
    BRIDGE_INPUTS, MAX_DECISION_BUCKETS, NUM_SHARES, SELECTED_COMMITMENT_INPUTS,
};
pub use gadgets::elgamal::spend_auth_g_affine;
pub use gadgets::vote_commitment::{vote_commitment_hash_v2, DOMAIN_VC_V2};
pub use params::{MAX_PROPOSAL_AUTHORITY, VOTE_COMM_TREE_DEPTH};
pub use prove_error::ProveError;
pub use shares_hash::shares_hash_from_comms;

pub mod delegation;
pub mod encrypt_choice;
pub mod share_reveal;
pub mod vote_proof;
