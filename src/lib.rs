//! Governance ZKP circuits for the Zally voting protocol.
//!
//! Contains four circuits:
//! - **Delegation** (ZKP #1): Proves delegation of voting rights.
//! - **Vote Proof** (ZKP #2): Proves a valid, authorized vote.
//! - **Share Reveal** (ZKP #3): Proves a revealed share belongs to a registered vote commitment.
//! - **Proxy Delegation** (ZKP #4): Proves a valid delegation of vote weight to a public delegate's pool.

#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

pub use voting_crypto_deps::{
    pasta_curves::group::{self, ff},
    rand,
};

mod domain_tags;
mod encoding;
mod gadgets;
mod params;
mod protocol_hash;
mod prove_error;
mod shares_hash;

pub use gadgets::elgamal::spend_auth_g_affine;
pub use params::{MAX_PROPOSAL_AUTHORITY, VOTE_COMM_TREE_DEPTH};
pub use prove_error::ProveError;
pub use shares_hash::{share_commitment, shares_hash, shares_hash_from_comms};

pub mod delegation;
pub mod proxy_delegation;
pub mod share_reveal;
pub mod vote_proof;
