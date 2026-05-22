//! Governance ZKP circuits for the Zally voting protocol.
//!
//! Contains three circuits:
//! - **Delegation** (ZKP #1): Proves delegation of voting rights.
//! - **Vote Proof** (ZKP #2): Proves a valid, authorized vote.
//! - **Share Reveal** (ZKP #3): Proves a revealed share belongs to a registered vote commitment.

#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

mod circuit;
mod domain_tags;
mod protocol_hash;
mod prove_error;
mod shares_hash;

pub use prove_error::ProveError;
pub use shares_hash::shares_hash_from_comms;

pub mod delegation;

pub mod vote_proof;

pub mod share_reveal;
