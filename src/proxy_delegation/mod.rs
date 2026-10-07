//! PROTOTYPE: Delegation-commitment proof (ZKP #4) for proxy delegation.
//!
//! Throwaway prototype used to measure rows, proof size, and prove time.
//! Spends a full-authority VAN exactly like ZKP #2 and emits a successor VAN
//! with weight `W - w` plus a delegation commitment
//! `DC = Poseidon(DOMAIN_VC, round, shares_hash, 0, delegate_index)`.

mod circuit;

pub use circuit::{delegation_commitment_hash, Circuit, Instance, K};
