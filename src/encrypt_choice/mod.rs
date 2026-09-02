//! Encrypt-Choice ZKP circuit (ZKP 1.5).
//!
//! The decision-bound auxiliary proof of a weighted vote. Started in the
//! background the moment the voter selects a choice, it carries all the
//! ElGamal elliptic-curve work of the vote so the interactive cast proof
//! (ZKP #2) contains no ECC at all; the two proofs are submitted together in
//! one vote bundle, bound by a shared public bridge commitment.
//!
//! The circuit verifies 5 conditions:
//! - **Condition 1**: One-Hot Decision (boolean selectors summing to one).
//! - **Condition 2**: Active-Bucket Confinement (boolean prefix flags summing
//!   to the public `decision_bucket_count`; the selector is zero in inactive
//!   buckets).
//! - **Condition 3**: Encryption Integrity (`16 × 8` ElGamal ciphertexts with
//!   non-zero PRF randomness, selector-chosen C2, shared weight points, and
//!   30-bit share range checks).
//! - **Condition 4**: Selected Commitments (wide Poseidon commitment per
//!   share; see `crate::bridge`).
//! - **Condition 5**: Bridge Integrity (compact bridge over round, proposal,
//!   bucket count, and all `(weight, commitment)` pairs, bound to the public
//!   instance).

mod builder;
mod circuit;
mod prove;

pub(crate) use builder::derive_vote_shares;

pub use builder::{
    build_encrypt_choice, restore_encrypt_choice, ElGamalCiphertextBytes, EncryptChoiceBuildError,
    EncryptChoiceBundle, EncryptedWeightedShareOutput,
};
pub use circuit::{Circuit, Instance, K};
pub use prove::{
    create_encrypt_choice_proof, encrypt_choice_cached_keys, encrypt_choice_params,
    encrypt_choice_proving_key, verify_encrypt_choice_proof, warm_encrypt_choice_keys,
};
