//! Real Halo2 prove/verify for the encrypt-choice circuit (ZKP 1.5).
//!
//! Follows the same pattern as `vote_proof/prove.rs` but for the
//! five-condition encrypt-choice circuit at K=11 on the fully parallel
//! (~1,100-column) layout.

use std::{string::String, vec::Vec};

use voting_crypto_deps::halo2_proofs::{
    pasta::EqAffine,
    plonk::{self, keygen_pk, keygen_vk},
    poly::commitment::Params,
};

use super::circuit::{Circuit, Instance, K};
use crate::{
    prove_error::{create_proof_bytes, verify_proof_bytes},
    ProveError,
};

// ================================================================
// Cached params + keys
// ================================================================

// Keygen is deterministic and expensive. Compute it once per process and
// reuse it for all subsequent proofs and verifications.
pub type EncryptChoiceKeys = (
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
);

static ENCRYPT_CHOICE_KEYS_CACHE: std::sync::OnceLock<Result<EncryptChoiceKeys, String>> =
    std::sync::OnceLock::new();

/// Return cached params and proving/verifying keys for the encrypt-choice
/// circuit.
///
/// Params generation and key generation are deterministic and expensive enough
/// to dominate the first proof or verification call. Compute the full tuple
/// once per process so warm-up covers both the SRS params and the keys.
pub fn encrypt_choice_cached_keys() -> Result<&'static EncryptChoiceKeys, ProveError> {
    match ENCRYPT_CHOICE_KEYS_CACHE.get_or_init(|| {
        let params = encrypt_choice_params();
        encrypt_choice_proving_key(&params)
            .map(|(pk, vk)| (params, pk, vk))
            .map_err(|error| error.to_string())
    }) {
        Ok(keys) => Ok(keys),
        Err(error) => Err(ProveError::CachedKeygen(error.clone())),
    }
}

/// Warm the process-lifetime encrypt-choice params/proving-key cache.
///
/// This lets callers pay deterministic keygen before the first user-visible
/// proof generation or verification path needs the params and keys. The
/// wallet should call this as soon as a vote becomes likely — the wide
/// (~1,100-column) K=11 encrypt-choice keygen is the most expensive of the
/// vote circuits.
pub fn warm_encrypt_choice_keys() -> Result<(), ProveError> {
    encrypt_choice_cached_keys().map(|_| ())
}

// ================================================================
// Params / key generation (public API, non-cached fallbacks)
// ================================================================

/// Generate the IPA params (SRS) for the encrypt-choice circuit.
/// Deterministic for a given `K`.
pub fn encrypt_choice_params() -> Params<EqAffine> {
    Params::new(K)
}

/// Generate the proving and verifying keys for the encrypt-choice circuit.
///
/// Uses `Circuit::default()` (all witnesses unknown) as the empty circuit
/// for key generation — the same pattern as the Orchard action circuit.
pub fn encrypt_choice_proving_key(
    params: &Params<EqAffine>,
) -> Result<(plonk::ProvingKey<EqAffine>, plonk::VerifyingKey<EqAffine>), ProveError> {
    let empty_circuit = Circuit::default();
    let vk = keygen_vk(params, &empty_circuit).map_err(ProveError::KeygenVk)?;
    let pk = keygen_pk(params, vk.clone(), &empty_circuit).map_err(ProveError::KeygenPk)?;
    Ok((pk, vk))
}

// ================================================================
// Prove
// ================================================================

/// Create a real Halo2 proof for the encrypt-choice circuit.
///
/// Returns the serialized proof bytes. Returns an error if the caller
/// provides a circuit without all witnesses populated or an instance
/// that Halo2 cannot prove against.
///
/// **Expensive**: K=11 proof generation over ~1,100 columns should run in
/// release mode. Params and keys are cached so only the first call pays
/// keygen.
pub fn create_encrypt_choice_proof(
    circuit: Circuit,
    instance: &Instance,
) -> Result<Vec<u8>, ProveError> {
    let (params, pk, _vk) = encrypt_choice_cached_keys()?;

    let public_inputs = instance.to_halo2_instance();

    create_proof_bytes(params, pk, circuit, &public_inputs)
}

// ================================================================
// Verify
// ================================================================

/// Verify an encrypt-choice proof given serialized proof bytes and the typed
/// public inputs.
///
/// Returns `Ok(())` if verification succeeds, or an error message.
///
/// # Caller-authenticated inputs
///
/// Every public input is bound into the proof transcript, but the proof
/// cannot tell the verifier whether caller-provided governance values are the
/// right ones. The following fields of `instance` MUST be sourced from their
/// authority before calling this function; substituting them is not
/// detectable from the proof alone.
///
/// ## Governance session parameters
///
/// - `instance.voting_round_id` — must identify the active voting round the
///   verifier is accepting.
/// - `instance.proposal_id` — must be in the active proposal set for
///   `voting_round_id`.
/// - `instance.decision_bucket_count` — must equal the option count `D`
///   declared by governance for `proposal_id`. The circuit constrains the
///   private decision to `0..D` structurally but proves nothing about which
///   `D` is correct for the proposal; the verifier must also reject
///   `D < 2` (the circuit shape admits `D = 1`).
///
/// ## Election-authority public key
///
/// - `instance.ea_pk_x`, `instance.ea_pk_y` — must equal the election
///   authority's published session key for `voting_round_id`. The circuit
///   only proves the shares were encrypted under the caller-supplied key.
///   Accepting the key from the prover bundle either loses liveness (the
///   real EA cannot decrypt) or secrecy (a colluding key holder can).
///
/// # Bundle binding
///
/// - `instance.bridge` is proof-attested here, but its protocol meaning
///   comes from the vote bundle: the verifier MUST check it equals the cast
///   proof's public bridge, and that both proofs carry the same
///   `voting_round_id`, `proposal_id`, and `decision_bucket_count`.
///   `vote_proof::verify_vote_bundle` performs these checks.
pub fn verify_encrypt_choice_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    let (params, _pk, vk) = encrypt_choice_cached_keys().map_err(|error| error.to_string())?;

    let public_inputs = instance.to_halo2_instance();

    verify_proof_bytes("encrypt choice", params, vk, proof, &public_inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::PrimeField;
    use voting_crypto_deps::pasta_curves::pallas;

    fn minimal_instance() -> Instance {
        Instance::from_parts(
            pallas::Base::from(1),
            pallas::Base::from(2),
            pallas::Base::from(3),
            pallas::Base::from(4),
            pallas::Base::from(5),
            pallas::Base::from(6),
        )
    }

    #[test]
    fn public_input_count_matches_instance_layout() {
        let instance = minimal_instance();

        assert_eq!(
            instance.to_halo2_instance().len(),
            Instance::NUM_PUBLIC_INPUTS
        );
        let serialized: Vec<u8> = instance
            .to_halo2_instance()
            .into_iter()
            .flat_map(|input| input.to_repr())
            .collect();
        assert_eq!(serialized.len(), Instance::NUM_PUBLIC_INPUTS * 32);
    }

    #[test]
    fn instance_offsets_match_field_order() {
        use super::super::circuit::{
            BRIDGE_PUBLIC_OFFSET, DECISION_BUCKET_COUNT_PUBLIC_OFFSET, EA_PK_X_PUBLIC_OFFSET,
            EA_PK_Y_PUBLIC_OFFSET, PROPOSAL_ID_PUBLIC_OFFSET, VOTING_ROUND_ID_PUBLIC_OFFSET,
        };

        let instance = minimal_instance();
        let halo2 = instance.to_halo2_instance();

        assert_eq!(halo2[EA_PK_X_PUBLIC_OFFSET], instance.ea_pk_x);
        assert_eq!(halo2[EA_PK_Y_PUBLIC_OFFSET], instance.ea_pk_y);
        assert_eq!(halo2[BRIDGE_PUBLIC_OFFSET], instance.bridge);
        assert_eq!(
            halo2[DECISION_BUCKET_COUNT_PUBLIC_OFFSET],
            instance.decision_bucket_count
        );
        assert_eq!(
            halo2[VOTING_ROUND_ID_PUBLIC_OFFSET],
            instance.voting_round_id
        );
        assert_eq!(halo2[PROPOSAL_ID_PUBLIC_OFFSET], instance.proposal_id);
    }

    // TODO: VK-stability tripwire, mirroring `vote_proof::prove`. A mismatch
    // means either the circuit shape changed (and the VK must be regenerated
    // and redistributed) or an unintended drift has been introduced.
    #[test]
    #[ignore = "runs wide K=11 keygen; run with `cargo test --release -- --ignored vk_fingerprint_unchanged`"]
    fn vk_fingerprint_unchanged() {
        let (_, _, vk) = encrypt_choice_cached_keys().expect("encrypt choice keys");
        let pinned = format!("{:?}", vk.pinned());
        let fingerprint = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(pinned.as_bytes());
        let actual: &[u8] = fingerprint.as_bytes();

        let expected: [u8; 32] = [
            0x39, 0xcd, 0xb3, 0x57, 0x46, 0xa5, 0x85, 0xe0, 0xd9, 0x42, 0x02, 0x72, 0xa8, 0xf1,
            0xef, 0x35, 0x9d, 0x21, 0x46, 0xbf, 0x40, 0xee, 0x70, 0xaf, 0x34, 0x19, 0x2a, 0xc7,
            0xdf, 0xf2, 0x57, 0x3b,
        ];

        assert_eq!(
            actual,
            expected.as_slice(),
            "encrypt choice VK fingerprint changed; if intentional, update `expected` to:\n{:02x?}",
            actual,
        );
    }
}
