//! Real Halo2 prove/verify for the Share Reveal circuit (ZKP #3).
//!
//! Follows the same pattern as `delegation/prove.rs` but for the
//! 5-condition share reveal circuit at K=10.

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
// Params / key generation
// ================================================================

pub type ShareRevealKeys = (
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
);

static SHARE_REVEAL_PK_CACHE: std::sync::OnceLock<Result<ShareRevealKeys, String>> =
    std::sync::OnceLock::new();

/// Generate the IPA params (SRS) for the share reveal circuit.
/// Deterministic for a given `K`.
///
/// **Expensive**: K=10 params generation takes measurable setup time.
/// Callers should cache the result.
pub fn share_reveal_params() -> Params<EqAffine> {
    Params::new(K)
}

/// Generate the proving and verifying keys for the share reveal circuit.
///
/// Uses `Circuit::default()` (all witnesses unknown) as the empty circuit
/// for key generation — the same pattern as the delegation circuit.
///
/// **Expensive**: first call involves full circuit layout. Callers should
/// cache the result alongside the params.
pub fn share_reveal_proving_key(
    params: &Params<EqAffine>,
) -> Result<(plonk::ProvingKey<EqAffine>, plonk::VerifyingKey<EqAffine>), ProveError> {
    let empty_circuit = Circuit::default();
    let vk = keygen_vk(params, &empty_circuit).map_err(ProveError::KeygenVk)?;
    let pk = keygen_pk(params, vk.clone(), &empty_circuit).map_err(ProveError::KeygenPk)?;
    Ok((pk, vk))
}

/// Return cached params and proving/verifying keys for the share reveal circuit.
///
/// Key generation is deterministic and expensive enough to dominate helper
/// proving latency if repeated for every revealed share. Compute it once per
/// process and reuse it for both proving and verification.
pub fn share_reveal_cached_keys() -> Result<&'static ShareRevealKeys, ProveError> {
    match SHARE_REVEAL_PK_CACHE.get_or_init(|| {
        let params = share_reveal_params();
        share_reveal_proving_key(&params)
            .map(|(pk, vk)| (params, pk, vk))
            .map_err(|error| error.to_string())
    }) {
        Ok(keys) => Ok(keys),
        Err(error) => Err(ProveError::CachedKeygen(error.clone())),
    }
}

/// Warm the process-lifetime share reveal params/proving-key cache.
///
/// This lets callers pay deterministic keygen before the first user-visible
/// proof generation or verification path needs the key.
pub fn warm_share_reveal_keys() -> Result<(), ProveError> {
    share_reveal_cached_keys().map(|_| ())
}

// ================================================================
// Prove
// ================================================================

/// Create a real Halo2 proof for the share reveal circuit.
///
/// Returns the serialized proof bytes. Returns an error if the caller
/// provides a circuit without all witnesses populated or an instance
/// that Halo2 cannot prove against.
///
/// **Expensive**: proof generation should run in release mode.
/// Params and keys are cached so only the first call pays keygen.
pub fn create_share_reveal_proof(
    circuit: Circuit,
    instance: &Instance,
) -> Result<Vec<u8>, ProveError> {
    let (params, pk, _vk) = share_reveal_cached_keys()?;

    let public_inputs = instance.to_halo2_instance();

    create_proof_bytes(params, pk, circuit, &public_inputs)
}

// ================================================================
// Verify
// ================================================================

/// Verify a share reveal circuit proof given serialized proof bytes and
/// the 37 public inputs.
///
/// Returns `Ok(())` if verification succeeds, or an error message.
///
/// # Caller-authenticated inputs
///
/// `constrain_instance` pins each public input to whatever value the
/// *verifier* supplies; the protocol cannot tell whether that value was
/// the *right* one. The following fields of `instance` MUST be sourced
/// from a trusted channel (authenticated chain state, a signed
/// governance announcement) before calling this function. Substituting
/// them is not detectable from the proof alone:
///
/// - `instance.proposal_id` — must come from the active session's
///   published proposal list (the same value bound into the matching
///   vote-proof's `vote_commitment`).
/// - `instance.voting_round_id` — must come from the same governance
///   announcement as `proposal_id`.
/// - `instance.vote_comm_tree_root` — must be the vote commitment tree
///   root at the announced snapshot height (verifier looks it up by
///   height, not by accepting it from the prover bundle).
/// - `instance.decision_bucket_count` — must equal the option count `D`
///   declared by governance for `proposal_id` (the same value bound into
///   the matching vote-proof's `vote_commitment`).
///
/// # Caller-supplied values bound transitively by the proof
///
/// The revealed ciphertext coordinates are public values supplied by the
/// caller. The circuit does not recover them from ZKP #2, because vote-proof
/// publishes only the aggregate `vote_commitment`. Instead, condition 4 binds
/// them by proving the 34-input weighted selected commitment
/// (`crate::bridge::selected_share_commitment`) over a private `blind` and
/// the full public bucket ciphertext vector equals one of the 16 private
/// share commitments, which are then bound to `vote_comm_tree_root` through
/// `share_comms -> shares_hash -> vote_commitment -> Merkle path`. This
/// category is sound under the selected-commitment Poseidon
/// preimage-resistance assumption, but it is not a direct
/// `constrain_instance` derivation from other public inputs.
///
/// - `instance.ciphertexts` (all 32 coordinate slots)
///
/// # Proof-attested outputs
///
/// The following public inputs are derived outside the circuit but
/// constrained in-circuit against authenticated inputs and private witnesses;
/// successful verification is itself their authentication and the caller does
/// not need a separate trusted channel:
///
/// - `instance.share_nullifier`
pub fn verify_share_reveal_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    let (params, _pk, vk) = share_reveal_cached_keys().map_err(|error| error.to_string())?;

    let public_inputs = instance.to_halo2_instance();

    verify_proof_bytes("share_reveal", params, vk, proof, &public_inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        share_reveal::{build_share_reveal, ShareRevealBundle},
        ProveError, VOTE_COMM_TREE_DEPTH,
    };
    use voting_crypto_deps::halo2_proofs::plonk;
    use voting_crypto_deps::pasta_curves::pallas;

    fn valid_bundle() -> ShareRevealBundle {
        let blinds: [pallas::Base; 16] =
            core::array::from_fn(|i| pallas::Base::from(1_001 + i as u64));
        let ciphertexts: [crate::WeightedShareCiphertexts; 16] = core::array::from_fn(|i| {
            crate::WeightedShareCiphertexts(core::array::from_fn(|bucket| {
                let base = 2_001 + 1_000 * i as u64 + 4 * bucket as u64;
                crate::CiphertextCoordinates {
                    c1_x: pallas::Base::from(base),
                    c2_x: pallas::Base::from(base + 1),
                    c1_y: pallas::Base::from(base + 2),
                    c2_y: pallas::Base::from(base + 3),
                }
            }))
        });
        let share_comms =
            core::array::from_fn(|i| crate::selected_share_commitment(blinds[i], &ciphertexts[i]));
        let share_index = 2usize;

        build_share_reveal(
            [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH],
            0,
            share_comms,
            blinds[share_index],
            &ciphertexts[share_index],
            share_index as u32,
            pallas::Base::from(3),
            pallas::Base::from(999),
            pallas::Base::from(4),
        )
    }

    fn minimal_instance() -> Instance {
        Instance::from_parts(
            pallas::Base::from(1),
            crate::WeightedShareCiphertexts(
                [crate::CiphertextCoordinates {
                    c1_x: pallas::Base::from(2),
                    c2_x: pallas::Base::from(3),
                    c1_y: pallas::Base::from(4),
                    c2_y: pallas::Base::from(5),
                }; crate::MAX_DECISION_BUCKETS],
            ),
            pallas::Base::from(6),
            pallas::Base::from(7),
            pallas::Base::from(8),
            pallas::Base::from(9),
        )
    }

    #[test]
    fn create_share_reveal_proof_signature_returns_result() {
        let _: fn(Circuit, &Instance) -> Result<Vec<u8>, ProveError> = create_share_reveal_proof;
    }

    #[test]
    fn create_share_reveal_proof_returns_err_for_missing_witnesses() {
        let instance = minimal_instance();
        let err = create_share_reveal_proof(Circuit::default(), &instance).unwrap_err();

        assert!(matches!(err, ProveError::Halo2(plonk::Error::Synthesis)));
    }

    #[test]
    #[ignore = "long-running real proof roundtrip; run with `cargo test -- --ignored real_proof_roundtrip`"]
    fn real_proof_roundtrip_stays_within_downstream_limit() {
        // Keep this aligned with vote-sdk's consensus proof-size limit.
        const DOWNSTREAM_MAX_PROOF_SIZE: usize = 15 * 1_024;

        let ShareRevealBundle { circuit, instance } = valid_bundle();
        let proof = create_share_reveal_proof(circuit, &instance)
            .expect("share reveal proof creation should succeed");

        verify_share_reveal_proof(&proof, &instance)
            .expect("share reveal verifier should accept the generated proof");
        assert!(
            proof.len() <= DOWNSTREAM_MAX_PROOF_SIZE,
            "share reveal proof is {} bytes, exceeding the downstream {}-byte limit",
            proof.len(),
            DOWNSTREAM_MAX_PROOF_SIZE,
        );
    }

    // TODO(sean): VK-stability tripwire. Hashes the `PinnedVerificationKey`
    // debug repr and compares against a baked-in fingerprint. A mismatch means
    // either the circuit shape changed (and the VK must be regenerated and
    // redistributed) or an unintended drift has been introduced.
    #[test]
    #[ignore = "runs K=10 keygen; run with `cargo test -- --ignored vk_fingerprint_unchanged`"]
    fn vk_fingerprint_unchanged() {
        let (_, _, vk) = share_reveal_cached_keys().expect("share reveal keys");
        let pinned = format!("{:?}", vk.pinned());
        let fingerprint = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(pinned.as_bytes());
        let actual: &[u8] = fingerprint.as_bytes();

        let expected: [u8; 32] = [
            0x0c, 0x0c, 0x04, 0x21, 0xdf, 0x7f, 0x70, 0x43, 0x1c, 0xfc, 0x63, 0xf1, 0x5f, 0x31,
            0x1b, 0xd6, 0x6e, 0x77, 0x6f, 0x73, 0x34, 0x32, 0x24, 0x00, 0x66, 0xd9, 0xe1, 0x78,
            0xf1, 0x65, 0x84, 0x1b,
        ];

        assert_eq!(
            actual,
            expected.as_slice(),
            "share reveal VK fingerprint changed; if intentional, update `expected` to:\n{:02x?}",
            actual,
        );
    }
}
