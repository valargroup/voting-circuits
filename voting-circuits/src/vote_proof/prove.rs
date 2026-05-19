//! Real Halo2 prove/verify for the vote proof circuit (ZKP #2).
//!
//! Follows the same pattern as `delegation/prove.rs` but for the
//! 12-condition vote proof circuit at K=14.

use std::string::String;
use std::vec::Vec;

use halo2_proofs::{
    pasta::EqAffine,
    plonk::{self, keygen_pk, keygen_vk, verify_proof, SingleVerifier},
    poly::commitment::Params,
    transcript::{Blake2bRead, Challenge255},
};

use crate::prove_error::create_proof_bytes;
use crate::ProveError;

use super::circuit::{Circuit, Instance, K};

// ================================================================
// Cached params + keys
// ================================================================

// Keygen is deterministic and expensive (~30s on device). Compute once
// per process and reuse for all subsequent proofs and verifications.
static VOTE_PROOF_KEYS_CACHE: std::sync::OnceLock<(
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
)> = std::sync::OnceLock::new();

/// Return cached params and proving/verifying keys for the vote proof circuit.
///
/// Params generation and key generation are deterministic and expensive enough
/// to dominate the first proof or verification call. Compute the full tuple
/// once per process so warm-up covers both the SRS params and the keys.
pub fn vote_proof_cached_keys() -> &'static (
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
) {
    VOTE_PROOF_KEYS_CACHE.get_or_init(|| {
        let params = vote_proof_params();
        let (pk, vk) = vote_proof_proving_key(&params);
        (params, pk, vk)
    })
}

/// Warm the process-lifetime vote proof params/proving-key cache.
///
/// This lets callers pay deterministic keygen before the first user-visible
/// proof generation or verification path needs the params and keys.
pub fn warm_vote_proof_keys() {
    let _ = vote_proof_cached_keys();
}

// ================================================================
// Params / key generation (public API, non-cached fallbacks)
// ================================================================

/// Generate the IPA params (SRS) for the vote proof circuit.
/// Deterministic for a given `K`.
pub fn vote_proof_params() -> Params<EqAffine> {
    Params::new(K)
}

/// Generate the proving and verifying keys for the vote proof circuit.
///
/// Uses `Circuit::default()` (all witnesses unknown) as the empty circuit
/// for key generation — the same pattern as the Orchard action circuit.
pub fn vote_proof_proving_key(
    params: &Params<EqAffine>,
) -> (plonk::ProvingKey<EqAffine>, plonk::VerifyingKey<EqAffine>) {
    let empty_circuit = Circuit::default();
    let vk = keygen_vk(params, &empty_circuit).expect("vote_proof keygen_vk should not fail");
    let pk = keygen_pk(params, vk.clone(), &empty_circuit)
        .expect("vote_proof keygen_pk should not fail");
    (pk, vk)
}

// ================================================================
// Prove
// ================================================================

/// Create a real Halo2 proof for the vote proof circuit.
///
/// Returns the serialized proof bytes. Returns an error if the caller
/// provides a circuit without all witnesses populated or an instance
/// that Halo2 cannot prove against.
///
/// **Expensive**: K=14 proof generation takes ~30-60 seconds in release mode.
/// Params and keys are cached so only the first call pays keygen.
pub fn create_vote_proof(circuit: Circuit, instance: &Instance) -> Result<Vec<u8>, ProveError> {
    let (params, pk, _vk) = vote_proof_cached_keys();

    let public_inputs = instance.to_halo2_instance();

    create_proof_bytes(params, pk, circuit, &public_inputs)
}

// ================================================================
// Verify
// ================================================================

/// Verify a vote proof circuit proof given serialized proof bytes and
/// the typed public inputs.
///
/// Returns `Ok(())` if verification succeeds, or an error message.
///
/// # Caller-authenticated inputs
///
/// Some public inputs are constrained to witness-derived cells, and every
/// public input is bound into the proof transcript. Neither property tells the
/// verifier whether caller-provided governance or chain values are the right
/// ones. The following fields of `instance` MUST be sourced from a trusted
/// channel (a signed governance announcement, a signed chain head) before
/// calling this function. Substituting them is not detectable from the proof
/// alone. Most notably, an attacker can pick their own `ea_pk` (`= sk*G` for
/// an `sk` they know) and decrypt all posted shares, with the proof still
/// verifying:
///
/// - `instance.proposal_id` — must come from the active session's
///   published proposal list.
/// - `instance.voting_round_id` — must come from the same governance
///   announcement as `proposal_id`.
/// - `instance.vote_comm_tree_root` - must be the vote commitment tree
///   root at `vote_comm_tree_anchor_height` (verifier looks it up by
///   height, not by accepting it from the prover bundle).
/// - `instance.vote_comm_tree_anchor_height` - must be a valid chain
///   height accepted by the consuming chain's anchor-validity check. This
///   slot is transcript-bound but not constrained to any circuit witness.
/// - `instance.ea_pk_x`, `instance.ea_pk_y` — must come from the
///   election authority's published session key for `voting_round_id`.
///   Wiring `ea_pk` from the same bundle that carries the proof lets a
///   malicious client choose a key it controls.
///
/// # Proof-attested outputs
///
/// The following public inputs are derived outside the circuit but
/// constrained in-circuit against authenticated inputs and private witnesses;
/// successful verification is itself their authentication and the caller does
/// not need a separate trusted channel:
///
/// - `instance.van_nullifier`
/// - `instance.r_vpk_x`, `instance.r_vpk_y`
/// - `instance.vote_authority_note_new`
/// - `instance.vote_commitment`
pub fn verify_vote_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    let (params, _pk, vk) = vote_proof_cached_keys();

    let public_inputs = instance.to_halo2_instance();

    let strategy = SingleVerifier::new(params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(params, vk, strategy, &[&[&public_inputs]], &mut transcript)
        .map_err(|e| format!("vote proof verification failed: {:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProveError;
    use halo2_proofs::plonk;
    use pasta_curves::group::ff::PrimeField;
    use pasta_curves::pallas;

    fn serialize_instance(instance: &Instance) -> Vec<u8> {
        instance
            .to_halo2_instance()
            .into_iter()
            .flat_map(|input| input.to_repr())
            .collect()
    }

    fn minimal_instance() -> Instance {
        Instance::from_parts(
            pallas::Base::from(1),
            pallas::Base::from(2),
            pallas::Base::from(3),
            pallas::Base::from(4),
            pallas::Base::from(5),
            pallas::Base::from(6),
            pallas::Base::from(7),
            pallas::Base::from(8),
            pallas::Base::from(9),
            pallas::Base::from(10),
            pallas::Base::from(11),
        )
    }

    #[test]
    fn create_vote_proof_signature_returns_result() {
        let _: fn(Circuit, &Instance) -> Result<Vec<u8>, ProveError> = create_vote_proof;
    }

    #[test]
    #[ignore = "long-running K=14 proof keygen; run when touching vote proof creation"]
    fn create_vote_proof_returns_err_for_missing_witnesses() {
        let instance = minimal_instance();
        let err = create_vote_proof(Circuit::default(), &instance).unwrap_err();

        assert!(matches!(err, ProveError::Halo2(plonk::Error::Synthesis)));
    }

    #[test]
    fn public_input_count_matches_instance_layout() {
        let instance = minimal_instance();

        assert_eq!(
            instance.to_halo2_instance().len(),
            Instance::NUM_PUBLIC_INPUTS
        );
        assert_eq!(
            serialize_instance(&instance).len(),
            Instance::NUM_PUBLIC_INPUTS * 32
        );
    }

    #[test]
    #[ignore = "expensive end-to-end proof generation; run with --ignored when touching verification"]
    fn typed_verify_accepts_proof_created_by_typed_builder() {
        use crate::circuit::elgamal::spend_auth_g_affine;
        use crate::vote_proof::build_vote_proof_from_delegation;
        use group::Curve;
        use orchard::keys::SpendingKey;

        let sk = SpendingKey::from_bytes([0x42; 32]).expect("valid test spending key");
        let ea_pk =
            (pallas::Point::from(spend_auth_g_affine()) * pallas::Scalar::from(42u64)).to_affine();
        let bundle = build_vote_proof_from_delegation(
            &sk,
            1,
            12_500_000,
            pallas::Base::from(0xDEAD_u64),
            pallas::Base::from(0xCAFE_u64),
            [pallas::Base::zero(); crate::vote_proof::VOTE_COMM_TREE_DEPTH],
            0,
            123,
            1,
            1,
            ea_pk,
            pallas::Scalar::from(7u64),
            65535,
            true,
        )
        .expect("vote proof builder should produce a valid proof");

        verify_vote_proof(&bundle.proof, &bundle.instance)
            .expect("typed verifier should accept the builder's proof and public inputs");
    }
}
