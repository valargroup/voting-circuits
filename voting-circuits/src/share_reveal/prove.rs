//! Real Halo2 prove/verify for the Share Reveal circuit (ZKP #3).
//!
//! Follows the same pattern as `delegation/prove.rs` but for the
//! 5-condition share reveal circuit at K=11.

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
// Params / key generation
// ================================================================

static SHARE_REVEAL_PK_CACHE: std::sync::OnceLock<(
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
)> = std::sync::OnceLock::new();

/// Generate the IPA params (SRS) for the share reveal circuit.
/// Deterministic for a given `K`.
///
/// **Expensive**: K=11 params generation takes ~1 second.
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
) -> (plonk::ProvingKey<EqAffine>, plonk::VerifyingKey<EqAffine>) {
    let empty_circuit = Circuit::default();
    let vk = keygen_vk(params, &empty_circuit).expect("share_reveal keygen_vk should not fail");
    let pk = keygen_pk(params, vk.clone(), &empty_circuit)
        .expect("share_reveal keygen_pk should not fail");
    (pk, vk)
}

/// Return cached params and proving/verifying keys for the share reveal circuit.
///
/// Key generation is deterministic and expensive enough to dominate helper
/// proving latency if repeated for every revealed share. Compute it once per
/// process and reuse it for both proving and verification.
pub fn share_reveal_cached_keys() -> &'static (
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
) {
    SHARE_REVEAL_PK_CACHE.get_or_init(|| {
        let params = share_reveal_params();
        let (pk, vk) = share_reveal_proving_key(&params);
        (params, pk, vk)
    })
}

/// Warm the process-lifetime share reveal params/proving-key cache.
///
/// This lets callers pay deterministic keygen before the first user-visible
/// proof generation or verification path needs the key.
pub fn warm_share_reveal_keys() {
    let _ = share_reveal_cached_keys();
}

// ================================================================
// Prove
// ================================================================

/// Create a real Halo2 proof for the share reveal circuit.
///
/// Returns the serialized proof bytes. Returns an error if the caller
/// provides a circuit without all witnesses populated or an instance
/// with 9 public inputs that Halo2 cannot prove against.
///
/// **Expensive**: K=11 proof generation takes ~5-15 seconds in release mode.
/// Params and keys are cached so only the first call pays keygen.
pub fn create_share_reveal_proof(
    circuit: Circuit,
    instance: &Instance,
) -> Result<Vec<u8>, ProveError> {
    let (params, pk, _vk) = share_reveal_cached_keys();

    let public_inputs = instance.to_halo2_instance();

    create_proof_bytes(params, pk, circuit, &public_inputs)
}

// ================================================================
// Verify
// ================================================================

/// Verify a share reveal circuit proof given serialized proof bytes and
/// the 9 public inputs.
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
/// - `instance.vote_decision` — the on-chain reveal of the voter's
///   choice; the caller must accept it only as part of the same chain
///   bundle that carries the proof, not from an untrusted side channel
///   (the proof binds it but does not assert it equals any particular
///   value).
///
/// # Proof-attested outputs
///
/// The following public inputs are derived outside the circuit but
/// constrained in-circuit against authenticated inputs and private witnesses;
/// successful verification is itself their authentication and the caller does
/// not need a separate trusted channel:
///
/// - `instance.share_nullifier`
/// - `instance.enc_share_c1_x`, `instance.enc_share_c1_y`
/// - `instance.enc_share_c2_x`, `instance.enc_share_c2_y`
pub fn verify_share_reveal_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    let (params, _pk, vk) = share_reveal_cached_keys();

    let public_inputs = instance.to_halo2_instance();

    let strategy = SingleVerifier::new(params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(params, vk, strategy, &[&[&public_inputs]], &mut transcript)
        .map_err(|e| format!("share_reveal verification failed: {:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProveError;
    use halo2_proofs::plonk;
    use pasta_curves::pallas;

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
}
