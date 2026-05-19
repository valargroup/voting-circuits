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
/// that Halo2 cannot prove against.
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
/// the [`Instance::NUM_PUBLIC_INPUTS`] public inputs.
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
/// - `instance.share_index` — the on-chain revealed slot; the proof binds it
///   to the selected share commitment, but the caller must pass the same value
///   carried by the external reveal payload.
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
    use crate::circuit::elgamal::{elgamal_encrypt, spend_auth_g_affine};
    use crate::share_reveal::{build_share_reveal, ShareRevealBundle};
    use crate::vote_proof::{poseidon_hash_2, share_commitment, VOTE_COMM_TREE_DEPTH};
    use crate::ProveError;
    use halo2_proofs::plonk;
    use pasta_curves::group::ff::PrimeField;
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
            pallas::Base::from(10),
        )
    }

    fn serialize_instance(instance: &Instance) -> Vec<u8> {
        instance
            .to_halo2_instance()
            .into_iter()
            .flat_map(|input| input.to_repr())
            .collect()
    }

    fn make_share_reveal_bundle() -> ShareRevealBundle {
        let ea_pk = pallas::Point::from(spend_auth_g_affine()) * pallas::Scalar::from(42u64);

        let shares: [u64; 16] = [625; 16];
        let randomness: [pallas::Base; 16] =
            core::array::from_fn(|i| pallas::Base::from((i as u64 + 1) * 101));
        let share_blinds: [pallas::Base; 16] =
            core::array::from_fn(|i| pallas::Base::from(1001u64 + i as u64));
        let mut c1_x = [pallas::Base::zero(); 16];
        let mut c2_x = [pallas::Base::zero(); 16];
        let mut c1_y = [pallas::Base::zero(); 16];
        let mut c2_y = [pallas::Base::zero(); 16];
        for i in 0..16 {
            let (cx1, cx2, cy1, cy2) =
                elgamal_encrypt(pallas::Base::from(shares[i]), randomness[i], ea_pk);
            c1_x[i] = cx1;
            c2_x[i] = cx2;
            c1_y[i] = cy1;
            c2_y[i] = cy2;
        }

        let share_comms: [pallas::Base; 16] = core::array::from_fn(|i| {
            share_commitment(
                i as u32,
                share_blinds[i],
                c1_x[i],
                c2_x[i],
                c1_y[i],
                c2_y[i],
            )
        });

        let mut empty_roots = [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH];
        empty_roots[0] = poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero());
        for i in 1..VOTE_COMM_TREE_DEPTH {
            empty_roots[i] = poseidon_hash_2(empty_roots[i - 1], empty_roots[i - 1]);
        }

        let share_idx: u32 = 2;
        build_share_reveal(
            empty_roots,
            0,
            share_comms,
            share_blinds[share_idx as usize],
            c1_x[share_idx as usize],
            c2_x[share_idx as usize],
            c1_y[share_idx as usize],
            c2_y[share_idx as usize],
            share_idx,
            pallas::Base::from(3u64),
            pallas::Base::from(1u64),
            pallas::Base::from(999u64),
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
    fn from_parts_matches_public_input_order() {
        let expected = vec![
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
        ];
        let instance = Instance::from_parts(
            expected[0],
            expected[1],
            expected[2],
            expected[3],
            expected[4],
            expected[5],
            expected[6],
            expected[7],
            expected[8],
            expected[9],
        );

        assert_eq!(instance.to_halo2_instance(), expected);
    }

    #[test]
    #[ignore = "expensive end-to-end proof generation; run with --ignored when touching share reveal verification"]
    fn typed_verify_accepts_proof_created_by_typed_builder() {
        let bundle = make_share_reveal_bundle();
        let proof = create_share_reveal_proof(bundle.circuit.clone(), &bundle.instance)
            .expect("share reveal proof generation should succeed");

        verify_share_reveal_proof(&proof, &bundle.instance)
            .expect("typed verifier should accept the builder's proof and public inputs");
    }
}
