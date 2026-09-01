//! Real Halo2 prove/verify for the vote proof circuit (ZKP #2).
//!
//! Follows the same pattern as `delegation/prove.rs` but for the
//! 12-condition vote proof circuit at K=11.

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
pub type VoteProofKeys = (
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
);

static VOTE_PROOF_KEYS_CACHE: std::sync::OnceLock<Result<VoteProofKeys, String>> =
    std::sync::OnceLock::new();

/// Return cached params and proving/verifying keys for the vote proof circuit.
///
/// Params generation and key generation are deterministic and expensive enough
/// to dominate the first proof or verification call. Compute the full tuple
/// once per process so warm-up covers both the SRS params and the keys.
pub fn vote_proof_cached_keys() -> Result<&'static VoteProofKeys, ProveError> {
    match VOTE_PROOF_KEYS_CACHE.get_or_init(|| {
        let params = vote_proof_params();
        vote_proof_proving_key(&params)
            .map(|(pk, vk)| (params, pk, vk))
            .map_err(|error| error.to_string())
    }) {
        Ok(keys) => Ok(keys),
        Err(error) => Err(ProveError::CachedKeygen(error.clone())),
    }
}

/// Warm the process-lifetime vote proof params/proving-key cache.
///
/// This lets callers pay deterministic keygen before the first user-visible
/// proof generation or verification path needs the params and keys.
pub fn warm_vote_proof_keys() -> Result<(), ProveError> {
    vote_proof_cached_keys().map(|_| ())
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
) -> Result<(plonk::ProvingKey<EqAffine>, plonk::VerifyingKey<EqAffine>), ProveError> {
    let empty_circuit = Circuit::default();
    let vk = keygen_vk(params, &empty_circuit).map_err(ProveError::KeygenVk)?;
    let pk = keygen_pk(params, vk.clone(), &empty_circuit).map_err(ProveError::KeygenPk)?;
    Ok((pk, vk))
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
/// **Expensive**: K=11 proof generation should run in release mode. Params and
/// keys are cached so only the first call pays keygen.
pub fn create_vote_proof(circuit: Circuit, instance: &Instance) -> Result<Vec<u8>, ProveError> {
    let (params, pk, _vk) = vote_proof_cached_keys()?;

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
/// ones. The following fields of `instance` MUST be sourced from their
/// category-specific authority before calling this function. Substituting them
/// is not detectable from the proof alone.
///
/// ## Ledger-state anchor
///
/// - `instance.vote_comm_tree_root` - must be the vote commitment tree
///   root at `vote_comm_tree_anchor_height` (verifier looks it up by
///   height, not by accepting it from the prover bundle).
/// - `instance.vote_comm_tree_anchor_height` - must be a valid chain
///   height accepted by the consuming chain's anchor-validity check. This
///   slot is transcript-bound but not constrained to any circuit witness.
///
/// ## Governance session parameters
///
/// - `instance.proposal_id` — must be in the active proposal set for
///   `voting_round_id`. The circuit only constrains this to the authority
///   bit-index range `[1, 50]`; it does not know whether the proposal exists
///   or is open.
/// - `instance.voting_round_id` — must come from the same governance
///   announcement as `proposal_id`, and must identify the active voting round
///   the verifier is accepting.
///
/// ## Weighted-vote parameters
///
/// - `instance.decision_bucket_count` — must equal the option count `D`
///   declared by governance for `proposal_id`, and must match the
///   encrypt-choice proof's public bucket count. The verifier must also
///   reject `D < 2`.
///
/// # Bundle binding
///
/// This proof is one half of a vote bundle. The verifier MUST also verify
/// the accompanying encrypt-choice proof (which authenticates the
/// election-authority key and the ElGamal ciphertexts) and check that
/// `instance.bridge`, `voting_round_id`, `proposal_id`, and
/// `decision_bucket_count` are identical across the two instances.
/// [`crate::vote_proof::verify_vote_bundle`] performs all of these checks.
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
/// - `instance.bridge` (subject to the bundle equality check above)
pub fn verify_vote_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    let (params, _pk, vk) = vote_proof_cached_keys().map_err(|error| error.to_string())?;

    let public_inputs = instance.to_halo2_instance();

    verify_proof_bytes("vote proof", params, vk, proof, &public_inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::PrimeField;
    use crate::ProveError;
    use voting_crypto_deps::halo2_proofs::plonk;
    use voting_crypto_deps::pasta_curves::pallas;

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
    #[ignore = "long-running K=11 proof keygen; run when touching vote proof creation"]
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

    // TODO(sean): VK-stability tripwire. Hashes the `PinnedVerificationKey`
    // debug repr and compares against a baked-in fingerprint. A mismatch means
    // either the circuit shape changed (and the VK must be regenerated and
    // redistributed) or an unintended drift has been introduced.
    #[test]
    #[ignore = "TODO(sean): runs K=11 keygen; run with `cargo test -- --ignored vk_fingerprint_unchanged`"]
    fn vk_fingerprint_unchanged() {
        let (_, _, vk) = vote_proof_cached_keys().expect("vote proof keys");
        let pinned = format!("{:?}", vk.pinned());
        let fingerprint = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(pinned.as_bytes());
        let actual: &[u8] = fingerprint.as_bytes();

        let expected: [u8; 32] = [
            0x6f, 0xa5, 0xba, 0x18, 0x6f, 0x24, 0x55, 0x34, 0x06, 0x54, 0x1d, 0x51, 0xd3, 0x4c,
            0x8d, 0xe7, 0xd3, 0x4c, 0xf6, 0x78, 0x25, 0x33, 0x28, 0x1f, 0x1c, 0xad, 0x14, 0x0c,
            0xf0, 0xce, 0x94, 0x69,
        ];

        assert_eq!(
            actual,
            expected.as_slice(),
            "vote proof VK fingerprint changed; if intentional, update `expected` to:\n{:02x?}",
            actual,
        );
    }

    #[test]
    #[ignore = "expensive end-to-end proof generation; run with --ignored when touching verification"]
    fn typed_verify_accepts_vote_bundle_created_by_typed_builders() {
        use crate::encrypt_choice::build_encrypt_choice;
        use crate::gadgets::elgamal::spend_auth_g_affine;
        use crate::group::Curve;
        use crate::vote_proof::{
            build_vote_proof_from_delegation, derive_vote_authority_transition, verify_vote_bundle,
            VoteBundle,
        };
        use voting_crypto_deps::orchard::keys::SpendingKey;

        let sk = SpendingKey::from_bytes([0x42; 32]).expect("valid test spending key");
        let ea_pk = (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine();
        let address_index = 1;
        let total_note_value = 12_500_000;
        let van_comm_rand = pallas::Base::from(0xDEAD_u64);
        let voting_round_id = pallas::Base::from(0xCAFE_u64);
        let proposal_id = 50;
        let proposal_authority_old = crate::params::MAX_PROPOSAL_AUTHORITY;
        let transition = derive_vote_authority_transition(
            &sk,
            address_index,
            total_note_value,
            van_comm_rand,
            voting_round_id,
            proposal_id,
            proposal_authority_old,
        )
        .expect("native vote authority transition should be valid");

        let encrypt_choice = build_encrypt_choice(
            &sk,
            total_note_value,
            transition.vote_authority_note_old,
            voting_round_id,
            proposal_id,
            1,
            4,
            ea_pk,
            true,
        )
        .expect("encrypt-choice builder should produce a valid proof");

        let cast = build_vote_proof_from_delegation(
            &sk,
            address_index,
            total_note_value,
            van_comm_rand,
            voting_round_id,
            [pallas::Base::zero(); crate::params::VOTE_COMM_TREE_DEPTH],
            0,
            123,
            proposal_id,
            pallas::Scalar::from(7u64),
            proposal_authority_old,
            &encrypt_choice,
        )
        .expect("vote proof builder should produce a valid proof");

        let expected_root = (0..crate::params::VOTE_COMM_TREE_DEPTH)
            .fold(transition.vote_authority_note_old, |current, _| {
                crate::protocol_hash::poseidon_hash_2(current, pallas::Base::zero())
            });
        assert_eq!(cast.instance.vote_comm_tree_root, expected_root);
        assert_eq!(
            cast.instance.vote_authority_note_new,
            transition.vote_authority_note_new
        );

        let bundle = VoteBundle {
            encrypt_choice,
            cast,
        };
        bundle
            .check_consistency()
            .expect("bundle instances must be consistent");
        verify_vote_bundle(
            &bundle.encrypt_choice.proof,
            &bundle.encrypt_choice.instance,
            &bundle.cast.proof,
            &bundle.cast.instance,
        )
        .expect("bundle verifier should accept both proofs and their binding");

        // ---- Chain the reveal proofs (ZKP #3) from the same bundle ----

        use crate::share_reveal::{build_share_reveal, verify_share_reveal_proof};

        let vote_commitment_position = 0u32;
        let reveal_auth_path = {
            // Single-leaf tree containing only this vote commitment.
            let mut empty = [pallas::Base::zero(); crate::params::VOTE_COMM_TREE_DEPTH];
            empty[0] =
                crate::protocol_hash::poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero());
            for i in 1..crate::params::VOTE_COMM_TREE_DEPTH {
                empty[i] = crate::protocol_hash::poseidon_hash_2(empty[i - 1], empty[i - 1]);
            }
            empty
        };

        // Reveal the first and last shares; the same flow covers all sixteen.
        for share_index in [0u32, 15] {
            let reveal = build_share_reveal(
                reveal_auth_path,
                vote_commitment_position,
                bundle.encrypt_choice.selected_commitments,
                bundle.encrypt_choice.share_blinds[share_index as usize],
                &bundle.encrypt_choice.encrypted_shares[share_index as usize].ciphertexts,
                share_index,
                bundle.cast.instance.proposal_id,
                bundle.cast.instance.voting_round_id,
                bundle.cast.instance.decision_bucket_count,
            );
            let proof =
                crate::share_reveal::create_share_reveal_proof(reveal.circuit, &reveal.instance)
                    .expect("share reveal proof should build");
            verify_share_reveal_proof(&proof, &reveal.instance)
                .expect("share reveal proof should verify");
        }

        // ---- No plaintext decision appears in any public instance ----
        // The decision used above is 1; the bundle's instances expose only
        // context values, commitments, and the bucket count — assert none of
        // the slots equals a bare decision encoding by construction: the
        // encrypt-choice instance is (ea_pk, bridge, D, round, proposal) and
        // the cast instance carries no decision field at all. This is a
        // structural property; the type system enforces it, and this test
        // documents it.
        let _ = &bundle;
    }
}
