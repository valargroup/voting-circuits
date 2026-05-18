//! Real Halo2 prove/verify for the vote proof circuit (ZKP #2).
//!
//! Follows the same pattern as `delegation/prove.rs` but for the
//! 11-condition vote proof circuit at K=14.

use std::string::String;
use std::vec::Vec;

use halo2_proofs::{
    pasta::EqAffine,
    plonk::{self, create_proof, keygen_pk, keygen_vk, verify_proof, SingleVerifier},
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use pasta_curves::{pallas, vesta};
use rand::rngs::OsRng;

use super::circuit::{Circuit, Instance, K};

/// Number of public inputs for the vote proof circuit.
const NUM_PUBLIC_INPUTS: usize = 9;

// ================================================================
// Cached params + keys
// ================================================================

// Keygen is deterministic and expensive (~30s on device). Compute once
// per process and reuse for all subsequent proofs and verifications.
static VOTE_PROOF_PK_CACHE: std::sync::OnceLock<(
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
)> = std::sync::OnceLock::new();

fn get_vote_proof_keys() -> &'static (
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
) {
    VOTE_PROOF_PK_CACHE.get_or_init(|| {
        let params = Params::new(K);
        let empty_circuit = Circuit::default();
        let vk = keygen_vk(&params, &empty_circuit).expect("vote_proof keygen_vk should not fail");
        let pk = keygen_pk(&params, vk.clone(), &empty_circuit)
            .expect("vote_proof keygen_pk should not fail");
        (params, pk, vk)
    })
}

/// Warm the process-lifetime vote proof params/proving-key cache.
///
/// This lets callers pay deterministic keygen before the first user-visible
/// proof generation path needs the key.
pub fn warm_vote_proof_keys() {
    let _ = get_vote_proof_keys();
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
/// Returns the serialized proof bytes. The caller must have constructed
/// a valid `Circuit` (with all witnesses populated) and a matching
/// `Instance` (9 public inputs).
///
/// **Expensive**: K=14 proof generation takes ~30-60 seconds in release mode.
/// Params and keys are cached so only the first call pays keygen.
pub fn create_vote_proof(circuit: Circuit, instance: &Instance) -> Vec<u8> {
    let (params, pk, _vk) = get_vote_proof_keys();

    let public_inputs = instance.to_halo2_instance();

    let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
    create_proof(
        params,
        pk,
        &[circuit],
        &[&[&public_inputs]],
        OsRng,
        &mut transcript,
    )
    .expect("vote proof generation should not fail");
    transcript.finalize()
}

// ================================================================
// Verify
// ================================================================

/// Verify a vote proof circuit proof given serialized proof bytes and
/// the 9 public inputs.
///
/// Returns `Ok(())` if verification succeeds, or an error message.
///
/// # Caller-authenticated inputs
///
/// `constrain_instance` pins each public input to whatever value the
/// *verifier* supplies; the protocol cannot tell whether that value was
/// the *right* one. The following fields of `instance` MUST be sourced
/// from a trusted channel (a signed governance announcement, a signed
/// chain head) before calling this function. Substituting them is not
/// detectable from the proof alone — most notably, an attacker can pick
/// their own `ea_pk` (`= sk·G` for an `sk` they know) and decrypt all
/// posted shares, with the proof still verifying:
///
/// - `instance.proposal_id` — must come from the active session's
///   published proposal list.
/// - `instance.voting_round_id` — must come from the same governance
///   announcement as `proposal_id`.
/// - `instance.vote_comm_tree_root` — must be the vote commitment tree
///   root at `vote_comm_tree_anchor_height` (verifier looks it up by
///   height, not by accepting it from the prover bundle).
/// - `instance.vote_comm_tree_anchor_height` — must be a valid chain
///   height accepted by the consuming chain's anchor-validity check.
/// - `instance.ea_pk_x`, `instance.ea_pk_y` — must come from the
///   election authority's published session key for `voting_round_id`.
///   Wiring `ea_pk` from the same bundle that carries the proof lets a
///   malicious client choose a key it controls.
///
/// # Proof-attested outputs
///
/// The following fields are produced by the circuit from private
/// witnesses; successful verification is itself their authentication and
/// the caller does not need a separate trusted channel:
///
/// - `instance.van_nullifier`
/// - `instance.r_vpk_x`, `instance.r_vpk_y`
/// - `instance.vote_authority_note_new`
/// - `instance.vote_commitment`
pub fn verify_vote_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    let (params, _pk, vk) = get_vote_proof_keys();

    let public_inputs = instance.to_halo2_instance();

    let strategy = SingleVerifier::new(params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(params, vk, strategy, &[&[&public_inputs]], &mut transcript)
        .map_err(|e| format!("vote proof verification failed: {:?}", e))
}

/// Verify a vote proof circuit proof from raw field-element bytes.
///
/// This is the lower-level entry point used by the FFI layer. It takes
/// the proof bytes and a flat array of `NUM_PUBLIC_INPUTS × 32` bytes of
/// LE-encoded Pallas base field elements (the public inputs in canonical
/// order).
///
/// Returns `Ok(())` if verification succeeds, or an error message.
///
/// # Per-slot layout and caller authentication
///
/// The per-slot meaning of `public_inputs_bytes` matches the offsets
/// defined at the top of `vote_proof/circuit.rs`. Each entry is annotated
/// with whether it is *proof-attested* (the proof itself authenticates
/// the value) or *caller-authenticated* (the caller MUST source it from
/// a trusted channel — see `verify_vote_proof` for the same contract on
/// the typed entry point, including why wiring `ea_pk_x/y` from the
/// proof bundle is a custody-attack surface).
///
/// ```text
/// bytes[  0.. 32] = van_nullifier              [proof-attested]
/// bytes[ 32.. 64] = r_vpk_x                    [proof-attested]
/// bytes[ 64.. 96] = r_vpk_y                    [proof-attested]
/// bytes[ 96..128] = vote_authority_note_new    [proof-attested]
/// bytes[128..160] = vote_commitment            [proof-attested]
/// bytes[160..192] = vote_comm_tree_root        [caller-authenticated]
/// bytes[192..224] = vote_comm_tree_anchor_h    [caller-authenticated]
/// bytes[224..256] = proposal_id                [caller-authenticated]
/// bytes[256..288] = voting_round_id            [caller-authenticated]
/// bytes[288..320] = ea_pk_x                    [caller-authenticated]
/// bytes[320..352] = ea_pk_y                    [caller-authenticated]
/// ```
pub fn verify_vote_proof_raw(proof: &[u8], public_inputs_bytes: &[u8]) -> Result<(), String> {
    use pasta_curves::group::ff::PrimeField;

    let expected_len = NUM_PUBLIC_INPUTS * 32;
    if public_inputs_bytes.len() != expected_len {
        return Err(format!(
            "expected {} bytes ({} × 32) for public inputs, got {}",
            expected_len,
            NUM_PUBLIC_INPUTS,
            public_inputs_bytes.len()
        ));
    }

    // Deserialize each 32-byte chunk as a Pallas Fp element.
    // The vote proof circuit's public inputs live on the Vesta
    // scalar field, which is the same as the Pallas base field.
    let mut public_inputs: Vec<vesta::Scalar> = Vec::with_capacity(NUM_PUBLIC_INPUTS);
    for i in 0..NUM_PUBLIC_INPUTS {
        let start = i * 32;
        let mut repr = [0u8; 32];
        repr.copy_from_slice(&public_inputs_bytes[start..start + 32]);
        let fp_opt: Option<pallas::Base> = pallas::Base::from_repr(repr).into();
        match fp_opt {
            Some(f) => public_inputs.push(f),
            None => {
                return Err(format!(
                    "public input {} is not a canonical Pallas Fp encoding",
                    i
                ))
            }
        }
    }

    let (params, _pk, vk) = get_vote_proof_keys();

    let strategy = SingleVerifier::new(params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(params, vk, strategy, &[&[&public_inputs]], &mut transcript)
        .map_err(|e| format!("vote proof verification failed: {:?}", e))
}
