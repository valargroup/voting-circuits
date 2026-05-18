//! Real Halo2 prove/verify for the Share Reveal circuit (ZKP #3).
//!
//! Follows the same pattern as `delegation/prove.rs` but for the
//! 5-condition share reveal circuit at K=11.

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

// ================================================================
// Prove
// ================================================================

/// Create a real Halo2 proof for the share reveal circuit.
///
/// Returns the serialized proof bytes. The caller must have constructed
/// a valid `Circuit` (with all witnesses populated) and a matching
/// `Instance` (7 public inputs).
///
/// **Expensive**: K=11 proof generation takes ~5-15 seconds in release mode.
pub fn create_share_reveal_proof(circuit: Circuit, instance: &Instance) -> Vec<u8> {
    let (params, pk, _vk) = share_reveal_cached_keys();

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
    .expect("share_reveal proof generation should not fail");
    transcript.finalize()
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
/// The following fields are produced by the circuit from private
/// witnesses; successful verification is itself their authentication and
/// the caller does not need a separate trusted channel:
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

/// Verify a share reveal circuit proof from raw field-element bytes.
///
/// This is the lower-level entry point used by the FFI layer. It takes
/// the proof bytes and a flat array of 9 × 32-byte LE-encoded Pallas
/// base field elements (the public inputs in canonical order).
///
/// Returns `Ok(())` if verification succeeds, or an error message.
///
/// # Per-slot layout and caller authentication
///
/// The per-slot meaning of `public_inputs_bytes` matches the offsets
/// defined at the top of `share_reveal/circuit.rs`. Each entry is
/// annotated with whether it is *proof-attested* (the proof itself
/// authenticates the value) or *caller-authenticated* (the caller MUST
/// source it from a trusted channel — see `verify_share_reveal_proof`
/// for the same contract on the typed entry point).
///
/// ```text
/// bytes[  0.. 32] = share_nullifier      [proof-attested]
/// bytes[ 32.. 64] = enc_share_c1_x       [proof-attested]
/// bytes[ 64.. 96] = enc_share_c1_y       [proof-attested]
/// bytes[ 96..128] = enc_share_c2_x       [proof-attested]
/// bytes[128..160] = enc_share_c2_y       [proof-attested]
/// bytes[160..192] = proposal_id          [caller-authenticated]
/// bytes[192..224] = vote_decision        [caller-authenticated]
/// bytes[224..256] = vote_comm_tree_root  [caller-authenticated]
/// bytes[256..288] = voting_round_id      [caller-authenticated]
/// ```
pub fn verify_share_reveal_proof_raw(
    proof: &[u8],
    public_inputs_bytes: &[u8],
) -> Result<(), String> {
    use pasta_curves::group::ff::PrimeField;

    const NUM_PUBLIC_INPUTS: usize = 9;
    const EXPECTED_BYTES: usize = NUM_PUBLIC_INPUTS * 32;

    if public_inputs_bytes.len() != EXPECTED_BYTES {
        return Err(format!(
            "expected {} bytes ({} × 32) for public inputs, got {}",
            EXPECTED_BYTES,
            NUM_PUBLIC_INPUTS,
            public_inputs_bytes.len()
        ));
    }

    // Deserialize each 32-byte chunk as a Pallas Fp element.
    // Note: the share reveal circuit's public inputs live on the Vesta
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

    let (params, _pk, vk) = share_reveal_cached_keys();

    let strategy = SingleVerifier::new(params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(params, vk, strategy, &[&[&public_inputs]], &mut transcript)
        .map_err(|e| format!("share_reveal verification failed: {:?}", e))
}
