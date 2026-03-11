//! Real Halo2 prove/verify for the vote proof circuit (ZKP #2).
//!
//! Follows the same pattern as `delegation/prove.rs` but for the
//! 11-condition vote proof circuit at K=14.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use halo2_proofs::{
    pasta::EqAffine,
    plonk::{self, create_proof, keygen_pk, keygen_vk, verify_proof, SingleVerifier},
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use rand::rngs::OsRng;

use super::circuit::{Circuit, Instance, K};

// ================================================================
// Cached params + keys
// ================================================================

// Keygen is deterministic and expensive (~30s on device). Compute once
// per process and reuse for all subsequent proofs and verifications.
#[cfg(feature = "std")]
static VOTE_PROOF_PK_CACHE: std::sync::OnceLock<(
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
)> = std::sync::OnceLock::new();

#[cfg(feature = "std")]
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

// ================================================================
// Params / key generation (public API, non-cached fallbacks)
// ================================================================

/// Generate the IPA params (SRS) for the vote proof circuit.
/// Deterministic for a given `K`.
///
/// Prefer [`get_vote_proof_keys`] when the `std` feature is enabled —
/// it caches the result across calls.
pub fn vote_proof_params() -> Params<EqAffine> {
    Params::new(K)
}

/// Generate the proving and verifying keys for the vote proof circuit.
///
/// Uses `Circuit::default()` (all witnesses unknown) as the empty circuit
/// for key generation — the same pattern as the Orchard action circuit.
///
/// Prefer [`get_vote_proof_keys`] when the `std` feature is enabled —
/// it caches the result across calls.
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
/// `Instance` (11 public inputs).
///
/// **Expensive**: K=14 proof generation takes ~30-60 seconds in release mode.
/// Params and keys are cached (with `std`) so only the first call pays keygen.
pub fn create_vote_proof(circuit: Circuit, instance: &Instance) -> Vec<u8> {
    #[cfg(feature = "std")]
    let (params, pk, _vk) = get_vote_proof_keys();

    #[cfg(not(feature = "std"))]
    let (params_owned, pk, _vk) = {
        let p = vote_proof_params();
        let (pk, vk) = vote_proof_proving_key(&p);
        (p, pk, vk)
    };
    #[cfg(not(feature = "std"))]
    let params = &params_owned;

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
/// the 11 public inputs.
///
/// Returns `Ok(())` if verification succeeds, or an error message.
pub fn verify_vote_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    #[cfg(feature = "std")]
    let (params, _pk, vk) = get_vote_proof_keys();

    #[cfg(not(feature = "std"))]
    let (params_owned, _pk, vk) = {
        let p = vote_proof_params();
        let (pk, vk) = vote_proof_proving_key(&p);
        (p, pk, vk)
    };
    #[cfg(not(feature = "std"))]
    let params = &params_owned;

    let public_inputs = instance.to_halo2_instance();

    let strategy = SingleVerifier::new(params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(params, vk, strategy, &[&[&public_inputs]], &mut transcript)
        .map_err(|e| format!("vote proof verification failed: {:?}", e))
}
