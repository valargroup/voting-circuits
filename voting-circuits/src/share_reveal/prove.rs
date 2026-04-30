//! Real Halo2 prove/verify for the Share Reveal circuit (ZKP #3).
//!
//! Follows the same pattern as `delegation/prove.rs` but for the
//! 5-condition share reveal circuit at K=11.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

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

#[cfg(feature = "std")]
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
/// proving latency if repeated for every revealed share. With `std` enabled,
/// compute it once per process and reuse it for both proving and verification.
#[cfg(feature = "std")]
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
    #[cfg(feature = "std")]
    let (params, pk, _vk) = share_reveal_cached_keys();

    #[cfg(not(feature = "std"))]
    let (params_owned, pk_owned, _vk) = {
        let params = share_reveal_params();
        let (pk, vk) = share_reveal_proving_key(&params);
        (params, pk, vk)
    };
    #[cfg(not(feature = "std"))]
    let (params, pk) = (&params_owned, &pk_owned);

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
pub fn verify_share_reveal_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    #[cfg(feature = "std")]
    let (params, _pk, vk) = share_reveal_cached_keys();

    #[cfg(not(feature = "std"))]
    let (params_owned, _pk, vk_owned) = {
        let params = share_reveal_params();
        let (pk, vk) = share_reveal_proving_key(&params);
        (params, pk, vk)
    };
    #[cfg(not(feature = "std"))]
    let (params, vk) = (&params_owned, &vk_owned);

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

    #[cfg(feature = "std")]
    let (params, _pk, vk) = share_reveal_cached_keys();

    #[cfg(not(feature = "std"))]
    let (params_owned, _pk, vk_owned) = {
        let params = share_reveal_params();
        let (pk, vk) = share_reveal_proving_key(&params);
        (params, pk, vk)
    };
    #[cfg(not(feature = "std"))]
    let (params, vk) = (&params_owned, &vk_owned);

    let strategy = SingleVerifier::new(params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(params, vk, strategy, &[&[&public_inputs]], &mut transcript)
        .map_err(|e| format!("share_reveal verification failed: {:?}", e))
}
