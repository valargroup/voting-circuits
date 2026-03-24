//! Real Halo2 prove/verify for the delegation circuit (ZKP #1).
//!
//! Follows the same pattern as `sdk/circuits/src/toy.rs` but for the full
//! 15-condition delegation circuit at K=14.

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

/// Generate the IPA params (SRS) for the delegation circuit.
/// Deterministic for a given `K`.
///
/// **Expensive**: K=14 params generation takes several seconds.
/// Callers should cache the result.
pub fn delegation_params() -> Params<EqAffine> {
    Params::new(K)
}

/// Generate the proving and verifying keys for the delegation circuit.
///
/// Uses `Circuit::default()` (all witnesses unknown) as the empty circuit
/// for key generation — the same pattern as the Orchard action circuit.
///
/// **Expensive**: first call involves full circuit layout. Callers should
/// cache the result alongside the params.
pub fn delegation_proving_key(
    params: &Params<EqAffine>,
) -> (
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
) {
    let empty_circuit = Circuit::default();
    let vk = keygen_vk(params, &empty_circuit).expect("delegation keygen_vk should not fail");
    let pk = keygen_pk(params, vk.clone(), &empty_circuit)
        .expect("delegation keygen_pk should not fail");
    (pk, vk)
}

// ================================================================
// Prove
// ================================================================

/// Create a real Halo2 proof for the delegation circuit.
///
/// Returns the serialized proof bytes. The caller must have constructed
/// a valid `Circuit` (with all witnesses populated) and a matching
/// `Instance` (14 public inputs).
///
/// **Expensive**: K=14 proof generation takes ~30-60 seconds in release mode.
pub fn create_delegation_proof(circuit: Circuit, instance: &Instance) -> Vec<u8> {
    let params = delegation_params();
    let (pk, _vk) = delegation_proving_key(&params);

    let public_inputs = instance.to_halo2_instance();

    let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
    create_proof(
        &params,
        &pk,
        &[circuit],
        &[&[&public_inputs]],
        OsRng,
        &mut transcript,
    )
    .expect("delegation proof generation should not fail");
    transcript.finalize()
}

// ================================================================
// Verify
// ================================================================

/// Verify a delegation circuit proof given serialized proof bytes and
/// the 14 public inputs.
///
/// Returns `Ok(())` if verification succeeds, or an error message.
pub fn verify_delegation_proof(
    proof: &[u8],
    instance: &Instance,
) -> Result<(), String> {
    let params = delegation_params();
    let (_pk, vk) = delegation_proving_key(&params);

    let public_inputs = instance.to_halo2_instance();

    let strategy = SingleVerifier::new(&params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(&params, &vk, strategy, &[&[&public_inputs]], &mut transcript)
        .map_err(|e| format!("delegation verification failed: {:?}", e))
}

/// Verify a delegation circuit proof from raw field-element bytes.
///
/// This is the lower-level entry point used by the FFI layer. It takes
/// the proof bytes and a flat array of 14 × 32-byte LE-encoded Pallas
/// base field elements (the public inputs in canonical order).
///
/// Returns `Ok(())` if verification succeeds, or an error message.
pub fn verify_delegation_proof_raw(
    proof: &[u8],
    public_inputs_bytes: &[u8],
) -> Result<(), String> {
    use pasta_curves::group::ff::PrimeField;

    if public_inputs_bytes.len() != 14 * 32 {
        return Err(format!(
            "expected 448 bytes (14 × 32) for public inputs, got {}",
            public_inputs_bytes.len()
        ));
    }

    // Deserialize each 32-byte chunk as a Pallas Fp element.
    // Note: the delegation circuit's public inputs live on the Vesta
    // scalar field, which is the same as the Pallas base field.
    let mut public_inputs: Vec<vesta::Scalar> = Vec::with_capacity(14);
    for i in 0..14 {
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

    let params = delegation_params();
    let (_pk, vk) = delegation_proving_key(&params);

    let strategy = SingleVerifier::new(&params);
    let mut transcript = Blake2bRead::<_, EqAffine, Challenge255<_>>::init(proof);

    verify_proof(
        &params,
        &vk,
        strategy,
        &[&[&public_inputs]],
        &mut transcript,
    )
    .map_err(|e| format!("delegation verification failed: {:?}", e))
}

#[cfg(test)]
mod prove_tests {
    use super::*;
    use crate::delegation::builder::{build_delegation_bundle, RealNoteInput};
    use crate::delegation::imt::{ImtProvider, SpacedLeafImtProvider};
    use orchard::{
        keys::{FullViewingKey, Scope, SpendingKey},
        note::{commitment::ExtractedNoteCommitment, Note, Rho},
        tree::{MerkleHashOrchard, MerklePath},
        value::NoteValue,
    };
    use ff::Field;
    use incrementalmerkletree::{Hashable, Level};
    use pasta_curves::pallas;
    use rand::rngs::OsRng;
    use crate::delegation::circuit::K;

    #[test]
    fn real_proof_roundtrip() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let output_recipient = fvk.address_at(1u32, Scope::External);
        let vote_round_id = pallas::Base::random(&mut rng);
        let van_comm_rand = pallas::Base::random(&mut rng);
        let alpha = pallas::Scalar::random(&mut rng);

        // Create a single real note
        let recipient = fvk.address_at(0u32, Scope::External);
        let (_, _, dummy) = Note::dummy(&mut rng, None);
        let note = Note::new(recipient, NoteValue::from_raw(13_000_000), Rho::from_nf_old(dummy.nullifier(&fvk)), &mut rng);
        let cmx = ExtractedNoteCommitment::from(note.commitment());
        let leaf = MerkleHashOrchard::from_cmx(&cmx);
        let empty = MerkleHashOrchard::empty_leaf();
        let mut leaves = [empty; 2];
        leaves[0] = leaf;
        let l1 = MerkleHashOrchard::combine(Level::from(0), &leaves[0], &leaves[1]);
        let mut current = l1;
        for level in 1..32u8 {
            current = MerkleHashOrchard::combine(Level::from(level), &current, &MerkleHashOrchard::empty_root(Level::from(level)));
        }
        let nc_root = current.inner();
        let mut auth_path = [empty; 32];
        auth_path[0] = leaves[1];
        for level in 1..32u8 {
            auth_path[level as usize] = MerkleHashOrchard::empty_root(Level::from(level));
        }
        let merkle_path = MerklePath::from_parts(0u32, auth_path);
        let imt = SpacedLeafImtProvider::new();
        let real_nf = note.nullifier(&fvk);
        let imt_proof = imt.non_membership_proof(real_nf.0).unwrap();

        let input = RealNoteInput { note, fvk: fvk.clone(), merkle_path, imt_proof, scope: Scope::External };
        let bundle = build_delegation_bundle(
            vec![input], &fvk, alpha, output_recipient, vote_round_id, nc_root, van_comm_rand, &imt, &mut rng, None,
        ).unwrap();

        let proof = create_delegation_proof(bundle.circuit, &bundle.instance);
        verify_delegation_proof(&proof, &bundle.instance).expect("real proof roundtrip failed");
    }
}
