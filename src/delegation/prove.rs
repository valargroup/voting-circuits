//! Real Halo2 prove/verify for the delegation circuit (ZKP #1).
//!
//! Follows the same pattern as `sdk/circuits/src/toy.rs` but for the full
//! 15-condition delegation circuit at K=12.

use std::{string::String, vec::Vec};

use halo2_proofs::{
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

pub type DelegationKeys = (
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
);

static DELEGATION_PK_CACHE: std::sync::OnceLock<Result<DelegationKeys, String>> =
    std::sync::OnceLock::new();

/// Generate the IPA params (SRS) for the delegation circuit.
/// Deterministic for a given `K`.
///
/// **Expensive**: K=12 params generation takes several seconds.
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
) -> Result<(plonk::ProvingKey<EqAffine>, plonk::VerifyingKey<EqAffine>), ProveError> {
    let empty_circuit = Circuit::default();
    let vk = keygen_vk(params, &empty_circuit).map_err(ProveError::KeygenVk)?;
    let pk = keygen_pk(params, vk.clone(), &empty_circuit).map_err(ProveError::KeygenPk)?;
    Ok((pk, vk))
}

/// Return cached params and proving/verifying keys for the delegation circuit.
///
/// Key generation is deterministic and expensive enough to dominate proof and
/// verification latency if repeated. Compute it once per process and reuse it.
pub fn delegation_cached_keys() -> Result<&'static DelegationKeys, ProveError> {
    match DELEGATION_PK_CACHE.get_or_init(|| {
        let params = delegation_params();
        delegation_proving_key(&params)
            .map(|(pk, vk)| (params, pk, vk))
            .map_err(|error| error.to_string())
    }) {
        Ok(keys) => Ok(keys),
        Err(error) => Err(ProveError::CachedKeygen(error.clone())),
    }
}

/// Warm the process-lifetime delegation params/proving-key cache.
///
/// This lets callers pay deterministic keygen before the first user-visible
/// proof generation or verification path needs the key.
pub fn warm_delegation_keys() -> Result<(), ProveError> {
    delegation_cached_keys().map(|_| ())
}

// ================================================================
// Prove
// ================================================================

/// Create a real Halo2 proof for the delegation circuit.
///
/// Returns the serialized proof bytes. Returns an error if the caller
/// provides a circuit without all witnesses populated or an instance
/// that Halo2 cannot prove against.
///
/// Params and keys are cached so only the first call pays keygen. Use the
/// delegation Criterion benchmark for release-mode proving measurements.
pub fn create_delegation_proof(
    circuit: Circuit,
    instance: &Instance,
) -> Result<Vec<u8>, ProveError> {
    let (params, pk, _vk) = delegation_cached_keys()?;

    let public_inputs = instance.to_halo2_instance();

    create_proof_bytes(params, pk, circuit, &public_inputs)
}

// ================================================================
// Verify
// ================================================================

/// Verify a delegation circuit proof given serialized proof bytes and
/// the 14 public inputs.
///
/// Returns `Ok(())` if verification succeeds, or an error message.
///
/// # Caller-authenticated inputs
///
/// `constrain_instance` pins each public input to whatever value the
/// *verifier* supplies; the protocol cannot tell whether that value was
/// the *right* one. The following fields of `instance` MUST be sourced
/// from their category-specific authority before calling this function.
/// Substituting them is not detectable from the proof alone.
///
/// ## Ledger-state anchors
///
/// These roots must be retrieved from the chain state at a snapshot height
/// that the verifier independently accepts. This crate's public-input vector
/// does not carry that height, so a chain ante handler or off-chain verifier
/// must validate the height-to-root lookup outside this API. Do not take these
/// values from the prover's bundle.
///
/// The proof does not encode a note version or pool identifier. Supplying an
/// Orchard root can therefore admit Orchard commitments; Ironwood-only callers
/// MUST supply the authenticated Ironwood root.
///
/// - `instance.nc_root` — the Ironwood note commitment tree root at the
///   verifier-pinned snapshot height.
/// - `instance.nf_imt_root` — the alternate-nullifier IMT root at the same
///   snapshot height as `nc_root`.
///
/// ## Session parameters
///
/// - `instance.vote_round_id` — must come from the same governance
///   announcement as the active voting session.
///
/// # Proof-attested outputs
///
/// The following public inputs are derived outside the circuit but
/// constrained in-circuit against authenticated inputs and private witnesses:
///
/// - `instance.van_comm`
/// - `instance.dom` — `Poseidon("governance authorization", vote_round_id)`.
///
/// The following fields are produced by the circuit from private
/// witnesses; successful verification is itself their authentication and
/// the caller does not need a separate trusted channel:
///
/// - `instance.nf_signed`
/// - `instance.rk_x` / `instance.rk_y`
/// - `instance.cmx_new`
/// - `instance.gov_null[..]`
pub fn verify_delegation_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    let (params, _pk, vk) = delegation_cached_keys().map_err(|error| error.to_string())?;

    let public_inputs = instance.to_halo2_instance();

    verify_proof_bytes("delegation", params, vk, proof, &public_inputs)
}

#[cfg(test)]
mod prove_tests {
    use super::*;
    use crate::delegation::builder::{build_delegation_bundle, RealNoteInput};
    use crate::delegation::imt::{ImtProvider, SpacedLeafImtProvider};
    use crate::ProveError;
    use ff::Field;
    use halo2_proofs::plonk;
    use incrementalmerkletree::{Hashable, Level};
    use orchard::{
        keys::{FullViewingKey, Scope, SpendValidatingKey, SpendingKey},
        note::{commitment::ExtractedNoteCommitment, nullifier::Nullifier, Note, NoteVersion, Rho},
        tree::{MerkleHashOrchard, MerklePath},
        value::NoteValue,
    };
    use pasta_curves::pallas;
    use rand::rngs::OsRng;

    fn minimal_instance() -> Instance {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let ak: SpendValidatingKey = fvk.into();
        let rk = ak.randomize(&pallas::Scalar::from(1));

        Instance::from_parts(
            Nullifier::from_inner(pallas::Base::from(1)),
            rk,
            pallas::Base::from(2),
            pallas::Base::from(3),
            pallas::Base::from(4),
            pallas::Base::from(5),
            pallas::Base::from(6),
            [pallas::Base::from(7); 5],
            pallas::Base::from(8),
        )
        .expect("test rk must be non-identity")
    }

    #[test]
    fn create_delegation_proof_signature_returns_result() {
        let _: fn(Circuit, &Instance) -> Result<Vec<u8>, ProveError> = create_delegation_proof;
    }

    #[test]
    #[ignore = "long-running K=12 proof keygen; run when touching delegation proof creation"]
    fn create_delegation_proof_returns_err_for_missing_witnesses() {
        let instance = minimal_instance();
        let err = create_delegation_proof(Circuit::default(), &instance).unwrap_err();

        assert!(matches!(err, ProveError::Halo2(plonk::Error::Synthesis)));
    }

    // TODO(sean): VK-stability tripwire. Hashes the `PinnedVerificationKey`
    // debug repr and compares against a baked-in fingerprint. A mismatch means
    // either the circuit shape changed (and the VK must be regenerated and
    // redistributed) or an unintended drift has been introduced.
    #[test]
    #[ignore = "TODO(sean): runs K=12 keygen; run with `cargo test -- --ignored vk_fingerprint_unchanged`"]
    fn vk_fingerprint_unchanged() {
        let (_, _, vk) = delegation_cached_keys().expect("delegation keys");
        let pinned = format!("{:?}", vk.pinned());
        let fingerprint = blake2b_simd::Params::new()
            .hash_length(32)
            .hash(pinned.as_bytes());
        let actual: &[u8] = fingerprint.as_bytes();

        let expected: [u8; 32] = [
            0xd2, 0x5f, 0xc9, 0xfa, 0x59, 0x07, 0xef, 0x37, 0x42, 0x9f, 0x6f, 0x81, 0x4d, 0x61,
            0xeb, 0x5c, 0x51, 0x45, 0xf9, 0xb3, 0x5c, 0x05, 0x9c, 0x83, 0x2c, 0x0f, 0x70, 0x16,
            0xd8, 0x49, 0x3c, 0xe6,
        ];

        assert_eq!(
            actual,
            expected.as_slice(),
            "delegation VK fingerprint changed; if intentional, update `expected` to:\n{:02x?}",
            actual,
        );
    }

    #[test]
    #[ignore = "long-running real proof roundtrip; run with `cargo test -- --ignored`"]
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
        let (_, _, dummy) = Note::dummy(&mut rng, None, NoteVersion::V3);
        let note = Note::new(
            recipient,
            NoteValue::from_raw(13_000_000),
            Rho::from_nf_old(dummy.nullifier(&fvk)),
            NoteVersion::V3,
            &mut rng,
        );
        let cmx = ExtractedNoteCommitment::from(note.commitment());
        let leaf = MerkleHashOrchard::from_cmx(&cmx);
        let empty = MerkleHashOrchard::empty_leaf();
        let mut leaves = [empty; 2];
        leaves[0] = leaf;
        let l1 = MerkleHashOrchard::combine(Level::from(0), &leaves[0], &leaves[1]);
        let mut current = l1;
        for level in 1..32u8 {
            current = MerkleHashOrchard::combine(
                Level::from(level),
                &current,
                &MerkleHashOrchard::empty_root(Level::from(level)),
            );
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
        let imt_proof = imt.non_membership_proof(real_nf.inner()).unwrap();

        let input = RealNoteInput {
            note,
            fvk: fvk.clone(),
            merkle_path,
            imt_proof,
            scope: Scope::External,
        };
        let bundle = build_delegation_bundle(
            vec![input],
            &fvk,
            alpha,
            output_recipient,
            vote_round_id,
            nc_root,
            van_comm_rand,
            &imt,
            &mut rng,
            None,
        )
        .unwrap();

        let proof = create_delegation_proof(bundle.circuit, &bundle.instance)
            .expect("delegation proof creation should succeed");
        verify_delegation_proof(&proof, &bundle.instance).expect("real proof roundtrip failed");
    }
}
