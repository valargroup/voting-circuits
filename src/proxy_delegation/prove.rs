//! Real Halo2 prove/verify for the proxy delegation circuit (ZKP #4).
//!
//! Follows the same pattern as `vote_proof/prove.rs`, for the 14-condition
//! proxy delegation circuit at K=11.

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

/// Params and proving and verifying keys for the proxy delegation circuit.
pub type ProxyDelegationKeys = (
    Params<EqAffine>,
    plonk::ProvingKey<EqAffine>,
    plonk::VerifyingKey<EqAffine>,
);

static PROXY_DELEGATION_KEYS_CACHE: std::sync::OnceLock<Result<ProxyDelegationKeys, String>> =
    std::sync::OnceLock::new();

/// Return cached params and proving/verifying keys for the proxy delegation
/// circuit.
///
/// Params generation and key generation are deterministic and expensive enough
/// to dominate the first proof or verification call. Compute the full tuple
/// once per process so warm-up covers both the SRS params and the keys. A
/// prover that must free them between phases uses [`ProxyDelegationProver`].
pub fn proxy_delegation_cached_keys() -> Result<&'static ProxyDelegationKeys, ProveError> {
    match PROXY_DELEGATION_KEYS_CACHE.get_or_init(|| {
        let params = proxy_delegation_params();
        proxy_delegation_proving_key(&params)
            .map(|(pk, vk)| (params, pk, vk))
            .map_err(|error| error.to_string())
    }) {
        Ok(keys) => Ok(keys),
        Err(error) => Err(ProveError::CachedKeygen(error.clone())),
    }
}

/// Warm the process-lifetime proxy delegation params/proving-key cache.
///
/// This lets callers pay deterministic keygen before the first user-visible
/// proof generation or verification path needs the params and keys. It also
/// arms the prepared commitment tables through
/// [`prepare_proxy_delegation_proving`]: a long-lived prover can use those
/// tables on supported pool widths. Verifiers can use the cached-key accessor
/// directly to avoid preparation costs.
pub fn warm_proxy_delegation_keys() -> Result<(), ProveError> {
    prepare_proxy_delegation_proving().map(|_| ())
}

/// Arm the prepared commitment tables over this circuit's SRS.
///
/// On Zakura, `Params::prepare_commitments` builds
/// fixed-base tables that the prover's polynomial commitments then evaluate
/// through. Key generation does not build them: it precomputes the
/// polynomial-side data (fixed cosets, FFT twiddles, quotient plans) and
/// stops there.
///
/// Call this once from a long-lived prover, off the proving critical path —
/// [`warm_proxy_delegation_keys`] already does, so callers that warm need not
/// call it separately. Preparation is idempotent and single-flighted;
/// concurrent callers share one attempt. Do not fan it out across a worker
/// pool, which would occupy the workers its own internal parallelism wants.
///
/// # Cost
///
/// Preparation retains fixed-base tables that scale with `2^K`. Measured for
/// the `K = 12` delegation circuit (`benches/delegation.rs`): 26,426.91 KiB,
/// against a proving key that is already 154,706 KiB. This circuit is
/// `K = 11`.
///
/// # When it pays off
///
/// halo2 routes through the prepared tables only on pools of at most eight
/// effective threads (ten for `K = 11` on AArch64 macOS); past that the
/// unprepared planner out-scales them and halo2 falls back on its own. So
/// proving gains depend on the circuit and host. Warming still pays table
/// construction time and retains memory on wide pools and verifier-only
/// callers, even when the large tables are not used.
///
/// # Returns
///
/// Whether the tables were actually built and cached. `false` means arming
/// was a no-op (no prepared backend, or the backend declined) and proving
/// simply keeps its unprepared path; it is not an error, and unlike a key
/// generation failure it is never memoized as one. With the LRZ backend,
/// this warms the keys and returns `Ok(false)` without building tables.
pub fn prepare_proxy_delegation_proving() -> Result<bool, ProveError> {
    proxy_delegation_cached_keys()
        .map(|(params, _pk, _vk)| crate::prove_error::prepare_commitments(params))
}

// ================================================================
// Caller-owned keys
// ================================================================

/// Proxy delegation params and keys owned by the caller.
///
/// The keys behind [`proxy_delegation_cached_keys`] live for the rest of the
/// process. A phased prover instead loads this handle for its ZKP #4 phase
/// and drops it before the next phase, which frees the proving key and any
/// prepared tables. Key generation is deterministic, so these are the same
/// keys, and a proof from either verifies under both.
pub struct ProxyDelegationProver {
    keys: ProxyDelegationKeys,
}

impl core::fmt::Debug for ProxyDelegationProver {
    // Keys are large, so they are left out.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ProxyDelegationProver")
            .finish_non_exhaustive()
    }
}

impl ProxyDelegationProver {
    /// Generates the params and keys, as the first cached-key call does,
    /// without touching that cache.
    pub fn load() -> Result<Self, ProveError> {
        let params = proxy_delegation_params();
        let (pk, vk) = proxy_delegation_proving_key(&params)?;
        Ok(ProxyDelegationProver {
            keys: (params, pk, vk),
        })
    }

    /// Arms the prepared commitment tables on this handle's params and
    /// returns whether they were built, with the costs and semantics of
    /// [`prepare_proxy_delegation_proving`]. The tables are freed with the
    /// handle.
    pub fn prepare(&self) -> bool {
        crate::prove_error::prepare_commitments(&self.keys.0)
    }

    /// Proves as [`create_proxy_delegation_proof`] does, with this handle's
    /// keys.
    pub fn create_proof(
        &self,
        circuit: Circuit,
        instance: &Instance,
    ) -> Result<Vec<u8>, ProveError> {
        let (params, pk, _vk) = &self.keys;
        create_proof_bytes(params, pk, circuit, &instance.to_halo2_instance())
    }

    /// Verifies as [`verify_proxy_delegation_proof`] does, with this handle's
    /// keys. Its caller contract applies unchanged.
    pub fn verify_proof(&self, proof: &[u8], instance: &Instance) -> Result<(), String> {
        let (params, _pk, vk) = &self.keys;
        verify_proof_bytes(
            "proxy delegation",
            params,
            vk,
            proof,
            &instance.to_halo2_instance(),
        )
    }

    /// The params and keys this handle owns.
    pub fn keys(&self) -> &ProxyDelegationKeys {
        &self.keys
    }
}

// ================================================================
// Params / key generation (public API, non-cached fallbacks)
// ================================================================

/// Generate the IPA params (SRS) for the proxy delegation circuit.
/// Deterministic for a given `K`.
pub fn proxy_delegation_params() -> Params<EqAffine> {
    Params::new(K)
}

/// Generate the proving and verifying keys for the proxy delegation circuit.
///
/// Uses `Circuit::default()` (all witnesses unknown) as the empty circuit
/// for key generation — the same pattern as the Orchard action circuit.
pub fn proxy_delegation_proving_key(
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

/// Create a real Halo2 proof for the proxy delegation circuit.
///
/// Returns the serialized proof bytes. Returns an error if the caller
/// provides a circuit without all witnesses populated or an instance
/// that Halo2 cannot prove against.
///
/// **Expensive**: K=11 proof generation should run in release mode. Params and
/// keys are cached so only the first call pays keygen.
pub fn create_proxy_delegation_proof(
    circuit: Circuit,
    instance: &Instance,
) -> Result<Vec<u8>, ProveError> {
    let (params, pk, _vk) = proxy_delegation_cached_keys()?;

    let public_inputs = instance.to_halo2_instance();

    create_proof_bytes(params, pk, circuit, &public_inputs)
}

// ================================================================
// Verify
// ================================================================

/// Verify a proxy delegation proof given serialized proof bytes and the typed
/// public inputs.
///
/// Returns `Ok(())` if verification succeeds, or an error message.
///
/// # Caller-authenticated inputs
///
/// Every public input is bound into the proof transcript, but the proof cannot
/// tell the verifier whether caller-provided chain or governance values are
/// the right ones. The following fields of `instance` MUST come from their
/// authority before calling this function:
///
/// - `instance.vote_comm_tree_root` - the root the chain accepts for this
///   spend: the vote commitment tree root at `vote_comm_tree_anchor_height`,
///   or the single-leaf root of a VAN that the same submission created.
/// - `instance.vote_comm_tree_anchor_height` - a height the chain's
///   anchor-validity check accepts. No condition constrains it.
/// - `instance.voting_round_id` - the active voting round.
/// - `instance.ea_pk_x`, `instance.ea_pk_y` - the election authority's
///   published key for `voting_round_id`. A substituted key lets its holder
///   decrypt the delegated shares, or leaves them undecryptable.
///
/// # Proof-attested outputs
///
/// Successful verification authenticates these, though the chain must still
/// check that both nullifiers are unspent:
///
/// - `instance.van_nullifier`
/// - `instance.r_vpk_x`, `instance.r_vpk_y`
/// - `instance.vote_authority_note_new`
/// - `instance.delegation_commitment`
/// - `instance.dc_slot_nullifier`
///
/// The delegate index inside the delegation commitment is private and
/// unbounded; the chain checks it when a share is revealed.
pub fn verify_proxy_delegation_proof(proof: &[u8], instance: &Instance) -> Result<(), String> {
    let (params, _pk, vk) = proxy_delegation_cached_keys().map_err(|error| error.to_string())?;

    let public_inputs = instance.to_halo2_instance();

    verify_proof_bytes("proxy delegation", params, vk, proof, &public_inputs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gadgets::elgamal::spend_auth_g_affine;
    use crate::group::Curve;
    use crate::params::VOTE_COMM_TREE_DEPTH;
    use crate::proxy_delegation::{
        build_proxy_delegation_proof, ProxyDelegationBundle, ShareLayout,
    };
    use voting_crypto_deps::halo2_proofs::plonk;
    use voting_crypto_deps::orchard::keys::SpendingKey;
    use voting_crypto_deps::pasta_curves::pallas;

    /// vote-sdk's consensus limit on a proof's size (`MaxProofSize`).
    const DOWNSTREAM_MAX_PROOF_SIZE: usize = 15 * 1_024;

    /// A bundle that delegates 4,800 of a VAN's 10,000 ballots in `layout`.
    fn valid_bundle(layout: ShareLayout) -> ProxyDelegationBundle {
        let sk = SpendingKey::from_bytes([0x42; 32]).expect("valid test spending key");
        build_proxy_delegation_proof(
            &sk,
            1,
            10_000,
            pallas::Base::from(0xDEAD_u64),
            pallas::Base::from(0xCAFE_u64),
            [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH],
            0,
            123,
            (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine(),
            pallas::Scalar::from(7u64),
            17,
            4_800,
            3,
            [0x5e; 32],
            layout,
        )
        .expect("valid proxy delegation inputs")
    }

    /// Proves `layout`'s bundle and checks the proof's size, that it verifies,
    /// and that changing any one public input breaks it.
    fn assert_real_proof_roundtrip(layout: ShareLayout) {
        let ProxyDelegationBundle {
            circuit, instance, ..
        } = valid_bundle(layout);
        let proof = create_proxy_delegation_proof(circuit, &instance)
            .expect("proxy delegation proof creation should succeed");
        assert!(
            proof.len() <= DOWNSTREAM_MAX_PROOF_SIZE,
            "proxy delegation proof is {} bytes, exceeding the downstream {}-byte limit",
            proof.len(),
            DOWNSTREAM_MAX_PROOF_SIZE,
        );

        verify_proxy_delegation_proof(&proof, &instance)
            .expect("proxy delegation verifier should accept the builder's proof");
        let (params, _, vk) = proxy_delegation_cached_keys().unwrap();
        verify_proof_bytes(
            "proxy delegation",
            &proxy_delegation_params(),
            vk,
            &proof,
            &instance.to_halo2_instance(),
        )
        .expect("fresh verifier params");

        let mut wrong_dc = instance.clone();
        wrong_dc.delegation_commitment += pallas::Base::one();
        assert!(verify_proxy_delegation_proof(&proof, &wrong_dc).is_err());
        for offset in 0..Instance::NUM_PUBLIC_INPUTS {
            let mut public_inputs = instance.to_halo2_instance();
            public_inputs[offset] += pallas::Base::one();
            assert!(
                verify_proof_bytes("proxy delegation", params, vk, &proof, &public_inputs).is_err(),
                "a changed public input {offset} must fail verification"
            );
        }
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
    fn create_proxy_delegation_proof_signature_returns_result() {
        let _: fn(Circuit, &Instance) -> Result<Vec<u8>, ProveError> =
            create_proxy_delegation_proof;
    }

    #[test]
    #[ignore = "long-running K=11 proof keygen; run when touching proxy delegation proof creation"]
    fn create_proxy_delegation_proof_returns_err_for_missing_witnesses() {
        let err =
            create_proxy_delegation_proof(Circuit::default(), &minimal_instance()).unwrap_err();

        assert!(matches!(err, ProveError::Halo2(plonk::Error::Synthesis)));
    }

    /// The pinned fingerprint of the proxy delegation VK.
    const EXPECTED_VK_FINGERPRINT: [u8; 32] = [
        0x4e, 0xc7, 0xe2, 0x4b, 0x9b, 0xbf, 0x27, 0x58, 0x07, 0xa6, 0x13, 0xd3, 0xd3, 0xea, 0x94,
        0xcf, 0xae, 0x21, 0x50, 0x83, 0xb7, 0xaa, 0x91, 0x37, 0x4b, 0x79, 0xde, 0x52, 0xbf, 0x7d,
        0x35, 0xaa,
    ];

    /// Hashes the `PinnedVerificationKey` debug repr of `vk`.
    fn vk_fingerprint(vk: &plonk::VerifyingKey<EqAffine>) -> Vec<u8> {
        let pinned = format!("{:?}", vk.pinned());
        blake2b_simd::Params::new()
            .hash_length(32)
            .hash(pinned.as_bytes())
            .as_bytes()
            .to_vec()
    }

    /// VK-stability tripwire: hashes the `PinnedVerificationKey` debug repr. A
    /// mismatch means the circuit shape changed, so the verifying key must be
    /// regenerated and redistributed, or that it drifted unintentionally.
    #[test]
    #[ignore = "runs K=11 keygen; run with `cargo test -- --ignored vk_fingerprint_unchanged`"]
    fn vk_fingerprint_unchanged() {
        let (_, _, vk) = proxy_delegation_cached_keys().expect("proxy delegation keys");
        let actual = vk_fingerprint(vk);

        assert_eq!(
            actual,
            EXPECTED_VK_FINGERPRINT.as_slice(),
            "proxy delegation VK fingerprint changed; if intentional, update `EXPECTED_VK_FINGERPRINT` to:\n{:02x?}",
            actual,
        );
    }

    /// The caller-owned prover loads the pinned VK, and its proofs and the
    /// cached keys' proofs verify under both. Dropping it and loading again
    /// gives a working prover.
    #[test]
    #[ignore = "long-running real proof roundtrip; run with `cargo test -- --ignored owned_prover`"]
    fn owned_prover_matches_the_cached_keys() {
        fn is_send_sync<T: Send + Sync>() {}
        is_send_sync::<ProxyDelegationProver>();

        let ProxyDelegationBundle {
            circuit, instance, ..
        } = valid_bundle(ShareLayout::Standard);
        let prover = ProxyDelegationProver::load().expect("load the owned keys");
        assert_eq!(vk_fingerprint(&prover.keys().2), EXPECTED_VK_FINGERPRINT);
        assert_eq!(prover.prepare(), !cfg!(feature = "lrz"));
        let owned = prover
            .create_proof(circuit.clone(), &instance)
            .expect("owned proof");
        prover
            .verify_proof(&owned, &instance)
            .expect("owned verifier");
        verify_proxy_delegation_proof(&owned, &instance).expect("cached verifier");
        drop(prover);

        let cached = create_proxy_delegation_proof(circuit, &instance).expect("cached proof");
        let prover = ProxyDelegationProver::load().expect("reload the owned keys");
        prover
            .verify_proof(&cached, &instance)
            .expect("owned verifier, cached proof");
        let mut wrong_dc = instance.clone();
        wrong_dc.delegation_commitment += pallas::Base::one();
        assert!(prover.verify_proof(&owned, &wrong_dc).is_err());
    }

    #[test]
    #[ignore = "long-running real proof roundtrip; run with `cargo test -- --ignored real_proof_roundtrip`"]
    fn real_proof_roundtrip() {
        assert_real_proof_roundtrip(ShareLayout::Standard);
    }

    #[test]
    #[ignore = "long-running real proof roundtrip; run with `cargo test -- --ignored real_proof_roundtrip`"]
    fn single_share_real_proof_roundtrip() {
        assert_real_proof_roundtrip(ShareLayout::SingleShare);
    }

    #[test]
    fn prepare_proxy_delegation_proving_signature_returns_result() {
        let _: fn() -> Result<bool, ProveError> = prepare_proxy_delegation_proving;
    }

    // Exercise the warm entry point first, then repeat preparation and prove
    // within the same narrow pool. LRZ must retain ordinary proving behavior.
    #[test]
    #[ignore = "real proof with prepared commitments; also exercised in CI"]
    fn prepared_proving_roundtrip() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(6)
            .build()
            .unwrap()
            .install(|| {
                warm_proxy_delegation_keys().expect("warm keys");
                let first = prepare_proxy_delegation_proving().expect("prepare keys");
                assert_eq!(first, !cfg!(feature = "lrz"));
                assert_eq!(
                    prepare_proxy_delegation_proving().expect("repeat preparation"),
                    first
                );
                real_proof_roundtrip();
            });
    }
}
