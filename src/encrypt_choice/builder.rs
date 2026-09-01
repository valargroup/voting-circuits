//! Encrypt-choice proof builder (ZKP 1.5).
//!
//! Constructs the decision-bound auxiliary proof for a weighted vote: all
//! `16 × 8` ElGamal bucket ciphertexts, the selected per-share commitments,
//! and the compact bridge that the cast proof (ZKP #2) later re-opens.
//!
//! Every secret — bucket randomness, blinds, the share decomposition and its
//! shuffle — is derived deterministically by `crate::vote_prf` from the
//! spending key and the `(round, proposal, VAN)` context, so re-running this
//! builder after a crash reproduces byte-identical witnesses, and the cast
//! builder can independently re-derive the same shares to cross-check the
//! bundle it consumes.

use std::{string::String, vec::Vec};

use crate::ff::PrimeField;
use crate::group::{Curve, GroupEncoding};
use voting_crypto_deps::halo2_proofs::circuit::Value;
use voting_crypto_deps::orchard::keys::SpendingKey;
use voting_crypto_deps::pasta_curves::{
    arithmetic::{Coordinates, CurveAffine},
    pallas,
};

use super::{
    circuit::{Circuit, Instance},
    prove::{create_encrypt_choice_proof, verify_encrypt_choice_proof},
};
use crate::{
    bridge::{
        bridge_commitment, selected_share_commitment, CiphertextCoordinates,
        WeightedShareCiphertexts, MAX_DECISION_BUCKETS, NUM_SHARES,
    },
    gadgets::elgamal::{base_to_scalar, spend_auth_g_affine},
    params::{BALLOT_DIVISOR, SHARE_VALUE_LIMIT},
    vote_prf::{
        denomination_split, derive_share_blind, derive_weighted_share_randomness,
        deterministic_shuffle,
    },
    ProveError,
};

/// Compressed wire encoding of one ElGamal ciphertext.
#[derive(Debug, Clone, Copy)]
pub struct ElGamalCiphertextBytes {
    /// Compressed C1 point (32 bytes).
    pub c1: [u8; 32],
    /// Compressed C2 point (32 bytes).
    pub c2: [u8; 32],
}

/// One weight share's complete encrypted bucket vector.
#[derive(Debug, Clone)]
pub struct EncryptedWeightedShareOutput {
    /// Share index (0-15).
    pub share_index: u32,
    /// Plaintext share value (encrypted into the decision bucket; all other
    /// buckets encrypt zero).
    pub plaintext_value: u64,
    /// Affine coordinates of all bucket ciphertexts, canonical bucket order.
    /// These are the exact values committed by the selected commitment.
    pub ciphertexts: WeightedShareCiphertexts,
    /// Compressed point encodings for wire payloads, canonical bucket order.
    pub compressed: [ElGamalCiphertextBytes; MAX_DECISION_BUCKETS],
    /// Per-bucket El Gamal randomness (32 bytes, LE pallas::Base repr),
    /// deterministically derived via the bucket-indexed vote PRF.
    pub randomness: [[u8; 32]; MAX_DECISION_BUCKETS],
}

/// Result of building an encrypt-choice proof.
///
/// Everything the cast (ZKP #2) and share-reveal (ZKP #3) builders need from
/// the auxiliary proof lives here; this bundle is the single source of truth
/// for the vote's ciphertexts, blinds, and commitments.
#[derive(Debug)]
pub struct EncryptChoiceBundle {
    /// Serialized Halo2 proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for the proof.
    pub instance: Instance,
    /// The compact bridge commitment (equals `instance.bridge`); the cast
    /// proof must expose the identical public value.
    pub bridge: pallas::Base,
    /// The 16 plaintext weight shares, in ballots.
    pub shares: [u64; NUM_SHARES],
    /// Per-share commitment blinds (PRF-derived; share-reveal needs the
    /// revealed share's blind).
    pub share_blinds: [pallas::Base; NUM_SHARES],
    /// Per-share selected commitments over the full bucket ciphertext
    /// vector; witnessed by ZKP #2 (bridge re-opening) and ZKP #3.
    pub selected_commitments: [pallas::Base; NUM_SHARES],
    /// The complete encrypted bucket vectors for all 16 shares.
    pub encrypted_shares: [EncryptedWeightedShareOutput; NUM_SHARES],
    /// Public active bucket count `D` for the proposal.
    pub decision_bucket_count: u64,
}

/// Errors that can occur during encrypt-choice proof construction.
#[derive(Debug)]
pub enum EncryptChoiceBuildError {
    /// The active bucket count is outside `[2, 8]`.
    InvalidBucketCount(u64),
    /// The private decision is not in `[0, decision_bucket_count)`.
    InvalidDecision {
        /// The rejected decision index.
        decision: u64,
        /// The active bucket count it must be below.
        decision_bucket_count: u64,
    },
    /// The total note value cannot be split into valid shares.
    InvalidShares(String),
    /// The election authority's public key is the identity point.
    InvalidElectionPublicKey,
    /// A derived El Gamal ciphertext point was the identity point.
    InvalidEncryptedShare(String),
    /// Halo2 proof creation failed.
    Prove(ProveError),
    /// A caller-supplied cached proof did not verify against the witness and
    /// instance reconstructed from the current vote context.
    InvalidCachedProof(String),
}

impl From<ProveError> for EncryptChoiceBuildError {
    fn from(error: ProveError) -> Self {
        EncryptChoiceBuildError::Prove(error)
    }
}

impl core::fmt::Display for EncryptChoiceBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EncryptChoiceBuildError::InvalidBucketCount(count) => {
                write!(
                    f,
                    "decision_bucket_count must be in [2, {}], got {}",
                    MAX_DECISION_BUCKETS, count
                )
            }
            EncryptChoiceBuildError::InvalidDecision {
                decision,
                decision_bucket_count,
            } => {
                write!(
                    f,
                    "decision {} is not in [0, {})",
                    decision, decision_bucket_count
                )
            }
            EncryptChoiceBuildError::InvalidShares(msg) => {
                write!(f, "invalid shares: {}", msg)
            }
            EncryptChoiceBuildError::InvalidElectionPublicKey => {
                write!(f, "invalid election public key: identity point")
            }
            EncryptChoiceBuildError::InvalidEncryptedShare(msg) => {
                write!(f, "invalid encrypted share: {}", msg)
            }
            EncryptChoiceBuildError::Prove(error) => {
                write!(f, "proof generation failed: {error}")
            }
            EncryptChoiceBuildError::InvalidCachedProof(error) => {
                write!(f, "cached encrypt-choice proof is invalid: {error}")
            }
        }
    }
}

impl std::error::Error for EncryptChoiceBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EncryptChoiceBuildError::Prove(error) => Some(error),
            _ => None,
        }
    }
}

fn pallas_coordinates(point: pallas::Affine) -> Option<Coordinates<pallas::Affine>> {
    point.coordinates().into()
}

/// Derives the 16 weight shares for a vote, in ballots.
///
/// This is the same derivation the cast builder performs; both proofs of a
/// bundle must witness identical shares.
pub(crate) fn derive_vote_shares(
    sk: &SpendingKey,
    num_ballots: u64,
    voting_round_id: pallas::Base,
    proposal_id: u64,
    vote_authority_note_old: pallas::Base,
    single_share: bool,
) -> Result<[u64; NUM_SHARES], String> {
    let shares: [u64; NUM_SHARES] = if single_share {
        // Last-moment mode: put entire weight in share[0], rest are zero.
        let mut s = [0u64; NUM_SHARES];
        s[0] = num_ballots;
        s
    } else {
        let mut s = denomination_split(
            num_ballots,
            sk,
            voting_round_id,
            proposal_id,
            vote_authority_note_old,
        );
        deterministic_shuffle(
            &mut s,
            sk,
            voting_round_id,
            proposal_id,
            vote_authority_note_old,
        );
        s
    };

    for (i, &s) in shares.iter().enumerate() {
        if s >= SHARE_VALUE_LIMIT {
            return Err(format!("share {} = {} exceeds 2^30", i, s));
        }
    }
    Ok(shares)
}

/// Fully assembled encrypt-choice witnesses and outputs, minus the proof.
///
/// Internal seam between witness construction and proving so MockProver
/// tests can exercise the exact builder-produced witnesses.
pub(crate) struct EncryptChoiceAssembly {
    pub(crate) circuit: Circuit,
    pub(crate) instance: Instance,
    pub(crate) bridge: pallas::Base,
    pub(crate) shares: [u64; NUM_SHARES],
    pub(crate) share_blinds: [pallas::Base; NUM_SHARES],
    pub(crate) selected_commitments: [pallas::Base; NUM_SHARES],
    pub(crate) encrypted_shares: [EncryptedWeightedShareOutput; NUM_SHARES],
}

impl EncryptChoiceAssembly {
    fn into_bundle(self, proof: Vec<u8>, decision_bucket_count: u64) -> EncryptChoiceBundle {
        EncryptChoiceBundle {
            proof,
            instance: self.instance,
            bridge: self.bridge,
            shares: self.shares,
            share_blinds: self.share_blinds,
            selected_commitments: self.selected_commitments,
            encrypted_shares: self.encrypted_shares,
            decision_bucket_count,
        }
    }
}

/// Build a real encrypt-choice proof (ZKP 1.5).
///
/// # Arguments
///
/// * `sk` - The SpendingKey used during delegation (ZKP #1).
/// * `total_note_value` - Sum of delegated note values in raw zatoshi.
///   Internally converted to ballot count via floor-division by
///   `BALLOT_DIVISOR`, exactly as the cast builder does.
/// * `vote_authority_note_old` - The VAN commitment this vote will consume,
///   from [`crate::vote_proof::derive_vote_authority_transition`]. Binds all
///   PRF derivations to the specific VAN; the cast proof must consume the
///   same VAN.
/// * `voting_round_id` - The active governance round identifier. The caller
///   must authenticate it from the round announcement.
/// * `proposal_id` - Which proposal to vote on. The caller must ensure it is
///   active for `voting_round_id`.
/// * `decision` - The voter's private choice, in `[0, decision_bucket_count)`.
///   Never leaves the device; only the encrypted one-hot vector is proven.
/// * `decision_bucket_count` - The proposal's public option count `D`, in
///   `[2, 8]`. The caller must authenticate it from the proposal
///   declaration.
/// * `ea_pk` - Election authority public key (Pallas affine point). The
///   caller must authenticate this against the active round's governance
///   announcement.
/// * `single_share` - Last-moment share layout: the entire weight in share 0.
///   Uses a distinct PRF randomness domain from the standard layout.
///
/// # Caller contract
///
/// All El Gamal randomness and blind factors are derived deterministically
/// from `sk` and the `(voting_round_id, proposal_id, vote_authority_note_old,
/// share, bucket)` context. Re-running this builder with the same inputs is
/// idempotent — including after a crash — and never reuses a nonce across
/// buckets, shares, layouts, VANs, rounds, or proposals.
///
/// **Expensive**: the wide K=11 proof generation should run in release mode
/// and is intended to start in the background as soon as the voter selects
/// their choice, ahead of the interactive cast step.
#[allow(clippy::too_many_arguments)]
pub fn build_encrypt_choice(
    sk: &SpendingKey,
    total_note_value: u64,
    vote_authority_note_old: pallas::Base,
    voting_round_id: pallas::Base,
    proposal_id: u64,
    decision: u64,
    decision_bucket_count: u64,
    ea_pk: pallas::Affine,
    single_share: bool,
) -> Result<EncryptChoiceBundle, EncryptChoiceBuildError> {
    let assembly = assemble_encrypt_choice(
        sk,
        total_note_value,
        vote_authority_note_old,
        voting_round_id,
        proposal_id,
        decision,
        decision_bucket_count,
        ea_pk,
        single_share,
    )?;

    // Clone rather than move: `into_bundle` consumes the rest of the assembly,
    // and a partial move would leave it unusable.
    let proof = create_encrypt_choice_proof(assembly.circuit.clone(), &assembly.instance)?;

    Ok(assembly.into_bundle(proof, decision_bucket_count))
}

/// Restores an encrypt-choice bundle from a cached proof.
///
/// Only the opaque proof comes from storage. Every witness, ciphertext,
/// commitment, bridge value, and public input is reconstructed from the
/// current spending key and vote context, and the proof is verified against
/// that reconstructed instance before the bundle is returned. A proof cached
/// for another choice, key, VAN, round, proposal, election-authority key,
/// bucket count, weight, or share layout is rejected before ZKP #2 can use it.
///
/// # Caller contract
///
/// `voting_round_id`, `proposal_id`, `decision_bucket_count`, and `ea_pk` must
/// come from the authenticated governance announcement. `total_note_value`
/// and `vote_authority_note_old` must come from the current durable delegation
/// generation and ordered VAN chain. `decision` is private wallet state.
/// `proof` is untrusted caller-shaped data; malformed or stale bytes return an
/// error rather than panicking.
#[allow(clippy::too_many_arguments)]
pub fn restore_encrypt_choice(
    proof: Vec<u8>,
    sk: &SpendingKey,
    total_note_value: u64,
    vote_authority_note_old: pallas::Base,
    voting_round_id: pallas::Base,
    proposal_id: u64,
    decision: u64,
    decision_bucket_count: u64,
    ea_pk: pallas::Affine,
    single_share: bool,
) -> Result<EncryptChoiceBundle, EncryptChoiceBuildError> {
    let assembly = assemble_encrypt_choice(
        sk,
        total_note_value,
        vote_authority_note_old,
        voting_round_id,
        proposal_id,
        decision,
        decision_bucket_count,
        ea_pk,
        single_share,
    )?;

    verify_encrypt_choice_proof(&proof, &assembly.instance)
        .map_err(EncryptChoiceBuildError::InvalidCachedProof)?;

    Ok(assembly.into_bundle(proof, decision_bucket_count))
}

/// Assembles the complete witness set and public inputs without proving.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_encrypt_choice(
    sk: &SpendingKey,
    total_note_value: u64,
    vote_authority_note_old: pallas::Base,
    voting_round_id: pallas::Base,
    proposal_id: u64,
    decision: u64,
    decision_bucket_count: u64,
    ea_pk: pallas::Affine,
    single_share: bool,
) -> Result<EncryptChoiceAssembly, EncryptChoiceBuildError> {
    if !(2..=MAX_DECISION_BUCKETS as u64).contains(&decision_bucket_count) {
        return Err(EncryptChoiceBuildError::InvalidBucketCount(
            decision_bucket_count,
        ));
    }
    if decision >= decision_bucket_count {
        return Err(EncryptChoiceBuildError::InvalidDecision {
            decision,
            decision_bucket_count,
        });
    }

    let ea_pk_coords =
        pallas_coordinates(ea_pk).ok_or(EncryptChoiceBuildError::InvalidElectionPublicKey)?;
    let ea_pk_x = *ea_pk_coords.x();
    let ea_pk_y = *ea_pk_coords.y();

    // ---- Shares (identical derivation to the cast builder) ----

    let num_ballots = total_note_value / BALLOT_DIVISOR;
    let shares = derive_vote_shares(
        sk,
        num_ballots,
        voting_round_id,
        proposal_id,
        vote_authority_note_old,
        single_share,
    )
    .map_err(EncryptChoiceBuildError::InvalidShares)?;

    // ---- Encrypt every (share, bucket) with independent PRF randomness ----

    let g = spend_auth_g_affine();
    let mut randomness = [[pallas::Base::zero(); NUM_SHARES]; MAX_DECISION_BUCKETS];
    let mut encrypted_shares: Vec<EncryptedWeightedShareOutput> = Vec::with_capacity(NUM_SHARES);

    for share in 0..NUM_SHARES {
        let mut coords = [CiphertextCoordinates {
            c1_x: pallas::Base::zero(),
            c2_x: pallas::Base::zero(),
            c1_y: pallas::Base::zero(),
            c2_y: pallas::Base::zero(),
        }; MAX_DECISION_BUCKETS];
        let mut compressed = [ElGamalCiphertextBytes {
            c1: [0u8; 32],
            c2: [0u8; 32],
        }; MAX_DECISION_BUCKETS];
        let mut randomness_bytes = [[0u8; 32]; MAX_DECISION_BUCKETS];

        for bucket in 0..MAX_DECISION_BUCKETS {
            let r = derive_weighted_share_randomness(
                sk,
                voting_round_id,
                proposal_id,
                vote_authority_note_old,
                share as u8,
                bucket as u8,
                single_share,
            );
            randomness[bucket][share] = r;
            randomness_bytes[bucket] = r.to_repr();
            let r_scalar = base_to_scalar(r)
                .expect("derive_weighted_share_randomness guarantees scalar-range");

            let plaintext = if bucket as u64 == decision {
                shares[share]
            } else {
                0
            };
            let v_scalar = base_to_scalar(pallas::Base::from(plaintext))
                .expect("30-bit share value is in scalar range");

            let c1_point = (g * r_scalar).to_affine();
            let c2_point = (g * v_scalar + ea_pk * r_scalar).to_affine();

            let c1_coords = pallas_coordinates(c1_point).ok_or_else(|| {
                EncryptChoiceBuildError::InvalidEncryptedShare(format!(
                    "share {share} bucket {bucket} c1 is identity"
                ))
            })?;
            let c2_coords = pallas_coordinates(c2_point).ok_or_else(|| {
                EncryptChoiceBuildError::InvalidEncryptedShare(format!(
                    "share {share} bucket {bucket} c2 is identity"
                ))
            })?;

            coords[bucket] = CiphertextCoordinates {
                c1_x: *c1_coords.x(),
                c2_x: *c2_coords.x(),
                c1_y: *c1_coords.y(),
                c2_y: *c2_coords.y(),
            };
            compressed[bucket] = ElGamalCiphertextBytes {
                c1: c1_point.to_bytes(),
                c2: c2_point.to_bytes(),
            };
        }

        encrypted_shares.push(EncryptedWeightedShareOutput {
            share_index: share as u32,
            plaintext_value: shares[share],
            ciphertexts: WeightedShareCiphertexts(coords),
            compressed,
            randomness: randomness_bytes,
        });
    }
    let encrypted_shares: [EncryptedWeightedShareOutput; NUM_SHARES] = encrypted_shares
        .try_into()
        .expect("sixteen encrypted shares");

    // ---- Blinds, selected commitments, bridge ----

    let share_blinds: [pallas::Base; NUM_SHARES] = core::array::from_fn(|i| {
        derive_share_blind(
            sk,
            voting_round_id,
            proposal_id,
            vote_authority_note_old,
            i as u8,
        )
    });
    let selected_commitments: [pallas::Base; NUM_SHARES] = core::array::from_fn(|i| {
        selected_share_commitment(share_blinds[i], &encrypted_shares[i].ciphertexts)
    });

    let proposal_id_base = pallas::Base::from(proposal_id);
    let decision_bucket_count_base = pallas::Base::from(decision_bucket_count);
    let weights_and_comms: [(pallas::Base, pallas::Base); NUM_SHARES] =
        core::array::from_fn(|i| (pallas::Base::from(shares[i]), selected_commitments[i]));
    let bridge = bridge_commitment(
        voting_round_id,
        proposal_id_base,
        decision_bucket_count_base,
        &weights_and_comms,
    );

    // ---- Build circuit witnesses ----

    let circuit = Circuit {
        shares: core::array::from_fn(|i| Value::known(pallas::Base::from(shares[i]))),
        blinds: share_blinds.map(Value::known),
        randomness: randomness.map(|per_bucket| per_bucket.map(Value::known)),
        selectors: core::array::from_fn(|bucket| {
            Value::known(if bucket as u64 == decision {
                pallas::Base::one()
            } else {
                pallas::Base::zero()
            })
        }),
        active: core::array::from_fn(|bucket| {
            Value::known(if (bucket as u64) < decision_bucket_count {
                pallas::Base::one()
            } else {
                pallas::Base::zero()
            })
        }),
        ea_pk: Value::known(ea_pk),
    };

    let instance = Instance::from_parts(
        ea_pk_x,
        ea_pk_y,
        bridge,
        decision_bucket_count_base,
        voting_round_id,
        proposal_id_base,
    );

    Ok(EncryptChoiceAssembly {
        circuit,
        instance,
        bridge,
        shares,
        share_blinds,
        selected_commitments,
        encrypted_shares,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sk() -> SpendingKey {
        SpendingKey::from_bytes([0x42; 32]).expect("valid spending key")
    }

    #[test]
    fn rejects_bucket_count_out_of_range() {
        for bad in [0u64, 1, MAX_DECISION_BUCKETS as u64 + 1] {
            let err = build_encrypt_choice(
                &test_sk(),
                12_500_000,
                pallas::Base::from(1u64),
                pallas::Base::from(2u64),
                3,
                0,
                bad,
                spend_auth_g_affine(),
                false,
            )
            .unwrap_err();
            assert!(matches!(
                err,
                EncryptChoiceBuildError::InvalidBucketCount(count) if count == bad
            ));
        }
    }

    #[test]
    fn rejects_decision_outside_active_buckets() {
        let err = build_encrypt_choice(
            &test_sk(),
            12_500_000,
            pallas::Base::from(1u64),
            pallas::Base::from(2u64),
            3,
            5,
            5,
            spend_auth_g_affine(),
            false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EncryptChoiceBuildError::InvalidDecision {
                decision: 5,
                decision_bucket_count: 5,
            }
        ));
    }

    #[test]
    #[ignore = "expensive wide K=11 keygen and proof; run with --release -- --ignored when touching encrypt-choice proving"]
    fn real_proof_roundtrip() {
        use crate::encrypt_choice::verify_encrypt_choice_proof;

        let bundle = build_encrypt_choice(
            &test_sk(),
            12_500_000,
            pallas::Base::from(0xDEAD_u64),
            pallas::Base::from(0xCAFE_u64),
            3,
            1,
            4,
            spend_auth_g_affine(),
            false,
        )
        .expect("encrypt-choice builder should produce a valid proof");

        verify_encrypt_choice_proof(&bundle.proof, &bundle.instance)
            .expect("typed verifier should accept the builder's proof");
        std::println!(
            "encrypt-choice proof size: {} bytes ({:.1} KiB)",
            bundle.proof.len(),
            bundle.proof.len() as f64 / 1024.0
        );

        // Determinism: rebuilding produces the identical bundle values.
        let again = build_encrypt_choice(
            &test_sk(),
            12_500_000,
            pallas::Base::from(0xDEAD_u64),
            pallas::Base::from(0xCAFE_u64),
            3,
            1,
            4,
            spend_auth_g_affine(),
            false,
        )
        .expect("rebuild succeeds");
        assert_eq!(bundle.bridge, again.bridge);
        assert_eq!(bundle.shares, again.shares);
        assert_eq!(bundle.selected_commitments, again.selected_commitments);
    }

    #[test]
    fn derive_vote_shares_is_deterministic_and_sums() {
        let sk = test_sk();
        let round = pallas::Base::from(7u64);
        let van = pallas::Base::from(11u64);

        let a = derive_vote_shares(&sk, 12_345, round, 3, van, false).expect("valid shares");
        let b = derive_vote_shares(&sk, 12_345, round, 3, van, false).expect("valid shares");
        assert_eq!(a, b);
        assert_eq!(a.iter().sum::<u64>(), 12_345);

        let single = derive_vote_shares(&sk, 12_345, round, 3, van, true).expect("valid shares");
        assert_eq!(single[0], 12_345);
        assert!(single[1..].iter().all(|&s| s == 0));
    }
}
