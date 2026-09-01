//! Deterministic vote-secret derivation and share decomposition.
//!
//! This module owns the Blake2b-512 PRF that derives every per-vote secret —
//! El Gamal randomness, share commitment blinds, remainder weights, and the
//! shuffle seed — plus the denomination-based share decomposition that both
//! the encrypt-choice (ZKP 1.5) and vote-proof (ZKP #2) builders must agree
//! on. Keeping one implementation guarantees that both proofs of a vote
//! bundle derive identical shares and secrets for the same
//! `(sk, round, proposal, VAN)` inputs, and that a crashed client can
//! re-derive everything without persisting secrets.

use std::vec::Vec;

use crate::ff::{Field, FromUniformBytes, PrimeField};
use voting_crypto_deps::orchard::keys::SpendingKey;
use voting_crypto_deps::pasta_curves::pallas;

use crate::{domain_tags, gadgets::elgamal::base_to_scalar};

/// Number of shares per vote.
pub(crate) const NUM_SHARES: usize = 16;

/// Standard denomination values for share decomposition (ballots, descending).
///
/// | Ballots    | ZEC         |
/// |------------|-------------|
/// | 10,000,000 | 1,250,000   |
/// | 1,000,000  | 125,000     |
/// | 100,000    | 12,500      |
/// | 10,000     | 1,250       |
/// | 1,000      | 125         |
/// | 100        | 12.5        |
/// | 10         | 1.25        |
/// | 1          | 0.125       |
const DENOMINATIONS: [u64; 8] = [10_000_000, 1_000_000, 100_000, 10_000, 1_000, 100, 10, 1];

/// Maximum slots used for standard denomination shares.
///
/// The remaining `NUM_SHARES - MAX_DENOM_SHARES` slots (7) are reserved for
/// random-valued shares produced by [`distribute_remainder`].  This ensures
/// every voter's share array contains a mix of standard denominations and
/// non-standard values, preventing the EA from reconstructing exact balances
/// by matching denomination patterns.
const MAX_DENOM_SHARES: usize = 9;

// The remainder slots must have enough room for meaningful PRF-weighted
// spreading. This fires at compile time if someone changes the constants.
const _: () = assert!(
    NUM_SHARES - MAX_DENOM_SHARES >= 7,
    "need at least 7 remainder slots for PRF-weighted distribution"
);

/// Decompose `num_ballots` into [`NUM_SHARES`] shares using a greedy
/// denomination strategy with randomized remainder distribution.
///
/// 1. **Greedy fill**: place the largest standard denominations that fit,
///    consuming up to [`MAX_DENOM_SHARES`] slots.
/// 2. **Remainder split**: if a non-zero remainder exists, distribute it
///    across all free slots using deterministic PRF-derived weights.
/// 3. The caller then shuffles the result via [`deterministic_shuffle`].
///
/// The randomized remainder prevents a single non-standard value from
/// fingerprinting the voter's exact balance.
pub(crate) fn denomination_split(
    num_ballots: u64,
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
) -> [u64; NUM_SHARES] {
    let mut shares = [0u64; NUM_SHARES];
    let mut remaining = num_ballots;
    let mut idx = 0;

    // Phase 1: Greedy fill — place the largest standard denominations that
    // fit, consuming up to MAX_DENOM_SHARES (9) slots. These "tier" values
    // are shared across many voters, forming the per-share anonymity set.
    for &d in &DENOMINATIONS {
        while remaining >= d && idx < MAX_DENOM_SHARES {
            shares[idx] = d;
            remaining -= d;
            idx += 1;
        }
    }

    // Phase 2: Remainder distribution — spread any leftover across the free
    // slots (at least 7, enforced by the const assert above) using
    // PRF-derived weights so no single non-standard value fingerprints the
    // exact balance.
    if remaining > 0 {
        distribute_remainder(
            &mut shares[idx..],
            remaining,
            sk,
            round_id,
            proposal_id,
            van_commitment,
            idx as u8,
        );
    }

    shares
}

/// Spread `remainder` across `slots` using PRF-derived weights.
///
/// Each slot gets `floor(remainder * weight_i / total_weight)` with any
/// rounding residual added one-per-slot to the first slots. Every slot
/// receives at least 1 to maximize dispersion across all available slots.
fn distribute_remainder(
    slots: &mut [u64],
    remainder: u64,
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
    base_index: u8,
) {
    let n = slots.len() as u64;
    // The greedy phase fills at most MAX_DENOM_SHARES (9) slots, so
    // n >= NUM_SHARES - MAX_DENOM_SHARES >= 7 (enforced by const assert).

    // Edge case: if the remainder is smaller than the number of slots, we
    // can't put at least 1 in every slot. Just give 1 ballot to as many
    // slots as we can and leave the rest at zero.
    // Example: remainder=3, n=7 → slots = [1, 1, 1, 0, 0, 0, 0]
    if remainder < n {
        for i in 0..(remainder as usize) {
            slots[i] = 1;
        }
        return;
    }

    // Ensure every slot gets at least 1 ballot so all 7 slots carry part
    // of the remainder. This maximizes dispersion — concentrating the
    // remainder in fewer slots would make each value larger and more
    // informative if decrypted individually.
    // Example: remainder=300, n=7 → distributable=293
    let distributable = remainder - n;

    // Derive a PRF weight per slot. Each weight is a 32-bit pseudorandom
    // value from BLAKE2b, unique per (voter, VAN, proposal, slot index).
    // The `| 1` ensures no weight is zero (avoids a slot getting nothing).
    let mut weights = Vec::with_capacity(slots.len());
    let mut total_weight: u64 = 0;
    for i in 0..slots.len() {
        let hash = vote_share_prf(
            sk,
            domain_tags::VOTE_PRF_DOMAIN_REMAINDER,
            round_id,
            proposal_id,
            van_commitment,
            base_index.wrapping_add(i as u8),
        );
        let w = u32::from_le_bytes(hash[0..4].try_into().unwrap()) as u64 | 1;
        weights.push(w);
        total_weight += w;
    }

    // Give each slot its reserved 1 ballot plus a weighted share of the
    // distributable portion: floor(distributable * weight_i / total_weight).
    // Integer division truncates, so we track how much was actually assigned.
    let mut assigned: u64 = 0;
    for i in 0..slots.len() {
        let share = ((distributable as u128 * weights[i] as u128) / total_weight as u128) as u64;
        slots[i] = 1 + share;
        assigned += share;
    }

    // The floor divisions above may leave a small leftover (at most n-1
    // ballots). Distribute it one-per-slot to the first slots. This is
    // deterministic — same PRF weights → same leftover → same correction.
    let leftover = distributable - assigned;
    for i in 0..(leftover as usize) {
        slots[i] += 1;
    }
}

/// Core PRF: BLAKE2b-512 bound to the spending key with voting-specific
/// personalization and domain-separated inputs.
///
/// `PRF(sk, domain, round_id, proposal_id, van_commitment, share_index)`
///   = BLAKE2b-512("ZcashVote_Expand", sk || domain || round_id || proposal_id_le64 || van_commitment || share_index_u8)
///
/// The `van_commitment` field binds the derivation to a specific VAN.
/// Without it, a user with multiple VANs (from >5 notes in Phase 1)
/// voting on the same proposal would derive identical El Gamal nonces,
/// enabling a classic nonce-reuse attack on the ciphertexts.
pub(crate) fn vote_share_prf(
    sk: &SpendingKey,
    domain: u8,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
    share_index: u8,
) -> [u8; 64] {
    *blake2b_simd::Params::new()
        .hash_length(64)
        .personal(domain_tags::VOTE_PRF_PERSONALIZATION)
        .to_state()
        .update(sk.to_bytes())
        .update(&[domain])
        .update(&round_id.to_repr())
        .update(&proposal_id.to_le_bytes())
        .update(&van_commitment.to_repr())
        .update(&[share_index])
        .finalize()
        .as_array()
}

/// Bucket-indexed PRF variant for weighted encrypt-choice derivations.
///
/// `PRF(sk, domain, round_id, proposal_id, van_commitment, share_index, bucket_index)`
///   = BLAKE2b-512("ZcashVote_Expand", sk || domain || round_id || proposal_id_le64
///                 || van_commitment || share_index_u8 || bucket_index_u8)
///
/// The preimage is the [`vote_share_prf`] preimage with the bucket byte
/// appended, and MUST only be used with domains reserved for bucket-indexed
/// streams (`VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED*`). Domain separation, not the
/// preimage length, is what keeps this stream independent from the
/// share-indexed one.
pub(crate) fn vote_share_bucket_prf(
    sk: &SpendingKey,
    domain: u8,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
    share_index: u8,
    bucket_index: u8,
) -> [u8; 64] {
    *blake2b_simd::Params::new()
        .hash_length(64)
        .personal(domain_tags::VOTE_PRF_PERSONALIZATION)
        .to_state()
        .update(sk.to_bytes())
        .update(&[domain])
        .update(&round_id.to_repr())
        .update(&proposal_id.to_le_bytes())
        .update(&van_commitment.to_repr())
        .update(&[share_index])
        .update(&[bucket_index])
        .finalize()
        .as_array()
}

/// Derive deterministic El Gamal randomness `r_{i,j}` for one
/// `(share, bucket)` ciphertext of a weighted encrypt-choice vote.
///
/// Every `(share, bucket)` pair gets independent randomness — zero-plaintext
/// buckets included — so decrypting one ciphertext reveals nothing about the
/// others. Standard and single-share layouts use distinct PRF domains because
/// they encrypt different plaintexts for the same indices.
///
/// Returns a non-zero `pallas::Base` element that is also a valid
/// `pallas::Scalar` (p_base < q_scalar on Pallas).
pub(crate) fn derive_weighted_share_randomness(
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
    share_index: u8,
    bucket_index: u8,
    single_share: bool,
) -> pallas::Base {
    let domain = if single_share {
        domain_tags::VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED_SINGLE_SHARE
    } else {
        domain_tags::VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED
    };
    let hash = vote_share_bucket_prf(
        sk,
        domain,
        round_id,
        proposal_id,
        van_commitment,
        share_index,
        bucket_index,
    );
    nonzero_base_from_prf(hash)
}

/// Reduce 64 PRF bytes to a non-zero base-field element usable as El Gamal
/// randomness.
fn nonzero_base_from_prf(hash: [u8; 64]) -> pallas::Base {
    let r = pallas::Base::from_uniform_bytes(&hash);
    if bool::from(r.is_zero()) {
        // Preserve deterministic derivation while satisfying the circuit's
        // non-zero randomness gate in the negligible exact-zero case.
        return pallas::Base::one();
    }
    debug_assert!(base_to_scalar(r).is_some(), "p < q guarantees Base→Scalar");
    r
}

/// Derive deterministic blind factor `blind_i` for a share commitment.
pub(crate) fn derive_share_blind(
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
    share_index: u8,
) -> pallas::Base {
    let hash = vote_share_prf(
        sk,
        domain_tags::VOTE_PRF_DOMAIN_BLIND,
        round_id,
        proposal_id,
        van_commitment,
        share_index,
    );
    pallas::Base::from_uniform_bytes(&hash)
}

/// Deterministic Fisher-Yates shuffle of the shares array.
///
/// Prevents the sorted denomination order from leaking balance information
/// through share indices. An adversary seeing (index, decrypted_value)
/// would otherwise learn the denomination's rank in the sorted
/// decomposition, tightening its estimate of the voter's total balance.
/// Shuffling makes each index equally likely to hold any denomination.
///
/// The permutation is derived from the same PRF used for El Gamal randomness
/// and blind factors, with a distinct shuffle domain separator.
/// Share index 0 is used for the PRF call (the seed depends on the VAN, round,
/// and proposal — not on the permutation step) to produce 64 bytes of
/// pseudorandom data, which is consumed 4 bytes at a time for modular indices.
pub(crate) fn deterministic_shuffle(
    shares: &mut [u64; NUM_SHARES],
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
) {
    // The share index is hardcoded to 0 here because the shuffle
    // function only needs one PRF call to seed the entire Fisher
    // Yates shuffle. It doesn't need per-share derivations.
    let seed = vote_share_prf(
        sk,
        domain_tags::VOTE_PRF_DOMAIN_SHUFFLE,
        round_id,
        proposal_id,
        van_commitment,
        0,
    );
    for i in (1..NUM_SHARES).rev() {
        // Each iteration consumes the next 4-byte slice of the seed as a
        // random u32: i=15 reads seed[0..4], i=14 reads seed[4..8], …,
        // i=1 reads seed[56..60] (15 draws × 4 bytes = 60 of the 64-byte seed).
        let byte_offset = (NUM_SHARES - 1 - i) * 4;
        let rand_bytes: [u8; 4] = seed[byte_offset..byte_offset + 4]
            .try_into()
            .expect("64-byte seed has room for 15 × 4-byte draws");
        let j = (u32::from_le_bytes(rand_bytes) as usize) % (i + 1);
        shares.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vote_share_bucket_prf_has_frozen_test_vector() {
        let sk = SpendingKey::from_bytes([0x42; 32]).expect("valid test spending key");
        let hash = vote_share_bucket_prf(
            &sk,
            domain_tags::VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED,
            pallas::Base::from(7u64),
            3,
            pallas::Base::from(11u64),
            2,
            5,
        );

        let expected: [u8; 64] = [
            0x6e, 0x88, 0x5e, 0xc2, 0x33, 0xbe, 0xfb, 0x75, 0x28, 0x8e, 0xe5, 0x6b, 0xc9, 0x7f,
            0xe5, 0x02, 0xaa, 0xdc, 0xe1, 0xf8, 0x95, 0x37, 0x84, 0x35, 0x81, 0x11, 0xe7, 0x78,
            0x0c, 0x94, 0x5e, 0x8e, 0xe8, 0x2f, 0x37, 0x98, 0x1c, 0x99, 0x89, 0xa8, 0xd0, 0xc3,
            0xfe, 0x54, 0xd8, 0xa2, 0x2f, 0xca, 0x73, 0xa6, 0xd2, 0x9d, 0x83, 0xda, 0x94, 0x0b,
            0x22, 0x94, 0xf4, 0x31, 0xc5, 0xfd, 0x0e, 0x03,
        ];
        if hash != expected {
            panic!(
                "vote_share_bucket_prf frozen vector mismatch; if intentional, update to:\n{:?}",
                hash
            );
        }
    }

    #[test]
    fn bucket_prf_differs_from_share_prf_and_across_buckets() {
        let sk = SpendingKey::from_bytes([0x42; 32]).expect("valid test spending key");
        let round = pallas::Base::from(7u64);
        let van = pallas::Base::from(11u64);

        let bucket0 = vote_share_bucket_prf(
            &sk,
            domain_tags::VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED,
            round,
            3,
            van,
            2,
            0,
        );
        let bucket1 = vote_share_bucket_prf(
            &sk,
            domain_tags::VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED,
            round,
            3,
            van,
            2,
            1,
        );
        assert_ne!(bucket0, bucket1, "bucket index must separate streams");

        let legacy = vote_share_prf(&sk, domain_tags::VOTE_PRF_DOMAIN_ELGAMAL, round, 3, van, 2);
        assert_ne!(bucket0, legacy, "weighted domain must separate streams");
    }

    #[test]
    fn derive_weighted_share_randomness_is_deterministic_and_layout_separated() {
        let sk = SpendingKey::from_bytes([0x24; 32]).expect("valid test spending key");
        let round = pallas::Base::from(9u64);
        let van = pallas::Base::from(13u64);

        let a = derive_weighted_share_randomness(&sk, round, 4, van, 1, 3, false);
        let b = derive_weighted_share_randomness(&sk, round, 4, van, 1, 3, false);
        assert_eq!(a, b, "derivation must be deterministic");
        assert!(!bool::from(a.is_zero()));
        assert!(base_to_scalar(a).is_some());

        let single = derive_weighted_share_randomness(&sk, round, 4, van, 1, 3, true);
        assert_ne!(a, single, "layouts must use distinct PRF domains");
    }
}
