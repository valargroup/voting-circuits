//! Vote proof builder (ZKP #2).
//!
//! Constructs a vote proof from delegation key material, a vote commitment
//! tree witness, and vote parameters. Lives inside the orchard crate to
//! access `pub(crate)` key internals.
//!
//! El Gamal encryption randomness and share blind factors are derived
//! deterministically via a Blake2b-512 PRF keyed by the spending key
//! and bound to the specific VAN being spent, enabling crash recovery
//! without persisting secrets and preventing nonce reuse across VANs.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use ff::{FromUniformBytes, PrimeField};
use group::{Curve, GroupEncoding};
use halo2_proofs::circuit::Value;
use pasta_curves::{arithmetic::CurveAffine, pallas};

use orchard::keys::{FullViewingKey, Scope, SpendAuthorizingKey, SpendingKey};

use super::circuit::{
    share_commitment, shares_hash, van_integrity_hash, van_nullifier_hash, vote_commitment_hash,
    Circuit, Instance, VOTE_COMM_TREE_DEPTH,
};
use super::prove::create_vote_proof;
use super::{base_to_scalar, spend_auth_g_affine};

/// Ballot divisor — must match `delegation::circuit::BALLOT_DIVISOR`.
const BALLOT_DIVISOR: u64 = 12_500_000;

/// Number of shares per vote.
const NUM_SHARES: usize = 16;

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

/// Decompose `num_ballots` into [`NUM_SHARES`] shares using a greedy
/// denomination strategy with randomized remainder distribution.
///
/// 1. **Greedy fill**: place the largest standard denominations that fit,
///    consuming up to [`MAX_DENOM_SHARES`] slots.
/// 2. **Remainder split**: if a non-zero remainder exists and there are ≥ 2
///    free slots, distribute the remainder across all free slots using
///    deterministic PRF-derived weights. If only 1 free slot remains, the
///    remainder goes there as-is.
/// 3. The caller then shuffles the result via [`deterministic_shuffle`].
///
/// The randomized remainder prevents a single non-standard value from
/// fingerprinting the voter's exact balance.
pub fn denomination_split(
    num_ballots: u64,
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
) -> [u64; NUM_SHARES] {
    let mut shares = [0u64; NUM_SHARES];
    let mut remaining = num_ballots;
    let mut idx = 0;

    for &d in &DENOMINATIONS {
        while remaining >= d && idx < MAX_DENOM_SHARES {
            shares[idx] = d;
            remaining -= d;
            idx += 1;
        }
    }

    if remaining > 0 {
        let free_slots = NUM_SHARES - idx;
        if free_slots >= 2 {
            distribute_remainder(
                &mut shares[idx..],
                remaining,
                sk,
                round_id,
                proposal_id,
                van_commitment,
                idx as u8,
            );
        } else {
            shares[idx] = remaining;
        }
    }

    shares
}

/// Spread `remainder` across `slots` using PRF-derived weights.
///
/// Each slot gets `floor(remainder * weight_i / total_weight)` with any
/// rounding residual added one-per-slot to the first slots.  Every slot
/// receives at least 1 so that no zero-valued slot leaks "this was padding."
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
    debug_assert!(n >= 2, "caller ensures at least 2 free slots");

    if remainder < n {
        // Not enough to put ≥1 in every slot; fill as many as possible.
        for i in 0..(remainder as usize) {
            slots[i] = 1;
        }
        return;
    }

    // Reserve 1 ballot per slot so every piece is non-zero.
    let distributable = remainder - n;

    // Derive a PRF weight per slot.
    let mut weights = Vec::with_capacity(slots.len());
    let mut total_weight: u64 = 0;
    for i in 0..slots.len() {
        let hash = vote_share_prf(
            sk,
            DOMAIN_REMAINDER,
            round_id,
            proposal_id,
            van_commitment,
            base_index.wrapping_add(i as u8),
        );
        let w = u32::from_le_bytes(hash[0..4].try_into().unwrap()) as u64 | 1;
        weights.push(w);
        total_weight += w;
    }

    // Weighted proportional split with rounding correction.
    let mut assigned: u64 = 0;
    for i in 0..slots.len() {
        let share = distributable * weights[i] / total_weight;
        slots[i] = 1 + share;
        assigned += share;
    }

    // Distribute any leftover from integer truncation, one per slot.
    let leftover = distributable - assigned;
    for i in 0..(leftover as usize) {
        slots[i] += 1;
    }
}

/// Encrypted share output from the vote proof builder.
///
/// Contains the El Gamal ciphertext components (compressed point bytes),
/// plaintext share value, and encryption randomness. Returned so the caller
/// can build reveal-share payloads using the exact ciphertexts committed in the proof.
#[derive(Debug, Clone)]
pub struct EncryptedShareOutput {
    /// Compressed El Gamal C1 point (32 bytes).
    pub c1: [u8; 32],
    /// Compressed El Gamal C2 point (32 bytes).
    pub c2: [u8; 32],
    /// Share index (0-15).
    pub share_index: u32,
    /// Plaintext share value.
    pub plaintext_value: u64,
    /// El Gamal randomness r (32 bytes, LE pallas::Base repr).
    /// Deterministically derived from (sk, round_id, proposal_id, van_commitment, share_index).
    pub randomness: [u8; 32],
}

/// Result of building a vote proof.
#[derive(Debug)]
pub struct VoteProofBundle {
    /// Serialized Halo2 proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for the proof.
    pub instance: Instance,
    /// Compressed r_vpk (32 bytes) for sighash computation and signature verification.
    pub r_vpk_bytes: [u8; 32],
    /// Encrypted shares generated during proof construction.
    /// These are the exact ciphertexts committed in the vote commitment hash
    /// and must be used for reveal-share payloads.
    pub encrypted_shares: [EncryptedShareOutput; 16],
    /// Poseidon hash of all encrypted share x-coordinates.
    /// Intermediate value: vote_commitment = H(DOMAIN_VC, voting_round_id, shares_hash, proposal_id, vote_decision).
    /// Needed by the helper server to verify share payloads.
    pub shares_hash: pallas::Base,
    /// Per-share blind factors for blinded commitments.
    /// share_comm_i = Poseidon(blind_i, c1_i_x, c2_i_x).
    /// Deterministically derived from (sk, round_id, proposal_id, van_commitment, share_index).
    pub share_blinds: [pallas::Base; 16],
    /// Pre-computed per-share Poseidon commitments.
    /// share_comm_i = Poseidon(blind_i, c1_i_x, c2_i_x).
    /// Provided as public inputs to ZKP #3 (share reveal) so the helper
    /// server only needs the primary share's blind, not all 16.
    pub share_comms: [pallas::Base; 16],
}

/// Errors that can occur during vote proof construction.
#[derive(Debug)]
pub enum VoteProofBuildError {
    /// A share randomness value could not be converted to a scalar.
    InvalidRandomness(String),
    /// The total note value cannot be split into valid shares.
    InvalidShares(String),
}

impl core::fmt::Display for VoteProofBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VoteProofBuildError::InvalidRandomness(msg) => {
                write!(f, "invalid randomness: {}", msg)
            }
            VoteProofBuildError::InvalidShares(msg) => {
                write!(f, "invalid shares: {}", msg)
            }
        }
    }
}

/// Extract the voting spending key scalar from a SpendingKey.
///
/// This replicates the sign-correction logic from `SpendAuthorizingKey::from`:
/// `ask = PRF_expand(sk)`, then negate if the resulting ak has ỹ = 1.
fn extract_vsk(sk: &SpendingKey) -> pallas::Scalar {
    let ask_raw = SpendAuthorizingKey::derive_inner(sk);
    let g = pallas::Point::from(spend_auth_g_affine());
    let ak_point = (g * ask_raw).to_affine();
    let ak_bytes = ak_point.to_bytes();

    // If the sign bit of ak is 1, the real ask was negated.
    if (ak_bytes.as_ref()[31] >> 7) == 1 {
        -ask_raw
    } else {
        ask_raw
    }
}

/// Blake2b-512 personalization for vote share secret derivation.
/// Distinct from Zcash's `"Zcash_ExpandSeed"` to avoid domain collisions.
const VOTE_PRF_PERSONALIZATION: &[u8; 16] = b"ZcashVote_Expand";

/// Domain separator for El Gamal encryption randomness.
const DOMAIN_ELGAMAL: u8 = 0x00;
/// Domain separator for share commitment blind factors.
const DOMAIN_BLIND: u8 = 0x01;
/// Domain separator for share-order shuffle seed.
const DOMAIN_SHUFFLE: u8 = 0x02;
/// Domain separator for remainder distribution weights.
const DOMAIN_REMAINDER: u8 = 0x03;

/// Core PRF: BLAKE2b-512 keyed by the spending key with voting-specific
/// personalization and domain-separated inputs.
///
/// `PRF(sk, domain, round_id, proposal_id, van_commitment, share_index)`
///   = BLAKE2b-512("ZcashVote_Expand", sk || domain || round_id || proposal_id_le64 || van_commitment || share_index_u8)
///
/// The `van_commitment` field binds the derivation to a specific VAN.
/// Without it, a user with multiple VANs (from >5 notes in Phase 1)
/// voting on the same proposal would derive identical El Gamal nonces,
/// enabling a classic nonce-reuse attack on the ciphertexts.
fn vote_share_prf(
    sk: &SpendingKey,
    domain: u8,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
    share_index: u8,
) -> [u8; 64] {
    *blake2b_simd::Params::new()
        .hash_length(64)
        .personal(VOTE_PRF_PERSONALIZATION)
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

/// Derive deterministic El Gamal randomness `r_i` for a share.
///
/// Returns a `pallas::Base` element that is also a valid `pallas::Scalar`.
/// We reduce mod p_base first; since p_base < q_scalar on the Pallas curve,
/// every Base element is representable as a Scalar.
pub fn derive_share_randomness(
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
    share_index: u8,
) -> pallas::Base {
    let hash = vote_share_prf(
        sk,
        DOMAIN_ELGAMAL,
        round_id,
        proposal_id,
        van_commitment,
        share_index,
    );
    let r = pallas::Base::from_uniform_bytes(&hash);
    debug_assert!(base_to_scalar(r).is_some(), "p < q guarantees Base→Scalar");
    r
}

/// Derive deterministic blind factor `blind_i` for a share commitment.
pub fn derive_share_blind(
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
    share_index: u8,
) -> pallas::Base {
    let hash = vote_share_prf(
        sk,
        DOMAIN_BLIND,
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
/// through share indices. When shares are distributed across multiple helper
/// servers, a server seeing (index, decrypted_value) would otherwise learn
/// the denomination's rank in the sorted decomposition, tightening its
/// estimate of the voter's total balance. Shuffling makes each index
/// equally likely to hold any denomination.
///
/// The permutation is derived from the same PRF used for El Gamal randomness
/// and blind factors, with a distinct domain separator (`DOMAIN_SHUFFLE`).
/// Share index 0 is used for the PRF call (the seed depends on the VAN, round,
/// and proposal — not on the permutation step) to produce 64 bytes of
/// pseudorandom data, which is consumed 4 bytes at a time for modular indices.
fn deterministic_shuffle(
    shares: &mut [u64; NUM_SHARES],
    sk: &SpendingKey,
    round_id: pallas::Base,
    proposal_id: u64,
    van_commitment: pallas::Base,
) {
    let seed = vote_share_prf(sk, DOMAIN_SHUFFLE, round_id, proposal_id, van_commitment, 0);
    for i in (1..NUM_SHARES).rev() {
        let byte_offset = (NUM_SHARES - 1 - i) * 4;
        let rand_bytes: [u8; 4] = seed[byte_offset..byte_offset + 4]
            .try_into()
            .expect("64-byte seed has room for 15 × 4-byte draws");
        let j = (u32::from_le_bytes(rand_bytes) as usize) % (i + 1);
        shares.swap(i, j);
    }
}

/// Build a real vote proof (ZKP #2) from delegation key material.
///
/// This function constructs the full vote proof circuit, computes all
/// public inputs, and generates a Halo2 proof.
///
/// # Arguments
///
/// * `sk` - The SpendingKey used during delegation (ZKP #1).
/// * `address_index` - The diversifier index of the output recipient
///   address used in delegation (typically 1).
/// * `total_note_value` - Sum of delegated note values in raw zatoshi (e.g. 15_000_000).
///   Internally converted to ballot count via floor-division by BALLOT_DIVISOR.
/// * `van_comm_rand` - The blinding factor used for the VAN in delegation.
/// * `voting_round_id` - The vote round identifier (Pallas base field element).
/// * `vote_comm_tree_path` - Merkle authentication path (24 siblings) for
///   the VAN in the vote commitment tree.
/// * `vote_comm_tree_position` - Leaf position of the VAN in the tree.
/// * `anchor_height` - The block height at which the tree was snapshotted
///   (must match the on-chain commitment tree root).
/// * `proposal_id` - Which proposal to vote on (1-indexed, must be in [1, 15]).
/// * `vote_decision` - The voter's choice.
/// * `ea_pk` - Election authority public key (Pallas affine point from session).
/// * `alpha_v` - Spend auth randomizer for the voting hotkey. The caller
///   retains this to sign the sighash with `rsk_v = ask_v.randomize(&alpha_v)`.
///
/// El Gamal encryption randomness (`r_i`) and share blind factors (`blind_i`)
/// are derived deterministically from `sk`, `voting_round_id`, `proposal_id`,
/// `vote_authority_note_old`, and each share's index via a Blake2b-512 PRF.
/// Including the VAN commitment prevents nonce reuse when the same user has
/// multiple VANs (from >5 notes in Phase 1) voting on the same proposal.
/// This allows the client to re-derive the same secrets after a crash without
/// persisting them.
///
/// **Expensive**: K=14 proof generation takes ~30-60 seconds in release mode.
#[allow(clippy::too_many_arguments)]
pub fn build_vote_proof_from_delegation(
    sk: &SpendingKey,
    address_index: u32,
    total_note_value: u64,
    van_comm_rand: pallas::Base,
    voting_round_id: pallas::Base,
    vote_comm_tree_path: [pallas::Base; VOTE_COMM_TREE_DEPTH],
    vote_comm_tree_position: u32,
    anchor_height: u32,
    proposal_id: u64,
    vote_decision: u64,
    ea_pk: pallas::Affine,
    alpha_v: pallas::Scalar,
    proposal_authority_old_u64: u64,
) -> Result<VoteProofBundle, VoteProofBuildError> {
    // ---- Key derivation (matches delegation's key hierarchy) ----

    let vsk = extract_vsk(sk);
    let fvk: FullViewingKey = sk.into();
    let vsk_nk = fvk.nk().inner();
    let rivk_v = fvk.rivk(Scope::External).inner();

    let address = fvk.address_at(address_index, Scope::External);
    let vpk_g_d_affine = address.g_d().to_affine();
    let vpk_pk_d_affine = address.pk_d().inner().to_affine();

    let vpk_g_d_x = *vpk_g_d_affine.coordinates().unwrap().x();
    let vpk_pk_d_x = *vpk_pk_d_affine.coordinates().unwrap().x();

    // ---- Fast key-chain consistency checks (instant, no circuit) ----
    {
        use core::iter;
        use group::ff::PrimeFieldBits;
        use halo2_gadgets::sinsemilla::primitives::CommitDomain;
        use orchard::constants::{fixed_bases::COMMIT_IVK_PERSONALIZATION, L_ORCHARD_BASE};

        // Check 1: [vsk] * SpendAuthG must match the ak from the FullViewingKey.
        let ak_from_vsk = (pallas::Point::from(spend_auth_g_affine()) * vsk).to_affine();
        let fvk_bytes = fvk.to_bytes();
        let ak_from_fvk_bytes: [u8; 32] = fvk_bytes[0..32].try_into().unwrap();
        let ak_from_fvk: pallas::Affine = {
            let opt: Option<pallas::Point> = pallas::Point::from_bytes(&ak_from_fvk_bytes).into();
            opt.expect("ak from fvk must be a valid point").to_affine()
        };
        assert_eq!(
            ak_from_vsk, ak_from_fvk,
            "extract_vsk bug: [vsk]*SpendAuthG != ak from FullViewingKey"
        );

        // Check 2: CommitIvk(ak_x, nk, rivk) must produce an ivk where [ivk]*g_d == pk_d.
        let ak_x = *ak_from_vsk.coordinates().unwrap().x();
        let domain = CommitDomain::new(COMMIT_IVK_PERSONALIZATION);
        let ivk = domain
            .short_commit(
                iter::empty()
                    .chain(ak_x.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE))
                    .chain(vsk_nk.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE)),
                &rivk_v,
            )
            .expect("CommitIvk must not produce bottom");
        let ivk_scalar = base_to_scalar(ivk).expect("ivk must be convertible to scalar");
        let pk_d_derived = (pallas::Point::from(vpk_g_d_affine) * ivk_scalar).to_affine();
        assert_eq!(
            pk_d_derived, vpk_pk_d_affine,
            "CommitIvk chain mismatch: [ivk]*g_d != pk_d from address"
        );

        std::eprintln!("[BUILDER] key-chain consistency checks passed");
    }

    // ---- Proposal authority ----

    let proposal_authority_old = pallas::Base::from(proposal_authority_old_u64);
    let one_shifted = pallas::Base::from(1u64 << proposal_id);
    let proposal_authority_new = proposal_authority_old - one_shifted;

    // ---- Ballot scaling (must match ZKP #1's BALLOT_DIVISOR) ----

    let num_ballots = total_note_value / BALLOT_DIVISOR;
    let num_ballots_base = pallas::Base::from(num_ballots);

    // ---- VAN integrity hashes ----
    // The VAN commitment hashes num_ballots (not raw zatoshi), matching
    // the delegation circuit (ZKP #1 condition 7).

    let vote_authority_note_old = van_integrity_hash(
        vpk_g_d_x,
        vpk_pk_d_x,
        num_ballots_base,
        voting_round_id,
        proposal_authority_old,
        van_comm_rand,
    );

    let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);

    let vote_authority_note_new = van_integrity_hash(
        vpk_g_d_x,
        vpk_pk_d_x,
        num_ballots_base,
        voting_round_id,
        proposal_authority_new,
        van_comm_rand,
    );

    // ---- Shares (denomination-based split of num_ballots into 16 parts) ----
    // Each share must be in [0, 2^30) for the range check.
    // Shares sum to num_ballots (ballot count), not raw zatoshi.

    let mut shares_u64 = denomination_split(
        num_ballots,
        sk,
        voting_round_id,
        proposal_id,
        vote_authority_note_old,
    );
    deterministic_shuffle(
        &mut shares_u64,
        sk,
        voting_round_id,
        proposal_id,
        vote_authority_note_old,
    );

    // Verify all shares are in range
    for (i, &s) in shares_u64.iter().enumerate() {
        if s >= (1u64 << 30) {
            return Err(VoteProofBuildError::InvalidShares(format!(
                "share {} = {} exceeds 2^30",
                i, s
            )));
        }
    }

    let shares_base: [pallas::Base; 16] =
        core::array::from_fn(|i| pallas::Base::from(shares_u64[i]));

    // ---- El Gamal encryption of shares ----
    //
    // Encrypts each share and captures both the x-coordinates (for circuit constraints)
    // and the full compressed point bytes (for reveal-share payloads).

    let ea_pk_point = pallas::Point::from(ea_pk);
    let ea_pk_x = *ea_pk.coordinates().unwrap().x();
    let ea_pk_y = *ea_pk.coordinates().unwrap().y();

    let g = pallas::Point::from(spend_auth_g_affine());
    let mut enc_c1_x = [pallas::Base::zero(); 16];
    let mut enc_c2_x = [pallas::Base::zero(); 16];
    let mut share_randomness = [pallas::Base::zero(); 16];
    let mut enc_share_outputs: [EncryptedShareOutput; 16] =
        core::array::from_fn(|i| EncryptedShareOutput {
            c1: [0u8; 32],
            c2: [0u8; 32],
            share_index: i as u32,
            plaintext_value: shares_u64[i],
            randomness: [0u8; 32],
        });

    for i in 0..16 {
        let r = derive_share_randomness(
            sk,
            voting_round_id,
            proposal_id,
            vote_authority_note_old,
            i as u8,
        );
        share_randomness[i] = r;
        let r_scalar = base_to_scalar(r).expect("derive_share_randomness guarantees scalar-range");
        let v_scalar = base_to_scalar(shares_base[i]).expect("share value in range");

        let c1_point = (g * r_scalar).to_affine();
        let c2_point = (g * v_scalar + ea_pk_point * r_scalar).to_affine();

        enc_c1_x[i] = *c1_point.coordinates().unwrap().x();
        enc_c2_x[i] = *c2_point.coordinates().unwrap().x();

        enc_share_outputs[i].c1 = c1_point.to_bytes();
        enc_share_outputs[i].c2 = c2_point.to_bytes();
        enc_share_outputs[i].randomness = r.to_repr();
    }

    let share_blinds: [pallas::Base; 16] = core::array::from_fn(|i| {
        derive_share_blind(
            sk,
            voting_round_id,
            proposal_id,
            vote_authority_note_old,
            i as u8,
        )
    });
    let share_comms: [pallas::Base; 16] =
        core::array::from_fn(|i| share_commitment(share_blinds[i], enc_c1_x[i], enc_c2_x[i]));
    let shares_hash_val = shares_hash(share_blinds, enc_c1_x, enc_c2_x);

    // ---- Condition 4: r_vpk = ak + [alpha_v] * G ----
    // alpha_v is now provided by the caller so they can sign with rsk_v.
    let ak_point = pallas::Point::from(spend_auth_g_affine()) * vsk;
    let r_vpk = (ak_point + pallas::Point::from(spend_auth_g_affine()) * alpha_v).to_affine();
    let r_vpk_x = *r_vpk.coordinates().unwrap().x();
    let r_vpk_y = *r_vpk.coordinates().unwrap().y();
    let r_vpk_bytes: [u8; 32] = r_vpk.to_bytes();

    // ---- Vote commitment ----

    let proposal_id_base = pallas::Base::from(proposal_id);
    let vote_decision_base = pallas::Base::from(vote_decision);
    let vote_commitment = vote_commitment_hash(
        voting_round_id,
        shares_hash_val,
        proposal_id_base,
        vote_decision_base,
    );

    // ---- Vote commitment tree root (from auth path) ----
    // Recompute the root from the leaf + auth path to set as public input.

    let vote_comm_tree_root = {
        use super::circuit::poseidon_hash_2;

        let mut current = vote_authority_note_old;
        for level in 0..VOTE_COMM_TREE_DEPTH {
            let sibling = vote_comm_tree_path[level];
            if vote_comm_tree_position & (1 << level) == 0 {
                current = poseidon_hash_2(current, sibling);
            } else {
                current = poseidon_hash_2(sibling, current);
            }
        }
        current
    };

    // ---- Build circuit ----

    let mut circuit = Circuit::with_van_witnesses(
        Value::known(vote_comm_tree_path),
        Value::known(vote_comm_tree_position),
        Value::known(vpk_g_d_affine),
        Value::known(vpk_pk_d_affine),
        Value::known(num_ballots_base),
        Value::known(proposal_authority_old),
        Value::known(van_comm_rand),
        Value::known(vote_authority_note_old),
        Value::known(vsk),
        Value::known(rivk_v),
        Value::known(vsk_nk),
        Value::known(alpha_v),
    );
    circuit.one_shifted = Value::known(one_shifted);
    circuit.shares = shares_base.map(Value::known);
    circuit.enc_share_c1_x = enc_c1_x.map(Value::known);
    circuit.enc_share_c2_x = enc_c2_x.map(Value::known);
    circuit.share_blinds = share_blinds.map(Value::known);
    circuit.share_randomness = share_randomness.map(Value::known);
    circuit.ea_pk = Value::known(ea_pk);
    circuit.vote_decision = Value::known(vote_decision_base);

    // ---- Build instance (public inputs) ----

    let anchor_height_base = pallas::Base::from(u64::from(anchor_height));
    let instance = Instance::from_parts(
        van_nullifier,
        r_vpk_x,
        r_vpk_y,
        vote_authority_note_new,
        vote_commitment,
        vote_comm_tree_root,
        anchor_height_base,
        proposal_id_base,
        voting_round_id,
        ea_pk_x,
        ea_pk_y,
    );

    // ---- MockProver check ----

    {
        use halo2_proofs::dev::MockProver;
        let mock_circuit = circuit.clone();
        let prover = MockProver::run(
            super::circuit::K,
            &mock_circuit,
            vec![instance.to_halo2_instance()],
        )
        .expect("MockProver::run should not fail");

        if let Err(failures) = prover.verify() {
            return Err(VoteProofBuildError::InvalidShares(format!(
                "circuit constraints not satisfied: {} failure(s): {:?}",
                failures.len(),
                failures,
            )));
        }
        std::eprintln!("[BUILDER] MockProver passed");
    }

    // ---- Generate proof ----

    let proof = create_vote_proof(circuit, &instance);

    Ok(VoteProofBundle {
        proof,
        instance,
        r_vpk_bytes,
        encrypted_shares: enc_share_outputs,
        shares_hash: shares_hash_val,
        share_blinds,
        share_comms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sk() -> SpendingKey {
        SpendingKey::from_bytes([0x42; 32]).expect("valid spending key")
    }

    fn test_round_id() -> pallas::Base {
        pallas::Base::from(0xCAFE_u64)
    }

    fn test_van() -> pallas::Base {
        pallas::Base::from(0xDEAD_u64)
    }

    #[test]
    fn derive_share_randomness_is_deterministic() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let a = derive_share_randomness(&sk, round_id, 1, van, 0);
        let b = derive_share_randomness(&sk, round_id, 1, van, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn derive_share_blind_is_deterministic() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let a = derive_share_blind(&sk, round_id, 1, van, 0);
        let b = derive_share_blind(&sk, round_id, 1, van, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn derive_share_randomness_is_valid_scalar() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        for i in 0..16u8 {
            let r = derive_share_randomness(&sk, round_id, 1, van, i);
            assert!(
                base_to_scalar(r).is_some(),
                "r_{} must be convertible to scalar",
                i
            );
        }
    }

    #[test]
    fn different_share_index_gives_different_values() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let r0 = derive_share_randomness(&sk, round_id, 1, van, 0);
        let r1 = derive_share_randomness(&sk, round_id, 1, van, 1);
        assert_ne!(r0, r1);

        let b0 = derive_share_blind(&sk, round_id, 1, van, 0);
        let b1 = derive_share_blind(&sk, round_id, 1, van, 1);
        assert_ne!(b0, b1);
    }

    #[test]
    fn different_proposal_id_gives_different_values() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let r_p1 = derive_share_randomness(&sk, round_id, 1, van, 0);
        let r_p2 = derive_share_randomness(&sk, round_id, 2, van, 0);
        assert_ne!(r_p1, r_p2);
    }

    #[test]
    fn different_round_id_gives_different_values() {
        let sk = test_sk();
        let van = test_van();
        let r_a = derive_share_randomness(&sk, pallas::Base::from(1u64), 1, van, 0);
        let r_b = derive_share_randomness(&sk, pallas::Base::from(2u64), 1, van, 0);
        assert_ne!(r_a, r_b);
    }

    #[test]
    fn randomness_and_blind_differ_for_same_inputs() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let r = derive_share_randomness(&sk, round_id, 1, van, 0);
        let b = derive_share_blind(&sk, round_id, 1, van, 0);
        assert_ne!(r, b, "domain separation must prevent r == blind");
    }

    #[test]
    fn all_16_shares_are_distinct() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let randoms: Vec<_> = (0..16u8)
            .map(|i| derive_share_randomness(&sk, round_id, 1, van, i))
            .collect();
        let blinds: Vec<_> = (0..16u8)
            .map(|i| derive_share_blind(&sk, round_id, 1, van, i))
            .collect();
        for i in 0..16 {
            for j in (i + 1)..16 {
                assert_ne!(randoms[i], randoms[j], "r_{} == r_{}", i, j);
                assert_ne!(blinds[i], blinds[j], "blind_{} == blind_{}", i, j);
            }
        }
    }

    #[test]
    fn different_van_commitment_gives_different_values() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van_a = pallas::Base::from(0xAAAA_u64);
        let van_b = pallas::Base::from(0xBBBB_u64);
        for i in 0..16u8 {
            let r_a = derive_share_randomness(&sk, round_id, 1, van_a, i);
            let r_b = derive_share_randomness(&sk, round_id, 1, van_b, i);
            assert_ne!(r_a, r_b, "r_{} must differ across VANs", i);

            let b_a = derive_share_blind(&sk, round_id, 1, van_a, i);
            let b_b = derive_share_blind(&sk, round_id, 1, van_b, i);
            assert_ne!(b_a, b_b, "blind_{} must differ across VANs", i);
        }
    }

    // ---- denomination_split tests ----
    //
    // Visual key:
    //   D = denomination (standard value, blends across voters)
    //   R = random (PRF-derived, prevents exact balance fingerprint)
    //   0 = zero (encrypted with fresh randomness, indistinguishable from non-zero)
    //
    // Layout: [0..8] = greedy denom slots | [9..15] = remainder / random slots
    // After shuffle, positions are randomized — these show the pre-shuffle array.

    /// Helper: print shares array for visual inspection during --nocapture runs.
    fn show(label: &str, shares: &[u64; 16]) {
        let parts: Vec<String> = shares
            .iter()
            .map(|&v| {
                if v == 0 {
                    "0".into()
                } else if v >= 1_000_000 {
                    format!("{}M", v / 1_000_000)
                } else if v >= 1_000 {
                    format!("{}K", v / 1_000)
                } else {
                    format!("{}", v)
                }
            })
            .collect();
        std::eprintln!("  {}: [{}]", label, parts.join(", "));
    }

    #[test]
    fn denom_split_zero_ballots() {
        // 0 ballots — all slots empty
        // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(0, &sk, rid, 1, van);
        show("0 ballots", &shares);
        assert_eq!(shares, [0; 16]);
    }

    #[test]
    fn denom_split_single_ballot() {
        // 1 ballot (0.125 ZEC) — smallest denomination
        // [D:1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(1, &sk, rid, 1, van);
        show("1 ballot (0.125 ZEC)", &shares);
        assert_eq!(shares[0], 1);
        for i in 1..16 {
            assert_eq!(shares[i], 0);
        }
    }

    #[test]
    fn denom_split_sub_zec() {
        // 4 ballots (0.5 ZEC)
        // [D:1, D:1, D:1, D:1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(4, &sk, rid, 1, van);
        show("4 ballots (0.5 ZEC)", &shares);
        assert_eq!(shares[0..4], [1; 4]);
        for i in 4..16 {
            assert_eq!(shares[i], 0);
        }
    }

    #[test]
    fn denom_split_one_zec() {
        // 8 ballots (1 ZEC)
        // [D:1, D:1, D:1, D:1, D:1, D:1, D:1, D:1, 0, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(8, &sk, rid, 1, van);
        show("8 ballots (1 ZEC)", &shares);
        assert_eq!(shares[0..8], [1; 8]);
        for i in 8..16 {
            assert_eq!(shares[i], 0);
        }
    }

    #[test]
    fn denom_split_small_balance() {
        // 50 ballots (6.25 ZEC) — 5 denom slots, all standard
        // [D:10, D:10, D:10, D:10, D:10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(50, &sk, rid, 1, van);
        show("50 ballots (6.25 ZEC)", &shares);
        assert_eq!(shares[0..5], [10; 5]);
        for i in 5..16 {
            assert_eq!(shares[i], 0);
        }
    }

    #[test]
    fn denom_split_all_denoms_exact() {
        // 11,111 ballots (1,388.9 ZEC) — one of each denom, no remainder
        // [D:10K, D:1K, D:100, D:10, D:1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(11_111, &sk, rid, 1, van);
        show("11,111 ballots (1,388.9 ZEC)", &shares);
        assert_eq!(shares[0], 10_000);
        assert_eq!(shares[1], 1_000);
        assert_eq!(shares[2], 100);
        assert_eq!(shares[3], 10);
        assert_eq!(shares[4], 1);
        for i in 5..16 {
            assert_eq!(shares[i], 0);
        }
    }

    #[test]
    fn denom_split_medium_holder_with_remainder() {
        // 4,800 ballots (600 ZEC) — greedy fills 9 (4×1K + 5×100 = 4,500), remainder 300
        // [D:1K, D:1K, D:1K, D:1K, D:100, D:100, D:100, D:100, D:100, R, R, R, R, R, R, R]
        //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^
        //  9 denomination slots (4,500)                                  7 random slots (300)
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(4_800, &sk, rid, 1, van);
        show("4,800 ballots (600 ZEC)", &shares);
        assert_eq!(shares[0..4], [1_000; 4]);
        assert_eq!(shares[4..9], [100; 5]);
        let remainder_sum: u64 = shares[9..16].iter().sum();
        assert_eq!(remainder_sum, 300);
        for i in 9..16 {
            assert!(shares[i] > 0, "remainder slot {} should be non-zero", i);
        }
        assert_eq!(shares.iter().sum::<u64>(), 4_800);
    }

    #[test]
    fn denom_split_high_hamming_weight() {
        // 999 ballots (124.875 ZEC) — greedy fills 9 (9×100 = 900), remainder 99
        // [D:100, D:100, D:100, D:100, D:100, D:100, D:100, D:100, D:100, R, R, R, R, R, R, R]
        //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
        //  9 denomination slots (900)                                       7 random slots (99)
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(999, &sk, rid, 1, van);
        show("999 ballots (124.875 ZEC)", &shares);
        assert_eq!(shares[0..9], [100; 9]);
        let remainder_sum: u64 = shares[9..16].iter().sum();
        assert_eq!(remainder_sum, 99);
        for i in 9..16 {
            assert!(shares[i] > 0, "remainder slot {} should be non-zero", i);
        }
    }

    #[test]
    fn denom_split_exact_denomination_match() {
        // 3M ballots (375 ZEC) — 3 denom slots, no remainder
        // [D:1M, D:1M, D:1M, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(3_000_000, &sk, rid, 1, van);
        show("3M ballots (375 ZEC)", &shares);
        assert_eq!(shares[0..3], [1_000_000; 3]);
        for i in 3..16 {
            assert_eq!(shares[i], 0);
        }
    }

    #[test]
    fn denom_split_8m_ballots() {
        // 8M ballots (1M ZEC) — 8 denom slots, no remainder
        // [D:1M, D:1M, D:1M, D:1M, D:1M, D:1M, D:1M, D:1M, 0, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(8_000_000, &sk, rid, 1, van);
        show("8M ballots (1M ZEC)", &shares);
        assert_eq!(shares[0..8], [1_000_000; 8]);
        for i in 8..16 {
            assert_eq!(shares[i], 0);
        }
    }

    #[test]
    fn denom_split_fills_all_9_denom_slots() {
        // 90M ballots (11.25M ZEC) — all 9 denom slots filled, no remainder
        // [D:10M, D:10M, D:10M, D:10M, D:10M, D:10M, D:10M, D:10M, D:10M, 0, 0, 0, 0, 0, 0, 0]
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(90_000_000, &sk, rid, 1, van);
        show("90M ballots (11.25M ZEC)", &shares);
        assert_eq!(shares[0..9], [10_000_000; 9]);
        for i in 9..16 {
            assert_eq!(shares[i], 0);
        }
    }

    #[test]
    fn denom_split_overflow_into_remainder() {
        // 100M ballots (12.5M ZEC) — 9 denom slots full (9×10M), remainder 10M in 7 random slots
        // [D:10M, D:10M, D:10M, D:10M, D:10M, D:10M, D:10M, D:10M, D:10M, R, R, R, R, R, R, R]
        //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^
        //  9 denomination slots (90M)                                        7 random slots (10M)
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(100_000_000, &sk, rid, 1, van);
        show("100M ballots (12.5M ZEC)", &shares);
        assert_eq!(shares[0..9], [10_000_000; 9]);
        let remainder_sum: u64 = shares[9..16].iter().sum();
        assert_eq!(remainder_sum, 10_000_000);
        for i in 9..16 {
            assert!(shares[i] > 0, "remainder slot {} should be non-zero", i);
        }
    }

    #[test]
    fn denom_split_mixed_with_remainder() {
        // 1,234,567 ballots (154,320.9 ZEC) — 9 denom slots, remainder distributed
        // [D:1M, D:100K, D:100K, D:10K, D:10K, D:10K, D:1K, D:1K, D:1K, R, R, R, R, R, R, R]
        //  greedy: 1M + 200K + 30K + 3K = 1,233,000                      remainder: 1,567
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(1_234_567, &sk, rid, 1, van);
        show("1,234,567 ballots (154K ZEC)", &shares);
        assert_eq!(shares[0], 1_000_000);
        assert_eq!(shares[1..3], [100_000; 2]);
        assert_eq!(shares[3..6], [10_000; 3]);
        assert_eq!(shares[6..9], [1_000; 3]);
        let remainder_sum: u64 = shares[9..16].iter().sum();
        assert_eq!(remainder_sum, 1_567);
        assert_eq!(shares.iter().sum::<u64>(), 1_234_567);
    }

    #[test]
    fn denom_split_small_remainder_fewer_than_free_slots() {
        // 10,000,003 ballots — 1 denom slot (10M), remainder 3 across 7 free slots
        // [D:10M, R:1, R:1, R:1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        //  remainder 3 < 7 free slots, so only 3 of 7 get a value
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let shares = denomination_split(10_000_003, &sk, rid, 1, van);
        show("10,000,003 ballots", &shares);
        assert_eq!(shares[0], 10_000_000);
        let remainder_sum: u64 = shares[1..16].iter().sum();
        assert_eq!(remainder_sum, 3);
        assert_eq!(shares.iter().sum::<u64>(), 10_000_003);
    }

    // ---- invariant tests ----

    #[test]
    fn denom_split_sum_invariant() {
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let test_values: [u64; 14] = [
            0,
            1,
            50,
            99,
            100,
            999,
            1_000,
            10_000,
            100_000,
            1_000_000,
            8_234_567,
            20_000_000,
            80_000_000,
            168_000_000,
        ];
        for &v in &test_values {
            let shares = denomination_split(v, &sk, rid, 1, van);
            assert_eq!(
                shares.iter().sum::<u64>(),
                v,
                "sum invariant violated for {}",
                v
            );
        }
    }

    #[test]
    fn denom_split_all_shares_in_range() {
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let test_values: [u64; 8] = [
            1,
            10_000,
            1_000_000,
            8_234_567,
            15_000_000,
            20_000_000,
            80_000_000,
            168_000_000,
        ];
        for &v in &test_values {
            let shares = denomination_split(v, &sk, rid, 1, van);
            for (i, &s) in shares.iter().enumerate() {
                assert!(
                    s < (1u64 << 30),
                    "share {} = {} exceeds 2^30 for {}",
                    i,
                    s,
                    v
                );
            }
        }
    }

    // ---- remainder randomization tests ----

    #[test]
    fn remainder_is_deterministic() {
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let a = denomination_split(999, &sk, rid, 1, van);
        let b = denomination_split(999, &sk, rid, 1, van);
        assert_eq!(a, b);
    }

    #[test]
    fn remainder_differs_across_proposals() {
        // Same balance, different proposal_id → same denoms, different random remainder
        let sk = test_sk();
        let rid = test_round_id();
        let van = test_van();
        let a = denomination_split(999, &sk, rid, 1, van);
        let b = denomination_split(999, &sk, rid, 2, van);
        show("999 ballots, proposal 1", &a);
        show("999 ballots, proposal 2", &b);
        assert_eq!(a[0..9], b[0..9], "denomination slots should be identical");
        assert_ne!(
            a[9..16],
            b[9..16],
            "remainder should differ across proposals"
        );
    }

    #[test]
    fn remainder_differs_across_vans() {
        // Same balance, different VAN → same denoms, different random remainder
        let sk = test_sk();
        let rid = test_round_id();
        let van_a = pallas::Base::from(0xAAAA_u64);
        let van_b = pallas::Base::from(0xBBBB_u64);
        let a = denomination_split(999, &sk, rid, 1, van_a);
        let b = denomination_split(999, &sk, rid, 1, van_b);
        show("999 ballots, VAN A", &a);
        show("999 ballots, VAN B", &b);
        assert_eq!(a[0..9], b[0..9], "denomination slots should be identical");
        assert_ne!(a[9..16], b[9..16], "remainder should differ across VANs");
    }

    // ---- deterministic_shuffle tests ----

    #[test]
    fn shuffle_preserves_sum() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let mut shares = denomination_split(8_234_567, &sk, round_id, 1, van);
        let sum_before = shares.iter().sum::<u64>();
        deterministic_shuffle(&mut shares, &sk, round_id, 1, van);
        assert_eq!(shares.iter().sum::<u64>(), sum_before);
    }

    #[test]
    fn shuffle_preserves_multiset() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let original = denomination_split(4_800, &sk, round_id, 1, van);
        let mut shuffled = original;
        deterministic_shuffle(&mut shuffled, &sk, round_id, 1, van);
        let mut sorted_orig = original;
        sorted_orig.sort();
        let mut sorted_shuf = shuffled;
        sorted_shuf.sort();
        assert_eq!(sorted_orig, sorted_shuf, "shuffle must be a permutation");
    }

    #[test]
    fn shuffle_is_deterministic() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let mut a = denomination_split(4_800, &sk, round_id, 1, van);
        let mut b = denomination_split(4_800, &sk, round_id, 1, van);
        deterministic_shuffle(&mut a, &sk, round_id, 1, van);
        deterministic_shuffle(&mut b, &sk, round_id, 1, van);
        assert_eq!(a, b, "same inputs must produce same permutation");
    }

    #[test]
    fn shuffle_differs_across_proposals() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let mut a = denomination_split(4_800, &sk, round_id, 1, van);
        let mut b = denomination_split(4_800, &sk, round_id, 1, van);
        deterministic_shuffle(&mut a, &sk, round_id, 1, van);
        deterministic_shuffle(&mut b, &sk, round_id, 2, van);
        assert_ne!(
            a, b,
            "different proposals should produce different permutations"
        );
    }

    #[test]
    fn shuffle_differs_across_vans() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van_a = pallas::Base::from(0xAAAA_u64);
        let van_b = pallas::Base::from(0xBBBB_u64);
        let mut a = denomination_split(4_800, &sk, round_id, 1, van_a);
        let mut b = denomination_split(4_800, &sk, round_id, 1, van_b);
        deterministic_shuffle(&mut a, &sk, round_id, 1, van_a);
        deterministic_shuffle(&mut b, &sk, round_id, 1, van_b);
        assert_ne!(a, b, "different VANs should produce different permutations");
    }

    #[test]
    fn shuffle_actually_reorders() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let original = denomination_split(4_800, &sk, round_id, 1, van);
        let mut shuffled = original;
        deterministic_shuffle(&mut shuffled, &sk, round_id, 1, van);
        assert_ne!(
            original, shuffled,
            "shuffle should reorder (vanishingly unlikely to be identity for 12 non-zero shares)"
        );
    }
}
