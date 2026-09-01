//! Vote proof builder (ZKP #2, compact cast circuit).
//!
//! Constructs the cast vote proof from delegation key material, a vote
//! commitment tree witness, vote parameters, and the encrypt-choice bundle
//! (ZKP 1.5) carrying the vote's ciphertexts and selected commitments.
//!
//! The builder re-derives the weight shares with the same deterministic PRF
//! pipeline as the encrypt-choice builder and cross-checks the supplied
//! bundle against them, so the two proofs of a [`VoteBundle`] always witness
//! identical weights and commitments.

use std::{string::String, vec::Vec};

use crate::group::{Curve, GroupEncoding};
use voting_crypto_deps::halo2_proofs::circuit::Value;
use voting_crypto_deps::orchard::keys::{FullViewingKey, Scope, SpendAuthorizingKey, SpendingKey};
use voting_crypto_deps::pasta_curves::{
    arithmetic::{Coordinates, CurveAffine},
    pallas,
};

use super::{
    circuit::{
        van_integrity_hash, van_nullifier_hash, vote_commitment_hash_v2, Circuit, Instance,
        MAX_PROPOSAL_ID,
    },
    prove::{create_vote_proof, verify_vote_proof},
};
use crate::{
    bridge::{bridge_commitment, selected_share_commitment, NUM_SHARES},
    encrypt_choice::{self, derive_vote_shares, verify_encrypt_choice_proof, EncryptChoiceBundle},
    gadgets::elgamal::{base_to_scalar, spend_auth_g_affine},
    params::{BALLOT_DIVISOR, MAX_PROPOSAL_AUTHORITY, VOTE_COMM_TREE_DEPTH},
    shares_hash::shares_hash_from_comms,
    ProveError,
};

type PallasAffineCoordinates = Coordinates<pallas::Affine>;

/// Result of building a cast vote proof.
///
/// The vote's ciphertexts, blinds, and selected commitments live in the
/// [`EncryptChoiceBundle`] this proof was built from; this bundle carries
/// only the cast proof's own outputs.
#[derive(Debug)]
pub struct VoteProofBundle {
    /// Serialized Halo2 proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for the proof.
    pub instance: Instance,
    /// Compressed r_vpk (32 bytes) for sighash computation and signature verification.
    pub r_vpk_bytes: [u8; 32],
    /// Poseidon hash of the 16 selected commitments.
    /// This value is exported for reveal-share helpers, but it is not a Halo2
    /// public input. The vote proof binds it through `instance.vote_commitment`:
    /// vote_commitment = H(DOMAIN_VC_V2, voting_round_id, shares_hash,
    /// proposal_id, decision_bucket_count).
    pub shares_hash: pallas::Base,
}

/// A complete two-proof vote: the encrypt-choice proof (ZKP 1.5) and the
/// cast proof (ZKP #2) bound by their shared public bridge.
#[derive(Debug)]
pub struct VoteBundle {
    /// The decision-bound auxiliary proof carrying all ElGamal ciphertexts.
    pub encrypt_choice: EncryptChoiceBundle,
    /// The compact cast proof.
    pub cast: VoteProofBundle,
}

/// A bundle-level consistency failure between the two proofs' instances.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoteBundleError(pub String);

impl core::fmt::Display for VoteBundleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "inconsistent vote bundle: {}", self.0)
    }
}

impl std::error::Error for VoteBundleError {}

/// Native cross-checks binding the two instances of a vote bundle.
///
/// A verifier must replicate exactly these checks (plus authenticating
/// `ea_pk`, `voting_round_id`, `proposal_id`, and `decision_bucket_count`
/// against governance session data) before accepting the two proofs as one
/// vote. [`verify_vote_bundle`] performs them together with both proof
/// verifications.
pub fn check_vote_bundle_consistency(
    encrypt_choice: &encrypt_choice::Instance,
    cast: &Instance,
) -> Result<(), VoteBundleError> {
    if encrypt_choice.bridge != cast.bridge {
        return Err(VoteBundleError("bridge values differ".into()));
    }
    if encrypt_choice.voting_round_id != cast.voting_round_id {
        return Err(VoteBundleError("voting round ids differ".into()));
    }
    if encrypt_choice.proposal_id != cast.proposal_id {
        return Err(VoteBundleError("proposal ids differ".into()));
    }
    if encrypt_choice.decision_bucket_count != cast.decision_bucket_count {
        return Err(VoteBundleError("decision bucket counts differ".into()));
    }
    Ok(())
}

impl VoteBundle {
    /// Runs the native bundle consistency checks on the two instances.
    pub fn check_consistency(&self) -> Result<(), VoteBundleError> {
        check_vote_bundle_consistency(&self.encrypt_choice.instance, &self.cast.instance)
    }
}

/// Verifies a complete vote bundle: instance consistency plus both proofs.
///
/// The caller must still authenticate the governance-sourced instance fields
/// of both proofs; see [`verify_vote_proof`] and
/// [`crate::encrypt_choice::verify_encrypt_choice_proof`].
pub fn verify_vote_bundle(
    encrypt_choice_proof: &[u8],
    encrypt_choice_instance: &encrypt_choice::Instance,
    cast_proof: &[u8],
    cast_instance: &Instance,
) -> Result<(), String> {
    check_vote_bundle_consistency(encrypt_choice_instance, cast_instance)
        .map_err(|error| error.to_string())?;
    verify_encrypt_choice_proof(encrypt_choice_proof, encrypt_choice_instance)?;
    verify_vote_proof(cast_proof, cast_instance)
}

/// Native VAN values for one proposal-authority transition.
///
/// This helper output lets clients derive a complete ordered authority chain
/// before starting expensive proof generation. It uses the same address,
/// ballot scaling, and VAN hash implementation as [`build_vote_proof_from_delegation`]
/// and does not change the vote circuit or its public inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoteAuthorityTransition {
    /// VAN consumed by the vote proof.
    pub vote_authority_note_old: pallas::Base,
    /// VAN produced by the vote proof.
    pub vote_authority_note_new: pallas::Base,
    /// Proposal-authority bitmask witnessed by the consumed VAN.
    pub proposal_authority_old: u64,
    /// Proposal-authority bitmask after consuming `proposal_id`.
    pub proposal_authority_new: u64,
}

/// Errors that can occur during vote proof construction.
#[derive(Debug)]
pub enum VoteProofBuildError {
    /// The total note value cannot be split into valid shares.
    InvalidShares(String),
    /// The randomized voting public key is the identity point.
    InvalidRandomizedVotingPublicKey,
    /// The proposal identifier is outside the supported 1-indexed range.
    InvalidProposalId(u64),
    /// The proposal-authority bitmask exceeds the circuit maximum.
    InvalidProposalAuthority(u64),
    /// The encrypt-choice bundle does not match this vote's context.
    EncryptChoiceMismatch(String),
    /// The selected proposal's authority bit has already been consumed.
    ProposalAuthorityConsumed {
        /// Selected proposal identifier.
        proposal_id: u64,
        /// Current proposal-authority bitmask.
        proposal_authority: u64,
    },
    /// Halo2 proof creation failed.
    Prove(ProveError),
}

impl From<ProveError> for VoteProofBuildError {
    fn from(error: ProveError) -> Self {
        VoteProofBuildError::Prove(error)
    }
}

impl core::fmt::Display for VoteProofBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VoteProofBuildError::InvalidShares(msg) => {
                write!(f, "invalid shares: {}", msg)
            }
            VoteProofBuildError::InvalidRandomizedVotingPublicKey => {
                write!(f, "invalid randomized voting public key: identity point")
            }
            VoteProofBuildError::EncryptChoiceMismatch(msg) => {
                write!(f, "encrypt-choice bundle mismatch: {}", msg)
            }
            VoteProofBuildError::InvalidProposalId(proposal_id) => {
                write!(
                    f,
                    "proposal_id must be in [1, {}], got {}",
                    MAX_PROPOSAL_ID - 1,
                    proposal_id
                )
            }
            VoteProofBuildError::InvalidProposalAuthority(proposal_authority) => {
                write!(
                    f,
                    "proposal_authority must be at most {}, got {}",
                    MAX_PROPOSAL_AUTHORITY, proposal_authority
                )
            }
            VoteProofBuildError::ProposalAuthorityConsumed {
                proposal_id,
                proposal_authority,
            } => {
                write!(
                    f,
                    "proposal {} is not authorized by proposal_authority {}",
                    proposal_id, proposal_authority
                )
            }
            VoteProofBuildError::Prove(error) => {
                write!(f, "proof generation failed: {error}")
            }
        }
    }
}

impl std::error::Error for VoteProofBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VoteProofBuildError::Prove(error) => Some(error),
            _ => None,
        }
    }
}

fn pallas_coordinates(point: pallas::Affine) -> Option<PallasAffineCoordinates> {
    point.coordinates().into()
}

/// Extract the voting spending key scalar from a SpendingKey.
///
/// This replicates the sign-correction logic from `SpendAuthorizingKey::from`:
/// `ask = PRF_expand(sk)`, then negate if the resulting ak has ỹ = 1.
fn extract_vsk(sk: &SpendingKey) -> pallas::Scalar {
    let ask_raw = SpendAuthorizingKey::derive_inner(sk);
    let ak_point = (spend_auth_g_affine() * ask_raw).to_affine();
    let ak_bytes = ak_point.to_bytes();

    // If the sign bit of ak is 1, the real ask was negated.
    if (ak_bytes[31] >> 7) == 1 {
        -ask_raw
    } else {
        ask_raw
    }
}

fn next_proposal_authority(
    proposal_authority_old: u64,
    proposal_id: u64,
) -> Result<u64, VoteProofBuildError> {
    if proposal_id == 0 || proposal_id >= MAX_PROPOSAL_ID as u64 {
        return Err(VoteProofBuildError::InvalidProposalId(proposal_id));
    }
    if proposal_authority_old > MAX_PROPOSAL_AUTHORITY {
        return Err(VoteProofBuildError::InvalidProposalAuthority(
            proposal_authority_old,
        ));
    }

    let proposal_bit = 1u64 << proposal_id;
    if proposal_authority_old & proposal_bit == 0 {
        return Err(VoteProofBuildError::ProposalAuthorityConsumed {
            proposal_id,
            proposal_authority: proposal_authority_old,
        });
    }

    Ok(proposal_authority_old - proposal_bit)
}

fn derive_vote_authority_transition_from_address(
    vpk_g_d_x: pallas::Base,
    vpk_pk_d_x: pallas::Base,
    total_note_value: u64,
    van_comm_rand: pallas::Base,
    voting_round_id: pallas::Base,
    proposal_id: u64,
    proposal_authority_old: u64,
) -> Result<VoteAuthorityTransition, VoteProofBuildError> {
    let proposal_authority_new = next_proposal_authority(proposal_authority_old, proposal_id)?;
    let num_ballots = pallas::Base::from(total_note_value / BALLOT_DIVISOR);
    let old = van_integrity_hash(
        vpk_g_d_x,
        vpk_pk_d_x,
        num_ballots,
        voting_round_id,
        pallas::Base::from(proposal_authority_old),
        van_comm_rand,
    );
    let new = van_integrity_hash(
        vpk_g_d_x,
        vpk_pk_d_x,
        num_ballots,
        voting_round_id,
        pallas::Base::from(proposal_authority_new),
        van_comm_rand,
    );

    Ok(VoteAuthorityTransition {
        vote_authority_note_old: old,
        vote_authority_note_new: new,
        proposal_authority_old,
        proposal_authority_new,
    })
}

/// Derives one VAN authority transition without constructing a proof.
///
/// Clients can call this repeatedly, feeding each returned
/// `proposal_authority_new` into the next call, to plan an ordered sequence of
/// vote proofs. The returned VANs are exactly those derived by
/// [`build_vote_proof_from_delegation`] for the same inputs.
/// Pass [`crate::MAX_PROPOSAL_AUTHORITY`] as `proposal_authority_old` for the
/// first transition produced by a fresh delegation.
pub fn derive_vote_authority_transition(
    sk: &SpendingKey,
    address_index: u32,
    total_note_value: u64,
    van_comm_rand: pallas::Base,
    voting_round_id: pallas::Base,
    proposal_id: u64,
    proposal_authority_old: u64,
) -> Result<VoteAuthorityTransition, VoteProofBuildError> {
    let fvk: FullViewingKey = sk.into();
    let address = fvk.address_at(address_index, Scope::External);
    let g_d = address.g_d().to_affine();
    let pk_d = address.pk_d().inner().to_affine();
    let g_d_x = *pallas_coordinates(g_d)
        .expect("Orchard address g_d is non-identity by construction")
        .x();
    let pk_d_x = *pallas_coordinates(pk_d)
        .expect("Orchard address pk_d is non-identity by construction")
        .x();

    derive_vote_authority_transition_from_address(
        g_d_x,
        pk_d_x,
        total_note_value,
        van_comm_rand,
        voting_round_id,
        proposal_id,
        proposal_authority_old,
    )
}

/// Build a real cast vote proof (ZKP #2) from delegation key material and an
/// encrypt-choice bundle.
///
/// This function constructs the compact cast circuit, computes all public
/// inputs, and generates a Halo2 proof. The ElGamal ciphertexts themselves
/// are proven by the supplied [`EncryptChoiceBundle`] (ZKP 1.5); this proof
/// re-opens its bridge commitment, binding both proofs to the same weights
/// and selected commitments.
///
/// # Arguments
///
/// * `sk` - The SpendingKey used during delegation (ZKP #1).
/// * `address_index` - The diversifier index of the output recipient
///   address used in delegation (typically 1).
/// * `total_note_value` - Sum of delegated note values in raw zatoshi (e.g. 15_000_000).
///   Internally converted to ballot count via floor-division by BALLOT_DIVISOR
///   (the delegation circuit's condition 8 constrains this relation; see the
///   delegation README §8 for the precise proven statement).
/// * `van_comm_rand` - The blinding factor used for the VAN in delegation.
/// * `voting_round_id` - The active governance round identifier (Pallas base
///   field element). The caller must authenticate it from the round
///   announcement.
/// * `vote_comm_tree_path` - Merkle authentication path (24 siblings) for
///   the VAN in the vote commitment tree.
/// * `vote_comm_tree_position` - Leaf position of the VAN in the tree.
/// * `anchor_height` - Caller-authenticated chain height used by the verifier
///   or chain to source the vote commitment tree root. The circuit carries this
///   as a public input but does not derive or constrain the height itself.
/// * `proposal_id` - Which proposal to vote on (1-indexed, must be in [1, 50]).
///   The builder checks only this circuit-supported range; the caller must
///   ensure the proposal is active for `voting_round_id`.
/// * `alpha_v` - Spend auth randomizer for the voting hotkey. The caller
///   retains this to sign the sighash with `rsk_v = ask_v.randomize(&alpha_v)`.
/// * `proposal_authority_old_u64` - The authority bitmask of the consumed VAN.
/// * `encrypt_choice` - The encrypt-choice bundle previously built by
///   [`crate::encrypt_choice::build_encrypt_choice`] for the **same**
///   `(sk, total_note_value, VAN, voting_round_id, proposal_id)` context and
///   share layout. The builder re-derives the shares and the bridge from its
///   own inputs and rejects a bundle that does not match, so a stale or
///   cross-context bundle cannot produce an inconsistent vote.
///
/// # Caller contract
///
/// `alpha_v` is a secret, one-time spend-auth randomizer and MUST be drawn
/// from a CSPRNG such as `OsRng` for each vote proof. `van_comm_rand` is the
/// secret VAN commitment blinding factor originally used by
/// `delegation::build_delegation_bundle`; pass the retained value unchanged.
/// `voting_round_id`, `anchor_height`, `proposal_id`, and the vote commitment
/// tree witness are authenticated session parameters: the builder constrains
/// proofs to the supplied values but cannot prove they came from the intended
/// chain state or governance announcement.
///
/// **Expensive**: K=11 proof generation should run in release mode; it is the
/// interactive step of a vote, after the encrypt-choice proof has already
/// been produced in the background.
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
    alpha_v: pallas::Scalar,
    proposal_authority_old_u64: u64,
    encrypt_choice: &EncryptChoiceBundle,
) -> Result<VoteProofBundle, VoteProofBuildError> {
    let proposal_authority_new_u64 =
        next_proposal_authority(proposal_authority_old_u64, proposal_id)?;

    // ---- Key derivation (matches delegation's key hierarchy) ----

    let vsk = extract_vsk(sk);
    let fvk: FullViewingKey = sk.into();
    let vsk_nk = fvk.nk().inner();
    let rivk_v = fvk.rivk(Scope::External).inner();

    let address = fvk.address_at(address_index, Scope::External);
    let vpk_g_d = address.g_d();
    let vpk_pk_d = address.pk_d().inner();
    let vpk_g_d_affine = vpk_g_d.to_affine();
    let vpk_pk_d_affine = vpk_pk_d.to_affine();

    let vpk_g_d_coords = pallas_coordinates(vpk_g_d_affine)
        .expect("orchard address g_d is non-identity by construction");
    let vpk_pk_d_coords = pallas_coordinates(vpk_pk_d_affine)
        .expect("orchard address pk_d is non-identity by construction");
    let vpk_g_d_x = *vpk_g_d_coords.x();
    let vpk_pk_d_x = *vpk_pk_d_coords.x();

    // ---- Fast key-chain consistency checks (instant, no circuit) ----
    {
        use crate::ff::PrimeFieldBits;
        use core::iter;
        use voting_crypto_deps::halo2_gadgets::sinsemilla::primitives::CommitDomain;
        use voting_crypto_deps::orchard::constants::{
            fixed_bases::COMMIT_IVK_PERSONALIZATION, L_ORCHARD_BASE,
        };

        // Check 1: [vsk] * SpendAuthG must match the ak from the FullViewingKey.
        let ak_from_vsk = (spend_auth_g_affine() * vsk).to_affine();
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
        let ak_from_vsk_coords = pallas_coordinates(ak_from_vsk)
            .expect("valid Orchard spending keys have nonzero spend authorizing keys");
        let ak_x = *ak_from_vsk_coords.x();
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
        let pk_d_derived = (*vpk_g_d * ivk_scalar).to_affine();
        assert_eq!(
            pk_d_derived, vpk_pk_d_affine,
            "CommitIvk chain mismatch: [ivk]*g_d != pk_d from address"
        );
    }

    // ---- Proposal authority ----

    let proposal_authority_old = pallas::Base::from(proposal_authority_old_u64);
    let one_shifted = pallas::Base::from(1u64 << proposal_id);

    // ---- Ballot scaling (must match ZKP #1's BALLOT_DIVISOR) ----

    let num_ballots = total_note_value / BALLOT_DIVISOR;
    let num_ballots_base = pallas::Base::from(num_ballots);

    // ---- VAN integrity hashes ----
    // The VAN commitment hashes num_ballots (not raw zatoshi), matching
    // the delegation circuit (ZKP #1 condition 7).

    let transition = derive_vote_authority_transition_from_address(
        vpk_g_d_x,
        vpk_pk_d_x,
        total_note_value,
        van_comm_rand,
        voting_round_id,
        proposal_id,
        proposal_authority_old_u64,
    )?;
    debug_assert_eq!(
        transition.proposal_authority_new,
        proposal_authority_new_u64
    );
    let vote_authority_note_old = transition.vote_authority_note_old;

    let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);

    let vote_authority_note_new = transition.vote_authority_note_new;

    // ---- Cross-check the encrypt-choice bundle ----
    //
    // Re-derive the shares and commitment blinds this VAN authorizes (both
    // possible layouts), re-open every selected commitment against the
    // bundle's reveal data, and recompute the bridge. Checking the
    // VAN-derived blinds is necessary in single-share mode, where equal
    // weights alone cannot distinguish bundles prepared for different VANs.

    let single_share_layout = {
        let mut layout = None;
        for single_share in [false, true] {
            let candidate = derive_vote_shares(
                sk,
                num_ballots,
                voting_round_id,
                proposal_id,
                vote_authority_note_old,
                single_share,
            )
            .map_err(VoteProofBuildError::InvalidShares)?;
            if candidate == encrypt_choice.shares {
                layout = Some(single_share);
                break;
            }
        }
        layout.ok_or_else(|| {
            VoteProofBuildError::EncryptChoiceMismatch(
                "bundle shares do not match this vote's derived shares".into(),
            )
        })?
    };
    let _ = single_share_layout;

    let shares_u64 = encrypt_choice.shares;
    let shares_base: [pallas::Base; NUM_SHARES] =
        core::array::from_fn(|i| pallas::Base::from(shares_u64[i]));

    let expected_share_blinds: [pallas::Base; NUM_SHARES] = core::array::from_fn(|i| {
        crate::vote_prf::derive_share_blind(
            sk,
            voting_round_id,
            proposal_id,
            vote_authority_note_old,
            i as u8,
        )
    });
    if encrypt_choice.share_blinds != expected_share_blinds {
        return Err(VoteProofBuildError::EncryptChoiceMismatch(
            "bundle commitment blinds do not match this vote's VAN".into(),
        ));
    }
    for (i, encrypted_share) in encrypt_choice.encrypted_shares.iter().enumerate() {
        if encrypted_share.share_index != i as u32
            || encrypted_share.plaintext_value != shares_u64[i]
        {
            return Err(VoteProofBuildError::EncryptChoiceMismatch(format!(
                "bundle reveal data does not match share {i}"
            )));
        }
        let expected_commitment =
            selected_share_commitment(expected_share_blinds[i], &encrypted_share.ciphertexts);
        if encrypt_choice.selected_commitments[i] != expected_commitment {
            return Err(VoteProofBuildError::EncryptChoiceMismatch(format!(
                "bundle selected commitment {i} does not open to its reveal data"
            )));
        }
    }

    let proposal_id_base = pallas::Base::from(proposal_id);
    let decision_bucket_count_base = pallas::Base::from(encrypt_choice.decision_bucket_count);
    let weights_and_comms: [(pallas::Base, pallas::Base); NUM_SHARES] =
        core::array::from_fn(|i| (shares_base[i], encrypt_choice.selected_commitments[i]));
    let expected_bridge = bridge_commitment(
        voting_round_id,
        proposal_id_base,
        decision_bucket_count_base,
        &weights_and_comms,
    );
    if expected_bridge != encrypt_choice.bridge
        || encrypt_choice.instance.bridge != encrypt_choice.bridge
        || encrypt_choice.instance.voting_round_id != voting_round_id
        || encrypt_choice.instance.proposal_id != proposal_id_base
        || encrypt_choice.instance.decision_bucket_count != decision_bucket_count_base
    {
        return Err(VoteProofBuildError::EncryptChoiceMismatch(
            "bundle bridge or context does not match this vote".into(),
        ));
    }

    // ---- shares_hash and vote commitment (v2) ----

    let shares_hash_val = shares_hash_from_comms(encrypt_choice.selected_commitments);

    // ---- Condition 4: r_vpk = ak + [alpha_v] * G = [vsk + alpha_v] * G ----
    // alpha_v is provided by the caller so they can sign with rsk_v.
    let r_vpk = (spend_auth_g_affine() * (vsk + alpha_v)).to_affine();
    let r_vpk_coords =
        pallas_coordinates(r_vpk).ok_or(VoteProofBuildError::InvalidRandomizedVotingPublicKey)?;
    let r_vpk_x = *r_vpk_coords.x();
    let r_vpk_y = *r_vpk_coords.y();
    let r_vpk_bytes: [u8; 32] = r_vpk.to_bytes();

    let vote_commitment = vote_commitment_hash_v2(
        voting_round_id,
        shares_hash_val,
        proposal_id_base,
        decision_bucket_count_base,
    );

    // ---- Vote commitment tree root (from auth path) ----
    // Recompute the root from the leaf + auth path to set as public input.

    let vote_comm_tree_root = {
        use crate::protocol_hash::poseidon_hash_2;

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
    circuit.selected_commitments = encrypt_choice.selected_commitments.map(Value::known);

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
        encrypt_choice.bridge,
        decision_bucket_count_base,
    );

    // ---- Generate proof ----

    let proof = create_vote_proof(circuit, &instance)?;

    Ok(VoteProofBundle {
        proof,
        instance,
        r_vpk_bytes,
        shares_hash: shares_hash_val,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::Field;
    use crate::params::SHARE_VALUE_LIMIT;
    use crate::vote_prf::{
        denomination_split, derive_share_blind, derive_weighted_share_randomness,
        deterministic_shuffle, vote_share_prf,
    };

    fn test_sk() -> SpendingKey {
        SpendingKey::from_bytes([0x42; 32]).expect("valid spending key")
    }

    fn test_round_id() -> pallas::Base {
        pallas::Base::from(0xCAFE_u64)
    }

    fn test_van() -> pallas::Base {
        pallas::Base::from(0xDEAD_u64)
    }

    /// Builds a consistent (but unproven) encrypt-choice bundle for builder
    /// tests: shares and blinds derived exactly as the cast builder re-derives
    /// them, selected commitments over synthetic ciphertexts, and a matching
    /// bridge and instance.
    fn fake_encrypt_choice_bundle(
        sk: &SpendingKey,
        total_note_value: u64,
        vote_authority_note_old: pallas::Base,
        voting_round_id: pallas::Base,
        proposal_id: u64,
        decision_bucket_count: u64,
        single_share: bool,
    ) -> EncryptChoiceBundle {
        use crate::encrypt_choice::{ElGamalCiphertextBytes, EncryptedWeightedShareOutput};
        use crate::{CiphertextCoordinates, WeightedShareCiphertexts, MAX_DECISION_BUCKETS};

        let shares = derive_vote_shares(
            sk,
            total_note_value / BALLOT_DIVISOR,
            voting_round_id,
            proposal_id,
            vote_authority_note_old,
            single_share,
        )
        .expect("test shares should be valid");
        let zero_coords = CiphertextCoordinates {
            c1_x: pallas::Base::zero(),
            c2_x: pallas::Base::zero(),
            c1_y: pallas::Base::zero(),
            c2_y: pallas::Base::zero(),
        };
        let share_blinds: [pallas::Base; NUM_SHARES] = core::array::from_fn(|i| {
            derive_share_blind(
                sk,
                voting_round_id,
                proposal_id,
                vote_authority_note_old,
                i as u8,
            )
        });
        let encrypted_shares: [EncryptedWeightedShareOutput; NUM_SHARES] =
            core::array::from_fn(|i| EncryptedWeightedShareOutput {
                share_index: i as u32,
                plaintext_value: shares[i],
                ciphertexts: WeightedShareCiphertexts([zero_coords; MAX_DECISION_BUCKETS]),
                compressed: [ElGamalCiphertextBytes {
                    c1: [0u8; 32],
                    c2: [0u8; 32],
                }; MAX_DECISION_BUCKETS],
                randomness: [[0u8; 32]; MAX_DECISION_BUCKETS],
            });
        let selected_commitments: [pallas::Base; NUM_SHARES] = core::array::from_fn(|i| {
            selected_share_commitment(share_blinds[i], &encrypted_shares[i].ciphertexts)
        });
        let weights_and_comms: [(pallas::Base, pallas::Base); NUM_SHARES] =
            core::array::from_fn(|i| (pallas::Base::from(shares[i]), selected_commitments[i]));
        let bridge = bridge_commitment(
            voting_round_id,
            pallas::Base::from(proposal_id),
            pallas::Base::from(decision_bucket_count),
            &weights_and_comms,
        );
        EncryptChoiceBundle {
            proof: Vec::new(),
            instance: encrypt_choice::Instance::from_parts(
                pallas::Base::zero(),
                pallas::Base::zero(),
                bridge,
                pallas::Base::from(decision_bucket_count),
                voting_round_id,
                pallas::Base::from(proposal_id),
            ),
            bridge,
            shares,
            share_blinds,
            selected_commitments,
            encrypted_shares,
            decision_bucket_count,
        }
    }

    #[test]
    fn vote_share_prf_has_frozen_test_vector() {
        let hash = vote_share_prf(
            &test_sk(),
            crate::domain_tags::VOTE_PRF_DOMAIN_ELGAMAL,
            test_round_id(),
            1,
            test_van(),
            0,
        );

        assert_eq!(
            hash,
            [
                0x62, 0x03, 0x29, 0x9b, 0x2c, 0x58, 0x4b, 0xa6, 0x37, 0x4d, 0xbe, 0xd6, 0x45, 0x71,
                0x6f, 0x03, 0x31, 0x56, 0x95, 0x6f, 0xf1, 0x88, 0x8e, 0x75, 0x41, 0x43, 0xb1, 0xf5,
                0x54, 0xea, 0xb5, 0xb0, 0x6b, 0xdf, 0x7d, 0xca, 0xd4, 0x5a, 0xc2, 0xf4, 0xb9, 0x6a,
                0xe4, 0x5b, 0xb9, 0x98, 0xd0, 0x5b, 0x4a, 0x8f, 0x12, 0x49, 0x52, 0xb3, 0x0b, 0x19,
                0xc1, 0xaf, 0x89, 0x35, 0x8a, 0x96, 0xe0, 0x2c,
            ]
        );
    }

    #[test]
    fn build_vote_proof_rejects_invalid_proposal_id() {
        let sk = test_sk();

        for proposal_id in [0, MAX_PROPOSAL_ID as u64, 64] {
            let bundle = fake_encrypt_choice_bundle(
                &sk,
                BALLOT_DIVISOR,
                test_van(),
                test_round_id(),
                proposal_id,
                4,
                true,
            );
            let err = build_vote_proof_from_delegation(
                &sk,
                1,
                BALLOT_DIVISOR,
                test_van(),
                test_round_id(),
                [pallas::Base::from(0u64); VOTE_COMM_TREE_DEPTH],
                0,
                123,
                proposal_id,
                pallas::Scalar::from(7u64),
                65535,
                &bundle,
            )
            .expect_err("invalid proposal_id should be rejected before proof generation");

            assert!(matches!(
                err,
                VoteProofBuildError::InvalidProposalId(rejected) if rejected == proposal_id
            ));
        }
    }

    #[test]
    fn authority_transition_derives_an_ordered_van_chain() {
        let sk = test_sk();
        let first = derive_vote_authority_transition(
            &sk,
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            1,
            65535,
        )
        .expect("first transition should be valid");
        let second = derive_vote_authority_transition(
            &sk,
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            2,
            first.proposal_authority_new,
        )
        .expect("second transition should be valid");

        assert_eq!(first.proposal_authority_new, 65533);
        assert_eq!(second.proposal_authority_old, first.proposal_authority_new);
        assert_eq!(second.proposal_authority_new, 65529);
        assert_eq!(
            second.vote_authority_note_old,
            first.vote_authority_note_new
        );
    }

    #[test]
    fn authority_transition_accepts_maximum_proposal_id() {
        let transition = derive_vote_authority_transition(
            &test_sk(),
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            50,
            MAX_PROPOSAL_AUTHORITY,
        )
        .expect("proposal 50 should consume the highest usable authority bit");

        assert_eq!(transition.proposal_authority_old, MAX_PROPOSAL_AUTHORITY);
        assert_eq!(
            transition.proposal_authority_new,
            MAX_PROPOSAL_AUTHORITY - (1 << 50)
        );
    }

    #[test]
    fn authority_transition_rejects_invalid_or_consumed_authority() {
        let sk = test_sk();
        let first_invalid = MAX_PROPOSAL_AUTHORITY + 1;
        let out_of_range = derive_vote_authority_transition(
            &sk,
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            1,
            first_invalid,
        )
        .expect_err("authority must fit the circuit range");
        assert!(matches!(
            out_of_range,
            VoteProofBuildError::InvalidProposalAuthority(rejected)
                if rejected == first_invalid
        ));

        let consumed = derive_vote_authority_transition(
            &sk,
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            2,
            65531,
        )
        .expect_err("proposal 2 bit is clear");
        assert!(matches!(
            consumed,
            VoteProofBuildError::ProposalAuthorityConsumed {
                proposal_id: 2,
                proposal_authority: 65531
            }
        ));
    }

    #[test]
    fn build_vote_proof_rejects_mismatched_encrypt_choice_bundle() {
        let sk = test_sk();

        // Bundle built for a different VAN: derived shares differ.
        let bundle = fake_encrypt_choice_bundle(
            &sk,
            BALLOT_DIVISOR * 100,
            pallas::Base::from(0xBEEF_u64),
            test_round_id(),
            1,
            4,
            false,
        );
        let err = build_vote_proof_from_delegation(
            &sk,
            1,
            BALLOT_DIVISOR * 101,
            test_van(),
            test_round_id(),
            [pallas::Base::from(0u64); VOTE_COMM_TREE_DEPTH],
            0,
            123,
            1,
            pallas::Scalar::from(7u64),
            65535,
            &bundle,
        )
        .expect_err("bundle with mismatched shares must be rejected");
        assert!(matches!(err, VoteProofBuildError::EncryptChoiceMismatch(_)));
    }

    #[test]
    fn build_vote_proof_rejects_single_share_bundle_from_another_van() {
        let sk = test_sk();
        let total_note_value = BALLOT_DIVISOR * 100;
        let current = derive_vote_authority_transition(
            &sk,
            1,
            total_note_value,
            test_van(),
            test_round_id(),
            1,
            65535,
        )
        .expect("current transition should be valid");
        let other = derive_vote_authority_transition(
            &sk,
            1,
            total_note_value,
            pallas::Base::from(0xBEEF_u64),
            test_round_id(),
            1,
            65535,
        )
        .expect("other transition should be valid");
        assert_ne!(
            current.vote_authority_note_old,
            other.vote_authority_note_old
        );

        let bundle = fake_encrypt_choice_bundle(
            &sk,
            total_note_value,
            other.vote_authority_note_old,
            test_round_id(),
            1,
            4,
            true,
        );

        // Single-share layouts have identical weights across VANs, and the
        // bridge itself does not contain the VAN. The VAN-derived commitment
        // blind check must therefore be what rejects this stale bundle.
        let current_shares = derive_vote_shares(
            &sk,
            total_note_value / BALLOT_DIVISOR,
            test_round_id(),
            1,
            current.vote_authority_note_old,
            true,
        )
        .expect("current shares should be valid");
        assert_eq!(bundle.shares, current_shares);
        let current_weights_and_comms: [(pallas::Base, pallas::Base); NUM_SHARES] =
            core::array::from_fn(|i| {
                (
                    pallas::Base::from(current_shares[i]),
                    bundle.selected_commitments[i],
                )
            });
        assert_eq!(
            bundle.bridge,
            bridge_commitment(
                test_round_id(),
                pallas::Base::from(1u64),
                pallas::Base::from(4u64),
                &current_weights_and_comms,
            )
        );

        let err = build_vote_proof_from_delegation(
            &sk,
            1,
            total_note_value,
            test_van(),
            test_round_id(),
            [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH],
            0,
            123,
            1,
            pallas::Scalar::from(7u64),
            65535,
            &bundle,
        )
        .expect_err("a single-share bundle from another VAN must be rejected");

        assert!(matches!(
            err,
            VoteProofBuildError::EncryptChoiceMismatch(message)
                if message.contains("commitment blinds")
        ));
    }

    #[test]
    fn build_vote_proof_rejects_tampered_bundle_bridge() {
        let sk = test_sk();
        let transition = derive_vote_authority_transition(
            &sk,
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            1,
            65535,
        )
        .expect("valid transition");
        let mut bundle = fake_encrypt_choice_bundle(
            &sk,
            BALLOT_DIVISOR,
            transition.vote_authority_note_old,
            test_round_id(),
            1,
            4,
            true,
        );
        // Tamper with one selected commitment after the bridge was computed.
        bundle.selected_commitments[0] += pallas::Base::one();

        let err = build_vote_proof_from_delegation(
            &sk,
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            [pallas::Base::from(0u64); VOTE_COMM_TREE_DEPTH],
            0,
            123,
            1,
            pallas::Scalar::from(7u64),
            65535,
            &bundle,
        )
        .expect_err("tampered bundle bridge must be rejected");
        assert!(matches!(err, VoteProofBuildError::EncryptChoiceMismatch(_)));
    }

    #[test]
    fn build_vote_proof_rejects_identity_r_vpk() {
        let sk = test_sk();
        let transition = derive_vote_authority_transition(
            &sk,
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            1,
            65535,
        )
        .expect("valid transition");
        let bundle = fake_encrypt_choice_bundle(
            &sk,
            BALLOT_DIVISOR,
            transition.vote_authority_note_old,
            test_round_id(),
            1,
            4,
            true,
        );
        let err = build_vote_proof_from_delegation(
            &sk,
            1,
            BALLOT_DIVISOR,
            test_van(),
            test_round_id(),
            [pallas::Base::from(0u64); VOTE_COMM_TREE_DEPTH],
            0,
            123,
            1,
            -extract_vsk(&sk),
            65535,
            &bundle,
        )
        .expect_err("alpha_v = -vsk should make r_vpk the identity");

        assert!(matches!(
            err,
            VoteProofBuildError::InvalidRandomizedVotingPublicKey
        ));
    }

    #[test]
    fn different_share_layout_gives_different_c1() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        for i in 0..16u8 {
            let standard_r = derive_weighted_share_randomness(&sk, round_id, 1, van, i, 0, false);
            let single_r = derive_weighted_share_randomness(&sk, round_id, 1, van, i, 0, true);
            let standard_c1 = spend_auth_g_affine()
                * base_to_scalar(standard_r).expect("standard randomness must be scalar-range");
            let single_c1 = spend_auth_g_affine()
                * base_to_scalar(single_r).expect("single-share randomness must be scalar-range");
            assert_ne!(
                standard_c1, single_c1,
                "share {i} must use a different C1 across layouts"
            );
        }
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
    fn derive_weighted_share_randomness_is_nonzero_valid_scalar() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        for i in 0..16u8 {
            let r = derive_weighted_share_randomness(&sk, round_id, 1, van, i, 3, false);
            assert!(
                bool::from(!r.is_zero()),
                "r_{} must be non-zero for the circuit hardening gate",
                i
            );
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
        let r0 = derive_weighted_share_randomness(&sk, round_id, 1, van, 0, 0, false);
        let r1 = derive_weighted_share_randomness(&sk, round_id, 1, van, 1, 0, false);
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
        let r_p1 = derive_weighted_share_randomness(&sk, round_id, 1, van, 0, 0, false);
        let r_p2 = derive_weighted_share_randomness(&sk, round_id, 2, van, 0, 0, false);
        assert_ne!(r_p1, r_p2);
    }

    #[test]
    fn different_round_id_gives_different_values() {
        let sk = test_sk();
        let van = test_van();
        let r_a =
            derive_weighted_share_randomness(&sk, pallas::Base::from(1u64), 1, van, 0, 0, false);
        let r_b =
            derive_weighted_share_randomness(&sk, pallas::Base::from(2u64), 1, van, 0, 0, false);
        assert_ne!(r_a, r_b);
    }

    #[test]
    fn randomness_and_blind_differ_for_same_inputs() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let r = derive_weighted_share_randomness(&sk, round_id, 1, van, 0, 0, false);
        let b = derive_share_blind(&sk, round_id, 1, van, 0);
        assert_ne!(r, b, "domain separation must prevent r == blind");
    }

    #[test]
    fn all_16_shares_are_distinct() {
        let sk = test_sk();
        let round_id = test_round_id();
        let van = test_van();
        let randoms: Vec<_> = (0..16u8)
            .map(|i| derive_weighted_share_randomness(&sk, round_id, 1, van, i, 0, false))
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
            let r_a = derive_weighted_share_randomness(&sk, round_id, 1, van_a, i, 0, false);
            let r_b = derive_weighted_share_randomness(&sk, round_id, 1, van_b, i, 0, false);
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
                    s < SHARE_VALUE_LIMIT,
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

    #[test]
    fn prove_error_maps_into_build_error() {
        let err = VoteProofBuildError::from(ProveError::Halo2(
            voting_crypto_deps::halo2_proofs::plonk::Error::Synthesis,
        ));

        assert!(matches!(err, VoteProofBuildError::Prove(_)));
    }
}
