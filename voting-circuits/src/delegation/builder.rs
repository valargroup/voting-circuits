//! Multi-note delegation bundle builder.
//!
//! Orchestrates the creation of a complete delegation proof:
//! a single merged circuit proving all 15 conditions for up to
//! `circuit::MAX_REAL_NOTES` notes.
//! Handles padding unused note slots with zero-value notes that still carry
//! valid IMT non-membership proofs against the real tree root.

use ff::Field;
use group::{Curve, Group, GroupEncoding};
use halo2_proofs::circuit::Value;
use pasta_curves::{
    arithmetic::{CurveAffine, CurveExt},
    pallas,
};
use rand::RngCore;
use std::vec::Vec;

use orchard::{
    keys::{FullViewingKey, NullifierDerivingKey, Scope, SpendValidatingKey},
    note::{
        commitment::ExtractedNoteCommitment, nullifier::Nullifier, Note, NoteCommitment,
        RandomSeed, Rho,
    },
    spec::NonIdentityPallasPoint,
    tree::MerklePath,
    value::NoteValue,
};

use super::{
    circuit::{self, rho_binding_hash, van_commitment_hash, NoteSlotWitness},
    imt::{derive_nullifier_domain, gov_null_hash, ImtProofData, ImtProvider},
};

// Hash-to-curve personalization for synthetic padding `g_d_pad` points.
//
// Domain-separated from Orchard's `KEY_DIVERSIFICATION_PERSONALIZATION`
// (`"z.cash:Orchard-gd"`) so that `g_d_pad = hash_to_curve(PADDING_PERSONALIZATION)(...)`
// is not generated through Orchard diversifier selection. Any accidental point
// collision is treated as a negligible hash-to-curve collision rather than a
// burned diversifier-index overlap.
pub(crate) const PADDING_PERSONALIZATION: &str = "shielded-vote/padding-v1";

/// Rho and rseed for a single padded note, captured during Phase 1 (PCZT construction).
#[derive(Clone, Debug)]
pub struct PaddedNoteData {
    /// Rho bytes (32 bytes, LE encoding of pallas::Base).
    pub rho: [u8; 32],
    /// Random seed bytes (32 bytes).
    pub rseed: [u8; 32],
}

/// Randomness captured during Phase 1 (PCZT construction) that must be reused
/// in Phase 2 (ZK proving) so the prover commits to the same nf_signed/cmx_new
/// that the signer committed to via the ZIP-244 sighash.
#[derive(Clone, Debug)]
pub struct PrecomputedRandomness {
    /// Rho + rseed for each padded note (0–4 entries).
    /// Padding note addresses are synthesized during proof building.
    pub padded_notes: Vec<PaddedNoteData>,
    /// Rseed for the signed (keystone) note.
    pub rseed_signed: [u8; 32],
    /// Rseed for the output note.
    pub rseed_output: [u8; 32],
}

/// Which precomputed note input failed validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecomputedRandomnessLocation {
    /// A padded note entry by index in `PrecomputedRandomness::padded_notes`.
    PaddedNote(usize),
    /// The synthetic signed note.
    SignedNote,
    /// The output note.
    OutputNote,
}

/// Input for a single real note in the delegation.
#[derive(Debug)]
pub struct RealNoteInput {
    /// The note being delegated.
    pub note: Note,
    /// The note's full viewing key.
    pub fvk: FullViewingKey,
    /// Merkle authentication path for the note commitment.
    pub merkle_path: MerklePath,
    /// IMT non-membership proof for this note's nullifier.
    ///
    /// This must satisfy [`ImtProofData`]'s tree contract and authenticate to
    /// the same nullifier IMT root used for the bundle.
    pub imt_proof: ImtProofData,
    /// Whether this note uses the internal (change) or external scope.
    pub scope: Scope,
}

/// Complete delegation bundle: a single circuit proving all 15 conditions.
#[derive(Debug)]
pub struct DelegationBundle {
    /// The merged delegation circuit.
    pub circuit: circuit::Circuit,
    /// Public inputs (14 field elements).
    pub instance: circuit::Instance,
}

// Wraps a `pallas::Point` as `NonIdentityPallasPoint` after asserting non-identity.
//
// Synthesized padding points come from `hash_to_curve` and scalar multiplication
// by `ivk`; both are cryptographically non-identity (identity from `hash_to_curve`
// is negligible, and `ivk = 0` is already rejected by `CommitIvk`'s ⊥ branch upstream).
// We assert here so the invariant fails at the construction site rather than
// silently flowing into the witness — the in-circuit `NonIdentityPoint::new`
// would catch identity at proof time, but a build-time assert is cheaper to debug.
fn assert_non_identity(point: pallas::Point) -> NonIdentityPallasPoint {
    assert!(
        !bool::from(point.is_identity()),
        "padding point must not be the identity"
    );
    NonIdentityPallasPoint::from_bytes(&point.to_affine().to_bytes())
        .expect("non-identity point round-trips through canonical encoding")
}

// Derives the synthetic `(g_d_pad, pk_d_pad)` pair for padding slot `slot_index`.
// `g_d_pad` is domain-separated from Orchard's `DiversifyHash`, so the pair is
// intentionally not a valid `orchard::Address`. `pk_d_pad = [ivk] * g_d_pad`
// satisfies condition 11 by construction (callers must pass the external ivk,
// since padding pins `is_internal = false`). Both points are wrapped as
// `NonIdentityPallasPoint` via `assert_non_identity` so the invariant fails at
// the builder rather than at proof time.
pub(crate) fn padding_points(
    slot_index: usize,
    ivk: pallas::Scalar,
) -> (NonIdentityPallasPoint, NonIdentityPallasPoint) {
    let slot_index = u32::try_from(slot_index).expect("padding slot index fits in u32");
    let g_d_pad = pallas::Point::hash_to_curve(PADDING_PERSONALIZATION)(&slot_index.to_le_bytes());
    let pk_d_pad = g_d_pad * ivk;
    (assert_non_identity(g_d_pad), assert_non_identity(pk_d_pad))
}

// Generates a random seed for a given rho.
// The random seed is used to derive the psi and rcm for the note.
// Note: Orchard's RandomSeed::random is not exposed. If it was exposed,
// we could use it here instead of sampling a random seed.
fn random_seed_for_rho(rho: &Rho, rng: &mut impl RngCore) -> RandomSeed {
    loop {
        let mut rseed = [0u8; 32];
        rng.fill_bytes(&mut rseed);
        let rseed = RandomSeed::from_bytes(rseed, rho);
        if bool::from(rseed.is_some()) {
            return rseed.unwrap();
        }
    }
}

// A single padding note slot in the delegation. Fields are `pub(crate)` so
// `delegation::circuit` tests can drive the same source-of-truth padding
// construction the production builder uses.
pub(crate) struct PaddingSlot {
    pub(crate) witness: NoteSlotWitness,
    pub(crate) cmx: pallas::Base,
    pub(crate) v_raw: u64,
    pub(crate) gov_null: pallas::Base,
    #[cfg(test)]
    pub(crate) real_nf: pallas::Base,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_padding_slot(
    slot_index: usize,
    pad_idx: usize,
    nk: &NullifierDerivingKey,
    dom: pallas::Base,
    ivk: pallas::Scalar,
    imt_provider: &impl ImtProvider,
    rng: &mut impl RngCore,
    precomputed: Option<&PrecomputedRandomness>,
) -> Result<PaddingSlot, DelegationBuildError> {
    let (g_d_pad, pk_d_pad) = padding_points(slot_index, ivk);

    let (rho, rseed) = if let Some(pre) = precomputed {
        // Reuse randomness so the prover commits to the same values.
        if pad_idx >= pre.padded_notes.len() {
            return Err(DelegationBuildError::MissingPrecomputedPaddedNote {
                index: pad_idx,
                actual: pre.padded_notes.len(),
            });
        }
        let pd = &pre.padded_notes[pad_idx];
        let rho = Rho::from_bytes(&pd.rho)
            .into_option()
            .ok_or(DelegationBuildError::InvalidPrecomputedRho { index: pad_idx })?;
        let location = PrecomputedRandomnessLocation::PaddedNote(pad_idx);
        let rseed = RandomSeed::from_bytes(pd.rseed, &rho)
            .into_option()
            .ok_or(DelegationBuildError::InvalidPrecomputedRseed { location })?;
        (rho, rseed)
    } else {
        let rho = Rho::from_nf_old(Nullifier::from_inner(pallas::Base::random(&mut *rng)));
        let rseed = random_seed_for_rho(&rho, &mut *rng);
        (rho, rseed)
    };

    let psi = rseed.psi(&rho);
    let rcm = rseed.rcm(&rho);
    let cm = NoteCommitment::derive(
        g_d_pad.to_affine().to_bytes(),
        pk_d_pad.to_affine().to_bytes(),
        NoteValue::ZERO,
        rho.into_inner(),
        psi,
        rcm.clone(),
    )
    .expect("padding note commitment must not be bottom");
    let cm_point = cm.inner();
    let cmx = ExtractedNoteCommitment::from(cm.clone()).inner();

    let real_nf = Nullifier::derive(nk, rho.into_inner(), psi, cm).inner();
    let gov_null = gov_null_hash(nk.inner(), dom, real_nf);
    let imt_proof = imt_provider.non_membership_proof(real_nf)?;

    // Merkle path is unconstrained for zero-value padding because condition 10
    // is gated by v=0; IMT non-membership and address ownership still run.
    let merkle_path = MerklePath::dummy(&mut *rng);
    let witness = NoteSlotWitness {
        g_d: Value::known(g_d_pad),
        pk_d: Value::known(pk_d_pad),
        v: Value::known(NoteValue::ZERO),
        rho: Value::known(rho.into_inner()),
        psi: Value::known(psi),
        rcm: Value::known(rcm),
        cm: Value::known(cm_point),
        path: Value::known(merkle_path.auth_path()),
        pos: Value::known(merkle_path.position()),
        imt_nf_bounds: Value::known(imt_proof.nf_bounds),
        imt_leaf_pos: Value::known(imt_proof.leaf_pos),
        imt_path: Value::known(imt_proof.path),
        is_internal: Value::known(false),
    };

    Ok(PaddingSlot {
        witness,
        cmx,
        v_raw: 0,
        gov_null,
        #[cfg(test)]
        real_nf,
    })
}

#[cfg(test)]
pub(crate) struct PaddingSlotForTesting {
    pub witness: NoteSlotWitness,
    pub cmx: pallas::Base,
    pub gov_null: pallas::Base,
    pub real_nf: pallas::Base,
}

#[cfg(test)]
pub(crate) fn build_padding_slot_for_testing(
    slot_index: usize,
    pad_idx: usize,
    fvk: &FullViewingKey,
    _ak: &SpendValidatingKey,
    dom: pallas::Base,
    imt_provider: &impl ImtProvider,
    rng: &mut impl RngCore,
) -> Result<PaddingSlotForTesting, DelegationBuildError> {
    let padding = build_padding_slot(
        slot_index,
        pad_idx,
        fvk.nk(),
        dom,
        fvk.ivk_scalar(Scope::External),
        imt_provider,
        rng,
        None,
    )?;

    Ok(PaddingSlotForTesting {
        witness: padding.witness,
        cmx: padding.cmx,
        gov_null: padding.gov_null,
        real_nf: padding.real_nf,
    })
}

/// Errors from delegation bundle construction.
#[derive(Clone, Debug)]
pub enum DelegationBuildError {
    /// Must have 1 to `circuit::MAX_REAL_NOTES` real notes.
    InvalidNoteCount(usize),
    /// Public input construction failed.
    Instance(circuit::InstanceError),
    /// A required precomputed padded note entry is missing.
    MissingPrecomputedPaddedNote { index: usize, actual: usize },
    /// A precomputed padded note rho is not a canonical field encoding.
    InvalidPrecomputedRho { index: usize },
    /// A precomputed rseed is not valid for the note rho.
    InvalidPrecomputedRseed {
        location: PrecomputedRandomnessLocation,
    },
    /// Precomputed note components do not produce a valid Orchard note.
    InvalidPrecomputedNote {
        location: PrecomputedRandomnessLocation,
    },
    /// IMT proof fetch failed for a padded note nullifier.
    ImtFetchFailed(super::imt::ImtError),
}

impl From<circuit::InstanceError> for DelegationBuildError {
    fn from(e: circuit::InstanceError) -> Self {
        DelegationBuildError::Instance(e)
    }
}

impl From<super::imt::ImtError> for DelegationBuildError {
    fn from(e: super::imt::ImtError) -> Self {
        DelegationBuildError::ImtFetchFailed(e)
    }
}

impl std::fmt::Display for DelegationBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DelegationBuildError::InvalidNoteCount(n) => {
                write!(
                    f,
                    "invalid note count: {} (expected 1–{})",
                    n,
                    circuit::MAX_REAL_NOTES
                )
            }
            DelegationBuildError::Instance(e) => {
                write!(f, "instance construction failed: {e}")
            }
            DelegationBuildError::MissingPrecomputedPaddedNote { index, actual } => {
                write!(
                    f,
                    "missing precomputed padded note at index {index} (got {actual} entries)"
                )
            }
            DelegationBuildError::InvalidPrecomputedRho { index } => {
                write!(f, "invalid precomputed padded note rho at index {index}")
            }
            DelegationBuildError::InvalidPrecomputedRseed { location } => {
                write!(f, "invalid precomputed rseed for {location}")
            }
            DelegationBuildError::InvalidPrecomputedNote { location } => {
                write!(f, "invalid precomputed note components for {location}")
            }
            DelegationBuildError::ImtFetchFailed(e) => {
                write!(f, "IMT proof fetch failed: {e}")
            }
        }
    }
}

impl std::fmt::Display for PrecomputedRandomnessLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrecomputedRandomnessLocation::PaddedNote(index) => {
                write!(f, "padded note {index}")
            }
            PrecomputedRandomnessLocation::SignedNote => write!(f, "signed note"),
            PrecomputedRandomnessLocation::OutputNote => write!(f, "output note"),
        }
    }
}

/// Build a complete delegation bundle with 1 to `circuit::MAX_REAL_NOTES`
/// real notes and padding.
///
/// # Arguments
///
/// - `real_notes`: 1 to `circuit::MAX_REAL_NOTES` real notes with their keys,
///   Merkle paths, and IMT proofs.
/// - `fvk`: The delegator's full viewing key (shared across all real notes).
/// - `alpha`: Spend auth randomizer for the keystone signature.
/// - `output_recipient`: Address of the voting hotkey (output note recipient).
/// - `vote_round_id`: Voting round identifier.
/// - `nc_root`: Note commitment tree root (shared ledger-state anchor).
///   The caller must pin this from the chain's note commitment tree at the
///   verifier-accepted snapshot height; the builder does not authenticate it.
/// - `van_comm_rand`: Blinding factor for the governance commitment.
/// - `imt_provider`: Provider for the bundle-wide alternate-nullifier IMT root
///   and padded-note IMT non-membership proofs. Every real-note proof in
///   `real_notes` must authenticate to this provider's root, and the caller
///   must ensure the provider root is from the same ledger snapshot as
///   `nc_root`.
/// - `rng`: Random number generator.
/// - `precomputed`: If `Some`, reuse Phase 1 randomness for padded/signed/output notes
///   (ZCA-74 fix). If `None`, sample fresh randomness (backward compat for tests).
#[allow(clippy::too_many_arguments)]
pub fn build_delegation_bundle(
    real_notes: Vec<RealNoteInput>,
    fvk: &FullViewingKey,
    alpha: pallas::Scalar,
    output_recipient: orchard::Address,
    vote_round_id: pallas::Base,
    nc_root: pallas::Base,
    van_comm_rand: pallas::Base,
    imt_provider: &impl ImtProvider,
    rng: &mut impl RngCore,
    precomputed: Option<&PrecomputedRandomness>,
) -> Result<DelegationBundle, DelegationBuildError> {
    // The circuit exposes a fixed MAX_REAL_NOTES shape; callers split larger
    // wallets into multiple delegation proofs rather than changing the VK.
    let n_real = real_notes.len();
    if n_real == 0 || n_real > circuit::MAX_REAL_NOTES {
        return Err(DelegationBuildError::InvalidNoteCount(n_real));
    }

    // Snapshot the IMT root — all per-note non-membership proofs must be against this root.
    let nf_imt_root = imt_provider.root();

    // Derive key material.
    let nk = fvk.nk();
    let nk_val = nk.inner();
    let ak: SpendValidatingKey = fvk.clone().into();
    let ivk = fvk.ivk_scalar(Scope::External);

    // Derive the nullifier domain for this round (ZIP §Nullifier Domains).
    let dom = derive_nullifier_domain(vote_round_id);

    // Collect per-note data.
    let mut note_slots = Vec::with_capacity(circuit::MAX_REAL_NOTES);
    let mut cmx_values = Vec::with_capacity(circuit::MAX_REAL_NOTES);
    let mut v_values = Vec::with_capacity(circuit::MAX_REAL_NOTES);
    let mut gov_nulls = Vec::with_capacity(circuit::MAX_REAL_NOTES);

    // Process real notes: derive psi/rcm from rseed, compute the note commitment,
    // real nullifier, and gov nullifier, then pack everything into a NoteSlotWitness.
    for input in &real_notes {
        let note = &input.note;
        let rho = note.rho();
        let psi = note.rseed().psi(&rho);
        let rcm = note.rseed().rcm(&rho);
        let cm = note.commitment();
        let cmx = ExtractedNoteCommitment::from(cm.clone()).inner();
        let v_raw = note.value().inner();
        let recipient = note.recipient();

        // Condition 12: real nullifier for IMT non-membership.
        let real_nf = note.nullifier(fvk);
        // Condition 14: alternate nullifier = Poseidon(domain tag, nk, dom, real_nf).
        let gov_null = gov_null_hash(nk_val, dom, real_nf.inner());

        let slot = NoteSlotWitness {
            g_d: Value::known(recipient.g_d()),
            pk_d: Value::known(recipient.pk_d().inner()),
            v: Value::known(note.value()),
            rho: Value::known(rho.into_inner()),
            psi: Value::known(psi),
            rcm: Value::known(rcm),
            cm: Value::known(cm.inner()),
            path: Value::known(input.merkle_path.auth_path()),
            pos: Value::known(input.merkle_path.position()),
            imt_nf_bounds: Value::known(input.imt_proof.nf_bounds),
            imt_leaf_pos: Value::known(input.imt_proof.leaf_pos),
            imt_path: Value::known(input.imt_proof.path),
            is_internal: Value::known(matches!(input.scope, Scope::Internal)),
        };

        note_slots.push(slot);
        cmx_values.push(cmx);
        v_values.push(v_raw);
        gov_nulls.push(gov_null);
    }

    // Pad remaining slots with zero-value dummy notes (ZIP §Note Padding).
    // Dummy notes use v=0, which gates condition 10 (Merkle path) via
    // v * (root - anchor) = 0. All other conditions run unconditionally.
    for i in n_real..circuit::MAX_REAL_NOTES {
        let pad_idx = i - n_real; // index into precomputed.padded_notes
        let padding = build_padding_slot(i, pad_idx, nk, dom, ivk, imt_provider, rng, precomputed)?;

        note_slots.push(padding.witness);
        cmx_values.push(padding.cmx);
        v_values.push(padding.v_raw);
        gov_nulls.push(padding.gov_null);
    }

    let notes: [NoteSlotWitness; circuit::MAX_REAL_NOTES] =
        note_slots.try_into().unwrap_or_else(|_| unreachable!());

    // Condition 8: ballot scaling.
    // num_ballots = floor(v_total / BALLOT_DIVISOR)
    let v_total_u64: u64 = v_values.iter().sum();
    let num_ballots_u64 = v_total_u64 / circuit::BALLOT_DIVISOR;
    let remainder_u64 = v_total_u64 % circuit::BALLOT_DIVISOR;
    let num_ballots_field = pallas::Base::from(num_ballots_u64);

    // Condition 7: gov commitment integrity.
    // van_comm = Poseidon(DOMAIN_VAN, g_d_new_x, pk_d_new_x, num_ballots,
    //                     vote_round_id, MAX_PROPOSAL_AUTHORITY, van_comm_rand)
    // Extract the output address as two x-coordinates (vpk representation).

    let g_d_new_x = *output_recipient
        .g_d()
        .to_affine()
        .coordinates()
        .unwrap()
        .x();
    let pk_d_new_x = *output_recipient
        .pk_d()
        .inner()
        .to_affine()
        .coordinates()
        .unwrap()
        .x();

    let van_comm = van_commitment_hash(
        g_d_new_x,
        pk_d_new_x,
        num_ballots_field,
        vote_round_id,
        van_comm_rand,
    );

    // Condition 3: rho binding.
    // rho_signed = Poseidon(domain, cmx_1, cmx_2, cmx_3, cmx_4, cmx_5, van_comm, vote_round_id)
    // Binds the keystone note to the exact notes being delegated.
    let rho = rho_binding_hash(
        cmx_values[0],
        cmx_values[1],
        cmx_values[2],
        cmx_values[3],
        cmx_values[4],
        van_comm,
        vote_round_id,
    );

    // Construct the keystone (signed) note (ZIP §Dummy Signed Note).
    // Value is 1 so that hardware wallets (Keystone) render the transaction.
    // The rho is bound to the delegation via condition 3.
    let sender_address = fvk.address_at(0u32, Scope::External);
    let signed_rho = Rho::from_nf_old(Nullifier::from_inner(rho));
    let signed_note = if let Some(pre) = precomputed {
        let location = PrecomputedRandomnessLocation::SignedNote;
        let rseed = RandomSeed::from_bytes(pre.rseed_signed, &signed_rho)
            .into_option()
            .ok_or(DelegationBuildError::InvalidPrecomputedRseed { location })?;
        Note::from_parts(sender_address, NoteValue::from_raw(1), signed_rho, rseed)
            .into_option()
            .ok_or(DelegationBuildError::InvalidPrecomputedNote { location })?
    } else {
        Note::new(
            sender_address,
            NoteValue::from_raw(1),
            signed_rho,
            &mut *rng,
        )
    };

    // Condition 2: nullifier integrity — nf_signed is a public input.
    let nf_signed = signed_note.nullifier(fvk);

    // Condition 6: output note commitment integrity.
    // The output note is sent to the voting hotkey address with rho = nf_signed.
    let output_rho = Rho::from_nf_old(nf_signed);
    let output_note = if let Some(pre) = precomputed {
        let location = PrecomputedRandomnessLocation::OutputNote;
        let rseed = RandomSeed::from_bytes(pre.rseed_output, &output_rho)
            .into_option()
            .ok_or(DelegationBuildError::InvalidPrecomputedRseed { location })?;
        Note::from_parts(output_recipient, NoteValue::ZERO, output_rho, rseed)
            .into_option()
            .ok_or(DelegationBuildError::InvalidPrecomputedNote { location })?
    } else {
        Note::new(output_recipient, NoteValue::ZERO, output_rho, &mut *rng)
    };
    let cmx_new = ExtractedNoteCommitment::from(output_note.commitment()).inner();

    // Condition 4: spend authority — rk is the randomized spend key.
    let rk = ak.randomize(&alpha);

    // Assemble the circuit (private witnesses) and instance (public inputs).
    // The caller runs keygen + create_proof on the circuit, then submits
    // the proof + instance to the vote chain. The verifier only needs
    // the instance, proof, and verification key.
    let circuit = circuit::Circuit::from_note_unchecked(fvk, &signed_note, alpha)
        .with_output_note(&output_note)
        .with_notes(notes)
        .with_van_comm_rand(van_comm_rand)
        .with_ballot_scaling(
            pallas::Base::from(num_ballots_u64),
            pallas::Base::from(remainder_u64),
        );

    let instance = circuit::Instance::from_parts(
        nf_signed,
        rk,
        cmx_new,
        van_comm,
        vote_round_id,
        nc_root,
        nf_imt_root,
        [
            gov_nulls[0],
            gov_nulls[1],
            gov_nulls[2],
            gov_nulls[3],
            gov_nulls[4],
        ],
        dom,
    )?;

    Ok(DelegationBundle { circuit, instance })
}

// ================================================================
// Test-only
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delegation::imt::{ImtError, SpacedLeafImtProvider};
    use ff::Field;
    use halo2_proofs::dev::MockProver;
    use incrementalmerkletree::{Hashable, Level};
    use orchard::{
        constants::MERKLE_DEPTH_ORCHARD,
        keys::{FullViewingKey, Scope, SpendingKey},
        note::{commitment::ExtractedNoteCommitment, Note, Rho},
        tree::{MerkleHashOrchard, MerklePath},
        value::NoteValue,
    };
    use pasta_curves::pallas;
    use rand::rngs::OsRng;
    use std::cell::{Cell, RefCell};

    /// Merged circuit K value.
    const K: u32 = 14;

    #[derive(Debug)]
    struct RecordingImtProvider {
        proof: ImtProofData,
        error: Option<ImtError>,
        requested_nfs: RefCell<Vec<pallas::Base>>,
    }

    impl RecordingImtProvider {
        fn returning(proof: ImtProofData) -> Self {
            Self {
                proof,
                error: None,
                requested_nfs: RefCell::new(Vec::new()),
            }
        }

        fn failing(error: ImtError) -> Self {
            Self {
                proof: test_imt_proof(),
                error: Some(error),
                requested_nfs: RefCell::new(Vec::new()),
            }
        }
    }

    impl ImtProvider for RecordingImtProvider {
        fn root(&self) -> pallas::Base {
            self.proof.root
        }

        fn non_membership_proof(&self, nf: pallas::Base) -> Result<ImtProofData, ImtError> {
            self.requested_nfs.borrow_mut().push(nf);
            match &self.error {
                Some(error) => Err(error.clone()),
                None => Ok(self.proof.clone()),
            }
        }
    }

    fn test_imt_proof() -> ImtProofData {
        ImtProofData {
            root: pallas::Base::from(900u64),
            nf_bounds: [
                pallas::Base::from(10u64),
                pallas::Base::from(20u64),
                pallas::Base::from(30u64),
            ],
            leaf_pos: 7,
            path: std::array::from_fn(|i| pallas::Base::from(1_000u64 + i as u64)),
        }
    }

    fn precomputed_padding_note(rng: &mut impl RngCore) -> (PaddedNoteData, Rho, RandomSeed) {
        let rho = Rho::from_nf_old(Nullifier::from_inner(pallas::Base::random(&mut *rng)));
        let rseed = random_seed_for_rho(&rho, &mut *rng);
        (
            PaddedNoteData {
                rho: rho.to_bytes(),
                rseed: *rseed.as_bytes(),
            },
            rho,
            rseed,
        )
    }

    fn assert_known<T>(value: &Value<T>, f: impl FnOnce(&T) -> bool) {
        let checked = Cell::new(false);
        value.assert_if_known(|actual| {
            checked.set(true);
            f(actual)
        });
        assert!(checked.get(), "expected known witness value");
    }

    fn assert_padding_slot_matches(
        padding: &PaddingSlot,
        slot_index: usize,
        nk: &NullifierDerivingKey,
        dom: pallas::Base,
        ivk: pallas::Scalar,
        rho: Rho,
        rseed: RandomSeed,
        imt_proof: &ImtProofData,
        requested_nfs: &[pallas::Base],
    ) {
        let (g_d_pad, pk_d_pad) = padding_points(slot_index, ivk);
        let psi = rseed.psi(&rho);
        let rcm = rseed.rcm(&rho);
        let cm = NoteCommitment::derive(
            g_d_pad.to_affine().to_bytes(),
            pk_d_pad.to_affine().to_bytes(),
            NoteValue::ZERO,
            rho.into_inner(),
            psi,
            rcm.clone(),
        )
        .expect("padding note commitment must not be bottom");
        let cm_point = cm.inner();
        let cmx = ExtractedNoteCommitment::from(cm.clone()).inner();
        let real_nf = Nullifier::derive(nk, rho.into_inner(), psi, cm).inner();

        assert_eq!(padding.cmx, cmx);
        assert_eq!(padding.v_raw, 0);
        assert_eq!(padding.gov_null, gov_null_hash(nk.inner(), dom, real_nf));
        assert_eq!(requested_nfs, &[real_nf]);

        assert_known(&padding.witness.g_d, |actual| *actual == g_d_pad);
        assert_known(&padding.witness.pk_d, |actual| *actual == pk_d_pad);
        assert_known(&padding.witness.v, |actual| *actual == NoteValue::ZERO);
        assert_known(&padding.witness.rho, |actual| *actual == rho.into_inner());
        assert_known(&padding.witness.psi, |actual| *actual == psi);
        assert_known(&padding.witness.rcm, |actual| actual.inner() == rcm.inner());
        assert_known(&padding.witness.cm, |actual| *actual == cm_point);
        assert_known(&padding.witness.imt_nf_bounds, |actual| {
            *actual == imt_proof.nf_bounds
        });
        assert_known(&padding.witness.imt_leaf_pos, |actual| {
            *actual == imt_proof.leaf_pos
        });
        assert_known(&padding.witness.imt_path, |actual| {
            *actual == imt_proof.path
        });
        assert_known(&padding.witness.is_internal, |actual| !*actual);
    }

    /// Helper: create 1 to `circuit::MAX_REAL_NOTES` real note inputs with a
    /// shared Merkle tree and anchor.
    ///
    /// Notes are placed at positions 0..n in the commitment tree. Returns
    /// `(inputs, nc_root)` where `nc_root` is the shared anchor.
    ///
    fn make_real_note_inputs(
        fvk: &FullViewingKey,
        values: &[u64],
        scopes: &[Scope],
        imt_provider: &impl ImtProvider,
        rng: &mut impl RngCore,
    ) -> (Vec<RealNoteInput>, pallas::Base) {
        let n = values.len();
        assert!(n >= 1 && n <= circuit::MAX_REAL_NOTES);
        assert_eq!(n, scopes.len());

        // Create notes.
        let mut notes = Vec::with_capacity(n);
        for (idx, &v) in values.iter().enumerate() {
            let recipient = fvk.address_at(0u32, scopes[idx]);
            let note_value = NoteValue::from_raw(v);
            let (_, _, dummy_parent) = Note::dummy(&mut *rng, None);
            let note = Note::new(
                recipient,
                note_value,
                Rho::from_nf_old(dummy_parent.nullifier(fvk)),
                &mut *rng,
            );
            notes.push(note);
        }

        // Extract leaf hashes, padding to 8 with empty leaves.
        let empty_leaf = MerkleHashOrchard::empty_leaf();
        let mut leaves = [empty_leaf; 8];
        for (i, note) in notes.iter().enumerate() {
            let cmx = ExtractedNoteCommitment::from(note.commitment());
            leaves[i] = MerkleHashOrchard::from_cmx(&cmx);
        }

        // Build the bottom three levels of the shared tree (8-leaf tree).
        let l1_0 = MerkleHashOrchard::combine(Level::from(0), &leaves[0], &leaves[1]);
        let l1_1 = MerkleHashOrchard::combine(Level::from(0), &leaves[2], &leaves[3]);
        let l1_2 = MerkleHashOrchard::combine(Level::from(0), &leaves[4], &leaves[5]);
        let l1_3 = MerkleHashOrchard::combine(Level::from(0), &leaves[6], &leaves[7]);
        let l2_0 = MerkleHashOrchard::combine(Level::from(1), &l1_0, &l1_1);
        let l2_1 = MerkleHashOrchard::combine(Level::from(1), &l1_2, &l1_3);
        let l3_0 = MerkleHashOrchard::combine(Level::from(2), &l2_0, &l2_1);

        // Hash up through the remaining levels with empty subtree siblings.
        let mut current = l3_0;
        for level in 3..MERKLE_DEPTH_ORCHARD {
            let sibling = MerkleHashOrchard::empty_root(Level::from(level as u8));
            current = MerkleHashOrchard::combine(Level::from(level as u8), &current, &sibling);
        }
        let nc_root = current.inner();

        // Build Merkle paths and RealNoteInputs.
        let l1 = [l1_0, l1_1, l1_2, l1_3];
        let l2 = [l2_0, l2_1];
        let mut inputs = Vec::with_capacity(n);
        for (i, note) in notes.into_iter().enumerate() {
            let mut auth_path = [MerkleHashOrchard::empty_leaf(); MERKLE_DEPTH_ORCHARD];
            auth_path[0] = leaves[i ^ 1];
            auth_path[1] = l1[(i >> 1) ^ 1];
            auth_path[2] = l2[1 - (i >> 2)];
            for level in 3..MERKLE_DEPTH_ORCHARD {
                auth_path[level] = MerkleHashOrchard::empty_root(Level::from(level as u8));
            }
            let merkle_path = MerklePath::from_parts(i as u32, auth_path);

            let real_nf = note.nullifier(fvk);
            let imt_proof = imt_provider.non_membership_proof(real_nf.inner()).unwrap();

            inputs.push(RealNoteInput {
                note,
                fvk: fvk.clone(),
                merkle_path,
                imt_proof,
                scope: scopes[i],
            });
        }

        (inputs, nc_root)
    }

    /// Helper: build a bundle with explicit scopes.
    fn build_bundle(values: &[u64], scopes: &[Scope]) -> DelegationBundle {
        assert_eq!(values.len(), scopes.len());
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let output_recipient = fvk.address_at(1u32, Scope::External);
        let vote_round_id = pallas::Base::random(&mut rng);
        let van_comm_rand = pallas::Base::random(&mut rng);
        let alpha = pallas::Scalar::random(&mut rng);

        let imt = SpacedLeafImtProvider::new();
        let (inputs, nc_root) = make_real_note_inputs(&fvk, values, scopes, &imt, &mut rng);

        let bundle = build_delegation_bundle(
            inputs,
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

        assert_delegation_output_shape(&bundle);
        bundle
    }

    fn build_single_note_bundle_with_precomputed(
        precomputed: &PrecomputedRandomness,
    ) -> Result<DelegationBundle, DelegationBuildError> {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();

        build_single_note_bundle_with_fvk_and_precomputed(&fvk, precomputed, &mut rng)
    }

    fn build_single_note_bundle_with_fvk_and_precomputed(
        fvk: &FullViewingKey,
        precomputed: &PrecomputedRandomness,
        rng: &mut impl RngCore,
    ) -> Result<DelegationBundle, DelegationBuildError> {
        let output_recipient = fvk.address_at(1u32, Scope::External);
        let vote_round_id = pallas::Base::random(&mut *rng);
        let van_comm_rand = pallas::Base::random(&mut *rng);
        let alpha = pallas::Scalar::random(&mut *rng);

        let imt = SpacedLeafImtProvider::new();
        let (inputs, nc_root) =
            make_real_note_inputs(fvk, &[13_000_000], &[Scope::External], &imt, &mut *rng);

        build_delegation_bundle(
            inputs,
            fvk,
            alpha,
            output_recipient,
            vote_round_id,
            nc_root,
            van_comm_rand,
            &imt,
            rng,
            Some(precomputed),
        )
    }

    fn make_valid_padded_note_data(rng: &mut impl RngCore) -> PaddedNoteData {
        let rho = Rho::from_nf_old(Nullifier::from_inner(pallas::Base::random(&mut *rng)));
        let rseed = random_seed_for_rho(&rho, rng);

        PaddedNoteData {
            rho: rho.to_bytes(),
            rseed: *rseed.as_bytes(),
        }
    }

    fn assert_delegation_output_shape(bundle: &DelegationBundle) {
        let pi = bundle.instance.to_halo2_instance();
        assert_eq!(pi.len(), 14, "delegation public input shape changed");
        assert_eq!(bundle.instance.gov_null.len(), 5);
        assert_eq!(pi[0], bundle.instance.nf_signed.inner());
        assert_eq!(pi[3], bundle.instance.cmx_new);
        assert_eq!(pi[4], bundle.instance.van_comm);
        assert_eq!(pi[5], bundle.instance.vote_round_id);
        assert_eq!(pi[6], bundle.instance.nc_root);
        assert_eq!(pi[7], bundle.instance.nf_imt_root);
        assert_eq!(&pi[8..13], &bundle.instance.gov_null);
        assert_eq!(pi[13], bundle.instance.dom);
    }

    fn verify_bundle(bundle: &DelegationBundle) {
        // Verify merged circuit.
        let pi = bundle.instance.to_halo2_instance();
        let prover = MockProver::run(K, &bundle.circuit, vec![pi]).unwrap();
        assert_eq!(prover.verify(), Ok(()), "merged circuit failed");
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_single_real_note() {
        let bundle = build_bundle(&[13_000_000], &[Scope::External]);
        verify_bundle(&bundle);
    }

    /// Build a bundle without verifying so callers can inspect the circuit
    /// witnesses. Mirrors `build_and_verify` minus the MockProver step and
    /// returns `(bundle, fvk, ak)` so the test can recompute the external IVK.
    fn build_bundle_for_inspection(
        values: &[u64],
        scopes: &[Scope],
    ) -> (DelegationBundle, FullViewingKey, SpendValidatingKey) {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let ak: SpendValidatingKey = fvk.clone().into();
        let output_recipient = fvk.address_at(1u32, Scope::External);
        let vote_round_id = pallas::Base::random(&mut rng);
        let van_comm_rand = pallas::Base::random(&mut rng);
        let alpha = pallas::Scalar::random(&mut rng);
        let imt = SpacedLeafImtProvider::new();
        let (inputs, nc_root) = make_real_note_inputs(&fvk, values, scopes, &imt, &mut rng);

        let bundle = build_delegation_bundle(
            inputs,
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
        (bundle, fvk, ak)
    }

    /// Build a bundle without verifying so callers can inspect the circuit
    /// witnesses. Mirrors `build_and_verify` minus the MockProver step and
    /// returns `(bundle, fvk, ak)` so the test can recompute the external IVK.
    fn build_bundle_for_inspection(
        values: &[u64],
        scopes: &[Scope],
    ) -> (DelegationBundle, FullViewingKey, SpendValidatingKey) {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let ak: SpendValidatingKey = fvk.clone().into();
        let output_recipient = fvk.address_at(1u32, Scope::External);
        let vote_round_id = pallas::Base::random(&mut rng);
        let van_comm_rand = pallas::Base::random(&mut rng);
        let alpha = pallas::Scalar::random(&mut rng);
        let imt = SpacedLeafImtProvider::new();
        let (inputs, nc_root) = make_real_note_inputs(&fvk, values, scopes, &imt, &mut rng);

        let bundle = build_delegation_bundle(
            inputs,
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
        (bundle, fvk, ak)
    }

    #[test]
    fn test_single_real_note_locks_padding_witnesses() {
        // With 1 real note, slots 1..5 must be padding and their (g_d, pk_d)
        // must come from `padding_points(slot_index, fvk.ivk_scalar(...))`.
        // Catches regressions where the synthetic padding path is silently
        // replaced (e.g. a fallback to `fvk.address_at(...)`) or where the
        // slot index passed to `padding_points` skews off-by-one.
        let (bundle, fvk, _ak) = build_bundle_for_inspection(&[13_000_000], &[Scope::External]);
        let ivk = fvk.ivk_scalar(Scope::External);
        let notes = bundle.circuit.notes_for_testing();
        for slot_index in 1..5 {
            let (expected_g_d, expected_pk_d) = padding_points(slot_index, ivk);
            assert_known(&notes[slot_index].g_d, |actual| *actual == expected_g_d);
            assert_known(&notes[slot_index].pk_d, |actual| *actual == expected_pk_d);
        }
    }

    #[test]
    fn test_five_real_notes_uses_no_padding() {
        // With 5 real notes there are no padding slots. Assert that no slot's
        // g_d matches the synthetic padding point for *any* slot index — this
        // catches an off-by-one in the `n_real..5` iteration boundary that
        // would smuggle a padding point into a real slot (which would silently
        // zero that slot's vote weight in condition 10's `v * (root - anchor)`
        // gate path because padding always has `v = 0`).
        let (bundle, fvk, _ak) =
            build_bundle_for_inspection(&[2_500_000; 5], &[Scope::External; 5]);
        let ivk = fvk.ivk_scalar(Scope::External);
        let padding_g_ds: Vec<_> = (0..5).map(|i| padding_points(i, ivk).0).collect();
        let notes = bundle.circuit.notes_for_testing();
        for slot_index in 0..5 {
            for pad_g_d in &padding_g_ds {
                assert_known(&notes[slot_index].g_d, |actual| actual != pad_g_d);
            }
        }
    }

    #[test]
    fn test_padding_points_are_synthetic_and_ivk_bound() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let ivk = fvk.ivk_scalar(Scope::External);

        for slot_index in 1..5 {
            let (g_d_pad, pk_d_pad) = padding_points(slot_index, ivk);
            let real_orchard_addr = fvk.address_at(slot_index as u32, Scope::External);

            // Deref `NonIdentityPallasPoint` to `pallas::Point` for arithmetic and
            // coordinate access; the wrapper enforces the non-identity invariant
            // at construction (`padding_points` -> `assert_non_identity`).
            assert_eq!(*pk_d_pad, *g_d_pad * ivk);
            assert_ne!(
                g_d_pad.to_affine().to_bytes(),
                real_orchard_addr.g_d().to_affine().to_bytes()
            );
            assert_ne!(
                pk_d_pad.to_affine().to_bytes(),
                real_orchard_addr.pk_d().to_bytes()
            );
        }
    }

    /// Locks the structural property that makes ZCA-450's fix correct: the
    /// hash-to-curve personalization for padding `g_d_pad` is **different**
    /// from Orchard's `KEY_DIVERSIFICATION_PERSONALIZATION`. Domain separation
    /// of `hash_to_curve` ensures the two personalizations produce disjoint
    /// images with overwhelming probability, so no real Orchard sender can
    /// derive a `g_d = DiversifyHash(d)` that collides with any padding
    /// `g_d_pad`. If either constant is ever renamed (here or upstream) so
    /// they coincide, this test fails loudly before a release ships a
    /// padding scheme that re-enters the real-Orchard address universe.
    ///
    /// Exhaustive testing of the inverse property — that no 88-bit `d`
    /// satisfies `DiversifyHash(d) == g_d_pad_i` — is infeasible (2^88
    /// preimage search), so we lock the construction-time invariant
    /// (different personalization → disjoint hash-to-curve domains)
    /// instead of the runtime invariant (no collision exists).
    #[test]
    fn test_padding_personalization_is_domain_separated_from_orchard() {
        use orchard::constants::KEY_DIVERSIFICATION_PERSONALIZATION;

        assert_ne!(
            PADDING_PERSONALIZATION, KEY_DIVERSIFICATION_PERSONALIZATION,
            "padding personalization must be domain-separated from Orchard's \
             DiversifyHash personalization; otherwise synthetic padding `g_d_pad` \
             can collide with real diversified bases and ZCA-450's fix regresses"
        );
    }

    // ---- Low-level Orchard API checks ----
    //
    // The builder calls these APIs directly for synthetic padding notes. These
    // tests use ordinary Orchard `Note` / `FullViewingKey` instances as ground
    // truth so the POC still catches accidental API or call-site drift.

    /// Builds a real Orchard `Note` for exercising the low-level Orchard APIs.
    fn fixture_real_note(
        scope: Scope,
        rng: &mut impl RngCore,
    ) -> (FullViewingKey, SpendValidatingKey, Note) {
        let sk = SpendingKey::random(rng);
        let fvk: FullViewingKey = (&sk).into();
        let ak: SpendValidatingKey = fvk.clone().into();
        let recipient = fvk.address_at(0u32, scope);
        let (_, _, dummy_parent) = Note::dummy(rng, None);
        let note = Note::new(
            recipient,
            NoteValue::from_raw(12_500_000),
            Rho::from_nf_old(dummy_parent.nullifier(&fvk)),
            rng,
        );
        (fvk, ak, note)
    }

    #[test]
    fn test_note_commitment_point_matches_orchard() {
        // The builder uses `NoteCommitment::derive` directly for synthetic
        // padding slots, so check it also matches ordinary Orchard notes.
        let mut rng = OsRng;
        for scope in [Scope::External, Scope::Internal] {
            let (_fvk, _ak, note) = fixture_real_note(scope, &mut rng);
            let recipient = note.recipient();
            let rho = note.rho();
            let psi = note.rseed().psi(&rho);
            let rcm = note.rseed().rcm(&rho);

            let derived = NoteCommitment::derive(
                recipient.g_d().to_affine().to_bytes(),
                recipient.pk_d().inner().to_affine().to_bytes(),
                note.value(),
                rho.into_inner(),
                psi,
                rcm,
            )
            .expect("fixture note commitment must not be bottom")
            .inner();
            let orchard = note.commitment().inner();

            assert_eq!(
                derived, orchard,
                "NoteCommitment::derive disagrees with ordinary Orchard note ({scope:?})"
            );
        }
    }

    #[test]
    fn test_derive_note_nullifier_matches_orchard() {
        // The builder uses `Nullifier::derive` directly for synthetic padding
        // slots, so check it also matches ordinary Orchard notes.
        let mut rng = OsRng;
        for scope in [Scope::External, Scope::Internal] {
            let (fvk, _ak, note) = fixture_real_note(scope, &mut rng);
            let nk = fvk.nk();
            let rho = note.rho();
            let psi = note.rseed().psi(&rho);
            let cm = note.commitment();

            let derived = Nullifier::derive(nk, rho.into_inner(), psi, cm).inner();
            let orchard = note.nullifier(&fvk).inner();

            assert_eq!(
                derived, orchard,
                "Nullifier::derive disagrees with ordinary Orchard note ({scope:?})"
            );
        }
    }

    #[test]
    fn test_fvk_ivk_scalar_matches_orchard_address_derivation() {
        // Check the exposed IVK scalar via the diversified-address invariant
        // `pk_d = [ivk_external] * g_d` on real external-scope Orchard addresses.
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let ivk = fvk.ivk_scalar(Scope::External);

        // Sweep several diversifier indices to catch accidental fixed-index
        // shortcuts in the derivation.
        for idx in [0u32, 1, 7, 1234] {
            let addr = fvk.address_at(idx, Scope::External);
            assert_eq!(
                *addr.g_d() * ivk,
                *addr.pk_d().inner(),
                "FullViewingKey::ivk_scalar drifted: [ivk] * g_d != pk_d at diversifier index {idx}"
            );
        }

        // Sanity: the external ivk must NOT validate internal-scope addresses,
        // catching a bug where `rivk(Scope::External)` is silently swapped for
        // `rivk(Scope::Internal)`.
        let internal_addr = fvk.address_at(0u32, Scope::Internal);
        assert_ne!(
            *internal_addr.g_d() * ivk,
            *internal_addr.pk_d().inner(),
            "FullViewingKey::ivk_scalar incorrectly validates an internal-scope address"
        );
    }

    #[test]
    fn test_build_padding_slot_fresh_randomness_populates_strict_witnesses() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let nk = fvk.nk();
        let dom = derive_nullifier_domain(pallas::Base::random(&mut rng));
        let ivk = fvk.ivk_scalar(Scope::External);
        let imt_proof = test_imt_proof();
        let imt = RecordingImtProvider::returning(imt_proof.clone());

        let padding = build_padding_slot(3, 0, nk, dom, ivk, &imt, &mut rng, None).unwrap();

        let (g_d_pad, pk_d_pad) = padding_points(3, ivk);
        assert_known(&padding.witness.g_d, |actual| *actual == g_d_pad);
        assert_known(&padding.witness.pk_d, |actual| *actual == pk_d_pad);
        let generated_padding_values = padding
            .witness
            .rho
            .as_ref()
            .copied()
            .zip(padding.witness.psi.as_ref().copied())
            .zip(padding.witness.rcm.as_ref().cloned())
            .zip(padding.witness.cm.as_ref().copied());
        assert_known(
            &generated_padding_values,
            |(((rho_inner, psi), rcm), cm_witness)| {
                let cm = NoteCommitment::derive(
                    g_d_pad.to_affine().to_bytes(),
                    pk_d_pad.to_affine().to_bytes(),
                    NoteValue::ZERO,
                    *rho_inner,
                    *psi,
                    rcm.clone(),
                )
                .expect("padding note commitment must not be bottom");
                let cm_point = cm.inner();
                let cmx = ExtractedNoteCommitment::from(cm.clone()).inner();
                let real_nf = Nullifier::derive(nk, *rho_inner, *psi, cm).inner();

                *cm_witness == cm_point
                    && padding.cmx == cmx
                    && padding.gov_null == gov_null_hash(nk.inner(), dom, real_nf)
                    && imt.requested_nfs.borrow().as_slice() == [real_nf]
            },
        );
        assert_known(&padding.witness.v, |actual| *actual == NoteValue::ZERO);
        assert_known(&padding.witness.is_internal, |actual| !*actual);
        assert_known(&padding.witness.imt_nf_bounds, |actual| {
            *actual == imt_proof.nf_bounds
        });
        assert_known(&padding.witness.imt_leaf_pos, |actual| {
            *actual == imt_proof.leaf_pos
        });
        assert_known(&padding.witness.imt_path, |actual| {
            *actual == imt_proof.path
        });
        assert_eq!(padding.v_raw, 0);
        assert_eq!(imt.requested_nfs.borrow().len(), 1);
    }

    #[test]
    fn test_build_padding_slot_reuses_selected_precomputed_randomness() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let nk = fvk.nk();
        let dom = derive_nullifier_domain(pallas::Base::random(&mut rng));
        let ivk = fvk.ivk_scalar(Scope::External);
        let imt_proof = test_imt_proof();
        let imt = RecordingImtProvider::returning(imt_proof.clone());

        let (unused_pd, _, _) = precomputed_padding_note(&mut rng);
        let (selected_pd, selected_rho, selected_rseed) = precomputed_padding_note(&mut rng);
        let precomputed = PrecomputedRandomness {
            padded_notes: vec![unused_pd, selected_pd],
            rseed_signed: [0; 32],
            rseed_output: [0; 32],
        };

        let padding =
            build_padding_slot(4, 1, nk, dom, ivk, &imt, &mut rng, Some(&precomputed)).unwrap();

        let requested_nfs = imt.requested_nfs.borrow();
        assert_padding_slot_matches(
            &padding,
            4,
            nk,
            dom,
            ivk,
            selected_rho,
            selected_rseed,
            &imt_proof,
            &requested_nfs,
        );
    }

    #[test]
    fn test_build_padding_slot_propagates_imt_errors() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let imt = RecordingImtProvider::failing(ImtError("fixture failure".to_string()));

        let result = build_padding_slot(
            2,
            0,
            fvk.nk(),
            derive_nullifier_domain(pallas::Base::random(&mut rng)),
            fvk.ivk_scalar(Scope::External),
            &imt,
            &mut rng,
            None,
        );

        assert!(matches!(
            result,
            Err(DelegationBuildError::ImtFetchFailed(ImtError(message)))
                if message == "fixture failure"
        ));
        assert_eq!(imt.requested_nfs.borrow().len(), 1);
    }

    #[test]
    fn test_build_padding_slot_rejects_missing_precomputed_padding_entry() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let imt = RecordingImtProvider::returning(test_imt_proof());
        let precomputed = PrecomputedRandomness {
            padded_notes: vec![],
            rseed_signed: [0; 32],
            rseed_output: [0; 32],
        };

        let result = build_padding_slot(
            1,
            0,
            fvk.nk(),
            derive_nullifier_domain(pallas::Base::random(&mut rng)),
            fvk.ivk_scalar(Scope::External),
            &imt,
            &mut rng,
            Some(&precomputed),
        );

        assert!(matches!(
            result,
            Err(DelegationBuildError::MissingPrecomputedPaddedNote {
                index: 0,
                actual: 0
            })
        ));
    }

    #[test]
    fn test_single_real_note_locks_padding_witnesses() {
        // With 1 real note, slots 1..5 must be padding and their (g_d, pk_d)
        // must come from `padding_points(slot_index, fvk.ivk_scalar(...))`.
        // Catches regressions where the synthetic padding path is silently
        // replaced (e.g. a fallback to `fvk.address_at(...)`) or where the
        // slot index passed to `padding_points` skews off-by-one.
        let (bundle, fvk, _ak) = build_bundle_for_inspection(&[13_000_000], &[Scope::External]);
        let ivk = fvk.ivk_scalar(Scope::External);
        let notes = bundle.circuit.notes_for_testing();
        for slot_index in 1..5 {
            let (expected_g_d, expected_pk_d) = padding_points(slot_index, ivk);
            assert_known(&notes[slot_index].g_d, |actual| *actual == expected_g_d);
            assert_known(&notes[slot_index].pk_d, |actual| *actual == expected_pk_d);
        }
    }

    #[test]
    fn test_five_real_notes_uses_no_padding() {
        // With 5 real notes there are no padding slots. Assert that no slot's
        // g_d matches the synthetic padding point for *any* slot index — this
        // catches an off-by-one in the `n_real..5` iteration boundary that
        // would smuggle a padding point into a real slot (which would silently
        // zero that slot's vote weight in condition 10's `v * (root - anchor)`
        // gate path because padding always has `v = 0`).
        let (bundle, fvk, _ak) =
            build_bundle_for_inspection(&[2_500_000; 5], &[Scope::External; 5]);
        let ivk = fvk.ivk_scalar(Scope::External);
        let padding_g_ds: Vec<_> = (0..5).map(|i| padding_points(i, ivk).0).collect();
        let notes = bundle.circuit.notes_for_testing();
        for slot_index in 0..5 {
            for pad_g_d in &padding_g_ds {
                assert_known(&notes[slot_index].g_d, |actual| actual != pad_g_d);
            }
        }
    }

    #[test]
    fn test_padding_points_are_synthetic_and_ivk_bound() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let ivk = fvk.ivk_scalar(Scope::External);

        for slot_index in 1..5 {
            let (g_d_pad, pk_d_pad) = padding_points(slot_index, ivk);
            let real_orchard_addr = fvk.address_at(slot_index as u32, Scope::External);

            // Deref `NonIdentityPallasPoint` to `pallas::Point` for arithmetic and
            // coordinate access; the wrapper enforces the non-identity invariant
            // at construction (`padding_points` -> `assert_non_identity`).
            assert_eq!(*pk_d_pad, *g_d_pad * ivk);
            assert_ne!(
                g_d_pad.to_affine().to_bytes(),
                real_orchard_addr.g_d().to_affine().to_bytes()
            );
            assert_ne!(
                pk_d_pad.to_affine().to_bytes(),
                real_orchard_addr.pk_d().to_bytes()
            );
        }
    }

    #[test]
    fn test_four_real_notes_builds_expected_output_shape() {
        // 3,200,000 x 4 = 12,800,000 → num_ballots = 1, remainder = 300,000.
        build_bundle(
            &[3_200_000, 3_200_000, 3_200_000, 3_200_000],
            &[
                Scope::External,
                Scope::External,
                Scope::External,
                Scope::External,
            ],
        );
    }

    #[test]
    fn test_two_real_notes_builds_expected_output_shape() {
        build_bundle(&[7_000_000, 7_000_000], &[Scope::External, Scope::External]);
    }

    #[test]
    fn test_min_weight_boundary_builds_expected_output_shape() {
        // v_total = 12,500,000 exactly → num_ballots = 1, remainder = 0. Should pass.
        build_bundle(&[12_500_000], &[Scope::External]);
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_below_one_ballot() {
        // v_total = 12,499,999 → num_ballots = 0. Circuit should fail
        // (non-zero check on num_ballots causes nb_minus_one to wrap).
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let output_recipient = fvk.address_at(1u32, Scope::External);
        let vote_round_id = pallas::Base::random(&mut rng);
        let van_comm_rand = pallas::Base::random(&mut rng);
        let alpha = pallas::Scalar::random(&mut rng);

        let imt = SpacedLeafImtProvider::new();
        let (inputs, nc_root) =
            make_real_note_inputs(&fvk, &[12_499_999], &[Scope::External], &imt, &mut rng);

        let bundle = build_delegation_bundle(
            inputs,
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

        let pi = bundle.instance.to_halo2_instance();
        let prover = MockProver::run(K, &bundle.circuit, vec![pi]).unwrap();
        assert!(prover.verify().is_err(), "below one ballot should fail");
    }

    #[test]
    fn test_three_ballots_builds_expected_output_shape() {
        // 3 notes × 12,500,000 = 37,500,000 → num_ballots = 3, remainder = 0.
        build_bundle(
            &[12_500_000, 12_500_000, 12_500_000],
            &[Scope::External, Scope::External, Scope::External],
        );
    }

    #[test]
    fn test_zero_notes_error() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let output_recipient = fvk.address_at(1u32, Scope::External);
        let imt = SpacedLeafImtProvider::new();

        let result = build_delegation_bundle(
            vec![],
            &fvk,
            pallas::Scalar::random(&mut rng),
            output_recipient,
            pallas::Base::random(&mut rng),
            pallas::Base::random(&mut rng),
            pallas::Base::random(&mut rng),
            &imt,
            &mut rng,
            None,
        );

        assert!(matches!(
            result,
            Err(DelegationBuildError::InvalidNoteCount(0))
        ));
    }

    #[test]
    fn test_five_real_notes_builds_expected_output_shape() {
        // 2,500,000 x 5 = 12,500,000 → num_ballots = 1, remainder = 0.
        build_bundle(
            &[2_500_000, 2_500_000, 2_500_000, 2_500_000, 2_500_000],
            &[
                Scope::External,
                Scope::External,
                Scope::External,
                Scope::External,
                Scope::External,
            ],
        );
    }

    #[test]
    fn test_six_notes_error() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let output_recipient = fvk.address_at(1u32, Scope::External);
        let imt = SpacedLeafImtProvider::new();

        let (inputs, _) = make_real_note_inputs(
            &fvk,
            &[3_000_000, 3_000_000, 3_000_000, 3_000_000, 3_000_000],
            &[
                Scope::External,
                Scope::External,
                Scope::External,
                Scope::External,
                Scope::External,
            ],
            &imt,
            &mut rng,
        );
        // Add a 6th note by extending.
        let mut inputs = inputs;
        let (extra, _) =
            make_real_note_inputs(&fvk, &[3_000_000], &[Scope::External], &imt, &mut rng);
        inputs.extend(extra);

        let result = build_delegation_bundle(
            inputs,
            &fvk,
            pallas::Scalar::random(&mut rng),
            output_recipient,
            pallas::Base::random(&mut rng),
            pallas::Base::random(&mut rng),
            pallas::Base::random(&mut rng),
            &imt,
            &mut rng,
            None,
        );

        assert!(matches!(
            result,
            Err(DelegationBuildError::InvalidNoteCount(6))
        ));
    }

    #[test]
    fn test_missing_precomputed_padded_note_returns_error() {
        let precomputed = PrecomputedRandomness {
            padded_notes: vec![],
            rseed_signed: [0u8; 32],
            rseed_output: [0u8; 32],
        };

        let result = build_single_note_bundle_with_precomputed(&precomputed);

        assert!(matches!(
            result,
            Err(DelegationBuildError::MissingPrecomputedPaddedNote {
                index: 0,
                actual: 0
            })
        ));
    }

    #[test]
    fn test_partial_precomputed_padded_notes_returns_later_missing_error() {
        let mut rng = OsRng;
        let sk = SpendingKey::random(&mut rng);
        let fvk: FullViewingKey = (&sk).into();
        let precomputed = PrecomputedRandomness {
            padded_notes: vec![
                make_valid_padded_note_data(&mut rng),
                make_valid_padded_note_data(&mut rng),
            ],
            rseed_signed: [0u8; 32],
            rseed_output: [0u8; 32],
        };

        let result =
            build_single_note_bundle_with_fvk_and_precomputed(&fvk, &precomputed, &mut rng);

        assert!(matches!(
            result,
            Err(DelegationBuildError::MissingPrecomputedPaddedNote {
                index: 2,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_invalid_precomputed_padded_rho_returns_error() {
        let precomputed = PrecomputedRandomness {
            padded_notes: vec![PaddedNoteData {
                rho: [0xffu8; 32],
                rseed: [0u8; 32],
            }],
            rseed_signed: [0u8; 32],
            rseed_output: [0u8; 32],
        };

        let result = build_single_note_bundle_with_precomputed(&precomputed);

        assert!(matches!(
            result,
            Err(DelegationBuildError::InvalidPrecomputedRho { index: 0 })
        ));
    }

    #[test]
    fn test_single_internal_note_builds_expected_output_shape() {
        build_bundle(&[13_000_000], &[Scope::Internal]);
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_mixed_scope_notes() {
        let bundle = build_bundle(
            &[4_000_000, 4_000_000, 3_000_000, 2_000_000],
            &[
                Scope::External,
                Scope::Internal,
                Scope::External,
                Scope::Internal,
            ],
        );
        verify_bundle(&bundle);
    }

    #[test]
    fn test_all_internal_notes_builds_expected_output_shape() {
        build_bundle(
            &[4_000_000, 4_000_000, 3_000_000, 2_000_000],
            &[
                Scope::Internal,
                Scope::Internal,
                Scope::Internal,
                Scope::Internal,
            ],
        );
    }
}
