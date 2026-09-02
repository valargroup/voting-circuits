//! The Vote Proof circuit implementation (ZKP #2).
//!
//! Proves that a registered voter is casting a valid vote, without
//! revealing which VAN they hold. Currently implements:
//!
//! - **Condition 1**: VAN Membership (Poseidon Merkle path, `constrain_instance`).
//! - **Condition 2**: VAN Integrity (Poseidon hash).
//! - **Condition 3**: Diversified Address Integrity (`vpk_pk_d = [ivk_v] * vpk_g_d` via CommitIvk).
//! - **Condition 4**: Spend Authority — `r_vpk = vsk.ak + [alpha_v] * G` (fixed-base mul + point add, `constrain_instance`).
//! - **Condition 5**: VAN Nullifier Integrity (nested Poseidon, `constrain_instance`).
//! - **Condition 6**: Proposal Authority Decrement (custom bit-decomposition chip with a `(proposal_id, 2^proposal_id)` lookup; see `gadgets/authority_decrement.rs`).
//! - **Condition 7**: New VAN Integrity (Poseidon hash, `constrain_instance`).
//! - **Condition 8**: Shares Sum Correctness (AddChip, `constrain_equal`).
//! - **Condition 9**: Shares Range (LookupRangeCheck, `[0, 2^30)`).
//! - **Condition 10**: Shares Hash Integrity (Poseidon `ConstantLength<16>` over 16 blinded share commitments; output flows to condition 12).
//! - **Condition 11**: Encryption Integrity (ECC variable-base mul, `constrain_equal`).
//! - **Condition 12**: Vote Commitment Integrity (Poseidon `ConstantLength<5>`, `constrain_instance`).
//!
//! Conditions 1–4 and 5–12 are fully constrained in-circuit.
//!
//! ## Conditions overview
//!
//! VAN ownership and spending:
//! - **Condition 1**: VAN Membership — Merkle path from `vote_authority_note_old`
//!   to `vote_comm_tree_root`.
//! - **Condition 2**: VAN Integrity — `vote_authority_note_old` is the two-layer
//!   Poseidon hash (ZKP 1–compatible: core then finalize with rand). *(implemented)*
//! - **Condition 3**: Diversified Address Integrity — `vpk_pk_d = [ivk_v] * vpk_g_d`
//!   where `ivk_v = CommitIvk(ExtractP([vsk]*SpendAuthG), vsk.nk)`. *(implemented)*
//! - **Condition 4**: Spend Authority — `r_vpk = vsk.ak + [alpha_v] * G`; enforced in-circuit (fixed-base mul + point add, `constrain_instance`).
//! - **Condition 5**: VAN Nullifier Integrity — `van_nullifier` is correctly
//!   derived from `vsk.nk`. *(implemented)*
//!
//! New VAN construction:
//! - **Condition 6**: Proposal Authority Decrement — `proposal_authority_new =
//!   proposal_authority_old - (1 << proposal_id)`, with bitmask range [0, 2^51). *(implemented)*
//! - **Condition 7**: New VAN Integrity — same two-layer structure as condition 2
//!   but with decremented authority. *(implemented)*
//!
//! Vote commitment construction:
//! - **Condition 8**: Shares Sum Correctness — `sum(shares_1..16) = total_note_value`.
//!   *(implemented)*
//! - **Condition 9**: Shares Range — each `shares_j` in `[0, 2^30)`.
//!   *(implemented)*
//! - **Condition 10**: Shares Hash Integrity — `shares_hash = H(enc_share_1..16)`.
//!   `shares_hash` is an internal wire, not a public instance. *(implemented)*
//! - **Condition 11**: Encryption Integrity — each `enc_share_i = ElGamal(shares_i, r_i, ea_pk)`.
//!   *(implemented)*
//! - **Condition 12**: Vote Commitment Integrity — `vote_commitment = H(DOMAIN_VC, voting_round_id,
//!   shares_hash, proposal_id, vote_decision)`, and this terminal commitment is
//!   the public binding for condition 10. *(implemented)*

use std::vec::Vec;

use voting_crypto_deps::halo2_gadgets::{
    ecc::{
        chip::{EccChip, EccConfig},
        CircuitVersion, NonIdentityPoint, ScalarFixed,
    },
    poseidon::{
        primitives::{self as poseidon, ConstantLength},
        Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
    },
    sinsemilla::chip::{SinsemillaChip, SinsemillaConfig},
    utilities::lookup_range_check::{LookupRangeCheck, LookupRangeCheckConfig},
};
use voting_crypto_deps::halo2_proofs::{
    circuit::{floor_planner, AssignedCell, Layouter, Value},
    plonk::{self, Advice, Column, ConstraintSystem, Fixed, Instance as InstanceColumn},
};
use voting_crypto_deps::orchard::{
    circuit::{
        commit_ivk::{CommitIvkChip, CommitIvkConfig},
        gadget::{
            add_chip::{AddChip, AddConfig},
            assign_free_advice, AddInstruction,
        },
    },
    constants::{OrchardCommitDomains, OrchardFixedBases, OrchardHashDomains},
};
use voting_crypto_deps::pasta_curves::{pallas, vesta};

use super::gadgets::authority_decrement::{AuthorityDecrementChip, AuthorityDecrementConfig};
use crate::{
    bridge::{compute_bridge_in_circuit, NUM_SHARES},
    domain_tags,
    gadgets::{
        address_ownership::{prove_address_ownership, spend_auth_g_mul},
        poseidon_merkle::{synthesize_poseidon_merkle_path, MerkleSwapGate},
        van_integrity, vote_commitment,
    },
    params::{
        PROPOSAL_AUTHORITY_BITS, RANGE_CHECK_WORD_BITS, SHARE_VALUE_RANGE_WORDS,
        VOTE_COMM_TREE_DEPTH,
    },
    shares_hash::compute_shares_hash_from_comms_in_circuit,
};

// ================================================================
// Constants
// ================================================================

/// Circuit size (2^K rows).
///
/// K=11 (2,048 rows). The ElGamal encryption work moved to the
/// encrypt-choice circuit (ZKP 1.5); this compact cast circuit carries no
/// ECC beyond conditions 3–4. Condition 10′'s bridge hash and conditions
/// 11′–12′'s shares hash and vote commitment run on the dedicated
/// four-column Poseidon track; condition 6 has its own dedicated advice
/// lane. [`voting_crypto_deps::halo2_proofs::dev::CircuitCost`] reports a
/// 1,662-row high-water mark (18.8% headroom) with 24 advice columns.
///
/// Key contributors (rough per-region heights, not per-column sums):
/// - 24-level Merkle path: 24 Poseidon regions stacked sequentially — the
///   tallest single stack in the circuit.
/// - 10-bit Sinsemilla/range-check lookup table: 1,024 fixed rows.
///
/// Run the `row_budget` diagnostic to re-measure after circuit changes:
///   `cargo test vote_proof::circuit::tests::row_budget -- --nocapture --ignored --test-threads=1`
pub const K: u32 = 11;

pub(super) use van_integrity::DOMAIN_VAN;
pub(super) use vote_commitment::DOMAIN_VC_V2;

/// Maximum proposal_id bit index (exclusive upper bound). `proposal_id` is in
/// `[1, MAX_PROPOSAL_ID)`, i.e. valid values are 1–50. Bit 0 is permanently
/// reserved as the sentinel/unset value and is rejected by the non-zero gate
/// in `AuthorityDecrementChip` (`q_cond_6`).
///
/// # Indexing Convention
///
/// `proposal_id` is **1-indexed** throughout the entire stack:
///
/// - **On-chain (`MsgCreateVotingSession`)**: proposals carry `Id = 1, 2, …, N`.
/// - **On-chain (`ValidateProposalId`)**: rejects `proposal_id < 1`.
/// - **Circuit (this file)**: `proposal_id` serves as the bit-position in the
///   51-bit `proposal_authority` bitmask. The `proposal_id != 0` gate ensures
///   bit 0 is never selected, so the effective bit range is `[1, 50]`.
/// - **Client (`zcash_voting::zkp2`)**: must validate `proposal_id` in `[1, 50]`
///   before building the proof.
///
/// Bit 0 of `proposal_authority` is always set and never decremented, acting as
/// a structural invariant rather than a usable slot.
pub(super) const MAX_PROPOSAL_ID: usize = PROPOSAL_AUTHORITY_BITS;

// ================================================================
// Public input offsets (11 field elements).
// ================================================================

/// Public input offset for the VAN nullifier (prevents double-vote).
const VAN_NULLIFIER_PUBLIC_OFFSET: usize = 0;
/// Public input offset for the randomized voting public key (condition 4: Spend Authority).
/// x-coordinate of r_vpk = vsk.ak + [alpha_v] * G.
const R_VPK_X_PUBLIC_OFFSET: usize = 1;
/// Public input offset for r_vpk y-coordinate.
const R_VPK_Y_PUBLIC_OFFSET: usize = 2;
/// Public input offset for the new VAN commitment (with decremented authority).
const VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET: usize = 3;
/// Public input offset for the vote commitment hash.
const VOTE_COMMITMENT_PUBLIC_OFFSET: usize = 4;
/// Public input offset for the vote commitment tree root.
const VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET: usize = 5;
/// Public input offset for the tree anchor height.
// The circuit does not constrain this slot to a witness cell. It is
// transcript-bound metadata whose meaning is authenticated by the verifier's
// caller. In the chain path, the ante handler looks up the commitment root at
// msg.VoteCommTreeAnchorHeight and passes that root as VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET,
// which the circuit does constrain. This keeps the binding between height and
// root in the chain state lookup rather than in this proof.
#[allow(dead_code)]
const VOTE_COMM_TREE_ANCHOR_HEIGHT_PUBLIC_OFFSET: usize = 6;
/// Public input offset for the proposal identifier.
///
/// In-circuit constraint: `proposal_id` is in `[1, MAX_PROPOSAL_ID)` via the
/// authority-decrement lookup. The caller must additionally verify that this
/// ID is in the active proposal set for `voting_round_id`.
const PROPOSAL_ID_PUBLIC_OFFSET: usize = 7;
/// Public input offset for the governance voting round identifier.
///
/// The circuit binds this value into the VAN nullifier, new VAN, and vote
/// commitment, but the caller must authenticate it from the active round's
/// governance announcement.
const VOTING_ROUND_ID_PUBLIC_OFFSET: usize = 8;
/// Public input offset for the compact bridge commitment.
///
/// The verifier must check that this equals the encrypt-choice proof's public
/// bridge value in the same vote bundle; the bridge binds the witnessed
/// weights and selected commitments to the ciphertexts proven by ZKP 1.5.
const BRIDGE_PUBLIC_OFFSET: usize = 9;
/// Public input offset for the active decision bucket count `D`.
///
/// Must be authenticated from the proposal's governance declaration and must
/// match the encrypt-choice proof's public bucket count.
const DECISION_BUCKET_COUNT_PUBLIC_OFFSET: usize = 10;

// ================================================================
// Out-of-circuit helpers
// ================================================================

pub(super) use van_integrity::van_integrity_hash;
pub(super) use vote_commitment::vote_commitment_hash_v2;

/// Returns the domain separator for the VAN nullifier inner hash.
///
/// The tag is defined in [`crate::domain_tags`], the central registry for
/// domain-separation constants and encoding rules.
pub(super) fn domain_van_nullifier() -> pallas::Base {
    domain_tags::vote_authority_spend()
}

/// Out-of-circuit VAN nullifier hash (condition 5).
///
/// ```text
/// van_nullifier = Poseidon(vsk_nk, domain_tag, voting_round_id, vote_authority_note_old)
/// ```
///
/// Single `ConstantLength<4>` call (2 permutations at rate=2).
/// Used by the builder and tests to compute the expected VAN nullifier.
pub(super) fn van_nullifier_hash(
    vsk_nk: pallas::Base,
    voting_round_id: pallas::Base,
    vote_authority_note_old: pallas::Base,
) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<4>, 3, 2>::init().hash([
        vsk_nk,
        domain_van_nullifier(),
        voting_round_id,
        vote_authority_note_old,
    ])
}

// ================================================================
// Config
// ================================================================

/// Configuration for the Vote Proof circuit.
///
/// Holds chip configs for Poseidon (conditions 1, 2, 5, 7, 10′–12′), AddChip
/// (condition 8), LookupRangeCheck (condition 9), ECC (conditions 3, 4),
/// the Merkle swap gate (condition 1), and the custom
/// `AuthorityDecrementChip` (condition 6; see `gadgets/authority_decrement.rs` —
/// uses neither AddChip nor LookupRangeCheck).
#[derive(Clone, Debug)]
pub struct Config {
    /// Public input column (11 field elements).
    primary: Column<InstanceColumn>,
    /// 10 advice columns for private witness data.
    ///
    /// Column layout follows the delegation circuit for consistency:
    /// - `advices[0..5]`: general witness assignment + Merkle swap gate.
    /// - `advices[5]`: Poseidon partial S-box column.
    /// - `advices[6..9]`: Poseidon state columns + AddChip output.
    /// - `advices[9]`: range check running sum.
    advices: [Column<Advice>; 10],
    /// Dedicated advice lane for condition 6's authority-decrement region and
    /// the condition 10′ witness cells, kept off the saturated primary track.
    aux_advices: [Column<Advice>; 10],
    /// Poseidon hash chip configuration.
    ///
    /// P128Pow5T3 with width 3, rate 2. Used for VAN integrity (condition 2),
    /// VAN nullifier (condition 5), new VAN integrity (condition 7), and the
    /// vote commitment Merkle path (condition 1).
    poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    /// Poseidon configuration on the dedicated hash track (conditions
    /// 10′–12′: bridge re-opening, shares hash, vote commitment).
    hash_poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    /// AddChip: constrains `a + b = c` on a single row.
    ///
    /// Uses advices[7] (a), advices[8] (b), advices[6] (c), matching
    /// the delegation circuit's column assignment.
    /// Used in condition 8 (shares sum correctness). Condition 6
    /// (proposal authority decrement) uses the dedicated
    /// `AuthorityDecrementChip` instead — it does not call `AddChip`.
    add_config: AddConfig,
    /// ECC configuration for conditions 3 and 4.
    ///
    /// Condition 3 proves `vpk_pk_d = [ivk_v] * vpk_g_d` via the CommitIvk chain:
    /// `[vsk] * SpendAuthG → ak → CommitIvk(ExtractP(ak), nk, rivk_v) → ivk_v → [ivk_v] * vpk_g_d`.
    /// Shares advice and fixed columns with Poseidon per delegation layout.
    ecc_config: EccConfig<OrchardFixedBases>,
    /// Sinsemilla chip configuration (condition 3: CommitIvk requires Sinsemilla).
    ///
    /// Uses advices[0..5] for Sinsemilla message hashing, advices[6] for
    /// witnessing message pieces, and lagrange_coeffs[0] for the fixed y_Q column.
    /// Also loads the 10-bit lookup table used by LookupRangeCheckConfig.
    sinsemilla_config:
        SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    /// CommitIvk chip configuration (condition 3: canonicity checks on ak || nk).
    ///
    /// Provides the custom gate and decomposition logic for the
    /// Sinsemilla-based `CommitIvk` commitment.
    commit_ivk_config: CommitIvkConfig,
    /// 10-bit lookup range check configuration.
    ///
    /// Uses advices[9] as the running-sum column. Each word is 10 bits,
    /// so `num_words` × 10 gives the total bit-width checked.
    /// Used in condition 9 to ensure each share is in `[0, 2^30)`.
    range_check: LookupRangeCheckConfig<pallas::Base, RANGE_CHECK_WORD_BITS>,
    /// Merkle conditional swap gate (condition 1).
    ///
    /// At each of the 24 Merkle tree levels, conditionally swaps
    /// (current, sibling) into (left, right) based on the position bit.
    /// Uses advices[0..5]: pos_bit, current, sibling, left, right.
    merkle_swap: MerkleSwapGate,
    /// Configuration for condition 6 (Proposal Authority Decrement).
    authority_decrement: AuthorityDecrementConfig,
}

impl Config {
    /// Constructs a Poseidon chip from this configuration.
    ///
    /// Width 3 (P128Pow5T3 state size), rate 2 (absorbs 2 field elements
    /// per permutation — halves the number of rounds vs rate 1).
    fn poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.poseidon_config.clone())
    }

    /// Constructs a Poseidon chip on the condition-10 track.
    fn hash_poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.hash_poseidon_config.clone())
    }

    /// Constructs an AddChip for field element addition (`c = a + b`).
    fn add_chip(&self) -> AddChip {
        AddChip::construct(self.add_config.clone())
    }

    /// Constructs an ECC chip for curve operations in conditions 3 and 4.
    fn ecc_chip(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(self.ecc_config.clone(), CircuitVersion::AnchoredBase)
    }

    /// Constructs a Sinsemilla chip (condition 3: CommitIvk).
    fn sinsemilla_chip(
        &self,
    ) -> SinsemillaChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases> {
        SinsemillaChip::construct(self.sinsemilla_config.clone())
    }

    /// Constructs a CommitIvk chip for canonicity checks (condition 3).
    fn commit_ivk_chip(&self) -> CommitIvkChip {
        CommitIvkChip::construct(self.commit_ivk_config.clone())
    }

    /// Returns the range check configuration (10-bit words).
    fn range_check_config(&self) -> LookupRangeCheckConfig<pallas::Base, RANGE_CHECK_WORD_BITS> {
        self.range_check
    }
}

// ================================================================
// Circuit
// ================================================================

/// The Vote Proof circuit (ZKP #2).
///
/// Proves that a registered voter is casting a valid vote, without
/// revealing which VAN they hold. Contains witness fields and constraint logic
/// for all 12 conditions.
///
/// Condition 4 constrains `r_vpk` in-circuit. The vote signature is verified
/// out-of-circuit under that constrained key.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    // === VAN ownership and spending (conditions 1–5) ===

    // Condition 1 (VAN Membership): Poseidon-based Merkle path from
    // vote_authority_note_old to vote_comm_tree_root.
    /// Merkle authentication path (sibling hashes at each tree level).
    pub(super) vote_comm_tree_path: Value<[pallas::Base; VOTE_COMM_TREE_DEPTH]>,
    /// Leaf position in the vote commitment tree.
    pub(super) vote_comm_tree_position: Value<u32>,

    // Condition 2 (VAN Integrity): two-layer hash matching ZKP 1 (delegation):
    // van_comm_core = Poseidon(DOMAIN_VAN, vpk_g_d.x, vpk_pk_d.x, total_note_value,
    //                          voting_round_id, proposal_authority_old);
    // vote_authority_note_old = Poseidon(van_comm_core, van_comm_rand).
    //
    // Condition 3 (Diversified Address Integrity): vpk_pk_d = [ivk_v] * vpk_g_d
    // where ivk_v = CommitIvk(ExtractP([vsk]*SpendAuthG), vsk.nk, rivk_v).
    // Full affine points are needed for condition 3's ECC operations;
    // x-coordinates are extracted in-circuit for Poseidon hashing (conditions 2, 7).
    /// Voting public key — diversified base point (from DiversifyHash(d)).
    /// This is the vpk_g_d component of the voting hotkey address.
    /// Condition 3 performs `[ivk_v] * vpk_g_d` to derive vpk_pk_d.
    pub(super) vpk_g_d: Value<pallas::Affine>,
    /// Voting public key — diversified transmission key (pk_d = [ivk_v] * g_d).
    /// This is the vpk_pk_d component of the voting hotkey address.
    /// Condition 3 (Diversified Address Integrity) constrains this to equal `[ivk_v] * vpk_g_d`.
    pub(super) vpk_pk_d: Value<pallas::Affine>,
    /// The voter's total delegated weight, denominated in ballots
    /// (1 ballot = 0.125 ZEC; converted from zatoshi by ZKP #1 condition 8 —
    /// see the delegation README §8 for the proven relation).
    pub(super) total_note_value: Value<pallas::Base>,
    // Condition 6:
    /// Remaining proposal authority bitmask in the old VAN.
    pub(super) proposal_authority_old: Value<pallas::Base>,
    /// Blinding randomness for the VAN commitment.
    pub(super) van_comm_rand: Value<pallas::Base>,
    /// The old VAN commitment (Poseidon hash output). Used as the Merkle
    /// leaf in condition 1 and constrained to equal the derived hash here.
    pub(super) vote_authority_note_old: Value<pallas::Base>,

    // Condition 3 (Diversified Address Integrity): prover controls the VAN address.
    // vpk_pk_d = [ivk_v] * vpk_g_d
    //   where ivk_v = CommitIvk_rivk_v(ExtractP([vsk]*SpendAuthG), vsk.nk)
    /// Voting spending key (scalar for ECC multiplication).
    /// Used in condition 3 for `[vsk] * SpendAuthG`.
    pub(super) vsk: Value<pallas::Scalar>,
    /// CommitIvk randomness for the ivk_v derivation (condition 3).
    /// Used as the blinding scalar in `CommitIvk(ak, nk, rivk_v)`.
    pub(super) rivk_v: Value<pallas::Scalar>,
    /// Spend auth randomizer for condition 4: r_vpk = vsk.ak + [alpha_v] * G.
    pub(super) alpha_v: Value<pallas::Scalar>,

    // Condition 5 (VAN Nullifier Integrity): nullifier deriving key.
    // Also used in condition 3 as the nk input to CommitIvk.
    /// Nullifier deriving key derived from vsk.
    pub(super) vsk_nk: Value<pallas::Base>,

    // Condition 6 (Proposal Authority Decrement): one_shifted = 2^proposal_id.
    /// `2^proposal_id`, supplied as a private witness and constrained by a lookup.
    ///
    /// Field arithmetic cannot express variable-exponent exponentiation as a
    /// polynomial gate, so the prover witnesses `one_shifted` directly. The lookup
    /// table `(0,1), (1,2), ..., (50,2^50)` then proves `one_shifted == 2^proposal_id`.
    /// The bit-decomposition region uses this value to compute
    /// `proposal_authority_new = proposal_authority_old - one_shifted`.
    pub(super) one_shifted: Value<pallas::Base>,

    // === Vote commitment construction (conditions 8–12) ===

    // Condition 8 (Shares Sum): sum(shares_1..16) = total_note_value.
    // Condition 9 (Shares Range): each share in [0, 2^30).
    /// Voting share vector (16 random shares that sum to total_note_value).
    /// The decomposition is chosen by the prover for amount privacy: the
    /// on-chain El Gamal ciphertexts reveal no weight fingerprint.
    pub(super) shares: [Value<pallas::Base>; 16],

    // Condition 10′ (Bridge Re-Opening): per-share selected commitments from
    // the encrypt-choice bundle. Each commits one share's blind and all 8
    // bucket ciphertext coordinates (see `crate::bridge`); the encrypt-choice
    // proof (ZKP 1.5) constrains them to real ElGamal encryptions, and this
    // circuit re-derives the public bridge from them and the witnessed
    // shares, binding both proofs to the same values.
    /// Per-share selected commitments (ZKP 1.5 outputs).
    pub(super) selected_commitments: [Value<pallas::Base>; NUM_SHARES],
}

impl Circuit {
    /// Creates a circuit with conditions 1–3 and 5–7 witnesses populated.
    ///
    /// All other witness fields are set to `Value::unknown()`.
    /// - Condition 1 uses `vote_authority_note_old` as the Merkle leaf,
    ///   with `vote_comm_tree_path` and `vote_comm_tree_position` for
    ///   the authentication path.
    /// - Condition 2 binds `vote_authority_note_old` to the Poseidon hash
    ///   of its components (using x-coordinates extracted from vpk_g_d, vpk_pk_d).
    /// - Condition 3 proves diversified address integrity via CommitIvk chain:
    ///   `[vsk] * SpendAuthG → ak → CommitIvk(ak, nk, rivk_v) → ivk_v → [ivk_v] * vpk_g_d = vpk_pk_d`.
    /// - Condition 5 reuses `vote_authority_note_old` and `voting_round_id`.
    /// - Condition 6 derives `proposal_authority_new` from
    ///   `proposal_authority_old`.
    /// - Condition 7 reuses all condition 2 witnesses except
    ///   `proposal_authority_old`, which is replaced by the
    ///   in-circuit `proposal_authority_new` from condition 6.
    pub(super) fn with_van_witnesses(
        vote_comm_tree_path: Value<[pallas::Base; VOTE_COMM_TREE_DEPTH]>,
        vote_comm_tree_position: Value<u32>,
        vpk_g_d: Value<pallas::Affine>,
        vpk_pk_d: Value<pallas::Affine>,
        total_note_value: Value<pallas::Base>,
        proposal_authority_old: Value<pallas::Base>,
        van_comm_rand: Value<pallas::Base>,
        vote_authority_note_old: Value<pallas::Base>,
        vsk: Value<pallas::Scalar>,
        rivk_v: Value<pallas::Scalar>,
        vsk_nk: Value<pallas::Base>,
        alpha_v: Value<pallas::Scalar>,
    ) -> Self {
        Circuit {
            vote_comm_tree_path,
            vote_comm_tree_position,
            vpk_g_d,
            vpk_pk_d,
            total_note_value,
            proposal_authority_old,
            van_comm_rand,
            vote_authority_note_old,
            vsk,
            rivk_v,
            alpha_v,
            vsk_nk,
            ..Default::default()
        }
    }
}

impl plonk::Circuit<pallas::Base> for Circuit {
    type Config = Config;
    type FloorPlanner = floor_planner::V1;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        // The primary 10 advice columns match the delegation circuit layout.
        // A second 10-column lane hosts condition 6's authority-decrement
        // region and the condition 10′ witness cells so they overlap the
        // saturated primary track. Four more columns hold the dedicated
        // Poseidon hash track (conditions 10′–12′). The chips tile within
        // the primary 10-column window:
        //
        //   advices[0..5]  — general witness assignment, Sinsemilla pair 1
        //                    message columns, and the Merkle swap gate
        //                    (pos_bit / current / sibling / left / right).
        //   advices[5]     — Poseidon partial S-box column; also the start of
        //                    Sinsemilla pair 2 main columns (advices[5..10]).
        //   advices[6..9]  — Poseidon width-3 state columns; AddChip uses these
        //                    same three columns (a=advices[7], b=advices[8],
        //                    c=advices[6]).
        //   advices[9]     — LookupRangeCheck running-sum column.
        let advices: [Column<Advice>; 10] = core::array::from_fn(|_| meta.advice_column());
        for col in &advices {
            meta.enable_equality(*col);
        }
        let aux_advices: [Column<Advice>; 10] = core::array::from_fn(|_| meta.advice_column());
        for col in &aux_advices {
            meta.enable_equality(*col);
        }
        let hash_advices: [Column<Advice>; 4] = core::array::from_fn(|_| meta.advice_column());
        // `Pow5Chip::configure` equality-enables the three state columns used
        // for cross-region handoffs. The partial-S-box column is internal
        // scratch space and must not be added to the permutation argument.

        // Instance column for public inputs.
        let primary = meta.instance_column();
        meta.enable_equality(primary);

        // 8 fixed columns shared between ECC and Poseidon chips.
        // Indices 0–1: Lagrange coefficients (ECC chip only).
        // Indices 2–4: Poseidon round constants A (rc_a).
        // Indices 5–7: Poseidon round constants B (rc_b).
        let lagrange_coeffs: [Column<Fixed>; 8] = core::array::from_fn(|_| meta.fixed_column());
        let rc_a = lagrange_coeffs[2..5].try_into().unwrap();
        let rc_b = lagrange_coeffs[5..8].try_into().unwrap();

        // Dedicated constants column, separate from the Lagrange coefficient
        // columns used by the ECC chip. This prevents collisions between
        // the ECC chip's fixed-base scalar multiplication tables and the
        // constant-zero cells created by strict range checks.
        let constants = meta.fixed_column();
        meta.enable_constant(constants);

        // AddChip: constrains `a + b = c` in a single row.
        // Column assignment matches the delegation circuit:
        //   a = advices[7], b = advices[8], c = advices[6].
        let add_config = AddChip::configure(meta, advices[7], advices[8], advices[6]);

        // Lookup table columns for Sinsemilla (3 columns) and range checks.
        // The first column (table_idx) is shared between Sinsemilla and
        // LookupRangeCheckConfig. SinsemillaChip::load populates all three
        // during synthesis (replacing the manual table loading).
        let table_idx = meta.lookup_table_column();
        let lookup = (
            table_idx,
            meta.lookup_table_column(),
            meta.lookup_table_column(),
        );

        // Range check configuration: 10-bit lookup words in advices[9].
        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], table_idx);

        // Primary ECC chip for conditions 3 and 4. It shares columns with
        // Poseidon per the delegation circuit layout.
        let ecc_config =
            EccChip::<OrchardFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);

        // Sinsemilla chip: required by CommitIvk for condition 3.
        // Uses advices[0..5] for Sinsemilla message hashing, advices[6] for
        // witnessing message pieces, and lagrange_coeffs[0] for the fixed
        // y_Q column. Shares the lookup table with LookupRangeCheckConfig.
        let sinsemilla_config = SinsemillaChip::configure(
            meta,
            advices[..5].try_into().unwrap(),
            advices[6],
            lagrange_coeffs[0],
            lookup,
            range_check,
            false,
        );

        // CommitIvk chip: canonicity checks on the ak || nk decomposition
        // inside the CommitIvk Sinsemilla commitment (condition 3).
        let commit_ivk_config = CommitIvkChip::configure(meta, advices);

        // Poseidon chip: P128Pow5T3 with width 3, rate 2.
        // State columns: advices[6..9] (3 columns for the width-3 state).
        // Partial S-box column: advices[5].
        // Round constants: lagrange_coeffs[2..5] (rc_a), [5..8] (rc_b).
        let poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
            meta,
            advices[6..9].try_into().unwrap(),
            advices[5],
            rc_a,
            rc_b,
        );
        let hash_round_constants: [Column<Fixed>; 6] =
            core::array::from_fn(|_| meta.fixed_column());
        let hash_poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
            meta,
            hash_advices[..3].try_into().unwrap(),
            hash_advices[3],
            hash_round_constants[..3].try_into().unwrap(),
            hash_round_constants[3..].try_into().unwrap(),
        );

        // Merkle conditional swap gate (condition 1).
        let merkle_swap = MerkleSwapGate::configure(
            meta,
            [advices[0], advices[1], advices[2], advices[3], advices[4]],
        );

        // Condition 6: Proposal Authority Decrement. The dedicated aux lane
        // lets the floor planner place its 52-row region alongside the
        // saturated primary track.
        let authority_decrement = AuthorityDecrementChip::configure(meta, aux_advices);

        Config {
            primary,
            advices,
            aux_advices,
            poseidon_config,
            hash_poseidon_config,
            add_config,
            ecc_config,
            sinsemilla_config,
            commit_ivk_config,
            range_check,
            merkle_swap,
            authority_decrement,
        }
    }

    #[allow(non_snake_case)]
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), plonk::Error> {
        // ---------------------------------------------------------------
        // Load the Sinsemilla generator lookup table.
        //
        // Populates the 10-bit lookup table and Sinsemilla generator
        // points. Required by CommitIvk (condition 3), and also provides
        // the range check table used by conditions 5 and 8.
        // ---------------------------------------------------------------
        SinsemillaChip::load(config.sinsemilla_config.clone(), &mut layouter)?;

        // Load (proposal_id, 2^proposal_id) lookup table for condition 6.
        AuthorityDecrementChip::load_table(&config.authority_decrement, &mut layouter)?;

        // Construct the primary ECC chip (used in conditions 3 and 4).
        let ecc_chip = config.ecc_chip();

        // ---------------------------------------------------------------
        // Witness assignment for condition 2.
        // ---------------------------------------------------------------

        // Copy voting_round_id from the instance column into an advice cell.
        // This creates an equality constraint between the advice cell and the
        // instance at offset VOTING_ROUND_ID_PUBLIC_OFFSET, ensuring the in-circuit value
        // matches the public input.
        let voting_round_id = layouter.assign_region(
            || "copy voting_round_id from instance",
            |mut region| {
                region.assign_advice_from_instance(
                    || "voting_round_id",
                    config.primary,
                    VOTING_ROUND_ID_PUBLIC_OFFSET,
                    config.advices[0],
                    0,
                )
            },
        )?;
        // Clone for condition 12 (vote commitment integrity) before
        // condition 2 consumes the original via van_integrity_poseidon.
        let voting_round_id_cond12 = voting_round_id.clone();

        // Witness vpk_g_d as a full non-identity curve point (condition 3 needs
        // the point for variable-base ECC mul; conditions 2/6 need the x-coordinate
        // for Poseidon hashing).
        let vpk_g_d_point = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "witness vpk_g_d"),
            self.vpk_g_d.map(|p| p),
        )?;
        let vpk_g_d = vpk_g_d_point.extract_p().inner().clone();

        // Witness vpk_pk_d as a full non-identity curve point (condition 3
        // constrains the derived point to equal this; conditions 2/6 use x-coordinate).
        let vpk_pk_d_point = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "witness vpk_pk_d"),
            self.vpk_pk_d.map(|p| p),
        )?;
        let vpk_pk_d = vpk_pk_d_point.extract_p().inner().clone();

        let total_note_value = assign_free_advice(
            layouter.namespace(|| "witness total_note_value"),
            config.advices[0],
            self.total_note_value,
        )?;

        let proposal_authority_old = assign_free_advice(
            layouter.namespace(|| "witness proposal_authority_old"),
            config.advices[0],
            self.proposal_authority_old,
        )?;

        let van_comm_rand = assign_free_advice(
            layouter.namespace(|| "witness van_comm_rand"),
            config.advices[0],
            self.van_comm_rand,
        )?;

        let vote_authority_note_old = assign_free_advice(
            layouter.namespace(|| "witness vote_authority_note_old"),
            config.advices[0],
            self.vote_authority_note_old,
        )?;

        // DOMAIN_VAN — constant-constrained so the value is baked into the
        // verification key and cannot be altered by a malicious prover.
        let domain_van = layouter.assign_region(
            || "DOMAIN_VAN constant",
            |mut region| {
                region.assign_advice_from_constant(
                    || "domain_van",
                    config.advices[0],
                    0,
                    pallas::Base::from(DOMAIN_VAN),
                )
            },
        )?;

        // ---------------------------------------------------------------
        // Witness assignment for conditions 3 and 4.
        //
        // vsk_nk is shared between condition 3 (CommitIvk input) and
        // condition 5 (VAN nullifier). Witnessed here so it's available
        // for condition 3 which runs before condition 5.
        // ---------------------------------------------------------------

        // Private witness: nullifier deriving key (shared by conditions 3, 4).
        let vsk_nk = assign_free_advice(
            layouter.namespace(|| "witness vsk_nk"),
            config.advices[0],
            self.vsk_nk,
        )?;

        // Clone cells that are consumed by condition 2's Poseidon hash but
        // reused in later conditions:
        // - vote_authority_note_old: also used in condition 1 (Merkle leaf).
        // - voting_round_id: also used in condition 5 (VAN nullifier).
        // - vpk_g_d, vpk_pk_d, total_note_value, voting_round_id,
        //   van_comm_rand, domain_van: also used in condition 7 (new VAN integrity).
        // - total_note_value: also used in condition 8 (shares sum check).
        // - vsk_nk: also used in condition 5 (VAN nullifier).
        let vote_authority_note_old_cond1 = vote_authority_note_old.clone();
        let voting_round_id_cond4 = voting_round_id.clone();
        let domain_van_cond6 = domain_van.clone();
        let vpk_g_d_cond6 = vpk_g_d.clone();
        let vpk_pk_d_cond6 = vpk_pk_d.clone();
        let total_note_value_cond6 = total_note_value.clone();
        let total_note_value_cond8 = total_note_value.clone();
        let voting_round_id_cond6 = voting_round_id.clone();
        let van_comm_rand_cond6 = van_comm_rand.clone();
        let vsk_nk_cond4 = vsk_nk.clone();

        // ---------------------------------------------------------------
        // Condition 2: VAN Integrity (ZKP 1–compatible two-layer hash).
        // van_comm_core = Poseidon(DOMAIN_VAN, vpk_g_d, vpk_pk_d, total_note_value,
        //                          voting_round_id, proposal_authority_old)
        // vote_authority_note_old = Poseidon(van_comm_core, van_comm_rand)
        // ---------------------------------------------------------------

        let derived_van = van_integrity::van_integrity_poseidon(
            &config.poseidon_config,
            &mut layouter,
            "Old VAN integrity",
            domain_van,
            vpk_g_d,
            vpk_pk_d,
            total_note_value,
            voting_round_id,
            proposal_authority_old.clone(),
            van_comm_rand,
        )?;

        // Constrain: derived VAN hash == witnessed vote_authority_note_old.
        layouter.assign_region(
            || "VAN integrity check",
            |mut region| region.constrain_equal(derived_van.cell(), vote_authority_note_old.cell()),
        )?;

        // ---------------------------------------------------------------
        // Condition 3: Diversified Address Integrity.
        //
        // vpk_pk_d = [ivk_v] * vpk_g_d where ivk_v = CommitIvk(ExtractP([vsk]*SpendAuthG), vsk_nk, rivk_v).
        // ---------------------------------------------------------------
        let vsk_scalar = ScalarFixed::new(
            ecc_chip.clone(),
            layouter.namespace(|| "cond3 vsk"),
            self.vsk,
        )?;
        let vsk_ak_point = spend_auth_g_mul(
            ecc_chip.clone(),
            layouter.namespace(|| "cond3 [vsk]G"),
            "cond3: [vsk] SpendAuthG",
            vsk_scalar,
        )?;
        let ak = vsk_ak_point.extract_p().inner().clone();
        let rivk_v_scalar = ScalarFixed::new(
            ecc_chip.clone(),
            layouter.namespace(|| "cond3 rivk_v"),
            self.rivk_v,
        )?;
        prove_address_ownership(
            config.sinsemilla_chip(),
            ecc_chip.clone(),
            config.commit_ivk_chip(),
            layouter.namespace(|| "cond3 address"),
            "cond3",
            ak,
            vsk_nk.clone(),
            rivk_v_scalar,
            &vpk_g_d_point,
            &vpk_pk_d_point,
        )?;

        // ---------------------------------------------------------------
        // Condition 4: Spend authority.
        // r_vpk = [alpha_v] * SpendAuthG + vsk_ak_point
        // ---------------------------------------------------------------
        // Spend authority: proves that the public r_vpk is a valid rerandomization of the prover's ak.
        // The out-of-circuit verifier checks that the vote signature is valid under r_vpk,
        // so this links the ZKP to the signature without revealing ak.
        //
        // Uses the shared gadget from crate::gadgets::spend_authority – a 1:1 copy of
        // the upstream Orchard spend authority check:
        //   https://github.com/zcash/orchard/blob/main/src/circuit.rs#L542-L558
        crate::gadgets::spend_authority::prove_spend_authority(
            ecc_chip.clone(),
            layouter.namespace(|| "cond4 spend authority"),
            self.alpha_v,
            &vsk_ak_point,
            config.primary,
            R_VPK_X_PUBLIC_OFFSET,
            R_VPK_Y_PUBLIC_OFFSET,
        )?;

        // ---------------------------------------------------------------
        // Condition 1: VAN Membership.
        //
        // MerklePath(vote_authority_note_old, position, path) = vote_comm_tree_root
        //
        // Poseidon-based Merkle path verification (24 levels). At each
        // level, the position bit determines child ordering: if bit=0,
        // current is the left child; if bit=1, current is the right child.
        //
        // The leaf is vote_authority_note_old, which is already constrained
        // to be a correct Poseidon hash by condition 2. This creates a
        // binding: the VAN integrity check and the Merkle membership proof
        // are tied to the same commitment.
        //
        // The hash function is Poseidon(left, right) with no level tag,
        // matching vote_commitment_tree::MerkleHashVote::combine.
        // ---------------------------------------------------------------
        {
            let root = synthesize_poseidon_merkle_path::<VOTE_COMM_TREE_DEPTH>(
                &config.merkle_swap,
                &config.poseidon_config,
                &mut layouter,
                config.advices[0],
                vote_authority_note_old_cond1,
                self.vote_comm_tree_position,
                self.vote_comm_tree_path,
                "cond1: merkle",
            )?;

            // Bind the computed Merkle root to the VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET
            // public input. The verifier checks that the voter's VAN is
            // a leaf in the published vote commitment tree.
            layouter.constrain_instance(
                root.cell(),
                config.primary,
                VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET,
            )?;
        }

        // ---------------------------------------------------------------
        // Witness assignment for condition 5.
        //
        // vsk_nk was already witnessed before condition 3 (shared between
        // conditions 3 and 5). The vsk_nk_cond4 clone is used here.
        // ---------------------------------------------------------------

        // "vote authority spend" domain tag — constant-constrained so the
        // value is baked into the verification key.
        let domain_van_nf = layouter.assign_region(
            || "DOMAIN_VAN_NULLIFIER constant",
            |mut region| {
                region.assign_advice_from_constant(
                    || "domain_van_nullifier",
                    config.advices[0],
                    0,
                    domain_van_nullifier(),
                )
            },
        )?;

        // ---------------------------------------------------------------
        // Condition 5: VAN Nullifier Integrity.
        // van_nullifier = Poseidon(vsk_nk, domain_tag, voting_round_id, vote_authority_note_old)
        //
        // Single ConstantLength<4> Poseidon hash (2 permutations at rate=2).
        //
        // voting_round_id and vote_authority_note_old are reused from
        // condition 2 via cell equality — these cells flow directly into
        // the Poseidon state without being re-witnessed.
        // ---------------------------------------------------------------

        let van_nullifier = {
            let hasher = PoseidonHash::<
                pallas::Base,
                _,
                poseidon::P128Pow5T3,
                ConstantLength<4>,
                3, // WIDTH
                2, // RATE
            >::init(
                config.poseidon_chip(),
                layouter.namespace(|| "VAN nullifier Poseidon init"),
            )?;
            hasher.hash(
                layouter.namespace(|| "Poseidon(vsk_nk, domain, round_id, van_old)"),
                [
                    vsk_nk_cond4,
                    domain_van_nf,
                    voting_round_id_cond4,
                    vote_authority_note_old,
                ],
            )?
        };

        // Bind the derived nullifier to the VAN_NULLIFIER_PUBLIC_OFFSET public input.
        // The verifier checks that the prover's computed nullifier matches
        // the publicly posted value, preventing double-voting.
        layouter.constrain_instance(
            van_nullifier.cell(),
            config.primary,
            VAN_NULLIFIER_PUBLIC_OFFSET,
        )?;

        // ---------------------------------------------------------------
        // Condition 6: Proposal Authority Decrement (bit decomposition).
        //
        // Step 1: Decompose proposal_authority_old into 51 bits b_i (boolean).
        // Step 2: Selector sel_i = 1 iff proposal_id == i; exactly one active;
        //         selected bit = sum(sel_i * b_i) = 1 (voter has authority).
        // Step 3: b_new_i = b_i*(1-sel_i); recompose to proposal_authority_new.
        // No diff/gap range check; decomposition proves [0, 2^51).
        // ---------------------------------------------------------------

        // Copy proposal_id from the public instance into an advice cell.
        let proposal_id = layouter.assign_region(
            || "copy proposal_id from instance",
            |mut region| {
                region.assign_advice_from_instance(
                    || "proposal_id",
                    config.primary,
                    PROPOSAL_ID_PUBLIC_OFFSET,
                    config.advices[0],
                    0,
                )
            },
        )?;

        let proposal_authority_new = AuthorityDecrementChip::assign(
            &config.authority_decrement,
            &mut layouter,
            proposal_id.clone(),
            proposal_authority_old,
            self.one_shifted,
        )?;

        // ---------------------------------------------------------------
        // Condition 7: New VAN Integrity (ZKP 1–compatible two-layer hash).
        //
        // Same structure as condition 2; proposal_authority_new (from
        // condition 6) replaces proposal_authority_old. vpk_g_d and vpk_pk_d
        // are unchanged (same diversified address).
        // ---------------------------------------------------------------

        let derived_van_new = van_integrity::van_integrity_poseidon(
            &config.poseidon_config,
            &mut layouter,
            "New VAN integrity",
            domain_van_cond6,
            vpk_g_d_cond6,
            vpk_pk_d_cond6,
            total_note_value_cond6,
            voting_round_id_cond6,
            proposal_authority_new,
            van_comm_rand_cond6,
        )?;

        // Bind the derived new VAN to the VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET public input.
        // The verifier checks that the new VAN commitment posted on-chain is
        // correctly formed with decremented proposal authority.
        layouter.constrain_instance(
            derived_van_new.cell(),
            config.primary,
            VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET,
        )?;

        // ---------------------------------------------------------------
        // Condition 8: Shares Sum Correctness.
        //
        // sum(share_0, ..., share_15) = total_note_value
        //
        // Proves the voting share decomposition is consistent with the
        // total delegated weight (in ballots). Uses 15 chained AddChip additions:
        //   partial_1  = share_0  + share_1
        //   partial_2  = partial_1  + share_2
        //   ...
        //   shares_sum = partial_14 + share_15
        // Then constrains shares_sum == total_note_value (from condition 2).
        // ---------------------------------------------------------------

        // Witness the 16 plaintext shares. These cells are also used
        // by condition 9 (range check) and condition 11 (El Gamal
        // encryption inputs).
        let share_cells: [_; 16] = (0..16usize)
            .map(|i| {
                assign_free_advice(
                    layouter.namespace(|| format!("witness share_{i}")),
                    config.advices[0],
                    self.shares[i],
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("always 16 elements");

        // Chain 15 additions: share_0 + share_1 + ... + share_15.
        let shares_sum = share_cells[1..].iter().enumerate().try_fold(
            share_cells[0].clone(),
            |acc, (i, share)| {
                config.add_chip().add(
                    layouter.namespace(|| format!("shares sum step {}", i + 1)),
                    &acc,
                    share,
                )
            },
        )?;

        // Constrain: shares_sum == total_note_value.
        // This ensures the 16 shares decompose the voter's total delegated
        // weight without creating or destroying value.
        layouter.assign_region(
            || "shares sum == total_note_value",
            |mut region| region.constrain_equal(shares_sum.cell(), total_note_value_cond8.cell()),
        )?;

        // ---------------------------------------------------------------
        // Condition 9: Shares Range.
        //
        // Each share_i in [0, 2^30)
        //
        // Motivation: the sum constraint (condition 8) holds in the
        // base field F_p, but El Gamal encryption operates in the
        // scalar field F_q via `share_i * G`. For Pallas, p ≠ q, so a
        // large base-field element (e.g. p − 50) reduces to a different
        // value mod q, breaking the correspondence between the
        // constrained sum and the encrypted values. Bounding each share
        // to [0, 2^30) guarantees both representations agree (no
        // modular reduction in either field), so the homomorphic tally
        // faithfully reflects condition 8's sum.
        //
        // Secondary benefit: after accumulation the EA decrypts to
        // `total_value * G` and must solve a bounded DLOG (BSGS) to
        // recover `total_value`. Bounded shares keep the per-decision
        // aggregate small enough for efficient recovery.
        //
        // Shares are denominated in ballots (1 ballot = 0.125 ZEC),
        // converted from zatoshi in ZKP #1's condition 8 (ballot
        // scaling). Uses 3 × 10-bit lookup words with strict mode,
        // giving [0, 2^30). halo2_gadgets v0.5's `short_range_check`
        // is private, so exact non-10-bit-aligned bounds (e.g. 24-bit)
        // are unavailable. 2^30 ballots ≈ 134M ZEC — well above the
        // 21M ZEC supply — so the bound is never binding in practice.
        //
        // If a share exceeds 2^30 (or wraps around the field, e.g.
        // from underflow), the 3-word decomposition produces a non-zero
        // z_3 running sum, which fails the strict check.
        // ---------------------------------------------------------------

        // Share cells are cloned because copy_check takes ownership;
        // the originals remain available for condition 11 (El Gamal).
        for (i, cell) in share_cells.iter().enumerate() {
            config.range_check_config().copy_check(
                layouter.namespace(|| format!("share_{i} < 2^30")),
                cell.clone(),
                SHARE_VALUE_RANGE_WORDS,
                true, // strict: running sum terminates at 0
            )?;
        }

        // ---------------------------------------------------------------
        // Condition 10′: Bridge Re-Opening.
        //
        // bridge = Poseidon(ENCRYPT_CHOICE_BRIDGE_DOMAIN, voting_round_id,
        //                   proposal_id, decision_bucket_count,
        //                   w_0, selected_comm_0, ..., w_15, selected_comm_15)
        //
        // The 16 selected commitments are witnessed from the encrypt-choice
        // bundle; the weights are the same condition-8/9 share cells that sum
        // to total_note_value, so the pre-encrypted weights proven by
        // ZKP 1.5 are exactly the shares this VAN authorizes. The derived
        // bridge is bound to BRIDGE_PUBLIC_OFFSET, and the verifier checks
        // it equals the encrypt-choice proof's public bridge.
        // ---------------------------------------------------------------

        let selected_commitments: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES] = (0
            ..NUM_SHARES)
            .map(|i| {
                assign_free_advice(
                    layouter.namespace(|| format!("witness selected_comm[{i}]")),
                    config.aux_advices[0],
                    self.selected_commitments[i],
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("always 16 elements");
        let selected_commitments_cond11: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES] =
            core::array::from_fn(|i| selected_commitments[i].clone());

        let decision_bucket_count = layouter.assign_region(
            || "copy decision_bucket_count from instance",
            |mut region| {
                region.assign_advice_from_instance(
                    || "decision_bucket_count",
                    config.primary,
                    DECISION_BUCKET_COUNT_PUBLIC_OFFSET,
                    config.aux_advices[1],
                    0,
                )
            },
        )?;
        let decision_bucket_count_cond12 = decision_bucket_count.clone();

        let bridge = compute_bridge_in_circuit(
            config.hash_poseidon_chip(),
            layouter.namespace(|| "cond10: bridge"),
            config.aux_advices[0],
            voting_round_id_cond12.clone(),
            proposal_id.clone(),
            decision_bucket_count,
            share_cells,
            selected_commitments,
        )?;
        layouter.constrain_instance(bridge.cell(), config.primary, BRIDGE_PUBLIC_OFFSET)?;

        // ---------------------------------------------------------------
        // Condition 11′: Shares Hash Integrity.
        //
        // shares_hash = Poseidon(selected_comm_0, ..., selected_comm_15)
        //
        // shares_hash is an internal wire; it is not bound to the instance
        // column. Condition 12′ folds it into the public vote commitment.
        // ZKP #3 recomputes the same hash from private witnesses when a
        // share is revealed.
        // ---------------------------------------------------------------

        let shares_hash = compute_shares_hash_from_comms_in_circuit(
            config.hash_poseidon_chip(),
            layouter.namespace(|| "cond11: shares hash"),
            selected_commitments_cond11,
        )?;

        // ---------------------------------------------------------------
        // Condition 12′: Vote Commitment Integrity (v2).
        //
        // vote_commitment = Poseidon(DOMAIN_VC_V2, voting_round_id,
        //                            shares_hash, proposal_id,
        //                            decision_bucket_count)
        //
        // The plaintext vote decision of v1 is gone: the decision is bound
        // only through the committed one-hot ciphertext vectors inside
        // shares_hash. Binding decision_bucket_count prevents replaying a
        // commitment under a proposal with a different option count.
        //
        // This is the value posted on-chain and later inserted into the
        // vote commitment tree. ZKP #3 (share reveal) opens individual
        // shares from this commitment.
        // ---------------------------------------------------------------

        // DOMAIN_VC_V2 — constant-constrained so the value is baked into the
        // verification key and cannot be altered by a malicious prover.
        let domain_vc = layouter.assign_region(
            || "DOMAIN_VC_V2 constant",
            |mut region| {
                region.assign_advice_from_constant(
                    || "domain_vc_v2",
                    config.advices[0],
                    0,
                    pallas::Base::from(DOMAIN_VC_V2),
                )
            },
        )?;

        let vote_commitment = vote_commitment::vote_commitment_poseidon(
            &config.hash_poseidon_config,
            &mut layouter,
            "cond12",
            domain_vc,
            voting_round_id_cond12,
            shares_hash,
            proposal_id,
            decision_bucket_count_cond12,
        )?;

        // Bind the derived vote commitment to the VOTE_COMMITMENT_PUBLIC_OFFSET public input.
        layouter.constrain_instance(
            vote_commitment.cell(),
            config.primary,
            VOTE_COMMITMENT_PUBLIC_OFFSET,
        )?;

        Ok(())
    }
}

// ================================================================
// Instance (public inputs)
// ================================================================

/// Public inputs to the Vote Proof circuit (11 field elements).
///
/// The voting client (prover) chooses these values when assembling the
/// proof; the verifier accepts them as the binding the proof must
/// satisfy and checks the proof without seeing any private witnesses.
/// The relationship is asymmetric: a malicious-custody client can
/// choose any public-input vector it likes, so the verifier must source
/// the *correct* values from authenticated chain state (see
/// [`crate::vote_proof::prove::verify_vote_proof`] for which fields
/// require caller authentication versus which are proof-attested
/// outputs).
///
/// Binding contract: `shares_hash` is deliberately absent from this public
/// instance vector. The circuit computes it as an internal condition-11′ cell
/// and exposes it to the verifier only through `vote_commitment`.
#[derive(Clone, Debug)]
pub struct Instance {
    /// The nullifier of the old VAN being spent (prevents double-vote).
    pub van_nullifier: pallas::Base,
    /// Randomized voting public key (condition 4): x-coordinate of r_vpk = vsk.ak + [alpha_v] * G.
    pub r_vpk_x: pallas::Base,
    /// Randomized voting public key: y-coordinate.
    pub r_vpk_y: pallas::Base,
    /// The new VAN commitment (with decremented proposal authority).
    pub vote_authority_note_new: pallas::Base,
    /// The vote commitment hash.
    pub vote_commitment: pallas::Base,
    /// Root of the vote commitment tree at anchor height.
    pub vote_comm_tree_root: pallas::Base,
    /// Caller-authenticated chain height used to source `vote_comm_tree_root`.
    ///
    /// This public input is transcript-bound but not constrained to a witness
    /// cell. Verifiers must check that `vote_comm_tree_root` is the chain root
    /// at this height.
    pub vote_comm_tree_anchor_height: pallas::Base,
    /// Governance session parameter: which proposal this vote is for.
    ///
    /// The circuit constrains this to `[1, 50]` through condition 6 and binds
    /// it into the new VAN and vote commitment. The verifier must separately
    /// check that it is active for `voting_round_id`.
    pub proposal_id: pallas::Base,
    /// Governance session parameter: the voting round identifier.
    ///
    /// The circuit binds this into the VAN nullifier, new VAN, and vote
    /// commitment, but cannot authenticate that it is the active round.
    pub voting_round_id: pallas::Base,
    /// Compact bridge commitment shared with the encrypt-choice proof.
    ///
    /// Proof-attested here, but its protocol meaning comes from the vote
    /// bundle: the verifier must check it equals the encrypt-choice proof's
    /// public bridge value.
    pub bridge: pallas::Base,
    /// Active decision bucket count `D` for the proposal.
    ///
    /// Must be authenticated from the proposal's governance declaration and
    /// must equal the encrypt-choice proof's public bucket count. The
    /// verifier must additionally reject `D < 2`.
    pub decision_bucket_count: pallas::Base,
}

impl Instance {
    /// Number of public inputs serialized by [`Self::to_halo2_instance`].
    pub const NUM_PUBLIC_INPUTS: usize = 11;

    /// Constructs an [`Instance`] from its constituent parts.
    ///
    /// Callers should authenticate `vote_comm_tree_root`,
    /// `vote_comm_tree_anchor_height`, `proposal_id`, `voting_round_id`, and
    /// `decision_bucket_count` out-of-band before passing them here.
    /// `proposal_id` must be active for `voting_round_id`; the circuit only
    /// checks the authority-bit index range. See
    /// [`crate::vote_proof::prove::verify_vote_proof`] for the trust
    /// contract. The remaining fields are proof-attested outputs derived
    /// outside the circuit but constrained in-circuit against authenticated
    /// inputs and private witnesses; `bridge` additionally requires the
    /// bundle-level equality check against the encrypt-choice instance.
    pub fn from_parts(
        van_nullifier: pallas::Base,
        r_vpk_x: pallas::Base,
        r_vpk_y: pallas::Base,
        vote_authority_note_new: pallas::Base,
        vote_commitment: pallas::Base,
        vote_comm_tree_root: pallas::Base,
        vote_comm_tree_anchor_height: pallas::Base,
        proposal_id: pallas::Base,
        voting_round_id: pallas::Base,
        bridge: pallas::Base,
        decision_bucket_count: pallas::Base,
    ) -> Self {
        Instance {
            van_nullifier,
            r_vpk_x,
            r_vpk_y,
            vote_authority_note_new,
            vote_commitment,
            vote_comm_tree_root,
            vote_comm_tree_anchor_height,
            proposal_id,
            voting_round_id,
            bridge,
            decision_bucket_count,
        }
    }

    /// Serializes public inputs for halo2 proof creation/verification.
    ///
    /// The order must match the instance column offsets defined at the
    /// top of this file (`VAN_NULLIFIER_PUBLIC_OFFSET`, `R_VPK_X_PUBLIC_OFFSET`,
    /// `R_VPK_Y_PUBLIC_OFFSET`, etc.).
    pub fn to_halo2_instance(&self) -> Vec<vesta::Scalar> {
        vec![
            self.van_nullifier,
            self.r_vpk_x,
            self.r_vpk_y,
            self.vote_authority_note_new,
            self.vote_commitment,
            self.vote_comm_tree_root,
            self.vote_comm_tree_anchor_height,
            self.proposal_id,
            self.voting_round_id,
            self.bridge,
            self.decision_bucket_count,
        ]
    }
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::bridge_commitment;
    use crate::ff::PrimeFieldBits;
    use crate::ff::{Field, PrimeField};
    use crate::gadgets::elgamal::{base_to_scalar, spend_auth_g_affine};
    use crate::group::{Curve, Group};
    use crate::params::SHARE_VALUE_LIMIT;
    use crate::protocol_hash::poseidon_hash_2;
    use crate::rand::rngs::OsRng;
    use crate::shares_hash::shares_hash_from_comms;
    use core::iter;
    use voting_crypto_deps::halo2_gadgets::sinsemilla::primitives::CommitDomain;
    use voting_crypto_deps::halo2_proofs::dev::MockProver;
    use voting_crypto_deps::pasta_curves::arithmetic::CurveAffine;
    use voting_crypto_deps::pasta_curves::pallas;

    use voting_crypto_deps::orchard::constants::{
        fixed_bases::COMMIT_IVK_PERSONALIZATION, L_ORCHARD_BASE,
    };

    /// Out-of-circuit voting key derivation for tests.
    ///
    /// Given a voting spending key (vsk), nullifier key (nk), and CommitIvk
    /// randomness (rivk_v), derives the full voting address:
    ///
    /// 1. `ak = [vsk] * SpendAuthG` (spend validating key)
    /// 2. `ak_x = ExtractP(ak)` (x-coordinate)
    /// 3. `ivk_v = CommitIvk(ak_x, nk, rivk_v)` (incoming viewing key)
    /// 4. `g_d = random non-identity point` (diversified base)
    /// 5. `pk_d = [ivk_v] * g_d` (diversified transmission key)
    ///
    /// Returns `(g_d_affine, pk_d_affine, ak_x)` for use as circuit witnesses.
    fn derive_voting_address(
        vsk: pallas::Scalar,
        nk: pallas::Base,
        rivk_v: pallas::Scalar,
    ) -> (pallas::Affine, pallas::Affine) {
        // Step 1: ak = [vsk] * SpendAuthG
        let g = spend_auth_g_affine();
        let ak_point = g * vsk;
        let ak_x = *ak_point.to_affine().coordinates().unwrap().x();

        // Step 2: ivk_v = CommitIvk(ak_x, nk, rivk_v)
        let domain = CommitDomain::new(COMMIT_IVK_PERSONALIZATION);
        let ivk_v = domain
            .short_commit(
                iter::empty()
                    .chain(ak_x.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE))
                    .chain(nk.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE)),
                &rivk_v,
            )
            .expect("CommitIvk should not produce ⊥ for random inputs");

        // Step 3: g_d = random non-identity point
        // Using a deterministic point derived from a fixed seed ensures
        // reproducibility while avoiding the identity point.
        let g_d = pallas::Point::generator() * pallas::Scalar::from(12345u64);
        let g_d_affine = g_d.to_affine();

        // Step 4: pk_d = [ivk_v] * g_d
        let ivk_v_scalar = base_to_scalar(ivk_v).expect("ivk_v must be < scalar field modulus");
        let pk_d = g_d * ivk_v_scalar;
        let pk_d_affine = pk_d.to_affine();

        (g_d_affine, pk_d_affine)
    }

    /// Default proposal_id and decision bucket count for tests.
    const TEST_PROPOSAL_ID: u64 = 3;
    const TEST_BUCKET_COUNT: u64 = 4;

    /// Deterministic stand-in selected commitments for cast-circuit tests.
    ///
    /// ZKP #2 does not verify the ElGamal validity of these values (that is
    /// ZKP 1.5's job); it only re-opens the bridge and hash chain over them,
    /// so opaque field elements are sufficient here.
    fn test_selected_commitments() -> [pallas::Base; 16] {
        core::array::from_fn(|i| pallas::Base::from(0x5e1e_c7ed_u64 + i as u64))
    }

    /// Sets the condition 10′–12′ witnesses on a circuit and returns the
    /// derived `(bridge, vote_commitment)` for the instance.
    fn set_conditions_10_to_12(
        circuit: &mut Circuit,
        shares_u64: [u64; 16],
        proposal_id: u64,
        voting_round_id: pallas::Base,
    ) -> (pallas::Base, pallas::Base) {
        let selected_commitments = test_selected_commitments();
        circuit.selected_commitments = selected_commitments.map(Value::known);
        let weights_and_comms: [(pallas::Base, pallas::Base); 16] =
            core::array::from_fn(|i| (pallas::Base::from(shares_u64[i]), selected_commitments[i]));
        let bridge = bridge_commitment(
            voting_round_id,
            pallas::Base::from(proposal_id),
            pallas::Base::from(TEST_BUCKET_COUNT),
            &weights_and_comms,
        );
        let vote_commitment = vote_commitment_hash_v2(
            voting_round_id,
            shares_hash_from_comms(selected_commitments),
            pallas::Base::from(proposal_id),
            pallas::Base::from(TEST_BUCKET_COUNT),
        );
        (bridge, vote_commitment)
    }

    /// Build valid test data for all 12 conditions.
    ///
    /// Returns a circuit with correctly-hashed VAN witnesses, valid
    /// shares, real El Gamal ciphertexts, and a matching instance.
    fn build_single_leaf_merkle_path(
        leaf: pallas::Base,
    ) -> ([pallas::Base; VOTE_COMM_TREE_DEPTH], u32, pallas::Base) {
        let auth_path = empty_vote_comm_tree_path();
        let mut current = leaf;
        for i in 0..VOTE_COMM_TREE_DEPTH {
            current = poseidon_hash_2(current, auth_path[i]);
        }
        (auth_path, 0, current)
    }

    fn empty_vote_comm_tree_path() -> [pallas::Base; VOTE_COMM_TREE_DEPTH] {
        let mut empty_roots = [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH];
        empty_roots[0] = poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero());
        for i in 1..VOTE_COMM_TREE_DEPTH {
            empty_roots[i] = poseidon_hash_2(empty_roots[i - 1], empty_roots[i - 1]);
        }
        empty_roots
    }

    fn build_left_leaf_merkle_path_with_sibling(
        left_leaf: pallas::Base,
        right_leaf: pallas::Base,
    ) -> ([pallas::Base; VOTE_COMM_TREE_DEPTH], u32, pallas::Base) {
        let mut auth_path = empty_vote_comm_tree_path();
        auth_path[0] = right_leaf;

        let mut current = left_leaf;
        for i in 0..VOTE_COMM_TREE_DEPTH {
            current = poseidon_hash_2(current, auth_path[i]);
        }
        (auth_path, 0, current)
    }

    struct VoteReuseFixture {
        vsk: pallas::Scalar,
        vsk_nk: pallas::Base,
        rivk_v: pallas::Scalar,
        alpha_v: pallas::Scalar,
        vpk_g_d_affine: pallas::Affine,
        vpk_pk_d_affine: pallas::Affine,
        total_note_value: pallas::Base,
        proposal_authority_old: pallas::Base,
        proposal_id: u64,
        van_comm_rand: pallas::Base,
        shares_u64: [u64; 16],
    }

    impl VoteReuseFixture {
        fn new() -> Self {
            let mut rng = OsRng;
            let vsk = pallas::Scalar::random(&mut rng);
            let vsk_nk = pallas::Base::random(&mut rng);
            let rivk_v = pallas::Scalar::random(&mut rng);
            let alpha_v = pallas::Scalar::random(&mut rng);
            let (vpk_g_d_affine, vpk_pk_d_affine) = derive_voting_address(vsk, vsk_nk, rivk_v);

            Self {
                vsk,
                vsk_nk,
                rivk_v,
                alpha_v,
                vpk_g_d_affine,
                vpk_pk_d_affine,
                total_note_value: pallas::Base::from(10_000u64),
                proposal_authority_old: pallas::Base::from(13u64),
                proposal_id: TEST_PROPOSAL_ID,
                van_comm_rand: pallas::Base::random(&mut rng),
                shares_u64: [625; 16],
            }
        }

        fn vpk_x_coordinates(&self) -> (pallas::Base, pallas::Base) {
            (
                *self.vpk_g_d_affine.coordinates().unwrap().x(),
                *self.vpk_pk_d_affine.coordinates().unwrap().x(),
            )
        }

        fn vote_authority_note_old(&self, voting_round_id: pallas::Base) -> pallas::Base {
            let (vpk_g_d_x, vpk_pk_d_x) = self.vpk_x_coordinates();
            van_integrity_hash(
                vpk_g_d_x,
                vpk_pk_d_x,
                self.total_note_value,
                voting_round_id,
                self.proposal_authority_old,
                self.van_comm_rand,
            )
        }

        fn vote_authority_note_new(&self, voting_round_id: pallas::Base) -> pallas::Base {
            let (vpk_g_d_x, vpk_pk_d_x) = self.vpk_x_coordinates();
            let proposal_authority_new =
                self.proposal_authority_old - pallas::Base::from(1u64 << self.proposal_id);
            van_integrity_hash(
                vpk_g_d_x,
                vpk_pk_d_x,
                self.total_note_value,
                voting_round_id,
                proposal_authority_new,
                self.van_comm_rand,
            )
        }

        fn build_vote_data(
            &self,
            voting_round_id: pallas::Base,
            auth_path: [pallas::Base; VOTE_COMM_TREE_DEPTH],
            position: u32,
            vote_comm_tree_root: pallas::Base,
            anchor_height: u64,
        ) -> (Circuit, Instance) {
            let vote_authority_note_old = self.vote_authority_note_old(voting_round_id);
            let vote_authority_note_new = self.vote_authority_note_new(voting_round_id);
            let van_nullifier =
                van_nullifier_hash(self.vsk_nk, voting_round_id, vote_authority_note_old);

            let g = spend_auth_g_affine();
            let r_vpk = (g * (self.vsk + self.alpha_v)).to_affine();
            let r_vpk_x = *r_vpk.coordinates().unwrap().x();
            let r_vpk_y = *r_vpk.coordinates().unwrap().y();

            let mut circuit = Circuit::with_van_witnesses(
                Value::known(auth_path),
                Value::known(position),
                Value::known(self.vpk_g_d_affine),
                Value::known(self.vpk_pk_d_affine),
                Value::known(self.total_note_value),
                Value::known(self.proposal_authority_old),
                Value::known(self.van_comm_rand),
                Value::known(vote_authority_note_old),
                Value::known(self.vsk),
                Value::known(self.rivk_v),
                Value::known(self.vsk_nk),
                Value::known(self.alpha_v),
            );
            circuit.one_shifted = Value::known(pallas::Base::from(1u64 << self.proposal_id));
            circuit.shares = self.shares_u64.map(|s| Value::known(pallas::Base::from(s)));
            let (bridge, vote_commitment) = set_conditions_10_to_12(
                &mut circuit,
                self.shares_u64,
                self.proposal_id,
                voting_round_id,
            );

            let instance = Instance::from_parts(
                van_nullifier,
                r_vpk_x,
                r_vpk_y,
                vote_authority_note_new,
                vote_commitment,
                vote_comm_tree_root,
                pallas::Base::from(anchor_height),
                pallas::Base::from(self.proposal_id),
                voting_round_id,
                bridge,
                pallas::Base::from(TEST_BUCKET_COUNT),
            );

            (circuit, instance)
        }
    }

    /// Build test (circuit, instance) with given proposal_authority_old,
    /// proposal_id, and optional spend-authority randomizer.
    /// proposal_authority_old must have the proposal_id-th bit set (spec bitmask).
    fn make_test_data_with_authority_proposal_and_alpha(
        proposal_authority_old: pallas::Base,
        proposal_id: u64,
        alpha_v_override: Option<pallas::Scalar>,
    ) -> (Circuit, Instance) {
        let mut rng = OsRng;

        // Condition 3 (spend authority): derive proper voting key hierarchy.
        // vsk → ak → ivk_v → (vpk_g_d, vpk_pk_d) through CommitIvk chain.
        let vsk = pallas::Scalar::random(&mut rng);
        let vsk_nk = pallas::Base::random(&mut rng);
        let rivk_v = pallas::Scalar::random(&mut rng);
        let alpha_v = alpha_v_override.unwrap_or_else(|| pallas::Scalar::random(&mut rng));

        let (vpk_g_d_affine, vpk_pk_d_affine) = derive_voting_address(vsk, vsk_nk, rivk_v);

        // Condition 4: r_vpk = ak + [alpha_v] * G = [vsk + alpha_v] * G
        let g = spend_auth_g_affine();
        let r_vpk = (g * (vsk + alpha_v)).to_affine();
        let r_vpk_x = *r_vpk.coordinates().unwrap().x();
        let r_vpk_y = *r_vpk.coordinates().unwrap().y();

        // Extract x-coordinates for Poseidon hashing (conditions 2, 6).
        let vpk_g_d_x = *vpk_g_d_affine.coordinates().unwrap().x();
        let vpk_pk_d_x = *vpk_pk_d_affine.coordinates().unwrap().x();

        // total_note_value must be small enough that all 16 shares
        // fit in [0, 2^30) for condition 9's range check.
        let total_note_value = pallas::Base::from(10_000u64);
        let voting_round_id = pallas::Base::random(&mut rng);
        let van_comm_rand = pallas::Base::random(&mut rng);

        let vote_authority_note_old = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_old,
            van_comm_rand,
        );
        let (auth_path, position, vote_comm_tree_root) =
            build_single_leaf_merkle_path(vote_authority_note_old);
        let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);
        // Spec: proposal_authority_new = proposal_authority_old - (1 << proposal_id).
        let one_shifted = pallas::Base::from(1u64 << proposal_id);
        let proposal_authority_new = proposal_authority_old - one_shifted;
        let vote_authority_note_new = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_new,
            van_comm_rand,
        );

        // Create shares that sum to total_note_value (conditions 8 + 9).
        // Each share must be in [0, 2^30) for condition 9's range check.
        let shares_u64: [u64; 16] = [625; 16]; // sum = 10000

        // Condition 11: El Gamal encryption of shares under ea_pk.

        let mut circuit = Circuit::with_van_witnesses(
            Value::known(auth_path),
            Value::known(position),
            Value::known(vpk_g_d_affine),
            Value::known(vpk_pk_d_affine),
            Value::known(total_note_value),
            Value::known(proposal_authority_old),
            Value::known(van_comm_rand),
            Value::known(vote_authority_note_old),
            Value::known(vsk),
            Value::known(rivk_v),
            Value::known(vsk_nk),
            Value::known(alpha_v),
        );
        circuit.one_shifted = Value::known(one_shifted);
        circuit.shares = shares_u64.map(|s| Value::known(pallas::Base::from(s)));

        // Conditions 10′–12′: bridge, shares hash, and vote commitment.
        let (bridge, vote_commitment) =
            set_conditions_10_to_12(&mut circuit, shares_u64, proposal_id, voting_round_id);

        let instance = Instance::from_parts(
            van_nullifier,
            r_vpk_x,
            r_vpk_y,
            vote_authority_note_new,
            vote_commitment,
            vote_comm_tree_root,
            pallas::Base::zero(),
            pallas::Base::from(proposal_id),
            voting_round_id,
            bridge,
            pallas::Base::from(TEST_BUCKET_COUNT),
        );

        (circuit, instance)
    }

    fn make_test_data_with_authority_and_proposal(
        proposal_authority_old: pallas::Base,
        proposal_id: u64,
    ) -> (Circuit, Instance) {
        make_test_data_with_authority_proposal_and_alpha(proposal_authority_old, proposal_id, None)
    }

    fn make_test_data_with_authority(proposal_authority_old: pallas::Base) -> (Circuit, Instance) {
        make_test_data_with_authority_and_proposal(proposal_authority_old, TEST_PROPOSAL_ID)
    }

    fn make_test_data() -> (Circuit, Instance) {
        // proposal_authority_old must have bit TEST_PROPOSAL_ID set (spec bitmask).
        // 5 | (1 << 3) = 13 so we can vote on proposal 3 and get new = 5.
        make_test_data_with_authority(pallas::Base::from(13u64))
    }

    // ================================================================
    // Condition 2 (VAN Integrity) tests
    // ================================================================

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn van_integrity_valid_proof() {
        let (circuit, instance) = make_test_data();

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();

        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn van_integrity_wrong_hash_fails() {
        let mut rng = OsRng;
        let (_, mut instance) = make_test_data();

        // Deliberately wrong VAN value — condition 2 constrain_equal will fail.
        let wrong_van = pallas::Base::random(&mut rng);
        let (auth_path, position, root) = build_single_leaf_merkle_path(wrong_van);
        instance.vote_comm_tree_root = root;

        // Use properly derived keys (condition 3 would pass) but the VAN
        // hash won't match wrong_van, so condition 2 fails.
        let vsk = pallas::Scalar::random(&mut rng);
        let vsk_nk = pallas::Base::random(&mut rng);
        let rivk_v = pallas::Scalar::random(&mut rng);
        let alpha_v = pallas::Scalar::random(&mut rng);
        let (vpk_g_d_affine, vpk_pk_d_affine) = derive_voting_address(vsk, vsk_nk, rivk_v);
        let g = spend_auth_g_affine();
        let r_vpk = (g * (vsk + alpha_v)).to_affine();
        instance.r_vpk_x = *r_vpk.coordinates().unwrap().x();
        instance.r_vpk_y = *r_vpk.coordinates().unwrap().y();

        let shares_u64: [u64; 16] = [625; 16];

        // Use authority 13 (bit 3 set) and one_shifted = 8 so condition 6 is consistent;
        // only condition 2 (VAN hash) should fail due to wrong_van.
        let proposal_authority_old = pallas::Base::from(13u64);
        let van_comm_rand = pallas::Base::random(&mut rng);
        let mut circuit = Circuit::with_van_witnesses(
            Value::known(auth_path),
            Value::known(position),
            Value::known(vpk_g_d_affine),
            Value::known(vpk_pk_d_affine),
            Value::known(pallas::Base::from(10_000u64)),
            Value::known(proposal_authority_old),
            Value::known(van_comm_rand),
            Value::known(wrong_van),
            Value::known(vsk),
            Value::known(rivk_v),
            Value::known(vsk_nk),
            Value::known(alpha_v),
        );
        circuit.one_shifted = Value::known(pallas::Base::from(1u64 << TEST_PROPOSAL_ID));
        circuit.shares = shares_u64.map(|s| Value::known(pallas::Base::from(s)));
        let (bridge, vc) = set_conditions_10_to_12(
            &mut circuit,
            shares_u64,
            TEST_PROPOSAL_ID,
            instance.voting_round_id,
        );
        instance.bridge = bridge;
        instance.vote_commitment = vc;
        instance.proposal_id = pallas::Base::from(TEST_PROPOSAL_ID);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        // Should fail: derived hash ≠ witnessed vote_authority_note_old.
        assert!(prover.verify().is_err());
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn van_integrity_wrong_round_id_fails() {
        let (circuit, mut instance) = make_test_data();

        // Supply a DIFFERENT voting_round_id in the instance.
        instance.voting_round_id = pallas::Base::random(&mut OsRng);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        // Should fail: the voting_round_id from the instance doesn't match
        // the one hashed into the VAN (condition 2).
        assert!(prover.verify().is_err());
    }

    #[test]
    fn round_scoped_van_redelegation_changes_nullifier() {
        let fixture = VoteReuseFixture::new();
        let round_1 = pallas::Base::from(0xCAFEu64);
        let round_2 = pallas::Base::from(0xCAFFu64);

        let van_round_1 = fixture.vote_authority_note_old(round_1);
        let van_round_2 = fixture.vote_authority_note_old(round_2);
        assert_ne!(
            van_round_1, van_round_2,
            "voting_round_id is part of the VAN preimage"
        );

        let nullifier_round_1 = van_nullifier_hash(fixture.vsk_nk, round_1, van_round_1);
        let nullifier_round_2 = van_nullifier_hash(fixture.vsk_nk, round_2, van_round_2);
        assert_ne!(
            nullifier_round_1, nullifier_round_2,
            "honest redelegation in a new round must not collide with the old round"
        );
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn round_scoped_van_redelegation_verifies_with_distinct_nullifiers() {
        let fixture = VoteReuseFixture::new();
        let round_1 = pallas::Base::from(0xCAFEu64);
        let round_2 = pallas::Base::from(0xCAFFu64);

        let van_round_1 = fixture.vote_authority_note_old(round_1);
        let (path_round_1, position_round_1, root_round_1) =
            build_single_leaf_merkle_path(van_round_1);
        let (circuit_round_1, instance_round_1) =
            fixture.build_vote_data(round_1, path_round_1, position_round_1, root_round_1, 10);

        let van_round_2 = fixture.vote_authority_note_old(round_2);
        let (path_round_2, position_round_2, root_round_2) =
            build_single_leaf_merkle_path(van_round_2);
        let (circuit_round_2, instance_round_2) =
            fixture.build_vote_data(round_2, path_round_2, position_round_2, root_round_2, 20);

        assert_ne!(van_round_1, van_round_2);
        assert_ne!(
            instance_round_1.van_nullifier,
            instance_round_2.van_nullifier
        );

        let prover_round_1 = MockProver::run(
            K,
            &circuit_round_1,
            vec![instance_round_1.to_halo2_instance()],
        )
        .unwrap();
        assert_eq!(prover_round_1.verify(), Ok(()));

        let prover_round_2 = MockProver::run(
            K,
            &circuit_round_2,
            vec![instance_round_2.to_halo2_instance()],
        )
        .unwrap();
        assert_eq!(prover_round_2.verify(), Ok(()));
    }

    /// Verifies the out-of-circuit helper produces deterministic results.
    #[test]
    fn van_integrity_hash_deterministic() {
        let mut rng = OsRng;

        let vpk_g_d = pallas::Base::random(&mut rng);
        let vpk_pk_d = pallas::Base::random(&mut rng);
        let val = pallas::Base::random(&mut rng);
        let round = pallas::Base::random(&mut rng);
        let auth = pallas::Base::random(&mut rng);
        let rand = pallas::Base::random(&mut rng);

        let h1 = van_integrity_hash(vpk_g_d, vpk_pk_d, val, round, auth, rand);
        let h2 = van_integrity_hash(vpk_g_d, vpk_pk_d, val, round, auth, rand);
        assert_eq!(h1, h2);

        // Changing any input changes the hash.
        let h3 = van_integrity_hash(
            pallas::Base::random(&mut rng),
            vpk_pk_d,
            val,
            round,
            auth,
            rand,
        );
        assert_ne!(h1, h3);
    }

    // ================================================================
    // Condition 3 (Diversified Address Integrity / Address Ownership) tests
    //
    // These tests ensure the circuit rejects witnesses that violate
    // vpk_pk_d = [ivk_v] * vpk_g_d. Without condition 3 enabled, they
    // would pass (invalid address ownership would not be detected).
    // ================================================================

    /// Using a different vsk in the circuit than was used to derive
    /// (vpk_g_d, vpk_pk_d) should fail condition 3 only: in-circuit
    /// [ivk']*vpk_g_d ≠ vpk_pk_d while VAN hash and nullifier stay valid.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn condition_3_wrong_vsk_fails() {
        let mut rng = OsRng;

        let vsk = pallas::Scalar::random(&mut rng);
        let vsk_nk = pallas::Base::random(&mut rng);
        let rivk_v = pallas::Scalar::random(&mut rng);
        let (vpk_g_d_affine, vpk_pk_d_affine) = derive_voting_address(vsk, vsk_nk, rivk_v);
        let vpk_g_d_x = *vpk_g_d_affine.coordinates().unwrap().x();
        let vpk_pk_d_x = *vpk_pk_d_affine.coordinates().unwrap().x();

        let total_note_value = pallas::Base::from(10_000u64);
        let voting_round_id = pallas::Base::random(&mut rng);
        let proposal_authority_old = pallas::Base::from(13u64);
        let proposal_id = 3u64;
        let van_comm_rand = pallas::Base::random(&mut rng);

        let vote_authority_note_old = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_old,
            van_comm_rand,
        );
        let (auth_path, position, vote_comm_tree_root) =
            build_single_leaf_merkle_path(vote_authority_note_old);
        let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);
        let one_shifted = pallas::Base::from(1u64 << proposal_id);
        let proposal_authority_new = proposal_authority_old - one_shifted;
        let vote_authority_note_new = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_new,
            van_comm_rand,
        );

        let shares_u64: [u64; 16] = [625; 16];

        let wrong_vsk = pallas::Scalar::random(&mut rng);
        assert_ne!(
            wrong_vsk, vsk,
            "test assumes distinct vsk with high probability"
        );
        let alpha_v = pallas::Scalar::random(&mut rng);
        let g = spend_auth_g_affine();
        let r_vpk = (g * (vsk + alpha_v)).to_affine();
        let r_vpk_x = *r_vpk.coordinates().unwrap().x();
        let r_vpk_y = *r_vpk.coordinates().unwrap().y();

        let mut circuit = Circuit::with_van_witnesses(
            Value::known(auth_path),
            Value::known(position),
            Value::known(vpk_g_d_affine),
            Value::known(vpk_pk_d_affine),
            Value::known(total_note_value),
            Value::known(proposal_authority_old),
            Value::known(van_comm_rand),
            Value::known(vote_authority_note_old),
            Value::known(wrong_vsk),
            Value::known(rivk_v),
            Value::known(vsk_nk),
            Value::known(alpha_v),
        );
        circuit.one_shifted = Value::known(one_shifted);
        circuit.shares = shares_u64.map(|s| Value::known(pallas::Base::from(s)));
        let (bridge, vc) =
            set_conditions_10_to_12(&mut circuit, shares_u64, proposal_id, voting_round_id);

        let instance = Instance::from_parts(
            van_nullifier,
            r_vpk_x,
            r_vpk_y,
            vote_authority_note_new,
            vc,
            vote_comm_tree_root,
            pallas::Base::zero(),
            pallas::Base::from(proposal_id),
            voting_round_id,
            bridge,
            pallas::Base::from(TEST_BUCKET_COUNT),
        );

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(
            prover.verify().is_err(),
            "condition 3 must reject wrong vsk"
        );
    }

    /// Using a vpk_pk_d that does not equal [ivk_v]*vpk_g_d should fail
    /// condition 3. Instance is built with a wrong vpk_pk_d for the VAN
    /// hash so condition 2 still passes; only condition 3 fails.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn condition_3_wrong_vpk_pk_d_fails() {
        let mut rng = OsRng;

        let vsk = pallas::Scalar::random(&mut rng);
        let vsk_nk = pallas::Base::random(&mut rng);
        let rivk_v = pallas::Scalar::random(&mut rng);
        let (vpk_g_d_affine, _vpk_pk_d_correct) = derive_voting_address(vsk, vsk_nk, rivk_v);
        let vpk_g_d_x = *vpk_g_d_affine.coordinates().unwrap().x();

        let wrong_vpk_pk_d_affine =
            (pallas::Point::generator() * pallas::Scalar::from(99999u64)).to_affine();
        let wrong_vpk_pk_d_x = *wrong_vpk_pk_d_affine.coordinates().unwrap().x();

        let total_note_value = pallas::Base::from(10_000u64);
        let voting_round_id = pallas::Base::random(&mut rng);
        let proposal_authority_old = pallas::Base::from(13u64);
        let proposal_id = 3u64;
        let van_comm_rand = pallas::Base::random(&mut rng);

        let vote_authority_note_old = van_integrity_hash(
            vpk_g_d_x,
            wrong_vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_old,
            van_comm_rand,
        );
        let (auth_path, position, vote_comm_tree_root) =
            build_single_leaf_merkle_path(vote_authority_note_old);
        let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);
        let one_shifted = pallas::Base::from(1u64 << proposal_id);
        let proposal_authority_new = proposal_authority_old - one_shifted;
        let vote_authority_note_new = van_integrity_hash(
            vpk_g_d_x,
            wrong_vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_new,
            van_comm_rand,
        );

        let shares_u64: [u64; 16] = [625; 16];

        let alpha_v = pallas::Scalar::random(&mut rng);
        let g = spend_auth_g_affine();
        let r_vpk = (g * (vsk + alpha_v)).to_affine();
        let r_vpk_x = *r_vpk.coordinates().unwrap().x();
        let r_vpk_y = *r_vpk.coordinates().unwrap().y();

        let mut circuit = Circuit::with_van_witnesses(
            Value::known(auth_path),
            Value::known(position),
            Value::known(vpk_g_d_affine),
            Value::known(wrong_vpk_pk_d_affine),
            Value::known(total_note_value),
            Value::known(proposal_authority_old),
            Value::known(van_comm_rand),
            Value::known(vote_authority_note_old),
            Value::known(vsk),
            Value::known(rivk_v),
            Value::known(vsk_nk),
            Value::known(alpha_v),
        );
        circuit.one_shifted = Value::known(one_shifted);
        circuit.shares = shares_u64.map(|s| Value::known(pallas::Base::from(s)));
        let (bridge, vc) =
            set_conditions_10_to_12(&mut circuit, shares_u64, proposal_id, voting_round_id);

        let instance = Instance::from_parts(
            van_nullifier,
            r_vpk_x,
            r_vpk_y,
            vote_authority_note_new,
            vc,
            vote_comm_tree_root,
            pallas::Base::zero(),
            pallas::Base::from(proposal_id),
            voting_round_id,
            bridge,
            pallas::Base::from(TEST_BUCKET_COUNT),
        );

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(
            prover.verify().is_err(),
            "condition 3 must reject wrong vpk_pk_d"
        );
    }

    // ================================================================
    // Condition 4 (Spend Authority) tests
    // ================================================================

    /// Wrong r_vpk public input should fail condition 4.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn condition_4_wrong_r_vpk_fails() {
        let (circuit, mut instance) = make_test_data();

        instance.r_vpk_x = pallas::Base::random(&mut OsRng);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(
            prover.verify().is_err(),
            "condition 4 must reject wrong r_vpk"
        );
    }

    /// Documents the current upstream-compatible relation: alpha_v = 0 is
    /// accepted when the public r_vpk is correspondingly equal to ak_P. This is
    /// a self-linking/coercion surface, not a proof-soundness failure; see
    /// THREAT_MODEL.md.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn condition_4_alpha_zero_is_accepted_by_relation() {
        let (circuit, instance) = make_test_data_with_authority_proposal_and_alpha(
            pallas::Base::from(13u64),
            TEST_PROPOSAL_ID,
            Some(pallas::Scalar::zero()),
        );

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    // ================================================================
    // Condition 5 (VAN Nullifier Integrity) tests
    // ================================================================

    /// Wrong VAN_NULLIFIER_PUBLIC_OFFSET public input should fail condition 5.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn van_nullifier_wrong_public_input_fails() {
        let (circuit, mut instance) = make_test_data();

        // Corrupt the VAN nullifier public input.
        instance.van_nullifier = pallas::Base::random(&mut OsRng);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();

        // Should fail: circuit-derived nullifier ≠ corrupted instance value.
        assert!(prover.verify().is_err());
    }

    /// Using a different vsk_nk in the circuit than was used to compute
    /// the instance nullifier should fail condition 5.
    /// Note: since vsk_nk is also used in CommitIvk (condition 3), the
    /// wrong value also breaks condition 3 — but the test still verifies
    /// that the proof fails as expected.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn van_nullifier_wrong_vsk_nk_fails() {
        let mut rng = OsRng;

        // Derive proper keys with the CORRECT vsk_nk.
        let vsk = pallas::Scalar::random(&mut rng);
        let vsk_nk = pallas::Base::random(&mut rng);
        let rivk_v = pallas::Scalar::random(&mut rng);
        let (vpk_g_d_affine, vpk_pk_d_affine) = derive_voting_address(vsk, vsk_nk, rivk_v);
        let vpk_g_d_x = *vpk_g_d_affine.coordinates().unwrap().x();
        let vpk_pk_d_x = *vpk_pk_d_affine.coordinates().unwrap().x();

        let total_note_value = pallas::Base::from(10_000u64);
        let voting_round_id = pallas::Base::random(&mut rng);
        let proposal_authority_old = pallas::Base::from(5u64); // bits 0 and 2 set
        let van_comm_rand = pallas::Base::random(&mut rng);
        let proposal_id = 0u64; // vote on proposal 0 so one_shifted = 1, new = 4

        let vote_authority_note_old = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_old,
            van_comm_rand,
        );
        let (auth_path, position, vote_comm_tree_root) =
            build_single_leaf_merkle_path(vote_authority_note_old);
        let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);
        let one_shifted = pallas::Base::from(1u64 << proposal_id);
        let proposal_authority_new = proposal_authority_old - one_shifted;
        let vote_authority_note_new = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_new,
            van_comm_rand,
        );

        // Use a DIFFERENT vsk_nk in the circuit.
        let wrong_vsk_nk = pallas::Base::random(&mut rng);
        let alpha_v = pallas::Scalar::random(&mut rng);
        let g = spend_auth_g_affine();
        let r_vpk = (g * (vsk + alpha_v)).to_affine();
        let r_vpk_x = *r_vpk.coordinates().unwrap().x();
        let r_vpk_y = *r_vpk.coordinates().unwrap().y();

        // Shares that sum to total_note_value (conditions 8 + 9).
        let shares_u64: [u64; 16] = [625; 16];

        // Condition 11: real El Gamal encryption.

        let mut circuit = Circuit::with_van_witnesses(
            Value::known(auth_path),
            Value::known(position),
            Value::known(vpk_g_d_affine),
            Value::known(vpk_pk_d_affine),
            Value::known(total_note_value),
            Value::known(proposal_authority_old),
            Value::known(van_comm_rand),
            Value::known(vote_authority_note_old),
            Value::known(vsk),
            Value::known(rivk_v),
            Value::known(wrong_vsk_nk),
            Value::known(alpha_v),
        );
        circuit.one_shifted = Value::known(one_shifted);
        circuit.shares = shares_u64.map(|s| Value::known(pallas::Base::from(s)));
        let (bridge, vc) =
            set_conditions_10_to_12(&mut circuit, shares_u64, proposal_id, voting_round_id);

        let instance = Instance::from_parts(
            van_nullifier,
            r_vpk_x,
            r_vpk_y,
            vote_authority_note_new,
            vc,
            vote_comm_tree_root,
            pallas::Base::zero(),
            pallas::Base::from(proposal_id),
            voting_round_id,
            bridge,
            pallas::Base::from(TEST_BUCKET_COUNT),
        );

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        // Should fail: circuit computes Poseidon(wrong_vsk_nk, inner_hash)
        // which ≠ the instance van_nullifier (computed with correct vsk_nk).
        // Also fails condition 3 since wrong_vsk_nk breaks CommitIvk derivation.
        assert!(prover.verify().is_err());
    }

    /// Verifies the out-of-circuit nullifier helper produces deterministic results.
    #[test]
    fn van_nullifier_hash_deterministic() {
        let mut rng = OsRng;

        let nk = pallas::Base::random(&mut rng);
        let round = pallas::Base::random(&mut rng);
        let van = pallas::Base::random(&mut rng);

        let h1 = van_nullifier_hash(nk, round, van);
        let h2 = van_nullifier_hash(nk, round, van);
        assert_eq!(h1, h2);

        // Changing any input changes the hash.
        let h3 = van_nullifier_hash(pallas::Base::random(&mut rng), round, van);
        assert_ne!(h1, h3);
    }

    #[test]
    fn van_nullifier_hash_frozen_vector() {
        assert_eq!(
            van_nullifier_hash(
                pallas::Base::from(1u64),
                pallas::Base::from(42u64),
                pallas::Base::from(100u64),
            ),
            pallas::Base::from_repr([
                114, 56, 62, 208, 155, 244, 76, 209, 125, 210, 149, 109, 176, 88, 34, 116, 123, 56,
                62, 216, 108, 204, 55, 120, 28, 155, 217, 186, 29, 159, 128, 2,
            ])
            .expect("frozen vector must be canonical")
        );
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn stale_and_current_anchor_proofs_for_same_van_share_nullifier() {
        let fixture = VoteReuseFixture::new();
        let voting_round_id = pallas::Base::from(0xCAFEu64);
        let stale_van = fixture.vote_authority_note_old(voting_round_id);
        let successor_van = fixture.vote_authority_note_new(voting_round_id);

        let (stale_path, stale_position, stale_root) = build_single_leaf_merkle_path(stale_van);
        let (stale_circuit, stale_instance) =
            fixture.build_vote_data(voting_round_id, stale_path, stale_position, stale_root, 10);

        let (current_path, current_position, current_root) =
            build_left_leaf_merkle_path_with_sibling(stale_van, successor_van);
        let (current_circuit, current_instance) = fixture.build_vote_data(
            voting_round_id,
            current_path,
            current_position,
            current_root,
            11,
        );

        assert_ne!(
            stale_root, current_root,
            "the successor VAN changes the supplied tree anchor"
        );
        assert_eq!(
            stale_instance.van_nullifier, current_instance.van_nullifier,
            "same (vsk_nk, voting_round_id, VAN) must collide for chain-side nullifier uniqueness"
        );

        // The circuit only proves membership in the supplied root. Freshness of
        // the height-to-root mapping is enforced by the chain ante handler.
        let stale_prover =
            MockProver::run(K, &stale_circuit, vec![stale_instance.to_halo2_instance()]).unwrap();
        assert_eq!(stale_prover.verify(), Ok(()));

        let current_prover = MockProver::run(
            K,
            &current_circuit,
            vec![current_instance.to_halo2_instance()],
        )
        .unwrap();
        assert_eq!(current_prover.verify(), Ok(()));
    }

    /// Verifies the domain tag is non-zero and deterministic.
    #[test]
    fn domain_van_nullifier_deterministic() {
        let d1 = domain_van_nullifier();
        let d2 = domain_van_nullifier();
        assert_eq!(d1, d2);

        // Must differ from DOMAIN_VAN (which is 0).
        assert_ne!(d1, pallas::Base::zero());
    }

    // ================================================================
    // Condition 6 (Proposal Authority Decrement) tests
    // ================================================================

    /// Proposal authority with only bit 0 set (value 1): vote on proposal 0, new = 0.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn proposal_authority_decrement_minimum_valid() {
        // proposal_id = 0 is now forbidden (sentinel value); use the next smallest valid id.
        // Authority = 2 = 0b0010 has exactly bit 1 set, so proposal_id = 1 is valid.
        // After decrement: proposal_authority_new = 0 (minimum possible outcome).
        let (circuit, instance) =
            make_test_data_with_authority_and_proposal(pallas::Base::from(2u64), 1);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    /// With proposal_authority_old = 0, the selected bit is 0 so the
    /// "run_selected = 1" constraint (selected bit was set) fails.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn proposal_authority_zero_fails() {
        let (circuit, instance) = make_test_data_with_authority(pallas::Base::zero());

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();

        assert!(prover.verify().is_err());
    }

    /// proposal_id = 0 is the dummy sentinel value and must be rejected (Cond 6, gate).
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn proposal_id_zero_fails() {
        // Authority = 1 = 0b0001 has bit 0 set, so this is otherwise a structurally
        // valid decrement — the only reason it must fail is the non-zero gate.
        let (circuit, instance) =
            make_test_data_with_authority_and_proposal(pallas::Base::one(), 0);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err(), "proposal_id = 0 must be rejected");
    }

    /// Full authority with proposal_id 50 verifies at the upper boundary.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn proposal_authority_full_authority_proposal_50_passes() {
        let (circuit, instance) = make_test_data_with_authority_and_proposal(
            pallas::Base::from(crate::params::MAX_PROPOSAL_AUTHORITY),
            50,
        );

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    /// Wrong vote_authority_note_new (e.g. not clearing the bit) fails condition 6.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn proposal_authority_wrong_new_fails() {
        let (circuit, mut instance) = make_test_data_with_authority_and_proposal(
            pallas::Base::from(crate::params::MAX_PROPOSAL_AUTHORITY),
            1,
        );

        instance.vote_authority_note_new = pallas::Base::random(&mut OsRng);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// authority=4 (0b0100, bit 2 set only), proposal_id=1 (bit 1 absent) →
    /// run_selected=0 at the terminal row, so "run_selected = 1" fails.
    /// Uses proposal_id=1 (not 0) to isolate this constraint from the
    /// proposal_id != 0 sentinel gate.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn proposal_authority_bit_not_set_fails() {
        let (circuit, instance) =
            make_test_data_with_authority_and_proposal(pallas::Base::from(4u64), 1);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// Condition 6 enforces run_sel = 1 (exactly one selector active) at the last bit row;
    /// see CONDITION_6_RUN_SEL_FIX.md. This test runs a valid proof (one selector) and
    /// verifies it passes; a zero-selector witness would be rejected by that gate.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn proposal_authority_condition6_run_sel_constraint() {
        let (circuit, instance) =
            make_test_data_with_authority_and_proposal(pallas::Base::from(3u64), 1);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    /// A value with bit 51 set lies outside the valid 51-bit bitmask and cannot
    /// be represented by the authority-decrement decomposition.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn proposal_authority_exceeds_51_bits_fails() {
        let first_invalid = crate::params::MAX_PROPOSAL_AUTHORITY + 1;
        let (circuit, instance) = make_test_data_with_authority(pallas::Base::from(first_invalid));
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(
            prover.verify().is_err(),
            "authority values wider than 51 bits must be rejected"
        );
    }

    // ================================================================
    // Condition 7 (New VAN Integrity) tests
    // ================================================================

    /// Wrong vote_authority_note_new public input should fail condition 7.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn new_van_integrity_wrong_public_input_fails() {
        let (circuit, mut instance) = make_test_data();

        // Corrupt the new VAN public input.
        instance.vote_authority_note_new = pallas::Base::random(&mut OsRng);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();

        // Should fail: circuit-derived new VAN ≠ corrupted instance value.
        assert!(prover.verify().is_err());
    }

    /// New VAN integrity with a large, valid 51-bit proposal authority.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn new_van_integrity_large_authority() {
        let authority = crate::params::MAX_PROPOSAL_AUTHORITY & !0b111u64;
        let (circuit, instance) = make_test_data_with_authority(pallas::Base::from(authority));

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    // ================================================================
    // Condition 1 (VAN Membership) tests
    // ================================================================

    /// Wrong vote_comm_tree_root in the instance should fail condition 1.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn van_membership_wrong_root_fails() {
        let (circuit, mut instance) = make_test_data();

        // Corrupt the tree root.
        instance.vote_comm_tree_root = pallas::Base::random(&mut OsRng);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// A VAN at a non-zero position in the tree should verify.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn van_membership_nonzero_position() {
        let mut rng = OsRng;

        // Derive proper voting key hierarchy.
        let vsk = pallas::Scalar::random(&mut rng);
        let vsk_nk = pallas::Base::random(&mut rng);
        let rivk_v = pallas::Scalar::random(&mut rng);
        let (vpk_g_d_affine, vpk_pk_d_affine) = derive_voting_address(vsk, vsk_nk, rivk_v);
        let vpk_g_d_x = *vpk_g_d_affine.coordinates().unwrap().x();
        let vpk_pk_d_x = *vpk_pk_d_affine.coordinates().unwrap().x();

        let total_note_value = pallas::Base::from(10_000u64);
        let voting_round_id = pallas::Base::random(&mut rng);
        let proposal_authority_old = pallas::Base::from(5u64); // bits 0 and 2 set
                                                               // proposal_id = 0 is now forbidden (sentinel); use proposal_id = 2 (bit 2 is set in 5).
        let proposal_id = 2u64;
        let van_comm_rand = pallas::Base::random(&mut rng);

        let vote_authority_note_old = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_old,
            van_comm_rand,
        );

        // Place the leaf at position 7 (binary: ...0111).
        let position: u32 = 7;
        let mut empty_roots = [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH];
        empty_roots[0] = poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero());
        for i in 1..VOTE_COMM_TREE_DEPTH {
            empty_roots[i] = poseidon_hash_2(empty_roots[i - 1], empty_roots[i - 1]);
        }
        let auth_path = empty_roots;
        let mut current = vote_authority_note_old;
        for i in 0..VOTE_COMM_TREE_DEPTH {
            if (position >> i) & 1 == 0 {
                current = poseidon_hash_2(current, auth_path[i]);
            } else {
                current = poseidon_hash_2(auth_path[i], current);
            }
        }
        let vote_comm_tree_root = current;

        let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);
        let one_shifted = pallas::Base::from(1u64 << proposal_id);
        let proposal_authority_new = proposal_authority_old - one_shifted;
        let vote_authority_note_new = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_new,
            van_comm_rand,
        );

        let alpha_v = pallas::Scalar::random(&mut rng);
        let g = spend_auth_g_affine();
        let r_vpk = (g * (vsk + alpha_v)).to_affine();
        let r_vpk_x = *r_vpk.coordinates().unwrap().x();
        let r_vpk_y = *r_vpk.coordinates().unwrap().y();

        // Shares that sum to total_note_value (conditions 8 + 9).
        let shares_u64: [u64; 16] = [625; 16];

        // Condition 11: real El Gamal encryption.

        let mut circuit = Circuit::with_van_witnesses(
            Value::known(auth_path),
            Value::known(position),
            Value::known(vpk_g_d_affine),
            Value::known(vpk_pk_d_affine),
            Value::known(total_note_value),
            Value::known(proposal_authority_old),
            Value::known(van_comm_rand),
            Value::known(vote_authority_note_old),
            Value::known(vsk),
            Value::known(rivk_v),
            Value::known(vsk_nk),
            Value::known(alpha_v),
        );
        circuit.one_shifted = Value::known(one_shifted);
        circuit.shares = shares_u64.map(|s| Value::known(pallas::Base::from(s)));
        let (bridge, vc) =
            set_conditions_10_to_12(&mut circuit, shares_u64, proposal_id, voting_round_id);

        let instance = Instance::from_parts(
            van_nullifier,
            r_vpk_x,
            r_vpk_y,
            vote_authority_note_new,
            vc,
            vote_comm_tree_root,
            pallas::Base::zero(),
            pallas::Base::from(proposal_id),
            voting_round_id,
            bridge,
            pallas::Base::from(TEST_BUCKET_COUNT),
        );

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    /// Poseidon hash-2 helper is deterministic.
    #[test]
    fn poseidon_hash_2_deterministic() {
        let mut rng = OsRng;
        let a = pallas::Base::random(&mut rng);
        let b = pallas::Base::random(&mut rng);

        assert_eq!(poseidon_hash_2(a, b), poseidon_hash_2(a, b));
        // Non-commutative.
        assert_ne!(poseidon_hash_2(a, b), poseidon_hash_2(b, a));
    }

    // ================================================================
    // Condition 8 (Shares Sum Correctness) tests
    // ================================================================

    /// Shares that do NOT sum to total_note_value should fail condition 8.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn shares_sum_wrong_total_fails() {
        let (mut circuit, instance) = make_test_data();

        // Corrupt shares[3] so the sum no longer equals total_note_value.
        // Use a small value that still passes condition 9's range check,
        // isolating the condition 8 failure.
        circuit.shares[3] = Value::known(pallas::Base::from(999u64));

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        // Should fail: shares sum ≠ total_note_value.
        assert!(prover.verify().is_err());
    }

    // ================================================================
    // Condition 9 (Shares Range) tests
    // ================================================================

    /// A share at the maximum valid value (2^30 - 1) should pass.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn shares_range_max_valid() {
        let max_share = pallas::Base::from(SHARE_VALUE_LIMIT - 1); // 1,073,741,823
        let total = (0..16).fold(pallas::Base::zero(), |acc, _| acc + max_share);

        let mut rng = OsRng;
        // Derive proper voting key hierarchy.
        let vsk = pallas::Scalar::random(&mut rng);
        let vsk_nk = pallas::Base::random(&mut rng);
        let rivk_v = pallas::Scalar::random(&mut rng);
        let (vpk_g_d_affine, vpk_pk_d_affine) = derive_voting_address(vsk, vsk_nk, rivk_v);
        let vpk_g_d_x = *vpk_g_d_affine.coordinates().unwrap().x();
        let vpk_pk_d_x = *vpk_pk_d_affine.coordinates().unwrap().x();

        let voting_round_id = pallas::Base::random(&mut rng);
        let proposal_authority_old = pallas::Base::from(5u64); // bits 0 and 2 set
                                                               // proposal_id = 0 is now forbidden (sentinel); use proposal_id = 2 (bit 2 is set in 5).
        let proposal_id = 2u64;
        let van_comm_rand = pallas::Base::random(&mut rng);

        let vote_authority_note_old = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total,
            voting_round_id,
            proposal_authority_old,
            van_comm_rand,
        );
        let (auth_path, position, vote_comm_tree_root) =
            build_single_leaf_merkle_path(vote_authority_note_old);
        let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);
        let one_shifted = pallas::Base::from(1u64 << proposal_id);
        let proposal_authority_new = proposal_authority_old - one_shifted;
        let vote_authority_note_new = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total,
            voting_round_id,
            proposal_authority_new,
            van_comm_rand,
        );

        // Condition 11: real El Gamal encryption with max-value shares.
        let max_share_u64 = SHARE_VALUE_LIMIT - 1;
        let shares_u64: [u64; 16] = [max_share_u64; 16];

        let alpha_v = pallas::Scalar::random(&mut rng);
        let g = spend_auth_g_affine();
        let r_vpk = (g * (vsk + alpha_v)).to_affine();
        let r_vpk_x = *r_vpk.coordinates().unwrap().x();
        let r_vpk_y = *r_vpk.coordinates().unwrap().y();

        let mut circuit = Circuit::with_van_witnesses(
            Value::known(auth_path),
            Value::known(position),
            Value::known(vpk_g_d_affine),
            Value::known(vpk_pk_d_affine),
            Value::known(total),
            Value::known(proposal_authority_old),
            Value::known(van_comm_rand),
            Value::known(vote_authority_note_old),
            Value::known(vsk),
            Value::known(rivk_v),
            Value::known(vsk_nk),
            Value::known(alpha_v),
        );
        circuit.one_shifted = Value::known(one_shifted);
        circuit.shares = [Value::known(max_share); 16];
        let (bridge, vc) =
            set_conditions_10_to_12(&mut circuit, shares_u64, proposal_id, voting_round_id);

        let instance = Instance::from_parts(
            van_nullifier,
            r_vpk_x,
            r_vpk_y,
            vote_authority_note_new,
            vc,
            vote_comm_tree_root,
            pallas::Base::zero(),
            pallas::Base::from(proposal_id),
            voting_round_id,
            bridge,
            pallas::Base::from(TEST_BUCKET_COUNT),
        );

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    /// A share at exactly 2^30 should fail the range check.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn shares_range_overflow_fails() {
        let (mut circuit, instance) = make_test_data();

        // Set share_0 to 2^30 (one above the max valid value).
        // This will fail condition 9 AND condition 8 (sum mismatch),
        // but the important thing is the circuit rejects it.
        circuit.shares[0] = Value::known(pallas::Base::from(SHARE_VALUE_LIMIT));

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// A share that is a large field element (simulating underflow
    /// from subtraction) should fail the range check.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn shares_range_field_wrap_fails() {
        let (mut circuit, instance) = make_test_data();

        // Set share_0 to p - 1 (a wrapped negative value).
        // The 10-bit decomposition will produce a huge residual.
        circuit.shares[0] = Value::known(-pallas::Base::one());

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// Shares that sum correctly to total_note_value but with shares[0] = 2^30
    /// (one above the per-share maximum). Condition 8 (sum check) passes because
    /// total_note_value is set to match the sum. Condition 9 (range check) must
    /// still reject the individual overflow, confirming it checks each share
    /// independently — a correct sum does not bypass the per-share range gate.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn shares_range_single_overflow_correct_sum_fails() {
        let mut rng = OsRng;

        let overflow_share = pallas::Base::from(SHARE_VALUE_LIMIT);
        let normal_share_u64 = 625u64;
        // total_note_value = 2^30 + 15 * 625 so sum(shares) == total_note_value.
        let total_note_value = overflow_share + pallas::Base::from(15u64 * normal_share_u64);

        let vsk = pallas::Scalar::random(&mut rng);
        let vsk_nk = pallas::Base::random(&mut rng);
        let rivk_v = pallas::Scalar::random(&mut rng);
        let alpha_v = pallas::Scalar::random(&mut rng);
        let (vpk_g_d_affine, vpk_pk_d_affine) = derive_voting_address(vsk, vsk_nk, rivk_v);
        let vpk_g_d_x = *vpk_g_d_affine.coordinates().unwrap().x();
        let vpk_pk_d_x = *vpk_pk_d_affine.coordinates().unwrap().x();

        let voting_round_id = pallas::Base::random(&mut rng);
        let proposal_authority_old = pallas::Base::from(13u64); // bit 3 set
        let proposal_id = TEST_PROPOSAL_ID;
        let van_comm_rand = pallas::Base::random(&mut rng);

        let vote_authority_note_old = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_old,
            van_comm_rand,
        );
        let (auth_path, position, vote_comm_tree_root) =
            build_single_leaf_merkle_path(vote_authority_note_old);
        let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);
        let one_shifted = pallas::Base::from(1u64 << proposal_id);
        let proposal_authority_new = proposal_authority_old - one_shifted;
        let vote_authority_note_new = van_integrity_hash(
            vpk_g_d_x,
            vpk_pk_d_x,
            total_note_value,
            voting_round_id,
            proposal_authority_new,
            van_comm_rand,
        );

        // shares[0] overflows (2^30); shares[1..16] are valid (625 each).
        // The encryption is computed with these exact values so condition 11 is consistent.
        let shares_u64: [u64; 16] = {
            let mut arr = [normal_share_u64; 16];
            arr[0] = SHARE_VALUE_LIMIT;
            arr
        };

        let g = spend_auth_g_affine();
        let r_vpk = (g * (vsk + alpha_v)).to_affine();

        let mut circuit = Circuit::with_van_witnesses(
            Value::known(auth_path),
            Value::known(position),
            Value::known(vpk_g_d_affine),
            Value::known(vpk_pk_d_affine),
            Value::known(total_note_value),
            Value::known(proposal_authority_old),
            Value::known(van_comm_rand),
            Value::known(vote_authority_note_old),
            Value::known(vsk),
            Value::known(rivk_v),
            Value::known(vsk_nk),
            Value::known(alpha_v),
        );
        circuit.one_shifted = Value::known(one_shifted);
        circuit.shares = shares_u64.map(|s| Value::known(pallas::Base::from(s)));

        let (bridge, vote_commitment) =
            set_conditions_10_to_12(&mut circuit, shares_u64, proposal_id, voting_round_id);

        let instance = Instance::from_parts(
            van_nullifier,
            *r_vpk.coordinates().unwrap().x(),
            *r_vpk.coordinates().unwrap().y(),
            vote_authority_note_new,
            vote_commitment,
            vote_comm_tree_root,
            pallas::Base::zero(),
            pallas::Base::from(proposal_id),
            voting_round_id,
            bridge,
            pallas::Base::from(TEST_BUCKET_COUNT),
        );

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        // Condition 8 (sum check) passes: shares sum to total_note_value.
        // Condition 9 (range check) must reject shares[0] = 2^30 regardless.
        assert!(
            prover.verify().is_err(),
            "range check must reject a share equal to 2^30 even when the total sum is correct"
        );
    }

    // ================================================================
    // Condition 10′ (Bridge Re-Opening) and 11′ (Shares Hash) tests
    // ================================================================

    /// Valid selected commitments with a matching bridge should pass.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn bridge_reopening_valid_proof() {
        let (circuit, instance) = make_test_data();

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    /// A corrupted selected commitment changes the derived bridge, which no
    /// longer matches the public bridge input.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn bridge_reopening_altered_commitment_fails() {
        let (mut circuit, instance) = make_test_data();

        circuit.selected_commitments[0] = Value::known(pallas::Base::random(&mut OsRng));

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// Reordering the selected commitments must fail: the bridge binds each
    /// commitment to its share position.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn bridge_reopening_reordered_commitments_fail() {
        let (mut circuit, instance) = make_test_data();

        let comms = test_selected_commitments();
        circuit.selected_commitments[0] = Value::known(comms[1]);
        circuit.selected_commitments[1] = Value::known(comms[0]);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// Moving weight between shares while preserving the condition-8 sum must
    /// fail: the bridge binds each individual weight.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn bridge_reopening_altered_shares_with_preserved_sum_fail() {
        let (mut circuit, instance) = make_test_data();

        // make_test_data uses 16 × 625 shares; 626 + 624 preserves the sum.
        circuit.shares[0] = Value::known(pallas::Base::from(626u64));
        circuit.shares[1] = Value::known(pallas::Base::from(624u64));

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// A wrong bridge value in the instance must fail.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn bridge_reopening_wrong_instance_bridge_fails() {
        let (circuit, mut instance) = make_test_data();

        instance.bridge = pallas::Base::random(&mut OsRng);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// A wrong decision bucket count must fail: it is bound into both the
    /// bridge and the vote commitment.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn bridge_reopening_wrong_bucket_count_fails() {
        let (circuit, mut instance) = make_test_data();

        instance.decision_bucket_count = pallas::Base::from(TEST_BUCKET_COUNT + 1);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// A bridge assembled for a different round or proposal must fail: both
    /// context values are folded into the bridge preimage.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn bridge_reopening_cross_context_replay_fails() {
        let (circuit, instance) = make_test_data();

        let mut wrong_round = instance.clone();
        wrong_round.voting_round_id = pallas::Base::random(&mut OsRng);
        let prover = MockProver::run(K, &circuit, vec![wrong_round.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());

        let mut wrong_proposal = instance;
        wrong_proposal.proposal_id = pallas::Base::from(TEST_PROPOSAL_ID + 1);
        let prover =
            MockProver::run(K, &circuit, vec![wrong_proposal.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    // ================================================================
    // Condition 12 (Vote Commitment Integrity) tests
    // ================================================================

    /// Valid vote commitment (full Poseidon chain) should pass.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn vote_commitment_integrity_valid_proof() {
        let (circuit, instance) = make_test_data();

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    /// A wrong proposal_id in the instance should fail condition 12:
    /// the in-circuit proposal_id (copied from instance) will produce
    /// a different vote_commitment.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn vote_commitment_wrong_proposal_id_fails() {
        let (circuit, mut instance) = make_test_data();

        // Corrupt the proposal_id in the instance.
        instance.proposal_id = pallas::Base::from(999u64);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// A wrong vote_commitment in the instance should fail.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn vote_commitment_wrong_instance_fails() {
        let (circuit, mut instance) = make_test_data();

        // Corrupt the vote_commitment public input.
        instance.vote_commitment = pallas::Base::random(&mut OsRng);

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// The out-of-circuit vote_commitment_hash_v2 helper is deterministic.
    #[test]
    fn vote_commitment_hash_v2_deterministic() {
        let mut rng = OsRng;

        let rid = pallas::Base::random(&mut rng);
        let sh = pallas::Base::random(&mut rng);
        let pid = pallas::Base::from(5u64);
        let bucket_count = pallas::Base::from(4u64);

        let h1 = vote_commitment_hash_v2(rid, sh, pid, bucket_count);
        let h2 = vote_commitment_hash_v2(rid, sh, pid, bucket_count);
        assert_eq!(h1, h2);

        // Changing any input changes the hash.
        let h3 = vote_commitment_hash_v2(rid, sh, pallas::Base::from(6u64), bucket_count);
        assert_ne!(h1, h3);

        // Changing voting_round_id changes the hash.
        let h4 = vote_commitment_hash_v2(pallas::Base::from(999u64), sh, pid, bucket_count);
        assert_ne!(h1, h4);

        // DOMAIN_VC_V2 ensures separation from VAN hashes.
        // (Different arity prevents confusion, but domain tag adds defense-in-depth.)
        assert_ne!(h1, pallas::Base::zero());
    }

    // ================================================================
    // Instance and circuit sanity
    // ================================================================

    /// Instance must serialize to exactly 11 public inputs.
    #[test]
    fn instance_has_eleven_public_inputs() {
        let (_, instance) = make_test_data();
        assert_eq!(instance.to_halo2_instance().len(), 11);
    }

    /// Default circuit (all witnesses unknown) must not produce a valid proof.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn default_circuit_with_valid_instance_fails() {
        let (_, instance) = make_test_data();
        let circuit = Circuit::default();

        // Synthesis failure is also acceptable.
        if let Ok(prover) = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]) {
            assert!(prover.verify().is_err());
        }
    }

    /// Measures actual rows used by the vote-proof circuit via `CircuitCost::measure`.
    ///
    /// `CircuitCost` runs the floor planner against the circuit and tracks the
    /// highest row offset assigned in any column, giving the real "rows consumed"
    /// number rather than the theoretical 2^K capacity.
    ///
    /// Run with:
    ///   cargo test vote_proof::circuit::tests::row_budget -- --nocapture --ignored --test-threads=1
    #[test]
    #[ignore = "long-running row-budget diagnostic; run with `cargo test vote_proof::circuit::tests::row_budget -- --ignored --nocapture --test-threads=1`"]
    fn row_budget() {
        use std::println;
        use voting_crypto_deps::halo2_proofs::dev::CircuitCost;
        use voting_crypto_deps::pasta_curves::vesta;

        let (circuit, _) = make_test_data();

        // CircuitCost::measure runs the floor planner and returns layout statistics.
        // Fields are private, so extract them from the Debug representation.
        let cost = CircuitCost::<vesta::Point, _>::measure(K, &circuit);
        let debug = format!("{cost:?}");

        // Parse max_rows, max_advice_rows, max_fixed_rows from Debug string.
        let extract = |field: &str| -> usize {
            let prefix = format!("{field}: ");
            debug
                .split(&prefix)
                .nth(1)
                .and_then(|s| s.split([',', ' ', '}']).next())
                .and_then(|n| n.parse().ok())
                .unwrap_or(0)
        };

        let max_rows = extract("max_rows");
        let max_advice_rows = extract("max_advice_rows");
        let max_fixed_rows = extract("max_fixed_rows");
        let total_available = 1usize << K;

        println!("=== vote-proof circuit row budget (K={K}) ===");
        println!("  max_rows (floor-planner high-water mark): {max_rows}");
        println!("  max_advice_rows:                          {max_advice_rows}");
        println!("  max_fixed_rows:                           {max_fixed_rows}");
        println!("  2^K  (total available rows):              {total_available}");
        println!(
            "  headroom:                                 {}",
            total_available.saturating_sub(max_rows)
        );
        println!(
            "  utilisation:                              {:.1}%",
            100.0 * max_rows as f64 / total_available as f64
        );
        println!();
        println!("  Full debug: {debug}");

        // ---------------------------------------------------------------
        // Witness-independence check: Circuit::default() (all unknowns)
        // must produce exactly the same layout as the filled circuit.
        // If these differ, the row count depends on witness values and
        // the measurement above cannot be trusted as a production bound.
        // ---------------------------------------------------------------
        let cost_default = CircuitCost::<vesta::Point, _>::measure(K, &Circuit::default());
        let debug_default = format!("{cost_default:?}");
        let max_rows_default = debug_default
            .split("max_rows: ")
            .nth(1)
            .and_then(|s| s.split([',', ' ', '}']).next())
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or(0);
        if max_rows_default == max_rows {
            println!(
                "  Witness-independence: PASS \
                (Circuit::default() max_rows={max_rows_default} == filled max_rows={max_rows})"
            );
        } else {
            println!(
                "  Witness-independence: FAIL \
                (Circuit::default() max_rows={max_rows_default} != filled max_rows={max_rows}) \
                — row count depends on witness values!"
            );
        }

        // ---------------------------------------------------------------
        // VOTE_COMM_TREE_DEPTH sanity check: confirm the circuit constant
        // matches the canonical value in vote_commitment_tree::TREE_DEPTH
        // (24 as of this writing). A mismatch would mean test data uses a
        // shallower tree than production.
        // ---------------------------------------------------------------
        println!("  VOTE_COMM_TREE_DEPTH (circuit constant): {VOTE_COMM_TREE_DEPTH}");

        // ---------------------------------------------------------------
        // Minimum-K probe: find the smallest K at which MockProver passes.
        // Useful for evaluating whether K can be reduced.
        // ---------------------------------------------------------------
        for probe_k in 11u32..=K {
            let (c, inst) = make_test_data();
            match MockProver::run(probe_k, &c, vec![inst.to_halo2_instance()]) {
                Err(_) => {
                    println!("  K={probe_k}: not enough rows (synthesizer rejected)");
                    continue;
                }
                Ok(p) => match p.verify() {
                    Ok(()) => {
                        println!("  Minimum viable K: {probe_k} (2^{probe_k} = {} rows, {:.1}% headroom)",
                            1usize << probe_k,
                            100.0 * (1.0 - max_rows as f64 / (1usize << probe_k) as f64));
                        break;
                    }
                    Err(_) => println!("  K={probe_k}: too small"),
                },
            }
        }
    }
}
