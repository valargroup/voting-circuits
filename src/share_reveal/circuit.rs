//! The Share Reveal circuit implementation (ZKP #3).
//!
//! Proves that a publicly-revealed encrypted share came from a valid,
//! registered vote commitment — without revealing which one. The circuit
//! verifies 5 conditions:
//!
//! - **Condition 1**: VC Membership — Poseidon Merkle path from `vote_commitment`
//!   to `vote_comm_tree_root`.
//! - **Condition 2**: Vote Commitment Integrity — `vote_commitment =
//!   Poseidon(DOMAIN_VC_V2, voting_round_id, shares_hash, proposal_id,
//!   decision_bucket_count)`.
//! - **Condition 3**: Shares Hash Integrity — `shares_hash =
//!   Poseidon(share_comm_0, ..., share_comm_15)`, where share_comms are
//!   private witnesses transitively bound to the public tree root.
//! - **Condition 4**: Primary Share Binding — the voting client knows a
//!   blind such that `share_comms[share_index]` equals the weighted selected
//!   commitment over all 16 revealed bucket ciphertexts
//!   (see `crate::bridge` for the authoritative 34-input shape; both
//!   coordinates of every point defend against ciphertext
//!   sign-malleability), binding the publicly revealed encrypted bucket
//!   vector to the committed set.
//! - **Condition 5**: Share Nullifier Integrity — `share_nullifier` is
//!   correctly derived as
//!   `Poseidon(domain_tag, vote_commitment, share_index, blind)`.
//!   `blind` is the share commitment blinding factor — a secret held by
//!   the voting client (the host program that built ZKP 1.5 / #2 and now
//!   builds this reveal proof). Using the blind (rather than a
//!   ciphertext coordinate) ensures the nullifier is not publicly
//!   derivable from on-chain data, since ciphertext coordinates are
//!   posted as public inputs alongside the proof. Round, proposal, bucket
//!   count, and `shares_hash` bind through the `vote_commitment` preimage;
//!   `share_comms` bind one hop earlier through `shares_hash`. The resulting
//!   `vote_commitment` is checked against the vote commitment tree.
//!
//! ## Privacy
//!
//! Only the primary share's blind is supplied as a private witness, so
//! the voting client does not need to surface the other 15 blinds when
//! it assembles the reveal. The 16 `share_comms` are private witnesses —
//! they never appear on chain, preserving share-level unlinkability.
//! Soundness is guaranteed because `share_comms` are transitively bound
//! to the public `vote_comm_tree_root` via
//! `shares_hash → vote_commitment → Merkle path`; the revealed ciphertext
//! coordinates bind to the selected `share_comm` through Poseidon preimage
//! resistance of `Poseidon(blind, c1_x, c2_x, c1_y, c2_y)`.
//!
//! Authoritative hash sources: `crate::bridge` owns the weighted
//! selected-commitment preimage, `crate::shares_hash` owns the aggregate
//! `Poseidon<16>` shares hash, `crate::gadgets::vote_commitment` owns the
//! vote commitment preimage, and `crate::domain_tags` owns the share-spend
//! domain tag encoding. This module's prose points to those owners rather
//! than defining competing formulas.
//!
//! ## Column layout
//!
//! - 17 advice columns: 9 shared by the general gadgets and primary Poseidon
//!   configuration, 4 for the second Merkle Poseidon configuration, and 4
//!   for the dedicated wide-hash Poseidon configuration (condition 4's
//!   34-input selected commitment).
//! - 20 explicitly allocated fixed columns for the three Poseidon
//!   configurations and general constants.
//! - 1 instance column (69 public inputs).
//! - K = 10 (1,024 rows).

use std::vec::Vec;

use itertools::Itertools;
use voting_crypto_deps::halo2_gadgets::{
    poseidon::{
        primitives::{self as poseidon, ConstantLength},
        Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
    },
    utilities::bool_check,
};
use voting_crypto_deps::halo2_proofs::{
    circuit::{floor_planner, AssignedCell, Layouter, Value},
    plonk::{
        self, Advice, Column, ConstraintSystem, Constraints, Expression, Fixed,
        Instance as InstanceColumn, Selector,
    },
    poly::Rotation,
};
use voting_crypto_deps::orchard::circuit::gadget::assign_free_advice;
use voting_crypto_deps::pasta_curves::{pallas, vesta};

use crate::{
    bridge::{hash_selected_commitment_in_circuit, WeightedShareCiphertexts, MAX_DECISION_BUCKETS},
    gadgets::{
        poseidon_merkle::{synthesize_poseidon_merkle_path_with_config_schedule, MerkleSwapGate},
        vote_commitment,
    },
    params::VOTE_COMM_TREE_DEPTH,
    shares_hash::compute_shares_hash_from_comms_in_circuit,
};

// ================================================================
// Constants
// ================================================================

/// Circuit size (2^K rows).
///
/// K=10 (1,024 rows). Condition 4's 34-input selected-commitment hash (~17
/// Poseidon permutations) runs on a dedicated third Poseidon configuration
/// so the `V1` floor planner can overlap it with the Merkle and primary
/// tracks. `CircuitCost::measure` reports an 855-row high-water mark
/// (16.5% headroom).
///
/// Run the `row_budget` test to re-measure after circuit changes:
///   `cargo test row_budget -- --nocapture --ignored`
pub const K: u32 = 10;

/// Independent Poseidon configurations used by the vote-commitment Merkle path.
const MERKLE_CONFIGS: usize = 2;

/// The primary Poseidon configuration also handles all non-Merkle hashes, so
/// the 24 Merkle levels are weighted 8/16 across the two configurations.
const MERKLE_CONFIG_SCHEDULE: [usize; VOTE_COMM_TREE_DEPTH] = [
    0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
];

// ================================================================
// Public input offsets (37 = 4M + 5 field elements).
// ================================================================

/// Public input offset for the share nullifier (prevents double-counting).
const SHARE_NULLIFIER_PUBLIC_OFFSET: usize = 0;
/// Base offset of the revealed share's ciphertext coordinates.
///
/// Bucket `j` occupies offsets `1 + 4j .. 1 + 4j + 4` in the canonical
/// preimage order `c1_x, c2_x, c1_y, c2_y` (see
/// [`crate::bridge::WeightedShareCiphertexts::to_preimage`]). All values are
/// caller-supplied. Condition 4 binds the full vector transitively to the
/// committed vote by proving the 34-input weighted selected commitment over
/// them equals the muxed private `share_comm`; both coordinates of every
/// point are bound, preventing ciphertext sign-malleability.
const ENC_SHARE_COORDS_PUBLIC_OFFSET: usize = 1;
/// Public input offset for the proposal identifier.
const PROPOSAL_ID_PUBLIC_OFFSET: usize = 1 + 4 * MAX_DECISION_BUCKETS;
/// Public input offset for the vote commitment tree root.
const VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET: usize = PROPOSAL_ID_PUBLIC_OFFSET + 1;
/// Public input offset for the voting round identifier.
///
/// Constrained in-circuit: `voting_round_id` is hashed into `vote_commitment`
/// and `vote_commitment` is hashed into the share nullifier. That transitive
/// path binds the nullifier to a specific round. This prevents cross-round
/// proof replay because the commitment tree is global, not per-round, so
/// `vote_comm_tree_root` alone does not provide round scoping. The chain also
/// validates that `voting_round_id` matches an active session (Gov Steps V1
/// §5.4 "Out-of-circuit checks").
const VOTING_ROUND_ID_PUBLIC_OFFSET: usize = VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET + 1;
/// Public input offset for the active decision bucket count `D`.
///
/// Bound into the condition-2 vote commitment; the verifier must
/// authenticate it from the proposal's governance declaration.
const DECISION_BUCKET_COUNT_PUBLIC_OFFSET: usize = VOTING_ROUND_ID_PUBLIC_OFFSET + 1;

// ================================================================
// Out-of-circuit helpers
// ================================================================

/// Domain separator for share nullifiers, encoded as a Pallas base field element.
///
/// `"share spend"` → 32-byte zero-padded array → `Fp::from_repr`.
pub use crate::domain_tags::share_spend as domain_tag_share_spend;

/// Out-of-circuit share nullifier hash (condition 5).
///
/// ```text
/// share_nullifier = Poseidon(domain_tag, vote_commitment, share_index, blind)
/// ```
///
/// Single `ConstantLength<4>` call (2 permutations at rate=2).
/// `blind` is the share commitment blinding factor for this share index.
/// Because blinds are never posted on-chain, the nullifier cannot be
/// derived by an observer — even one who knows the vote commitment tree
/// contents and the public ciphertext coordinates. Round, proposal, decision,
/// and `shares_hash` bind through the `vote_commitment` preimage;
/// `share_comms` bind one hop earlier through `shares_hash`. The nullifier
/// deliberately consumes the parent vote commitment instead of re-hashing its
/// full preimage.
pub fn share_nullifier_hash(
    vote_commitment: pallas::Base,
    share_index: pallas::Base,
    blind: pallas::Base,
) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<4>, 3, 2>::init().hash([
        domain_tag_share_spend(),
        vote_commitment,
        share_index,
        blind,
    ])
}

// ================================================================
// Config
// ================================================================

/// Configuration for the Share Reveal circuit.
///
/// Holds the Poseidon chip config, the Merkle swap gate selector,
/// and the share commitment multiplexer gate selector.
#[derive(Clone, Debug)]
pub struct Config {
    /// Public input column (9 field elements).
    primary: Column<InstanceColumn>,
    /// 9 advice columns for private witness data.
    advices: [Column<Advice>; 9],
    /// Poseidon hash chip configuration.
    poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    /// Dedicated Poseidon configuration for condition 4's 34-input
    /// selected-commitment hash, so its ~33 permutations overlap the other
    /// tracks instead of extending them.
    wide_poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    /// Independent Poseidon configurations used by scheduled Merkle levels.
    merkle_poseidon_configs: [PoseidonConfig<pallas::Base, 3, 2>; MERKLE_CONFIGS],
    /// Merkle conditional swap gate (condition 1).
    merkle_swap: MerkleSwapGate,
    /// Selector for the share commitment multiplexer gate (condition 4).
    ///
    /// Fires on a 4-row block (9 advice columns, Rotation 0..3):
    ///   Row 0: sel_0..sel_8     (advices[0..9])
    ///   Row 1: sel_9..sel_15    (advices[0..7]),  comm_0..comm_1  (advices[7..9])
    ///   Row 2: comm_2..comm_10  (advices[0..9])
    ///   Row 3: comm_11..comm_15 (advices[0..5]),  selected_comm   (advices[5]),
    ///          share_index      (advices[6])
    ///
    /// Constraints:
    /// - Each sel_i is boolean.
    /// - Exactly one sel_i is 1.
    /// - share_index == Σ i * sel_i (index reconstruction, replaces 16 per-bit checks).
    /// - selected_comm = Σ sel_i * comm_i.
    q_share_comm_mux: Selector,
}

impl Config {
    /// Constructs a Poseidon chip from this configuration.
    fn poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.poseidon_config.clone())
    }

    /// Assigns a field-element constant to an advice cell so the value is
    /// baked into the verification key via `assign_advice_from_constant`.
    fn assign_constant(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        label: &'static str,
        value: pallas::Base,
    ) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
        layouter.assign_region(
            || label,
            |mut region| region.assign_advice_from_constant(|| label, self.advices[0], 0, value),
        )
    }
}

// ================================================================
// Circuit
// ================================================================

/// The Share Reveal circuit (ZKP #3).
///
/// Proves that a publicly-revealed encrypted share came from a valid,
/// registered vote commitment — without revealing which one.
#[derive(Clone, Debug)]
pub struct Circuit {
    // === Condition 1: VC Membership ===
    /// Merkle authentication path (sibling hashes at each tree level).
    pub(super) vote_comm_tree_path: Value<[pallas::Base; VOTE_COMM_TREE_DEPTH]>,
    /// Leaf position in the vote commitment tree.
    pub(super) vote_comm_tree_position: Value<u32>,

    // === Condition 3: Shares Hash Integrity ===
    /// Pre-computed per-share Poseidon commitments (private witnesses).
    ///
    /// Shape: see `crate::shares_hash` — five-input
    /// `Poseidon(blind, c1_x, c2_x, c1_y, c2_y)` including the
    /// y-coordinates that defend against ciphertext sign-malleability.
    /// Transitively bound to the public tree root via
    /// `shares_hash → vote_commitment → Merkle path`.
    pub(super) share_comms: [Value<pallas::Base>; 16],

    // === Condition 4: Primary Share Binding ===
    /// Blind factor for the revealed share. The synthesize body
    /// (see the "Condition 4: Primary Share Binding" region below) recomputes
    /// `Poseidon(primary_blind, c1_x, c2_x, c1_y, c2_y)` using the shared
    /// `crate::shares_hash` gadget and constrains it to equal
    /// `share_comms[share_index]`; the y-coordinates are the
    /// sign-malleability defense and the gadget is the single source of
    /// truth for the preimage shape.
    pub(super) primary_blind: Value<pallas::Base>,

    // === Share selection ===
    /// Which of the 16 shares is being revealed (0..15).
    pub(super) share_index: Value<pallas::Base>,

    // === Condition 5: Share Nullifier Integrity ===
    /// The vote commitment leaf value (links conditions 1, 2, and 5).
    pub(super) vote_commitment: Value<pallas::Base>,
}

impl Default for Circuit {
    fn default() -> Self {
        Self {
            vote_comm_tree_path: Value::unknown(),
            vote_comm_tree_position: Value::unknown(),
            share_comms: [Value::unknown(); 16],
            primary_blind: Value::unknown(),
            share_index: Value::unknown(),
            vote_commitment: Value::unknown(),
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
        // 9 advice columns — the minimum required by the three gadgets in this circuit:
        //   [0..4]  Merkle conditional swap gate (pos_bit, current, sibling, left, right).
        //   [5]     Poseidon Pow5T3 partial S-box column (internal to the chip).
        //   [6..8]  Poseidon width-3 state columns.
        // The share commitment mux gate (condition 4) reuses all 9 columns across
        // 4 rows to pack its 16 one-hot selectors + 16 commitments without needing
        // an additional column.
        let advices: [Column<Advice>; 9] = core::array::from_fn(|_| meta.advice_column());
        for col in &advices {
            meta.enable_equality(*col);
        }

        // Instance column for public inputs.
        let primary = meta.instance_column();
        meta.enable_equality(primary);

        // 8 fixed columns shared between Poseidon round constants and
        // general constants.
        let lagrange_coeffs: [Column<Fixed>; 8] = core::array::from_fn(|_| meta.fixed_column());
        let rc_a = lagrange_coeffs[2..5].try_into().unwrap();
        let rc_b = lagrange_coeffs[5..8].try_into().unwrap();

        // Enable constants via the first fixed column.
        meta.enable_constant(lagrange_coeffs[0]);

        // Poseidon chip: P128Pow5T3 with width 3, rate 2.
        // State columns: advices[6..8], partial S-box: advices[5].
        let poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
            meta,
            advices[6..9].try_into().unwrap(),
            advices[5],
            rc_a,
            rc_b,
        );

        // A second independent Poseidon configuration lets the V1 floor planner
        // overlap Merkle hashes assigned to different column sets. The hash chain
        // is preserved by equality-constrained copies between configuration outputs.
        let merkle_advice: [Column<Advice>; 4] = core::array::from_fn(|_| meta.advice_column());
        // `Pow5Chip::configure` equality-enables the three state columns used
        // for cross-region handoffs. The partial-S-box column is internal
        // scratch space and must not be added to the permutation argument.
        let merkle_round_constants: [Column<Fixed>; 6] =
            core::array::from_fn(|_| meta.fixed_column());
        let merkle_poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
            meta,
            merkle_advice[1..4].try_into().unwrap(),
            merkle_advice[0],
            merkle_round_constants[..3].try_into().unwrap(),
            merkle_round_constants[3..].try_into().unwrap(),
        );
        let merkle_poseidon_configs = [poseidon_config.clone(), merkle_poseidon_config];

        // Dedicated wide-hash Poseidon configuration (condition 4).
        let wide_advice: [Column<Advice>; 4] = core::array::from_fn(|_| meta.advice_column());
        let wide_round_constants: [Column<Fixed>; 6] =
            core::array::from_fn(|_| meta.fixed_column());
        let wide_poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
            meta,
            wide_advice[1..4].try_into().unwrap(),
            wide_advice[0],
            wide_round_constants[..3].try_into().unwrap(),
            wide_round_constants[3..].try_into().unwrap(),
        );

        // Merkle conditional swap gate (condition 1).
        let merkle_swap = MerkleSwapGate::configure(
            meta,
            [advices[0], advices[1], advices[2], advices[3], advices[4]],
        );

        // Share commitment multiplexer gate (condition 4).
        // Col →  [0]       [1]       [2]        [3]        [4]        [5]        [6]       [7]       [8]
        // ------+---------+---------+----------+----------+----------+----------+---------+---------+---------
        // Row 0 | sel[0]  | sel[1]  | sel[2]   | sel[3]   | sel[4]   | sel[5]   | sel[6]  | sel[7]  | sel[8]
        // Row 1 | sel[9]  | sel[10] | sel[11]  | sel[12]  | sel[13]  | sel[14]  | sel[15] | comm[0] | comm[1]
        // Row 2 | comm[2] | comm[3] | comm[4]  | comm[5]  | comm[6]  | comm[7]  | comm[8] | comm[9] |comm[10]
        // Row 3 | comm[11]| comm[12]| comm[13] | comm[14] | comm[15] | sel_comm | share_idx| —      | —
        let q_share_comm_mux = meta.selector();
        meta.create_gate("share commitment multiplexer", |meta| {
            let q = meta.query_selector(q_share_comm_mux);

            let sel: [_; 16] = [
                meta.query_advice(advices[0], Rotation::cur()),
                meta.query_advice(advices[1], Rotation::cur()),
                meta.query_advice(advices[2], Rotation::cur()),
                meta.query_advice(advices[3], Rotation::cur()),
                meta.query_advice(advices[4], Rotation::cur()),
                meta.query_advice(advices[5], Rotation::cur()),
                meta.query_advice(advices[6], Rotation::cur()),
                meta.query_advice(advices[7], Rotation::cur()),
                meta.query_advice(advices[8], Rotation::cur()),
                meta.query_advice(advices[0], Rotation::next()),
                meta.query_advice(advices[1], Rotation::next()),
                meta.query_advice(advices[2], Rotation::next()),
                meta.query_advice(advices[3], Rotation::next()),
                meta.query_advice(advices[4], Rotation::next()),
                meta.query_advice(advices[5], Rotation::next()),
                meta.query_advice(advices[6], Rotation::next()),
            ];

            let comm: [_; 16] = [
                meta.query_advice(advices[7], Rotation::next()),
                meta.query_advice(advices[8], Rotation::next()),
                meta.query_advice(advices[0], Rotation(2)),
                meta.query_advice(advices[1], Rotation(2)),
                meta.query_advice(advices[2], Rotation(2)),
                meta.query_advice(advices[3], Rotation(2)),
                meta.query_advice(advices[4], Rotation(2)),
                meta.query_advice(advices[5], Rotation(2)),
                meta.query_advice(advices[6], Rotation(2)),
                meta.query_advice(advices[7], Rotation(2)),
                meta.query_advice(advices[8], Rotation(2)),
                meta.query_advice(advices[0], Rotation(3)),
                meta.query_advice(advices[1], Rotation(3)),
                meta.query_advice(advices[2], Rotation(3)),
                meta.query_advice(advices[3], Rotation(3)),
                meta.query_advice(advices[4], Rotation(3)),
            ];

            let selected_comm = meta.query_advice(advices[5], Rotation(3));
            let share_index = meta.query_advice(advices[6], Rotation(3));

            let one = Expression::Constant(pallas::Base::one());

            // Boolean checks for all 16 selection bits.
            let bool_checks: Vec<(&'static str, Expression<pallas::Base>)> = (0..16)
                .map(|i| ("bool sel_i", bool_check(sel[i].clone())))
                .collect();

            // Sum check for selectors (only one is 1)
            let sum_expr = sel
                .iter()
                .skip(1)
                .fold(sel[0].clone(), |acc, s| acc + s.clone());
            let sum_check = ("sum sel == 1", sum_expr - one);

            // Index reconstruction: share_index == sum(i * sel[i]).
            //
            // Given bool + sum guarantees exactly one sel[j] = 1, the sum collapses
            // to j.
            let reconstructed = sel
                .iter()
                .enumerate()
                .skip(1)
                .fold(Expression::Constant(pallas::Base::zero()), |acc, (i, s)| {
                    acc + Expression::Constant(pallas::Base::from(i as u64)) * s.clone()
                });
            let index_reconstruct = ("index reconstruct", share_index.clone() - reconstructed);

            // Selected commitment must equal the dot product:
            // selected_comm == Σ sel[i] * comm[i]
            let comm_mux_expr = comm
                .iter()
                .zip_eq(sel.iter())
                .fold(selected_comm, |acc, (c, s)| acc - s.clone() * c.clone());
            let comm_mux = ("comm mux", comm_mux_expr);

            // What these four groups together guarantee:
            // The bool + sum constraints establish one-hotness.
            // Given one-hotness, the index reconstruction collapses to share_index == j where j is the unique set position.
            // The mux constraint then collapses to selected_comm == comm[j].
            // Combined with the constrain_equal(derived_comm, selected_comm), the full chain is:
            // derived_comm  ==  comm[share_index]  ==  share_comms[share_index]
            // The last equality is enforced by copy_advice.
            let mut constraints: Vec<(&'static str, Expression<pallas::Base>)> = bool_checks;
            constraints.push(sum_check);
            constraints.push(index_reconstruct);
            constraints.push(comm_mux);

            Constraints::with_selector(q, constraints)
        });

        Config {
            primary,
            advices,
            poseidon_config,
            wide_poseidon_config,
            merkle_poseidon_configs,
            merkle_swap,
            q_share_comm_mux,
        }
    }

    #[allow(non_snake_case)]
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), plonk::Error> {
        // ---------------------------------------------------------------
        // Witness private inputs.
        // ---------------------------------------------------------------

        let vote_commitment = assign_free_advice(
            layouter.namespace(|| "witness vote_commitment"),
            config.advices[0],
            self.vote_commitment,
        )?;
        // Clone for conditions 2 and 5 (Merkle path in condition 1 copies
        // the cell, so the original reference remains valid).
        let vote_commitment_cond2 = vote_commitment.clone();
        let vote_commitment_cond5 = vote_commitment.clone();

        let share_index = assign_free_advice(
            layouter.namespace(|| "witness share_index"),
            config.advices[0],
            self.share_index,
        )?;
        let share_index_cond5 = share_index.clone();

        let primary_blind = assign_free_advice(
            layouter.namespace(|| "witness primary_blind"),
            config.advices[0],
            self.primary_blind,
        )?;
        let primary_blind_cond5 = primary_blind.clone();

        // Copy proposal_id and vote_decision from instance into advice.
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

        let decision_bucket_count = layouter.assign_region(
            || "copy decision_bucket_count from instance",
            |mut region| {
                region.assign_advice_from_instance(
                    || "decision_bucket_count",
                    config.primary,
                    DECISION_BUCKET_COUNT_PUBLIC_OFFSET,
                    config.advices[0],
                    0,
                )
            },
        )?;

        // Copy voting_round_id from instance into advice.
        // Used in condition 2 (vote commitment integrity).
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
        let voting_round_id_cond2 = voting_round_id;

        // ---------------------------------------------------------------
        // Witness 16 share_comms as private advice cells.
        //
        // Transitively bound to the public vote_comm_tree_root via:
        //   share_comms → shares_hash → vote_commitment → Merkle root
        // ---------------------------------------------------------------

        let share_comms: [AssignedCell<pallas::Base, pallas::Base>; 16] = {
            let mut cells = Vec::with_capacity(16);
            for i in 0..16 {
                cells.push(assign_free_advice(
                    layouter.namespace(|| format!("witness share_comm[{i}]")),
                    config.advices[0],
                    self.share_comms[i],
                )?);
            }
            cells.try_into().unwrap()
        };

        // Clone for condition 4 mux (condition 3's Poseidon consumes them).
        let share_comms_cond4: [AssignedCell<pallas::Base, pallas::Base>; 16] =
            core::array::from_fn(|i| share_comms[i].clone());

        // ---------------------------------------------------------------
        // Condition 3: Shares Hash Integrity.
        //
        // shares_hash = Poseidon(share_comm_0, ..., share_comm_15)
        //
        // The share_comms are private witnesses. Soundness comes from the
        // transitive binding to the public tree root via condition 2 + 1.
        // ---------------------------------------------------------------

        let shares_hash = compute_shares_hash_from_comms_in_circuit(
            config.poseidon_chip(),
            layouter.namespace(|| "cond3: shares_hash from comms"),
            share_comms,
        )?;
        let shares_hash_cond2 = shares_hash.clone();

        // ---------------------------------------------------------------
        // Condition 4: Primary Share Binding.
        //
        // The ciphertext coordinates are caller-supplied public inputs. ZKP #2
        // publishes only the aggregate vote_commitment, not per-share
        // ciphertext coordinates, so this is a transitive hash binding rather
        // than a direct comparison against vote-proof public inputs.
        //
        // Proves that the ciphertext coordinates of the *revealed* share
        // correspond to the share commitment at the declared
        // `share_index`, by recomputing the commitment and matching it
        // against the muxed-out `share_comms[share_index]`:
        //   derived_comm = Poseidon(primary_blind, enc_c1_x, enc_c2_x,
        //                          enc_c1_y, enc_c2_y)
        //   share_comms[share_index] == derived_comm
        //
        // Defense-by-rejection: an adversary that has seen the on-chain
        // ciphertexts but does not hold the blind cannot claim the wrong share
        // is the revealed one. The recomputed commitment must match the muxed
        // `share_comms[share_index]`; otherwise condition 4 rejects. The
        // load-bearing assumption is Poseidon preimage resistance for the
        // share-commitment hash shape owned by `crate::shares_hash`.
        // ---------------------------------------------------------------

        let coords: [AssignedCell<pallas::Base, pallas::Base>; 4 * MAX_DECISION_BUCKETS] = {
            let mut cells = Vec::with_capacity(4 * MAX_DECISION_BUCKETS);
            for slot in 0..4 * MAX_DECISION_BUCKETS {
                cells.push(layouter.assign_region(
                    || format!("copy ciphertext coordinate {slot} from instance"),
                    |mut region| {
                        region.assign_advice_from_instance(
                            || format!("coordinate {slot}"),
                            config.primary,
                            ENC_SHARE_COORDS_PUBLIC_OFFSET + slot,
                            config.advices[0],
                            0,
                        )
                    },
                )?);
            }
            cells.try_into().expect("4M coordinate cells")
        };

        let derived_comm = hash_selected_commitment_in_circuit(
            PoseidonChip::construct(config.wide_poseidon_config.clone()),
            layouter.namespace(|| "cond4: weighted selected commitment"),
            config.advices[0],
            primary_blind,
            coords,
            0,
        )?;

        // Mux share_comms by share_index → selected_comm.
        //
        // Col →  [0]       [1]       [2]        [3]        [4]        [5]        [6]       [7]       [8]       [9]
        // ------+---------+---------+----------+----------+----------+----------+---------+---------+---------+---------
        // Row 0 | sel[0]  | sel[1]  | sel[2]   | sel[3]   | sel[4]   | sel[5]   | sel[6]  | sel[7]  | sel[8]  | sel[9]
        // Row 1 | sel[10] | sel[11] | sel[12]  | sel[13]  | sel[14]  | sel[15]  | comm[0] | comm[1] | comm[2] | comm[3]
        // Row 2 | comm[4] | comm[5] | comm[6]  | comm[7]  | comm[8]  | comm[9]  | comm[10]| comm[11]| comm[12]| comm[13]
        // Row 3 | comm[14]| comm[15]| sel_comm | share_idx| —        | —        | —       | —       | —       | —
        let selected_comm = layouter.assign_region(
            || "cond4: share commitment mux",
            |mut region| {
                config.q_share_comm_mux.enable(&mut region, 0)?;

                // Create a selector map
                let sel_values: [Value<pallas::Base>; 16] = core::array::from_fn(|i| {
                    self.share_index.map(|idx| {
                        if idx == pallas::Base::from(i as u64) {
                            pallas::Base::one()
                        } else {
                            pallas::Base::zero()
                        }
                    })
                });

                // Assign the one-hot selector bits into the region. We use assign_advice
                // (fresh allocation) because sel_values are computed locally and have no
                // prior cell to copy from. There are 16 bits spread across 9 advice
                // columns, so they spill from row 0 into the first 7 columns of row 1.
                // Layout table: (sel_start, count, advice_col_offset, row)
                for (sel_start, count, col_off, row) in [(0, 9, 0, 0), (9, 7, 0, 1)] {
                    for i in 0..count {
                        region.assign_advice(
                            || format!("sel_{}", sel_start + i),
                            config.advices[col_off + i],
                            row,
                            || sel_values[sel_start + i],
                        )?;
                    }
                }

                // Copy the 16 share commitments into the region. We use copy_advice
                // (equality-constrained copy) instead of assign_advice because these
                // cells were allocated earlier in separate regions; copy_advice ties
                // this cell to the original via the permutation argument, preventing
                // the prover from substituting a different value. The 16 commitments
                // also spill across multiple rows alongside the selector bits above.
                // Layout table: (comm_start, count, advice_col_offset, row)
                for (comm_start, count, col_off, row) in [(0, 2, 7, 1), (2, 9, 0, 2), (11, 5, 0, 3)]
                {
                    for i in 0..count {
                        share_comms_cond4[comm_start + i].copy_advice(
                            || format!("comm_{}", comm_start + i),
                            &mut region,
                            config.advices[col_off + i],
                            row,
                        )?;
                    }
                }

                // Select the correct commitment via dot product selector.
                // selected_comm_val = Σ sel[i] * comm[i]
                let selected_comm_val =
                    (0..16).fold(Value::known(pallas::Base::zero()), |acc, i| {
                        acc.zip(sel_values[i])
                            .zip(share_comms_cond4[i].value().copied())
                            .map(|((a, s), c)| a + s * c)
                    });
                let selected_comm = region.assign_advice(
                    || "selected_comm",
                    config.advices[5],
                    3,
                    || selected_comm_val,
                )?;

                share_index.copy_advice(|| "share_index", &mut region, config.advices[6], 3)?;

                Ok(selected_comm)
            },
        )?;

        // Ensure that the derived commitment is equal to selected
        layouter.assign_region(
            || "cond4: derived_comm == selected_comm",
            |mut region| region.constrain_equal(derived_comm.cell(), selected_comm.cell()),
        )?;

        // ---------------------------------------------------------------
        // Condition 2: Vote Commitment Integrity (v2).
        //
        // vote_commitment = Poseidon(DOMAIN_VC_V2, voting_round_id,
        //                            shares_hash, proposal_id,
        //                            decision_bucket_count)
        //
        // Same hash as the shared vote-commitment helper and the vote
        // commitment tree; the plaintext vote decision of v1 is gone.
        // ---------------------------------------------------------------

        // DOMAIN_VC_V2 constant (baked into the VK).
        let domain_vc = config.assign_constant(
            &mut layouter,
            "cond2: DOMAIN_VC_V2 constant",
            pallas::Base::from(vote_commitment::DOMAIN_VC_V2),
        )?;

        let derived_vc = vote_commitment::vote_commitment_poseidon(
            &config.poseidon_config,
            &mut layouter,
            "cond2",
            domain_vc,
            voting_round_id_cond2,
            shares_hash_cond2,
            proposal_id,
            decision_bucket_count,
        )?;

        // Constrain derived vote_commitment == witnessed vote_commitment.
        layouter.assign_region(
            || "cond2: vote_commitment equality",
            |mut region| region.constrain_equal(derived_vc.cell(), vote_commitment_cond2.cell()),
        )?;

        // ---------------------------------------------------------------
        // Condition 1: VC Membership.
        //
        // MerklePath(vote_commitment, position, path) = vote_comm_tree_root
        //
        // 24-level Poseidon Merkle path (LSB-first position bits).
        // Uses the shared poseidon_merkle gadget.
        // ---------------------------------------------------------------
        {
            let root = synthesize_poseidon_merkle_path_with_config_schedule::<
                VOTE_COMM_TREE_DEPTH,
                MERKLE_CONFIGS,
            >(
                &config.merkle_swap,
                &config.merkle_poseidon_configs,
                &MERKLE_CONFIG_SCHEDULE,
                &mut layouter,
                config.advices[0],
                vote_commitment,
                self.vote_comm_tree_position,
                self.vote_comm_tree_path,
                "cond1: merkle",
            )?;

            // Bind the computed Merkle root to the public input.
            layouter.constrain_instance(
                root.cell(),
                config.primary,
                VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET,
            )?;
        }

        // ---------------------------------------------------------------
        // Condition 5: Share Nullifier Integrity.
        //
        // share_nullifier = Poseidon(domain_tag, vote_commitment, share_index,
        //                            blind)
        //
        // Single ConstantLength<4> Poseidon hash (2 permutations at rate=2).
        // blind is the share commitment blinding factor — the secret that
        // makes the nullifier non-derivable from public on-chain data.
        // Unlike ciphertext coordinates (c1_x, c2_x), the blind is never
        // posted on-chain, so an observer cannot enumerate vote commitments
        // to link nullifiers to their source.
        // Round, proposal, decision, and shares_hash binding is transitive
        // through vote_commitment; share_comms bind one hop earlier through
        // shares_hash. A wrong public `vote_decision` or a wrong private
        // share-commitment set changes the condition-2 commitment and is
        // rejected by the Merkle path binding in condition 1.
        // ---------------------------------------------------------------
        {
            // "share spend" domain tag — constant-constrained so the
            // value is baked into the verification key.
            let domain_tag = config.assign_constant(
                &mut layouter,
                "cond5: DOMAIN_SHARE_SPEND constant",
                domain_tag_share_spend(),
            )?;

            let share_nullifier = PoseidonHash::<
                pallas::Base,
                _,
                poseidon::P128Pow5T3,
                ConstantLength<4>,
                3,
                2,
            >::init(
                config.poseidon_chip(),
                layouter.namespace(|| "cond5: share nullifier Poseidon init"),
            )?
            .hash(
                layouter.namespace(|| "cond5: Poseidon(tag, vc, idx, blind)"),
                [
                    domain_tag,
                    vote_commitment_cond5,
                    share_index_cond5,
                    primary_blind_cond5,
                ],
            )?;

            layouter.constrain_instance(
                share_nullifier.cell(),
                config.primary,
                SHARE_NULLIFIER_PUBLIC_OFFSET,
            )?;
        }

        Ok(())
    }
}

// ================================================================
// Instance (public inputs)
// ================================================================

/// Public inputs to the Share Reveal circuit (37 = 4M + 5 field elements).
///
/// The voting client (prover) chooses these values when assembling the
/// proof; the verifier accepts them as the binding the proof must
/// satisfy and checks the proof without seeing any private witnesses.
/// The relationship is asymmetric: a malicious-custody client can
/// choose any public-input vector it likes, so the verifier must source
/// the *correct* values from authenticated chain state (see
/// [`crate::share_reveal::prove::verify_share_reveal_proof`] for which
/// fields require caller authentication versus which are proof-attested
/// outputs).
///
/// The struct field order equals the Halo2 public input order; the
/// ciphertext vector expands to 64 elements in the canonical
/// [`WeightedShareCiphertexts::to_preimage`] order.
#[derive(Clone, Debug)]
pub struct Instance {
    /// Poseidon nullifier for this share (prevents double-counting).
    pub share_nullifier: pallas::Base,
    /// Caller-supplied ciphertext coordinates of all 8 decision buckets of
    /// the revealed share, bound through condition 4's weighted selected
    /// commitment. Both coordinates of every point are bound, preventing
    /// sign-malleability; the vector is not directly recovered from
    /// vote-proof public inputs.
    pub ciphertexts: WeightedShareCiphertexts,
    /// Which proposal this vote is for.
    pub proposal_id: pallas::Base,
    /// Root of the vote commitment tree at anchor height.
    pub vote_comm_tree_root: pallas::Base,
    /// The voting round identifier.
    pub voting_round_id: pallas::Base,
    /// The active decision bucket count `D` for the proposal.
    pub decision_bucket_count: pallas::Base,
}

impl Instance {
    /// Number of public inputs serialized by [`Self::to_halo2_instance`].
    pub const NUM_PUBLIC_INPUTS: usize = 5 + 4 * MAX_DECISION_BUCKETS;

    /// Constructs an [`Instance`] from its constituent parts.
    ///
    /// Callers should authenticate `proposal_id`, `vote_comm_tree_root`,
    /// `voting_round_id`, and `decision_bucket_count` out-of-band before
    /// passing them here — see
    /// [`crate::share_reveal::prove::verify_share_reveal_proof`] for the
    /// trust contract. The ciphertext vector is caller-supplied reveal data
    /// bound through the weighted selected commitment
    /// (`crate::bridge::selected_share_commitment`) and the transitive
    /// `share_comm -> shares_hash -> vote_commitment` chain. The
    /// `share_nullifier` is a proof-attested output.
    pub fn from_parts(
        share_nullifier: pallas::Base,
        ciphertexts: WeightedShareCiphertexts,
        proposal_id: pallas::Base,
        vote_comm_tree_root: pallas::Base,
        voting_round_id: pallas::Base,
        decision_bucket_count: pallas::Base,
    ) -> Self {
        Instance {
            share_nullifier,
            ciphertexts,
            proposal_id,
            vote_comm_tree_root,
            voting_round_id,
            decision_bucket_count,
        }
    }

    /// Serializes public inputs for halo2 proof creation/verification.
    ///
    /// The order must match the instance column offsets defined at the
    /// top of this file.
    pub fn to_halo2_instance(&self) -> Vec<vesta::Scalar> {
        let mut inputs = Vec::with_capacity(Self::NUM_PUBLIC_INPUTS);
        inputs.push(self.share_nullifier);
        inputs.extend(self.ciphertexts.to_preimage());
        inputs.push(self.proposal_id);
        inputs.push(self.vote_comm_tree_root);
        inputs.push(self.voting_round_id);
        inputs.push(self.decision_bucket_count);
        inputs
    }
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::PrimeField;
    use voting_crypto_deps::halo2_proofs::dev::MockProver;
    use voting_crypto_deps::pasta_curves::pallas;

    use crate::bridge::{selected_share_commitment, CiphertextCoordinates};
    use crate::gadgets::vote_commitment::vote_commitment_hash_v2;
    use crate::protocol_hash::poseidon_hash_2;
    use crate::shares_hash::shares_hash_from_comms;

    #[test]
    fn instance_to_halo2_instance_uses_public_input_offsets() {
        let share_nullifier = pallas::Base::from(10u64);
        let ciphertexts = test_ciphertexts(1_000);
        let proposal_id = pallas::Base::from(15u64);
        let vote_comm_tree_root = pallas::Base::from(17u64);
        let voting_round_id = pallas::Base::from(18u64);
        let decision_bucket_count = pallas::Base::from(4u64);

        let instance = Instance {
            share_nullifier,
            ciphertexts,
            proposal_id,
            vote_comm_tree_root,
            voting_round_id,
            decision_bucket_count,
        };

        let public_inputs = instance.to_halo2_instance();

        assert_eq!(public_inputs.len(), Instance::NUM_PUBLIC_INPUTS);
        assert_eq!(Instance::NUM_PUBLIC_INPUTS, 37);
        assert_eq!(
            public_inputs[SHARE_NULLIFIER_PUBLIC_OFFSET],
            share_nullifier
        );
        let preimage = ciphertexts.to_preimage();
        for (slot, value) in preimage.iter().enumerate() {
            assert_eq!(
                public_inputs[ENC_SHARE_COORDS_PUBLIC_OFFSET + slot],
                *value,
                "coordinate slot {slot} must match the canonical preimage order"
            );
        }
        // Pin the per-bucket layout explicitly for bucket 2.
        assert_eq!(
            public_inputs[ENC_SHARE_COORDS_PUBLIC_OFFSET + 8],
            instance.ciphertexts.0[2].c1_x
        );
        assert_eq!(
            public_inputs[ENC_SHARE_COORDS_PUBLIC_OFFSET + 9],
            instance.ciphertexts.0[2].c2_x
        );
        assert_eq!(
            public_inputs[ENC_SHARE_COORDS_PUBLIC_OFFSET + 10],
            instance.ciphertexts.0[2].c1_y
        );
        assert_eq!(
            public_inputs[ENC_SHARE_COORDS_PUBLIC_OFFSET + 11],
            instance.ciphertexts.0[2].c2_y
        );
        assert_eq!(public_inputs[PROPOSAL_ID_PUBLIC_OFFSET], proposal_id);
        assert_eq!(
            public_inputs[VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET],
            vote_comm_tree_root
        );
        assert_eq!(
            public_inputs[VOTING_ROUND_ID_PUBLIC_OFFSET],
            voting_round_id
        );
        assert_eq!(
            public_inputs[DECISION_BUCKET_COUNT_PUBLIC_OFFSET],
            decision_bucket_count
        );
    }

    /// Deterministic test ciphertext vector for one share.
    fn test_ciphertexts(seed: u64) -> WeightedShareCiphertexts {
        WeightedShareCiphertexts(core::array::from_fn(|bucket| {
            let base = seed + (4 * bucket) as u64;
            CiphertextCoordinates {
                c1_x: pallas::Base::from(base),
                c2_x: pallas::Base::from(base + 1),
                c1_y: pallas::Base::from(base + 2),
                c2_y: pallas::Base::from(base + 3),
            }
        }))
    }

    /// Returns `(ciphertexts, share_blinds, selected_commitments)` for a
    /// deterministic 16-share weighted witness set.
    fn weighted_test_shares(
        seed: u64,
    ) -> (
        [WeightedShareCiphertexts; 16],
        [pallas::Base; 16],
        [pallas::Base; 16],
    ) {
        let ciphertexts: [WeightedShareCiphertexts; 16] =
            core::array::from_fn(|i| test_ciphertexts(seed + 1_000 * i as u64));
        let share_blinds: [pallas::Base; 16] =
            core::array::from_fn(|i| pallas::Base::from(1001u64 + i as u64));
        let comms: [pallas::Base; 16] =
            core::array::from_fn(|i| selected_share_commitment(share_blinds[i], &ciphertexts[i]));
        (ciphertexts, share_blinds, comms)
    }

    fn make_test_data(share_idx: u32) -> (Circuit, Instance) {
        let (circuit, instance, _) = make_test_ballot(share_idx, 10_000);
        (circuit, instance)
    }

    fn make_test_ballot(share_idx: u32, ciphertext_seed: u64) -> (Circuit, Instance, pallas::Base) {
        let proposal_id = pallas::Base::from(3u64);
        let decision_bucket_count = pallas::Base::from(4u64);
        let voting_round_id = pallas::Base::from(999u64);

        let (ciphertexts, share_blinds, share_comms) = weighted_test_shares(ciphertext_seed);
        let shares_hash_val = shares_hash_from_comms(share_comms);

        let vote_commitment = vote_commitment_hash_v2(
            voting_round_id,
            shares_hash_val,
            proposal_id,
            decision_bucket_count,
        );

        let (auth_path, position, vote_comm_tree_root) =
            build_single_leaf_merkle_path(vote_commitment, 0);

        let share_index_fp = pallas::Base::from(share_idx as u64);
        let share_nullifier = share_nullifier_hash(
            vote_commitment,
            share_index_fp,
            share_blinds[share_idx as usize],
        );

        let circuit = Circuit {
            vote_comm_tree_path: Value::known(auth_path),
            vote_comm_tree_position: Value::known(position),
            share_comms: share_comms.map(Value::known),
            primary_blind: Value::known(share_blinds[share_idx as usize]),
            share_index: Value::known(share_index_fp),
            vote_commitment: Value::known(vote_commitment),
        };

        let instance = Instance::from_parts(
            share_nullifier,
            ciphertexts[share_idx as usize],
            proposal_id,
            vote_comm_tree_root,
            voting_round_id,
            decision_bucket_count,
        );

        (circuit, instance, vote_commitment)
    }

    fn build_single_leaf_merkle_path(
        leaf: pallas::Base,
        position: u32,
    ) -> ([pallas::Base; VOTE_COMM_TREE_DEPTH], u32, pallas::Base) {
        let mut empty_roots = [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH];
        empty_roots[0] = poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero());
        for i in 1..VOTE_COMM_TREE_DEPTH {
            empty_roots[i] = poseidon_hash_2(empty_roots[i - 1], empty_roots[i - 1]);
        }

        let auth_path = empty_roots;
        let mut current = leaf;
        for (level, sibling) in auth_path.iter().enumerate() {
            let (left, right) = if (position >> level) & 1 == 0 {
                (current, *sibling)
            } else {
                (*sibling, current)
            };
            current = poseidon_hash_2(left, right);
        }
        (auth_path, position, current)
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_valid() {
        let (circuit, instance) = make_test_data(0);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn merkle_schedule_accepts_mixed_nonzero_position() {
        // Exercise both left- and right-child branches across all 24 scheduled
        // levels, including handoffs between the two Poseidon configurations.
        const POSITION: u32 = 0xA5_5A_C3;

        let (mut circuit, mut instance, vote_commitment) = make_test_ballot(0, 10_000);
        let (path, position, root) = build_single_leaf_merkle_path(vote_commitment, POSITION);
        circuit.vote_comm_tree_path = Value::known(path);
        circuit.vote_comm_tree_position = Value::known(position);
        instance.vote_comm_tree_root = root;

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_valid_index_1() {
        let (circuit, instance) = make_test_data(1);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_valid_index_2() {
        let (circuit, instance) = make_test_data(2);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_valid_index_3() {
        let (circuit, instance) = make_test_data(3);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_valid_index_15() {
        let (circuit, instance) = make_test_data(15);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_wrong_merkle_root() {
        let (circuit, mut instance) = make_test_data(0);
        instance.vote_comm_tree_root = pallas::Base::from(12345u64);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    #[ignore = "exhaustive Merkle lane-schedule mutation diagnostic"]
    fn merkle_schedule_rejects_each_sibling_and_position_bit_mutation() {
        for level in 0..VOTE_COMM_TREE_DEPTH {
            let (mut circuit, instance, vote_commitment) = make_test_ballot(0, 10_000);
            let (mut path, _, _) = build_single_leaf_merkle_path(vote_commitment, 0);
            path[level] += pallas::Base::one();
            circuit.vote_comm_tree_path = Value::known(path);
            let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
            assert!(
                prover.verify().is_err(),
                "tampered sibling at level {level} must fail"
            );

            let (mut circuit, instance, _) = make_test_ballot(0, 10_000);
            circuit.vote_comm_tree_position = Value::known(1u32 << level);
            let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
            assert!(
                prover.verify().is_err(),
                "tampered position bit at level {level} must fail"
            );
        }
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_wrong_nullifier() {
        let (circuit, mut instance) = make_test_data(0);
        instance.share_nullifier = pallas::Base::from(99999u64);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_wrong_share_index() {
        let (circuit, instance) = make_test_data(0);
        // Publish a different share's ciphertext vector: the recomputed
        // selected commitment no longer matches the muxed share_comm[0].
        let (ciphertexts, _, _) = weighted_test_shares(10_000);
        let bad_instance = Instance::from_parts(
            instance.share_nullifier,
            ciphertexts[1],
            instance.proposal_id,
            instance.vote_comm_tree_root,
            instance.voting_round_id,
            instance.decision_bucket_count,
        );
        let prover = MockProver::run(K, &circuit, vec![bad_instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_wrong_bucket_count() {
        let (circuit, mut instance) = make_test_data(0);
        instance.decision_bucket_count = pallas::Base::from(5u64);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_wrong_voting_round_id() {
        let (circuit, mut instance) = make_test_data(0);
        instance.voting_round_id = pallas::Base::from(12345u64);
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_cannot_replay_across_vote_commitments() {
        let share_idx = 0;
        let (circuit_a, instance_a, vote_commitment_a) = make_test_ballot(share_idx, 10_000);
        let (_circuit_b, instance_b, vote_commitment_b) = make_test_ballot(share_idx, 50_000);

        assert_eq!(instance_a.voting_round_id, instance_b.voting_round_id);
        assert_eq!(instance_a.proposal_id, instance_b.proposal_id);
        assert_eq!(
            instance_a.decision_bucket_count,
            instance_b.decision_bucket_count
        );
        assert_ne!(vote_commitment_a, vote_commitment_b);
        assert_ne!(
            instance_a.vote_comm_tree_root,
            instance_b.vote_comm_tree_root
        );

        let prover_a =
            MockProver::run(K, &circuit_a, vec![instance_a.to_halo2_instance()]).unwrap();
        assert_eq!(prover_a.verify(), Ok(()));

        // Reuse ballot A's reveal witnesses, but authenticate them against
        // ballot B's distinct vote commitment tree root.
        let mut replay_instance = instance_a.clone();
        replay_instance.vote_comm_tree_root = instance_b.vote_comm_tree_root;
        let replay_prover =
            MockProver::run(K, &circuit_a, vec![replay_instance.to_halo2_instance()]).unwrap();
        assert!(replay_prover.verify().is_err());
    }

    /// Proves that flipping c1_y to -c1_y (sign malleability) is detected.
    /// The share reveal circuit binds to the full curve point via share_commitment(blind, c1_x, c2_x, c1_y, c2_y).
    /// Negating c1_y changes the commitment, so the proof must fail.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_sign_flip_detected() {
        let (circuit, mut instance) = make_test_data(0);
        instance.ciphertexts.0[3].c1_y = -instance.ciphertexts.0[3].c1_y;
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// Any single altered coordinate among the 64 must be rejected: the
    /// weighted selected commitment binds every slot.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_each_altered_coordinate_fails() {
        // Sampling all 32 slots is slow; exercise one coordinate of
        // each kind in the first, a middle, and the last bucket.
        for bucket in [0usize, 3, 7] {
            for kind in 0..4usize {
                let (circuit, mut instance) = make_test_data(0);
                let coords = &mut instance.ciphertexts.0[bucket];
                match kind {
                    0 => coords.c1_x += pallas::Base::one(),
                    1 => coords.c2_x += pallas::Base::one(),
                    2 => coords.c1_y += pallas::Base::one(),
                    _ => coords.c2_y += pallas::Base::one(),
                }
                let prover =
                    MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
                assert!(
                    prover.verify().is_err(),
                    "altered coordinate kind {kind} in bucket {bucket} must be rejected"
                );
            }
        }
    }

    /// A wrong blind must be rejected by the selected-commitment equality.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_wrong_blind_fails() {
        let (mut circuit, instance) = make_test_data(0);
        circuit.primary_blind = Value::known(pallas::Base::from(424242u64));
        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        assert!(prover.verify().is_err());
    }

    /// Tampers with share_comms[5] (a share other than the primary share at index 0).
    /// The share_comms are private witnesses but transitively bound to the public
    /// vote_comm_tree_root via:
    ///   share_comms → shares_hash (condition 3)
    ///   shares_hash → vote_commitment (condition 2)
    ///   vote_commitment → Merkle root (condition 1)
    /// Changing any share_comm alters shares_hash → vote_commitment, so the Merkle
    /// root computed in-circuit no longer matches the public instance root.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_share_reveal_tampered_share_comms_fails() {
        let (mut circuit, instance) = make_test_data(0);

        // Replace share_comms[5] (index ≠ primary share index 0) with a wrong value.
        // Any single-field substitution propagates through shares_hash → vote_commitment
        // → Merkle root, invalidating condition 1.
        circuit.share_comms[5] = Value::known(pallas::Base::from(99999u64));

        let prover = MockProver::run(K, &circuit, vec![instance.to_halo2_instance()]).unwrap();
        // Must fail: tampered share_comm → wrong shares_hash → wrong vote_commitment
        // → Merkle root computed in-circuit ≠ instance.vote_comm_tree_root.
        assert!(prover.verify().is_err());
    }

    #[test]
    fn share_nullifier_tracks_shares_hash_through_vote_commitment() {
        let voting_round_id = pallas::Base::from(42u64);
        let proposal_id = pallas::Base::from(7u64);
        let decision_bucket_count = pallas::Base::from(4u64);
        let shares_hash_a = pallas::Base::from(100u64);
        let shares_hash_b = pallas::Base::from(101u64);
        let share_index = pallas::Base::from(3u64);
        let blind = pallas::Base::from(200u64);

        let vote_commitment_a = vote_commitment_hash_v2(
            voting_round_id,
            shares_hash_a,
            proposal_id,
            decision_bucket_count,
        );
        let vote_commitment_b = vote_commitment_hash_v2(
            voting_round_id,
            shares_hash_b,
            proposal_id,
            decision_bucket_count,
        );
        assert_ne!(vote_commitment_a, vote_commitment_b);

        let share_nullifier_a = share_nullifier_hash(vote_commitment_a, share_index, blind);
        let share_nullifier_b = share_nullifier_hash(vote_commitment_b, share_index, blind);
        assert_ne!(share_nullifier_a, share_nullifier_b);

        assert_eq!(
            vote_commitment_a.to_repr(),
            [
                110, 183, 25, 111, 169, 124, 27, 220, 124, 219, 126, 128, 52, 98, 61, 174, 212,
                123, 35, 188, 205, 178, 219, 101, 55, 91, 155, 198, 193, 197, 131, 19,
            ]
        );
        assert_eq!(
            vote_commitment_b.to_repr(),
            [
                34, 157, 120, 250, 247, 73, 204, 166, 78, 255, 70, 184, 244, 79, 65, 172, 13, 5,
                185, 127, 37, 64, 137, 95, 53, 84, 1, 255, 114, 36, 184, 44,
            ]
        );
        assert_eq!(
            share_nullifier_a.to_repr(),
            [
                45, 134, 202, 17, 253, 95, 59, 251, 251, 60, 47, 231, 5, 77, 4, 226, 126, 54, 246,
                105, 235, 0, 148, 142, 192, 168, 9, 171, 96, 9, 66, 37,
            ]
        );
        assert_eq!(
            share_nullifier_b.to_repr(),
            [
                101, 72, 74, 223, 211, 229, 95, 226, 59, 130, 22, 109, 174, 40, 84, 137, 70, 154,
                119, 255, 250, 216, 209, 153, 134, 239, 111, 220, 217, 102, 188, 36,
            ]
        );
    }

    #[test]
    fn share_nullifier_hash_frozen_vector() {
        assert_eq!(
            share_nullifier_hash(
                pallas::Base::from(42u64),
                pallas::Base::from(3u64),
                pallas::Base::from(200u64),
            ),
            pallas::Base::from_repr([
                103, 140, 231, 81, 182, 191, 8, 141, 126, 173, 35, 129, 94, 244, 230, 146, 27, 161,
                255, 223, 211, 230, 26, 212, 86, 62, 15, 167, 99, 237, 233, 63,
            ])
            .expect("frozen vector must be canonical")
        );
    }

    #[test]
    fn test_share_reveal_domain_tag_matches_server() {
        assert_eq!(domain_tag_share_spend(), crate::domain_tags::share_spend());
    }

    /// Measures actual rows used by the share-reveal circuit via `CircuitCost::measure`.
    ///
    /// `CircuitCost` runs the floor planner against the circuit and tracks the
    /// highest row offset assigned in any column, giving the real "rows consumed"
    /// number rather than the theoretical 2^K capacity.
    ///
    /// Run with:
    ///   cargo test row_budget -- --nocapture --ignored
    #[test]
    #[ignore = "long-running row-budget diagnostic; run with `cargo test row_budget -- --ignored --nocapture`"]
    fn row_budget() {
        use std::println;
        use voting_crypto_deps::halo2_proofs::dev::CircuitCost;
        use voting_crypto_deps::pasta_curves::vesta;

        let (circuit, _) = make_test_data(0);

        let cost = CircuitCost::<vesta::Point, _>::measure(K, &circuit);
        let debug = format!("{cost:?}");

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

        println!("=== share-reveal circuit row budget (K={K}) ===");
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

        // Witness-independence check: Circuit::default() (all unknowns)
        // must produce exactly the same layout as the filled circuit.
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

        println!("  VOTE_COMM_TREE_DEPTH (circuit constant): {VOTE_COMM_TREE_DEPTH}");

        // Minimum-K probe: find the smallest K at which MockProver passes.
        for probe_k in 9u32..=K {
            let (c, inst) = make_test_data(0);
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
