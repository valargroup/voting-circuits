//! The Proxy Delegation circuit implementation (ZKP #4).
//!
//! The conditions are listed in the [module docs](super), and `synthesize`
//! marks where each one is enforced. The column layout is ZKP #2's without the
//! authority-decrement chip. Conditions 7 and 14 range-check on the second El
//! Gamal track's lookup column, and condition 14 hashes on its own Poseidon
//! configuration over four of that track's columns, which keeps both off the
//! saturated primary track.

use std::vec::Vec;

use voting_crypto_deps::composition::{
    assign_free_advice, AddChip, AddConfig, AddInstruction, CommitIvkChip, CommitIvkConfig,
    OrchardCommitDomains, OrchardFixedBases, OrchardHashDomains,
};
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
use voting_crypto_deps::pasta_curves::{pallas, vesta};

use crate::{
    domain_tags,
    gadgets::{
        address_ownership::{prove_address_ownership, spend_auth_g_mul},
        elgamal::{prove_elgamal_encryptions, EaPkInstanceLoc, SpendAuthGFixedBase30Config},
        nonzero::NonZeroConfig,
        poseidon_merkle::{synthesize_poseidon_merkle_path, MerkleSwapGate},
        spend_authority::prove_spend_authority,
        van_integrity::{self, DOMAIN_VAN},
        vote_commitment::{self, DOMAIN_VC},
    },
    params::{
        MAX_PROPOSAL_AUTHORITY, RANGE_CHECK_WORD_BITS, SHARE_VALUE_RANGE_WORDS,
        VOTE_COMM_TREE_DEPTH,
    },
    shares_hash::compute_shares_hash_in_circuit,
};

// ================================================================
// Constants
// ================================================================

/// Circuit size (2^K rows).
///
/// K=11 (2,048 rows). `CircuitCost` reports a 2,015-row high-water mark, set
/// by the primary track as in ZKP #2. Re-measure after any circuit change:
///   `cargo test --release proxy_delegation::circuit::tests::row_budget -- --ignored --nocapture`
pub const K: u32 = 11;

/// First share index assigned to the second El Gamal track.
const ELGAMAL_TRACK_SPLIT: usize = 8;

/// Number of condition-10 share commitments hashed on the primary track.
const PRIMARY_SHARES_HASH_COUNT: usize = 2;

/// Proposal id carried by every delegation commitment (condition 13).
///
/// ZKP #2 rejects proposal 0, so only this circuit can emit a commitment for
/// it, and ZKP #3 reveals such a share as `(0, d)` into delegate `d`'s pool.
/// It is a constant cell: changing it changes the verifying key.
pub(crate) const POOL_PROPOSAL_ID: u64 = 0;

/// Bit width of the DC slot index (condition 14), so one registration's VAN
/// chain can emit at most `2^DC_SLOT_BITS = 8` delegation commitments.
///
/// The cap holds because every slot nullifier is keyed by `(nk, round, rand)`,
/// which no successor VAN changes: condition 3 fixes `nk` for the address and
/// condition 6 keeps the address and `rand`. Any other circuit that rewrites a
/// full-authority VAN must keep `rand` too, or the cap resets. Changing this
/// value changes the verifying key.
pub(crate) const DC_SLOT_BITS: usize = 3;

// ================================================================
// Public input offsets (11 field elements)
// ================================================================

/// Offset of the VAN nullifier (condition 5).
const VAN_NULLIFIER_PUBLIC_OFFSET: usize = 0;
/// Offset of the `r_vpk` x-coordinate (condition 4).
const R_VPK_X_PUBLIC_OFFSET: usize = 1;
/// Offset of the `r_vpk` y-coordinate (condition 4).
const R_VPK_Y_PUBLIC_OFFSET: usize = 2;
/// Offset of the successor VAN (condition 6).
const VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET: usize = 3;
/// Offset of the delegation commitment (condition 13).
const DELEGATION_COMMITMENT_PUBLIC_OFFSET: usize = 4;
/// Offset of the vote commitment tree root (condition 1).
const VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET: usize = 5;
/// Offset of the tree anchor height, which no condition constrains (see
/// [`Instance::vote_comm_tree_anchor_height`]).
const VOTE_COMM_TREE_ANCHOR_HEIGHT_PUBLIC_OFFSET: usize = 6;
/// Offset of the voting round id.
const VOTING_ROUND_ID_PUBLIC_OFFSET: usize = 7;
/// Offset of the election authority public key x-coordinate (condition 11).
const EA_PK_X_PUBLIC_OFFSET: usize = 8;
/// Offset of the election authority public key y-coordinate (condition 11).
const EA_PK_Y_PUBLIC_OFFSET: usize = 9;
/// Offset of the DC slot nullifier (condition 14).
const DC_SLOT_NULLIFIER_PUBLIC_OFFSET: usize = 10;

// ================================================================
// Out-of-circuit helpers
// ================================================================

/// Out-of-circuit DC slot nullifier (condition 14):
///
/// ```text
/// dc_slot_nullifier = Poseidon(vsk_nk, TAG + slot, voting_round_id, van_comm_rand)
/// ```
///
/// where `TAG` is the `"proxy delegation slot"` domain tag. It computes the
/// formula for any `slot`; the circuit only accepts `slot < 2^DC_SLOT_BITS`.
pub fn dc_slot_nullifier_hash(
    vsk_nk: pallas::Base,
    voting_round_id: pallas::Base,
    van_comm_rand: pallas::Base,
    slot: u8,
) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<4>, 3, 2>::init().hash([
        vsk_nk,
        domain_tags::proxy_delegation_slot() + pallas::Base::from(u64::from(slot)),
        voting_round_id,
        van_comm_rand,
    ])
}

/// Out-of-circuit delegation commitment (condition 13):
///
/// ```text
/// DC = Poseidon(DOMAIN_VC, voting_round_id, shares_hash, POOL_PROPOSAL_ID, delegate_index)
/// ```
///
/// This is the vote commitment preimage with proposal 0 and the delegate index
/// as the decision, so ZKP #3 reveals DC shares without changes.
pub fn delegation_commitment_hash(
    voting_round_id: pallas::Base,
    shares_hash: pallas::Base,
    delegate_index: u32,
) -> pallas::Base {
    vote_commitment::vote_commitment_hash(
        voting_round_id,
        shares_hash,
        pallas::Base::from(POOL_PROPOSAL_ID),
        pallas::Base::from(u64::from(delegate_index)),
    )
}

// ================================================================
// Config
// ================================================================

/// Configuration for the Proxy Delegation circuit.
#[derive(Clone, Debug)]
pub struct Config {
    /// Public input column (11 field elements).
    primary: Column<InstanceColumn>,
    /// Primary advice columns, laid out as in ZKP #2.
    advices: [Column<Advice>; 10],
    /// First El Gamal track (condition 11, shares 0 to 7).
    elgamal_advices: [Column<Advice>; 10],
    /// Second El Gamal track (condition 11, shares 8 to 15). Conditions 7, 12
    /// and 14 also use it.
    elgamal_advices_b: [Column<Advice>; 10],
    /// Poseidon on the primary track (conditions 1, 2, 5, 6, 13 and part of 10).
    poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    /// Poseidon on the dedicated condition-10 track.
    hash_poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    /// Poseidon for the condition-14 slot nullifier, on `elgamal_advices_b[1..5]`.
    slot_poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    /// `a + b = c` gate (condition 8 sums and condition 14's `TAG + slot`).
    add_config: AddConfig,
    /// ECC chip for conditions 3 and 4.
    ecc_config: EccConfig<OrchardFixedBases>,
    /// ECC chip for the first El Gamal track.
    elgamal_ecc_config: EccConfig<OrchardFixedBases>,
    /// ECC chip for the second El Gamal track.
    elgamal_ecc_config_b: EccConfig<OrchardFixedBases>,
    /// Sinsemilla for CommitIvk (condition 3); also loads the 10-bit lookup table.
    sinsemilla_config:
        SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    /// CommitIvk canonicity checks (condition 3).
    commit_ivk_config: CommitIvkConfig,
    /// 10-bit lookup range check on `advices[9]` (condition 9).
    range_check: LookupRangeCheckConfig<pallas::Base, RANGE_CHECK_WORD_BITS>,
    /// 10-bit lookup range check on `elgamal_advices_b[9]` (conditions 7 and 14).
    range_check_b: LookupRangeCheckConfig<pallas::Base, RANGE_CHECK_WORD_BITS>,
    /// Merkle conditional swap gate (condition 1).
    merkle_swap: MerkleSwapGate,
    /// Inverse-witness non-zero gate on the first El Gamal track.
    nonzero: NonZeroConfig,
    /// Inverse-witness non-zero gate on the second El Gamal track (also condition 12).
    nonzero_b: NonZeroConfig,
    /// Unsigned 30-bit fixed-base multiplication for El Gamal on the first track.
    spend_auth_g_fixed_base_30: SpendAuthGFixedBase30Config,
    /// Unsigned 30-bit fixed-base multiplication for El Gamal on the second track.
    spend_auth_g_fixed_base_30_b: SpendAuthGFixedBase30Config,
}

impl Config {
    /// Constructs the primary-track Poseidon chip.
    fn poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.poseidon_config.clone())
    }

    /// Constructs the condition-10 Poseidon chip.
    fn hash_poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.hash_poseidon_config.clone())
    }

    /// Constructs the condition-14 Poseidon chip.
    fn slot_poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.slot_poseidon_config.clone())
    }

    /// Constructs the `a + b = c` chip.
    fn add_chip(&self) -> AddChip {
        AddChip::construct(self.add_config.clone())
    }

    /// Constructs the ECC chip for conditions 3 and 4.
    fn ecc_chip(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(self.ecc_config.clone(), CircuitVersion::AnchoredBase)
    }

    /// Constructs the ECC chip for the first El Gamal track.
    fn elgamal_ecc_chip(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(
            self.elgamal_ecc_config.clone(),
            CircuitVersion::AnchoredBase,
        )
    }

    /// Constructs the ECC chip for the second El Gamal track.
    fn elgamal_ecc_chip_b(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(
            self.elgamal_ecc_config_b.clone(),
            CircuitVersion::AnchoredBase,
        )
    }

    /// Constructs the Sinsemilla chip for CommitIvk (condition 3).
    fn sinsemilla_chip(
        &self,
    ) -> SinsemillaChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases> {
        SinsemillaChip::construct(self.sinsemilla_config.clone())
    }

    /// Constructs the CommitIvk chip (condition 3).
    fn commit_ivk_chip(&self) -> CommitIvkChip {
        CommitIvkChip::construct(self.commit_ivk_config.clone())
    }

    /// Assigns `value` to `advices[0]` as a fixed constant, which bakes it into
    /// the verifying key.
    fn constant(
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

/// Witnesses 16 values as free advice cells in `column`.
fn assign_16(
    layouter: &mut impl Layouter<pallas::Base>,
    column: Column<Advice>,
    name: &str,
    values: &[Value<pallas::Base>; 16],
) -> Result<[AssignedCell<pallas::Base, pallas::Base>; 16], plonk::Error> {
    let cells = values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            assign_free_advice(
                layouter.namespace(|| format!("witness {name}[{i}]")),
                column,
                *value,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(cells.try_into().expect("16 values give 16 cells"))
}

// ================================================================
// Circuit
// ================================================================

/// The Proxy Delegation circuit (ZKP #4).
///
/// Holds the private witnesses for the conditions in the
/// [module docs](super). `Circuit::default()` leaves every witness unknown
/// and is what key generation uses.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    /// Merkle authentication path of the old VAN (condition 1).
    pub(super) vote_comm_tree_path: Value<[pallas::Base; VOTE_COMM_TREE_DEPTH]>,
    /// Leaf position of the old VAN (condition 1).
    pub(super) vote_comm_tree_position: Value<u32>,
    /// Diversified base of the voting address (conditions 2, 3 and 6).
    pub(super) vpk_g_d: Value<pallas::Affine>,
    /// Transmission key of the voting address (conditions 2, 3 and 6).
    pub(super) vpk_pk_d: Value<pallas::Affine>,
    /// Old VAN weight `W` in ballots (conditions 2 and 8).
    pub(super) total_note_value: Value<pallas::Base>,
    /// Successor VAN weight `W_new = W - w` in ballots (conditions 6, 7 and 8).
    pub(super) remaining_note_value: Value<pallas::Base>,
    /// VAN blinding randomness, shared by the old and successor VANs
    /// (conditions 2, 6 and 14).
    pub(super) van_comm_rand: Value<pallas::Base>,
    /// Old VAN commitment (conditions 1, 2 and 5).
    pub(super) vote_authority_note_old: Value<pallas::Base>,
    /// Voting spending key (conditions 3 and 4).
    pub(super) vsk: Value<pallas::Scalar>,
    /// CommitIvk randomness (condition 3).
    pub(super) rivk_v: Value<pallas::Scalar>,
    /// Spend authorization randomizer (condition 4).
    pub(super) alpha_v: Value<pallas::Scalar>,
    /// Nullifier deriving key (conditions 3, 5 and 14).
    pub(super) vsk_nk: Value<pallas::Base>,
    /// The 16 delegated shares, summing to `w` (conditions 8, 9, 11 and 12).
    pub(super) shares: [Value<pallas::Base>; 16],
    /// x-coordinate of each share's El Gamal `C1` (conditions 10 and 11).
    pub(super) enc_share_c1_x: [Value<pallas::Base>; 16],
    /// x-coordinate of each share's El Gamal `C2` (conditions 10 and 11).
    pub(super) enc_share_c2_x: [Value<pallas::Base>; 16],
    /// y-coordinate of each share's El Gamal `C1` (conditions 10 and 11).
    pub(super) enc_share_c1_y: [Value<pallas::Base>; 16],
    /// y-coordinate of each share's El Gamal `C2` (conditions 10 and 11).
    pub(super) enc_share_c2_y: [Value<pallas::Base>; 16],
    /// Per-share commitment blinds (condition 10).
    pub(super) share_blinds: [Value<pallas::Base>; 16],
    /// Per-share El Gamal randomness (condition 11).
    pub(super) share_randomness: [Value<pallas::Base>; 16],
    /// Election authority public key (condition 11).
    pub(super) ea_pk: Value<pallas::Affine>,
    /// Delegate index `d`, hashed into the DC (condition 13) and constrained
    /// nowhere else.
    pub(super) delegate_index: Value<pallas::Base>,
    /// DC slot index in `[0, 2^DC_SLOT_BITS)` (condition 14).
    pub(super) dc_slot: Value<pallas::Base>,
}

impl plonk::Circuit<pallas::Base> for Circuit {
    type Config = Config;
    type FloorPlanner = floor_planner::V1;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        let advices: [Column<Advice>; 10] = core::array::from_fn(|_| meta.advice_column());
        for col in &advices {
            meta.enable_equality(*col);
        }
        let elgamal_advices: [Column<Advice>; 10] = core::array::from_fn(|_| meta.advice_column());
        for col in &elgamal_advices {
            meta.enable_equality(*col);
        }
        let elgamal_advices_b: [Column<Advice>; 10] =
            core::array::from_fn(|_| meta.advice_column());
        for col in &elgamal_advices_b {
            meta.enable_equality(*col);
        }
        let hash_advices: [Column<Advice>; 4] = core::array::from_fn(|_| meta.advice_column());

        let primary = meta.instance_column();
        meta.enable_equality(primary);

        // As in ZKP #2, the ECC Lagrange-coefficient columns also hold the
        // primary Poseidon round constants (2-7) and Sinsemilla's y_Q (0).
        let lagrange_coeffs: [Column<Fixed>; 8] = core::array::from_fn(|_| meta.fixed_column());
        let rc_a = lagrange_coeffs[2..5].try_into().unwrap();
        let rc_b = lagrange_coeffs[5..8].try_into().unwrap();
        let constants = meta.fixed_column();
        meta.enable_constant(constants);

        let add_config = AddChip::configure(meta, advices[7], advices[8], advices[6]);

        let table_idx = meta.lookup_table_column();
        let lookup = (
            table_idx,
            meta.lookup_table_column(),
            meta.lookup_table_column(),
        );
        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], table_idx);
        let elgamal_range_check =
            LookupRangeCheckConfig::configure(meta, elgamal_advices[9], table_idx);
        let elgamal_range_check_b =
            LookupRangeCheckConfig::configure(meta, elgamal_advices_b[9], table_idx);

        let ecc_config =
            EccChip::<OrchardFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);
        let elgamal_lagrange_coeffs: [Column<Fixed>; 8] =
            core::array::from_fn(|_| meta.fixed_column());
        let elgamal_ecc_config = EccChip::<OrchardFixedBases>::configure(
            meta,
            elgamal_advices,
            elgamal_lagrange_coeffs,
            elgamal_range_check,
        );
        let spend_auth_g_fixed_base_30 =
            SpendAuthGFixedBase30Config::configure(meta, elgamal_advices, elgamal_lagrange_coeffs);
        let elgamal_lagrange_coeffs_b: [Column<Fixed>; 8] =
            core::array::from_fn(|_| meta.fixed_column());
        let elgamal_ecc_config_b = EccChip::<OrchardFixedBases>::configure(
            meta,
            elgamal_advices_b,
            elgamal_lagrange_coeffs_b,
            elgamal_range_check_b,
        );
        let spend_auth_g_fixed_base_30_b = SpendAuthGFixedBase30Config::configure(
            meta,
            elgamal_advices_b,
            elgamal_lagrange_coeffs_b,
        );

        let sinsemilla_config = SinsemillaChip::configure(
            meta,
            advices[..5].try_into().unwrap(),
            advices[6],
            lagrange_coeffs[0],
            lookup,
            range_check,
            false,
        );
        let commit_ivk_config = CommitIvkChip::configure(meta, advices);

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

        let slot_round_constants: [Column<Fixed>; 6] =
            core::array::from_fn(|_| meta.fixed_column());
        let slot_poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
            meta,
            elgamal_advices_b[1..4].try_into().unwrap(),
            elgamal_advices_b[4],
            slot_round_constants[..3].try_into().unwrap(),
            slot_round_constants[3..].try_into().unwrap(),
        );
        let merkle_swap = MerkleSwapGate::configure(
            meta,
            [advices[0], advices[1], advices[2], advices[3], advices[4]],
        );
        let nonzero = NonZeroConfig::configure(meta, [elgamal_advices[0], elgamal_advices[1]]);
        let nonzero_b =
            NonZeroConfig::configure(meta, [elgamal_advices_b[0], elgamal_advices_b[1]]);

        Config {
            primary,
            advices,
            elgamal_advices,
            elgamal_advices_b,
            poseidon_config,
            hash_poseidon_config,
            slot_poseidon_config,
            add_config,
            ecc_config,
            elgamal_ecc_config,
            elgamal_ecc_config_b,
            sinsemilla_config,
            commit_ivk_config,
            range_check,
            range_check_b: elgamal_range_check_b,
            merkle_swap,
            nonzero,
            nonzero_b,
            spend_auth_g_fixed_base_30,
            spend_auth_g_fixed_base_30_b,
        }
    }

    // The region order below fixes the layout, and so the verifying key.
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), plonk::Error> {
        SinsemillaChip::load(config.sinsemilla_config.clone(), &mut layouter)?;
        let ecc_chip = config.ecc_chip();

        // ---------------------------------------------------------------
        // Shared witnesses.
        // ---------------------------------------------------------------
        let voting_round_id = layouter.assign_region(
            || "copy voting_round_id",
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

        let vpk_g_d_point = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "witness vpk_g_d"),
            self.vpk_g_d,
        )?;
        let vpk_g_d = vpk_g_d_point.extract_p().inner().clone();
        let vpk_pk_d_point = NonIdentityPoint::new(
            ecc_chip.clone(),
            layouter.namespace(|| "witness vpk_pk_d"),
            self.vpk_pk_d,
        )?;
        let vpk_pk_d = vpk_pk_d_point.extract_p().inner().clone();

        let total_note_value = assign_free_advice(
            layouter.namespace(|| "witness total_note_value"),
            config.advices[0],
            self.total_note_value,
        )?;
        let remaining_note_value = assign_free_advice(
            layouter.namespace(|| "witness remaining_note_value"),
            config.advices[0],
            self.remaining_note_value,
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
        let domain_van =
            config.constant(&mut layouter, "DOMAIN_VAN", pallas::Base::from(DOMAIN_VAN))?;
        // One constant cell serves as the authority of both VANs (conditions 2 and 6).
        let authority_max = config.constant(
            &mut layouter,
            "MAX_PROPOSAL_AUTHORITY",
            pallas::Base::from(MAX_PROPOSAL_AUTHORITY),
        )?;
        let vsk_nk = assign_free_advice(
            layouter.namespace(|| "witness vsk_nk"),
            config.advices[0],
            self.vsk_nk,
        )?;

        // ---------------------------------------------------------------
        // Condition 2: old VAN integrity, with full authority.
        // ---------------------------------------------------------------
        let derived_van = van_integrity::van_integrity_poseidon(
            &config.poseidon_config,
            &mut layouter,
            "Old VAN integrity",
            domain_van.clone(),
            vpk_g_d.clone(),
            vpk_pk_d.clone(),
            total_note_value.clone(),
            voting_round_id.clone(),
            authority_max.clone(),
            van_comm_rand.clone(),
        )?;
        layouter.assign_region(
            || "old VAN integrity check",
            |mut region| region.constrain_equal(derived_van.cell(), vote_authority_note_old.cell()),
        )?;

        // ---------------------------------------------------------------
        // Condition 3: diversified address integrity.
        // ---------------------------------------------------------------
        let vsk_scalar =
            ScalarFixed::new(ecc_chip.clone(), layouter.namespace(|| "vsk"), self.vsk)?;
        let vsk_ak_point = spend_auth_g_mul(
            ecc_chip.clone(),
            layouter.namespace(|| "[vsk] SpendAuthG"),
            "condition 3: [vsk] SpendAuthG",
            vsk_scalar,
        )?;
        let ak = vsk_ak_point.extract_p().inner().clone();
        let rivk_v_scalar = ScalarFixed::new(
            ecc_chip.clone(),
            layouter.namespace(|| "rivk_v"),
            self.rivk_v,
        )?;
        prove_address_ownership(
            config.sinsemilla_chip(),
            ecc_chip.clone(),
            config.commit_ivk_chip(),
            layouter.namespace(|| "condition 3"),
            "condition 3",
            ak,
            vsk_nk.clone(),
            rivk_v_scalar,
            &vpk_g_d_point,
            &vpk_pk_d_point,
        )?;

        // ---------------------------------------------------------------
        // Condition 4: spend authority.
        // ---------------------------------------------------------------
        prove_spend_authority(
            ecc_chip.clone(),
            layouter.namespace(|| "condition 4"),
            self.alpha_v,
            &vsk_ak_point,
            config.primary,
            R_VPK_X_PUBLIC_OFFSET,
            R_VPK_Y_PUBLIC_OFFSET,
        )?;

        // ---------------------------------------------------------------
        // Condition 1: old VAN membership.
        // ---------------------------------------------------------------
        let root = synthesize_poseidon_merkle_path::<VOTE_COMM_TREE_DEPTH>(
            &config.merkle_swap,
            &config.poseidon_config,
            &mut layouter,
            config.advices[0],
            vote_authority_note_old.clone(),
            self.vote_comm_tree_position,
            self.vote_comm_tree_path,
            "condition 1: merkle",
        )?;
        layouter.constrain_instance(
            root.cell(),
            config.primary,
            VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET,
        )?;

        // ---------------------------------------------------------------
        // Condition 5: VAN nullifier, derived exactly as in ZKP #2.
        // ---------------------------------------------------------------
        let domain_van_nf = config.constant(
            &mut layouter,
            "vote authority spend",
            domain_tags::vote_authority_spend(),
        )?;
        let van_nullifier =
            PoseidonHash::<pallas::Base, _, poseidon::P128Pow5T3, ConstantLength<4>, 3, 2>::init(
                config.poseidon_chip(),
                layouter.namespace(|| "van_nullifier init"),
            )?
            .hash(
                layouter.namespace(|| "Poseidon(nk, tag, round, van_old)"),
                [
                    vsk_nk.clone(),
                    domain_van_nf,
                    voting_round_id.clone(),
                    vote_authority_note_old,
                ],
            )?;
        layouter.constrain_instance(
            van_nullifier.cell(),
            config.primary,
            VAN_NULLIFIER_PUBLIC_OFFSET,
        )?;

        // ---------------------------------------------------------------
        // Condition 14: DC slot range and slot nullifier.
        // ---------------------------------------------------------------
        let dc_slot = assign_free_advice(
            layouter.namespace(|| "witness dc_slot"),
            config.advices[0],
            self.dc_slot,
        )?;
        config.range_check_b.copy_short_check(
            layouter.namespace(|| "dc_slot < 2^DC_SLOT_BITS"),
            dc_slot.clone(),
            DC_SLOT_BITS,
        )?;
        let slot_tag = config.constant(
            &mut layouter,
            "proxy delegation slot",
            domain_tags::proxy_delegation_slot(),
        )?;
        let slot_tag =
            config
                .add_chip()
                .add(layouter.namespace(|| "TAG + dc_slot"), &slot_tag, &dc_slot)?;
        let dc_slot_nullifier =
            PoseidonHash::<pallas::Base, _, poseidon::P128Pow5T3, ConstantLength<4>, 3, 2>::init(
                config.slot_poseidon_chip(),
                layouter.namespace(|| "dc_slot_nullifier init"),
            )?
            .hash(
                layouter.namespace(|| "Poseidon(nk, TAG + dc_slot, round, rand)"),
                [
                    vsk_nk,
                    slot_tag,
                    voting_round_id.clone(),
                    van_comm_rand.clone(),
                ],
            )?;
        layouter.constrain_instance(
            dc_slot_nullifier.cell(),
            config.primary,
            DC_SLOT_NULLIFIER_PUBLIC_OFFSET,
        )?;

        // ---------------------------------------------------------------
        // Condition 6: successor VAN with the same address, round and rand,
        // full authority and weight W_new.
        // ---------------------------------------------------------------
        let vote_authority_note_new = van_integrity::van_integrity_poseidon(
            &config.poseidon_config,
            &mut layouter,
            "New VAN integrity",
            domain_van,
            vpk_g_d,
            vpk_pk_d,
            remaining_note_value.clone(),
            voting_round_id.clone(),
            authority_max,
            van_comm_rand,
        )?;
        layouter.constrain_instance(
            vote_authority_note_new.cell(),
            config.primary,
            VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET,
        )?;

        // ---------------------------------------------------------------
        // Condition 7: W_new in [0, 2^30). Without it, w > W would wrap W_new
        // around the field and inflate the successor VAN.
        // ---------------------------------------------------------------
        config.range_check_b.copy_check(
            layouter.namespace(|| "W_new < 2^30"),
            remaining_note_value.clone(),
            SHARE_VALUE_RANGE_WORDS,
            true,
        )?;

        // ---------------------------------------------------------------
        // Condition 8: sum(shares) + W_new = W.
        // ---------------------------------------------------------------
        let share_cells = assign_16(&mut layouter, config.advices[0], "share", &self.shares)?;
        let shares_sum = share_cells[1..].iter().enumerate().try_fold(
            share_cells[0].clone(),
            |acc, (i, share)| {
                config.add_chip().add(
                    layouter.namespace(|| format!("sum step {}", i + 1)),
                    &acc,
                    share,
                )
            },
        )?;
        let conserved = config.add_chip().add(
            layouter.namespace(|| "w + W_new"),
            &shares_sum,
            &remaining_note_value,
        )?;
        layouter.assign_region(
            || "w + W_new = W",
            |mut region| region.constrain_equal(conserved.cell(), total_note_value.cell()),
        )?;

        // ---------------------------------------------------------------
        // Condition 12: w != 0.
        // ---------------------------------------------------------------
        config.nonzero_b.constrain_nonzero(
            layouter.namespace(|| "w != 0"),
            "w != 0",
            &shares_sum,
        )?;

        // ---------------------------------------------------------------
        // Condition 9: each share in [0, 2^30).
        // ---------------------------------------------------------------
        for (i, cell) in share_cells.iter().enumerate() {
            config.range_check.copy_check(
                layouter.namespace(|| format!("share_{i} < 2^30")),
                cell.clone(),
                SHARE_VALUE_RANGE_WORDS,
                true,
            )?;
        }

        // ---------------------------------------------------------------
        // Condition 10: shares hash over the 16 blinded share commitments.
        // It is an internal wire, bound publicly only through the DC.
        // ---------------------------------------------------------------
        let blinds = assign_16(
            &mut layouter,
            config.advices[0],
            "share_blind",
            &self.share_blinds,
        )?;
        let enc_c1_x = assign_16(
            &mut layouter,
            config.advices[0],
            "enc_c1_x",
            &self.enc_share_c1_x,
        )?;
        let enc_c2_x = assign_16(
            &mut layouter,
            config.advices[0],
            "enc_c2_x",
            &self.enc_share_c2_x,
        )?;
        let enc_c1_y = assign_16(
            &mut layouter,
            config.advices[0],
            "enc_c1_y",
            &self.enc_share_c1_y,
        )?;
        let enc_c2_y = assign_16(
            &mut layouter,
            config.advices[0],
            "enc_c2_y",
            &self.enc_share_c2_y,
        )?;
        let shares_hash = compute_shares_hash_in_circuit(
            || config.poseidon_chip(),
            || config.hash_poseidon_chip(),
            PRIMARY_SHARES_HASH_COUNT,
            layouter.namespace(|| "condition 10: shares hash"),
            blinds,
            enc_c1_x.clone(),
            enc_c2_x.clone(),
            enc_c1_y.clone(),
            enc_c2_y.clone(),
        )?;

        // ---------------------------------------------------------------
        // Condition 11: El Gamal encryption of each share under ea_pk, split
        // across the two El Gamal tracks.
        // ---------------------------------------------------------------
        let r_cells: [_; 16] = (0..16usize)
            .map(|i| {
                let column = if i < ELGAMAL_TRACK_SPLIT {
                    config.elgamal_advices[0]
                } else {
                    config.elgamal_advices_b[0]
                };
                assign_free_advice(
                    layouter.namespace(|| format!("witness r[{i}]")),
                    column,
                    self.share_randomness[i],
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("16 values give 16 cells");
        let ea_pk_loc = || EaPkInstanceLoc {
            instance: config.primary,
            x_row: EA_PK_X_PUBLIC_OFFSET,
            y_row: EA_PK_Y_PUBLIC_OFFSET,
        };
        prove_elgamal_encryptions(
            config.elgamal_ecc_chip(),
            config.nonzero,
            &config.spend_auth_g_fixed_base_30,
            layouter.namespace(|| "condition 11a"),
            "condition 11a",
            0..ELGAMAL_TRACK_SPLIT,
            self.ea_pk,
            ea_pk_loc(),
            share_cells.clone(),
            r_cells.clone(),
            enc_c1_x.clone(),
            enc_c2_x.clone(),
            enc_c1_y.clone(),
            enc_c2_y.clone(),
        )?;
        prove_elgamal_encryptions(
            config.elgamal_ecc_chip_b(),
            config.nonzero_b,
            &config.spend_auth_g_fixed_base_30_b,
            layouter.namespace(|| "condition 11b"),
            "condition 11b",
            ELGAMAL_TRACK_SPLIT..16,
            self.ea_pk,
            ea_pk_loc(),
            share_cells,
            r_cells,
            enc_c1_x,
            enc_c2_x,
            enc_c1_y,
            enc_c2_y,
        )?;

        // ---------------------------------------------------------------
        // Condition 13: DC = Poseidon(DOMAIN_VC, round, shares_hash, 0, d).
        // ---------------------------------------------------------------
        let domain_vc =
            config.constant(&mut layouter, "DOMAIN_VC", pallas::Base::from(DOMAIN_VC))?;
        let pool_proposal_id = config.constant(
            &mut layouter,
            "POOL_PROPOSAL_ID",
            pallas::Base::from(POOL_PROPOSAL_ID),
        )?;
        let delegate_index = assign_free_advice(
            layouter.namespace(|| "witness delegate_index"),
            config.advices[0],
            self.delegate_index,
        )?;
        let delegation_commitment = vote_commitment::vote_commitment_poseidon(
            &config.poseidon_config,
            &mut layouter,
            "condition 13",
            domain_vc,
            voting_round_id,
            shares_hash,
            pool_proposal_id,
            delegate_index,
        )?;
        layouter.constrain_instance(
            delegation_commitment.cell(),
            config.primary,
            DELEGATION_COMMITMENT_PUBLIC_OFFSET,
        )?;

        Ok(())
    }
}

// ================================================================
// Instance (public inputs)
// ================================================================

/// Public inputs to the Proxy Delegation circuit (11 field elements).
///
/// [`Instance::to_halo2_instance`] is the only definition of their order, so
/// verifiers must pack instances through it. As with ZKP #2, the verifier must
/// take `vote_comm_tree_root`, `vote_comm_tree_anchor_height`,
/// `voting_round_id` and `ea_pk_*` from authenticated chain state; the other
/// fields are outputs that the proof attests.
#[derive(Clone, Debug)]
pub struct Instance {
    /// Nullifier of the spent VAN (condition 5), equal to the one ZKP #2
    /// would publish for it.
    pub van_nullifier: pallas::Base,
    /// x-coordinate of the randomized voting key `r_vpk` (condition 4).
    pub r_vpk_x: pallas::Base,
    /// y-coordinate of `r_vpk` (condition 4).
    pub r_vpk_y: pallas::Base,
    /// Successor VAN, with weight `W - w` and full authority (condition 6).
    pub vote_authority_note_new: pallas::Base,
    /// Delegation commitment (condition 13).
    pub delegation_commitment: pallas::Base,
    /// Vote commitment tree root that contains the old VAN (condition 1).
    pub vote_comm_tree_root: pallas::Base,
    /// Chain height of `vote_comm_tree_root`.
    ///
    /// The proof transcript binds it, but no condition constrains it: the
    /// verifier must check that the root is the chain's root at this height.
    pub vote_comm_tree_anchor_height: pallas::Base,
    /// Voting round, bound into conditions 2, 5, 6, 13 and 14.
    pub voting_round_id: pallas::Base,
    /// x-coordinate of the election authority public key (condition 11).
    pub ea_pk_x: pallas::Base,
    /// y-coordinate of the election authority public key (condition 11).
    pub ea_pk_y: pallas::Base,
    /// DC slot nullifier (condition 14).
    pub dc_slot_nullifier: pallas::Base,
}

impl Instance {
    /// Number of public inputs packed by [`Self::to_halo2_instance`].
    pub const NUM_PUBLIC_INPUTS: usize = 11;

    /// Constructs an [`Instance`]; the arguments are in public-input order.
    pub fn from_parts(
        van_nullifier: pallas::Base,
        r_vpk_x: pallas::Base,
        r_vpk_y: pallas::Base,
        vote_authority_note_new: pallas::Base,
        delegation_commitment: pallas::Base,
        vote_comm_tree_root: pallas::Base,
        vote_comm_tree_anchor_height: pallas::Base,
        voting_round_id: pallas::Base,
        ea_pk_x: pallas::Base,
        ea_pk_y: pallas::Base,
        dc_slot_nullifier: pallas::Base,
    ) -> Self {
        Instance {
            van_nullifier,
            r_vpk_x,
            r_vpk_y,
            vote_authority_note_new,
            delegation_commitment,
            vote_comm_tree_root,
            vote_comm_tree_anchor_height,
            voting_round_id,
            ea_pk_x,
            ea_pk_y,
            dc_slot_nullifier,
        }
    }

    /// Packs the public inputs for halo2 proving and verification, placing
    /// each field at the offset the circuit constrains it at.
    pub fn to_halo2_instance(&self) -> Vec<vesta::Scalar> {
        let mut inputs = vec![vesta::Scalar::zero(); Self::NUM_PUBLIC_INPUTS];
        inputs[VAN_NULLIFIER_PUBLIC_OFFSET] = self.van_nullifier;
        inputs[R_VPK_X_PUBLIC_OFFSET] = self.r_vpk_x;
        inputs[R_VPK_Y_PUBLIC_OFFSET] = self.r_vpk_y;
        inputs[VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET] = self.vote_authority_note_new;
        inputs[DELEGATION_COMMITMENT_PUBLIC_OFFSET] = self.delegation_commitment;
        inputs[VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET] = self.vote_comm_tree_root;
        inputs[VOTE_COMM_TREE_ANCHOR_HEIGHT_PUBLIC_OFFSET] = self.vote_comm_tree_anchor_height;
        inputs[VOTING_ROUND_ID_PUBLIC_OFFSET] = self.voting_round_id;
        inputs[EA_PK_X_PUBLIC_OFFSET] = self.ea_pk_x;
        inputs[EA_PK_Y_PUBLIC_OFFSET] = self.ea_pk_y;
        inputs[DC_SLOT_NULLIFIER_PUBLIC_OFFSET] = self.dc_slot_nullifier;
        inputs
    }
}

// ================================================================
// Tests
// ================================================================

#[cfg(test)]
mod tests {
    use super::super::builder::{build_proxy_delegation_proof, ProxyDelegationBundle, ShareLayout};
    use super::super::vectors::single_leaf_path;
    use super::*;
    use crate::ff::{Field, PrimeField, PrimeFieldBits};
    use crate::gadgets::elgamal::{base_to_scalar, elgamal_encrypt, spend_auth_g_affine};
    use crate::gadgets::van_integrity::van_integrity_hash;
    use crate::group::{Curve, Group};
    use crate::params::BALLOT_DIVISOR;
    use crate::protocol_hash::poseidon_hash_2;
    use crate::rand::rngs::OsRng;
    use crate::share_reveal::ShareRevealBundle;
    use crate::shares_hash::{share_commitment, shares_hash};
    use crate::vote_proof::{derive_vote_authority_transition, van_nullifier_hash};
    use core::iter;
    use voting_crypto_deps::composition::{COMMIT_IVK_PERSONALIZATION, L_ORCHARD_BASE};
    use voting_crypto_deps::halo2_gadgets::sinsemilla::primitives::CommitDomain;
    use voting_crypto_deps::halo2_proofs::{
        dev::{metadata, MockProver},
        plonk::Any,
    };
    use voting_crypto_deps::orchard::keys::{FullViewingKey, SpendingKey};
    use voting_crypto_deps::pasta_curves::arithmetic::CurveAffine;
    use voting_crypto_deps::CircuitValue as _;

    /// Root of a tree holding only `leaf`, at position 0.
    fn single_leaf_root(leaf: pallas::Base) -> pallas::Base {
        single_leaf_path()
            .iter()
            .fold(leaf, |node, sibling| poseidon_hash_2(node, *sibling))
    }

    /// Derives the voting address `(g_d, pk_d)` that condition 3 binds to
    /// `(vsk, nk, rivk_v)`.
    fn derive_voting_address(
        vsk: pallas::Scalar,
        nk: pallas::Base,
        rivk_v: pallas::Scalar,
    ) -> (pallas::Affine, pallas::Affine) {
        let ak_x = *(spend_auth_g_affine() * vsk)
            .to_affine()
            .coordinates()
            .unwrap()
            .x();
        let ivk_v = CommitDomain::new(COMMIT_IVK_PERSONALIZATION)
            .short_commit(
                iter::empty()
                    .chain(ak_x.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE))
                    .chain(nk.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE)),
                &rivk_v,
            )
            .unwrap();
        let g_d = pallas::Point::generator() * pallas::Scalar::from(12345u64);
        let pk_d = g_d * base_to_scalar(ivk_v).unwrap();
        (g_d.to_affine(), pk_d.to_affine())
    }

    /// One ZKP #4 statement. Each negative test changes one field of the
    /// default to target one condition.
    #[derive(Clone)]
    struct Fixture {
        /// Old VAN weight `W`.
        total: u64,
        /// The 16 delegated shares; `w` is their sum.
        shares: [u64; 16],
        /// Successor weight; `W - w` when `None`.
        remaining_override: Option<pallas::Base>,
        /// Proposal authority of the old VAN.
        authority_old: u64,
        /// Delegate index the prover witnesses.
        delegate_index: u64,
        /// Delegate index in the public DC; `delegate_index` when `None`.
        instance_delegate_index: Option<u64>,
        /// Proposal id in the public DC.
        instance_dc_proposal: u64,
        /// DC slot.
        dc_slot: u64,
    }

    impl Default for Fixture {
        fn default() -> Self {
            let mut shares = [0u64; 16];
            shares[0] = 1000;
            shares[3] = 100;
            shares[7] = 7;
            Fixture {
                total: 10_000,
                shares,
                remaining_override: None,
                authority_old: MAX_PROPOSAL_AUTHORITY,
                delegate_index: 1234,
                instance_delegate_index: None,
                instance_dc_proposal: 0,
                dc_slot: 3,
            }
        }
    }

    /// The keys, round and VAN `rand` behind a statement. Statements about one
    /// VAN chain share them.
    #[derive(Clone, Copy)]
    struct Secrets {
        vsk: pallas::Scalar,
        nk: pallas::Base,
        rivk: pallas::Scalar,
        alpha: pallas::Scalar,
        round: pallas::Base,
        rand: pallas::Base,
    }

    impl Secrets {
        /// Draws every secret at random.
        fn random() -> Self {
            let mut rng = OsRng;
            Secrets {
                vsk: pallas::Scalar::random(&mut rng),
                nk: pallas::Base::random(&mut rng),
                rivk: pallas::Scalar::random(&mut rng),
                alpha: pallas::Scalar::random(&mut rng),
                round: pallas::Base::random(&mut rng),
                rand: pallas::Base::random(&mut rng),
            }
        }
    }

    /// A circuit and instance built from a [`Fixture`], plus the share data
    /// ZKP #3 needs to reveal a DC share and the openings that malicious
    /// witness tests rebuild public inputs from.
    struct Built {
        circuit: Circuit,
        instance: Instance,
        secrets: Secrets,
        /// x-coordinates of the voting address `(g_d, pk_d)`.
        address_x: (pallas::Base, pallas::Base),
        /// The spent VAN.
        van_old: pallas::Base,
        ea_pk: pallas::Affine,
        delegate_index: u64,
        randomness: [pallas::Base; 16],
        share_comms: [pallas::Base; 16],
        blinds: [pallas::Base; 16],
        /// `(x, y)` of each share's `C1`.
        c1: [(pallas::Base, pallas::Base); 16],
        /// `(x, y)` of each share's `C2`.
        c2: [(pallas::Base, pallas::Base); 16],
        round: pallas::Base,
        /// The honest DC for `delegate_index`, ignoring the instance overrides.
        dc: pallas::Base,
    }

    impl Built {
        /// The full-authority VAN of this address in this round with `weight`
        /// and `rand`.
        fn van(&self, weight: pallas::Base, rand: pallas::Base) -> pallas::Base {
            van_integrity_hash(
                self.address_x.0,
                self.address_x.1,
                weight,
                self.round,
                pallas::Base::from(MAX_PROPOSAL_AUTHORITY),
                rand,
            )
        }

        /// Replaces share `i`'s ciphertext witness and rebinds the public DC
        /// to the new ciphertexts, so conditions 10 and 13 still hold.
        fn set_ciphertext(&mut self, i: usize, (c1x, c2x, c1y, c2y): CiphertextCoords) {
            self.c1[i] = (c1x, c1y);
            self.c2[i] = (c2x, c2y);
            self.circuit.enc_share_c1_x[i] = Value::known(c1x);
            self.circuit.enc_share_c2_x[i] = Value::known(c2x);
            self.circuit.enc_share_c1_y[i] = Value::known(c1y);
            self.circuit.enc_share_c2_y[i] = Value::known(c2y);
            let sh = shares_hash(
                self.blinds,
                self.c1.map(|c| c.0),
                self.c2.map(|c| c.0),
                self.c1.map(|c| c.1),
                self.c2.map(|c| c.1),
            );
            self.share_comms = core::array::from_fn(|j| {
                share_commitment(
                    self.blinds[j],
                    self.c1[j].0,
                    self.c2[j].0,
                    self.c1[j].1,
                    self.c2[j].1,
                )
            });
            self.dc = delegation_commitment_hash(self.round, sh, self.delegate_index as u32);
            self.instance.delegation_commitment = self.dc;
        }

        /// Share `i`'s ciphertext coordinates, in [`elgamal_encrypt`] order.
        fn ciphertext(&self, i: usize) -> CiphertextCoords {
            (self.c1[i].0, self.c2[i].0, self.c1[i].1, self.c2[i].1)
        }
    }

    /// `(c1_x, c2_x, c1_y, c2_y)`, as [`elgamal_encrypt`] returns them.
    type CiphertextCoords = (pallas::Base, pallas::Base, pallas::Base, pallas::Base);

    /// Builds the witnesses and public inputs for `fx`, with fresh keys.
    fn build(fx: &Fixture) -> Built {
        build_with(fx, Secrets::random())
    }

    /// Builds the witnesses and public inputs for `fx` from `secrets`.
    fn build_with(fx: &Fixture, secrets: Secrets) -> Built {
        let mut rng = OsRng;
        let Secrets {
            vsk,
            nk,
            rivk,
            alpha,
            round,
            rand,
        } = secrets;
        let (g_d, pk_d) = derive_voting_address(vsk, nk, rivk);
        let g_d_x = *g_d.coordinates().unwrap().x();
        let pk_d_x = *pk_d.coordinates().unwrap().x();
        let w: u64 = fx.shares.iter().sum();
        let total = pallas::Base::from(fx.total);
        let remaining = fx
            .remaining_override
            .unwrap_or(pallas::Base::from(fx.total) - pallas::Base::from(w));

        let van_old = van_integrity_hash(
            g_d_x,
            pk_d_x,
            total,
            round,
            pallas::Base::from(fx.authority_old),
            rand,
        );
        let van_new = van_integrity_hash(
            g_d_x,
            pk_d_x,
            remaining,
            round,
            pallas::Base::from(MAX_PROPOSAL_AUTHORITY),
            rand,
        );
        let van_nullifier =
            poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<4>, 3, 2>::init().hash([
                nk,
                domain_tags::vote_authority_spend(),
                round,
                van_old,
            ]);
        let r_vpk = (spend_auth_g_affine() * (vsk + alpha)).to_affine();
        let ea_pk = (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine();

        let randomness: [pallas::Base; 16] =
            core::array::from_fn(|i| pallas::Base::from((i as u64 + 1) * 101));
        let blinds: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let mut c1x = [pallas::Base::zero(); 16];
        let mut c2x = [pallas::Base::zero(); 16];
        let mut c1y = [pallas::Base::zero(); 16];
        let mut c2y = [pallas::Base::zero(); 16];
        for i in 0..16 {
            (c1x[i], c2x[i], c1y[i], c2y[i]) =
                elgamal_encrypt(pallas::Base::from(fx.shares[i]), randomness[i], ea_pk).unwrap();
        }
        let sh = shares_hash(blinds, c1x, c2x, c1y, c2y);
        let share_comms: [pallas::Base; 16] =
            core::array::from_fn(|i| share_commitment(blinds[i], c1x[i], c2x[i], c1y[i], c2y[i]));
        let dc_instance = vote_commitment::vote_commitment_hash(
            round,
            sh,
            pallas::Base::from(fx.instance_dc_proposal),
            pallas::Base::from(fx.instance_delegate_index.unwrap_or(fx.delegate_index)),
        );

        let circuit = Circuit {
            vote_comm_tree_path: Value::known(single_leaf_path()),
            vote_comm_tree_position: Value::known(0),
            vpk_g_d: Value::known(g_d),
            vpk_pk_d: Value::known(pk_d),
            total_note_value: Value::known(total),
            remaining_note_value: Value::known(remaining),
            van_comm_rand: Value::known(rand),
            vote_authority_note_old: Value::known(van_old),
            vsk: Value::known(vsk),
            rivk_v: Value::known(rivk),
            alpha_v: Value::known(alpha),
            vsk_nk: Value::known(nk),
            shares: fx.shares.map(|s| Value::known(pallas::Base::from(s))),
            enc_share_c1_x: c1x.map(Value::known),
            enc_share_c2_x: c2x.map(Value::known),
            enc_share_c1_y: c1y.map(Value::known),
            enc_share_c2_y: c2y.map(Value::known),
            share_blinds: blinds.map(Value::known),
            share_randomness: randomness.map(Value::known),
            ea_pk: Value::known(ea_pk),
            delegate_index: Value::known(pallas::Base::from(fx.delegate_index)),
            dc_slot: Value::known(pallas::Base::from(fx.dc_slot)),
        };
        let instance = Instance::from_parts(
            van_nullifier,
            *r_vpk.coordinates().unwrap().x(),
            *r_vpk.coordinates().unwrap().y(),
            van_new,
            dc_instance,
            single_leaf_root(van_old),
            pallas::Base::from(77u64),
            round,
            *ea_pk.coordinates().unwrap().x(),
            *ea_pk.coordinates().unwrap().y(),
            dc_slot_nullifier_hash(nk, round, rand, fx.dc_slot as u8),
        );
        Built {
            circuit,
            instance,
            secrets,
            address_x: (g_d_x, pk_d_x),
            van_old,
            ea_pk,
            delegate_index: fx.delegate_index,
            randomness,
            share_comms,
            blinds,
            c1: core::array::from_fn(|i| (c1x[i], c1y[i])),
            c2: core::array::from_fn(|i| (c2x[i], c2y[i])),
            round,
            dc: delegation_commitment_hash(round, sh, fx.delegate_index as u32),
        }
    }

    /// Runs MockProver on `fx` and returns the rendered failures, which are
    /// empty when the statement holds.
    fn failures(fx: &Fixture) -> Vec<String> {
        built_failures(&build(fx))
    }

    /// Runs MockProver on `b` and returns the rendered failures.
    fn built_failures(b: &Built) -> Vec<String> {
        mock_failures(&b.circuit, &b.instance)
    }

    /// Runs MockProver on `circuit` against `instance` and returns the
    /// rendered failures.
    fn mock_failures(circuit: &Circuit, instance: &Instance) -> Vec<String> {
        let prover = MockProver::run(K, circuit, vec![instance.to_halo2_instance()])
            .expect("synthesis succeeds");
        match prover.verify() {
            Ok(()) => vec![],
            Err(errs) => errs.iter().map(|e| format!("{e}")).collect(),
        }
    }

    /// Returns whether `fx` verifies under MockProver.
    fn verifies(fx: &Fixture) -> bool {
        failures(fx).is_empty()
    }

    /// Returns whether `b` verifies under MockProver.
    fn built_verifies(b: &Built) -> bool {
        built_failures(b).is_empty()
    }

    /// Configures the circuit, to name its columns in failure checks.
    fn config() -> Config {
        <Circuit as plonk::Circuit<pallas::Base>>::configure(&mut ConstraintSystem::default())
    }

    /// Returns whether `fails` include a strict 3-word range check failing on
    /// `column`: `advices[9]` for condition 9, `elgamal_advices_b[9]` for
    /// condition 7.
    fn range_check_fails_on(fails: &[String], column: Column<Advice>) -> bool {
        let column = column_name(column);
        fails
            .iter()
            .any(|f| f.contains(&column) && f.contains("'3 words range check'"))
    }

    /// The name MockProver gives `column` in its failures.
    fn column_name(column: impl Into<Column<Any>>) -> String {
        format!("{:?}", metadata::Column::from(column.into()))
    }

    /// Returns whether `fails` include the public input at `offset`.
    fn fails_at_public_input(fails: &[String], offset: usize) -> bool {
        let column = column_name(config().primary);
        let row = format!("on row {offset})");
        fails
            .iter()
            .any(|f| f.contains(&column) && f.contains(&row))
    }

    /// Returns whether `fails` include a cell of region `region` on one of
    /// `columns`.
    fn fails_in(fails: &[String], region: &str, columns: &[Column<Advice>]) -> bool {
        let region = format!("('{region}')");
        fails
            .iter()
            .any(|f| f.contains(&region) && columns.iter().any(|c| f.contains(&column_name(*c))))
    }

    /// Asserts that `fails` is non-empty and that every public input is
    /// consistent, so a private condition rejected the witness.
    fn assert_rejected_privately(fails: &[String], what: &str) {
        assert!(!fails.is_empty(), "{what} must be rejected");
        let instance = column_name(config().primary);
        assert!(
            !fails.iter().any(|f| f.contains(&instance)),
            "{what}: every public input must hold: {fails:?}"
        );
    }

    /// Asserts that condition 3's `pk_d = [ivk] g_d` check rejected `b`.
    fn assert_rejected_by_address_ownership(b: &Built) {
        let fails = built_failures(b);
        assert_rejected_privately(&fails, "a foreign address");
        assert!(
            fails_in(&fails, "variable-base scalar mul", &config().advices),
            "condition 3 must fail: {fails:?}"
        );
    }

    /// Asserts that condition 11 rejected share `i` on its own El Gamal track
    /// and that the other track holds.
    fn assert_rejected_on_track_of(b: &Built, i: usize, what: &str) {
        let fails = built_failures(b);
        assert_rejected_privately(&fails, what);
        let config = config();
        let (own, other) = if i < ELGAMAL_TRACK_SPLIT {
            (config.elgamal_advices, config.elgamal_advices_b)
        } else {
            (config.elgamal_advices_b, config.elgamal_advices)
        };
        assert!(
            fails_in(&fails, "complete point addition", &own),
            "{what}: condition 11 must fail on share {i}'s track: {fails:?}"
        );
        assert!(
            !fails_in(&fails, "complete point addition", &other),
            "{what}: the other track must hold: {fails:?}"
        );
    }

    /// Asserts that `fx` fails condition 9 while condition 7 holds. Condition
    /// 11's unsigned 30-bit decomposition rejects such a share as well.
    fn assert_rejected_by_share_range(fx: &Fixture) {
        let fails = failures(fx);
        let config = config();
        assert!(
            range_check_fails_on(&fails, config.advices[9]),
            "condition 9 must fail: {fails:?}"
        );
        assert!(
            !range_check_fails_on(&fails, config.elgamal_advices_b[9]),
            "condition 7 must hold: {fails:?}"
        );
    }

    /// Asserts that `slot` is rejected only by condition 14's 3-bit range check.
    fn assert_rejected_by_slot_range_check(slot: u64) {
        let fails = failures(&Fixture {
            dc_slot: slot,
            ..Fixture::default()
        });
        assert!(!fails.is_empty(), "slot {slot} must be rejected");
        for f in &fails {
            assert!(
                f.starts_with("Lookup ") && f.contains("'Range check 3 bits'"),
                "slot {slot}: unexpected failure outside the slot range check: {f}"
            );
        }
    }

    /// Builds a ZKP #3 reveal of DC share `share` as `(POOL_PROPOSAL_ID, decision)`.
    fn reveal(b: &Built, share: usize, decision: u64) -> ShareRevealBundle {
        crate::share_reveal::build_share_reveal(
            single_leaf_path(),
            0,
            b.share_comms,
            b.blinds[share],
            b.c1[share].0,
            b.c2[share].0,
            b.c1[share].1,
            b.c2[share].1,
            share as u32,
            pallas::Base::from(POOL_PROPOSAL_ID),
            pallas::Base::from(decision),
            b.round,
        )
    }

    /// Returns whether `bundle`'s ZKP #3 circuit verifies against `instance`.
    fn reveal_verifies(
        bundle: &ShareRevealBundle,
        instance: &crate::share_reveal::Instance,
    ) -> bool {
        MockProver::run(
            crate::share_reveal::K,
            &bundle.circuit,
            vec![instance.to_halo2_instance()],
        )
        .expect("synthesis succeeds")
        .verify()
        .is_ok()
    }

    // ================================================================
    // Frozen constants, vectors and packing
    // ================================================================

    /// These constants are baked into the ZKP #4 verifying key and the
    /// chain's reveal rule, so they must never change.
    #[test]
    fn frozen_constants_are_pinned() {
        assert_eq!(POOL_PROPOSAL_ID, 0);
        assert_eq!(DC_SLOT_BITS, 3);
        assert_eq!(1usize << DC_SLOT_BITS, 8);
    }

    /// `TAG + s` for each valid slot only moves byte 0 of the
    /// `"proxy delegation slot"` tag, so the eight slot tags are distinct from
    /// each other and from the crate's other domain tags, and so are the
    /// slot nullifiers.
    #[test]
    fn slot_tags_are_distinct() {
        let n = 1usize << DC_SLOT_BITS;
        let tag = domain_tags::proxy_delegation_slot();
        let slot_tags: Vec<pallas::Base> =
            (0..n as u64).map(|s| tag + pallas::Base::from(s)).collect();
        for (s, t) in slot_tags.iter().enumerate() {
            let r = t.to_repr();
            assert_eq!(r[0], b'p' + s as u8);
            assert_eq!(&r[1..21], &b"proxy delegation slot"[1..]);
            assert!(r[21..].iter().all(|&x| x == 0));
        }
        let others = [
            pallas::Base::from(DOMAIN_VAN),
            pallas::Base::from(DOMAIN_VC),
            domain_tags::vote_authority_spend(),
            domain_tags::governance_authorization(),
            domain_tags::share_spend(),
        ];
        for (i, a) in slot_tags.iter().enumerate() {
            for b in &slot_tags[i + 1..] {
                assert_ne!(a, b);
            }
            assert!(
                !others.contains(a),
                "slot tag {i} collides with another tag"
            );
        }
        let (nk, round, rand) = (
            pallas::Base::from(1u64),
            pallas::Base::from(42u64),
            pallas::Base::from(5u64),
        );
        let nfs: Vec<_> = (0..n as u8)
            .map(|s| dc_slot_nullifier_hash(nk, round, rand, s))
            .collect();
        for (i, a) in nfs.iter().enumerate() {
            for b in &nfs[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    /// The DC is the vote commitment for proposal 0 with decision `d`.
    #[test]
    fn delegation_commitment_hash_frozen_vector() {
        let (round, sh) = (pallas::Base::from(42u64), pallas::Base::from(100u64));
        let dc = delegation_commitment_hash(round, sh, 7);
        assert_eq!(
            dc,
            vote_commitment::vote_commitment_hash(
                round,
                sh,
                pallas::Base::zero(),
                pallas::Base::from(7u64)
            )
        );
        assert_eq!(
            dc.to_repr(),
            [
                247, 121, 250, 52, 50, 53, 218, 40, 230, 53, 218, 119, 147, 220, 98, 255, 76, 208,
                35, 208, 35, 9, 136, 21, 184, 50, 80, 60, 16, 160, 147, 5,
            ]
        );
    }

    /// Pins the condition-14 slot nullifier formula.
    #[test]
    fn dc_slot_nullifier_hash_frozen_vector() {
        let nf = dc_slot_nullifier_hash(
            pallas::Base::from(1u64),
            pallas::Base::from(42u64),
            pallas::Base::from(5u64),
            3,
        );
        assert_eq!(
            nf.to_repr(),
            [
                201, 98, 204, 233, 194, 253, 198, 143, 148, 150, 211, 254, 103, 20, 47, 40, 213, 1,
                57, 57, 100, 87, 175, 120, 32, 255, 71, 120, 116, 0, 53, 16,
            ]
        );
    }

    // ================================================================
    // Accepted statements
    // ================================================================

    /// A partial delegation, with weight left in the successor, verifies.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_valid_partial() {
        assert!(verifies(&Fixture::default()));
    }

    /// Delegating all of `W` leaves a zero-weight successor.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_valid_full_delegation_successor_zero() {
        let fx = Fixture {
            total: 1107,
            ..Fixture::default()
        };
        assert!(verifies(&fx));
    }

    /// `W = 2^30`, the most ZKP #1 can mint, delegated in full over 16 shares.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_valid_max_weight() {
        let fx = Fixture {
            total: 1 << 30,
            shares: [(1 << 30) / 16; 16],
            ..Fixture::default()
        };
        assert!(verifies(&fx));
    }

    /// The circuit places no bound on the delegate index. The chain rejects
    /// `d = 0`, and any index past the registry, when a pool share is revealed.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_accepts_any_delegate_index() {
        for d in [0, u64::from(u32::MAX)] {
            let fx = Fixture {
                delegate_index: d,
                ..Fixture::default()
            };
            assert!(verifies(&fx), "d = {d} must verify");
        }
    }

    /// Condition 14 accepts the first and last slots.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_valid_slot_7() {
        for slot in [0, 7] {
            let fx = Fixture {
                dc_slot: slot,
                ..Fixture::default()
            };
            assert!(verifies(&fx), "slot {slot} must verify");
        }
    }

    // ================================================================
    // Rejected statements
    // ================================================================

    /// Condition 2: a VAN that has voted (authority below MAX) cannot delegate.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_non_max_authority() {
        let fx = Fixture {
            authority_old: MAX_PROPOSAL_AUTHORITY - 2,
            ..Fixture::default()
        };
        assert!(!verifies(&fx));
    }

    /// Condition 7: `w > W` would wrap `W_new` around the field.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_overdelegation_w_gt_w() {
        let fails = failures(&Fixture {
            total: 1000,
            ..Fixture::default()
        });
        assert!(
            range_check_fails_on(&fails, config().elgamal_advices_b[9]),
            "condition 7 must fail: {fails:?}"
        );
    }

    /// Condition 8: the successor cannot keep weight that was delegated.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_inflated_successor() {
        let fx = Fixture {
            remaining_override: Some(pallas::Base::from(10_000u64)),
            ..Fixture::default()
        };
        assert!(!verifies(&fx));
    }

    /// Condition 9: a share of `2^30` is rejected even though the sum and
    /// `W_new` are valid.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_share_overflow() {
        let mut fx = Fixture {
            total: (1 << 30) + 2000,
            ..Fixture::default()
        };
        fx.shares[0] = 1 << 30;
        assert_rejected_by_share_range(&fx);
    }

    /// Condition 12: a DC must carry weight.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_zero_weight() {
        let fx = Fixture {
            shares: [0; 16],
            ..Fixture::default()
        };
        assert!(!verifies(&fx));
    }

    /// Condition 13: the DC must commit to proposal 0.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_nonzero_pool_proposal() {
        let fx = Fixture {
            instance_dc_proposal: 1,
            ..Fixture::default()
        };
        assert!(!verifies(&fx));
    }

    /// Condition 13: the DC must commit to the witnessed delegate.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_wrong_delegate_in_dc() {
        let fx = Fixture {
            instance_delegate_index: Some(1235),
            ..Fixture::default()
        };
        assert!(!verifies(&fx));
    }

    /// Condition 14 rejects slot 8, the first past the cap.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_slot_8_at_range_check() {
        assert_rejected_by_slot_range_check(8);
    }

    /// Condition 14 rejects slots 15 and 16.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_slot_15() {
        assert_rejected_by_slot_range_check(15);
        assert_rejected_by_slot_range_check(16);
    }

    // ================================================================
    // Malicious witnesses
    //
    // Each test starts from an honest statement and changes witnesses, then
    // recomputes every public input those witnesses feed, so only the
    // targeted condition can reject it.
    // ================================================================

    /// Condition 3: a prover who knows a VAN's opening but not its keys cannot
    /// spend it, even with self-consistent nullifiers and `r_vpk`.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_spend_without_the_address_keys() {
        let fx = Fixture::default();
        let mut b = build(&fx);
        let thief = Secrets::random();
        b.circuit.vsk = Value::known(thief.vsk);
        b.circuit.vsk_nk = Value::known(thief.nk);
        b.circuit.rivk_v = Value::known(thief.rivk);
        b.circuit.alpha_v = Value::known(thief.alpha);
        let r_vpk = (spend_auth_g_affine() * (thief.vsk + thief.alpha)).to_affine();
        b.instance.r_vpk_x = *r_vpk.coordinates().unwrap().x();
        b.instance.r_vpk_y = *r_vpk.coordinates().unwrap().y();
        b.instance.van_nullifier = van_nullifier_hash(thief.nk, b.round, b.van_old);
        b.instance.dc_slot_nullifier =
            dc_slot_nullifier_hash(thief.nk, b.round, b.secrets.rand, fx.dc_slot as u8);
        assert_rejected_by_address_ownership(&b);
    }

    /// Condition 3 binds `nk` to the address, so a fresh `nk` cannot mint a
    /// fresh VAN nullifier (a double spend) or slot nullifier (a ninth DC).
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_nk_not_bound_to_the_address() {
        let fx = Fixture::default();
        let mut b = build(&fx);
        let nk = pallas::Base::random(&mut OsRng);
        b.circuit.vsk_nk = Value::known(nk);
        b.instance.van_nullifier = van_nullifier_hash(nk, b.round, b.van_old);
        b.instance.dc_slot_nullifier =
            dc_slot_nullifier_hash(nk, b.round, b.secrets.rand, fx.dc_slot as u8);
        assert_rejected_by_address_ownership(&b);
    }

    /// Condition 4: the public `r_vpk` must be `[vsk + alpha_v] G` for the
    /// witnessed `alpha_v`.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_r_vpk_not_from_the_spend_key() {
        let mut b = build(&Fixture::default());
        b.circuit.alpha_v = Value::known(b.secrets.alpha + pallas::Scalar::one());
        let fails = built_failures(&b);
        assert!(
            fails_at_public_input(&fails, R_VPK_X_PUBLIC_OFFSET),
            "{fails:?}"
        );
        assert!(
            fails_at_public_input(&fails, R_VPK_Y_PUBLIC_OFFSET),
            "{fails:?}"
        );
    }

    /// Condition 1: a VAN must be in the tree. An inflated copy of a real VAN
    /// is not, and neither is the real VAN at another position.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_van_outside_the_tree() {
        let secrets = Secrets::random();
        let real = build_with(&Fixture::default(), secrets);
        assert!(built_verifies(&real));

        let mut inflated = build_with(
            &Fixture {
                total: 1 << 30,
                ..Fixture::default()
            },
            secrets,
        );
        inflated.instance.vote_comm_tree_root = real.instance.vote_comm_tree_root;
        let mut moved = real;
        moved.circuit.vote_comm_tree_position = Value::known(1);
        for (b, what) in [(inflated, "inflated VAN"), (moved, "moved VAN")] {
            let fails = built_failures(&b);
            assert!(
                fails_at_public_input(&fails, VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET),
                "{what}: {fails:?}"
            );
        }
    }

    /// Condition 11 on both El Gamal tracks: a ciphertext of a different value
    /// is rejected at the edges of each track, even with the DC rebound to it.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_ciphertext_of_another_value() {
        let fx = Fixture::default();
        for i in [0, ELGAMAL_TRACK_SPLIT - 1, ELGAMAL_TRACK_SPLIT, 15] {
            let mut b = build(&fx);
            let forged = pallas::Base::from(fx.shares[i] + 1);
            b.set_ciphertext(
                i,
                elgamal_encrypt(forged, b.randomness[i], b.ea_pk).unwrap(),
            );
            assert_rejected_on_track_of(&b, i, "another value");
        }
    }

    /// Condition 11 binds `C1` to the witnessed randomness and both `y`
    /// coordinates, on each side of the track split.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_ciphertext_not_from_the_witnesses() {
        let fx = Fixture::default();
        for i in [ELGAMAL_TRACK_SPLIT - 1, ELGAMAL_TRACK_SPLIT] {
            let mut b = build(&fx);
            let other_r = b.randomness[i] + pallas::Base::one();
            let share = pallas::Base::from(fx.shares[i]);
            b.set_ciphertext(i, elgamal_encrypt(share, other_r, b.ea_pk).unwrap());
            assert_rejected_on_track_of(&b, i, "other randomness");

            // Negating C2 gives a valid point that encrypts -share.
            let mut b = build(&fx);
            let (c1x, c2x, c1y, c2y) = b.ciphertext(i);
            b.set_ciphertext(i, (c1x, c2x, c1y, -c2y));
            assert_rejected_on_track_of(&b, i, "negated C2");
        }
    }

    /// Condition 11 encrypts share `i` into ciphertext `i`, so the shares on
    /// either side of the track split cannot trade ciphertexts or weight.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_shares_crossed_between_tracks() {
        let fx = Fixture::default();
        let (a, z) = (ELGAMAL_TRACK_SPLIT - 1, ELGAMAL_TRACK_SPLIT);
        assert_ne!(fx.shares[a], fx.shares[z]);

        let mut b = build(&fx);
        let (ca, cz) = (b.ciphertext(a), b.ciphertext(z));
        b.set_ciphertext(a, cz);
        b.set_ciphertext(z, ca);
        let swapped = b;

        // Moving weight from share 7 to share 8 keeps conditions 8 and 9.
        let mut moved = build(&fx);
        moved.circuit.shares[a] = Value::known(pallas::Base::from(fx.shares[a] - 5));
        moved.circuit.shares[z] = Value::known(pallas::Base::from(fx.shares[z] + 5));

        let config = config();
        for (b, what) in [(swapped, "swapped ciphertexts"), (moved, "moved weight")] {
            let fails = built_failures(&b);
            assert_rejected_privately(&fails, what);
            for track in [config.elgamal_advices, config.elgamal_advices_b] {
                assert!(
                    fails_in(&fails, "complete point addition", &track),
                    "{what}: both tracks must fail: {fails:?}"
                );
            }
        }
    }

    /// Condition 10: the DC binds each share's blind.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_unbound_share_blind() {
        let mut b = build(&Fixture::default());
        b.circuit.share_blinds[ELGAMAL_TRACK_SPLIT] =
            Value::known(b.blinds[ELGAMAL_TRACK_SPLIT] + pallas::Base::one());
        let fails = built_failures(&b);
        assert!(
            fails_at_public_input(&fails, DELEGATION_COMMITMENT_PUBLIC_OFFSET),
            "{fails:?}"
        );
    }

    /// Condition 6: the successor keeps the old VAN's `rand`, which keys
    /// every slot nullifier, and its address.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_successor_keeps_rand_and_address() {
        let fx = Fixture::default();
        let remaining = pallas::Base::from(fx.total - fx.shares.iter().sum::<u64>());

        let mut fresh_rand = build(&fx);
        fresh_rand.instance.vote_authority_note_new =
            fresh_rand.van(remaining, pallas::Base::random(&mut OsRng));

        let mut moved = build(&fx);
        let other = build(&fx);
        moved.instance.vote_authority_note_new = other.van(remaining, moved.secrets.rand);

        for (b, what) in [(fresh_rand, "fresh rand"), (moved, "other address")] {
            let fails = built_failures(&b);
            assert!(
                fails_at_public_input(&fails, VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET),
                "{what}: {fails:?}"
            );
        }
    }

    /// ZKP #4 -> ZKP #4: the successor's DC for an already used slot publishes
    /// the same slot nullifier, so the chain rejects it, while a new slot
    /// gives a new one. No other slot nullifier verifies for that slot.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_slot_nullifier_is_stable_across_successors() {
        let secrets = Secrets::random();
        let first_fx = Fixture::default();
        let first = build_with(&first_fx, secrets);
        assert!(built_verifies(&first));

        let successor_fx = |dc_slot| Fixture {
            total: first_fx.total - first_fx.shares.iter().sum::<u64>(),
            dc_slot,
            ..Fixture::default()
        };
        let replay = build_with(&successor_fx(first_fx.dc_slot), secrets);
        assert_eq!(replay.van_old, first.instance.vote_authority_note_new);
        assert!(built_verifies(&replay));
        assert_eq!(
            replay.instance.dc_slot_nullifier,
            first.instance.dc_slot_nullifier
        );
        assert_ne!(replay.instance.van_nullifier, first.instance.van_nullifier);

        let next = build_with(&successor_fx(first_fx.dc_slot + 1), secrets);
        assert!(built_verifies(&next));
        assert_ne!(
            next.instance.dc_slot_nullifier,
            first.instance.dc_slot_nullifier
        );

        let mut fresh = build_with(&successor_fx(first_fx.dc_slot), secrets);
        fresh.instance.dc_slot_nullifier = dc_slot_nullifier_hash(
            secrets.nk,
            secrets.round,
            pallas::Base::random(&mut OsRng),
            first_fx.dc_slot as u8,
        );
        let fails = built_failures(&fresh);
        assert!(
            fails_at_public_input(&fails, DC_SLOT_NULLIFIER_PUBLIC_OFFSET),
            "{fails:?}"
        );
    }

    /// ZKP #2 -> ZKP #4: the VAN that ZKP #2 emits after a vote cannot
    /// delegate, because condition 2 fixes full authority.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_rejects_van_after_zkp2_vote() {
        let sk = SpendingKey::from_bytes([0x42; 32]).expect("valid test spending key");
        let fvk: FullViewingKey = (&sk).into();
        let nk = voting_crypto_deps::nullifier_key(&fvk).circuit_value();
        let (rand, round) = (
            pallas::Base::from(0xDEAD_u64),
            pallas::Base::from(0xCAFE_u64),
        );
        let w = 10_000;
        let transition = derive_vote_authority_transition(
            &sk,
            1,
            w * BALLOT_DIVISOR,
            rand,
            round,
            1,
            MAX_PROPOSAL_AUTHORITY,
        )
        .expect("valid ZKP #2 transition");

        let mut b = build_proxy_delegation_proof(
            &sk,
            1,
            w,
            rand,
            round,
            single_leaf_path(),
            0,
            123,
            (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine(),
            pallas::Scalar::from(7u64),
            17,
            4_800,
            0,
            [0x5e; 32],
            ShareLayout::Standard,
        )
        .expect("valid proxy delegation inputs");
        let failures = |b: &ProxyDelegationBundle| mock_failures(&b.circuit, &b.instance);
        assert_eq!(
            b.instance.vote_comm_tree_root,
            single_leaf_root(transition.vote_authority_note_old)
        );
        assert!(failures(&b).is_empty(), "the full-authority VAN delegates");

        let voted = transition.vote_authority_note_new;
        b.circuit.vote_authority_note_old = Value::known(voted);
        b.instance.vote_comm_tree_root = single_leaf_root(voted);
        b.instance.van_nullifier = van_nullifier_hash(nk, round, voted);
        assert_rejected_privately(&failures(&b), "the voted VAN");
    }

    // ================================================================
    // Single-share layout and ZKP #3 reveals
    // ================================================================

    /// Last-moment single-share layout: all of `w` in share 0, shares 1-15 zero.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp4_single_share_layout_accepts_and_reveals() {
        let mut fx = Fixture {
            shares: [0; 16],
            ..Fixture::default()
        };
        fx.shares[0] = 4_000;
        assert!(verifies(&fx), "single-share partial DC must verify");

        // Delegate everything at the 2^30 - 1 weight ceiling, so W_new = 0.
        let mut fx = Fixture {
            total: (1 << 30) - 1,
            shares: [0; 16],
            ..Fixture::default()
        };
        fx.shares[0] = (1 << 30) - 1;
        assert!(
            verifies(&fx),
            "single-share full DC at max weight must verify"
        );

        // ZKP #3 reveals share 0 as (0, d) against the DC leaf.
        let b = build(&fx);
        let rb = reveal(&b, 0, fx.delegate_index);
        assert_eq!(rb.instance.vote_comm_tree_root, single_leaf_root(b.dc));
        assert!(reveal_verifies(&rb, &rb.instance));

        // Share 0 at 2^30 with W = 2^30 + 1, so W_new = 1 passes condition 7.
        let mut fx = Fixture {
            total: (1 << 30) + 1,
            shares: [0; 16],
            ..Fixture::default()
        };
        fx.shares[0] = 1 << 30;
        assert_rejected_by_share_range(&fx);
    }

    /// ZKP #3, unchanged, reveals a DC share as `(0, d)`, including at
    /// `d = u32::MAX`, and rejects any other delegate or proposal.
    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn zkp3_reveals_dc_share_with_pool_proposal_and_large_index() {
        let fx = Fixture::default();
        let b = build(&fx);
        for share in [0, 7] {
            let rb = reveal(&b, share, fx.delegate_index);
            assert_eq!(rb.instance.vote_comm_tree_root, single_leaf_root(b.dc));
            assert!(reveal_verifies(&rb, &rb.instance));

            let mut wrong_delegate = rb.instance.clone();
            wrong_delegate.vote_decision = pallas::Base::from(fx.delegate_index + 1);
            assert!(!reveal_verifies(&rb, &wrong_delegate));

            let mut wrong_proposal = rb.instance.clone();
            wrong_proposal.proposal_id = pallas::Base::one();
            assert!(!reveal_verifies(&rb, &wrong_proposal));
        }

        let fx = Fixture {
            delegate_index: u64::from(u32::MAX),
            ..Fixture::default()
        };
        let b = build(&fx);
        let rb = reveal(&b, 0, fx.delegate_index);
        assert!(reveal_verifies(&rb, &rb.instance));
    }

    // ================================================================
    // Row budget
    // ================================================================

    /// Measures the floor planner's row high-water mark, column counts and
    /// lookups via `CircuitCost::measure`, and checks that the layout does not
    /// depend on witness values.
    #[test]
    #[ignore = "row-budget diagnostic; run with `cargo test --release proxy_delegation::circuit::tests::row_budget -- --ignored --nocapture`"]
    fn row_budget() {
        use std::println;
        use voting_crypto_deps::halo2_proofs::dev::CircuitCost;

        let filled = format!(
            "{:?}",
            CircuitCost::<vesta::Point, _>::measure(K, &build(&Fixture::default()).circuit)
        );
        let default = format!(
            "{:?}",
            CircuitCost::<vesta::Point, _>::measure(K, &Circuit::default())
        );
        // CircuitCost's fields are private, so read them from its Debug output.
        let field = |name: &str| -> usize {
            filled
                .split(&format!(" {name}: "))
                .nth(1)
                .and_then(|s| s.split([',', ' ', '}']).next())
                .and_then(|n| n.parse().ok())
                .unwrap_or_else(|| panic!("CircuitCost has no {name}"))
        };
        let max_rows = field("max_rows");
        let total_rows = 1usize << K;

        println!("=== proxy delegation circuit row budget (K={K}) ===");
        println!("  max_rows:           {max_rows} of {total_rows}");
        println!(
            "  headroom:           {}",
            total_rows.saturating_sub(max_rows)
        );
        println!("  max_advice_rows:    {}", field("max_advice_rows"));
        println!("  max_fixed_rows:     {}", field("max_fixed_rows"));
        println!("  advice columns:     {}", field("num_advice_columns"));
        println!("  fixed columns:      {}", field("num_fixed_columns"));
        println!("  instance columns:   {}", field("num_instance_columns"));
        println!("  lookups:            {}", field("lookups"));
        println!("  max degree:         {}", field("max_deg"));
        println!("  Full debug: {filled}");

        assert!(max_rows <= total_rows);
        assert_eq!(
            filled, default,
            "the layout must not depend on witness values"
        );
    }
}
