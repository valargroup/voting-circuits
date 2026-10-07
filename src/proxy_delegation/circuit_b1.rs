//! PROTOTYPE B1 variant: NSLOTS public encrypted pool slots in one proof.
//!
//! Conditions (numbering mirrors ZKP #2 where shared):
//! - C1  VAN membership (Poseidon Merkle path, root -> instance).
//! - C2  Old VAN integrity with proposal_authority = MAX (constant).
//! - C3  Diversified address integrity (CommitIvk chain).
//! - C4  Spend authority r_vpk -> instance.
//! - C5  VAN nullifier (identical to ZKP #2) -> instance.
//! - C6  Successor VAN integrity: same address/round/rand, MAX, weight W_new -> instance.
//! - C7  W_new in [0, 2^30) (strict 3-word lookup range check).
//! - C8  sum(shares) + W_new == W.
//! - C9  each share in [0, 2^30).
//! - C10 shares_hash (blinded share commitments).
//! - C11 El Gamal encryption integrity under ea_pk (instance).
//! - C12 w = sum(shares) != 0.
//! - C13 DC = Poseidon(DOMAIN_VC, round, shares_hash, 0, delegate_index) -> instance.

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

#[allow(unused_imports)]
use crate::{
    domain_tags,
    gadgets::{
        address_ownership::{prove_address_ownership, spend_auth_g_mul},
        elgamal::{prove_elgamal_encryptions, EaPkInstanceLoc, SpendAuthGFixedBase30Config},
        nonzero::NonZeroConfig,
        poseidon_merkle::{synthesize_poseidon_merkle_path, MerkleSwapGate},
        van_integrity, vote_commitment,
    },
    params::{
        MAX_PROPOSAL_AUTHORITY, RANGE_CHECK_WORD_BITS, SHARE_VALUE_RANGE_WORDS,
        VOTE_COMM_TREE_DEPTH,
    },
    shares_hash::compute_shares_hash_in_circuit,
};

/// Circuit size.
pub const K: u32 = 11;
pub const NSLOTS: usize = 10;
const SPLIT: usize = 5;
const SLOT_BASE: usize = 10;

const ELGAMAL_TRACK_SPLIT: usize = 8;
const PRIMARY_SHARES_HASH_COUNT: usize = 2;

/// Reserved proposal id for delegation commitments (baked as a constant).
pub(crate) const POOL_PROPOSAL_ID: u64 = 0;

const VAN_NULLIFIER_PUBLIC_OFFSET: usize = 0;
const R_VPK_X_PUBLIC_OFFSET: usize = 1;
const R_VPK_Y_PUBLIC_OFFSET: usize = 2;
const VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET: usize = 3;
const DELEGATION_COMMITMENT_PUBLIC_OFFSET: usize = 4;
const VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET: usize = 5;
#[allow(dead_code)]
const VOTE_COMM_TREE_ANCHOR_HEIGHT_PUBLIC_OFFSET: usize = 6;
const VOTING_ROUND_ID_PUBLIC_OFFSET: usize = 7;
const EA_PK_X_PUBLIC_OFFSET: usize = 8;
const EA_PK_Y_PUBLIC_OFFSET: usize = 9;

/// Native DC hash: `Poseidon(DOMAIN_VC, round, shares_hash, 0, delegate_index)`.
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

/// Circuit configuration.
#[derive(Clone, Debug)]
pub struct Config {
    primary: Column<InstanceColumn>,
    advices: [Column<Advice>; 10],
    elgamal_advices: [Column<Advice>; 10],
    elgamal_advices_b: [Column<Advice>; 10],
    poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    hash_poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    add_config: AddConfig,
    ecc_config: EccConfig<OrchardFixedBases>,
    elgamal_ecc_config: EccConfig<OrchardFixedBases>,
    elgamal_ecc_config_b: EccConfig<OrchardFixedBases>,
    sinsemilla_config:
        SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    commit_ivk_config: CommitIvkConfig,
    range_check: LookupRangeCheckConfig<pallas::Base, RANGE_CHECK_WORD_BITS>,
    range_check_b: LookupRangeCheckConfig<pallas::Base, RANGE_CHECK_WORD_BITS>,
    merkle_swap: MerkleSwapGate,
    nonzero: NonZeroConfig,
    nonzero_b: NonZeroConfig,
    spend_auth_g_fixed_base_30: SpendAuthGFixedBase30Config,
    spend_auth_g_fixed_base_30_b: SpendAuthGFixedBase30Config,
}

impl Config {
    fn poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.poseidon_config.clone())
    }
    fn hash_poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.hash_poseidon_config.clone())
    }
    fn add_chip(&self) -> AddChip {
        AddChip::construct(self.add_config.clone())
    }
    fn ecc_chip(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(self.ecc_config.clone(), CircuitVersion::AnchoredBase)
    }
    fn elgamal_ecc_chip(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(self.elgamal_ecc_config.clone(), CircuitVersion::AnchoredBase)
    }
    fn elgamal_ecc_chip_b(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(self.elgamal_ecc_config_b.clone(), CircuitVersion::AnchoredBase)
    }
    fn sinsemilla_chip(
        &self,
    ) -> SinsemillaChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases> {
        SinsemillaChip::construct(self.sinsemilla_config.clone())
    }
    fn commit_ivk_chip(&self) -> CommitIvkChip {
        CommitIvkChip::construct(self.commit_ivk_config.clone())
    }
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

/// ZKP #4 witnesses.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    pub(crate) vote_comm_tree_path: Value<[pallas::Base; VOTE_COMM_TREE_DEPTH]>,
    pub(crate) vote_comm_tree_position: Value<u32>,
    pub(crate) vpk_g_d: Value<pallas::Affine>,
    pub(crate) vpk_pk_d: Value<pallas::Affine>,
    /// Input VAN weight W (ballots).
    pub(crate) total_note_value: Value<pallas::Base>,
    /// Successor VAN weight W - w (ballots).
    pub(crate) remaining_note_value: Value<pallas::Base>,
    pub(crate) van_comm_rand: Value<pallas::Base>,
    pub(crate) vote_authority_note_old: Value<pallas::Base>,
    pub(crate) vsk: Value<pallas::Scalar>,
    pub(crate) rivk_v: Value<pallas::Scalar>,
    pub(crate) alpha_v: Value<pallas::Scalar>,
    pub(crate) vsk_nk: Value<pallas::Base>,
    pub(crate) shares: [Value<pallas::Base>; 16],
    pub(crate) enc_share_c1_x: [Value<pallas::Base>; 16],
    pub(crate) enc_share_c2_x: [Value<pallas::Base>; 16],
    pub(crate) enc_share_c1_y: [Value<pallas::Base>; 16],
    pub(crate) enc_share_c2_y: [Value<pallas::Base>; 16],
    pub(crate) share_blinds: [Value<pallas::Base>; 16],
    pub(crate) share_randomness: [Value<pallas::Base>; 16],
    pub(crate) ea_pk: Value<pallas::Affine>,
    /// Private delegate index (hidden inside DC).
    pub(crate) delegate_index: Value<pallas::Base>,
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

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), plonk::Error> {
        SinsemillaChip::load(config.sinsemilla_config.clone(), &mut layouter)?;
        let ecc_chip = config.ecc_chip();

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
            layouter.namespace(|| "witness W"),
            config.advices[0],
            self.total_note_value,
        )?;
        let remaining_note_value = assign_free_advice(
            layouter.namespace(|| "witness W_new"),
            config.advices[0],
            self.remaining_note_value,
        )?;
        let van_comm_rand = assign_free_advice(
            layouter.namespace(|| "witness van_comm_rand"),
            config.advices[0],
            self.van_comm_rand,
        )?;
        let vote_authority_note_old = assign_free_advice(
            layouter.namespace(|| "witness van_old"),
            config.advices[0],
            self.vote_authority_note_old,
        )?;
        let domain_van = config.constant(
            &mut layouter,
            "DOMAIN_VAN",
            pallas::Base::from(van_integrity::DOMAIN_VAN),
        )?;
        // C2/C6: authority is the MAX constant on both sides (baked into the VK).
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

        // ---- C2: old VAN integrity (authority == MAX) ----
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
            || "VAN integrity check",
            |mut region| region.constrain_equal(derived_van.cell(), vote_authority_note_old.cell()),
        )?;

        // ---- C3: address ownership ----
        let vsk_scalar = ScalarFixed::new(ecc_chip.clone(), layouter.namespace(|| "vsk"), self.vsk)?;
        let vsk_ak_point = spend_auth_g_mul(
            ecc_chip.clone(),
            layouter.namespace(|| "[vsk]G"),
            "c3: [vsk] SpendAuthG",
            vsk_scalar,
        )?;
        let ak = vsk_ak_point.extract_p().inner().clone();
        let rivk_v_scalar =
            ScalarFixed::new(ecc_chip.clone(), layouter.namespace(|| "rivk_v"), self.rivk_v)?;
        prove_address_ownership(
            config.sinsemilla_chip(),
            ecc_chip.clone(),
            config.commit_ivk_chip(),
            layouter.namespace(|| "c3 address"),
            "c3",
            ak,
            vsk_nk.clone(),
            rivk_v_scalar,
            &vpk_g_d_point,
            &vpk_pk_d_point,
        )?;

        // ---- C4: spend authority ----
        crate::gadgets::spend_authority::prove_spend_authority(
            ecc_chip.clone(),
            layouter.namespace(|| "c4 spend authority"),
            self.alpha_v,
            &vsk_ak_point,
            config.primary,
            R_VPK_X_PUBLIC_OFFSET,
            R_VPK_Y_PUBLIC_OFFSET,
        )?;

        // ---- C1: membership ----
        {
            let root = synthesize_poseidon_merkle_path::<VOTE_COMM_TREE_DEPTH>(
                &config.merkle_swap,
                &config.poseidon_config,
                &mut layouter,
                config.advices[0],
                vote_authority_note_old.clone(),
                self.vote_comm_tree_position,
                self.vote_comm_tree_path,
                "c1: merkle",
            )?;
            layouter.constrain_instance(
                root.cell(),
                config.primary,
                VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET,
            )?;
        }

        // ---- C5: VAN nullifier (identical derivation to ZKP #2) ----
        let domain_van_nf =
            config.constant(&mut layouter, "vote authority spend", domain_tags::vote_authority_spend())?;
        let van_nullifier = PoseidonHash::<
            pallas::Base,
            _,
            poseidon::P128Pow5T3,
            ConstantLength<4>,
            3,
            2,
        >::init(config.poseidon_chip(), layouter.namespace(|| "nf init"))?
        .hash(
            layouter.namespace(|| "Poseidon(nk, tag, round, van)"),
            [vsk_nk, domain_van_nf, voting_round_id.clone(), vote_authority_note_old],
        )?;
        layouter.constrain_instance(
            van_nullifier.cell(),
            config.primary,
            VAN_NULLIFIER_PUBLIC_OFFSET,
        )?;

        // ---- C6: successor VAN (same address/round/rand, MAX, weight W_new) ----
        let derived_van_new = van_integrity::van_integrity_poseidon(
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
            derived_van_new.cell(),
            config.primary,
            VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET,
        )?;

        // ---- C7: W_new in [0, 2^30) — anti-inflation (w <= W) ----
        config.range_check_b.copy_check(
            layouter.namespace(|| "W_new < 2^30"),
            remaining_note_value.clone(),
            SHARE_VALUE_RANGE_WORDS,
            true,
        )?;

        // ---- B1: NSLOTS public (delegate_index_j, Enc(w_j)) slots ----
        let share_cells: [_; 16] = (0..16usize)
            .map(|i| {
                assign_free_advice(
                    layouter.namespace(|| format!("witness w_{i}")),
                    config.advices[0],
                    if i < NSLOTS { self.shares[i] } else { Value::known(pallas::Base::zero()) },
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .expect("16");
        let shares_sum = share_cells[1..NSLOTS].iter().enumerate().try_fold(
            share_cells[0].clone(),
            |acc, (i, share)| {
                config
                    .add_chip()
                    .add(layouter.namespace(|| format!("sum step {}", i + 1)), &acc, share)
            },
        )?;
        let conserved = config.add_chip().add(
            layouter.namespace(|| "w + W_new"),
            &shares_sum,
            &remaining_note_value,
        )?;
        layouter.assign_region(
            || "w + W_new == W",
            |mut region| region.constrain_equal(conserved.cell(), total_note_value.cell()),
        )?;
        config
            .nonzero_b
            .constrain_nonzero(layouter.namespace(|| "w != 0"), "w != 0", &shares_sum)?;
        let wit16 = |f: &mut dyn FnMut(usize) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error>| -> Result<[AssignedCell<pallas::Base, pallas::Base>; 16], plonk::Error> {
            let v = (0..16).map(|i| f(i)).collect::<Result<Vec<_>, _>>()?;
            Ok(v.try_into().expect("16"))
        };
        let enc_c1 = wit16(&mut |i| assign_free_advice(layouter.namespace(|| format!("c1x {i}")), config.advices[0], self.enc_share_c1_x[i]))?;
        let enc_c2 = wit16(&mut |i| assign_free_advice(layouter.namespace(|| format!("c2x {i}")), config.advices[0], self.enc_share_c2_x[i]))?;
        let enc_c1_y = wit16(&mut |i| assign_free_advice(layouter.namespace(|| format!("c1y {i}")), config.advices[0], self.enc_share_c1_y[i]))?;
        let enc_c2_y = wit16(&mut |i| assign_free_advice(layouter.namespace(|| format!("c2y {i}")), config.advices[0], self.enc_share_c2_y[i]))?;
        for j in 0..NSLOTS {
            let base = SLOT_BASE + 5 * j;
            // slot j: [delegate_index (transcript-bound), c1x, c1y, c2x, c2y]
            layouter.constrain_instance(enc_c1[j].cell(), config.primary, base + 1)?;
            layouter.constrain_instance(enc_c1_y[j].cell(), config.primary, base + 2)?;
            layouter.constrain_instance(enc_c2[j].cell(), config.primary, base + 3)?;
            layouter.constrain_instance(enc_c2_y[j].cell(), config.primary, base + 4)?;
        }
        {
            let r_cells: [_; 16] = (0..16usize)
                .map(|i| {
                    let column = if i < SPLIT { config.elgamal_advices[0] } else { config.elgamal_advices_b[0] };
                    assign_free_advice(layouter.namespace(|| format!("r[{i}]")), column, self.share_randomness[i])
                })
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .expect("16");
            let loc = || EaPkInstanceLoc { instance: config.primary, x_row: EA_PK_X_PUBLIC_OFFSET, y_row: EA_PK_Y_PUBLIC_OFFSET };
            prove_elgamal_encryptions(config.elgamal_ecc_chip(), config.nonzero, &config.spend_auth_g_fixed_base_30,
                layouter.namespace(|| "b1a"), "b1a", 0..SPLIT, self.ea_pk, loc(),
                share_cells.clone(), r_cells.clone(), enc_c1.clone(), enc_c2.clone(), enc_c1_y.clone(), enc_c2_y.clone())?;
            prove_elgamal_encryptions(config.elgamal_ecc_chip_b(), config.nonzero_b, &config.spend_auth_g_fixed_base_30_b,
                layouter.namespace(|| "b1b"), "b1b", SPLIT..NSLOTS, self.ea_pk, loc(),
                share_cells, r_cells, enc_c1, enc_c2, enc_c1_y, enc_c2_y)?;
        }
        Ok(())
    }
}

/// Public inputs (10 field elements).
#[derive(Clone, Debug)]
pub struct Instance {
    pub van_nullifier: pallas::Base,
    pub r_vpk_x: pallas::Base,
    pub r_vpk_y: pallas::Base,
    pub vote_authority_note_new: pallas::Base,
    pub delegation_commitment: pallas::Base,
    pub vote_comm_tree_root: pallas::Base,
    pub vote_comm_tree_anchor_height: pallas::Base,
    pub voting_round_id: pallas::Base,
    pub ea_pk_x: pallas::Base,
    pub ea_pk_y: pallas::Base,
}

impl Instance {
    /// Halo2 ordering.
    pub fn to_halo2_instance(&self) -> Vec<vesta::Scalar> {
        vec![
            self.van_nullifier,
            self.r_vpk_x,
            self.r_vpk_y,
            self.vote_authority_note_new,
            self.delegation_commitment,
            self.vote_comm_tree_root,
            self.vote_comm_tree_anchor_height,
            self.voting_round_id,
            self.ea_pk_x,
            self.ea_pk_y,
        ]
    }
}

#[cfg(test)]
mod tests_b1;
