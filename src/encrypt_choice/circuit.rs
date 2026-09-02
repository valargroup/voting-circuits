//! The Encrypt-Choice circuit implementation (ZKP 1.5).
//!
//! Proves that a voter's 16 weight shares are each encrypted into all
//! [`MAX_DECISION_BUCKETS`] (8) ElGamal decision buckets under the
//! governance-announced election-authority key, with the voter's private
//! decision selecting which bucket of each share encrypts the weight (all
//! other buckets encrypt zero). The circuit verifies five conditions:
//!
//! - **Condition 1**: One-Hot Decision — the private selector vector is
//!   boolean and sums to one.
//! - **Condition 2**: Active-Bucket Confinement — witnessed boolean prefix
//!   flags are non-increasing, sum to the public `decision_bucket_count`, and
//!   the selector is zero in every inactive bucket. Together with condition 1
//!   this proves the private decision is in `0..D` without changing circuit
//!   shape.
//! - **Condition 3**: Encryption Integrity — for every `(bucket j, share i)`:
//!   `r_ij != 0`, `C1_ij = [r_ij]G`, `R_ij = [r_ij]PK`, `S_ij = R_ij + W_i`
//!   (complete addition), where `W_i = [w_i]G` is the shared weight point
//!   computed once per share via the 30-bit fixed-base gadget (which also
//!   range-checks `w_i < 2^30`). The selector then chooses the published C2
//!   coordinates: `C2_ij = R_ij + b_j·(S_ij − R_ij)`.
//! - **Condition 4**: Selected Commitments — for every share, the wide
//!   Poseidon commitment over its blind and all bucket ciphertext
//!   coordinates ([`crate::bridge::selected_share_commitment`]).
//! - **Condition 5**: Bridge Integrity — the compact bridge commitment over
//!   the round, proposal, bucket count, and every `(weight, commitment)`
//!   pair ([`crate::bridge::bridge_commitment`]), bound to the public
//!   instance. ZKP #2 re-opens the same bridge, binding both proofs to the
//!   same weights and ciphertexts.
//!
//! Authoritative hash sources: `crate::bridge` owns the selected-commitment
//! and bridge preimages; `crate::domain_tags` owns the tag encodings.
//!
//! ## Layout
//!
//! The fully parallel ("maxpar") decision-bound layout: 16 round-robin ECC
//! tracks (10 advice columns each) carry eight of the `16 × 8` ciphertext
//! jobs apiece, and 17 Poseidon tracks give each of the 16 wide
//! selected-commitment hashes and the bridge its own track. K = 11 — the
//! same ring size as every other vote circuit — traded for a wide proving
//! key.

use std::vec::Vec;

use voting_crypto_deps::halo2_gadgets::{
    ecc::{
        chip::{EccChip, EccConfig},
        CircuitVersion, FixedPointBaseField, NonIdentityPoint, ScalarVar,
    },
    poseidon::{primitives as poseidon, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig},
    utilities::lookup_range_check::{LookupRangeCheck, LookupRangeCheckConfig},
};
use voting_crypto_deps::halo2_proofs::{
    circuit::{floor_planner, AssignedCell, Layouter, Value},
    plonk::{
        self, Advice, Column, ConstraintSystem, Constraints, Expression, Fixed,
        Instance as InstanceColumn, Selector, TableColumn,
    },
    poly::Rotation,
};
use voting_crypto_deps::orchard::constants::{OrchardBaseFieldBases, OrchardFixedBases};
use voting_crypto_deps::pasta_curves::{pallas, vesta};

use crate::{
    bridge::{
        compute_bridge_in_circuit, hash_selected_commitment_in_circuit, MAX_DECISION_BUCKETS,
        NUM_SHARES,
    },
    gadgets::{elgamal::SpendAuthGFixedBase30Config, nonzero::NonZeroConfig},
};

// ================================================================
// Constants
// ================================================================

/// Circuit size (2^K rows).
///
/// K=11 for the fully parallel 16-ECC-track layout: each track carries
/// `16 × 8 / 16 = 8` ciphertext jobs of ~230 rows each, fitting under the
/// 1,024-row lookup table. `CircuitCost::measure` reports a 1,854-row
/// high-water mark with 231 advice / 447 fixed / 679 total columns and a
/// measured proof of ~58 KiB — the columns, not the rows, dominate keygen,
/// proof size, and proving memory.
///
/// Run the `row_budget` test to re-measure after circuit changes:
///   `cargo test encrypt_choice -- --nocapture --ignored row_budget`
pub const K: u32 = 11;

/// Round-robin ECC tracks carrying the per-(bucket, share) ciphertext jobs.
const ECC_TRACKS: usize = 16;

/// Parallel Poseidon tracks: one per selected commitment
/// (`share % HASH_TRACKS` covers tracks 0–15) plus a dedicated bridge track
/// (`NUM_SHARES % HASH_TRACKS = 16`).
const HASH_TRACKS: usize = 17;

// ================================================================
// Public input offsets (6 field elements).
// ================================================================

/// Public input offset for the election-authority public key x-coordinate.
pub(crate) const EA_PK_X_PUBLIC_OFFSET: usize = 0;
/// Public input offset for the election-authority public key y-coordinate.
///
/// Both coordinates are public to prevent sign-ambiguity attacks (using
/// −ea_pk would corrupt the tally).
pub(crate) const EA_PK_Y_PUBLIC_OFFSET: usize = 1;
/// Public input offset for the compact bridge commitment shared with ZKP #2.
pub(crate) const BRIDGE_PUBLIC_OFFSET: usize = 2;
/// Public input offset for the active decision bucket count `D`.
pub(crate) const DECISION_BUCKET_COUNT_PUBLIC_OFFSET: usize = 3;
/// Public input offset for the voting round identifier.
pub(crate) const VOTING_ROUND_ID_PUBLIC_OFFSET: usize = 4;
/// Public input offset for the proposal identifier.
pub(crate) const PROPOSAL_ID_PUBLIC_OFFSET: usize = 5;

// ================================================================
// Instance
// ================================================================

/// Typed public inputs for the encrypt-choice circuit.
///
/// Field order equals the Halo2 instance-column order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instance {
    /// Election-authority public key x-coordinate (offset 0). Must be
    /// authenticated against the governance-announced round key.
    pub ea_pk_x: pallas::Base,
    /// Election-authority public key y-coordinate (offset 1).
    pub ea_pk_y: pallas::Base,
    /// Compact bridge commitment (offset 2). Must equal the cast proof's
    /// public bridge value in the same vote bundle.
    pub bridge: pallas::Base,
    /// Active decision bucket count `D` (offset 3). Must be authenticated
    /// from the proposal's governance declaration; the circuit constrains
    /// `1 <= D <= 8` structurally (prefix flags summing to `D` with a
    /// one-hot selector inside), and the builder and verifier enforce
    /// `D >= 2`.
    pub decision_bucket_count: pallas::Base,
    /// Voting round identifier (offset 4).
    pub voting_round_id: pallas::Base,
    /// Proposal identifier (offset 5).
    pub proposal_id: pallas::Base,
}

impl Instance {
    /// Number of public inputs in the instance column.
    pub const NUM_PUBLIC_INPUTS: usize = 6;

    /// Builds an instance from its parts, in instance-column order.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        ea_pk_x: pallas::Base,
        ea_pk_y: pallas::Base,
        bridge: pallas::Base,
        decision_bucket_count: pallas::Base,
        voting_round_id: pallas::Base,
        proposal_id: pallas::Base,
    ) -> Self {
        Self {
            ea_pk_x,
            ea_pk_y,
            bridge,
            decision_bucket_count,
            voting_round_id,
            proposal_id,
        }
    }

    /// Serializes the instance in Halo2 instance-column order.
    pub fn to_halo2_instance(&self) -> Vec<vesta::Scalar> {
        vec![
            self.ea_pk_x,
            self.ea_pk_y,
            self.bridge,
            self.decision_bucket_count,
            self.voting_round_id,
            self.proposal_id,
        ]
    }
}

// ================================================================
// Config
// ================================================================

/// One round-robin ECC track: full Orchard ECC chip, non-zero gate, 30-bit
/// fixed-base gadget, and the C2 selection gate on ten shared advice columns.
#[derive(Clone, Debug)]
struct EncryptChoiceEccTrack {
    advices: [Column<Advice>; 10],
    ecc: EccConfig<OrchardFixedBases>,
    nonzero: NonZeroConfig,
    fixed_base_30: SpendAuthGFixedBase30Config,
    q_select_c2: Selector,
}

/// Configuration for the encrypt-choice circuit.
#[derive(Clone, Debug)]
pub struct Config {
    primary: Column<InstanceColumn>,
    work: [Column<Advice>; 3],
    table_idx: TableColumn,
    q_selector: Selector,
    q_active: Selector,
    q_active_mono: Selector,
    tracks: Vec<EncryptChoiceEccTrack>,
    poseidon: Vec<PoseidonConfig<pallas::Base, 3, 2>>,
}

// ================================================================
// Circuit
// ================================================================

/// The encrypt-choice circuit (ZKP 1.5).
///
/// All witnesses are private; see the module documentation for the proven
/// conditions and [`Instance`] for the public inputs.
#[derive(Clone, Debug)]
pub struct Circuit {
    /// The 16 weight shares `w_i` (each in `[0, 2^30)`).
    pub(crate) shares: [Value<pallas::Base>; NUM_SHARES],
    /// The 16 per-share commitment blinds.
    pub(crate) blinds: [Value<pallas::Base>; NUM_SHARES],
    /// ElGamal randomness, indexed `[bucket][share]`.
    pub(crate) randomness: [[Value<pallas::Base>; NUM_SHARES]; MAX_DECISION_BUCKETS],
    /// One-hot decision selector per bucket.
    pub(crate) selectors: [Value<pallas::Base>; MAX_DECISION_BUCKETS],
    /// Boolean active-bucket prefix flags (1 for buckets `< D`).
    pub(crate) active: [Value<pallas::Base>; MAX_DECISION_BUCKETS],
    /// Election-authority public key.
    pub(crate) ea_pk: Value<pallas::Affine>,
}

impl Default for Circuit {
    fn default() -> Self {
        Self {
            shares: [Value::unknown(); NUM_SHARES],
            blinds: [Value::unknown(); NUM_SHARES],
            randomness: [[Value::unknown(); NUM_SHARES]; MAX_DECISION_BUCKETS],
            selectors: [Value::unknown(); MAX_DECISION_BUCKETS],
            active: [Value::unknown(); MAX_DECISION_BUCKETS],
            ea_pk: Value::unknown(),
        }
    }
}

fn load_range_table(
    mut layouter: impl Layouter<pallas::Base>,
    table: TableColumn,
) -> Result<(), plonk::Error> {
    layouter.assign_table(
        || "10-bit range table",
        |mut table_region| {
            for value in 0..(1usize << 10) {
                table_region.assign_cell(
                    || "range value",
                    table,
                    value,
                    || Value::known(pallas::Base::from(value as u64)),
                )?;
            }
            Ok(())
        },
    )
}

impl plonk::Circuit<pallas::Base> for Circuit {
    type Config = Config;
    type FloorPlanner = floor_planner::V1;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        let primary = meta.instance_column();
        meta.enable_equality(primary);
        let work: [Column<Advice>; 3] = core::array::from_fn(|_| meta.advice_column());
        for column in work {
            meta.enable_equality(column);
        }
        let constants = meta.fixed_column();
        meta.enable_constant(constants);
        let table_idx = meta.lookup_table_column();

        // Condition 1: one-hot selector (boolean, running sum → 1).
        let q_selector = meta.selector();
        meta.create_gate("encrypt-choice one-hot selector", |meta| {
            let q = meta.query_selector(q_selector);
            let selector = meta.query_advice(work[0], Rotation::cur());
            let sum = meta.query_advice(work[1], Rotation::cur());
            let next_sum = meta.query_advice(work[1], Rotation::next());
            let one = Expression::Constant(pallas::Base::one());
            Constraints::with_selector(
                q,
                [
                    (
                        "selector is boolean",
                        selector.clone() * (one - selector.clone()),
                    ),
                    ("selector running sum", next_sum - sum - selector),
                ],
            )
        });

        // Condition 2: boolean prefix flags confining the selector to the
        // active buckets, with the running sum bound to the public D.
        let q_active = meta.selector();
        meta.create_gate("encrypt-choice active buckets", |meta| {
            let q = meta.query_selector(q_active);
            let selector = meta.query_advice(work[0], Rotation::cur());
            let active = meta.query_advice(work[1], Rotation::cur());
            let sum = meta.query_advice(work[2], Rotation::cur());
            let next_sum = meta.query_advice(work[2], Rotation::next());
            let one = Expression::Constant(pallas::Base::one());
            Constraints::with_selector(
                q,
                [
                    (
                        "active is boolean",
                        active.clone() * (one.clone() - active.clone()),
                    ),
                    (
                        "selector only in active buckets",
                        selector * (one - active.clone()),
                    ),
                    ("active running sum", next_sum - sum - active),
                ],
            )
        });
        let q_active_mono = meta.selector();
        meta.create_gate("encrypt-choice active prefix", |meta| {
            let q = meta.query_selector(q_active_mono);
            let active = meta.query_advice(work[1], Rotation::cur());
            let next_active = meta.query_advice(work[1], Rotation::next());
            let one = Expression::Constant(pallas::Base::one());
            Constraints::with_selector(q, [("no reactivation", next_active * (one - active))])
        });

        // Condition 3 tracks: ECC + non-zero + 30-bit fixed-base + C2 select.
        let mut tracks = Vec::with_capacity(ECC_TRACKS);
        for _ in 0..ECC_TRACKS {
            let advices: [Column<Advice>; 10] = core::array::from_fn(|_| meta.advice_column());
            for column in advices {
                meta.enable_equality(column);
            }
            let range = LookupRangeCheckConfig::configure(meta, advices[9], table_idx);
            let lagrange: [Column<Fixed>; 8] = core::array::from_fn(|_| meta.fixed_column());
            let ecc = EccChip::<OrchardFixedBases>::configure(meta, advices, lagrange, range);
            let fixed_base_30 = SpendAuthGFixedBase30Config::configure(meta, advices, lagrange);
            let nonzero = NonZeroConfig::configure(meta, [advices[0], advices[1]]);
            let q_select_c2 = meta.selector();
            meta.create_gate("encrypt-choice C2 selection", |meta| {
                let q = meta.query_selector(q_select_c2);
                let selector = meta.query_advice(advices[0], Rotation::cur());
                let rx = meta.query_advice(advices[1], Rotation::cur());
                let ry = meta.query_advice(advices[2], Rotation::cur());
                let sx = meta.query_advice(advices[3], Rotation::cur());
                let sy = meta.query_advice(advices[4], Rotation::cur());
                let c2x = meta.query_advice(advices[5], Rotation::cur());
                let c2y = meta.query_advice(advices[6], Rotation::cur());
                Constraints::with_selector(
                    q,
                    [
                        (
                            "select C2 x",
                            c2x - rx.clone() - selector.clone() * (sx - rx),
                        ),
                        ("select C2 y", c2y - ry.clone() - selector * (sy - ry)),
                    ],
                )
            });
            tracks.push(EncryptChoiceEccTrack {
                advices,
                ecc,
                nonzero,
                fixed_base_30,
                q_select_c2,
            });
        }

        // Conditions 4–5 tracks: parallel Poseidon configurations.
        let mut poseidon = Vec::with_capacity(HASH_TRACKS);
        for _ in 0..HASH_TRACKS {
            let advices: [Column<Advice>; 4] = core::array::from_fn(|_| meta.advice_column());
            let fixed: [Column<Fixed>; 6] = core::array::from_fn(|_| meta.fixed_column());
            poseidon.push(PoseidonChip::configure::<poseidon::P128Pow5T3>(
                meta,
                advices[..3].try_into().expect("three state columns"),
                advices[3],
                fixed[..3].try_into().expect("three first-round columns"),
                fixed[3..].try_into().expect("three second-round columns"),
            ));
        }

        Config {
            primary,
            work,
            table_idx,
            q_selector,
            q_active,
            q_active_mono,
            tracks,
            poseidon,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), plonk::Error> {
        load_range_table(layouter.namespace(|| "range table"), config.table_idx)?;

        // ---- Condition 1: one-hot selector ----
        let selector_cells = layouter.assign_region(
            || "private decision",
            |mut region| {
                region.assign_advice_from_constant(
                    || "initial selector sum",
                    config.work[1],
                    0,
                    pallas::Base::zero(),
                )?;
                let mut cells = Vec::with_capacity(MAX_DECISION_BUCKETS);
                let mut running = Value::known(pallas::Base::zero());
                let mut final_sum = None;
                for bucket in 0..MAX_DECISION_BUCKETS {
                    config.q_selector.enable(&mut region, bucket)?;
                    cells.push(region.assign_advice(
                        || format!("selector[{bucket}]"),
                        config.work[0],
                        bucket,
                        || self.selectors[bucket],
                    )?);
                    running = running + self.selectors[bucket];
                    final_sum = Some(region.assign_advice(
                        || format!("selector sum through {bucket}"),
                        config.work[1],
                        bucket + 1,
                        || running,
                    )?);
                }
                region.constrain_constant(
                    final_sum.expect("at least two buckets").cell(),
                    pallas::Base::one(),
                )?;
                Ok(cells)
            },
        )?;

        // ---- Condition 2: active-bucket confinement ----
        let active_sum = layouter.assign_region(
            || "active bucket confinement",
            |mut region| {
                region.assign_advice_from_constant(
                    || "initial active sum",
                    config.work[2],
                    0,
                    pallas::Base::zero(),
                )?;
                let mut running = Value::known(pallas::Base::zero());
                let mut final_sum = None;
                for bucket in 0..MAX_DECISION_BUCKETS {
                    config.q_active.enable(&mut region, bucket)?;
                    if bucket + 1 < MAX_DECISION_BUCKETS {
                        config.q_active_mono.enable(&mut region, bucket)?;
                    }
                    selector_cells[bucket].copy_advice(
                        || "selector",
                        &mut region,
                        config.work[0],
                        bucket,
                    )?;
                    region.assign_advice(
                        || format!("active[{bucket}]"),
                        config.work[1],
                        bucket,
                        || self.active[bucket],
                    )?;
                    running = running + self.active[bucket];
                    final_sum = Some(region.assign_advice(
                        || format!("active sum through {bucket}"),
                        config.work[2],
                        bucket + 1,
                        || running,
                    )?);
                }
                Ok(final_sum.expect("at least two buckets"))
            },
        )?;
        layouter.constrain_instance(
            active_sum.cell(),
            config.primary,
            DECISION_BUCKET_COUNT_PUBLIC_OFFSET,
        )?;

        // ---- Shares and blinds ----
        let (share_cells, blind_cells) = layouter.assign_region(
            || "share witnesses",
            |mut region| {
                let mut shares = Vec::with_capacity(NUM_SHARES);
                let mut blinds = Vec::with_capacity(NUM_SHARES);
                for share in 0..NUM_SHARES {
                    shares.push(region.assign_advice(
                        || format!("share[{share}]"),
                        config.work[0],
                        share,
                        || self.shares[share],
                    )?);
                    blinds.push(region.assign_advice(
                        || format!("blind[{share}]"),
                        config.work[1],
                        share,
                        || self.blinds[share],
                    )?);
                }
                Ok((shares, blinds))
            },
        )?;
        let share_cells: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES] =
            share_cells.try_into().expect("sixteen shares");
        let blind_cells: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES] =
            blind_cells.try_into().expect("sixteen blinds");

        // ---- Condition 3: W_i = [w_i]G (also range-checks w_i < 2^30) ----
        let mut weight_points = Vec::with_capacity(NUM_SHARES);
        for share in 0..NUM_SHARES {
            let track = &config.tracks[share % ECC_TRACKS];
            weight_points.push(track.fixed_base_30.mul(
                layouter.namespace(|| format!("W[{share}] = [share]G")),
                &share_cells[share],
            )?);
        }

        // ---- Condition 3: per-(bucket, share) ciphertexts, round-robin ----
        let mut c1_x = vec![vec![None; NUM_SHARES]; MAX_DECISION_BUCKETS];
        let mut c1_y = vec![vec![None; NUM_SHARES]; MAX_DECISION_BUCKETS];
        let mut c2_x = vec![vec![None; NUM_SHARES]; MAX_DECISION_BUCKETS];
        let mut c2_y = vec![vec![None; NUM_SHARES]; MAX_DECISION_BUCKETS];
        for track_index in 0..ECC_TRACKS {
            let track = &config.tracks[track_index];
            let ecc_chip = EccChip::construct(track.ecc.clone(), CircuitVersion::AnchoredBase);
            let ea_pk = NonIdentityPoint::new(
                ecc_chip.clone(),
                layouter.namespace(|| format!("track {track_index} EA key")),
                self.ea_pk,
            )?;
            layouter.constrain_instance(
                ea_pk.inner().x().cell(),
                config.primary,
                EA_PK_X_PUBLIC_OFFSET,
            )?;
            layouter.constrain_instance(
                ea_pk.inner().y().cell(),
                config.primary,
                EA_PK_Y_PUBLIC_OFFSET,
            )?;
            let spend_auth_g = FixedPointBaseField::from_inner(
                ecc_chip.clone(),
                OrchardBaseFieldBases::SpendAuthGBase,
            );

            for job in (track_index..NUM_SHARES * MAX_DECISION_BUCKETS).step_by(ECC_TRACKS) {
                let bucket = job / NUM_SHARES;
                let share = job % NUM_SHARES;
                let r_cell = layouter.assign_region(
                    || format!("randomness bucket {bucket} share {share}"),
                    |mut region| {
                        region.assign_advice(
                            || "r",
                            track.advices[0],
                            0,
                            || self.randomness[bucket][share],
                        )
                    },
                )?;
                track.nonzero.constrain_nonzero(
                    layouter.namespace(|| format!("r[{bucket}][{share}] != 0")),
                    "encrypt-choice ElGamal randomness != 0",
                    &r_cell,
                )?;
                let c1 = spend_auth_g.clone().mul(
                    layouter.namespace(|| format!("C1[{bucket}][{share}]")),
                    r_cell.clone(),
                )?;
                c1_x[bucket][share] = Some(c1.inner().x());
                c1_y[bucket][share] = Some(c1.inner().y());

                let r_scalar = ScalarVar::from_base(
                    ecc_chip.clone(),
                    layouter.namespace(|| format!("r[{bucket}][{share}] scalar")),
                    &r_cell,
                )?;
                let (r_pk, _) = ea_pk.mul(
                    layouter.namespace(|| format!("R[{bucket}][{share}]")),
                    r_scalar,
                )?;
                let rx = r_pk.inner().x();
                let ry = r_pk.inner().y();
                let (sx, sy) = track.fixed_base_30.add(
                    layouter.namespace(|| format!("S[{bucket}][{share}]")),
                    &rx,
                    &ry,
                    &weight_points[share].0,
                    &weight_points[share].1,
                )?;
                let (selected_x, selected_y) = layouter.assign_region(
                    || format!("select C2[{bucket}][{share}]"),
                    |mut region| {
                        track.q_select_c2.enable(&mut region, 0)?;
                        selector_cells[bucket].copy_advice(
                            || "selector",
                            &mut region,
                            track.advices[0],
                            0,
                        )?;
                        rx.copy_advice(|| "R x", &mut region, track.advices[1], 0)?;
                        ry.copy_advice(|| "R y", &mut region, track.advices[2], 0)?;
                        sx.copy_advice(|| "S x", &mut region, track.advices[3], 0)?;
                        sy.copy_advice(|| "S y", &mut region, track.advices[4], 0)?;
                        let selected_x_value = self.selectors[bucket]
                            .zip(rx.value().copied())
                            .zip(sx.value().copied())
                            .map(|((selector, r), s)| r + selector * (s - r));
                        let selected_y_value = self.selectors[bucket]
                            .zip(ry.value().copied())
                            .zip(sy.value().copied())
                            .map(|((selector, r), s)| r + selector * (s - r));
                        let x = region.assign_advice(
                            || "selected C2 x",
                            track.advices[5],
                            0,
                            || selected_x_value,
                        )?;
                        let y = region.assign_advice(
                            || "selected C2 y",
                            track.advices[6],
                            0,
                            || selected_y_value,
                        )?;
                        Ok((x, y))
                    },
                )?;
                c2_x[bucket][share] = Some(selected_x);
                c2_y[bucket][share] = Some(selected_y);
            }
        }

        // ---- Condition 4: selected commitments (shared bridge gadget) ----
        let coord = |cells: &Vec<Vec<Option<AssignedCell<pallas::Base, pallas::Base>>>>,
                     bucket: usize,
                     share: usize| {
            cells[bucket][share]
                .clone()
                .expect("every (bucket, share) job was scheduled")
        };
        let mut selected_commitments = Vec::with_capacity(NUM_SHARES);
        for share in 0..NUM_SHARES {
            let mut coords = Vec::with_capacity(4 * MAX_DECISION_BUCKETS);
            for bucket in 0..MAX_DECISION_BUCKETS {
                coords.extend([
                    coord(&c1_x, bucket, share),
                    coord(&c2_x, bucket, share),
                    coord(&c1_y, bucket, share),
                    coord(&c2_y, bucket, share),
                ]);
            }
            let coords: [AssignedCell<pallas::Base, pallas::Base>; 4 * MAX_DECISION_BUCKETS] =
                coords.try_into().expect("coordinate count is fixed");
            selected_commitments.push(hash_selected_commitment_in_circuit(
                PoseidonChip::construct(config.poseidon[share % HASH_TRACKS].clone()),
                layouter.namespace(|| format!("selected commitment {share}")),
                config.work[0],
                blind_cells[share].clone(),
                coords,
                share,
            )?);
        }
        let selected_commitments: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES] =
            selected_commitments
                .try_into()
                .expect("sixteen selected commitments");

        // ---- Condition 5: compact bridge (shared bridge gadget) ----
        let (round, proposal, bucket_count) = layouter.assign_region(
            || "bridge context",
            |mut region| {
                let round = region.assign_advice_from_instance(
                    || "round",
                    config.primary,
                    VOTING_ROUND_ID_PUBLIC_OFFSET,
                    config.work[1],
                    0,
                )?;
                let proposal = region.assign_advice_from_instance(
                    || "proposal",
                    config.primary,
                    PROPOSAL_ID_PUBLIC_OFFSET,
                    config.work[2],
                    0,
                )?;
                let bucket_count = region.assign_advice_from_instance(
                    || "bucket count",
                    config.primary,
                    DECISION_BUCKET_COUNT_PUBLIC_OFFSET,
                    config.work[0],
                    0,
                )?;
                Ok((round, proposal, bucket_count))
            },
        )?;
        let bridge = compute_bridge_in_circuit(
            PoseidonChip::construct(config.poseidon[NUM_SHARES % HASH_TRACKS].clone()),
            layouter.namespace(|| "bridge"),
            config.work[0],
            round,
            proposal,
            bucket_count,
            share_cells,
            selected_commitments,
        )?;
        layouter.constrain_instance(bridge.cell(), config.primary, BRIDGE_PUBLIC_OFFSET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voting_crypto_deps::halo2_proofs::dev::MockProver;

    use crate::encrypt_choice::builder::{assemble_encrypt_choice, EncryptChoiceAssembly};
    use crate::gadgets::elgamal::spend_auth_g_affine;
    use voting_crypto_deps::orchard::keys::SpendingKey;

    fn test_ea_pk() -> pallas::Affine {
        use crate::group::Curve;
        (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine()
    }

    fn valid_assembly(
        decision: u64,
        decision_bucket_count: u64,
        single_share: bool,
    ) -> EncryptChoiceAssembly {
        let sk = SpendingKey::from_bytes([0x42; 32]).expect("valid test spending key");
        assemble_encrypt_choice(
            &sk,
            // 12,345 ballots exercises denominations, remainder spread, and
            // (via the shuffle) zero-valued shares.
            12_345 * crate::params::BALLOT_DIVISOR,
            pallas::Base::from(0xDEAD_u64),
            pallas::Base::from(0xCAFE_u64),
            3,
            decision,
            decision_bucket_count,
            test_ea_pk(),
            single_share,
        )
        .expect("valid witness assembly")
    }

    fn assert_satisfied(assembly: &EncryptChoiceAssembly) {
        let prover = MockProver::run(
            K,
            &assembly.circuit,
            vec![assembly.instance.to_halo2_instance()],
        )
        .expect("mock prover runs");
        assert_eq!(prover.verify(), Ok(()));
    }

    fn assert_unsatisfied(circuit: &Circuit, instance: &Instance) {
        let prover = MockProver::run(K, circuit, vec![instance.to_halo2_instance()])
            .expect("mock prover runs");
        assert_ne!(prover.verify(), Ok(()));
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn accepts_first_middle_and_last_active_decisions() {
        for decision in [
            0,
            MAX_DECISION_BUCKETS as u64 / 2,
            MAX_DECISION_BUCKETS as u64 - 1,
        ] {
            assert_satisfied(&valid_assembly(
                decision,
                MAX_DECISION_BUCKETS as u64,
                false,
            ));
        }
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn accepts_boundary_bucket_counts() {
        assert_satisfied(&valid_assembly(1, 2, false));
        assert_satisfied(&valid_assembly(0, 2, false));
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn accepts_single_share_layout() {
        assert_satisfied(&valid_assembly(2, 5, true));
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn rejects_non_one_hot_selectors() {
        let mut assembly = valid_assembly(0, 4, false);
        // Two selected buckets.
        assembly.circuit.selectors[1] = Value::known(pallas::Base::one());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);

        // No selected bucket.
        let mut assembly = valid_assembly(0, 4, false);
        assembly.circuit.selectors[0] = Value::known(pallas::Base::zero());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);

        // Non-boolean selector "sum" trick: 2 in one bucket, -1 in another.
        let mut assembly = valid_assembly(0, 4, false);
        assembly.circuit.selectors[0] = Value::known(pallas::Base::one() + pallas::Base::one());
        assembly.circuit.selectors[1] = Value::known(-pallas::Base::one());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn rejects_selector_in_inactive_bucket() {
        let mut assembly = valid_assembly(0, 4, false);
        // Move the selection into bucket 5 >= D = 4.
        assembly.circuit.selectors[0] = Value::known(pallas::Base::zero());
        assembly.circuit.selectors[5] = Value::known(pallas::Base::one());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn rejects_non_prefix_or_miscounted_active_flags() {
        // Reactivation after a gap.
        let mut assembly = valid_assembly(0, 4, false);
        assembly.circuit.active[4] = Value::known(pallas::Base::zero());
        assembly.circuit.active[5] = Value::known(pallas::Base::one());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);

        // Active sum differs from the public bucket count.
        let mut assembly = valid_assembly(0, 4, false);
        assembly.circuit.active[4] = Value::known(pallas::Base::one());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn rejects_zero_randomness() {
        let mut assembly = valid_assembly(1, 4, false);
        assembly.circuit.randomness[2][7] = Value::known(pallas::Base::zero());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn rejects_wrong_or_negated_ea_key() {
        use crate::group::Curve;

        // A different key than the instance.
        let mut assembly = valid_assembly(1, 4, false);
        assembly.circuit.ea_pk =
            Value::known((spend_auth_g_affine() * pallas::Scalar::from(43u64)).to_affine());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);

        // The negated key (same x, flipped y) must also fail: both
        // coordinates are pinned to the instance.
        let mut assembly = valid_assembly(1, 4, false);
        assembly.circuit.ea_pk =
            Value::known((-(spend_auth_g_affine() * pallas::Scalar::from(42u64))).to_affine());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn rejects_out_of_range_share() {
        let mut assembly = valid_assembly(1, 4, false);
        // 2^30 fails the 30-bit fixed-base decomposition of W_i.
        assembly.circuit.shares[0] =
            Value::known(pallas::Base::from(crate::params::SHARE_VALUE_LIMIT));
        assert_unsatisfied(&assembly.circuit, &assembly.instance);
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn rejects_altered_share_blind_or_weight() {
        // Altering a blind changes the selected commitment, so the derived
        // bridge no longer matches the public instance.
        let mut assembly = valid_assembly(1, 4, false);
        assembly.circuit.blinds[3] = Value::known(pallas::Base::from(999u64));
        assert_unsatisfied(&assembly.circuit, &assembly.instance);

        // Altering a weight changes both W_i and the bridge preimage.
        let mut assembly = valid_assembly(1, 4, false);
        assembly.circuit.shares[3] = assembly.circuit.shares[3].map(|s| s + pallas::Base::one());
        assert_unsatisfied(&assembly.circuit, &assembly.instance);
    }

    #[test]
    #[ignore = "expensive wide K=11 MockProver synthesis; run when touching the encrypt-choice circuit"]
    fn rejects_replayed_context() {
        // A proof assembled for (round, proposal, D) must not verify under a
        // different round, proposal, bucket count, or bridge value.
        let assembly = valid_assembly(1, 4, false);

        let mut wrong_round = assembly.instance;
        wrong_round.voting_round_id += pallas::Base::one();
        assert_unsatisfied(&assembly.circuit, &wrong_round);

        let mut wrong_proposal = assembly.instance;
        wrong_proposal.proposal_id += pallas::Base::one();
        assert_unsatisfied(&assembly.circuit, &wrong_proposal);

        let mut wrong_count = assembly.instance;
        wrong_count.decision_bucket_count += pallas::Base::one();
        assert_unsatisfied(&assembly.circuit, &wrong_count);

        let mut wrong_bridge = assembly.instance;
        wrong_bridge.bridge += pallas::Base::one();
        assert_unsatisfied(&assembly.circuit, &wrong_bridge);
    }

    #[test]
    #[ignore = "diagnostic; run with --nocapture to inspect the row budget"]
    fn row_budget() {
        use voting_crypto_deps::halo2_proofs::dev::CircuitCost;

        let assembly = valid_assembly(1, 4, false);
        let cost = CircuitCost::<vesta::Point, Circuit>::measure(K, &assembly.circuit);
        println!("encrypt-choice circuit cost at K={K}:\n{:#?}", cost);
    }
}
