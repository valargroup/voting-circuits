//! IMT non-membership circuit gates and synthesis (condition 13).
//!
//! K=2 punctured-range model: each leaf stores `[nf_lo, nf_mid, nf_hi]` and
//! the leaf hash is `Poseidon3(nf_lo, nf_mid, nf_hi)` (ConstantLength<3>).
//!
//! Non-membership is proven by:
//! - Strict interval: `nf_lo < value < nf_hi` (via two offset range checks)
//! - Non-equality: `value != nf_mid` (via inverse witness)
//! - Merkle path to the committed root
//!
//! The Merkle conditional swap gate and path synthesis are provided by
//! [`crate::circuit::poseidon_merkle`].

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{self, Advice, Column, Constraints, Expression, Selector},
    poly::Rotation,
};
use pasta_curves::pallas;

use ff::Field;
use halo2_gadgets::{
    ecc::chip::EccConfig,
    poseidon::{
        primitives::{self as poseidon, ConstantLength},
        Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
    },
};

use orchard::circuit::gadget::assign_free_advice;
use orchard::constants::OrchardFixedBases;

use super::imt::IMT_DEPTH;
use crate::circuit::poseidon_merkle::{MerkleSwapGate, synthesize_poseidon_merkle_path};

// ================================================================
// PuncturedIntervalGate
// ================================================================

/// Punctured interval check gate proving `nf_lo < real_nf < nf_hi` AND `real_nf != nf_mid`.
///
/// **Layout** (3 rows):
/// - Row 0: `advices[0]`=nf_lo, `advices[1]`=nf_hi, `advices[2]`=real_nf
/// - Row 1: `advices[0]`=nf_mid, `advices[1]`=x_lo, `advices[2]`=x_hi
/// - Row 2: `advices[0]`=diff_inv
///
/// **Constraints** (q_interval):
/// - `x_lo = real_nf - nf_lo - 1`  (strict lower: nf > nf_lo)
/// - `x_hi = nf_hi - real_nf - 1`  (strict upper: nf < nf_hi)
///
/// **Constraints** (q_neq):
/// - `(real_nf - nf_mid) * diff_inv = 1`  (non-equality with interior nullifier)
///
/// Range checks on `x_lo` and `x_hi` to `[0, 2^250)` are applied by
/// [`synthesize_imt_non_membership`] via `lookup_config.copy_check` after this gate assigns them.
///
/// NOTE: The 250-bit range checks are only sound when every IMT bracket
/// has span `nf_hi - nf_lo < 2^250`. The IMT MUST be initialized with
/// sentinel nullifiers at multiples of 2^249 (so each K=2 leaf spans
/// at most `2 × 2^249 = 2^250`) before any real nullifiers are inserted.
#[derive(Clone, Debug)]
pub(crate) struct PuncturedIntervalGate {
    pub(crate) q_interval: Selector,
    pub(crate) q_neq: Selector,
    advices: [Column<Advice>; 3],
}

impl PuncturedIntervalGate {
    pub(crate) fn configure(
        meta: &mut plonk::ConstraintSystem<pallas::Base>,
        advices: [Column<Advice>; 3],
    ) -> Self {
        let q_interval = meta.selector();
        let q_neq = meta.selector();

        meta.create_gate("Punctured interval check", |meta| {
            let q = meta.query_selector(q_interval);
            let nf_lo = meta.query_advice(advices[0], Rotation::cur());
            let nf_hi = meta.query_advice(advices[1], Rotation::cur());
            let real_nf = meta.query_advice(advices[2], Rotation::cur());
            let x_lo = meta.query_advice(advices[1], Rotation::next());
            let x_hi = meta.query_advice(advices[2], Rotation::next());

            let one = Expression::Constant(pallas::Base::one());

            Constraints::with_selector(
                q,
                [
                    // Strict lower bound: x_lo = real_nf - nf_lo - 1.
                    // Range-checking x_lo to [0, 2^250) proves real_nf > nf_lo,
                    // since if real_nf <= nf_lo then (real_nf - nf_lo - 1) wraps
                    // to a huge field element exceeding 2^250.
                    ("x_lo = real_nf - nf_lo - 1", x_lo.clone() - (real_nf.clone() - nf_lo - one.clone())),
                    // Strict upper bound: x_hi = nf_hi - real_nf - 1.
                    // Range-checking x_hi to [0, 2^250) proves real_nf < nf_hi.
                    ("x_hi = nf_hi - real_nf - 1", x_hi.clone() - (nf_hi - real_nf - one)),
                ],
            )
        });

        meta.create_gate("Non-equality check (nf != nf_mid)", |meta| {
            let q = meta.query_selector(q_neq);
            let real_nf = meta.query_advice(advices[2], Rotation::cur());
            let nf_mid = meta.query_advice(advices[0], Rotation::next());
            let diff_inv = meta.query_advice(advices[0], Rotation(2));

            let one = Expression::Constant(pallas::Base::one());

            Constraints::with_selector(
                q,
                [
                    // (real_nf - nf_mid) * diff_inv = 1
                    // This is satisfiable iff real_nf != nf_mid, since
                    // nf_mid is a field element and the inverse exists
                    // only when the difference is nonzero.
                    ("(nf - nf_mid) * inv = 1", (real_nf - nf_mid) * diff_inv - one),
                ],
            )
        });

        PuncturedIntervalGate {
            q_interval,
            q_neq,
            advices,
        }
    }

    /// Assigns the punctured interval check region.
    /// Returns `(x_lo, x_hi)` for external range checking.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assign(
        &self,
        region: &mut halo2_proofs::circuit::Region<'_, pallas::Base>,
        offset: usize,
        nf_lo: &AssignedCell<pallas::Base, pallas::Base>,
        nf_mid: &AssignedCell<pallas::Base, pallas::Base>,
        nf_hi: &AssignedCell<pallas::Base, pallas::Base>,
        real_nf: &AssignedCell<pallas::Base, pallas::Base>,
    ) -> Result<
        (
            AssignedCell<pallas::Base, pallas::Base>,
            AssignedCell<pallas::Base, pallas::Base>,
        ),
        plonk::Error,
    > {
        // Enable both selectors at row 0.
        self.q_interval.enable(region, offset)?;
        self.q_neq.enable(region, offset)?;

        // Row 0: nf_lo, nf_hi, real_nf
        nf_lo.copy_advice(|| "nf_lo", region, self.advices[0], offset)?;
        nf_hi.copy_advice(|| "nf_hi", region, self.advices[1], offset)?;
        real_nf.copy_advice(|| "real_nf", region, self.advices[2], offset)?;

        // Row 1: nf_mid, x_lo, x_hi
        nf_mid.copy_advice(|| "nf_mid", region, self.advices[0], offset + 1)?;

        let x_lo = region.assign_advice(
            || "x_lo = real_nf - nf_lo - 1",
            self.advices[1],
            offset + 1,
            || {
                real_nf
                    .value()
                    .copied()
                    .zip(nf_lo.value().copied())
                    .map(|(nf, lo)| nf - lo - pallas::Base::one())
            },
        )?;

        let x_hi = region.assign_advice(
            || "x_hi = nf_hi - real_nf - 1",
            self.advices[2],
            offset + 1,
            || {
                nf_hi
                    .value()
                    .copied()
                    .zip(real_nf.value().copied())
                    .map(|(hi, nf)| hi - nf - pallas::Base::one())
            },
        )?;

        // Row 2: diff_inv = inverse(real_nf - nf_mid)
        region.assign_advice(
            || "diff_inv = 1/(real_nf - nf_mid)",
            self.advices[0],
            offset + 2,
            || {
                real_nf
                    .value()
                    .copied()
                    .zip(nf_mid.value().copied())
                    .map(|(nf, mid)| {
                        let diff = nf - mid;
                        debug_assert!(
                            bool::from(!diff.is_zero()),
                            "real_nf must not equal nf_mid — the nullifier is already in the tree"
                        );
                        diff.invert().unwrap_or(pallas::Base::zero())
                    })
            },
        )?;

        Ok((x_lo, x_hi))
    }
}

// ================================================================
// ImtNonMembershipConfig
// ================================================================

/// Bundles the Merkle swap gate, punctured interval gate, and the columns they need.
#[derive(Clone, Debug)]
pub(crate) struct ImtNonMembershipConfig {
    pub(crate) swap_gate: MerkleSwapGate,
    pub(crate) interval_gate: PuncturedIntervalGate,
    /// The first advice column, used for free-witness assignments.
    pub(crate) advice_0: Column<Advice>,
}

impl ImtNonMembershipConfig {
    /// Configures IMT gates. Uses `advices[0..5]` for the swap gate and
    /// `advices[0..3]` for the interval gate (overlapping is fine — different selectors).
    pub(crate) fn configure(
        meta: &mut plonk::ConstraintSystem<pallas::Base>,
        advices: &[Column<Advice>; 10],
    ) -> Self {
        let swap_gate = MerkleSwapGate::configure(
            meta,
            [advices[0], advices[1], advices[2], advices[3], advices[4]],
        );
        let interval_gate = PuncturedIntervalGate::configure(
            meta,
            [advices[0], advices[1], advices[2]],
        );
        ImtNonMembershipConfig {
            swap_gate,
            interval_gate,
            advice_0: advices[0],
        }
    }
}

// ================================================================
// synthesize_imt_non_membership
// ================================================================

/// Synthesizes the IMT non-membership proof for a single note slot (condition 13).
///
/// K=2 punctured-range model. Orchestrates:
/// 1. Witness nf_lo, nf_mid, nf_hi
/// 2. Poseidon3 leaf hash = Poseidon3(nf_lo, nf_mid, nf_hi)
/// 3. 29-level Merkle path via [`synthesize_poseidon_merkle_path`]
/// 4. Punctured interval check: nf_lo < real_nf < nf_hi AND real_nf != nf_mid
/// 5. Range checks on x_lo, x_hi to [0, 2^250)
///
/// Returns `imt_root` which the caller feeds into the `q_per_note` gate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn synthesize_imt_non_membership(
    imt_config: &ImtNonMembershipConfig,
    poseidon_config: &PoseidonConfig<pallas::Base, 3, 2>,
    ecc_config: &EccConfig<OrchardFixedBases>,
    layouter: &mut impl Layouter<pallas::Base>,
    imt_nf_bounds: Value<[pallas::Base; 3]>,
    imt_leaf_pos: Value<u32>,
    imt_path: Value<[pallas::Base; IMT_DEPTH]>,
    real_nf: &AssignedCell<pallas::Base, pallas::Base>,
    slot: usize,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let s = slot;

    // Witness the three nullifier boundaries.
    let imt_nf_lo = assign_free_advice(
        layouter.namespace(|| format!("note {s} imt_nf_lo")),
        imt_config.advice_0,
        imt_nf_bounds.map(|b| b[0]),
    )?;

    let imt_nf_mid = assign_free_advice(
        layouter.namespace(|| format!("note {s} imt_nf_mid")),
        imt_config.advice_0,
        imt_nf_bounds.map(|b| b[1]),
    )?;

    let imt_nf_hi = assign_free_advice(
        layouter.namespace(|| format!("note {s} imt_nf_hi")),
        imt_config.advice_0,
        imt_nf_bounds.map(|b| b[2]),
    )?;

    // Compute leaf hash: Poseidon3(nf_lo, nf_mid, nf_hi).
    let leaf_hash = {
        let poseidon_hasher = PoseidonHash::<
            pallas::Base,
            _,
            poseidon::P128Pow5T3,
            ConstantLength<3>,
            3,
            2,
        >::init(
            PoseidonChip::construct(poseidon_config.clone()),
            layouter.namespace(|| format!("note {s} imt leaf hash init")),
        )?;
        poseidon_hasher.hash(
            layouter.namespace(|| format!("note {s} Poseidon3(nf_lo, nf_mid, nf_hi)")),
            [imt_nf_lo.clone(), imt_nf_mid.clone(), imt_nf_hi.clone()],
        )?
    };

    // 29-level Poseidon Merkle path from leaf_hash to imt_root.
    let imt_root = synthesize_poseidon_merkle_path::<IMT_DEPTH>(
        &imt_config.swap_gate,
        poseidon_config,
        layouter,
        imt_config.advice_0,
        leaf_hash,
        imt_leaf_pos,
        imt_path,
        &format!("note {s} imt"),
    )?;

    // Punctured interval check: nf_lo < real_nf < nf_hi AND real_nf != nf_mid.
    let (x_lo, x_hi) = layouter.assign_region(
        || format!("note {s} punctured interval check"),
        |mut region| {
            imt_config.interval_gate.assign(
                &mut region,
                0,
                &imt_nf_lo,
                &imt_nf_mid,
                &imt_nf_hi,
                real_nf,
            )
        },
    )?;

    // Range checks enforce the strict interval inclusion.
    // x_lo in [0, 2^250) proves real_nf > nf_lo (strict).
    // x_hi in [0, 2^250) proves real_nf < nf_hi (strict).
    // 25 limbs × 10 bits = 250-bit range.
    ecc_config.lookup_config.copy_check(
        layouter.namespace(|| format!("note {s} x_lo < 2^250")),
        x_lo,
        25,
        true,
    )?;

    ecc_config.lookup_config.copy_check(
        layouter.namespace(|| format!("note {s} x_hi < 2^250")),
        x_hi,
        25,
        true,
    )?;

    Ok(imt_root)
}
