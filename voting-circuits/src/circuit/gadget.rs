//! Generic halo2 helpers shared across the governance circuits.
//!
//! These are framework-level utilities that aren't tied to anything orchard-
//! specific, so they live here rather than in `orchard::circuit::gadget`.

use ff::Field;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::{self, Advice, Column},
};

/// Bakes a constant into the verifier key by assigning it to a free advice
/// cell in a standalone region.
///
/// Prerequisites — both must hold at circuit configuration time:
///
/// 1. Some `Column<Fixed>` must be registered via
///    `meta.enable_constant(fixed_col)` on `&mut ConstraintSystem`. This
///    designates the column whose cells will hold the baked-in constant
///    values and whose permutation cells back the copy constraint that
///    forces the advice cell to match.
/// 2. The advice column passed here must have
///    `meta.enable_equality(advice_col)` called on it, so the copy
///    constraint from the fixed-column constants cell to this advice cell
///    can be added.
///
/// Mechanism: at synthesis time, `assign_advice_from_constant` queues the
/// `(value, advice_cell)` pair; the V1 floor planner later writes `value`
/// into an `enable_constant`-registered fixed column (which is part of the
/// VK) and adds a copy constraint between that fixed cell and the advice
/// cell. A malicious client driving the honest circuit therefore cannot put
/// any value other than `constant` in the advice cell without breaking the
/// permutation argument.
///
/// Counterpart of orchard's `assign_free_advice` for known-constant values.
pub fn assign_constant<F: Field>(
    mut layouter: impl Layouter<F>,
    column: Column<Advice>,
    constant: F,
) -> Result<AssignedCell<F, F>, plonk::Error> {
    layouter.assign_region(
        || "load constant",
        |mut region| region.assign_advice_from_constant(|| "constant", column, 0, constant),
    )
}
