//! Generic halo2 helpers shared across the governance circuits.
//!
//! These are framework-level utilities that aren't tied to anything orchard-
//! specific, so they live here rather than in `orchard::circuit::gadget`.

use ff::Field;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::{self, Advice, Column},
};

/// Bakes a constant into the verifier key by assigning it to a free advice cell
/// in a standalone region. The advice column must have constants enabled
/// (via `Region::enable_constant`) at circuit configuration time.
///
/// Counterpart of orchard's `assign_free_advice` for known-constant values:
/// the constant ends up baked into the verifier key, so a malicious prover
/// can't substitute a different value at proving time.
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
