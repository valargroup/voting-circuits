//! Generic halo2 helpers shared across the governance circuits.
//!
//! These are framework-level utilities that aren't tied to anything orchard-
//! specific, so they live here rather than in `orchard::circuit::gadget`.

use ff::Field;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::{self, Advice, Column},
};

/// Assigns a circuit-known constant to an advice cell via
/// `assign_advice_from_constant`.
///
/// Prerequisites: the circuit must configure a fixed constants column with
/// `meta.enable_constant(...)`, and the advice column passed here must be
/// equality-enabled with `meta.enable_equality(...)`. Halo2 places `constant`
/// in that fixed column and copy-constrains the returned advice cell to it, so
/// a malicious client driving the honest circuit cannot substitute another
/// value.
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
