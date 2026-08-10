//! Small inverse-witness gadget for rejecting exact zero witnesses.
//!
//! These checks harden relations that are algebraically valid at zero but leak
//! privacy when a prover chooses the degenerate value.

use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::{Advice, Assigned, Column, ConstraintSystem, Constraints, Error, Expression, Selector},
    poly::Rotation,
};
use pasta_curves::{group::ff::Field, pallas};

/// Configuration for field non-zero checks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NonZeroConfig {
    value: Column<Advice>,
    inverse: Column<Advice>,
    q_value_nonzero: Selector,
}

impl NonZeroConfig {
    /// Configures inverse-witness checks over the supplied advice columns.
    pub(crate) fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        advices: [Column<Advice>; 2],
    ) -> Self {
        let q_value_nonzero = meta.selector();

        meta.create_gate("field element != 0", |meta| {
            let q = meta.query_selector(q_value_nonzero);
            let value = meta.query_advice(advices[0], Rotation::cur());
            let inv = meta.query_advice(advices[1], Rotation::cur());
            let one = Expression::Constant(pallas::Base::one());
            Constraints::with_selector(q, [("value * inv = 1", value * inv - one)])
        });

        Self {
            value: advices[0],
            inverse: advices[1],
            q_value_nonzero,
        }
    }

    /// Constrains `cell != 0` by assigning and checking its multiplicative inverse.
    pub(crate) fn constrain_nonzero<V>(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        label: &'static str,
        cell: &AssignedCell<V, pallas::Base>,
    ) -> Result<(), Error>
    where
        for<'v> Assigned<pallas::Base>: From<&'v V>,
    {
        layouter.assign_region(
            || label,
            |mut region| {
                self.q_value_nonzero.enable(&mut region, 0)?;

                let value = cell.value_field().map(Assigned::evaluate);
                let value_copy: AssignedCell<pallas::Base, pallas::Base> = region
                    .assign_advice::<_, pallas::Base, _, _>(|| "value", self.value, 0, || value)?;
                region.constrain_equal(value_copy.cell(), cell.cell())?;

                let _: AssignedCell<pallas::Base, pallas::Base> = region
                    .assign_advice::<_, pallas::Base, _, _>(
                        || "value_inv",
                        self.inverse,
                        0,
                        || value.map(|value| value.invert().unwrap_or(pallas::Base::zero())),
                    )?;

                Ok(())
            },
        )
    }
}
