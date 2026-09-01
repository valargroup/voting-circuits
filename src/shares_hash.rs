//! Shared circuit gadget for the shares-hash computation used in ZKP #2 and ZKP #3.
//!
//! This module is the authoritative in-tree definition of the aggregate
//! encrypted-share hash. Both the vote-proof circuit (ZKP #2, condition 11′)
//! and the share-reveal circuit (ZKP #3, condition 3) call this
//! implementation rather than maintaining separate formula copies:
//!
//! ```text
//! shares_hash = Poseidon(share_comm_0, …, share_comm_15)
//! ```
//!
//! The 16 `share_comm_i` values are the weighted selected commitments owned
//! by [`crate::bridge`] — each a wide Poseidon commitment over one share's
//! blind and all 16 bucket ciphertext coordinates.
//!
//! `shares_hash` is a reusable internal circuit value, not a public instance
//! by itself. ZKP #2 binds it to the verifier only by feeding it into the
//! public vote commitment, while ZKP #3 binds it transitively through the same
//! vote commitment tree path.
//!
//! This module extracts those constraints into a single, auditable gadget so
//! that both circuits provably execute the same hash logic.

use voting_crypto_deps::halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip,
};
use voting_crypto_deps::halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk,
};
use voting_crypto_deps::pasta_curves::pallas;

/// Computes the shares hash in-circuit from pre-computed share commitments:
///
/// ```text
/// shares_hash = Poseidon(share_comm_0, …, share_comm_15)
/// ```
///
/// The caller supplies the 16 `share_comm` cells — witnessed selected
/// commitments in ZKP #2's bridge re-opening and ZKP #3's private witnesses.
///
/// Returns an internal cell; callers must bind it transitively through their
/// own public commitment path.
pub(crate) fn compute_shares_hash_from_comms_in_circuit(
    poseidon_chip: PoseidonChip<pallas::Base, 3, 2>,
    mut layouter: impl Layouter<pallas::Base>,
    share_comms: [AssignedCell<pallas::Base, pallas::Base>; 16],
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let hasher = PoseidonHash::<
        pallas::Base,
        _,
        poseidon::P128Pow5T3,
        ConstantLength<16>,
        3, // WIDTH
        2, // RATE
    >::init(
        poseidon_chip,
        layouter.namespace(|| "shares_hash Poseidon init"),
    )?;
    hasher.hash(
        layouter.namespace(|| "shares_hash = Poseidon(share_comms)"),
        share_comms,
    )
}

/// Native counterpart of [`compute_shares_hash_from_comms_in_circuit`].
///
/// Computes `Poseidon(share_comm_0, …, share_comm_15)` outside the circuit.
pub fn shares_hash_from_comms(share_comms: [pallas::Base; 16]) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<16>, 3, 2>::init().hash(share_comms)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ff::{Field, PrimeField};
    use crate::rand::rngs::OsRng;
    use voting_crypto_deps::halo2_gadgets::poseidon::Pow5Config as PoseidonConfig;
    use voting_crypto_deps::halo2_proofs::{
        circuit::{floor_planner, Value},
        dev::MockProver,
        plonk::{Advice, Column, ConstraintSystem, Fixed, Instance as InstanceColumn},
    };

    // ---------------------------------------------------------------
    // Shared minimal circuit config (Poseidon only, no ECC).
    // ---------------------------------------------------------------

    #[derive(Clone)]
    struct TestConfig {
        primary: Column<InstanceColumn>,
        advice: Column<Advice>,
        poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    }

    impl TestConfig {
        fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self {
            let primary = meta.instance_column();
            meta.enable_equality(primary);

            // 5 advice columns: [0] general witness, [1..4] Poseidon state.
            let advices: [Column<Advice>; 5] = core::array::from_fn(|_| meta.advice_column());
            for col in &advices {
                meta.enable_equality(*col);
            }

            let fixed: [Column<Fixed>; 6] = core::array::from_fn(|_| meta.fixed_column());
            // Dedicated constants column required by Poseidon strict range checks.
            let constants = meta.fixed_column();
            meta.enable_constant(constants);
            let poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
                meta,
                advices[1..4].try_into().unwrap(),
                advices[4],
                fixed[0..3].try_into().unwrap(),
                fixed[3..6].try_into().unwrap(),
            );

            TestConfig {
                primary,
                advice: advices[0],
                poseidon_config,
            }
        }

        fn poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
            PoseidonChip::construct(self.poseidon_config.clone())
        }
    }

    /// Witnesses a single field element into the advice column.
    fn witness(
        mut layouter: impl Layouter<pallas::Base>,
        col: Column<Advice>,
        val: Value<pallas::Base>,
    ) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
        layouter.assign_region(
            || "witness",
            |mut region| region.assign_advice(|| "val", col, 0, || val),
        )
    }

    fn base_from_repr(bytes: [u8; 32]) -> pallas::Base {
        pallas::Base::from_repr(bytes).expect("frozen vector must be canonical")
    }

    #[test]
    fn shares_hash_from_comms_frozen_vector() {
        let comms: [pallas::Base; 16] =
            core::array::from_fn(|i| pallas::Base::from(2000 + i as u64));

        assert_eq!(
            shares_hash_from_comms(comms),
            base_from_repr([
                66, 178, 247, 131, 58, 182, 190, 147, 179, 149, 118, 94, 239, 231, 109, 165, 61,
                42, 59, 57, 145, 92, 175, 87, 153, 205, 72, 118, 189, 200, 77, 35,
            ])
        );
    }

    // ================================================================
    // compute_shares_hash_from_comms_in_circuit
    // ================================================================

    /// Minimal circuit: computes `compute_shares_hash_from_comms_in_circuit`
    /// from 16 pre-computed share_comms and constrains to instance row 0.
    #[derive(Clone)]
    struct ComputeSharesHashFromCommsCircuit {
        share_comms: [pallas::Base; 16],
    }

    impl Default for ComputeSharesHashFromCommsCircuit {
        fn default() -> Self {
            Self {
                share_comms: [pallas::Base::zero(); 16],
            }
        }
    }

    impl plonk::Circuit<pallas::Base> for ComputeSharesHashFromCommsCircuit {
        type Config = TestConfig;
        type FloorPlanner = floor_planner::V1;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
            TestConfig::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<pallas::Base>,
        ) -> Result<(), plonk::Error> {
            let mut comm_cells = Vec::with_capacity(16);
            for i in 0..16 {
                comm_cells.push(witness(
                    layouter.namespace(|| format!("comm_{i}")),
                    config.advice,
                    Value::known(self.share_comms[i]),
                )?);
            }
            let comms: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                comm_cells.try_into().unwrap();

            let result = super::compute_shares_hash_from_comms_in_circuit(
                config.poseidon_chip(),
                layouter.namespace(|| "hash_from_comms"),
                comms,
            )?;
            layouter.constrain_instance(result.cell(), config.primary, 0)
        }
    }

    /// The from-comms gadget matches the native computation.
    #[test]
    fn shares_hash_from_comms_matches_native() {
        let mut rng = OsRng;
        let comms: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let expected = super::shares_hash_from_comms(comms);

        let circuit = ComputeSharesHashFromCommsCircuit { share_comms: comms };
        let prover =
            MockProver::run(12, &circuit, vec![vec![expected]]).expect("MockProver::run failed");
        assert_eq!(prover.verify(), Ok(()));
    }

    /// Corrupting any single share_comm changes the output.
    #[test]
    fn shares_hash_from_comms_wrong_comm_fails() {
        let mut rng = OsRng;
        let comms: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let expected = super::shares_hash_from_comms(comms);

        let mut bad_comms = comms;
        bad_comms[7] = pallas::Base::random(&mut rng);
        let circuit = ComputeSharesHashFromCommsCircuit {
            share_comms: bad_comms,
        };
        let prover =
            MockProver::run(12, &circuit, vec![vec![expected]]).expect("MockProver::run failed");
        assert!(prover.verify().is_err());
    }

    /// Every input position contributes to the hash.
    #[test]
    fn all_16_positions_are_hashed() {
        let comms: [pallas::Base; 16] =
            core::array::from_fn(|i| pallas::Base::from(3000 + i as u64));
        let baseline = shares_hash_from_comms(comms);

        for position in 0..16 {
            let mut altered = comms;
            altered[position] += pallas::Base::one();
            assert_ne!(
                baseline,
                shares_hash_from_comms(altered),
                "position {position} must bind"
            );
        }
    }
}
