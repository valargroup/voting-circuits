//! Shared circuit gadget for the shares-hash computation used in ZKP #2 and ZKP #3.
//!
//! This module is the authoritative in-tree definition of the two-level
//! encrypted-share hash. Both the vote-proof circuit (ZKP #2, condition 10) and
//! the share-reveal circuit (ZKP #3, condition 3) call this implementation
//! rather than maintaining separate formula copies:
//!
//! ```text
//! share_comm_i = Poseidon(DOMAIN_SHARE_COMM, blind_i,
//!                         c1_i_x, c2_i_x, c1_i_y, c2_i_y)   for i ∈ 0..16
//! shares_hash  = Poseidon(share_comm_0, …, share_comm_15)
//! ```
//!
//! The y-coordinates are included to bind each share commitment to the exact
//! curve point, preventing ciphertext sign-malleability attacks where an
//! adversary negates ElGamal ciphertext points without invalidating the ZKP.
//!
//! `shares_hash` is a reusable internal circuit value, not a public instance
//! by itself. ZKP #2 binds it to the verifier only by feeding it into the
//! public vote commitment, while ZKP #3 binds it transitively through the same
//! vote commitment tree path.
//!
//! This module extracts those constraints into a single, auditable gadget so
//! that both circuits provably execute the same hash logic.

use halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip,
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk,
};
use itertools::Itertools;
use pasta_curves::pallas;

pub use crate::circuit::share_commitment::share_commitment;
use crate::circuit::share_commitment::share_commitment_poseidon;

/// Native full two-level shares hash:
///
/// ```text
/// share_comm_i = Poseidon(DOMAIN_SHARE_COMM, blind_i,
///                         c1_i_x, c2_i_x, c1_i_y, c2_i_y)   for i ∈ 0..16
/// shares_hash  = Poseidon(share_comm_0, …, share_comm_15)
/// ```
///
/// Native counterpart of [`compute_shares_hash_in_circuit`].
pub fn shares_hash(
    share_blinds: [pallas::Base; 16],
    enc_share_c1_x: [pallas::Base; 16],
    enc_share_c2_x: [pallas::Base; 16],
    enc_share_c1_y: [pallas::Base; 16],
    enc_share_c2_y: [pallas::Base; 16],
) -> pallas::Base {
    let comms: [pallas::Base; 16] = core::array::from_fn(|i| {
        share_commitment(
            share_blinds[i],
            enc_share_c1_x[i],
            enc_share_c2_x[i],
            enc_share_c1_y[i],
            enc_share_c2_y[i],
        )
    });
    shares_hash_from_comms(comms)
}

/// Computes a single blinded per-share commitment in-circuit:
///
/// ```text
/// share_comm = Poseidon(DOMAIN_SHARE_COMM, blind, c1_x, c2_x, c1_y, c2_y)
/// ```
///
/// `domain_share_comm` must be assigned with
/// `crate::circuit::share_commitment::assign_domain_share_comm`, which pins
/// the domain tag into the verification key.
///
/// The y-coordinates bind the commitment to the exact curve point, preventing
/// ciphertext sign-malleability. The `index` is used only for namespace labels
/// and has no effect on the constraint system.
pub(crate) fn hash_share_commitment_in_circuit(
    chip: PoseidonChip<pallas::Base, 3, 2>,
    mut layouter: impl Layouter<pallas::Base>,
    domain_share_comm: AssignedCell<pallas::Base, pallas::Base>,
    blind: AssignedCell<pallas::Base, pallas::Base>,
    enc_c1_x: AssignedCell<pallas::Base, pallas::Base>,
    enc_c2_x: AssignedCell<pallas::Base, pallas::Base>,
    enc_c1_y: AssignedCell<pallas::Base, pallas::Base>,
    enc_c2_y: AssignedCell<pallas::Base, pallas::Base>,
    index: usize,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    share_commitment_poseidon(
        chip,
        &mut layouter,
        &format!("share_comm_{index} = Poseidon(DOMAIN_SHARE_COMM, blind, c1_x, c2_x, c1_y, c2_y)[{index}]"),
        domain_share_comm,
        blind,
        enc_c1_x,
        enc_c2_x,
        enc_c1_y,
        enc_c2_y,
    )
}

/// Computes the two-level shares hash in-circuit:
///
/// ```text
/// share_comm_i = Poseidon(DOMAIN_SHARE_COMM, blind_i,
///                         c1_i_x, c2_i_x, c1_i_y, c2_i_y)   for i ∈ 0..16
/// shares_hash  = Poseidon(share_comm_0, …, share_comm_15)
/// ```
///
/// # Arguments
///
/// * `poseidon_chip` — A closure that returns a fresh `PoseidonChip` each time
///   it is called. It is called 17 times: once per per-share hash and once for
///   the outer hash. Typically `|| config.poseidon_chip()`.
/// * `layouter` — The circuit layouter.
/// * `domain_share_comm` — Constant cell for `DOMAIN_SHARE_COMM`.
/// * `blinds` — The 16 per-share blind factors.
/// * `enc_c1_x` — The 16 El Gamal `C1` x-coordinates.
/// * `enc_c2_x` — The 16 El Gamal `C2` x-coordinates.
/// * `enc_c1_y` — The 16 El Gamal `C1` y-coordinates.
/// * `enc_c2_y` — The 16 El Gamal `C2` y-coordinates.
///
/// Returns the internal `shares_hash` cell. The caller is responsible for
/// consuming that cell in a public binding, such as the vote commitment hash;
/// this gadget does not constrain the result to an instance column.
pub(crate) fn compute_shares_hash_in_circuit(
    poseidon_chip: impl Fn() -> PoseidonChip<pallas::Base, 3, 2>,
    mut layouter: impl Layouter<pallas::Base>,
    domain_share_comm: AssignedCell<pallas::Base, pallas::Base>,
    blinds: [AssignedCell<pallas::Base, pallas::Base>; 16],
    enc_c1_x: [AssignedCell<pallas::Base, pallas::Base>; 16],
    enc_c2_x: [AssignedCell<pallas::Base, pallas::Base>; 16],
    enc_c1_y: [AssignedCell<pallas::Base, pallas::Base>; 16],
    enc_c2_y: [AssignedCell<pallas::Base, pallas::Base>; 16],
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let share_comms: [_; 16] = IntoIterator::into_iter(blinds)
        .zip_eq(enc_c1_x)
        .zip_eq(enc_c2_x)
        .zip_eq(enc_c1_y)
        .zip_eq(enc_c2_y)
        .enumerate()
        .map(|(i, ((((blind, c1x), c2x), c1y), c2y))| {
            hash_share_commitment_in_circuit(
                poseidon_chip(),
                layouter.namespace(|| format!("share_comm_{i}")),
                domain_share_comm.clone(),
                blind,
                c1x,
                c2x,
                c1y,
                c2y,
                i,
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("always 16 elements");

    // Outer hash: shares_hash = Poseidon(share_comm_0, …, share_comm_15)
    let hasher = PoseidonHash::<
        pallas::Base,
        _,
        poseidon::P128Pow5T3,
        ConstantLength<16>,
        3, // WIDTH
        2, // RATE
    >::init(
        poseidon_chip(),
        layouter.namespace(|| "shares_hash Poseidon init"),
    )?;
    hasher.hash(
        layouter.namespace(|| "shares_hash = Poseidon(share_comms)"),
        share_comms,
    )
}

/// Computes the shares hash in-circuit from pre-computed share commitments:
///
/// ```text
/// shares_hash = Poseidon(share_comm_0, …, share_comm_15)
/// ```
///
/// Unlike [`compute_shares_hash_in_circuit`], this skips the per-share
/// blind hashing (level 1) because the caller already provides the 16
/// `share_comm` values. ZKP #3 supplies them as private advice cells and
/// binds them transitively through `shares_hash`, the vote commitment, and
/// the vote commitment tree root.
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

    use ff::{Field, PrimeField};
    use halo2_gadgets::poseidon::Pow5Config as PoseidonConfig;
    use halo2_proofs::{
        circuit::{floor_planner, Value},
        dev::MockProver,
        plonk::{Advice, Column, ConstraintSystem, Fixed, Instance as InstanceColumn},
    };
    use rand::rngs::OsRng;

    use crate::circuit::share_commitment as share_commitment_hash;

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

    // ================================================================
    // hash_share_commitment_in_circuit
    // ================================================================

    fn base_from_repr(bytes: [u8; 32]) -> pallas::Base {
        pallas::Base::from_repr(bytes).expect("frozen vector must be canonical")
    }

    #[test]
    fn share_commitment_frozen_vector() {
        let actual = share_commitment(
            pallas::Base::from(1u64),
            pallas::Base::from(2u64),
            pallas::Base::from(3u64),
            pallas::Base::from(4u64),
            pallas::Base::from(5u64),
        );

        assert_eq!(
            actual,
            base_from_repr([
                183, 66, 173, 64, 240, 83, 206, 161, 132, 211, 79, 38, 240, 12, 144, 142, 247, 139,
                173, 56, 54, 59, 51, 73, 42, 113, 240, 242, 21, 103, 150, 29,
            ])
        );
    }

    #[test]
    fn shares_hash_frozen_vector() {
        let blinds = core::array::from_fn(|i| pallas::Base::from((i + 1) as u64));
        let enc_c1_x = core::array::from_fn(|i| pallas::Base::from((i + 17) as u64));
        let enc_c2_x = core::array::from_fn(|i| pallas::Base::from((i + 33) as u64));
        let enc_c1_y = core::array::from_fn(|i| pallas::Base::from((i + 49) as u64));
        let enc_c2_y = core::array::from_fn(|i| pallas::Base::from((i + 65) as u64));

        assert_eq!(
            shares_hash(blinds, enc_c1_x, enc_c2_x, enc_c1_y, enc_c2_y),
            base_from_repr([
                125, 88, 190, 64, 180, 158, 228, 46, 43, 173, 80, 255, 152, 160, 47, 234, 86, 36,
                157, 87, 187, 167, 86, 239, 58, 45, 222, 42, 111, 6, 63, 28,
            ])
        );
    }

    /// Minimal circuit: computes `hash_share_commitment_in_circuit` and
    /// constrains the result to instance row 0.
    #[derive(Clone, Default)]
    struct HashShareCommCircuit {
        blind: pallas::Base,
        c1_x: pallas::Base,
        c2_x: pallas::Base,
        c1_y: pallas::Base,
        c2_y: pallas::Base,
    }

    impl plonk::Circuit<pallas::Base> for HashShareCommCircuit {
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
            let blind = witness(
                layouter.namespace(|| "blind"),
                config.advice,
                Value::known(self.blind),
            )?;
            let c1x = witness(
                layouter.namespace(|| "c1_x"),
                config.advice,
                Value::known(self.c1_x),
            )?;
            let c2x = witness(
                layouter.namespace(|| "c2_x"),
                config.advice,
                Value::known(self.c2_x),
            )?;
            let c1y = witness(
                layouter.namespace(|| "c1_y"),
                config.advice,
                Value::known(self.c1_y),
            )?;
            let c2y = witness(
                layouter.namespace(|| "c2_y"),
                config.advice,
                Value::known(self.c2_y),
            )?;

            let domain_share_comm =
                share_commitment_hash::assign_domain_share_comm(&mut layouter, config.advice)?;
            let result = hash_share_commitment_in_circuit(
                config.poseidon_chip(),
                layouter.namespace(|| "hash_share_comm"),
                domain_share_comm,
                blind,
                c1x,
                c2x,
                c1y,
                c2y,
                0,
            )?;
            layouter.constrain_instance(result.cell(), config.primary, 0)
        }
    }

    /// In-circuit result matches the native `share_commitment` helper.
    #[test]
    fn hash_share_commitment_matches_native() {
        let mut rng = OsRng;
        let blind = pallas::Base::random(&mut rng);
        let c1_x = pallas::Base::random(&mut rng);
        let c2_x = pallas::Base::random(&mut rng);
        let c1_y = pallas::Base::random(&mut rng);
        let c2_y = pallas::Base::random(&mut rng);

        let expected = share_commitment(blind, c1_x, c2_x, c1_y, c2_y);
        let circuit = HashShareCommCircuit {
            blind,
            c1_x,
            c2_x,
            c1_y,
            c2_y,
        };
        let prover =
            MockProver::run(10, &circuit, vec![vec![expected]]).expect("MockProver::run failed");
        assert_eq!(prover.verify(), Ok(()));
    }

    /// Swapping c1 and c2 produces a different hash (input order matters).
    #[test]
    fn hash_share_commitment_input_order_matters() {
        let mut rng = OsRng;
        let blind = pallas::Base::random(&mut rng);
        let c1_x = pallas::Base::random(&mut rng);
        let c2_x = pallas::Base::random(&mut rng);
        let c1_y = pallas::Base::random(&mut rng);
        let c2_y = pallas::Base::random(&mut rng);

        let wrong = share_commitment(blind, c2_x, c1_x, c2_y, c1_y);
        let circuit = HashShareCommCircuit {
            blind,
            c1_x,
            c2_x,
            c1_y,
            c2_y,
        };
        let prover =
            MockProver::run(10, &circuit, vec![vec![wrong]]).expect("MockProver::run failed");
        assert!(prover.verify().is_err());
    }

    /// Negating a y-coordinate (simulating sign-bit flip) changes the hash.
    #[test]
    fn hash_share_commitment_y_negate_changes_hash() {
        let mut rng = OsRng;
        let blind = pallas::Base::random(&mut rng);
        let c1_x = pallas::Base::random(&mut rng);
        let c2_x = pallas::Base::random(&mut rng);
        let c1_y = pallas::Base::random(&mut rng);
        let c2_y = pallas::Base::random(&mut rng);

        let correct = share_commitment(blind, c1_x, c2_x, c1_y, c2_y);
        let negated = share_commitment(blind, c1_x, c2_x, -c1_y, c2_y);
        assert_ne!(
            correct, negated,
            "negating c1_y must change the share commitment"
        );
    }

    // ================================================================
    // compute_shares_hash_in_circuit
    // ================================================================

    /// Minimal circuit: computes `compute_shares_hash_in_circuit` over 16
    /// shares and constrains the result to instance row 0.
    #[derive(Clone)]
    struct ComputeSharesHashCircuit {
        blinds: [pallas::Base; 16],
        enc_c1_x: [pallas::Base; 16],
        enc_c2_x: [pallas::Base; 16],
        enc_c1_y: [pallas::Base; 16],
        enc_c2_y: [pallas::Base; 16],
    }

    impl Default for ComputeSharesHashCircuit {
        fn default() -> Self {
            Self {
                blinds: [pallas::Base::zero(); 16],
                enc_c1_x: [pallas::Base::zero(); 16],
                enc_c2_x: [pallas::Base::zero(); 16],
                enc_c1_y: [pallas::Base::zero(); 16],
                enc_c2_y: [pallas::Base::zero(); 16],
            }
        }
    }

    impl plonk::Circuit<pallas::Base> for ComputeSharesHashCircuit {
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
            let mut blind_cells = Vec::with_capacity(16);
            let mut c1x_cells = Vec::with_capacity(16);
            let mut c2x_cells = Vec::with_capacity(16);
            let mut c1y_cells = Vec::with_capacity(16);
            let mut c2y_cells = Vec::with_capacity(16);
            for i in 0..16 {
                blind_cells.push(witness(
                    layouter.namespace(|| format!("blind_{i}")),
                    config.advice,
                    Value::known(self.blinds[i]),
                )?);
                c1x_cells.push(witness(
                    layouter.namespace(|| format!("c1x_{i}")),
                    config.advice,
                    Value::known(self.enc_c1_x[i]),
                )?);
                c2x_cells.push(witness(
                    layouter.namespace(|| format!("c2x_{i}")),
                    config.advice,
                    Value::known(self.enc_c2_x[i]),
                )?);
                c1y_cells.push(witness(
                    layouter.namespace(|| format!("c1y_{i}")),
                    config.advice,
                    Value::known(self.enc_c1_y[i]),
                )?);
                c2y_cells.push(witness(
                    layouter.namespace(|| format!("c2y_{i}")),
                    config.advice,
                    Value::known(self.enc_c2_y[i]),
                )?);
            }
            let blinds: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                blind_cells.try_into().unwrap();
            let enc_c1_x: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                c1x_cells.try_into().unwrap();
            let enc_c2_x: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                c2x_cells.try_into().unwrap();
            let enc_c1_y: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                c1y_cells.try_into().unwrap();
            let enc_c2_y: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                c2y_cells.try_into().unwrap();

            let domain_share_comm =
                share_commitment_hash::assign_domain_share_comm(&mut layouter, config.advice)?;
            let result = compute_shares_hash_in_circuit(
                || config.poseidon_chip(),
                layouter.namespace(|| "compute_shares_hash"),
                domain_share_comm,
                blinds,
                enc_c1_x,
                enc_c2_x,
                enc_c1_y,
                enc_c2_y,
            )?;
            layouter.constrain_instance(result.cell(), config.primary, 0)
        }
    }

    /// In-circuit result matches the native `shares_hash` helper.
    #[test]
    fn compute_shares_hash_matches_native() {
        let mut rng = OsRng;
        let blinds: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));

        let expected = shares_hash(blinds, enc_c1_x, enc_c2_x, enc_c1_y, enc_c2_y);
        let circuit = ComputeSharesHashCircuit {
            blinds,
            enc_c1_x,
            enc_c2_x,
            enc_c1_y,
            enc_c2_y,
        };
        // K=12 (4096 rows) comfortably fits the 16 inner Poseidon(6) regions
        // plus the outer Poseidon(16) region.
        let prover =
            MockProver::run(12, &circuit, vec![vec![expected]]).expect("MockProver::run failed");
        assert_eq!(prover.verify(), Ok(()));
    }

    /// Corrupting any single enc_c1_x value changes the output.
    #[test]
    fn compute_shares_hash_wrong_enc_c1_fails() {
        let mut rng = OsRng;
        let blinds: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));

        let correct = shares_hash(blinds, enc_c1_x, enc_c2_x, enc_c1_y, enc_c2_y);

        let mut circuit = ComputeSharesHashCircuit {
            blinds,
            enc_c1_x,
            enc_c2_x,
            enc_c1_y,
            enc_c2_y,
        };
        circuit.enc_c1_x[2] = pallas::Base::random(&mut rng);

        let prover =
            MockProver::run(12, &circuit, vec![vec![correct]]).expect("MockProver::run failed");
        assert!(prover.verify().is_err());
    }

    /// Every one of the 16 share positions contributes to the native output hash.
    ///
    /// `compute_shares_hash_matches_native` covers the in-circuit/native
    /// equivalence once; this test keeps the per-position coverage without
    /// running a separate K=12 prover for every position.
    #[test]
    fn all_16_share_positions_are_hashed() {
        let mut rng = OsRng;
        let blinds: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));

        let correct = shares_hash(blinds, enc_c1_x, enc_c2_x, enc_c1_y, enc_c2_y);

        for i in 0..16 {
            let mut perturbed_enc_c1_x = enc_c1_x;
            perturbed_enc_c1_x[i] += pallas::Base::one();

            assert_ne!(
                shares_hash(blinds, perturbed_enc_c1_x, enc_c2_x, enc_c1_y, enc_c2_y),
                correct,
                "perturbing enc_c1_x[{i}] did not change the shares_hash"
            );
        }
    }

    /// Corrupting a blind factor changes the output.
    #[test]
    fn compute_shares_hash_wrong_blind_fails() {
        let mut rng = OsRng;
        let blinds: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));

        let correct = shares_hash(blinds, enc_c1_x, enc_c2_x, enc_c1_y, enc_c2_y);

        let mut circuit = ComputeSharesHashCircuit {
            blinds,
            enc_c1_x,
            enc_c2_x,
            enc_c1_y,
            enc_c2_y,
        };
        circuit.blinds[0] = pallas::Base::random(&mut rng);

        let prover =
            MockProver::run(12, &circuit, vec![vec![correct]]).expect("MockProver::run failed");
        assert!(prover.verify().is_err());
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

    /// Minimal circuit: computes both in-circuit `shares_hash` paths from the
    /// same witness and constrains their outputs equal without a native oracle.
    #[derive(Clone)]
    struct SharesHashInCircuitEquivalenceCircuit {
        blinds: [pallas::Base; 16],
        enc_c1_x: [pallas::Base; 16],
        enc_c2_x: [pallas::Base; 16],
        enc_c1_y: [pallas::Base; 16],
        enc_c2_y: [pallas::Base; 16],
    }

    impl Default for SharesHashInCircuitEquivalenceCircuit {
        fn default() -> Self {
            Self {
                blinds: [pallas::Base::zero(); 16],
                enc_c1_x: [pallas::Base::zero(); 16],
                enc_c2_x: [pallas::Base::zero(); 16],
                enc_c1_y: [pallas::Base::zero(); 16],
                enc_c2_y: [pallas::Base::zero(); 16],
            }
        }
    }

    impl plonk::Circuit<pallas::Base> for SharesHashInCircuitEquivalenceCircuit {
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
            let mut blind_cells = Vec::with_capacity(16);
            let mut c1x_cells = Vec::with_capacity(16);
            let mut c2x_cells = Vec::with_capacity(16);
            let mut c1y_cells = Vec::with_capacity(16);
            let mut c2y_cells = Vec::with_capacity(16);
            for i in 0..16 {
                blind_cells.push(witness(
                    layouter.namespace(|| format!("blind_{i}")),
                    config.advice,
                    Value::known(self.blinds[i]),
                )?);
                c1x_cells.push(witness(
                    layouter.namespace(|| format!("c1x_{i}")),
                    config.advice,
                    Value::known(self.enc_c1_x[i]),
                )?);
                c2x_cells.push(witness(
                    layouter.namespace(|| format!("c2x_{i}")),
                    config.advice,
                    Value::known(self.enc_c2_x[i]),
                )?);
                c1y_cells.push(witness(
                    layouter.namespace(|| format!("c1y_{i}")),
                    config.advice,
                    Value::known(self.enc_c1_y[i]),
                )?);
                c2y_cells.push(witness(
                    layouter.namespace(|| format!("c2y_{i}")),
                    config.advice,
                    Value::known(self.enc_c2_y[i]),
                )?);
            }

            let blinds_full: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                core::array::from_fn(|i| blind_cells[i].clone());
            let enc_c1_x_full: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                core::array::from_fn(|i| c1x_cells[i].clone());
            let enc_c2_x_full: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                core::array::from_fn(|i| c2x_cells[i].clone());
            let enc_c1_y_full: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                core::array::from_fn(|i| c1y_cells[i].clone());
            let enc_c2_y_full: [AssignedCell<pallas::Base, pallas::Base>; 16] =
                core::array::from_fn(|i| c2y_cells[i].clone());

            let domain_share_comm =
                share_commitment_hash::assign_domain_share_comm(&mut layouter, config.advice)?;
            let full_hash = compute_shares_hash_in_circuit(
                || config.poseidon_chip(),
                layouter.namespace(|| "full shares_hash path"),
                domain_share_comm.clone(),
                blinds_full,
                enc_c1_x_full,
                enc_c2_x_full,
                enc_c1_y_full,
                enc_c2_y_full,
            )?;

            let share_comms: [AssignedCell<pallas::Base, pallas::Base>; 16] = (0..16)
                .map(|i| {
                    hash_share_commitment_in_circuit(
                        config.poseidon_chip(),
                        layouter.namespace(|| format!("from-comms share_comm_{i}")),
                        domain_share_comm.clone(),
                        blind_cells[i].clone(),
                        c1x_cells[i].clone(),
                        c2x_cells[i].clone(),
                        c1y_cells[i].clone(),
                        c2y_cells[i].clone(),
                        i,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .expect("always 16 elements");

            let from_comms_hash = super::compute_shares_hash_from_comms_in_circuit(
                config.poseidon_chip(),
                layouter.namespace(|| "from-comms shares_hash path"),
                share_comms,
            )?;

            layouter.assign_region(
                || "full shares_hash == from-comms shares_hash",
                |mut region| region.constrain_equal(full_hash.cell(), from_comms_hash.cell()),
            )
        }
    }

    /// The from-comms gadget matches the two-level native computation.
    #[test]
    fn shares_hash_from_comms_matches_native() {
        let mut rng = OsRng;
        let blinds: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_x: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c1_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
        let enc_c2_y: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));

        let comms: [pallas::Base; 16] = core::array::from_fn(|i| {
            share_commitment(
                blinds[i],
                enc_c1_x[i],
                enc_c2_x[i],
                enc_c1_y[i],
                enc_c2_y[i],
            )
        });
        let expected = super::shares_hash_from_comms(comms);

        assert_eq!(
            expected,
            shares_hash(blinds, enc_c1_x, enc_c2_x, enc_c1_y, enc_c2_y)
        );

        let circuit = ComputeSharesHashFromCommsCircuit { share_comms: comms };
        let prover =
            MockProver::run(12, &circuit, vec![vec![expected]]).expect("MockProver::run failed");
        assert_eq!(prover.verify(), Ok(()));
    }

    /// The full two-level in-circuit path and the from-comms in-circuit path
    /// agree on the same witnesses without relying on a native expected value.
    #[test]
    fn compute_shares_hash_in_circuit_matches_from_comms_in_circuit() {
        let mut rng = OsRng;
        let circuit = SharesHashInCircuitEquivalenceCircuit {
            blinds: core::array::from_fn(|_| pallas::Base::random(&mut rng)),
            enc_c1_x: core::array::from_fn(|_| pallas::Base::random(&mut rng)),
            enc_c2_x: core::array::from_fn(|_| pallas::Base::random(&mut rng)),
            enc_c1_y: core::array::from_fn(|_| pallas::Base::random(&mut rng)),
            enc_c2_y: core::array::from_fn(|_| pallas::Base::random(&mut rng)),
        };

        // K=13 fits the two complete 17-Poseidon paths used by this equivalence test.
        let prover = MockProver::run(13, &circuit, vec![vec![]]).expect("MockProver::run failed");
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
}
