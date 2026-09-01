//! Shared formula module for the encrypt-choice (ZKP 1.5) commit-and-prove
//! seam.
//!
//! This module is the authoritative in-tree definition of the two hashes that
//! bind the encrypt-choice proof (ZKP 1.5) to the cast vote proof (ZKP #2)
//! and the share-reveal proof (ZKP #3):
//!
//! ```text
//! selected_comm_i = Poseidon(WEIGHTED_SHARE_COMM_DOMAIN, blind_i,
//!                            for j in 0..8: c1_j_x, c2_j_x, c1_j_y, c2_j_y)
//! bridge          = Poseidon(ENCRYPT_CHOICE_BRIDGE_DOMAIN,
//!                            voting_round_id, proposal_id, decision_bucket_count,
//!                            w_0, selected_comm_0, ..., w_15, selected_comm_15)
//! ```
//!
//! `selected_comm_i` commits one weight share's full 8-bucket ElGamal
//! ciphertext vector (the decision-selected `C2` per bucket). Both coordinates
//! of every point are bound to prevent ciphertext sign-malleability, and the
//! blind keeps the commitment hiding when ZKP #3 later opens it publicly.
//!
//! `bridge` is the public seam value shared by ZKP 1.5 and ZKP #2. It binds
//! the weights to their ciphertext commitments and folds in the round,
//! proposal, and active bucket count so an encrypt-choice proof cannot be
//! replayed under a different voting context.
//!
//! The encrypt-choice circuit, the vote-proof circuit, the share-reveal
//! circuit, the builders, and the tests all call this implementation; none of
//! them may maintain a separate formula copy. Domain tags are owned by
//! [`crate::domain_tags`] and are assigned in-circuit via
//! `assign_advice_from_constant`, baking them into every consuming
//! verification key.

use voting_crypto_deps::halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip,
};
use voting_crypto_deps::halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::{self, Advice, Column},
};
use voting_crypto_deps::pasta_curves::pallas;

use crate::domain_tags;

/// Fixed number of decision buckets in every weighted vote circuit shape.
///
/// Each proposal publicly declares an active `decision_bucket_count = D` with
/// `2 <= D <= MAX_DECISION_BUCKETS`; inactive buckets are proof-bound
/// encryptions of zero. Changing this value changes every circuit shape and
/// verification key that consumes this module.
pub const MAX_DECISION_BUCKETS: usize = 8;

/// Number of weight shares per vote (shared protocol constant).
pub const NUM_SHARES: usize = 16;

/// Poseidon input width of one selected share commitment:
/// domain, blind, then four coordinates per bucket.
pub const SELECTED_COMMITMENT_INPUTS: usize = 2 + 4 * MAX_DECISION_BUCKETS;

/// Poseidon input width of the compact bridge commitment:
/// domain, round, proposal, bucket count, then one `(weight, commitment)`
/// pair per share.
pub const BRIDGE_INPUTS: usize = 4 + 2 * NUM_SHARES;

/// Affine coordinates of one ElGamal ciphertext `(C1, C2)`.
///
/// The field order is also the canonical hash preimage order per bucket:
/// `c1_x, c2_x, c1_y, c2_y` — matching the pre-existing five-input share
/// commitment convention of [`crate::shares_hash`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CiphertextCoordinates {
    /// x-coordinate of `C1 = [r]G`.
    pub c1_x: pallas::Base,
    /// x-coordinate of `C2 = [m]G + [r]PK`.
    pub c2_x: pallas::Base,
    /// y-coordinate of `C1`.
    pub c1_y: pallas::Base,
    /// y-coordinate of `C2`.
    pub c2_y: pallas::Base,
}

/// All [`MAX_DECISION_BUCKETS`] bucket ciphertexts of one weight share, in
/// canonical bucket order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightedShareCiphertexts(pub [CiphertextCoordinates; MAX_DECISION_BUCKETS]);

impl WeightedShareCiphertexts {
    /// Flattens the ciphertext vector into the canonical hash preimage order:
    /// `c1_x, c2_x, c1_y, c2_y` per bucket, buckets ascending.
    pub fn to_preimage(&self) -> [pallas::Base; 4 * MAX_DECISION_BUCKETS] {
        let mut out = [pallas::Base::zero(); 4 * MAX_DECISION_BUCKETS];
        for (bucket, coords) in self.0.iter().enumerate() {
            out[4 * bucket] = coords.c1_x;
            out[4 * bucket + 1] = coords.c2_x;
            out[4 * bucket + 2] = coords.c1_y;
            out[4 * bucket + 3] = coords.c2_y;
        }
        out
    }
}

/// Native selected share commitment:
///
/// ```text
/// selected_comm = Poseidon(WEIGHTED_SHARE_COMM_DOMAIN, blind,
///                          for j in 0..8: c1_j_x, c2_j_x, c1_j_y, c2_j_y)
/// ```
///
/// Native counterpart of [`hash_selected_commitment_in_circuit`].
pub fn selected_share_commitment(
    blind: pallas::Base,
    ciphertexts: &WeightedShareCiphertexts,
) -> pallas::Base {
    let mut message = [pallas::Base::zero(); SELECTED_COMMITMENT_INPUTS];
    message[0] = domain_tags::weighted_share_commitment();
    message[1] = blind;
    message[2..].copy_from_slice(&ciphertexts.to_preimage());
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<SELECTED_COMMITMENT_INPUTS>, 3, 2>::init()
        .hash(message)
}

/// Native compact bridge commitment:
///
/// ```text
/// bridge = Poseidon(ENCRYPT_CHOICE_BRIDGE_DOMAIN, round, proposal, D,
///                   w_0, selected_comm_0, ..., w_15, selected_comm_15)
/// ```
///
/// Native counterpart of [`compute_bridge_in_circuit`].
pub fn bridge_commitment(
    voting_round_id: pallas::Base,
    proposal_id: pallas::Base,
    decision_bucket_count: pallas::Base,
    weights_and_comms: &[(pallas::Base, pallas::Base); NUM_SHARES],
) -> pallas::Base {
    let mut message = [pallas::Base::zero(); BRIDGE_INPUTS];
    message[0] = domain_tags::encrypt_choice_bridge();
    message[1] = voting_round_id;
    message[2] = proposal_id;
    message[3] = decision_bucket_count;
    for (share, (weight, comm)) in weights_and_comms.iter().enumerate() {
        message[4 + 2 * share] = *weight;
        message[5 + 2 * share] = *comm;
    }
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<BRIDGE_INPUTS>, 3, 2>::init()
        .hash(message)
}

/// Assigns a domain-tag constant into `column` so it is baked into the
/// verification key of the consuming circuit.
///
/// Requires the circuit to have called `meta.enable_constant` on a fixed
/// column and `meta.enable_equality` on `column`.
fn assign_domain_constant(
    mut layouter: impl Layouter<pallas::Base>,
    column: Column<Advice>,
    name: &'static str,
    value: pallas::Base,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    layouter.assign_region(
        || name,
        |mut region| region.assign_advice_from_constant(|| name, column, 0, value),
    )
}

/// Computes one selected share commitment in-circuit:
///
/// ```text
/// selected_comm = Poseidon(WEIGHTED_SHARE_COMM_DOMAIN, blind,
///                          coords[0], ..., coords[31])
/// ```
///
/// `coords` must already be in the canonical preimage order
/// (`c1_x, c2_x, c1_y, c2_y` per bucket, buckets ascending) — the same order
/// produced by [`WeightedShareCiphertexts::to_preimage`]. The domain tag is
/// assigned as a circuit constant in `domain_column`. The `index` is used
/// only for namespace labels.
pub(crate) fn hash_selected_commitment_in_circuit(
    chip: PoseidonChip<pallas::Base, 3, 2>,
    mut layouter: impl Layouter<pallas::Base>,
    domain_column: Column<Advice>,
    blind: AssignedCell<pallas::Base, pallas::Base>,
    coords: [AssignedCell<pallas::Base, pallas::Base>; 4 * MAX_DECISION_BUCKETS],
    index: usize,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let domain = assign_domain_constant(
        layouter.namespace(|| format!("selected_comm_{index} domain")),
        domain_column,
        "weighted share commitment domain",
        domain_tags::weighted_share_commitment(),
    )?;

    let mut message = Vec::with_capacity(SELECTED_COMMITMENT_INPUTS);
    message.push(domain);
    message.push(blind);
    message.extend(coords);
    let message: [AssignedCell<pallas::Base, pallas::Base>; SELECTED_COMMITMENT_INPUTS] = message
        .try_into()
        .expect("selected commitment input length is fixed");

    let hasher = PoseidonHash::<
        pallas::Base,
        _,
        poseidon::P128Pow5T3,
        ConstantLength<SELECTED_COMMITMENT_INPUTS>,
        3,
        2,
    >::init(
        chip,
        layouter.namespace(|| format!("selected_comm_{index} Poseidon init")),
    )?;
    hasher.hash(
        layouter.namespace(|| format!("selected_comm_{index}")),
        message,
    )
}

/// Computes the compact bridge commitment in-circuit:
///
/// ```text
/// bridge = Poseidon(ENCRYPT_CHOICE_BRIDGE_DOMAIN, round, proposal, D,
///                   w_0, selected_comm_0, ..., w_15, selected_comm_15)
/// ```
///
/// The context cells (`round`, `proposal`, `bucket_count`) and the per-share
/// weight and commitment cells must already be constrained by the caller
/// (copied from the instance column or derived from constrained witnesses).
/// The domain tag is assigned as a circuit constant in `domain_column`.
///
/// Returns the bridge cell. The caller is responsible for binding it to its
/// public instance slot.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_bridge_in_circuit(
    chip: PoseidonChip<pallas::Base, 3, 2>,
    mut layouter: impl Layouter<pallas::Base>,
    domain_column: Column<Advice>,
    voting_round_id: AssignedCell<pallas::Base, pallas::Base>,
    proposal_id: AssignedCell<pallas::Base, pallas::Base>,
    decision_bucket_count: AssignedCell<pallas::Base, pallas::Base>,
    weights: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES],
    selected_commitments: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES],
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let domain = assign_domain_constant(
        layouter.namespace(|| "bridge domain"),
        domain_column,
        "encrypt choice bridge domain",
        domain_tags::encrypt_choice_bridge(),
    )?;

    let mut message = Vec::with_capacity(BRIDGE_INPUTS);
    message.extend([domain, voting_round_id, proposal_id, decision_bucket_count]);
    for (weight, comm) in weights.into_iter().zip(selected_commitments) {
        message.push(weight);
        message.push(comm);
    }
    let message: [AssignedCell<pallas::Base, pallas::Base>; BRIDGE_INPUTS] =
        message.try_into().expect("bridge input length is fixed");

    let hasher = PoseidonHash::<
        pallas::Base,
        _,
        poseidon::P128Pow5T3,
        ConstantLength<BRIDGE_INPUTS>,
        3,
        2,
    >::init(chip, layouter.namespace(|| "bridge Poseidon init"))?;
    hasher.hash(layouter.namespace(|| "bridge"), message)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ff::{Field, PrimeField};
    use crate::rand::rngs::OsRng;
    use voting_crypto_deps::halo2_gadgets::poseidon::Pow5Config as PoseidonConfig;
    use voting_crypto_deps::halo2_proofs::{
        circuit::{floor_planner, Layouter, Value},
        dev::MockProver,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Fixed, Instance as InstanceColumn},
    };

    fn random_ciphertexts() -> WeightedShareCiphertexts {
        let mut rng = OsRng;
        WeightedShareCiphertexts(core::array::from_fn(|_| CiphertextCoordinates {
            c1_x: pallas::Base::random(&mut rng),
            c2_x: pallas::Base::random(&mut rng),
            c1_y: pallas::Base::random(&mut rng),
            c2_y: pallas::Base::random(&mut rng),
        }))
    }

    #[test]
    fn selected_share_commitment_frozen_vector() {
        let blind = pallas::Base::from(7u64);
        let ciphertexts = WeightedShareCiphertexts(core::array::from_fn(|bucket| {
            let base = (4 * bucket) as u64;
            CiphertextCoordinates {
                c1_x: pallas::Base::from(100 + base),
                c2_x: pallas::Base::from(101 + base),
                c1_y: pallas::Base::from(102 + base),
                c2_y: pallas::Base::from(103 + base),
            }
        }));

        let comm = selected_share_commitment(blind, &ciphertexts);

        // Frozen output: any change to the domain tag, preimage order, or
        // Poseidon parameters must be caught here.
        let expected: [u8; 32] = [
            0xc1, 0x9b, 0x9e, 0x4a, 0x5d, 0x36, 0xc2, 0x1e, 0x9d, 0xfd, 0xab, 0xdc, 0x3d, 0x71,
            0x49, 0x36, 0xa0, 0x4a, 0x1f, 0x55, 0x6c, 0x25, 0x82, 0xdf, 0x37, 0x9b, 0xc5, 0xdc,
            0xa9, 0x3c, 0x56, 0x14,
        ];
        if comm.to_repr() != expected {
            panic!(
                "selected_share_commitment frozen vector mismatch; if intentional, update to:\n{:?}",
                comm.to_repr()
            );
        }
    }

    #[test]
    fn bridge_commitment_frozen_vector() {
        let weights_and_comms: [(pallas::Base, pallas::Base); NUM_SHARES] =
            core::array::from_fn(|i| {
                (
                    pallas::Base::from(1000 + i as u64),
                    pallas::Base::from(2000 + i as u64),
                )
            });

        let bridge = bridge_commitment(
            pallas::Base::from(7u64),
            pallas::Base::from(3u64),
            pallas::Base::from(5u64),
            &weights_and_comms,
        );

        let expected: [u8; 32] = [
            0xdb, 0xa8, 0x34, 0xbe, 0x26, 0x32, 0x2a, 0x62, 0x8e, 0xf2, 0xbd, 0x8b, 0xf9, 0x00,
            0x86, 0x55, 0x29, 0x7f, 0xc6, 0xef, 0x79, 0x1b, 0xf3, 0xef, 0x6c, 0x3f, 0x7b, 0x45,
            0xad, 0x66, 0x1b, 0x31,
        ];
        if bridge.to_repr() != expected {
            panic!(
                "bridge_commitment frozen vector mismatch; if intentional, update to:\n{:?}",
                bridge.to_repr()
            );
        }
    }

    #[test]
    fn selected_share_commitment_is_sensitive_to_every_preimage_slot() {
        let mut rng = OsRng;
        let blind = pallas::Base::random(&mut rng);
        let ciphertexts = random_ciphertexts();
        let baseline = selected_share_commitment(blind, &ciphertexts);

        let altered_blind = selected_share_commitment(blind + pallas::Base::one(), &ciphertexts);
        assert_ne!(baseline, altered_blind, "blind slot must bind");

        for slot in 0..4 * MAX_DECISION_BUCKETS {
            let mut preimage = ciphertexts.to_preimage();
            preimage[slot] += pallas::Base::one();
            let altered =
                WeightedShareCiphertexts(core::array::from_fn(|bucket| CiphertextCoordinates {
                    c1_x: preimage[4 * bucket],
                    c2_x: preimage[4 * bucket + 1],
                    c1_y: preimage[4 * bucket + 2],
                    c2_y: preimage[4 * bucket + 3],
                }));
            assert_ne!(
                baseline,
                selected_share_commitment(blind, &altered),
                "coordinate slot {slot} must bind"
            );
        }
    }

    #[test]
    fn bridge_commitment_is_sensitive_to_every_preimage_slot() {
        let mut rng = OsRng;
        let round = pallas::Base::random(&mut rng);
        let proposal = pallas::Base::random(&mut rng);
        let bucket_count = pallas::Base::random(&mut rng);
        let weights_and_comms: [(pallas::Base, pallas::Base); NUM_SHARES] =
            core::array::from_fn(|_| {
                (
                    pallas::Base::random(&mut rng),
                    pallas::Base::random(&mut rng),
                )
            });
        let baseline = bridge_commitment(round, proposal, bucket_count, &weights_and_comms);

        assert_ne!(
            baseline,
            bridge_commitment(
                round + pallas::Base::one(),
                proposal,
                bucket_count,
                &weights_and_comms
            ),
            "round slot must bind"
        );
        assert_ne!(
            baseline,
            bridge_commitment(
                round,
                proposal + pallas::Base::one(),
                bucket_count,
                &weights_and_comms
            ),
            "proposal slot must bind"
        );
        assert_ne!(
            baseline,
            bridge_commitment(
                round,
                proposal,
                bucket_count + pallas::Base::one(),
                &weights_and_comms
            ),
            "bucket count slot must bind"
        );

        for share in 0..NUM_SHARES {
            let mut altered = weights_and_comms;
            altered[share].0 += pallas::Base::one();
            assert_ne!(
                baseline,
                bridge_commitment(round, proposal, bucket_count, &altered),
                "weight slot {share} must bind"
            );

            let mut altered = weights_and_comms;
            altered[share].1 += pallas::Base::one();
            assert_ne!(
                baseline,
                bridge_commitment(round, proposal, bucket_count, &altered),
                "commitment slot {share} must bind"
            );
        }
    }

    #[test]
    fn bridge_and_selected_commitment_domains_disjoint_preimage_shapes() {
        // The two hashes have different arities (34 vs 36), so a collision
        // between them would require a Poseidon-level break; this test pins
        // the arities so a refactor cannot silently make them collide.
        assert_eq!(SELECTED_COMMITMENT_INPUTS, 34);
        assert_eq!(BRIDGE_INPUTS, 36);
    }

    #[test]
    fn vote_proof_readme_tracks_selected_commitment_shape() {
        let readme = include_str!("vote_proof/README.md");
        let description = readme
            .split_once("- **selected_comm_i**:")
            .expect("README must document selected_comm_i")
            .1
            .split_once("- **voting_round_id")
            .expect("README selected_comm_i description must have a section boundary")
            .0;

        let expected_arity = format!("{SELECTED_COMMITMENT_INPUTS}-input Poseidon commitment");
        let expected_buckets = format!("all {MAX_DECISION_BUCKETS} bucket");
        assert!(
            description.contains(&expected_arity),
            "README selected_comm_i description must contain `{expected_arity}`"
        );
        assert!(
            description.contains(&expected_buckets),
            "README selected_comm_i description must contain `{expected_buckets}`"
        );
    }

    // ---------------------------------------------------------------
    // In-circuit equivalence
    // ---------------------------------------------------------------

    /// Minimal Poseidon-only circuit computing one selected commitment and
    /// one bridge, binding both to the instance column.
    #[derive(Clone)]
    struct EquivalenceCircuit {
        blind: Value<pallas::Base>,
        coords: [Value<pallas::Base>; 4 * MAX_DECISION_BUCKETS],
        round: Value<pallas::Base>,
        proposal: Value<pallas::Base>,
        bucket_count: Value<pallas::Base>,
        weights: [Value<pallas::Base>; NUM_SHARES],
        comms: [Value<pallas::Base>; NUM_SHARES],
    }

    impl Default for EquivalenceCircuit {
        fn default() -> Self {
            Self {
                blind: Value::unknown(),
                coords: [Value::unknown(); 4 * MAX_DECISION_BUCKETS],
                round: Value::unknown(),
                proposal: Value::unknown(),
                bucket_count: Value::unknown(),
                weights: [Value::unknown(); NUM_SHARES],
                comms: [Value::unknown(); NUM_SHARES],
            }
        }
    }

    fn assign_free(
        layouter: &mut impl Layouter<pallas::Base>,
        column: Column<Advice>,
        name: &'static str,
        value: Value<pallas::Base>,
    ) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
        layouter.assign_region(
            || name,
            |mut region| region.assign_advice(|| name, column, 0, || value),
        )
    }

    #[derive(Clone)]
    struct EquivalenceConfig {
        primary: Column<InstanceColumn>,
        advice: Column<Advice>,
        poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    }

    impl Circuit<pallas::Base> for EquivalenceCircuit {
        type Config = EquivalenceConfig;
        type FloorPlanner = floor_planner::V1;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
            let primary = meta.instance_column();
            meta.enable_equality(primary);

            let advices: [Column<Advice>; 5] = core::array::from_fn(|_| meta.advice_column());
            for col in &advices {
                meta.enable_equality(*col);
            }
            let fixed: [Column<Fixed>; 6] = core::array::from_fn(|_| meta.fixed_column());
            let constants = meta.fixed_column();
            meta.enable_constant(constants);

            let poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
                meta,
                advices[1..4].try_into().unwrap(),
                advices[4],
                fixed[..3].try_into().unwrap(),
                fixed[3..].try_into().unwrap(),
            );

            EquivalenceConfig {
                primary,
                advice: advices[0],
                poseidon_config,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<pallas::Base>,
        ) -> Result<(), plonk::Error> {
            let blind = assign_free(&mut layouter, config.advice, "blind", self.blind)?;
            let coords: [AssignedCell<pallas::Base, pallas::Base>; 4 * MAX_DECISION_BUCKETS] = {
                let mut cells = Vec::with_capacity(4 * MAX_DECISION_BUCKETS);
                for (i, value) in self.coords.iter().enumerate() {
                    cells.push(layouter.assign_region(
                        || format!("coord {i}"),
                        |mut region| region.assign_advice(|| "coord", config.advice, 0, || *value),
                    )?);
                }
                cells.try_into().expect("64 coordinate cells")
            };

            let selected = hash_selected_commitment_in_circuit(
                PoseidonChip::construct(config.poseidon_config.clone()),
                layouter.namespace(|| "selected commitment"),
                config.advice,
                blind,
                coords,
                0,
            )?;
            layouter.constrain_instance(selected.cell(), config.primary, 0)?;

            let round = assign_free(&mut layouter, config.advice, "round", self.round)?;
            let proposal = assign_free(&mut layouter, config.advice, "proposal", self.proposal)?;
            let bucket_count = assign_free(
                &mut layouter,
                config.advice,
                "bucket count",
                self.bucket_count,
            )?;
            let weights: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES] = {
                let mut cells = Vec::with_capacity(NUM_SHARES);
                for value in self.weights.iter() {
                    cells.push(assign_free(&mut layouter, config.advice, "weight", *value)?);
                }
                cells.try_into().expect("16 weight cells")
            };
            let comms: [AssignedCell<pallas::Base, pallas::Base>; NUM_SHARES] = {
                let mut cells = Vec::with_capacity(NUM_SHARES);
                for value in self.comms.iter() {
                    cells.push(assign_free(&mut layouter, config.advice, "comm", *value)?);
                }
                cells.try_into().expect("16 commitment cells")
            };

            let bridge = compute_bridge_in_circuit(
                PoseidonChip::construct(config.poseidon_config.clone()),
                layouter.namespace(|| "bridge"),
                config.advice,
                round,
                proposal,
                bucket_count,
                weights,
                comms,
            )?;
            layouter.constrain_instance(bridge.cell(), config.primary, 1)
        }
    }

    #[test]
    fn in_circuit_hashes_match_native() {
        let mut rng = OsRng;
        let blind = pallas::Base::random(&mut rng);
        let ciphertexts = random_ciphertexts();
        let round = pallas::Base::from(7u64);
        let proposal = pallas::Base::from(3u64);
        let bucket_count = pallas::Base::from(5u64);
        let weights: [pallas::Base; NUM_SHARES] =
            core::array::from_fn(|i| pallas::Base::from(10 + i as u64));
        let comms: [pallas::Base; NUM_SHARES] =
            core::array::from_fn(|_| pallas::Base::random(&mut rng));

        let expected_selected = selected_share_commitment(blind, &ciphertexts);
        let weights_and_comms: [(pallas::Base, pallas::Base); NUM_SHARES] =
            core::array::from_fn(|i| (weights[i], comms[i]));
        let expected_bridge = bridge_commitment(round, proposal, bucket_count, &weights_and_comms);

        let circuit = EquivalenceCircuit {
            blind: Value::known(blind),
            coords: ciphertexts.to_preimage().map(Value::known),
            round: Value::known(round),
            proposal: Value::known(proposal),
            bucket_count: Value::known(bucket_count),
            weights: weights.map(Value::known),
            comms: comms.map(Value::known),
        };

        let prover = MockProver::run(13, &circuit, vec![vec![expected_selected, expected_bridge]])
            .expect("mock prover runs");
        assert_eq!(prover.verify(), Ok(()));
    }

    #[test]
    fn in_circuit_hashes_reject_wrong_instance() {
        let circuit = EquivalenceCircuit {
            blind: Value::known(pallas::Base::one()),
            coords: [Value::known(pallas::Base::one()); 4 * MAX_DECISION_BUCKETS],
            round: Value::known(pallas::Base::one()),
            proposal: Value::known(pallas::Base::one()),
            bucket_count: Value::known(pallas::Base::one()),
            weights: [Value::known(pallas::Base::one()); NUM_SHARES],
            comms: [Value::known(pallas::Base::one()); NUM_SHARES],
        };

        let prover = MockProver::run(
            13,
            &circuit,
            vec![vec![pallas::Base::zero(), pallas::Base::zero()]],
        )
        .expect("mock prover runs");
        assert_ne!(prover.verify(), Ok(()));
    }
}
