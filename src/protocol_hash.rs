//! Canonical Poseidon primitives used by protocol hash helpers.
//!
//! These wrappers centralize the P128Pow5T3, width 3, rate 2 parameters so
//! native helpers, in-circuit gadgets, and test oracles cannot drift into
//! separate definitions.

use halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip,
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk,
};
use pasta_curves::pallas;

/// Computes the protocol Poseidon hash for a fixed number of field inputs.
pub(crate) fn poseidon_hash<const N: usize>(inputs: [pallas::Base; N]) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<N>, 3, 2>::init().hash(inputs)
}

/// Computes `Poseidon(a, b)` with P128Pow5T3, width 3, rate 2.
///
/// Used for Poseidon Merkle path computation and tests. This is the same hash
/// function used by `vote_commitment_tree::MerkleHashVote::combine`.
pub fn poseidon_hash_2(a: pallas::Base, b: pallas::Base) -> pallas::Base {
    poseidon_hash([a, b])
}

/// Computes `Poseidon(a, b, c)` with P128Pow5T3, width 3, rate 2.
pub(crate) fn poseidon_hash_3(a: pallas::Base, b: pallas::Base, c: pallas::Base) -> pallas::Base {
    poseidon_hash([a, b, c])
}

/// Synthesizes the protocol Poseidon hash for a fixed number of field inputs.
pub(crate) fn poseidon_hash_in_circuit<const N: usize>(
    chip: PoseidonChip<pallas::Base, 3, 2>,
    mut layouter: impl Layouter<pallas::Base>,
    label: &str,
    inputs: [AssignedCell<pallas::Base, pallas::Base>; N],
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let hasher =
        PoseidonHash::<pallas::Base, _, poseidon::P128Pow5T3, ConstantLength<N>, 3, 2>::init(
            chip,
            layouter.namespace(|| format!("{label} Poseidon init")),
        )?;
    hasher.hash(
        layouter.namespace(|| format!("{label} Poseidon hash")),
        inputs,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poseidon_hash_2_vectors_are_frozen() {
        let vectors = [
            (
                (pallas::Base::zero(), pallas::Base::zero()),
                pallas::Base::from_raw([
                    2_216_552_556_888_936_826,
                    5_603_939_996_197_372_455,
                    14_405_608_055_658_021_993,
                    257_921_357_662_939_124,
                ]),
            ),
            (
                (pallas::Base::from(1u64), pallas::Base::from(2u64)),
                pallas::Base::from_raw([
                    9_905_064_880_589_628_236,
                    4_271_102_183_789_020_803,
                    219_221_386_113_680_107,
                    3_843_160_293_085_321_624,
                ]),
            ),
            (
                (pallas::Base::from(42u64), pallas::Base::from(99u64)),
                pallas::Base::from_raw([
                    10_378_353_436_149_985_154,
                    12_698_039_334_634_083_230,
                    17_905_004_260_684_481_651,
                    1_912_763_018_532_593_885,
                ]),
            ),
        ];

        for ((a, b), expected) in vectors {
            assert_eq!(poseidon_hash_2(a, b), expected);
        }
    }

    #[test]
    fn poseidon_hash_3_vectors_are_frozen() {
        let vectors = [
            (
                (
                    pallas::Base::zero(),
                    pallas::Base::zero(),
                    pallas::Base::zero(),
                ),
                pallas::Base::from_raw([
                    6_945_653_648_026_230_712,
                    9_935_878_616_126_350_886,
                    3_440_240_271_760_316_840,
                    1_075_975_577_291_643_089,
                ]),
            ),
            (
                (
                    pallas::Base::from(1u64),
                    pallas::Base::from(2u64),
                    pallas::Base::from(3u64),
                ),
                pallas::Base::from_raw([
                    5_801_387_003_377_482_986,
                    10_049_809_810_059_466_806,
                    6_856_205_520_035_286_116,
                    1_777_587_896_617_418_398,
                ]),
            ),
            (
                (
                    pallas::Base::from(42u64),
                    pallas::Base::from(99u64),
                    pallas::Base::from(123u64),
                ),
                pallas::Base::from_raw([
                    1_718_157_347_811_741_002,
                    3_896_949_896_271_120_622,
                    9_242_511_883_964_356_336,
                    3_076_042_013_626_979_766,
                ]),
            ),
        ];

        for ((a, b, c), expected) in vectors {
            assert_eq!(poseidon_hash_3(a, b, c), expected);
        }
    }
}
