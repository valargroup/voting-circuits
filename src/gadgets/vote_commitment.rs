//! Vote Commitment integrity gadget.
//!
//! Authoritative in-tree definition of the 5-input Poseidon hash used by both
//! ZKP #2 (vote proof, condition 12) and ZKP #3 (share reveal, condition 2).
//!
//! ```text
//! vote_commitment = Poseidon(DOMAIN_VC, voting_round_id,
//!                            shares_hash, proposal_id, vote_decision)
//! ```
//!
//! The domain tag bakes into the verification key, preventing a
//! client misuse driving the honest circuit from substituting VAN
//! commitments for vote commitments in the shared tree.

use pasta_curves::pallas;

use halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk,
};

pub use crate::domain_tags::DOMAIN_VC;

// ================================================================
// Out-of-circuit helper
// ================================================================

/// Out-of-circuit vote commitment hash.
///
/// This is the authoritative native implementation of the vote commitment
/// preimage shared by vote proof and share reveal:
/// ```text
/// Poseidon(DOMAIN_VC, voting_round_id, shares_hash, proposal_id, vote_decision)
/// ```
///
/// Used by builders and tests to compute the expected vote commitment.
/// Must produce identical output to the in-circuit gadget.
pub(crate) fn vote_commitment_hash(
    voting_round_id: pallas::Base,
    shares_hash: pallas::Base,
    proposal_id: pallas::Base,
    vote_decision: pallas::Base,
) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<5>, 3, 2>::init().hash([
        pallas::Base::from(DOMAIN_VC),
        voting_round_id,
        shares_hash,
        proposal_id,
        vote_decision,
    ])
}

// ================================================================
// In-circuit gadget
// ================================================================

/// In-circuit vote commitment hash.
///
/// Computes `Poseidon(domain_vc, voting_round_id, shares_hash, proposal_id, vote_decision)`
/// matching the out-of-circuit helper above.
///
/// Takes a `PoseidonConfig` so it can be used by any circuit that
/// configures a compatible Poseidon chip (P128Pow5T3, width 3, rate 2).
/// The `domain_vc` cell must be assigned via `assign_advice_from_constant`
/// so the value is baked into the verification key.
///
/// Used by ZKP #2 (vote proof, condition 12) and ZKP #3 (share reveal,
/// condition 2).
pub(crate) fn vote_commitment_poseidon(
    poseidon_config: &PoseidonConfig<pallas::Base, 3, 2>,
    layouter: &mut impl Layouter<pallas::Base>,
    label: &str,
    domain_vc: AssignedCell<pallas::Base, pallas::Base>,
    voting_round_id: AssignedCell<pallas::Base, pallas::Base>,
    shares_hash: AssignedCell<pallas::Base, pallas::Base>,
    proposal_id: AssignedCell<pallas::Base, pallas::Base>,
    vote_decision: AssignedCell<pallas::Base, pallas::Base>,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let message = [
        domain_vc,
        voting_round_id,
        shares_hash,
        proposal_id,
        vote_decision,
    ];
    let hasher =
        PoseidonHash::<pallas::Base, _, poseidon::P128Pow5T3, ConstantLength<5>, 3, 2>::init(
            PoseidonChip::construct(poseidon_config.clone()),
            layouter.namespace(|| format!("{label} Poseidon init")),
        )?;
    hasher.hash(
        layouter.namespace(|| format!("{label} Poseidon(DOMAIN_VC, ...)")),
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::PrimeField;

    fn base_from_repr(bytes: [u8; 32]) -> pallas::Base {
        pallas::Base::from_repr(bytes).expect("frozen vector must be canonical")
    }

    #[test]
    fn vote_commitment_hash_frozen_vector() {
        let actual = vote_commitment_hash(
            pallas::Base::from(42u64),
            pallas::Base::from(100u64),
            pallas::Base::from(7u64),
            pallas::Base::from(1u64),
        );

        assert_eq!(
            actual,
            base_from_repr([
                246, 84, 48, 178, 227, 178, 234, 71, 2, 178, 177, 211, 238, 120, 238, 157, 174, 5,
                29, 244, 76, 128, 250, 245, 139, 137, 84, 246, 108, 197, 47, 31,
            ])
        );
    }
}
