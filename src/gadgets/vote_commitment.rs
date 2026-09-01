//! Vote Commitment integrity gadget.
//!
//! Authoritative in-tree definition of the 5-input Poseidon hash used by both
//! ZKP #2 (vote proof, condition 12′) and ZKP #3 (share reveal, condition 2).
//!
//! ```text
//! vote_commitment = Poseidon(DOMAIN_VC_V2, voting_round_id,
//!                            shares_hash, proposal_id, decision_bucket_count)
//! ```
//!
//! The domain tag bakes into the verification key, preventing a
//! client misuse driving the honest circuit from substituting VAN
//! commitments for vote commitments in the shared tree.

use voting_crypto_deps::pasta_curves::pallas;

use voting_crypto_deps::halo2_gadgets::poseidon::{
    primitives::{self as poseidon, ConstantLength},
    Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
};
use voting_crypto_deps::halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk,
};

pub use crate::domain_tags::DOMAIN_VC_V2;

// ================================================================
// Out-of-circuit helper
// ================================================================

/// Out-of-circuit weighted (v2) vote commitment hash.
///
/// Authoritative native implementation of the versioned vote commitment used
/// by the weighted encrypt-choice design:
/// ```text
/// Poseidon(DOMAIN_VC_V2, voting_round_id, shares_hash, proposal_id,
///          decision_bucket_count)
/// ```
///
/// The plaintext `vote_decision` slot of v1 is replaced by the public
/// `decision_bucket_count`; the decision itself is bound only through the
/// committed one-hot ciphertext vectors inside `shares_hash`. Binding the
/// bucket count prevents replaying a commitment under a proposal with a
/// different option count.
pub fn vote_commitment_hash_v2(
    voting_round_id: pallas::Base,
    shares_hash: pallas::Base,
    proposal_id: pallas::Base,
    decision_bucket_count: pallas::Base,
) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<5>, 3, 2>::init().hash([
        pallas::Base::from(DOMAIN_VC_V2),
        voting_round_id,
        shares_hash,
        proposal_id,
        decision_bucket_count,
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
///
/// The v2 (weighted) formula reuses this same gadget: the caller passes a
/// `DOMAIN_VC_V2` constant cell as `domain_vc` and the public
/// `decision_bucket_count` cell in the `vote_decision` slot, matching
/// [`vote_commitment_hash_v2`].
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
    use crate::ff::PrimeField;

    #[test]
    fn vote_commitment_hash_v2_frozen_vector() {
        let actual = vote_commitment_hash_v2(
            pallas::Base::from(42u64),
            pallas::Base::from(100u64),
            pallas::Base::from(7u64),
            pallas::Base::from(5u64),
        );

        let expected: [u8; 32] = [
            146, 133, 76, 184, 20, 171, 210, 83, 163, 222, 84, 105, 115, 4, 147, 89, 243, 26, 124,
            138, 88, 94, 238, 149, 247, 235, 37, 179, 209, 162, 130, 21,
        ];
        if actual.to_repr() != expected {
            panic!(
                "vote_commitment_hash_v2 frozen vector mismatch; if intentional, update to:\n{:?}",
                actual.to_repr()
            );
        }
    }
}
