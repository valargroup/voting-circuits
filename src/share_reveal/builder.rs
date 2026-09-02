//! Share Reveal bundle builder.
//!
//! Constructs the [`Circuit`] and [`Instance`] from high-level inputs
//! (Merkle path, selected commitments, revealed bucket ciphertexts, vote
//! metadata). The builder computes all derived values (shares_hash,
//! vote_commitment, share_nullifier, tree root) so the caller only provides
//! raw witness data.

use voting_crypto_deps::halo2_proofs::circuit::Value;
use voting_crypto_deps::pasta_curves::pallas;

use super::circuit::{share_nullifier_hash, Circuit, Instance};
use crate::{
    bridge::WeightedShareCiphertexts, gadgets::vote_commitment::vote_commitment_hash_v2,
    params::VOTE_COMM_TREE_DEPTH, protocol_hash::poseidon_hash_2,
    shares_hash::shares_hash_from_comms,
};

/// Complete share reveal bundle: circuit + public inputs.
#[derive(Clone, Debug)]
pub struct ShareRevealBundle {
    /// The share reveal circuit with all witnesses populated.
    pub circuit: Circuit,
    /// Public inputs (37 field elements).
    pub instance: Instance,
}

/// Build a share reveal bundle from high-level inputs.
///
/// # Arguments
///
/// - `merkle_auth_path`: The 24 sibling hashes from the vote commitment tree.
/// - `merkle_position`: Leaf position in the vote commitment tree.
/// - `selected_commitments`: The 16 per-share weighted selected commitments
///   from the vote's encrypt-choice bundle
///   ([`crate::encrypt_choice::EncryptChoiceBundle::selected_commitments`]).
/// - `primary_blind`: Blind factor for the revealed share (at `share_index`).
/// - `revealed`: All 16 bucket ciphertext coordinates of the revealed share
///   ([`crate::encrypt_choice::EncryptedWeightedShareOutput::ciphertexts`]).
///   Published as public inputs and bound to
///   `selected_commitments[share_index]` through the weighted
///   selected-commitment hash.
/// - `share_index`: Which of the 16 shares is being revealed (0..15).
/// - `proposal_id`: Proposal identifier (as a field element).
/// - `voting_round_id`: Voting round identifier (as a field element).
/// - `decision_bucket_count`: The proposal's public option count `D` (as a
///   field element); must equal the value bound by the vote's cast proof.
///
/// # Caller contract
///
/// `selected_commitments`, `primary_blind`, and `revealed` are cross-circuit
/// outputs from `encrypt_choice::build_encrypt_choice`. Pass the bundle's
/// `share_blinds[share_index]`, `encrypted_shares[share_index].ciphertexts`,
/// and full `selected_commitments` array unchanged; drawing a fresh blind
/// breaks the selected-commitment constraint and can invalidate the reveal.
/// `proposal_id`, `voting_round_id`, `decision_bucket_count`, and the vote
/// commitment tree witness are authenticated session parameters supplied by
/// the caller.
#[allow(clippy::too_many_arguments)]
pub fn build_share_reveal(
    merkle_auth_path: [pallas::Base; VOTE_COMM_TREE_DEPTH],
    merkle_position: u32,
    selected_commitments: [pallas::Base; 16],
    primary_blind: pallas::Base,
    revealed: &WeightedShareCiphertexts,
    share_index: u32,
    proposal_id: pallas::Base,
    voting_round_id: pallas::Base,
    decision_bucket_count: pallas::Base,
) -> ShareRevealBundle {
    let shares_hash = shares_hash_from_comms(selected_commitments);

    let vote_commitment = vote_commitment_hash_v2(
        voting_round_id,
        shares_hash,
        proposal_id,
        decision_bucket_count,
    );

    let vote_comm_tree_root = {
        let mut current = vote_commitment;
        for (i, sibling) in merkle_auth_path
            .iter()
            .enumerate()
            .take(VOTE_COMM_TREE_DEPTH)
        {
            let bit = (merkle_position >> i) & 1;
            let (left, right) = if bit == 0 {
                (current, *sibling)
            } else {
                (*sibling, current)
            };
            current = poseidon_hash_2(left, right);
        }
        current
    };

    let share_index_fp = pallas::Base::from(share_index as u64);
    let share_nullifier = share_nullifier_hash(vote_commitment, share_index_fp, primary_blind);

    let circuit = Circuit {
        vote_comm_tree_path: Value::known(merkle_auth_path),
        vote_comm_tree_position: Value::known(merkle_position),
        share_comms: selected_commitments.map(Value::known),
        primary_blind: Value::known(primary_blind),
        share_index: Value::known(share_index_fp),
        vote_commitment: Value::known(vote_commitment),
    };

    let instance = Instance::from_parts(
        share_nullifier,
        *revealed,
        proposal_id,
        vote_comm_tree_root,
        voting_round_id,
        decision_bucket_count,
    );

    ShareRevealBundle { circuit, instance }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voting_crypto_deps::halo2_proofs::dev::MockProver;
    use voting_crypto_deps::pasta_curves::pallas;

    use crate::bridge::{selected_share_commitment, CiphertextCoordinates, MAX_DECISION_BUCKETS};

    use super::super::circuit::K;

    /// Deterministic test ciphertext vector for one share.
    fn test_ciphertexts(seed: u64) -> WeightedShareCiphertexts {
        WeightedShareCiphertexts(core::array::from_fn(|bucket| {
            let base = seed + (4 * bucket) as u64;
            CiphertextCoordinates {
                c1_x: pallas::Base::from(base),
                c2_x: pallas::Base::from(base + 1),
                c1_y: pallas::Base::from(base + 2),
                c2_y: pallas::Base::from(base + 3),
            }
        }))
    }

    #[test]
    #[ignore = "long-running Halo2 circuit test; run with `cargo test -- --ignored`"]
    fn test_builder_round_trip() {
        let share_blinds: [pallas::Base; 16] =
            core::array::from_fn(|i| pallas::Base::from(1001u64 + i as u64));
        let ciphertexts: [WeightedShareCiphertexts; 16] =
            core::array::from_fn(|i| test_ciphertexts(10_000 + 1_000 * i as u64));
        let selected_commitments: [pallas::Base; 16] =
            core::array::from_fn(|i| selected_share_commitment(share_blinds[i], &ciphertexts[i]));

        let mut empty_roots = [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH];
        empty_roots[0] = poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero());
        for i in 1..VOTE_COMM_TREE_DEPTH {
            empty_roots[i] = poseidon_hash_2(empty_roots[i - 1], empty_roots[i - 1]);
        }

        let share_idx: u32 = 2;
        let bundle = build_share_reveal(
            empty_roots,
            0,
            selected_commitments,
            share_blinds[share_idx as usize],
            &ciphertexts[share_idx as usize],
            share_idx,
            pallas::Base::from(3u64),
            pallas::Base::from(999u64),
            pallas::Base::from(MAX_DECISION_BUCKETS as u64),
        );

        let prover = MockProver::run(
            K,
            &bundle.circuit,
            vec![bundle.instance.to_halo2_instance()],
        )
        .unwrap();
        assert_eq!(prover.verify(), Ok(()));
    }
}
