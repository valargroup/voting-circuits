//! Per-share commitment hash used by ZKP #2 and ZKP #3.
//!
//! ```text
//! share_commitment = Poseidon(DOMAIN_SHARE_COMM, blind, c1_x, c2_x, c1_y, c2_y)
//! ```
//!
//! The domain tag is assigned as a circuit constant so the value is baked into
//! the verification keys. The blind remains a private witness and keeps
//! observers from recomputing `shares_hash` from posted encrypted shares.

use halo2_gadgets::poseidon::Pow5Chip as PoseidonChip;
use halo2_proofs::{
    circuit::{AssignedCell, Layouter},
    plonk::{self, Advice, Column},
};
use pasta_curves::pallas;

pub use crate::domain_tags::DOMAIN_SHARE_COMM;
use crate::protocol_hash::{poseidon_hash, poseidon_hash_in_circuit};

/// Out-of-circuit per-share blinded commitment.
///
/// The y-coordinates bind the commitment to the exact curve point, not just
/// the x-coordinate. Without them, an attacker can negate the El Gamal
/// ciphertext without invalidating the ZKP and corrupt the homomorphic tally.
pub fn share_commitment(
    blind: pallas::Base,
    c1_x: pallas::Base,
    c2_x: pallas::Base,
    c1_y: pallas::Base,
    c2_y: pallas::Base,
) -> pallas::Base {
    poseidon_hash([
        pallas::Base::from(DOMAIN_SHARE_COMM),
        blind,
        c1_x,
        c2_x,
        c1_y,
        c2_y,
    ])
}

/// Assigns `DOMAIN_SHARE_COMM` as a circuit constant.
pub(crate) fn assign_domain_share_comm(
    layouter: &mut impl Layouter<pallas::Base>,
    advice: Column<Advice>,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    layouter.assign_region(
        || "DOMAIN_SHARE_COMM constant",
        |mut region| {
            region.assign_advice_from_constant(
                || "domain_share_comm",
                advice,
                0,
                pallas::Base::from(DOMAIN_SHARE_COMM),
            )
        },
    )
}

/// Synthesizes one per-share commitment.
///
/// The caller must pass a `domain_share_comm` cell assigned by
/// [`assign_domain_share_comm`] so the domain tag is constrained as a constant.
pub(crate) fn share_commitment_poseidon(
    chip: PoseidonChip<pallas::Base, 3, 2>,
    layouter: &mut impl Layouter<pallas::Base>,
    label: &str,
    domain_share_comm: AssignedCell<pallas::Base, pallas::Base>,
    blind: AssignedCell<pallas::Base, pallas::Base>,
    enc_c1_x: AssignedCell<pallas::Base, pallas::Base>,
    enc_c2_x: AssignedCell<pallas::Base, pallas::Base>,
    enc_c1_y: AssignedCell<pallas::Base, pallas::Base>,
    enc_c2_y: AssignedCell<pallas::Base, pallas::Base>,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    poseidon_hash_in_circuit(
        chip,
        layouter.namespace(|| label),
        label,
        [
            domain_share_comm,
            blind,
            enc_c1_x,
            enc_c2_x,
            enc_c1_y,
            enc_c2_y,
        ],
    )
}
