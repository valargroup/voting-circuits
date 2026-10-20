//! Voting-owned snapshot of Orchard's Halo2 composition implementation.
//!
//! Inherited from zakura-core/common at 0b19c5cfff9fcf8d78bbd0424ea884aca63a921e.
//! See README.md for provenance, permitted adaptations, and acceptance criteria.
#![allow(dead_code)]

mod commit_ivk;
pub(crate) mod constants;
mod gadget;
mod note_commit;
mod spec;
pub(crate) mod witness;

mod backend {
    pub(crate) use crate::pasta_curves::group::{self, ff};
    pub(crate) use crate::{halo2_gadgets, halo2_proofs, pasta_curves};
}

pub use commit_ivk::{gadgets::commit_ivk, CommitIvkChip, CommitIvkConfig};
pub use constants::fixed_bases::spend_auth_g::{
    generator as spend_auth_generator, U as SPEND_AUTH_U, Z as SPEND_AUTH_Z,
};
pub use constants::fixed_bases::{
    OrchardBaseFieldBases, OrchardFixedBases, OrchardFixedBasesFull, COMMIT_IVK_PERSONALIZATION,
    NOTE_COMMITMENT_PERSONALIZATION,
};
pub use constants::{
    OrchardCommitDomains, OrchardHashDomains, L_ORCHARD_BASE, L_VALUE, MERKLE_DEPTH_ORCHARD,
};
pub use gadget::{
    add_chip::{AddChip, AddConfig},
    assign_free_advice, derive_nullifier, AddInstruction,
};
pub use note_commit::{gadgets::note_commit, NoteCommitChip, NoteCommitConfig};

#[cfg(test)]
mod key_vectors;
#[cfg(all(test, feature = "rand"))]
mod tests;
