//! Orchard fixed bases.

use std::vec::Vec;

use super::{L_ORCHARD_SCALAR, L_VALUE};

use crate::legacy::backend::halo2_gadgets::ecc::{
    chip::{BaseFieldElem, FixedPoint, FullScalar, ShortScalar},
    FixedPoints,
};

use crate::legacy::backend::pasta_curves::pallas;

/// Precomputed table for the `CommitIvk` commitment randomness base.
#[path = "fixed_bases/commit_ivk_r.rs"]
pub mod commit_ivk_r;
#[path = "fixed_bases/lagrange_coeffs.rs"]
mod lagrange_coeffs;
/// Precomputed table for the `NoteCommit` commitment randomness base.
#[path = "fixed_bases/note_commit_r.rs"]
pub mod note_commit_r;
/// Precomputed table for the nullifier base `K^Orchard`.
#[path = "fixed_bases/nullifier_k.rs"]
pub mod nullifier_k;
/// Precomputed table for the spend authorization base `G^Orchard`.
#[path = "fixed_bases/spend_auth_g.rs"]
pub mod spend_auth_g;
/// Precomputed table for the value commitment randomness base.
#[path = "fixed_bases/value_commit_r.rs"]
pub mod value_commit_r;
/// Precomputed table for the value commitment value base.
#[path = "fixed_bases/value_commit_v.rs"]
pub mod value_commit_v;

/// SWU hash-to-curve personalization for the spending key base point and
/// the nullifier base point K^Orchard
pub const ORCHARD_PERSONALIZATION: &str = "z.cash:Orchard";

/// SWU hash-to-curve personalization for the value commitment generator
pub const VALUE_COMMITMENT_PERSONALIZATION: &str = "z.cash:Orchard-cv";

/// SWU hash-to-curve value for the value commitment generator
pub const VALUE_COMMITMENT_V_BYTES: [u8; 1] = *b"v";

/// SWU hash-to-curve value for the value commitment generator
pub const VALUE_COMMITMENT_R_BYTES: [u8; 1] = *b"r";

/// SWU hash-to-curve personalization for the note commitment generator
pub const NOTE_COMMITMENT_PERSONALIZATION: &str = "z.cash:Orchard-NoteCommit";

/// SWU hash-to-curve personalization for the IVK commitment generator
pub const COMMIT_IVK_PERSONALIZATION: &str = "z.cash:Orchard-CommitIvk";

/// Window size for fixed-base scalar multiplication
pub const FIXED_BASE_WINDOW_SIZE: usize = 3;

/// $2^{`FIXED_BASE_WINDOW_SIZE`}$
pub const H: usize = 1 << FIXED_BASE_WINDOW_SIZE;

/// Number of windows for a full-width scalar
pub const NUM_WINDOWS: usize = L_ORCHARD_SCALAR.div_ceil(FIXED_BASE_WINDOW_SIZE);

/// Number of windows for a short signed scalar
pub const NUM_WINDOWS_SHORT: usize = L_VALUE.div_ceil(FIXED_BASE_WINDOW_SIZE);

/// Fixed bases used in scalar mul where the scalar is a base field element.
///
/// The ECC chip's `FixedPoints::Base` associated type must be a single type,
/// so both `NullifierK` and `SpendAuthGBase` are wrapped in this enum.
/// `SpendAuthGBase` reuses the same generator and U/Z tables as
/// `OrchardFixedBasesFull::SpendAuthG` (same 85-window structure over the
/// 255-bit pallas base field), allowing `FixedPointBaseField::mul` to accept
/// an `AssignedCell` directly — no variable-base `NonIdentityPoint` witness
/// needed.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OrchardBaseFieldBases {
    /// Nullifier key generator $\mathcal{K}^{\mathsf{Orchard}}$, used in
    /// DeriveNullifier.
    NullifierK,
    /// SpendAuthG with full 85-window tables, accepting a base-field scalar via
    /// an `AssignedCell` input (no variable-base `NonIdentityPoint` witness
    /// required). Reuses the same generator and U/Z tables as
    /// `OrchardFixedBasesFull::SpendAuthG`.
    SpendAuthGBase,
}

/// Fixed bases used in scalar mul where the scalar is a short (64-bit) signed
/// value.
///
/// `FixedPoints::ShortScalar` must be a single type, so both `ValueCommitV` and
/// `SpendAuthGShort` are wrapped in this enum. `SpendAuthGShort` uses the same
/// generator as `SpendAuthG` but with 22-window precomputed tables, enabling
/// `FixedPointShort::mul` for short signed scalars.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OrchardShortScalarBases {
    /// Value commitment generator for Orchard (original short base).
    ValueCommitV,
    /// SpendAuthG with short (22-window) precomputed tables, for multiplication
    /// by short signed scalars via `FixedPointShort::mul`.
    SpendAuthGShort,
}

/// Carrier type for the `FixedPoints<pallas::Affine>` impl that wires Orchard's
/// per-slot fixed-base enums into halo2_gadgets's ECC, Sinsemilla, and Merkle
/// chips. Carries no value-level state — the concrete fixed bases live in
/// `OrchardFixedBasesFull`, `OrchardBaseFieldBases`, and `OrchardShortScalarBases`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct OrchardFixedBases;

impl From<NullifierK> for OrchardBaseFieldBases {
    fn from(_: NullifierK) -> Self {
        Self::NullifierK
    }
}

impl From<ValueCommitV> for OrchardShortScalarBases {
    fn from(_: ValueCommitV) -> Self {
        Self::ValueCommitV
    }
}

/// The Orchard fixed bases used in scalar mul with full-width scalars.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OrchardFixedBasesFull {
    /// Randomness base for the `CommitIvk` commitment.
    CommitIvkR,
    /// Randomness base for the `NoteCommit` commitment.
    NoteCommitR,
    /// Randomness base for value commitments.
    ValueCommitR,
    /// Spend authorization base `G^Orchard`.
    SpendAuthG,
}

/// NullifierK is used in scalar mul with a base field element.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct NullifierK;

/// ValueCommitV is used in scalar mul with a short signed scalar.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ValueCommitV;

impl FixedPoints<pallas::Affine> for OrchardFixedBases {
    type FullScalar = OrchardFixedBasesFull;
    type Base = OrchardBaseFieldBases;
    type ShortScalar = OrchardShortScalarBases;
}

impl FixedPoint<pallas::Affine> for OrchardFixedBasesFull {
    type FixedScalarKind = FullScalar;

    fn generator(&self) -> pallas::Affine {
        match self {
            Self::CommitIvkR => commit_ivk_r::generator(),
            Self::NoteCommitR => note_commit_r::generator(),
            Self::ValueCommitR => value_commit_r::generator(),
            Self::SpendAuthG => spend_auth_g::generator(),
        }
    }

    fn u(&self) -> Vec<[[u8; 32]; H]> {
        match self {
            Self::CommitIvkR => commit_ivk_r::U.to_vec(),
            Self::NoteCommitR => note_commit_r::U.to_vec(),
            Self::ValueCommitR => value_commit_r::U.to_vec(),
            Self::SpendAuthG => spend_auth_g::U.to_vec(),
        }
    }

    fn z(&self) -> Vec<u64> {
        match self {
            Self::CommitIvkR => commit_ivk_r::Z.to_vec(),
            Self::NoteCommitR => note_commit_r::Z.to_vec(),
            Self::ValueCommitR => value_commit_r::Z.to_vec(),
            Self::SpendAuthG => spend_auth_g::Z.to_vec(),
        }
    }

    fn lagrange_coeffs(&self) -> Vec<[pallas::Base; H]> {
        match self {
            Self::CommitIvkR => lagrange_coeffs::COMMIT_IVK_R.to_vec(),
            Self::NoteCommitR => lagrange_coeffs::NOTE_COMMIT_R.to_vec(),
            Self::ValueCommitR => lagrange_coeffs::VALUE_COMMIT_R.to_vec(),
            Self::SpendAuthG => lagrange_coeffs::SPEND_AUTH_G.to_vec(),
        }
    }
}

impl FixedPoint<pallas::Affine> for OrchardBaseFieldBases {
    type FixedScalarKind = BaseFieldElem;

    fn generator(&self) -> pallas::Affine {
        match self {
            Self::NullifierK => nullifier_k::generator(),
            Self::SpendAuthGBase => spend_auth_g::generator(),
        }
    }

    fn u(&self) -> Vec<[[u8; 32]; H]> {
        match self {
            Self::NullifierK => nullifier_k::U.to_vec(),
            // SpendAuthG's full-scalar U/Z tables have the same 85-window
            // structure as the base-field-element variant (pallas::Base and
            // pallas::Scalar are both 255-bit); the precomputed values depend
            // only on the generator and window layout, not the scalar kind.
            Self::SpendAuthGBase => spend_auth_g::U.to_vec(),
        }
    }

    fn z(&self) -> Vec<u64> {
        match self {
            Self::NullifierK => nullifier_k::Z.to_vec(),
            Self::SpendAuthGBase => spend_auth_g::Z.to_vec(),
        }
    }

    fn lagrange_coeffs(&self) -> Vec<[pallas::Base; H]> {
        match self {
            Self::NullifierK => lagrange_coeffs::NULLIFIER_K.to_vec(),
            Self::SpendAuthGBase => lagrange_coeffs::SPEND_AUTH_G.to_vec(),
        }
    }
}

impl FixedPoint<pallas::Affine> for OrchardShortScalarBases {
    type FixedScalarKind = ShortScalar;

    fn generator(&self) -> pallas::Affine {
        match self {
            Self::ValueCommitV => value_commit_v::generator(),
            Self::SpendAuthGShort => spend_auth_g::generator(),
        }
    }

    fn u(&self) -> Vec<[[u8; 32]; H]> {
        match self {
            Self::ValueCommitV => value_commit_v::U_SHORT.to_vec(),
            Self::SpendAuthGShort => spend_auth_g::U_SHORT.to_vec(),
        }
    }

    fn z(&self) -> Vec<u64> {
        match self {
            Self::ValueCommitV => value_commit_v::Z_SHORT.to_vec(),
            Self::SpendAuthGShort => spend_auth_g::Z_SHORT.to_vec(),
        }
    }

    fn lagrange_coeffs(&self) -> Vec<[pallas::Base; H]> {
        match self {
            Self::ValueCommitV => lagrange_coeffs::VALUE_COMMIT_V.to_vec(),
            Self::SpendAuthGShort => lagrange_coeffs::SPEND_AUTH_G_SHORT.to_vec(),
        }
    }
}
