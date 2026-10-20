//! Constants used in the Orchard protocol.
#[path = "constants/fixed_bases.rs"]
pub mod fixed_bases;
#[path = "constants/sinsemilla.rs"]
pub mod sinsemilla;

pub use self::sinsemilla::{OrchardCommitDomains, OrchardHashDomains};
pub use fixed_bases::{NullifierK, OrchardFixedBases, OrchardFixedBasesFull};

/// $\mathsf{MerkleDepth^{Orchard}}$
pub const MERKLE_DEPTH_ORCHARD: usize = 32;

/// The Pallas scalar field modulus is $q = 2^{254} + \mathsf{t_q}$.
/// <https://github.com/zcash/pasta>
pub(crate) const T_Q: u128 = 45560315531506369815346746415080538113;

/// The Pallas base field modulus is $p = 2^{254} + \mathsf{t_p}$.
/// <https://github.com/zcash/pasta>
pub(crate) const T_P: u128 = 45560315531419706090280762371685220353;

/// $\ell^\mathsf{Orchard}_\mathsf{base}$
pub const L_ORCHARD_BASE: usize = 255;

/// $\ell^\mathsf{Orchard}_\mathsf{scalar}$
pub const L_ORCHARD_SCALAR: usize = 255;

/// $\ell_\mathsf{value}$
pub const L_VALUE: usize = 64;

/// SWU hash-to-curve personalization for the group hash for key diversification
pub const KEY_DIVERSIFICATION_PERSONALIZATION: &str = "z.cash:Orchard-gd";
