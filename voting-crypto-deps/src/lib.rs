//! Provides one stable cryptography import surface for shielded voting.
//!
//! Without the `zakura` feature, this crate reexports the crates.io Zcash
//! packages. Enabling `zakura` switches those reexports to the corresponding
//! `zakura-*` packages under their familiar Rust crate names.
//!
//! The facade itself supports Rust 1.86 in upstream mode. The Zakura packages
//! currently require Rust 1.88.

#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

#[cfg(not(feature = "zakura"))]
pub use upstream_halo2_gadgets as halo2_gadgets;
#[cfg(not(feature = "zakura"))]
pub use upstream_halo2_poseidon as halo2_poseidon;
#[cfg(not(feature = "zakura"))]
pub use upstream_halo2_proofs as halo2_proofs;
#[cfg(not(feature = "zakura"))]
pub use upstream_orchard as orchard;
#[cfg(not(feature = "zakura"))]
pub use upstream_pasta_curves as pasta_curves;
#[cfg(not(feature = "zakura"))]
pub use upstream_sinsemilla as sinsemilla;

#[cfg(feature = "zakura")]
pub use zakura_halo2_gadgets as halo2_gadgets;
#[cfg(feature = "zakura")]
pub use zakura_halo2_poseidon as halo2_poseidon;
#[cfg(feature = "zakura")]
pub use zakura_halo2_proofs as halo2_proofs;
#[cfg(feature = "zakura")]
pub use zakura_orchard as orchard;
#[cfg(feature = "zakura")]
pub use zakura_pasta_curves as pasta_curves;
#[cfg(feature = "zakura")]
pub use zakura_sinsemilla as sinsemilla;
