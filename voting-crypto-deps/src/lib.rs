//! Selects one coherent cryptography dependency family for shielded voting.
//!
//! The default `upstream` feature reexports the crates.io Zcash packages.
//! Zakura consumers must disable default features and enable `zakura`, which
//! reexports the corresponding `zakura-*` packages under their familiar Rust
//! crate names. Enabling both backends, or neither backend, is an error.
//!
//! The facade itself supports Rust 1.86 in upstream mode. The Zakura packages
//! currently require Rust 1.88.

#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

#[cfg(all(feature = "upstream", feature = "zakura"))]
compile_error!("features `upstream` and `zakura` cannot be enabled together");

#[cfg(not(any(feature = "upstream", feature = "zakura")))]
compile_error!("enable exactly one of the `upstream` or `zakura` features");

#[cfg(feature = "upstream")]
pub use upstream_halo2_gadgets as halo2_gadgets;
#[cfg(feature = "upstream")]
pub use upstream_halo2_poseidon as halo2_poseidon;
#[cfg(feature = "upstream")]
pub use upstream_halo2_proofs as halo2_proofs;
#[cfg(feature = "upstream")]
pub use upstream_orchard as orchard;
#[cfg(feature = "upstream")]
pub use upstream_pasta_curves as pasta_curves;
#[cfg(feature = "upstream")]
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
