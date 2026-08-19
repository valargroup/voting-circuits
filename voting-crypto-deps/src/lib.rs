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
#[cfg(feature = "upstream-validator")]
pub use upstream_reddsa as reddsa;
#[cfg(feature = "upstream")]
pub use upstream_sinsemilla as sinsemilla;
#[cfg(feature = "upstream-validator")]
pub use upstream_zcash_primitives as zcash_primitives;

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
#[cfg(feature = "zakura-validator")]
pub use zakura_reddsa as reddsa;
#[cfg(feature = "zakura")]
pub use zakura_sinsemilla as sinsemilla;
#[cfg(feature = "zakura-validator")]
pub use zakura_zcash_primitives as zcash_primitives;

#[cfg(all(
    test,
    any(feature = "upstream-validator", feature = "zakura-validator")
))]
mod tests {
    use super::{orchard, zcash_primitives};

    #[test]
    fn transaction_primitives_use_the_selected_orchard_type() {
        let parsed =
            zcash_primitives::transaction::components::orchard::read_action_without_auth(&[][..]);
        let _: Option<orchard::Action<()>> = parsed.ok();
    }
}
