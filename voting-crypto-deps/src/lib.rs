//! Selects one coherent cryptography dependency family for shielded voting.
//!
//! The default `upstream` feature reexports the complete crates.io Zcash
//! package family. Consumers that need a smaller graph can disable default
//! features and select fine-grained features such as `upstream-vct`. Zakura
//! consumers use the corresponding `zakura-*` features. Mixing the two
//! package families, or selecting neither family, is an error.
//!
//! The facade itself supports Rust 1.86 in upstream mode. The Zakura packages
//! currently require Rust 1.88.

#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

#[cfg(all(
    any(
        feature = "upstream-pasta",
        feature = "upstream-gadgets",
        feature = "upstream-poseidon",
        feature = "upstream-proofs",
        feature = "upstream-orchard",
        feature = "upstream-rand",
        feature = "upstream-sinsemilla",
        feature = "upstream-validator",
    ),
    any(
        feature = "zakura-pasta",
        feature = "zakura-gadgets",
        feature = "zakura-poseidon",
        feature = "zakura-proofs",
        feature = "zakura-orchard",
        feature = "zakura-rand",
        feature = "zakura-sinsemilla",
        feature = "zakura-validator",
    )
))]
compile_error!("upstream and Zakura dependency features cannot be enabled together");

#[cfg(not(any(
    feature = "upstream-pasta",
    feature = "upstream-gadgets",
    feature = "upstream-poseidon",
    feature = "upstream-proofs",
    feature = "upstream-orchard",
    feature = "upstream-rand",
    feature = "upstream-sinsemilla",
    feature = "upstream-validator",
    feature = "zakura-pasta",
    feature = "zakura-gadgets",
    feature = "zakura-poseidon",
    feature = "zakura-proofs",
    feature = "zakura-orchard",
    feature = "zakura-rand",
    feature = "zakura-sinsemilla",
    feature = "zakura-validator",
)))]
compile_error!("enable at least one upstream or Zakura dependency feature");

#[cfg(feature = "upstream-gadgets")]
pub use upstream_halo2_gadgets as halo2_gadgets;
#[cfg(feature = "upstream-poseidon")]
pub use upstream_halo2_poseidon as halo2_poseidon;
#[cfg(feature = "upstream-proofs")]
pub use upstream_halo2_proofs as halo2_proofs;
#[cfg(feature = "upstream-orchard")]
pub use upstream_orchard as orchard;
#[cfg(feature = "upstream-pasta")]
pub use upstream_pasta_curves as pasta_curves;
#[cfg(feature = "upstream-rand")]
pub use upstream_rand as rand;
#[cfg(feature = "upstream-validator")]
pub use upstream_reddsa as reddsa;
#[cfg(feature = "upstream-sinsemilla")]
pub use upstream_sinsemilla as sinsemilla;
#[cfg(feature = "upstream-validator")]
pub use upstream_zcash_primitives as zcash_primitives;

#[cfg(feature = "zakura-gadgets")]
pub use zakura_halo2_gadgets as halo2_gadgets;
#[cfg(feature = "zakura-poseidon")]
pub use zakura_halo2_poseidon as halo2_poseidon;
#[cfg(feature = "zakura-proofs")]
pub use zakura_halo2_proofs as halo2_proofs;
#[cfg(feature = "zakura-orchard")]
pub use zakura_orchard as orchard;
#[cfg(feature = "zakura-pasta")]
pub use zakura_pasta_curves as pasta_curves;
#[cfg(feature = "zakura-rand")]
pub mod rand {
    pub use zakura_rand::{CryptoRng, Rng};

    pub mod rngs {
        use core::convert::Infallible;
        use zakura_rand::{
            rand_core::{TryCryptoRng, TryRng, UnwrapErr},
            rngs::SysRng,
        };

        #[derive(Clone, Copy, Debug, Default)]
        pub struct OsRng;

        impl TryRng for OsRng {
            type Error = Infallible;

            fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
                UnwrapErr(SysRng).try_next_u32()
            }

            fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
                UnwrapErr(SysRng).try_next_u64()
            }

            fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
                UnwrapErr(SysRng).try_fill_bytes(dst)
            }
        }

        impl TryCryptoRng for OsRng {}
    }
}
#[cfg(feature = "zakura-validator")]
pub use zakura_reddsa as reddsa;
#[cfg(feature = "zakura-sinsemilla")]
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
