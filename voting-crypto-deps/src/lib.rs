//! Selects one coherent cryptography dependency family for shielded voting.
//!
//! The default features reexport the complete Zakura package family. Consumers
//! that need a smaller graph can disable default features and select
//! fine-grained features such as `vct`. LRZ consumers use the corresponding
//! `lrz-*` features. Mixing the two package families, or selecting neither
//! family, is an error.
//!
//! The facade itself supports Rust 1.86 in LRZ mode. The default Zakura
//! packages currently require Rust 1.91.

#![deny(missing_debug_implementations)]
#![deny(unsafe_code)]

#[cfg(all(
    any(
        feature = "pasta",
        feature = "gadgets",
        feature = "poseidon",
        feature = "proofs",
        feature = "orchard",
        feature = "rand",
        feature = "sinsemilla",
        feature = "validator",
    ),
    any(
        feature = "lrz-pasta",
        feature = "lrz-gadgets",
        feature = "lrz-poseidon",
        feature = "lrz-proofs",
        feature = "lrz-orchard",
        feature = "lrz-rand",
        feature = "lrz-sinsemilla",
        feature = "lrz-validator",
    )
))]
compile_error!("Zakura and LRZ dependency features cannot be enabled together");

#[cfg(not(any(
    feature = "pasta",
    feature = "gadgets",
    feature = "poseidon",
    feature = "proofs",
    feature = "orchard",
    feature = "rand",
    feature = "sinsemilla",
    feature = "validator",
    feature = "lrz-pasta",
    feature = "lrz-gadgets",
    feature = "lrz-poseidon",
    feature = "lrz-proofs",
    feature = "lrz-orchard",
    feature = "lrz-rand",
    feature = "lrz-sinsemilla",
    feature = "lrz-validator",
)))]
compile_error!("enable at least one Zakura or LRZ dependency feature");

#[cfg(feature = "gadgets")]
pub use ::halo2_gadgets;
#[cfg(feature = "poseidon")]
pub use ::halo2_poseidon;
#[cfg(feature = "proofs")]
pub use ::halo2_proofs;
#[cfg(feature = "orchard")]
pub use ::orchard;
#[cfg(feature = "pasta")]
pub use ::pasta_curves;
#[cfg(feature = "rand")]
pub mod rand {
    pub use ::rand::{CryptoRng, Rng};

    pub mod rngs {
        use ::rand::{
            rand_core::{TryCryptoRng, TryRng, UnwrapErr},
            rngs::SysRng,
        };
        use core::convert::Infallible;

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
#[cfg(feature = "validator")]
pub use ::reddsa;
#[cfg(feature = "sinsemilla")]
pub use ::sinsemilla;
#[cfg(feature = "validator")]
pub use ::zcash_primitives;

#[cfg(feature = "lrz-gadgets")]
pub use lrz_halo2_gadgets as halo2_gadgets;
#[cfg(feature = "lrz-poseidon")]
pub use lrz_halo2_poseidon as halo2_poseidon;
#[cfg(feature = "lrz-proofs")]
pub use lrz_halo2_proofs as halo2_proofs;
#[cfg(feature = "lrz-orchard")]
pub use lrz_orchard as orchard;
#[cfg(feature = "lrz-pasta")]
pub use lrz_pasta_curves as pasta_curves;
#[cfg(feature = "lrz-rand")]
pub use lrz_rand as rand;
#[cfg(feature = "lrz-validator")]
pub use lrz_reddsa as reddsa;
#[cfg(feature = "lrz-sinsemilla")]
pub use lrz_sinsemilla as sinsemilla;
#[cfg(feature = "lrz-validator")]
pub use lrz_zcash_primitives as zcash_primitives;

/// Arms halo2's prepared commitment tables over `params`, where the selected
/// backend has them.
///
/// Key generation precomputes the polynomial-side proving data and stops
/// there. The Zakura backend can additionally build fixed-base tables that the
/// prover's polynomial commitments then evaluate through; this arms them. The
/// LRZ backend has no such API, so there this reports `false` and proving
/// keeps its ordinary path — the same answer the Zakura backend gives when it
/// declines.
///
/// Preparation is idempotent, single-flighted, and shared with every clone of
/// `params`; it is never serialized, so call it again after `Params::read`.
/// Call it once from a long-lived prover, off the proving critical path, and
/// not fanned out across a worker pool.
///
/// Arming is a pure performance change: it alters only the route by which the
/// commitment MSMs are evaluated, never their results, so proofs and verifying
/// keys are unaffected.
///
/// Returns whether tables were actually built and cached.
#[cfg(feature = "proofs")]
pub fn prepare_commitments<C: halo2_proofs::arithmetic::CurveAffine>(
    params: &halo2_proofs::poly::commitment::Params<C>,
) -> bool {
    params.prepare_commitments()
}

/// LRZ counterpart of the Zakura [`prepare_commitments`]; always `false`,
/// because upstream halo2 has no prepared commitment tables to arm.
#[cfg(feature = "lrz-proofs")]
pub fn prepare_commitments<C: halo2_proofs::arithmetic::CurveAffine>(
    _params: &halo2_proofs::poly::commitment::Params<C>,
) -> bool {
    false
}

#[cfg(all(test, any(feature = "validator", feature = "lrz-validator")))]
mod tests {
    use super::{orchard, zcash_primitives};

    #[test]
    fn transaction_primitives_use_the_selected_orchard_type() {
        let parsed =
            zcash_primitives::transaction::components::orchard::read_action_without_auth(&[][..]);
        let _: Option<orchard::Action<()>> = parsed.ok();
    }
}
