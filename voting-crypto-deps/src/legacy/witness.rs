//! Adapt documented Orchard encodings to voting's pinned circuit backend.
//!
//! The FVK layout and native derivations are inherited from Common's keys.rs,
//! note.rs, and backend/halo2/spec.rs at the revision recorded in README.md.
//! This is a circuit compatibility implementation, not a new protocol codec.
use crate::pasta_curves::group::{
    ff::{FromUniformBytes, PrimeField},
    Group, GroupEncoding,
};
use crate::{
    orchard::{self, keys::Scope},
    pasta_curves::{arithmetic::CurveExt, pallas},
};
use zcash_spec::PrfExpand;

const COMPONENT_BYTES: usize = 32;
const FVK_COMPONENTS: usize = 3;
const PRF_EXPAND_PERSONALIZATION: &[u8; 16] = b"Zcash_ExpandSeed";
const V3_RCM_DOMAIN: u8 = 0x0b;
const DIVERSIFY_DOMAIN: &str = "z.cash:Orchard-gd";

macro_rules! private_component {
    ($name:ident) => {
        #[derive(Clone, Copy)]
        pub struct $name([u8; COMPONENT_BYTES]);
        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(concat!(stringify!($name), "([redacted])"))
            }
        }
        impl $name {
            pub fn to_bytes(&self) -> [u8; COMPONENT_BYTES] {
                self.0
            }
        }
    };
}
private_component!(NullifierDerivingKey);
private_component!(CommitIvkRandomness);

/// Orchard raw FVK encoding: ak || nk || rivk, three canonical 32-byte components.
fn viewing_components(
    key: &orchard::keys::FullViewingKey,
) -> [[u8; COMPONENT_BYTES]; FVK_COMPONENTS] {
    let bytes = key.to_bytes();
    core::array::from_fn(|i| {
        bytes[i * COMPONENT_BYTES..(i + 1) * COMPONENT_BYTES]
            .try_into()
            .unwrap()
    })
}
pub(crate) fn nullifier_key(key: &orchard::keys::FullViewingKey) -> NullifierDerivingKey {
    NullifierDerivingKey(viewing_components(key)[1])
}
pub(crate) fn viewing_randomness(
    key: &orchard::keys::FullViewingKey,
    scope: Scope,
) -> CommitIvkRandomness {
    let [ak, nk, rivk] = viewing_components(key);
    CommitIvkRandomness(match scope {
        Scope::External => rivk,
        Scope::Internal => pallas::Scalar::from_uniform_bytes(
            &PrfExpand::ORCHARD_RIVK_INTERNAL.with(&rivk, &ak, &nk),
        )
        .to_repr(),
    })
}
pub(crate) fn note_psi(seed: &orchard::note::RandomSeed, rho: &orchard::note::Rho) -> pallas::Base {
    pallas::Base::from_uniform_bytes(&PrfExpand::PSI.with(seed.as_bytes(), &rho.to_bytes()))
}
pub(crate) fn note_trapdoor(
    version: orchard::note::NoteVersion,
    seed: &orchard::note::RandomSeed,
    rho: &orchard::note::Rho,
    gd: &crate::NonIdentityPallasPoint,
    pkd: &crate::NonIdentityPallasPoint,
    value: orchard::value::NoteValue,
    psi: pallas::Base,
) -> pallas::Scalar {
    match version {
        orchard::note::NoteVersion::V2 => pallas::Scalar::from_uniform_bytes(
            &PrfExpand::ORCHARD_RCM.with(seed.as_bytes(), &rho.to_bytes()),
        ),
        orchard::note::NoteVersion::V3 => {
            let mut h = blake2b_simd::Params::new()
                .hash_length(64)
                .personal(PRF_EXPAND_PERSONALIZATION)
                .to_state();
            h.update(seed.as_bytes());
            h.update(&[V3_RCM_DOMAIN]);
            h.update(&gd.to_bytes());
            h.update(&pkd.to_bytes());
            h.update(&value.inner().to_le_bytes());
            h.update(&rho.to_bytes());
            h.update(&psi.to_repr());
            pallas::Scalar::from_uniform_bytes(h.finalize().as_array())
        }
    }
}
pub(crate) fn diversified_base(address: &orchard::Address) -> crate::NonIdentityPallasPoint {
    let hash = pallas::Point::hash_to_curve(DIVERSIFY_DOMAIN);
    let point = hash(address.diversifier().as_array());
    // Orchard's defined identity substitution; choice remains constant time.
    let point = <pallas::Point as subtle::ConditionallySelectable>::conditional_select(
        &point,
        &hash(&[]),
        point.is_identity(),
    );
    crate::NonIdentityPallasPoint::from_bytes(&point.to_bytes())
        .expect("Orchard diversification returns a nonidentity point")
}
pub(crate) fn transmission_key(address: &orchard::Address) -> crate::NonIdentityPallasPoint {
    let bytes = address.to_raw_address_bytes();
    crate::NonIdentityPallasPoint::from_bytes(bytes[11..].try_into().unwrap())
        .expect("validated Orchard address has a nonidentity transmission key")
}
pub(crate) fn note_commitment(note: &orchard::note::Note) -> pallas::Point {
    let address = note.recipient();
    let gd = diversified_base(&address);
    let pkd = transmission_key(&address);
    let rho = note.rho();
    let psi = note_psi(note.rseed(), &rho);
    let rcm = note_trapdoor(
        note.version(),
        note.rseed(),
        &rho,
        &gd,
        &pkd,
        note.value(),
        psi,
    );
    crate::note_commitment_point(
        *gd,
        *pkd,
        note.value(),
        pallas::Base::from_repr(rho.to_bytes()).unwrap(),
        psi,
        rcm,
    )
    .expect("validated Orchard note has a valid commitment")
}
