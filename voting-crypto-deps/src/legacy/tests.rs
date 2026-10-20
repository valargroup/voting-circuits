use super::{key_vectors, spend_auth_generator, witness};
use crate::pasta_curves::group::{
    ff::{Field, FromUniformBytes, PrimeField},
    Curve, GroupEncoding,
};
use crate::{
    orchard::{
        self,
        keys::{FullViewingKey, Scope, SpendingKey},
        note::{Note, NoteVersion, RandomSeed, Rho},
        value::NoteValue,
    },
    pasta_curves::{arithmetic::CurveAffine, pallas},
};
use crate::{CircuitValue, NonIdentityPallasPoint};

#[test]
fn frozen_protocol_vectors_cover_viewing_scopes_and_both_sign_corrections() {
    let mut signs = [false; 2];
    for vector in key_vectors::test_vectors() {
        let sk = SpendingKey::from_bytes(vector.sk).unwrap();
        let fvk = FullViewingKey::from(&sk);
        assert_eq!(witness::nullifier_key(&fvk).to_bytes(), vector.nk);
        assert_eq!(
            witness::viewing_randomness(&fvk, Scope::External).to_bytes(),
            vector.rivk
        );
        assert_eq!(
            witness::viewing_randomness(&fvk, Scope::Internal).to_bytes(),
            vector.internal_rivk
        );
        let ask = crate::spend_authorizing_scalar(&sk);
        assert_eq!(ask.to_repr(), vector.ask);
        assert_eq!((spend_auth_generator() * ask).to_bytes(), vector.ak);
        let raw = pallas::Scalar::from_uniform_bytes(
            &zcash_spec::PrfExpand::ORCHARD_ASK.with(&vector.sk),
        );
        signs[((spend_auth_generator() * raw).to_bytes()[31] >> 7) as usize] = true;
        let native = orchard::keys::SpendAuthorizingKey::from(&sk)
            .randomize(&crate::spend_auth_randomizer(&pallas::Scalar::ZERO));
        let vk = orchard::primitives::redpallas::VerificationKey::from(&native);
        let sig = native.sign(crate::rand::rngs::OsRng, b"voting-owned key agreement");
        vk.verify(b"voting-owned key agreement", &sig).unwrap();
        assert!(vk
            .verify(b"altered voting-owned key agreement", &sig)
            .is_err());
        let address = fvk.address(
            orchard::keys::Diversifier::from_bytes(vector.default_d),
            Scope::External,
        );
        assert_eq!(
            witness::transmission_key(&address).to_bytes(),
            vector.default_pk_d
        );
        let rho = Rho::from_bytes(&vector.note_rho).unwrap();
        let seed = RandomSeed::from_bytes(vector.note_rseed, &rho).unwrap();
        let gd = witness::diversified_base(&address);
        let pkd = witness::transmission_key(&address);
        let rcm = witness::note_trapdoor(
            NoteVersion::V3,
            &seed,
            &rho,
            &gd,
            &pkd,
            NoteValue::from_raw(vector.note_v),
            witness::note_psi(&seed, &rho),
        );
        assert_eq!(rcm.to_repr(), vector.note_qr_rcm);
    }
    assert_eq!(signs, [true, true]);
}

#[test]
fn native_notes_agree_across_scopes_versions_and_value_boundaries() {
    use crate::halo2_poseidon::{ConstantLength, Hash, P128Pow5T3};
    for vector in key_vectors::test_vectors() {
        let sk = SpendingKey::from_bytes(vector.sk).unwrap();
        let fvk = FullViewingKey::from(&sk);
        for scope in [Scope::External, Scope::Internal] {
            let address = fvk.address(
                orchard::keys::Diversifier::from_bytes(vector.default_d),
                scope,
            );
            let rho = Rho::from_bytes(&vector.note_rho).unwrap();
            let seed = RandomSeed::from_bytes(vector.note_rseed, &rho).unwrap();
            for version in [NoteVersion::V2, NoteVersion::V3] {
                for value in [0, 1, u64::MAX] {
                    let note =
                        Note::from_parts(address, NoteValue::from_raw(value), rho, seed, version)
                            .unwrap();
                    let cm = witness::note_commitment(&note);
                    let native_cmx =
                        orchard::note::ExtractedNoteCommitment::from(note.commitment()).to_bytes();
                    assert_eq!(
                        cm.to_affine().coordinates().unwrap().x().to_repr(),
                        native_cmx
                    );
                    let nk = witness::nullifier_key(&fvk).circuit_value();
                    let rho_field = pallas::Base::from_repr(rho.to_bytes()).unwrap();
                    let prf = Hash::<_, P128Pow5T3, ConstantLength<2>, 3, 2>::init()
                        .hash([nk, rho_field]);
                    let scalar =
                        pallas::Scalar::from_repr((prf + witness::note_psi(&seed, &rho)).to_repr())
                            .unwrap();
                    let k = super::constants::fixed_bases::nullifier_k::generator();
                    let nf_point = cm + k * scalar;
                    assert_eq!(
                        nf_point.to_affine().coordinates().unwrap().x().to_repr(),
                        note.nullifier(&fvk).to_bytes()
                    );
                    let ivk_bytes = fvk.to_ivk(scope).to_bytes();
                    let ivk = pallas::Base::from_repr(ivk_bytes[32..].try_into().unwrap()).unwrap();
                    let ivk = pallas::Scalar::from_repr(ivk.to_repr()).unwrap();
                    assert_eq!(
                        *witness::diversified_base(&address) * ivk,
                        *witness::transmission_key(&address)
                    );
                }
            }
        }
    }
}

#[test]
fn point_parser_rejects_identity_invalid_and_modulus_boundary_encodings() {
    use crate::pasta_curves::group::Group;
    assert!(NonIdentityPallasPoint::from_bytes(&pallas::Point::identity().to_bytes()).is_none());
    assert!(NonIdentityPallasPoint::from_bytes(&[0xff; 32]).is_none());
    let mut modulus = (-pallas::Base::ONE).to_repr();
    for byte in &mut modulus {
        let (sum, carry) = byte.overflowing_add(1);
        *byte = sum;
        if !carry {
            break;
        }
    }
    assert!(NonIdentityPallasPoint::from_bytes(&modulus).is_none());
    let generator = spend_auth_generator().to_bytes();
    assert_eq!(
        NonIdentityPallasPoint::from_bytes(&generator)
            .unwrap()
            .to_bytes(),
        generator
    );
}

#[test]
fn synthetic_v3_trapdoors_match_the_original_common_bridge() {
    // Frozen from Common e018495880249a37c804920b0b0da09d572a4591's
    // public bridge with the fixed, public test seed and synthetic points below.
    let rho = Rho::from_bytes(&pallas::Base::ONE.to_repr()).unwrap();
    let seed = RandomSeed::from_bytes([7; 32], &rho).unwrap();
    let gd = NonIdentityPallasPoint::from_bytes(&spend_auth_generator().to_bytes()).unwrap();
    let pkd = NonIdentityPallasPoint::from_bytes(
        &(spend_auth_generator() * pallas::Scalar::from(2)).to_bytes(),
    )
    .unwrap();
    let cases = [
        (
            0,
            "9b059fb09aa68f29d4961e289711ef771e4fb8bcb6e29ff4bceba51382d03336",
        ),
        (
            1,
            "6a91bbdccdb4ba9e54d68d860a8f9f85e6794cfec6f9cf76f890ee0e858da30e",
        ),
        (
            u64::MAX,
            "12ea145f1c22841e0c9d7dc26989693be1dc578f04e005646f549b0d04c74537",
        ),
    ];
    for (value, expected) in cases {
        let expected: [u8; 32] =
            core::array::from_fn(|i| u8::from_str_radix(&expected[i * 2..i * 2 + 2], 16).unwrap());
        let actual = witness::note_trapdoor(
            NoteVersion::V3,
            &seed,
            &rho,
            &gd,
            &pkd,
            NoteValue::from_raw(value),
            witness::note_psi(&seed, &rho),
        );
        assert_eq!(actual.to_repr(), expected);
    }
}
