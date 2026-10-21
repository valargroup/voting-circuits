#![cfg(any(feature = "orchard", feature = "lrz-orchard"))]

use voting_crypto_deps::{
    composition::{spend_auth_generator, NonIdentityPallasPoint},
    orchard::keys::{FullViewingKey, SpendValidatingKey, SpendingKey},
    pasta_curves::{
        group::{ff::PrimeField, Group, GroupEncoding},
        pallas,
    },
    spend_authorizing_scalar, CircuitValue,
};

#[test]
fn spending_scalar_matches_public_key_for_both_signs() {
    // Frozen PRF_expand(sk, 0x06) values reduced modulo the Pallas scalar order.
    let fixtures = [
        (
            0,
            "20669426733999045941002756217236585338641686720835387998358833163862666371260",
        ),
        (
            3,
            "28804734343413892612679563040438627094074937225352888070312827785222692342209",
        ),
    ];
    let mut signs_seen = [false; 2];
    for (seed, raw) in fixtures {
        let raw = pallas::Scalar::from_str_vartime(raw).unwrap();
        let sign = (spend_auth_generator() * raw).to_bytes()[31] >> 7;
        signs_seen[sign as usize] = true;
        let key = SpendingKey::from_bytes([seed; 32]).unwrap();
        let actual = spend_authorizing_scalar(&key);
        assert_eq!(actual, if sign == 1 { -raw } else { raw });
        let ak = SpendValidatingKey::from(FullViewingKey::from(&key));
        assert_eq!(spend_auth_generator() * actual, ak.circuit_value());
    }
    assert_eq!(signs_seen, [true, true]);
}

#[test]
fn point_adapter_accepts_canonical_points_and_rejects_invalid_encodings() {
    let bytes = spend_auth_generator().to_bytes();
    let point =
        Option::<NonIdentityPallasPoint>::from(NonIdentityPallasPoint::from_bytes(&bytes)).unwrap();
    assert_eq!(point.to_bytes(), bytes);
    for bytes in [pallas::Point::identity().to_bytes(), [0xff; 32]] {
        assert!(
            Option::<NonIdentityPallasPoint>::from(NonIdentityPallasPoint::from_bytes(&bytes))
                .is_none()
        );
    }
}
