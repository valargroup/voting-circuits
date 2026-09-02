//! Protocol domain-separation tag registry.
//!
//! This module is the crate-local source of truth for domain separator
//! constants. It does not define whole hash formulas; the hash-owning
//! modules remain responsible for their preimage layout.
//!
//! ## Encoding rule
//!
//! - Small numeric tags are used when several protocol-internal hashes share
//!   the same Poseidon shape and live in the same commitment tree. These are
//!   encoded as `pallas::Base::from(tag)`.
//! - Short ASCII string tags are used when the preimage should be readable at
//!   the byte level because the hash separates cross-protocol or cross-circuit
//!   replay domains. These are encoded as a zero-padded 32-byte little-endian
//!   integer and parsed as a canonical Pallas base-field element.
//! - Single-byte PRF domain tags are used as byte preimage fields when one
//!   BLAKE2b construction produces several independent vote-proof streams.

use crate::ff::PrimeField;
use voting_crypto_deps::pasta_curves::pallas;

/// Domain tag for Vote Authority Note commitments in the shared vote tree.
pub const DOMAIN_VAN: u64 = 0;

/// Domain tag for legacy (v1) vote commitments in the shared vote tree.
///
/// Retired by the weighted encrypt-choice design; kept registered so the
/// distinctness tests prevent any future tag from colliding with historical
/// v1 commitments.
#[allow(dead_code)]
pub const DOMAIN_VC: u64 = 1;

/// Domain tag for weighted (encrypt-choice) vote commitments in the shared
/// vote tree.
///
/// Version 2 of the vote commitment replaces the plaintext `vote_decision`
/// preimage slot with the public `decision_bucket_count`; the distinct domain
/// guarantees v1 and v2 commitments can never collide in the shared tree.
pub const DOMAIN_VC_V2: u64 = 2;

/// Blake2b-512 personalization for vote share secret derivation.
///
/// Distinct from Zcash's `"Zcash_ExpandSeed"` personalization to avoid
/// collisions with Zcash key-derivation streams that use similar inputs.
pub const VOTE_PRF_PERSONALIZATION: &[u8; 16] = b"ZcashVote_Expand";

/// PRF domain for legacy single-ciphertext El Gamal randomness (retired;
/// kept registered so the distinctness tests prevent reuse).
#[allow(dead_code)]
pub const VOTE_PRF_DOMAIN_ELGAMAL: u8 = 0x00;
/// PRF domain for vote-proof share commitment blind factors.
pub const VOTE_PRF_DOMAIN_BLIND: u8 = 0x01;
/// PRF domain for vote-proof share-order shuffle seed.
pub const VOTE_PRF_DOMAIN_SHUFFLE: u8 = 0x02;
/// PRF domain for vote-proof remainder distribution weights.
pub const VOTE_PRF_DOMAIN_REMAINDER: u8 = 0x03;
/// PRF domain for legacy single-share El Gamal randomness (retired; kept
/// registered so the distinctness tests prevent reuse).
#[allow(dead_code)]
pub const VOTE_PRF_DOMAIN_ELGAMAL_SINGLE_SHARE: u8 = 0x04;
/// PRF domain for weighted (per-bucket) encrypt-choice El Gamal randomness.
///
/// Used with the bucket-indexed PRF variant: the preimage appends a bucket
/// byte after the share index, making every `(share, bucket)` randomizer
/// unique for a given `(sk, round, proposal, VAN)`.
pub const VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED: u8 = 0x05;
/// PRF domain for single-share-layout weighted encrypt-choice El Gamal
/// randomness.
pub const VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED_SINGLE_SHARE: u8 = 0x06;

/// Encodes a short ASCII tag as a canonical Pallas base-field element.
///
/// The 31-byte length limit leaves the high byte zero, which makes the
/// little-endian integer strictly smaller than the Pallas base-field modulus
/// for the tags used by this crate.
fn string_domain_tag(tag: &[u8]) -> pallas::Base {
    assert!(
        tag.len() < 32 && tag.is_ascii(),
        "domain tags must be short ASCII strings"
    );
    let mut bytes = [0u8; 32];
    bytes[..tag.len()].copy_from_slice(tag);
    pallas::Base::from_repr(bytes).expect("short ASCII tag is canonical")
}

/// Domain tag for ZKP #2 VAN nullifiers.
pub fn vote_authority_spend() -> pallas::Base {
    string_domain_tag(b"vote authority spend")
}

/// Domain tag for ZKP #1 governance alternate-nullifier domains.
pub fn governance_authorization() -> pallas::Base {
    string_domain_tag(b"governance authorization")
}

/// Domain tag for ZKP #3 share nullifiers.
pub fn share_spend() -> pallas::Base {
    string_domain_tag(b"share spend")
}

/// Domain tag for weighted selected-share commitments (ZKP 1.5 / #2 / #3).
///
/// Prefixes the wide Poseidon commitment over one share's blind and all 16
/// bucket ciphertext coordinates. See `crate::bridge` for the authoritative
/// preimage layout.
pub fn weighted_share_commitment() -> pallas::Base {
    string_domain_tag(b"weighted share commitment")
}

/// Domain tag for the encrypt-choice bridge commitment (ZKP 1.5 / #2 seam).
///
/// Prefixes the compact bridge hash binding round, proposal, bucket count,
/// and every `(weight, selected commitment)` pair. See `crate::bridge` for
/// the authoritative preimage layout.
pub fn encrypt_choice_bridge() -> pallas::Base {
    string_domain_tag(b"encrypt choice bridge")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::PrimeField;

    fn assert_string_tag(name: &str, actual: pallas::Base, expected_prefix: &[u8]) {
        let mut expected = [0u8; 32];
        expected[..expected_prefix.len()].copy_from_slice(expected_prefix);
        assert_eq!(
            actual.to_repr(),
            expected,
            "{name} domain tag must remain a zero-padded ASCII field element"
        );
    }

    #[test]
    fn numeric_domain_tag_values_are_pinned() {
        assert_eq!(DOMAIN_VAN, 0);
        assert_eq!(DOMAIN_VC, 1);
        assert_eq!(DOMAIN_VC_V2, 2);
        assert_eq!(pallas::Base::from(DOMAIN_VAN), pallas::Base::zero());
        assert_eq!(pallas::Base::from(DOMAIN_VC), pallas::Base::one());
        assert_eq!(
            pallas::Base::from(DOMAIN_VC_V2),
            pallas::Base::one() + pallas::Base::one()
        );
    }

    #[test]
    fn string_domain_tag_values_are_pinned() {
        assert_string_tag(
            "vote authority spend",
            vote_authority_spend(),
            b"vote authority spend",
        );
        assert_string_tag(
            "governance authorization",
            governance_authorization(),
            b"governance authorization",
        );
        assert_string_tag("share spend", share_spend(), b"share spend");
        assert_string_tag(
            "weighted share commitment",
            weighted_share_commitment(),
            b"weighted share commitment",
        );
        assert_string_tag(
            "encrypt choice bridge",
            encrypt_choice_bridge(),
            b"encrypt choice bridge",
        );
    }

    #[test]
    fn vote_prf_domain_values_are_pinned() {
        assert_eq!(VOTE_PRF_PERSONALIZATION, b"ZcashVote_Expand");
        assert_eq!(VOTE_PRF_DOMAIN_ELGAMAL, 0x00);
        assert_eq!(VOTE_PRF_DOMAIN_BLIND, 0x01);
        assert_eq!(VOTE_PRF_DOMAIN_SHUFFLE, 0x02);
        assert_eq!(VOTE_PRF_DOMAIN_REMAINDER, 0x03);
        assert_eq!(VOTE_PRF_DOMAIN_ELGAMAL_SINGLE_SHARE, 0x04);
        assert_eq!(VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED, 0x05);
        assert_eq!(VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED_SINGLE_SHARE, 0x06);
    }

    #[test]
    fn protocol_domain_tags_are_distinct() {
        let tags = [
            ("van commitment", pallas::Base::from(DOMAIN_VAN)),
            ("vote commitment", pallas::Base::from(DOMAIN_VC)),
            ("weighted vote commitment", pallas::Base::from(DOMAIN_VC_V2)),
            ("vote authority spend", vote_authority_spend()),
            ("governance authorization", governance_authorization()),
            ("share spend", share_spend()),
            ("weighted share commitment", weighted_share_commitment()),
            ("encrypt choice bridge", encrypt_choice_bridge()),
        ];

        for (i, (left_name, left)) in tags.iter().enumerate() {
            for (right_name, right) in tags.iter().skip(i + 1) {
                assert_ne!(
                    left, right,
                    "domain tags must be distinct: {left_name} and {right_name}"
                );
            }
        }
    }

    #[test]
    fn vote_prf_domains_are_distinct() {
        let domains = [
            ("elgamal", VOTE_PRF_DOMAIN_ELGAMAL),
            ("blind", VOTE_PRF_DOMAIN_BLIND),
            ("shuffle", VOTE_PRF_DOMAIN_SHUFFLE),
            ("remainder", VOTE_PRF_DOMAIN_REMAINDER),
            ("single-share elgamal", VOTE_PRF_DOMAIN_ELGAMAL_SINGLE_SHARE),
            ("weighted elgamal", VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED),
            (
                "single-share weighted elgamal",
                VOTE_PRF_DOMAIN_ELGAMAL_WEIGHTED_SINGLE_SHARE,
            ),
        ];

        for (i, (left_name, left)) in domains.iter().enumerate() {
            for (right_name, right) in domains.iter().skip(i + 1) {
                assert_ne!(
                    left, right,
                    "vote PRF domains must be distinct: {left_name} and {right_name}"
                );
            }
        }
    }
}
