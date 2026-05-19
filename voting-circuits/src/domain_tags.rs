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

use ff::PrimeField;
use pasta_curves::pallas;

/// Domain tag for Vote Authority Note commitments in the shared vote tree.
pub const DOMAIN_VAN: u64 = 0;

/// Domain tag for vote commitments in the shared vote tree.
pub const DOMAIN_VC: u64 = 1;

/// Blake2b-512 personalization for vote share secret derivation.
///
/// Distinct from Zcash's `"Zcash_ExpandSeed"` personalization to avoid
/// collisions with Zcash key-derivation streams that use similar inputs.
pub(crate) const VOTE_PRF_PERSONALIZATION: &[u8; 16] = b"ZcashVote_Expand";

/// PRF domain for vote-proof El Gamal encryption randomness.
pub(crate) const VOTE_PRF_DOMAIN_ELGAMAL: u8 = 0x00;
/// PRF domain for vote-proof share commitment blind factors.
pub(crate) const VOTE_PRF_DOMAIN_BLIND: u8 = 0x01;
/// PRF domain for vote-proof share-order shuffle seed.
pub(crate) const VOTE_PRF_DOMAIN_SHUFFLE: u8 = 0x02;
/// PRF domain for vote-proof remainder distribution weights.
pub(crate) const VOTE_PRF_DOMAIN_REMAINDER: u8 = 0x03;

/// Encodes a short ASCII tag as a canonical Pallas base-field element.
///
/// The 31-byte length limit leaves the high byte zero, which makes the
/// little-endian integer strictly smaller than the Pallas base-field modulus
/// for the tags used by this crate.
pub(crate) fn string_domain_tag(tag: &[u8]) -> pallas::Base {
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
pub(crate) fn governance_authorization() -> pallas::Base {
    string_domain_tag(b"governance authorization")
}

/// Domain tag for ZKP #1 delegation rho binding.
pub(crate) fn delegation_rho_binding() -> pallas::Base {
    string_domain_tag(b"delegation rho binding")
}

/// Domain tag for ZKP #1 governance alternate nullifiers.
pub(crate) fn governance_nullifier() -> pallas::Base {
    string_domain_tag(b"governance nullifier")
}

/// Domain tag for ZKP #3 share nullifiers.
pub fn share_spend() -> pallas::Base {
    string_domain_tag(b"share spend")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_domain_tags_are_distinct() {
        let tags = [
            ("van commitment", pallas::Base::from(DOMAIN_VAN)),
            ("vote commitment", pallas::Base::from(DOMAIN_VC)),
            ("vote authority spend", vote_authority_spend()),
            ("governance authorization", governance_authorization()),
            ("delegation rho binding", delegation_rho_binding()),
            ("governance nullifier", governance_nullifier()),
            ("share spend", share_spend()),
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
