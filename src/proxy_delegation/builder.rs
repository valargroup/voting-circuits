//! Proxy delegation builder (ZKP #4).
//!
//! Builds the circuit and public inputs that delegate part of a VAN's weight
//! to a delegate. The VAN is spent as ZKP #2's builder spends it, and every
//! share secret comes from the caller's `dc_seed` through a BLAKE2b-512 PRF,
//! so the same inputs always rebuild the same delegation commitment.

use std::string::String;

use crate::ff::{Field, FromUniformBytes, PrimeField};
use crate::group::{Curve, GroupEncoding};
use voting_crypto_deps::halo2_proofs::circuit::Value;
use voting_crypto_deps::orchard::keys::{FullViewingKey, Scope, SpendingKey};
use voting_crypto_deps::pasta_curves::{
    arithmetic::{Coordinates, CurveAffine},
    pallas,
};

use super::circuit::{
    dc_slot_nullifier_hash, delegation_commitment_hash, Circuit, Instance, DC_SLOT_BITS,
};
use crate::{
    domain_tags,
    gadgets::{
        elgamal::{base_to_scalar, spend_auth_g_affine},
        van_integrity::van_integrity_hash,
    },
    params::{MAX_PROPOSAL_AUTHORITY, SHARE_VALUE_LIMIT, VOTE_COMM_TREE_DEPTH},
    protocol_hash::poseidon_hash_2,
    shares_hash::{share_commitment, shares_hash},
    vote_proof::{
        denomination_split_with, extract_vsk, shuffle_with_seed, van_nullifier_hash,
        EncryptedShareOutput,
    },
};

/// How a delegation commitment splits its weight `w` into 16 shares.
///
/// The discriminant is the layout number (0 or 1) that the client's `dc_seed`
/// derivation binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShareLayout {
    /// Layout 0: ZKP #2's denomination split of `w`, shuffled.
    Standard = 0,
    /// Layout 1: all of `w` in share 0 and zero in shares 1 to 15, as in
    /// ZKP #2's last-moment single-share mode.
    SingleShare = 1,
}

/// Output of [`build_proxy_delegation_proof`]: the circuit to prove, and every
/// value the client needs to submit the delegation and deliver its shares.
#[derive(Clone, Debug)]
pub struct ProxyDelegationBundle {
    /// The circuit with every witness set; prove it with
    /// [`create_proxy_delegation_proof`](super::create_proxy_delegation_proof).
    pub circuit: Circuit,
    /// The public inputs, which carry the VAN nullifier, `r_vpk`, the
    /// successor VAN, the delegation commitment and the DC slot nullifier.
    pub instance: Instance,
    /// Compressed `r_vpk`, the key that verifies the batch signature.
    pub r_vpk_bytes: [u8; 32],
    /// The caller's `alpha_v`, kept with `r_vpk`: the batch is signed with
    /// `rsk_v = ask_v.randomize(&alpha_v)`.
    pub alpha_v: pallas::Scalar,
    /// The 16 encrypted shares that the delegation commitment binds.
    pub encrypted_shares: [EncryptedShareOutput; 16],
    /// Hash of the 16 share commitments. It is bound publicly only through
    /// `instance.delegation_commitment`.
    pub shares_hash: pallas::Base,
    /// Per-share commitment blinds.
    pub share_blinds: [pallas::Base; 16],
    /// Per-share commitments, which ZKP #3 takes to reveal a share.
    pub share_comms: [pallas::Base; 16],
}

/// Errors from [`build_proxy_delegation_proof`].
#[derive(Debug)]
pub enum ProxyDelegationBuildError {
    /// `dc_slot` is not below `2^DC_SLOT_BITS = 8`.
    InvalidDcSlot(u8),
    /// The delegate index is 0, which no delegate holds.
    ZeroDelegateIndex,
    /// The delegated weight is 0 ballots.
    ZeroDelegatedWeight,
    /// The delegated weight is more than the VAN holds.
    DelegatedWeightExceedsVan {
        /// The weight to delegate, in ballots.
        delegated_ballots: u64,
        /// The VAN's weight, in ballots.
        van_ballots: u64,
    },
    /// The VAN weight is above `2^30` ballots, which no VAN can hold.
    InvalidVanBallots(u64),
    /// A share is not below `2^30`.
    InvalidShares(String),
    /// The election authority's public key is the identity point.
    InvalidElectionPublicKey,
    /// The randomized voting public key is the identity point.
    InvalidRandomizedVotingPublicKey,
    /// A derived El Gamal ciphertext point was the identity point.
    InvalidEncryptedShare(String),
}

impl core::fmt::Display for ProxyDelegationBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDcSlot(slot) => {
                write!(f, "dc_slot must be below {}, got {slot}", 1 << DC_SLOT_BITS)
            }
            Self::ZeroDelegateIndex => write!(f, "delegate_index must be at least 1"),
            Self::ZeroDelegatedWeight => write!(f, "delegated weight must be at least 1 ballot"),
            Self::DelegatedWeightExceedsVan {
                delegated_ballots,
                van_ballots,
            } => write!(
                f,
                "cannot delegate {delegated_ballots} ballots from a VAN holding {van_ballots}"
            ),
            Self::InvalidVanBallots(van_ballots) => write!(
                f,
                "VAN weight must be at most {SHARE_VALUE_LIMIT} ballots, got {van_ballots}"
            ),
            Self::InvalidShares(msg) => write!(f, "invalid shares: {msg}"),
            Self::InvalidElectionPublicKey => {
                write!(f, "invalid election public key: identity point")
            }
            Self::InvalidRandomizedVotingPublicKey => {
                write!(f, "invalid randomized voting public key: identity point")
            }
            Self::InvalidEncryptedShare(msg) => write!(f, "invalid encrypted share: {msg}"),
        }
    }
}

impl std::error::Error for ProxyDelegationBuildError {}

/// Returns the affine coordinates of `point`, or `None` for the identity.
fn point_xy(point: pallas::Affine) -> Option<(pallas::Base, pallas::Base)> {
    let coordinates: Option<Coordinates<pallas::Affine>> = point.coordinates().into();
    coordinates.map(|c| (*c.x(), *c.y()))
}

/// ZKP #4 share PRF:
///
/// ```text
/// BLAKE2b-512("ZcashVoteProxyEx", dc_seed || domain || share_index)
/// ```
///
/// This is ZKP #2's vote share PRF with `dc_seed` as the key. It hashes no
/// round, proposal or VAN, because `dc_seed` already binds them (see the
/// caller contract of [`build_proxy_delegation_proof`]).
fn proxy_share_prf(dc_seed: &[u8; 32], domain: u8, share_index: u8) -> [u8; 64] {
    *blake2b_simd::Params::new()
        .hash_length(64)
        .personal(domain_tags::PROXY_PRF_PERSONALIZATION)
        .to_state()
        .update(dc_seed)
        .update(&[domain])
        .update(&[share_index])
        .finalize()
        .as_array()
}

/// Derives the El Gamal randomness of share `share_index`.
///
/// The two layouts use different domains, so rebuilding a DC in the other
/// layout never reuses a nonce. The result is non-zero, as condition 11
/// requires.
fn derive_share_randomness(
    dc_seed: &[u8; 32],
    share_index: u8,
    layout: ShareLayout,
) -> pallas::Base {
    let domain = match layout {
        ShareLayout::Standard => domain_tags::PROXY_PRF_DOMAIN_ELGAMAL,
        ShareLayout::SingleShare => domain_tags::PROXY_PRF_DOMAIN_ELGAMAL_SINGLE_SHARE,
    };
    let r = pallas::Base::from_uniform_bytes(&proxy_share_prf(dc_seed, domain, share_index));
    if bool::from(r.is_zero()) {
        // Stay deterministic in the negligible exact-zero case.
        return pallas::Base::one();
    }
    r
}

/// Derives the commitment blind of share `share_index`.
fn derive_share_blind(dc_seed: &[u8; 32], share_index: u8) -> pallas::Base {
    pallas::Base::from_uniform_bytes(&proxy_share_prf(
        dc_seed,
        domain_tags::PROXY_PRF_DOMAIN_BLIND,
        share_index,
    ))
}

/// Builds the ZKP #4 circuit and public inputs that delegate
/// `delegated_ballots` of a VAN's weight to delegate `delegate_index`.
///
/// Prove the result with
/// [`create_proxy_delegation_proof`](super::create_proxy_delegation_proof).
/// The VAN must have full proposal authority, that is, it has not voted.
///
/// # Arguments
///
/// * `sk`, `address_index`, `van_comm_rand`, `voting_round_id`,
///   `vote_comm_tree_path`, `vote_comm_tree_position`, `anchor_height`, `ea_pk`
///   and `alpha_v` - as for
///   [`build_vote_proof_from_delegation`](crate::vote_proof::build_vote_proof_from_delegation),
///   which also derives the tree root from the path.
/// * `van_ballots` - The VAN's weight `W` in ballots: for a VAN from ZKP #1,
///   `total_note_value / 12,500,000` rounded down; for a ZKP #4 successor,
///   the weight it kept.
/// * `delegate_index` - The delegate `d`, at least 1. The circuit does not
///   bound it; the chain counts the shares only if `d` is a registered
///   delegate.
/// * `delegated_ballots` - The weight `w` to delegate, in `[1, W]`. The
///   successor VAN keeps `W - w`.
/// * `dc_slot` - This DC's slot in `[0, 8)` among the delegations of the VAN
///   chain in this round.
/// * `dc_seed` - The seed of every share secret.
/// * `layout` - The share layout.
///
/// # Caller contract
///
/// `dc_seed` must be secret and unique to the DC's VAN, round, delegate,
/// weight and layout, which the client's derivation hashes in. A seed reused
/// for different shares reuses El Gamal nonces.
///
/// The output is a deterministic function of the inputs, so a rebuild with
/// the same inputs, including `alpha_v`, gives byte-identical public inputs,
/// shares and commitments. `alpha_v` must otherwise be fresh from a CSPRNG for
/// each DC. As for ZKP #2, the caller must authenticate `voting_round_id`,
/// `anchor_height`, `ea_pk` and the tree witness.
pub fn build_proxy_delegation_proof(
    sk: &SpendingKey,
    address_index: u32,
    van_ballots: u64,
    van_comm_rand: pallas::Base,
    voting_round_id: pallas::Base,
    vote_comm_tree_path: [pallas::Base; VOTE_COMM_TREE_DEPTH],
    vote_comm_tree_position: u32,
    anchor_height: u32,
    ea_pk: pallas::Affine,
    alpha_v: pallas::Scalar,
    delegate_index: u32,
    delegated_ballots: u64,
    dc_slot: u8,
    dc_seed: [u8; 32],
    layout: ShareLayout,
) -> Result<ProxyDelegationBundle, ProxyDelegationBuildError> {
    if u32::from(dc_slot) >= 1 << DC_SLOT_BITS {
        return Err(ProxyDelegationBuildError::InvalidDcSlot(dc_slot));
    }
    if delegate_index == 0 {
        return Err(ProxyDelegationBuildError::ZeroDelegateIndex);
    }
    if delegated_ballots == 0 {
        return Err(ProxyDelegationBuildError::ZeroDelegatedWeight);
    }
    if delegated_ballots > van_ballots {
        return Err(ProxyDelegationBuildError::DelegatedWeightExceedsVan {
            delegated_ballots,
            van_ballots,
        });
    }
    // ZKP #1 mints at most 2^30 ballots and condition 7 keeps every successor
    // below 2^30, so no VAN holds more.
    if van_ballots > SHARE_VALUE_LIMIT {
        return Err(ProxyDelegationBuildError::InvalidVanBallots(van_ballots));
    }
    let (ea_pk_x, ea_pk_y) =
        point_xy(ea_pk).ok_or(ProxyDelegationBuildError::InvalidElectionPublicKey)?;

    // ---- Keys and address, derived as in ZKP #2 ----

    let vsk = extract_vsk(sk);
    let fvk: FullViewingKey = sk.into();
    let vsk_nk = fvk.nk().inner();
    let rivk_v = fvk.rivk(Scope::External).inner();
    let address = fvk.address_at(address_index, Scope::External);
    let vpk_g_d = address.g_d().to_affine();
    let vpk_pk_d = address.pk_d().inner().to_affine();
    let (g_d_x, _) =
        point_xy(vpk_g_d).expect("Orchard address g_d is non-identity by construction");
    let (pk_d_x, _) =
        point_xy(vpk_pk_d).expect("Orchard address pk_d is non-identity by construction");

    // ---- VANs and nullifiers (conditions 2, 5, 6 and 14) ----

    let remaining_ballots = van_ballots - delegated_ballots;
    let authority = pallas::Base::from(MAX_PROPOSAL_AUTHORITY);
    let vote_authority_note_old = van_integrity_hash(
        g_d_x,
        pk_d_x,
        pallas::Base::from(van_ballots),
        voting_round_id,
        authority,
        van_comm_rand,
    );
    let vote_authority_note_new = van_integrity_hash(
        g_d_x,
        pk_d_x,
        pallas::Base::from(remaining_ballots),
        voting_round_id,
        authority,
        van_comm_rand,
    );
    let van_nullifier = van_nullifier_hash(vsk_nk, voting_round_id, vote_authority_note_old);
    let dc_slot_nullifier = dc_slot_nullifier_hash(vsk_nk, voting_round_id, van_comm_rand, dc_slot);

    // ---- Shares ----

    let shares: [u64; 16] = match layout {
        ShareLayout::Standard => {
            let mut shares = denomination_split_with(delegated_ballots, |slot| {
                proxy_share_prf(&dc_seed, domain_tags::PROXY_PRF_DOMAIN_REMAINDER, slot)
            });
            // One PRF call seeds the whole shuffle, so its share index is 0.
            let seed = proxy_share_prf(&dc_seed, domain_tags::PROXY_PRF_DOMAIN_SHUFFLE, 0);
            shuffle_with_seed(&mut shares, &seed);
            shares
        }
        ShareLayout::SingleShare => {
            let mut shares = [0; 16];
            shares[0] = delegated_ballots;
            shares
        }
    };
    for (i, &share) in shares.iter().enumerate() {
        if share >= SHARE_VALUE_LIMIT {
            return Err(ProxyDelegationBuildError::InvalidShares(format!(
                "share {i} = {share} is not below 2^30"
            )));
        }
    }

    // ---- El Gamal encryption (condition 11) ----

    let g = spend_auth_g_affine();
    let share_randomness: [pallas::Base; 16] =
        core::array::from_fn(|i| derive_share_randomness(&dc_seed, i as u8, layout));
    let mut enc_c1_x = [pallas::Base::zero(); 16];
    let mut enc_c2_x = [pallas::Base::zero(); 16];
    let mut enc_c1_y = [pallas::Base::zero(); 16];
    let mut enc_c2_y = [pallas::Base::zero(); 16];
    let mut encrypted_shares: [EncryptedShareOutput; 16] =
        core::array::from_fn(|i| EncryptedShareOutput {
            c1: [0u8; 32],
            c2: [0u8; 32],
            share_index: i as u32,
            plaintext_value: shares[i],
            randomness: share_randomness[i].to_repr(),
        });
    for i in 0..16 {
        let r = base_to_scalar(share_randomness[i]).expect("p < q makes every Base a Scalar");
        let c1 = (g * r).to_affine();
        let c2 = (g * pallas::Scalar::from(shares[i]) + ea_pk * r).to_affine();
        let identity = |component: &str| {
            ProxyDelegationBuildError::InvalidEncryptedShare(format!(
                "share {i} {component} is identity"
            ))
        };
        (enc_c1_x[i], enc_c1_y[i]) = point_xy(c1).ok_or_else(|| identity("c1"))?;
        (enc_c2_x[i], enc_c2_y[i]) = point_xy(c2).ok_or_else(|| identity("c2"))?;
        encrypted_shares[i].c1 = c1.to_bytes();
        encrypted_shares[i].c2 = c2.to_bytes();
    }

    // ---- Share commitments and DC (conditions 10 and 13) ----

    let share_blinds: [pallas::Base; 16] =
        core::array::from_fn(|i| derive_share_blind(&dc_seed, i as u8));
    let share_comms: [pallas::Base; 16] = core::array::from_fn(|i| {
        share_commitment(
            share_blinds[i],
            enc_c1_x[i],
            enc_c2_x[i],
            enc_c1_y[i],
            enc_c2_y[i],
        )
    });
    let shares_hash_val = shares_hash(share_blinds, enc_c1_x, enc_c2_x, enc_c1_y, enc_c2_y);
    let delegation_commitment =
        delegation_commitment_hash(voting_round_id, shares_hash_val, delegate_index);

    // ---- Spend authority (condition 4) ----

    let r_vpk = (g * (vsk + alpha_v)).to_affine();
    let (r_vpk_x, r_vpk_y) =
        point_xy(r_vpk).ok_or(ProxyDelegationBuildError::InvalidRandomizedVotingPublicKey)?;

    // ---- Tree root (condition 1) ----

    let vote_comm_tree_root = vote_comm_tree_path.iter().enumerate().fold(
        vote_authority_note_old,
        |node, (level, sibling)| {
            if vote_comm_tree_position & (1 << level) == 0 {
                poseidon_hash_2(node, *sibling)
            } else {
                poseidon_hash_2(*sibling, node)
            }
        },
    );

    let circuit = Circuit {
        vote_comm_tree_path: Value::known(vote_comm_tree_path),
        vote_comm_tree_position: Value::known(vote_comm_tree_position),
        vpk_g_d: Value::known(vpk_g_d),
        vpk_pk_d: Value::known(vpk_pk_d),
        total_note_value: Value::known(pallas::Base::from(van_ballots)),
        remaining_note_value: Value::known(pallas::Base::from(remaining_ballots)),
        van_comm_rand: Value::known(van_comm_rand),
        vote_authority_note_old: Value::known(vote_authority_note_old),
        vsk: Value::known(vsk),
        rivk_v: Value::known(rivk_v),
        alpha_v: Value::known(alpha_v),
        vsk_nk: Value::known(vsk_nk),
        shares: shares.map(|share| Value::known(pallas::Base::from(share))),
        enc_share_c1_x: enc_c1_x.map(Value::known),
        enc_share_c2_x: enc_c2_x.map(Value::known),
        enc_share_c1_y: enc_c1_y.map(Value::known),
        enc_share_c2_y: enc_c2_y.map(Value::known),
        share_blinds: share_blinds.map(Value::known),
        share_randomness: share_randomness.map(Value::known),
        ea_pk: Value::known(ea_pk),
        delegate_index: Value::known(pallas::Base::from(u64::from(delegate_index))),
        dc_slot: Value::known(pallas::Base::from(u64::from(dc_slot))),
    };
    let instance = Instance::from_parts(
        van_nullifier,
        r_vpk_x,
        r_vpk_y,
        vote_authority_note_new,
        delegation_commitment,
        vote_comm_tree_root,
        pallas::Base::from(u64::from(anchor_height)),
        voting_round_id,
        ea_pk_x,
        ea_pk_y,
        dc_slot_nullifier,
    );

    Ok(ProxyDelegationBundle {
        circuit,
        instance,
        r_vpk_bytes: r_vpk.to_bytes(),
        alpha_v,
        encrypted_shares,
        shares_hash: shares_hash_val,
        share_blinds,
        share_comms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::Group;
    use crate::shares_hash::shares_hash_from_comms;
    use crate::vote_proof::derive_vote_authority_transition;

    /// One build's inputs; each test overrides the fields it exercises.
    #[derive(Clone)]
    struct Inputs {
        van_ballots: u64,
        ea_pk: pallas::Affine,
        alpha_v: pallas::Scalar,
        delegate_index: u32,
        delegated_ballots: u64,
        dc_slot: u8,
        layout: ShareLayout,
    }

    impl Default for Inputs {
        fn default() -> Self {
            Inputs {
                van_ballots: 10_000,
                ea_pk: (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine(),
                alpha_v: pallas::Scalar::from(7u64),
                delegate_index: 17,
                // 4 x 1,000 and 5 x 100 fill the denomination slots, so 300
                // ballots go to the remainder slots.
                delegated_ballots: 4_800,
                dc_slot: 3,
                layout: ShareLayout::Standard,
            }
        }
    }

    const DC_SEED: [u8; 32] = [0x5e; 32];

    fn test_sk() -> SpendingKey {
        SpendingKey::from_bytes([0x42; 32]).expect("valid spending key")
    }

    fn test_round_id() -> pallas::Base {
        pallas::Base::from(0xCAFE_u64)
    }

    fn test_van_comm_rand() -> pallas::Base {
        pallas::Base::from(0xDEAD_u64)
    }

    fn build(inputs: &Inputs) -> Result<ProxyDelegationBundle, ProxyDelegationBuildError> {
        build_proxy_delegation_proof(
            &test_sk(),
            1,
            inputs.van_ballots,
            test_van_comm_rand(),
            test_round_id(),
            [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH],
            0,
            123,
            inputs.ea_pk,
            inputs.alpha_v,
            inputs.delegate_index,
            inputs.delegated_ballots,
            inputs.dc_slot,
            DC_SEED,
            inputs.layout,
        )
    }

    fn plaintexts(bundle: &ProxyDelegationBundle) -> [u64; 16] {
        core::array::from_fn(|i| bundle.encrypted_shares[i].plaintext_value)
    }

    #[test]
    fn proxy_share_prf_has_frozen_test_vector() {
        let hash = proxy_share_prf(&DC_SEED, domain_tags::PROXY_PRF_DOMAIN_ELGAMAL, 0);

        assert_eq!(
            hash,
            [
                0x1c, 0x1c, 0xcb, 0xe3, 0x83, 0x5d, 0x8f, 0x5c, 0xb3, 0x7e, 0xd8, 0x29, 0x4b, 0x04,
                0xb1, 0x41, 0xcc, 0xb5, 0x88, 0xec, 0x82, 0xc8, 0xd5, 0x4b, 0x98, 0xc4, 0xe0, 0xce,
                0xa7, 0x7d, 0x66, 0x96, 0x3c, 0x16, 0xe0, 0x1e, 0x0d, 0xd5, 0x9c, 0x42, 0x6c, 0x25,
                0x7c, 0xaf, 0x32, 0xed, 0xa0, 0x17, 0xc9, 0xe8, 0x90, 0x60, 0xbe, 0x8f, 0xfe, 0x38,
                0xaf, 0xf0, 0xbf, 0x57, 0x0c, 0x31, 0xba, 0x6c,
            ]
        );
    }

    #[test]
    fn share_layout_numbers_are_pinned() {
        assert_eq!(ShareLayout::Standard as u8, 0);
        assert_eq!(ShareLayout::SingleShare as u8, 1);
    }

    #[test]
    fn rejects_dc_slot_past_the_cap() {
        for dc_slot in [8, u8::MAX] {
            let err = build(&Inputs {
                dc_slot,
                ..Inputs::default()
            })
            .expect_err("slots past 7 must be rejected");
            assert!(matches!(
                err,
                ProxyDelegationBuildError::InvalidDcSlot(rejected) if rejected == dc_slot
            ));
        }
    }

    #[test]
    fn rejects_zero_delegate_index() {
        let err = build(&Inputs {
            delegate_index: 0,
            ..Inputs::default()
        })
        .expect_err("delegate 0 must be rejected");
        assert!(matches!(err, ProxyDelegationBuildError::ZeroDelegateIndex));
    }

    #[test]
    fn rejects_zero_weight() {
        let err = build(&Inputs {
            delegated_ballots: 0,
            ..Inputs::default()
        })
        .expect_err("w = 0 must be rejected");
        assert!(matches!(
            err,
            ProxyDelegationBuildError::ZeroDelegatedWeight
        ));
    }

    #[test]
    fn rejects_weight_above_the_van() {
        let err = build(&Inputs {
            delegated_ballots: 10_001,
            ..Inputs::default()
        })
        .expect_err("w > W must be rejected");
        assert!(matches!(
            err,
            ProxyDelegationBuildError::DelegatedWeightExceedsVan {
                delegated_ballots: 10_001,
                van_ballots: 10_000,
            }
        ));

        // The zero-weight successor of a full delegation cannot delegate.
        let err = build(&Inputs {
            van_ballots: 0,
            delegated_ballots: 1,
            ..Inputs::default()
        })
        .expect_err("a W = 0 VAN must be rejected");
        assert!(matches!(
            err,
            ProxyDelegationBuildError::DelegatedWeightExceedsVan {
                delegated_ballots: 1,
                van_ballots: 0,
            }
        ));
    }

    #[test]
    fn rejects_weights_past_the_share_range() {
        let err = build(&Inputs {
            van_ballots: (1 << 30) + 1,
            ..Inputs::default()
        })
        .expect_err("no VAN holds more than 2^30 ballots");
        assert!(matches!(
            err,
            ProxyDelegationBuildError::InvalidVanBallots(rejected) if rejected == (1 << 30) + 1
        ));

        // One share cannot carry 2^30 ballots, but the standard split can.
        let all = Inputs {
            van_ballots: 1 << 30,
            delegated_ballots: 1 << 30,
            ..Inputs::default()
        };
        let err = build(&Inputs {
            layout: ShareLayout::SingleShare,
            ..all.clone()
        })
        .expect_err("a 2^30 single share must be rejected");
        assert!(matches!(err, ProxyDelegationBuildError::InvalidShares(_)));
        build(&all).expect("the standard layout splits 2^30 ballots");
    }

    #[test]
    fn rejects_identity_keys() {
        let err = build(&Inputs {
            ea_pk: pallas::Point::identity().to_affine(),
            ..Inputs::default()
        })
        .expect_err("identity ea_pk must be rejected");
        assert!(matches!(
            err,
            ProxyDelegationBuildError::InvalidElectionPublicKey
        ));

        let err = build(&Inputs {
            alpha_v: -extract_vsk(&test_sk()),
            ..Inputs::default()
        })
        .expect_err("alpha_v = -vsk makes r_vpk the identity");
        assert!(matches!(
            err,
            ProxyDelegationBuildError::InvalidRandomizedVotingPublicKey
        ));
    }

    /// A rebuild from the same inputs reproduces every public output, so a
    /// re-prove publishes the same DC, shares and commitments.
    #[test]
    fn output_is_deterministic() {
        for layout in [ShareLayout::Standard, ShareLayout::SingleShare] {
            let inputs = Inputs {
                layout,
                ..Inputs::default()
            };
            let a = build(&inputs).expect("valid inputs");
            let b = build(&inputs).expect("valid inputs");

            assert_eq!(
                a.instance.to_halo2_instance(),
                b.instance.to_halo2_instance()
            );
            assert_eq!(a.r_vpk_bytes, b.r_vpk_bytes);
            assert_eq!(a.shares_hash, b.shares_hash);
            assert_eq!(a.share_blinds, b.share_blinds);
            assert_eq!(a.share_comms, b.share_comms);
            for (x, y) in a.encrypted_shares.iter().zip(&b.encrypted_shares) {
                assert_eq!(
                    (x.c1, x.c2, x.share_index, x.plaintext_value, x.randomness),
                    (y.c1, y.c2, y.share_index, y.plaintext_value, y.randomness)
                );
            }
        }
    }

    /// The outputs agree with each other and with the layout, and ZKP #2 sees
    /// the same VANs: condition 5 publishes ZKP #2's nullifier for the spent
    /// VAN, and the successor is the VAN ZKP #2 opens for `W - w` ballots.
    #[test]
    fn outputs_are_consistent() {
        let sk = test_sk();
        let fvk: FullViewingKey = (&sk).into();
        // ZKP #2's full-authority VAN holding `ballots`.
        let zkp2_van = |ballots: u64| {
            derive_vote_authority_transition(
                &sk,
                1,
                ballots * crate::params::BALLOT_DIVISOR,
                test_van_comm_rand(),
                test_round_id(),
                1,
                MAX_PROPOSAL_AUTHORITY,
            )
            .expect("valid ZKP #2 transition")
            .vote_authority_note_old
        };

        for layout in [ShareLayout::Standard, ShareLayout::SingleShare] {
            let inputs = Inputs {
                layout,
                ..Inputs::default()
            };
            let b = build(&inputs).expect("valid inputs");
            let instance = &b.instance;

            assert_eq!(
                instance.van_nullifier,
                van_nullifier_hash(fvk.nk().inner(), test_round_id(), zkp2_van(10_000))
            );
            assert_eq!(instance.vote_authority_note_new, zkp2_van(5_200));

            let shares = plaintexts(&b);
            assert_eq!(shares.iter().sum::<u64>(), inputs.delegated_ballots);
            let nonzero: Vec<usize> = (0..16).filter(|&i| shares[i] != 0).collect();
            match layout {
                // 9 denomination shares and 7 remainder shares.
                ShareLayout::Standard => assert_eq!(nonzero.len(), 16),
                ShareLayout::SingleShare => assert_eq!(nonzero, [0]),
            }

            assert_eq!(shares_hash_from_comms(b.share_comms), b.shares_hash);
            assert_eq!(
                instance.delegation_commitment,
                delegation_commitment_hash(test_round_id(), b.shares_hash, inputs.delegate_index)
            );
            let r_vpk = pallas::Affine::from_xy(instance.r_vpk_x, instance.r_vpk_y)
                .expect("r_vpk is on the curve");
            assert_eq!(b.r_vpk_bytes, r_vpk.to_bytes());
        }
    }

    /// Rebuilding a DC in the other layout, even from the same seed, never
    /// reuses an El Gamal nonce, and no nonce equals a blind.
    #[test]
    fn layouts_never_share_a_nonce() {
        for i in 0..16u8 {
            let standard = derive_share_randomness(&DC_SEED, i, ShareLayout::Standard);
            let single = derive_share_randomness(&DC_SEED, i, ShareLayout::SingleShare);
            assert_ne!(standard, single, "share {i}");
            assert_ne!(standard, derive_share_blind(&DC_SEED, i), "share {i}");
        }
    }

    /// Pins the outputs a re-prove must reproduce byte for byte: the layout-0
    /// share split, the layout-1 randomness and shares hash, and both DCs.
    #[test]
    fn frozen_outputs() {
        let standard = build(&Inputs::default()).expect("valid inputs");
        let single = build(&Inputs {
            layout: ShareLayout::SingleShare,
            ..Inputs::default()
        })
        .expect("valid inputs");

        assert_eq!(
            plaintexts(&standard),
            [46, 42, 100, 39, 100, 1000, 100, 78, 1000, 100, 1000, 11, 69, 15, 1000, 100]
        );
        assert_eq!(
            standard.instance.delegation_commitment.to_repr(),
            [
                148, 116, 106, 32, 64, 201, 102, 242, 140, 239, 95, 129, 133, 233, 15, 8, 50, 120,
                186, 65, 131, 68, 10, 73, 63, 64, 208, 43, 134, 211, 189, 1,
            ]
        );
        assert_eq!(
            single.encrypted_shares[0].randomness,
            [
                209, 60, 29, 200, 194, 246, 153, 175, 139, 250, 196, 111, 220, 139, 192, 246, 102,
                172, 33, 1, 96, 206, 13, 23, 60, 2, 25, 194, 215, 57, 114, 25,
            ]
        );
        assert_eq!(
            single.shares_hash.to_repr(),
            [
                145, 180, 227, 154, 98, 72, 136, 106, 224, 201, 209, 90, 241, 58, 204, 252, 192,
                60, 230, 150, 173, 251, 192, 64, 163, 28, 162, 212, 124, 147, 172, 7,
            ]
        );
        assert_eq!(
            single.instance.delegation_commitment.to_repr(),
            [
                184, 21, 183, 126, 134, 6, 100, 171, 0, 126, 128, 30, 172, 64, 34, 13, 71, 163,
                230, 125, 116, 254, 249, 36, 45, 210, 21, 102, 248, 99, 191, 21,
            ]
        );
    }
}
