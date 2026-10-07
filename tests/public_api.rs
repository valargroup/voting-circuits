use voting_circuits::proxy_delegation::{
    build_proxy_delegation_proof, create_proxy_delegation_proof, verify_proxy_delegation_proof,
    Circuit, Instance, ProxyDelegationBuildError, ProxyDelegationBundle, ShareLayout,
};
use voting_circuits::{ProveError, MAX_PROPOSAL_AUTHORITY, VOTE_COMM_TREE_DEPTH};
use voting_crypto_deps::{orchard::keys::SpendingKey, pasta_curves::pallas};

#[test]
fn fresh_delegation_authority_is_public_and_canonical() {
    assert_eq!(MAX_PROPOSAL_AUTHORITY, (1u64 << 51) - 1);
}

/// Pins the signatures of the proxy delegation (ZKP #4) entry points.
#[test]
fn proxy_delegation_entry_points_are_public() {
    let _: fn(
        &SpendingKey,
        u32,
        u64,
        pallas::Base,
        pallas::Base,
        [pallas::Base; VOTE_COMM_TREE_DEPTH],
        u32,
        u32,
        pallas::Affine,
        pallas::Scalar,
        u32,
        u64,
        u8,
        [u8; 32],
        ShareLayout,
    ) -> Result<ProxyDelegationBundle, ProxyDelegationBuildError> = build_proxy_delegation_proof;
    let _: fn(Circuit, &Instance) -> Result<Vec<u8>, ProveError> = create_proxy_delegation_proof;
    let _: fn(&[u8], &Instance) -> Result<(), String> = verify_proxy_delegation_proof;
}
