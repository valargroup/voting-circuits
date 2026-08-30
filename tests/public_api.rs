use voting_circuits::MAX_PROPOSAL_AUTHORITY;

#[test]
fn fresh_delegation_authority_is_public_and_canonical() {
    assert_eq!(MAX_PROPOSAL_AUTHORITY, (1u64 << 51) - 1);
}
