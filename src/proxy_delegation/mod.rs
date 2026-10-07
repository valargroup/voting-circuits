//! Proxy Delegation ZKP circuit (ZKP #4).
//!
//! Spends a full-authority VAN as ZKP #2 does and emits a successor VAN with
//! weight `W - w`, plus a delegation commitment (DC) to 16 El Gamal shares of
//! `w` and a private delegate index `d`. The DC is a vote commitment for
//! proposal 0, so ZKP #3 reveals its shares into delegate `d`'s pool unchanged.
//!
//! The circuit verifies 14 conditions:
//! - **Condition 1**: VAN Membership (Poseidon Merkle path to `vote_comm_tree_root`).
//! - **Condition 2**: VAN Integrity, with `MAX_PROPOSAL_AUTHORITY` as a constant
//!   cell, so only a VAN that has not voted can delegate.
//! - **Condition 3**: Diversified Address Integrity (CommitIvk chain, as in ZKP #2).
//! - **Condition 4**: Spend Authority (`r_vpk`, as in ZKP #2).
//! - **Condition 5**: VAN Nullifier Integrity, identical to ZKP #2, so a VAN
//!   spent by either circuit is spent for both.
//! - **Condition 6**: Successor VAN Integrity: same address, round, `rand` and
//!   full authority, with weight `W_new`.
//! - **Condition 7**: `W_new` in `[0, 2^30)` (strict lookup range check).
//! - **Condition 8**: Conservation: `sum(shares) + W_new = W`.
//! - **Condition 9**: Shares Range: each share in `[0, 2^30)`.
//! - **Condition 10**: Shares Hash Integrity (16 blinded share commitments).
//! - **Condition 11**: Encryption Integrity: each share El Gamal-encrypted
//!   under `ea_pk`.
//! - **Condition 12**: Non-zero Weight: `w = sum(shares) != 0`.
//! - **Condition 13**: DC Integrity:
//!   `DC = Poseidon(DOMAIN_VC, voting_round_id, shares_hash, 0, d)`, with
//!   proposal 0 as a constant cell and no bound on `d`.
//! - **Condition 14**: DC Slot Nullifier: `dc_slot` in `[0, 8)` and
//!   `dc_slot_nullifier = Poseidon(nk, "proxy delegation slot" + dc_slot,
//!   voting_round_id, rand)`.
//!
//! Conditions 7, 8, 9 and 12 give `1 <= w <= W` over the integers. The chain,
//! not the circuit, keeps `d` on a registered delegate: it accepts a pool share
//! reveal only for `1 <= d < NextDelegateIndex`.

mod builder;
mod circuit;

pub use builder::{
    build_proxy_delegation_proof, ProxyDelegationBuildError, ProxyDelegationBundle, ShareLayout,
};
pub use circuit::{dc_slot_nullifier_hash, delegation_commitment_hash, Circuit, Instance, K};
