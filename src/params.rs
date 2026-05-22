//! Protocol-wide constants shared across the three ZKP circuits.

/// Ballot divisor for converting raw zatoshi balance to ballot count.
///
/// `num_ballots = floor(v_total / BALLOT_DIVISOR)`
///
/// ZKP #1 (delegation) commits to it; ZKP #2 (vote_proof) must agree.
pub const BALLOT_DIVISOR: u64 = 12_500_000;

/// Depth of the Poseidon-based vote commitment tree. Shared by ZKP #2
/// (vote_proof) Merkle membership and ZKP #3 (share_reveal) Merkle membership.
///
/// Reduced from Zcash's depth 32 (~4.3B) because governance voting
/// produces far fewer leaves than a full shielded pool. Each voter
/// generates 1 leaf per delegation + 2 per vote, so even 10K voters
/// × 50 proposals ≈ 1M leaves — well within 2^24 ≈ 16.7M capacity.
///
/// Must match `vote_commitment_tree::TREE_DEPTH`.
pub const VOTE_COMM_TREE_DEPTH: usize = 24;
