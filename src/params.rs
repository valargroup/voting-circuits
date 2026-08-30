//! Protocol-wide constants shared across the three ZKP circuits.

/// Ballot divisor for converting raw zatoshi balance to ballot count.
///
/// `num_ballots = floor(v_total / BALLOT_DIVISOR)`
///
/// ZKP #1 (delegation) commits to it; ZKP #2 (vote_proof) must agree.
///
/// Note: the delegation circuit's condition 8 proves a relation slightly
/// weaker than exact floor-division — a one-ballot under-claim is admissible
/// for ~34% of `v_total` values. Over-claim is impossible. See
/// `src/delegation/README.md` §8 ("Soundness scope") for the analysis and
/// the available tightening approaches.
pub const BALLOT_DIVISOR: u64 = 12_500_000;

/// Bit width of each encrypted vote share in ZKP #2.
///
/// This bound keeps vote-share values identical in the Pallas base and scalar
/// fields, and bounds the discrete logarithm recovered after decryption.
pub(crate) const SHARE_VALUE_BITS: usize = 30;

/// Exclusive upper bound for each encrypted vote share.
pub(crate) const SHARE_VALUE_LIMIT: u64 = 1 << SHARE_VALUE_BITS;

/// Bit width of each lookup range-check word.
pub(crate) const RANGE_CHECK_WORD_BITS: usize = 10;

/// Number of lookup words used to range-check each vote share.
pub(crate) const SHARE_VALUE_RANGE_WORDS: usize = SHARE_VALUE_BITS / RANGE_CHECK_WORD_BITS;

/// Maximum number of usable proposals in one voting round.
///
/// Proposal IDs are 1-indexed. Bit zero remains reserved as the unset sentinel,
/// so the proposal-authority bitmask needs one more bit than this limit.
pub(crate) const MAX_PROPOSALS: usize = 50;

/// Number of bits in the proposal-authority bitmask, including bit zero.
pub(crate) const PROPOSAL_AUTHORITY_BITS: usize = MAX_PROPOSALS + 1;

/// Full proposal authority assigned by a fresh delegation.
///
/// This mask includes reserved bit zero and usable bits 1 through 50.
/// Delegation constrains it as a circuit constant, so changing it changes the
/// delegation verification key.
pub const MAX_PROPOSAL_AUTHORITY: u64 = (1u64 << PROPOSAL_AUTHORITY_BITS) - 1;

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
