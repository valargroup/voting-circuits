//! IMT (Indexed Merkle Tree) utilities for the delegation proof system.
//!
//! Provides out-of-circuit helpers for building and verifying Poseidon-based
//! Indexed Merkle Tree non-membership proofs using K=2 punctured-range leaves.
//! Each leaf stores three sorted nullifier boundaries `[nf_lo, nf_mid, nf_hi]`
//! and the leaf hash is `Poseidon3(nf_lo, nf_mid, nf_hi)`, followed by a
//! standard Merkle path authenticating the leaf.
//! A non-membership proof shows that a nullifier falls strictly inside
//! `(nf_lo, nf_hi)` and is not equal to `nf_mid`.
//! Used by the delegation circuit and builder.

use alloc::string::String;
use ff::PrimeField;
use halo2_gadgets::poseidon::primitives::{self as poseidon, ConstantLength};
use pasta_curves::pallas;

/// Depth of the nullifier Indexed Merkle Tree Merkle path (Poseidon-based).
/// Total Poseidon calls per proof = 2 (leaf hash, ConstantLength<3>) + 29 (path) = 31.
pub const IMT_DEPTH: usize = 29;

/// Protocol identifier for governance authorization, encoded as a little-endian
/// Pallas field element. Used to derive the nullifier domain for this application.
pub(crate) fn gov_auth_domain_tag() -> pallas::Base {
    let mut bytes = [0u8; 32];
    bytes[..24].copy_from_slice(b"governance authorization");
    pallas::Base::from_repr(bytes).unwrap()
}

/// Compute Poseidon hash of two field elements (out of circuit).
pub(crate) fn poseidon_hash_2(a: pallas::Base, b: pallas::Base) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<2>, 3, 2>::init().hash([a, b])
}

/// Compute Poseidon hash of three field elements (out of circuit).
/// Uses `ConstantLength<3>` (width-3 sponge, 2 absorption blocks).
pub(crate) fn poseidon_hash_3(a: pallas::Base, b: pallas::Base, c: pallas::Base) -> pallas::Base {
    poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<3>, 3, 2>::init().hash([a, b, c])
}

/// Derive the nullifier domain for a voting round (out of circuit).
///
/// `dom = Poseidon("governance authorization", vote_round_id)`
///
/// The nullifier domain scopes alternate nullifiers to a specific application
/// instance (ZIP §Nullifier Domains). This application hashes its protocol
/// identifier with the vote round ID to produce a unique domain per round.
pub fn derive_nullifier_domain(vote_round_id: pallas::Base) -> pallas::Base {
    poseidon_hash_2(gov_auth_domain_tag(), vote_round_id)
}

/// Compute alternate nullifier out-of-circuit (ZIP §Alternate Nullifier Derivation).
///
/// `nf_dom = Poseidon(nk, dom, nf^old)`
///
/// where `dom` is the nullifier domain (see [`derive_nullifier_domain`]).
/// Single ConstantLength<3> call (2 permutations at rate=2).
pub(crate) fn gov_null_hash(
    nk: pallas::Base,
    dom: pallas::Base,
    real_nf: pallas::Base,
) -> pallas::Base {
    poseidon_hash_3(nk, dom, real_nf)
}

/// IMT non-membership proof data (K=2 punctured-range leaf model).
///
/// Each leaf stores three sorted nullifier boundaries `[nf_lo, nf_mid, nf_hi]`.
/// The leaf hash is `Poseidon3(nf_lo, nf_mid, nf_hi)`, followed by a standard
/// 29-level Merkle path to the root. Non-membership is proven by showing:
///   1. `nf_lo < value < nf_hi` (strict interval)
///   2. `value != nf_mid` (non-equality with interior nullifier)
#[derive(Clone, Debug)]
pub struct ImtProofData {
    /// The Merkle root of the IMT.
    pub root: pallas::Base,
    /// Three sorted nullifier boundaries: `[nf_lo, nf_mid, nf_hi]`.
    pub nf_bounds: [pallas::Base; 3],
    /// Position of the leaf in the tree.
    pub leaf_pos: u32,
    /// Sibling hashes along the 29-level Merkle path (pure siblings).
    pub path: [pallas::Base; IMT_DEPTH],
}

/// Error type for IMT proof fetching failures.
#[derive(Clone, Debug)]
pub struct ImtError(pub String);

impl core::fmt::Display for ImtError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "IMT error: {}", self.0)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ImtError {}

/// Trait for providing IMT non-membership proofs.
///
/// Implementations must return proofs against a consistent root — all proofs
/// from the same provider must share the same `root()` value.
pub trait ImtProvider {
    /// The current IMT root.
    fn root(&self) -> pallas::Base;
    /// Generate a non-membership proof for the given nullifier.
    fn non_membership_proof(&self, nf: pallas::Base) -> Result<ImtProofData, ImtError>;
}

// ================================================================
// SpacedLeafImtProvider (available for proof generation and tests)
// ================================================================

use alloc::vec::Vec;
use ff::Field;

/// Precomputed empty subtree hashes for the IMT (Poseidon-based).
///
/// `empty[0] = Poseidon3(0, 0, 0)` (hash of an all-zero punctured-range leaf),
/// `empty[i] = Poseidon(empty[i-1], empty[i-1])` for i >= 1.
pub fn empty_imt_hashes() -> Vec<pallas::Base> {
    let empty_leaf = poseidon_hash_3(pallas::Base::zero(), pallas::Base::zero(), pallas::Base::zero());
    let mut hashes = vec![empty_leaf];
    for _ in 1..=IMT_DEPTH {
        let prev = *hashes.last().unwrap();
        hashes.push(poseidon_hash_2(prev, prev));
    }
    hashes
}

/// Sentinel spacing exponent: sentinels are placed at `k * 2^249`.
/// With K=2 punctured ranges each leaf spans two consecutive intervals,
/// giving outer span `2 * 2^249 = 2^250` — matching the circuit's 250-bit
/// range check.
const SENTINEL_EXPONENT: u64 = 249;

/// Number of sentinel multiples: `0, 1*step, 2*step, ..., 32*step`.
/// `32 * 2^249 = 2^254` covers the Pallas field (p ≈ 2^254.9).
const SENTINEL_COUNT: u64 = 32;

/// Build the sorted, deduplicated, odd-count sentinel list used by both
/// [`SpacedLeafImtProvider`] and the production `prepare_nullifiers` path.
///
/// Sentinels: `k * 2^249` for `k = 0..=32`, plus `p - 1` to close the tail.
/// If the count is even after dedup, `Fp::from(2)` is inserted after sentinel 0
/// to make it odd (collision probability ≈ 2^{-254}).
fn build_sentinel_list() -> Vec<pallas::Base> {
    let step = pallas::Base::from(2u64).pow([SENTINEL_EXPONENT, 0, 0, 0]);
    let mut nfs: Vec<pallas::Base> = (0u64..=SENTINEL_COUNT)
        .map(|k| step * pallas::Base::from(k))
        .collect();
    nfs.push(-pallas::Base::one()); // p - 1
    nfs.sort();
    nfs.dedup();
    if nfs.len() % 2 == 0 {
        debug_assert_eq!(nfs[0], pallas::Base::zero(), "sentinel 0 must be first");
        nfs.insert(1, pallas::Base::from(2u64));
    }
    nfs
}

/// Build punctured-range triples from a sorted, deduplicated, odd-count
/// nullifier list. Mirrors `imt_tree::tree::build_punctured_ranges` so the
/// test fixture is protected by the same ordering invariant as production.
fn build_punctured_ranges_local(sorted_nfs: &[pallas::Base]) -> Vec<[pallas::Base; 3]> {
    let n = sorted_nfs.len();
    assert!(n >= 3, "need at least 3 sorted nullifiers, got {n}");
    assert!(n % 2 == 1, "sorted nullifier count must be odd (got {n})");

    let num_leaves = (n - 1) / 2;
    (0..num_leaves)
        .map(|i| {
            let base = i * 2;
            let (lo, mid, hi) = (sorted_nfs[base], sorted_nfs[base + 1], sorted_nfs[base + 2]);
            assert!(
                lo < mid && mid < hi,
                "punctured range {i} violates strict ordering: \
                 nf_lo={lo:?}, nf_mid={mid:?}, nf_hi={hi:?}"
            );
            [lo, mid, hi]
        })
        .collect()
}

/// Find the punctured-range index containing `value`. Mirrors
/// `imt_tree::tree::find_punctured_range_for_value`.
fn find_range_for_value(ranges: &[[pallas::Base; 3]], value: pallas::Base) -> Option<usize> {
    let i = ranges.partition_point(|[nf_lo, _, _]| *nf_lo < value);
    if i == 0 {
        return None;
    }
    let idx = i - 1;
    let [nf_lo, nf_mid, nf_hi] = ranges[idx];
    let offset = value - nf_lo;
    let span = nf_hi - nf_lo;
    if offset == pallas::Base::zero() || offset >= span {
        return None;
    }
    if value == nf_mid {
        return None;
    }
    Some(idx)
}

/// IMT provider with evenly-spaced K=2 punctured-range brackets.
///
/// Mirrors the production sentinel injection path: sentinels at `k * 2^249`
/// for `k = 0..=32`, plus `p - 1`, sorted, deduplicated, and padded to odd
/// count with `Fp::from(2)`. Each interior leaf spans exactly `2^250`,
/// satisfying the circuit's 250-bit range check. The tail leaf covers
/// `[32*step, p-1]` with span `≈ 2^126`, well under the bound.
///
/// Used for proof generation (fixture generators) and testing.
#[derive(Debug)]
pub struct SpacedLeafImtProvider {
    /// The root of the IMT.
    root: pallas::Base,
    /// Punctured-range triples: `[nf_lo, nf_mid, nf_hi]` for each leaf.
    leaves: Vec<[pallas::Base; 3]>,
    /// Bottom levels of the subtree (32-leaf subtree → 5 levels).
    /// `subtree_levels[0]` has 32 leaf hashes Poseidon3(nf_lo, nf_mid, nf_hi),
    /// `subtree_levels[5]` has 1 subtree root.
    subtree_levels: Vec<Vec<pallas::Base>>,
}

impl SpacedLeafImtProvider {
    /// Create a new spaced-leaf IMT provider (K=2 punctured-range model).
    ///
    /// Builds the same sentinel layout as the production `prepare_nullifiers`
    /// path: sorted, deduplicated, padded to odd count, then grouped into
    /// punctured-range triples with strict ordering validation.
    pub fn new() -> Self {
        let sorted_nfs = build_sentinel_list();
        let leaves = build_punctured_ranges_local(&sorted_nfs);
        let empty = empty_imt_hashes();

        let empty_leaf_hash = poseidon_hash_3(
            pallas::Base::zero(),
            pallas::Base::zero(),
            pallas::Base::zero(),
        );
        let mut level0 = vec![empty_leaf_hash; 32];
        for (k, bounds) in leaves.iter().enumerate() {
            level0[k] = poseidon_hash_3(bounds[0], bounds[1], bounds[2]);
        }

        let mut subtree_levels = vec![level0];
        for _l in 1..=5 {
            let prev = subtree_levels.last().unwrap();
            let mut current = Vec::with_capacity(prev.len() / 2);
            for j in 0..(prev.len() / 2) {
                current.push(poseidon_hash_2(prev[2 * j], prev[2 * j + 1]));
            }
            subtree_levels.push(current);
        }

        let mut root = subtree_levels[5][0];
        for l in 5..IMT_DEPTH {
            root = poseidon_hash_2(root, empty[l]);
        }

        SpacedLeafImtProvider {
            root,
            leaves,
            subtree_levels,
        }
    }
}

impl ImtProvider for SpacedLeafImtProvider {
    fn root(&self) -> pallas::Base {
        self.root
    }

    fn non_membership_proof(&self, nf: pallas::Base) -> Result<ImtProofData, ImtError> {
        let k = find_range_for_value(&self.leaves, nf).ok_or_else(|| {
            ImtError(alloc::format!(
                "nullifier {nf:?} not in any punctured range"
            ))
        })?;

        let nf_bounds = self.leaves[k];
        let leaf_pos = k as u32;

        let empty = empty_imt_hashes();
        let mut path = [pallas::Base::zero(); IMT_DEPTH];

        let mut idx = k;
        for l in 0..5 {
            let sibling_idx = idx ^ 1;
            path[l] = self.subtree_levels[l][sibling_idx];
            idx >>= 1;
        }

        for l in 5..IMT_DEPTH {
            path[l] = empty[l];
        }

        Ok(ImtProofData {
            root: self.root,
            nf_bounds,
            leaf_pos,
            path,
        })
    }
}
