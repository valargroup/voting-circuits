# Share Splitting: Parameter Rationale

## Context

When casting a vote (ZKP2), a voter's total delegated weight (`num\_ballots`) is split into 16 El Gamal-encrypted shares. Each share is revealed individually (ZKP3) in a way that is **unlinkable** to the voter or to the other 15 shares from the same vote. No observer can group 16 shares back to one voter.

The share-splitting algorithm is a **defense-in-depth** layer: if unlinkability were somehow broken (EA + helper collusion, side-channel, etc.), individual share values should not fingerprint the voter's exact balance. The algorithm creates *anonymity tiers* — many voters at different balances produce identical individual share values, so a decrypted share may reveal the tier but not the position within it.

## Algorithm overview

**Phase 1 — Greedy fill (up to 9 slots).** Repeatedly subtract the largest standard denomination from `[10M, 1M, 100K, 10K, 1K, 100, 10, 1]` (in ballots) that fits. Stop after 9 slots or when balance is zero.

*Example: 4,800 ballots (600 ZEC) → `[1K, 1K, 1K, 1K, 100, 100, 100, 100, 100]`, 300 remaining.*

**Phase 2 — Remainder distribution (up to 7 slots).** Spread any leftover across the remaining slots using PRF-derived weights (keyed by spending key, bound to round/proposal/VAN). Two voters with identical balances produce different remainder patterns.

*Example (continued): 300 ballots → `[78, 49, 50, 39, 11, 38, 35]`.*

**Phase 3 — Shuffle.** Deterministic Fisher-Yates permutation randomizes all 16 slot positions. Without this, the first slots always hold the largest denominations, leaking balance magnitude through position.

## Anonymity bands

The point of denominations (vs. a pure random split) is that a single decrypted share reveals the **tier** but not the exact balance. The 10× spacing between denominations defines the anonymity bands:

| Decrypted share       | Voter's balance could be        | Band width      |
| --------------------- | ------------------------------- | --------------- |
| `1` (0.125 ZEC)       | 0.125 ZEC – 1.125 ZEC           | ~1 ZEC          |
| `10` (1.25 ZEC)       | 1.25 ZEC – 12.375 ZEC           | ~11 ZEC         |
| `100` (12.5 ZEC)      | 12.5 ZEC – 124.875 ZEC          | ~112 ZEC        |
| `1K` (125 ZEC)        | 125 ZEC – 1,249.875 ZEC         | ~1,125 ZEC      |
| `10K` (1,250 ZEC)     | 1,250 ZEC – 12,499.875 ZEC      | ~11,250 ZEC     |
| `100K` (12,500 ZEC)   | 12,500 ZEC – 124,999.875 ZEC    | ~112,500 ZEC    |
| `1M` (125,000 ZEC)    | 125,000 ZEC – 1,249,999.875 ZEC | ~1,125,000 ZEC  |
| `10M` (1,250,000 ZEC) | 1,250,000 ZEC – 21,000,000 ZEC  | ~19,750,000 ZEC |

Any voter whose decomposition includes at least one share of that denomination falls in the band. With a pure random split, a share of `347` narrows the estimate to `~347 × 16 ≈ 5,500 ballots` — effectively an anonymity set of 1.

Base 10 is a pragmatic choice. Base 2 would make each decomposition unique (deterministic fingerprinting). Base 100+ would push most of the balance into remainder slots, reducing the number of standard-valued shares.
