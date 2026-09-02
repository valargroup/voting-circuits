# Encrypted Votes: Alternative Designs

## Status

This document is the design and benchmark record behind the encrypted
weighted-vote tally: the adopted construction, every alternative considered,
why each was rejected, and the measurements that drove the decisions.

The selected design — the weighted one-hot ElGamal construction with a
decision-bound ZKP 1.5 ("encrypt-choice") split — is now implemented in
production (`src/encrypt_choice/`, `src/vote_proof/`, `src/share_reveal/`,
`src/bridge.rs`), using the fully parallel K=11 layout. The code and its
module docs are the source of truth for the shipped protocol; this document
records the alternatives and the evidence.

All measurements below share one environment unless noted: Apple M4 Max, 16
logical CPUs, 128 GiB RAM, macOS arm64, `rustc 1.96.1`, commit `01ae783`,
default Zakura cryptography backend, `RAYON_NUM_THREADS=8`, warm proving
after an unmeasured warm-up, peak RSS via `/usr/bin/time -l`. Benchmarks
measured only the replacement vote-encryption cryptography — the production
VAN membership, ownership, spend-authority, and authority-decrement
constraints were excluded — so they are incremental costs, not totals.

## Adopted construction: weighted one-hot ElGamal

Replace the plaintext decision reveal with an encrypted weighted one-hot
vector. For a fixed circuit maximum `M = MAX_DECISION_BUCKETS = 16` and a
per-proposal public active count `2 <= D <= M`, each of the 16 weight shares
`w_i` is encrypted into every bucket:

```text
b_j in {0, 1};  sum_j(b_j) = 1;  b_j = 0 for j >= D
E_i,j = Enc_PK(b_j * w_i; r_i,j) = ([r_i,j]G, [b_j * w_i]G + [r_i,j]PK)
```

One shared one-hot selector is reused across all 16 shares (a vote cannot
split its shares across decisions). Inactive buckets are proof-bound
encryptions of zero with fresh independent randomness. Each share's `M`
ciphertexts are bound (both coordinates) into a blinded per-share commitment;
ZKP3 reveals one committed vector at a time, the chain adds the `D` active
ciphertexts into per-`(round, proposal, bucket)` accumulators after
duplicate-nullifier rejection, and only the aggregate ciphertexts are
decrypted (verifiably: DLEQ or threshold decryption shares) after the round
closes, with post-decryption weight-conservation checks. Fixing `M` in the
circuit shape supports variable option counts without per-proposal keys; an
exact-`D` circuit family was kept as a fallback and never needed.

### Shared weight-point optimization (adopted)

The straightforward circuit computes `[b_j * w_i]G` independently for all
`16M` buckets. The adopted variant computes `W_i = [w_i]G` once per share,
then per bucket `C1 = [r]G`, `R = [r]PK`, `S = R + W_i` (complete addition),
and selects the C2 coordinates with the constrained boolean selector:
`C2 = R + b_j * (S - R)` on both coordinates. This removes `16(M-1)` 30-bit
fixed-base multiplications; the per-bucket `[r]G` / `[r]PK` work remains
because zero ciphertexts still need fresh randomness. Identity handling: `R`
is constrained non-identity, `S` uses a complete-addition gadget, and
selection happens between two fully constrained candidates — never through
an identity-encoded input to incomplete addition.

Measured unified-circuit scaling (fully parallel layout, K=11 held at every
`M`; `M = 16` rows are single full-proof samples):

| M | Variant | Rows / K | Total columns | Prove median | Proof | Peak RSS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | straightforward | 1,937 / 11 | 303 | 228 ms | 27.1 KiB | 387 MiB |
| 8 | straightforward | 1,937 / 11 | 663 | 545 ms | 56.6 KiB | 816 MiB |
| 16 | straightforward | 1,937 / 11 | 1,143 | 978 ms | 95.9 KiB | 1,301 MiB |
| 16 | shared weight point | 1,854 / 11 | 1,154 | 925 ms | 96.3 KiB | 1,293 MiB |

The optimization is moderate (-2% to -9% proving), not proportional to `M`.
Columns, proof size, and memory grow roughly linearly with `M` (~20 advice
and ~40 fixed columns, ~5 KiB proof, ~70 MiB RSS per extra bucket).

### Generator-table reuse across ECC tracks (rejected)

Sharing one set of eight Lagrange fixed columns across all `2M` ECC tracks
saves 248 fixed columns and ~7.5 KiB of proof at `M = 16`, but serializes the
fixed-base regions: K grows with `M`, proving explodes from 925 ms to 11.55 s
and peak RSS from 1.29 GiB to 14.9 GiB. Not viable unless a future backend
can share immutable fixed-table storage without a placement conflict.

### Joint-randomizer ECC primitive (closed, upstream-only)

Each ciphertext binds one randomness cell to both `C1 = [r]G` and
`R = [r]PK`, but the fixed-base (3-bit windows) and variable-base (running
sum fused into the double-and-add gates) gadgets decompose `r` separately
and incompatibly. `ScalarVar::from_base` adds no constraints, so a `mul_pair`
wrapper has exactly the existing circuit shape — an exact-shape prototype
(32 variable-base muls, 4,832 rows at k=13, 215 ms) confirmed the expected
speedup is 1.0x. Two cheap-looking aggregations were rejected as unsound:
proving only `sum(C1_j) = [sum(r_j)]G` (per-bucket errors cancel) and
`C1 + λR = [r](G + λPK)` for circuit-fixed λ (cancelling errors are
choosable). A genuinely shared decomposition requires a new upstream-reviewed
`mul_pair` primitive in Zakura `halo2_gadgets`; no application-local copy of
ECC internals is permitted.

## Rejected: per-proof PK window tables

A since-removed feature-gated prototype compared repeated
`NonIdentityPoint::mul` against a four-bit advice-backed window table for
`PK`, constructed once per proof (64 windows × 16 constrained points,
multilinear selection gate). `PK` is a governance session parameter, so a
fixed-column table would force per-key verifying keys, and the pinned Halo2
API has no advice-backed lookup — the table must be interpolated in gates.

| Scalars | Variant | Rows / K | Total columns | Keygen | Prove median | Proof |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | variable-base | 2,416 / 12 | 31 | 48 ms | 104 ms | 4,096 B |
| 16 | 4-bit PK table | 3,969 / 12 | 75 | 431 ms | 171 ms | 8,832 B |
| 256 | variable-base | 38,656 / 16 | 31 | 948 ms | 1,731 ms | 4,352 B |
| 256 | 4-bit PK table | 34,689 / 16 | 75 | 26,067 ms | 2,844 ms | 9,088 B |

The table starts saving rows between 64 and 128 repeated multiplications
(~10% at 256), but the advice width and degree-six selection gate make
proving ~1.64x slower everywhere, with severe keygen and proof-size
regressions. Not integrated. Reopen only if the backend gains an efficient
dynamic/advice lookup or deployment accepts per-key fixed tables and
verifying keys.

## Rejected: Fiat–Shamir randomized batch validation in ZKP2

Publishing all `16M` ciphertexts in ZKP2 and checking two random linear
aggregate equations under a Fiat–Shamir challenge was rejected because:

1. it breaks ZKP2-to-ZKP3 unlinkability — a reveal matches directly to the
   public ZKP2 vector, regrouping all 16 shares of one vote;
2. the proposed circuit arithmetic used the wrong field (Halo2 arithmetic is
   mod the Pallas base field; group scalars are mod the scalar field);
3. keeping the vector hidden removes the native batching advantage — the
   verifier cannot recompute aggregates it does not possess, and a hidden
   in-circuit MSM likely costs more than the individual checks; and
4. repairing it needs a new protocol layer (hidden homomorphic vector
   commitments / linear-opening proofs) for an unproven benefit.

Deterministic sums, prover-supplied aggregates, and hidden vectors without a
sound linear-opening proof remain prohibited.

## Rejected: deferring ElGamal validation to ZKP3

A construction that committed weights, selectors, randomizers, and
coordinates in ZKP2 but proved the ElGamal equations only at reveal time (as
a per-share Fiat–Shamir batch in ZKP3) was rejected:

- **The ZKP3 helper would learn the decision.** The deferred proof witnesses
  `w_i`, the one-hot selector, and every `r_i,j`; zero knowledge hides them
  from verifiers, not from the machine generating the proof. With validation
  in ZKP2, ZKP3 only opens a commitment — the helper sees
  `[Enc(0), Enc(w_i), Enc(0)]` and needs no plaintext witnesses.
- **Invalid committed shares are detected too late.** ZKP2 would accept
  commitments over malformed ciphertexts that can never produce a valid ZKP3
  and are permanently untallyable — unrepairable under the accepted
  commitment, unlike a merely undelivered share whose nullifier is still
  unaccepted.

Consequently ElGamal correctness is proved deterministically before a vote
commitment is accepted, and ZKP3 stays a selective opening proof (publish the
`M` pairs, recompute the blinded per-share commitment, prove membership,
enforce the share nullifier) that never witnesses the decision, weight, or
randomizers.

## The ZKP 1.5 split

At `M = 16` the weighted replacement cryptography alone proves in ~925 ms on
the M4 Max (roughly 2 s on a phone) — meaningful interactive latency at
cast time. With the ECC optimizations above closed, the remaining lever is
moving work off the critical path: every ECC operation in the weighted
construction (`W_i`, `C1`, `R`, `S`) depends only on the shares, the
PRF-derived randomness, and the EA key. Because the randomness PRF is
deterministic and keyed on `(sk, round, proposal, VAN, share, bucket)`,
pre-generation is idempotent — a crash re-derives byte-identical witnesses.

Both variants are commit-and-prove splits joined by a Poseidon **bridge**:
the auxiliary proof and the cast proof expose the same public bridge value,
the cast proof re-opens it from its own witnessed weights (the same cells
that sum to `total_note_value`), and the verifier checks bridge, round,
proposal, and bucket-count equality across the two instances. Both proofs
are submitted together in one vote message — publishing the auxiliary proof
early would leak "this voter is preparing to vote on proposal P."

### Decision-independent pre-generation (rejected)

The original variant is provable *before* the voter picks an option: for
every share it proves all ECC relations plus, for every candidate decision
`d in 0..M`, the commitment ZKP3 would open if the voter picked `d` —
`16M` wide candidate hashes. The cast proof then witnesses the candidate
array, proves the one-hot selection by inner product
(`share_comm_i = sum_d b_d * cand_comm_{i,d}`), and re-opens per-share and
outer bridges.

It met its success criterion — cast-time proving at `M = 16` is **97 ms**
(9.6x below unified, nearly flat in `M`, K=11, no ECC) — but the exhaustive
hashing makes the background proof cost ~3x the unified proof:

| M | Pregen rows / K | Pregen prove | Cast prove | Bundle payload | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1,885 / 11 | 185 ms | 67 ms | 25.9 KiB | 279 MiB |
| 8 | 7,381 / 13 | 1,094 ms | 93 ms | 53.2 KiB | 2,138 MiB |
| 16 | 14,709 / 14 | 2,971 ms | 97 ms | 68.5 KiB | 5,867 MiB |

(A `wide` layout traded ~18% proving and ~20% RSS for 1.6x proof bytes; a
`maxpar` K=11 layout was attainable but its ~4,000 columns made the proof
5.9x larger with doubled verification. The hash-count multiplication, not
the layout, is the fundamental cost.) 5.9 GiB peak RSS at `M = 16` is
acceptable for desktop background work but too much for a phone in one
piece; the split-by-share-group shape (independent per-share workloads,
per-group bridges) was identified as the mobile fallback. This variant was
ultimately rejected because the product flow already knows the decision when
proving can start — the phone kicks off the auxiliary proof the moment the
voter selects a choice — so paying ~3x proving and ~4x memory to be
decision-independent buys nothing.

Also rejected in this space: **speculative full-proof pre-generation**
(background-prove `D` complete unified ZKP2s and submit the chosen one) —
cheapest to engineer and review, still the documented fallback for small-`D`
proposals, but unreasonable at 16+ options (~2 s per candidate proof on a
phone) and it pins `vote_comm_tree_root` anchors whose acceptance window
becomes a chain-policy dependency; **exact-`D` circuit families** (helps
small `D`, multiplies key management, no help at large `D`); and **UI-level
early proving** (hides a few hundred ms at most).

### Decision-bound split (adopted; production "encrypt-choice")

If the decision is already known, the auxiliary circuit applies the
constrained one-hot selectors directly to the `R`/`S` coordinates and hashes
only the *selected* ciphertext vector per share — 16 wide hashes instead of
`16M` — with one compact bridge:

```text
selected_comm_i = Poseidon(domain, blind_i,
                           for j in 0..M: C1.x, C2.x, C1.y, C2.y)
bridge = Poseidon(domain, round, proposal, D,
                  w_0, selected_comm_0, ..., w_15, selected_comm_15)
```

The cast circuit witnesses only the 16 weights and selected commitments,
re-opens the bridge, re-checks the weight sum and 30-bit ranges, and
computes the shares hash and versioned vote commitment. The decision stays
private; the auxiliary proof is on the post-decision critical path, so this
variant offers no latency win over the unified proof — its value is removing
the exhaustive-hashing penalty while keeping the cast proof ECC-free and
compatible with batch-cast and delegation-merge futures.

M=16 layout sweep (auxiliary + cast bundle):

| Layout | ECC / hash tracks | Aux rows / K / columns | Aux prove | Bundle seq. | Bundle verify | Bundle proof | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 4 / 4 | 14,773 / 14 / 172 | 1,061 ms | 1,149 ms | 8.34 ms | 26.7 KiB | 1.64 GiB |
| wide | 8 / 4 | 7,387 / 13 / 293 | 873 ms | 973 ms | 6.04 ms | 36.5 KiB | 1.44 GiB |
| maxpar | 32 / 17 | 1,854 / 11 / 1,164 | 1,033 ms | 1,148 ms | 8.78 ms | 108.0 KiB | 1.39 GiB |

Against the matched unified `M = 16` result (905 ms, 5.65 ms verify,
96.3 KiB), the `wide` bundle is 1.075x the proving latency at 0.38x the
payload; the exhaustive bundle for comparison costs ~3,362 ms and 5.73 GiB.
Running the two proofs concurrently is no faster than sequential at this
size (they contend for the same worker pool).

`wide` was the best measured point of the benchmark. **Production selected
`maxpar`** so that every vote circuit shares K=11 SRS parameters and peak
proving memory is lowest, accepting the ~100 KiB auxiliary proof; the
production circuit reproduces the benchmark shape exactly (1,854 rows,
1,164 columns, 97.1 KiB measured proof).

### Soundness obligations (carried into production)

The split is a standard commit-and-prove composition. Its obligations, all
reflected in the production integration and its adversarial tests:

- **Bridge binding and hiding**: registered domain tags
  (`crate::domain_tags`), one shared formula module (`crate::bridge`)
  consumed by builders, all three circuits, and tests; `blind_i` provides
  hiding, both coordinates of every point are bound (sign-malleability).
- **Same `w_i` in both proofs**: the cast proof recomputes the bridge from
  the same share cells that sum to `total_note_value`; the 30-bit range is
  proven in ZKP 1.5 (fixed-base decomposition) and re-checked at cast.
- **EA key binding**: ZKP 1.5 pins both `ea_pk` coordinates to its instance;
  the verifier authenticates them against governance session data.
- **Replay confinement**: round, proposal, and `D` are folded into the
  bridge preimage and exposed on both instances; the verifier checks
  cross-instance equality (`verify_vote_bundle`).
- **Randomizer discipline**: non-zero `r` in-circuit, complete addition for
  `S`, deterministic builder-side PRF derivation per
  `(sk, round, proposal, VAN, share, bucket)`.
- **Selection integrity**: boolean prefix-confined selectors, C2 selection
  between fully constrained candidates.
- **Blast radius**: shared-state underconstraints affect all buckets at
  once; the seam received the focused adversarial test suite (altered
  commitments/coordinates, bridge mismatch, non-one-hot and inactive
  selectors, altered/reordered shares, zero randomness, wrong EA keys,
  cross-round/proposal replay, mixed bundles) required by the adoption
  gates.

Accepted costs: total compute increases (the chain verifies two proofs per
vote and carries the auxiliary proof bytes), plus a new circuit, verifying
key, wire message, and versioning surface requiring its own cryptography
review.
