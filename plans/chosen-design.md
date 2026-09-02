# Chosen Design: Private Decisions via a Decision-Bound Proof Split

## Summary

Vote decisions are now private. Instead of committing to a plaintext
`vote_decision`, each vote encrypts its weight against a hidden one-hot
selector over `M = 8` decision buckets: for every one of the 16 weight
shares, the selected bucket encrypts the share's weight and the other buckets
encrypt zero, all under the election authority's ElGamal key with independent
randomness. Only aggregate per-bucket tallies are ever decrypted.

Two fixed protocol parameters appear throughout and must not be confused:
**16 shares** (`NUM_SHARES` — the per-vote weight decomposition, kept from
the original design for amount privacy) and **8 buckets**
(`MAX_DECISION_BUCKETS` — the vote-option ceiling). Every share is encrypted
into every bucket, so a vote carries 16 × 8 = 128 ciphertexts.

To keep this off the interactive critical path, the ElGamal work was split
out of ZKP #2 into a new auxiliary proof, **ZKP 1.5 "encrypt-choice"**
(`src/encrypt_choice/`), which the wallet starts in the background the
moment the voter selects a choice. ZKP #2 becomes a compact, ECC-free cast
circuit. The two proofs are submitted together as one `VoteBundle`, glued by
a shared public **bridge** commitment; ZKP #3 reveals a full per-share
ciphertext vector without repeating any ElGamal checks.

Each proposal publicly declares its active option count `D` (2..=8, the
vote-sdk proposal-option limit); inactive buckets are proof-bound
encryptions of zero, so one circuit shape and verifying key covers every
option count.

## The bridge (commit-and-prove seam)

The intuition: ZKP 1.5 proves "these ciphertexts correctly encrypt weights
`w_0..w_15` under a valid one-hot decision," and ZKP #2 proves "these same
weights are the shares my VAN authorizes." The two statements are joined by
one Poseidon value both proofs expose publicly:

```text
selected_comm_i = Poseidon(domain, blind_i,
                           all 8 bucket ciphertext coordinates of share i)
bridge          = Poseidon(domain, round, proposal, D,
                           w_0, selected_comm_0, ..., w_15, selected_comm_15)
```

Because the cast proof recomputes the bridge from the very share cells that
sum to `total_note_value`, the pre-encrypted weights and the authorized
weights are provably identical. Folding round, proposal, and `D` into the
preimage prevents replaying an encrypt-choice proof under a different voting
context. `src/bridge.rs` is the single native + in-circuit source of these
formulas.

## ZKP 1.5 — encrypt-choice (new)

Proves, in one background proof (K=11, fully parallel layout: 16 ECC / 17
Poseidon tracks):

1. **One-hot decision**: boolean selectors summing to one.
2. **Active-bucket confinement**: boolean prefix flags summing to the public
   `D`, with the selector zero in inactive buckets — so the private decision
   is in `0..D` without changing circuit shape.
3. **Encryption integrity** for all 16×8 ciphertexts: `r != 0`,
   `C1 = [r]G`, `R = [r]PK`, `S = R + W_i` (complete addition), with the
   shared weight point `W_i = [w_i]G` computed once per share (this also
   range-checks `w_i < 2^30`). The selector then chooses the published C2:
   `C2 = R + b_j·(S − R)` — the selected bucket encrypts the weight, all
   others encrypt zero.
4. The 16 selected commitments and the bridge.

Instance: `[ea_pk_x, ea_pk_y, bridge, D, round, proposal]` — the EA key is
authenticated here, not in ZKP #2. All randomness and blinds are derived
from a deterministic PRF keyed by `(sk, round, proposal, VAN, share,
bucket)`, so the proof is idempotent after a crash and the cast builder can
independently re-derive and cross-check the same shares.

## ZKP #2 — compact cast circuit (refactored in place)

Conditions 1–9 (VAN membership/integrity, address ownership, spend
authority, nullifier, authority decrement, shares sum, shares range) are
unchanged. The old conditions 10–12 — the 16 in-circuit ElGamal encryptions
and the decision-bearing commitment — are replaced by three cheap hash
steps:

- **10′ Bridge re-opening**: witness the 16 selected commitments and
  recompute the bridge over them and the condition-8 share cells; bind it to
  the instance.
- **11′ Shares hash**: `Poseidon<16>` over the selected commitments.
- **12′ Vote commitment v2**:
  `Poseidon(DOMAIN_VC_V2, round, shares_hash, proposal, D)` — the plaintext
  decision slot is gone; the decision is bound only through the committed
  ciphertext vectors inside `shares_hash`, and binding `D` prevents replay
  under a proposal with a different option count.

The instance keeps its 11 slots with offsets 0–8 unchanged; `ea_pk_x/y`
became `bridge` and `D`. With the ECC tracks removed the circuit stays at
K=11 with its high-water mark down from 2,015 to 1,662 rows — the
interactive proof contains no ElGamal ECC at all.

A verifier accepts a vote by verifying both proofs and checking bridge,
round, proposal, and `D` are equal across the two instances
(`vote_proof::verify_vote_bundle`), then authenticating the
governance-sourced fields as before.

## ZKP #3 — share reveal (refactored in place)

Reveals all 8 bucket ciphertexts of one share (the chain adds the `D`
active ones to its encrypted tallies), instead of one ciphertext plus the
plaintext decision:

- The instance grows to 37 elements: nullifier, 32 ciphertext coordinates,
  proposal, tree root, round, `D`. `vote_decision` is removed everywhere.
- Condition 4 recomputes the 34-input selected commitment over the revealed
  vector and the private blind, and matches it against the muxed private
  `share_comms[share_index]` — the same commitment shape ZKP 1.5 produced,
  so no ElGamal checks are repeated at reveal time and the helper that
  builds the proof never learns the decision, weight, or randomness.
- Conditions 1/3/5 (Merkle membership, shares hash, blind-keyed share
  nullifier) are unchanged in structure; condition 2 uses the v2 vote
  commitment.

The wide hash runs on a dedicated Poseidon track, keeping the circuit at
K=10 (855 of 1,024 rows); the proof stays well within the 15 KiB downstream
limit.

## Cost and rationale

The split does not reduce total computation — it moves it. Measured with the
full production circuits at `M = 8` (M4 Max, 8 proving threads, release,
warm keys):

| | Keygen | Prove (median) | Verify | Proof |
| --- | ---: | ---: | ---: | ---: |
| ZKP 1.5 encrypt-choice (K=11) | 734 ms | 531 ms | 3.8 ms | 57.7 KiB |
| ZKP #2 cast (K=11) | 96 ms | 81 ms | — | 7.1 KiB |
| Vote bundle (both + binding checks) | — | 611 ms seq. | 5.3 ms | 64.7 KiB |
| ZKP #3 share reveal (K=10) | 63 ms | 38 ms | 1.1 ms | 6.0 KiB |

Peak RSS for the whole three-circuit process is ~0.9 GiB. The interactive
cast proof is a complete production ZKP #2 (VAN membership, ownership,
authority decrement included) at 81 ms; the encrypt-choice proof runs in the
background between decision selection and submission. `M = 8` matches the
vote-sdk proposal-option limit; raising it later is a coordinated
circuit/VK upgrade. The price of the split is a second proof to verify, a
two-proof payload, and one more circuit and verifying key. Re-run the
numbers with `RAYON_NUM_THREADS=8 /usr/bin/time -l cargo run --release
--example vote_bench`. Alternatives considered and rejected are recorded in
`encrypted-votes-alternative-designs.md`.
