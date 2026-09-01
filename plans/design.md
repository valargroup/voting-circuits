# Private Vote Choice Design

## Status and scope

This document specifies a private-choice protocol for weighted voting. It
covers the proof statements and their composition, the data that crosses each
boundary, the security assumptions, and the obligations that remain outside
the zero-knowledge proofs.

The design hides which option a voter selected while preserving weighted
voting. It replaces the plaintext choice formerly carried by the cast proof
with a one-hot vector of ElGamal ciphertexts. Encryption is proved in a new
auxiliary circuit, ZKP 1.5 (`encrypt-choice`). The existing cast proof, ZKP 2,
proves that the encrypted weights are authorized, and ZKP 3 opens one
encrypted share at a time for tallying.

The privacy claim is relative to ordinary chain observers, reveal helpers
without the election secret key, and fewer than the required election-key
holders in a threshold deployment. These circuits do not prevent a party that
possesses the complete election secret key from decrypting an individual
published ciphertext. Aggregate-only decryption therefore depends on the
wider key-management and decryption protocol, not on these proofs.

## Parameters and terminology

Two fixed dimensions must be kept separate:

- A vote has `N = 16` weight-share slots. This dimension comes from the
  amount-privacy design.
- Each share has `M = 8` decision buckets. This is the protocol ceiling.

A proposal declares a public active bucket count `D`. The protocol accepts
`2 <= D <= 8`, and valid choices are zero-based indices in `[0, D)`. Every
share carries eight ciphertexts regardless of `D`, so one vote commits to
`16 * 8 = 128` ciphertexts.

For example, a voter with weight 7 choosing the middle option of a
three-option proposal conceptually produces:

```text
[Enc(0), Enc(7), Enc(0)]
```

The actual construction does this separately for all 16 share slots. It uses
one selector vector for the whole vote, so different shares cannot be sent to
different options.

The standard construction decomposes the authorized ballot weight into 16
shares. An optional last-moment layout places the weight in one slot and zero
in the others. That fallback uses a separate randomness domain and fits the
same proof relation, but it gives up the amount-privacy benefit of the
standard decomposition if individual shares are ever decrypted.

## Vote lifecycle

1. The wallet obtains the authenticated voting round, proposal, active option
   count `D`, and election-authority public key. It also determines the Vote
   Authority Note (VAN) that this vote will consume.
2. Once the voter selects an option, the wallet constructs and proves ZKP 1.5
   locally. It may cache the proof while the voter reviews the transaction.
3. At submission time, the wallet produces ZKP 2 against a current,
   acceptable vote-tree anchor. ZKP 2 consumes the VAN and binds its
   authorized shares to the ciphertext commitments already proved by
   ZKP 1.5.
4. The public vote message carries both proofs and their public instances.
   ZKP 1.5 is not published earlier: doing so would disclose that someone is
   preparing a vote for a particular round and proposal.
5. The wallet retains, or can deterministically reconstruct, the private
   shares, commitment blinds, ciphertexts, and other data needed to build
   ZKP 3 proofs. This is private prover state and must not be included in the
   public vote message.
6. Later, ZKP 3 reveals one share's eight ciphertexts without publishing the
   parent vote commitment or the share index. After verification and
   duplicate-nullifier rejection, the chain adds the active ciphertexts to
   the proposal's bucket accumulators.
7. After the reveal period, the election authorities decrypt aggregate bucket
   ciphertexts and recover the bounded integer totals.

The proof flow is:

```text
private choice + VAN context
            |
            v
 ZKP 1.5: valid one-hot encryptions
            |
       public bridge
            |
            v
 ZKP 2: authorized shares + vote commitment
            |
       vote commitment tree
            |
            v
 ZKP 3: one committed ciphertext vector
            |
   nullifier check and bucket aggregation
```

## Weighted one-hot ElGamal encryption

Let `w_i` be share `i`, let `b_j` select bucket `j`, and let `PK = [x]G` be
the election-authority public key. The selector satisfies:

```text
b_j in {0, 1}
sum_j b_j = 1
b_j = 0 for j >= D
```

For every share `i` and bucket `j`, the wallet derives a distinct nonzero
randomizer `r_i,j` and constructs:

```text
C1_i,j = [r_i,j]G
C2_i,j = [r_i,j]PK + [b_j * w_i]G
```

The selected bucket encrypts the share weight; every other bucket encrypts
zero. The randomizers are deterministic but pseudorandom to anyone who does
not know the spending key. Distinct domains and share/bucket indices prevent
reuse across VANs, rounds, proposals, layouts, shares, and buckets.

The circuit avoids recomputing `[w_i]G` for every bucket. It computes:

```text
W_i   = [w_i]G
R_i,j = [r_i,j]PK
S_i,j = R_i,j + W_i
C2_i,j = select(b_j, R_i,j, S_i,j)
```

`S_i,j` uses complete curve addition, and the coordinate selection is
constrained by the boolean selector. The same fixed-base decomposition that
computes `W_i` proves `0 <= w_i < 2^30`. This bound keeps the base-field
value used by the share-sum constraint equal to the scalar used for curve
multiplication.

The active-prefix constraints make `D` the number of active buckets and force
the selector into that prefix. They structurally permit `D = 1`; protocol
verifiers must independently enforce the governance rule `D >= 2`. Proof
verification alone is not that check.

## Deterministic secrets and cached proofs

Vote secrets are derived with a Blake2b-512 PRF keyed by the spending key and
separated by registered domains. The common context is:

```text
(domain, voting_round_id, proposal_id, VAN, share_index)
```

Encryption randomness additionally includes `bucket_index`. Commitment
blinds do not. Exact-zero encryption randomness is deterministically remapped
to one, and the circuit independently rejects a zero randomizer.

The PRF derivation and the standard denomination split are prover policies,
not proof constraints. ZKP 1.5 proves that its witnessed randomizers are
nonzero and that the resulting encryption equations hold; it does not prove
that the randomizers, blinds, or share layout came from Blake2b. Proof
construction is randomized even though the reconstructed witnesses and public
instance are deterministic.

This construction lets the wallet restore a cached ZKP 1.5 proof without
persisting its private witnesses. It reconstructs the shares, ciphertexts,
commitments, bridge, and public instance from the current context, then
verifies the cached proof against that reconstruction. A cache entry must be
discarded when the choice, weight or share layout, VAN, round, proposal, `D`,
or election-authority key changes.

The decision, `D`, and election-authority key are not part of the randomness
PRF. Two choice revisions for the same core context therefore reuse the same
per-bucket randomizers. Only the final revision may leave the wallet. If
ciphertext vectors from two choice revisions were published, their differences
would reveal which buckets changed. Changes to `D` or the authority key also
invalidate the cached proof and can leave repeated `C1` values if both versions
escape. Supporting multiple published revisions would require a revision
identifier in the PRF context or another explicit nonce policy.

## Selected commitments and the bridge

Each share's complete ciphertext vector is hidden behind a blinded,
domain-separated Poseidon commitment. The canonical preimage order for each
bucket is `C1.x, C2.x, C1.y, C2.y`:

```text
selected_comm_i = Poseidon(
    WEIGHTED_SHARE_COMMITMENT_DOMAIN,
    blind_i,
    C1_i,0.x, C2_i,0.x, C1_i,0.y, C2_i,0.y,
    ...
    C1_i,7.x, C2_i,7.x, C1_i,7.y, C2_i,7.y
)
```

This is a 34-element preimage. Binding both coordinates prevents a prover
from changing point signs without changing the commitment.

The public bridge binds all 16 weights and commitments to the voting context:

```text
bridge = Poseidon(
    ENCRYPT_CHOICE_BRIDGE_DOMAIN,
    voting_round_id,
    proposal_id,
    D,
    w_0, selected_comm_0,
    ...
    w_15, selected_comm_15
)
```

This is a 36-element preimage. These commitment and bridge definitions are
normative; proof producers and verifiers must use the exact domain tags and
preimage order shown here.

ZKP 1.5 computes the bridge from the weights it encrypted. ZKP 2 recomputes
the same bridge from the exact share cells that are range-checked and summed
to the ballot weight authorized by the VAN. Equality of the public bridge
values therefore joins the statements:

```text
ZKP 1.5: these bounded weights were encrypted under one valid private choice
ZKP 2:   these same weights are authorized by this VAN
```

ZKP 1.5 alone does not prove that the weights are authorized. That conclusion
belongs to the composed vote bundle.

## ZKP 1.5: encrypt-choice

ZKP 1.5 has six public inputs:

```text
(ea_pk_x, ea_pk_y, bridge, D, voting_round_id, proposal_id)
```

Its private witnesses include the 16 weights, one-hot selector, active-prefix
flags, 128 randomizers, ciphertext coordinates, and 16 commitment blinds. It
proves:

1. The selector is boolean and one-hot.
2. The selected bucket is in the active prefix.
3. The election key is a non-identity point and each randomizer is nonzero.
4. All 128 `C1` and `C2` values satisfy the ElGamal equations.
5. Every weight is below `2^30`.
6. The 16 selected commitments and the public bridge have the canonical
   preimages above.

Both election-key coordinates are public and constrained. The verifier must
still authenticate that point as the election key announced for the round;
the proof only establishes correctness relative to the supplied key.

## ZKP 2: compact cast proof

ZKP 2 retains the existing VAN membership, VAN integrity, address ownership,
spend authority, VAN nullifier, proposal-authority decrement, new-VAN
integrity, share-sum, and share-range conditions. It still performs
elliptic-curve work for address ownership and spend authority. What moved to
ZKP 1.5 is the ElGamal work for vote ciphertexts.

The old plaintext decision and election-authority key are absent. Their place
is taken by three hash operations:

1. Recompute the bridge from the authorized share cells and the 16 witnessed
   selected commitments.
2. Compute:

   ```text
   shares_hash = Poseidon(
       selected_comm_0, ..., selected_comm_15
   )
   ```

3. Compute the versioned vote commitment:

   ```text
   vote_commitment = Poseidon(
       DOMAIN_VC_V2,
       voting_round_id,
       shares_hash,
       proposal_id,
       D
   )
   ```

The choice is bound transitively through the committed ciphertext vectors;
it is never a ZKP 2 witness or public input.

The cast instance remains 11 field elements. Its first nine slots retain
their previous roles; the final two are now `bridge` and `D`. In particular,
the instance includes the VAN nullifier, randomized voting key, new VAN, vote
commitment, vote-tree root and anchor height, proposal, and round. The anchor
height is transcript-bound metadata but is not derived from a circuit
witness. The verifier must check that the supplied root is the accepted chain
root at that height.

Bundle verification verifies both proofs and checks equality of:

```text
bridge
voting_round_id
proposal_id
D
```

It does not authenticate governance or chain provenance.

## ZKP 3: share reveal

ZKP 3 publishes one share's complete eight-bucket ciphertext vector. Its 37
public inputs are:

```text
share_nullifier
32 ciphertext coordinates
proposal_id
vote_comm_tree_root
voting_round_id
D
```

The vote commitment, all 16 selected commitments, the selected share index,
the selected commitment, and the commitment blind remain private. The circuit
proves the following chain:

```text
public ciphertext coordinates
        |
selected_comm_i = Poseidon(domain, blind_i, coordinates)
        |
shares_hash = Poseidon(selected_comm_0, ..., selected_comm_15)
        |
vote_commitment = Poseidon(DOMAIN_VC_V2, round, shares_hash, proposal, D)
        |
private Merkle path
        |
public vote_comm_tree_root
```

It also exposes:

```text
share_nullifier = Poseidon(
    SHARE_SPEND_DOMAIN,
    vote_commitment,
    share_index,
    blind_i
)
```

The chain must reject a repeated share nullifier. That stateful uniqueness
check, rather than the circuit alone, prevents a share from being counted
twice. The blind hides the parent vote and share index from observers under
the Poseidon assumptions.

ZKP 3 does not repeat the ElGamal constraints and does not require the helper
to know the choice, weight, or encryption randomness. It does require the
full selected-commitment array, the chosen share's blind and ciphertexts, and
a valid vote-tree path. How those private inputs are stored and delivered to
a reveal helper is an operational decision outside the proof statement. A
remote helper with those inputs can recompute and correlate the parent vote
commitment even though the resulting public proof does not reveal it. Public
selective opening must therefore not be confused with unlinkability from the
machine generating the proof.

## Verifier and tally obligations

Before accepting a vote bundle, the protocol verifier must:

1. Authenticate the voting round, active proposal, `D`, and both coordinates
   of the election-authority key from governance state.
2. Enforce `2 <= D <= 8`.
3. Authenticate the cast proof's vote-tree root at its anchor height and
   apply the chain's anchor-validity policy.
4. Verify both proofs.
5. Check equality of the bridge, round, proposal, and `D` across their
   instances.
6. Reject a repeated VAN nullifier and verify the vote signature under the
   proof-attested randomized voting key.
7. Insert the resulting vote commitment and new VAN according to the chain
   transition rules.

Before accepting a share reveal, it must:

1. Authenticate the round, proposal, `D`, and vote-tree root from the
   applicable chain and governance state.
2. Verify ZKP 3.
3. Reject a repeated share nullifier.
4. Add the ciphertexts for buckets in `[0, D)` to the corresponding
   `(round, proposal, bucket)` accumulators. The remaining buckets are proved
   encryptions of zero but need not be accumulated.

These circuits do not implement reveal scheduling, completeness, or recovery
when a share is omitted. They also do not implement ciphertext accumulation,
threshold decryption, proofs of correct decryption such as DLEQ proofs, or
post-decryption weight-conservation checks. Those are required protocol
components. In particular, nullifiers prevent duplicates but do not ensure
that all 16 shares of every accepted vote are revealed before a deadline.

## Security properties and limits

Subject to the proof-system, Poseidon, PRF, and ElGamal assumptions, the
construction provides:

- **Choice validity:** one selector is shared across every share, and it
  selects one active bucket.
- **Weight authorization:** the public bridge joins the encrypted weights to
  the exact shares authorized by the consumed VAN.
- **Ciphertext binding:** commitment blinds make the selected commitments
  hiding, and both coordinates of every ciphertext point are committed.
- **Replay confinement:** round, proposal, and `D` appear in the bridge and
  vote commitment and are checked across the proof instances.
- **Deterministic recovery:** a wallet can reconstruct the auxiliary
  witnesses and validate a cached proof from the spending key and vote
  context.
- **Selective opening:** a share reveal proves membership in a registered
  vote without publishing the parent commitment or share index.

The construction does not provide:

- Privacy from a holder or colluding threshold of the election secret key if
  that party decrypts individual revealed ciphertexts.
- Protection against metadata that links proof generation, submission, or
  reveal traffic to a wallet.
- Reveal liveness or proof that all accepted weight was included in the final
  accumulators.
- Safe publication of multiple choice revisions under the current PRF
  context.
- A complete identity-encoding policy. The group relation can produce an
  identity `C2` in the negligible case `R_i,j = -W_i`. Proofs, wire formats,
  and tally code must either support one canonical identity encoding or
  consistently reject this case.
- An uncoordinated increase beyond eight buckets. Changing `M` changes
  circuit shapes, public reveal layouts, verifying keys, clients, and chain
  validation, and requires a versioned rollout.

The v2 vote-commitment domain and changed verification keys likewise require
coordinated deployment. A verifier must not infer compatibility from the
unchanged ZKP numbering.

## Performance and rationale

An `M = 8` configuration was benchmarked on an Apple M4 Max with eight
proving threads in release mode. Keys were generated once; proving was warmed
before five samples. Verification used ten encrypt-choice/bundle samples and
five share-reveal samples. The recorded run reported:

| Circuit | Key generation | Proving median | Verification median | Proof bytes |
| --- | ---: | ---: | ---: | ---: |
| ZKP 1.5 encrypt-choice, K=11 | 734 ms | 531 ms | 3.8 ms | 57.7 KiB |
| ZKP 2 cast, K=11 | 96 ms | 81 ms | included below | 7.1 KiB |
| Vote bundle | — | 611 ms sequential | 5.3 ms | 64.7 KiB |
| ZKP 3 share reveal, K=10 | 63 ms | 38 ms | 1.1 ms | 6.0 KiB |

Peak resident memory for the three-circuit benchmark process was
approximately 0.9 GiB. These byte counts cover serialized Halo2 proofs only.
They do not include public instances, signatures, transaction framing, or
other wire data.

The split does not reduce total computation. Its operational benefit is that
the 531 ms encrypt-choice proof can begin after option selection and overlap
the voter's review time, leaving an approximately 81 ms cast proof once the
cached proof is ready. If the voter submits immediately, ZKP 1.5 remains on
the choice-to-submission critical path. Changing the choice also requires a
new auxiliary proof.

The cost is a second proof verification, about 65 KiB of proof bytes, another
circuit and verification key, retained reveal state, and a larger integration
and review surface. Historical `M = 16` experiments and alternative layouts
are not measurements of this eight-bucket design.
