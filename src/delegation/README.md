# Delegation Circuit (ZKP 1)

A single circuit proving all 15 conditions of the delegation ZKP at K=12 (4,096 rows). The circuit handles the keystone note (conditions 1–8) and five per-note slots (conditions 9–15 ×5) in one proof.

The five Orchard Merkle paths distribute their levels across four independent
five-advice-column Sinsemilla lanes. The five IMT paths rotate their Poseidon
levels across those same lanes. This balanced layout uses 30 advice columns.

**Public inputs:** 14 field elements.
**Per-note slots:** 5 (`MAX_REAL_NOTES`; unused slots are padded with
zero-value notes). The fixed width is a protocol parameter: it gives every
delegation the same published `gov_null_1..5` shape, supports wallets with up
to five real notes per proof, and keeps the circuit within K=12. Wallets with
more than five notes produce multiple delegation proofs.

**Note value asymmetry:** the builder gives the keystone (signed) note value `1`
zatoshi, and the circuit requires its value to be nonzero, so Keystone-class
hardware wallets render the spend for user approval. The output (change) note
has value `0`. See conditions 1 and 6.

**Authoritative hash sources:** this README is explanatory. The in-tree source
of truth for reusable hash preimages is the owning module:
`crate::circuit::van_integrity` for `van_comm`, `crate::domain_tags` for
domain-tag encoding, `rho_binding_hash` in `circuit.rs`, and `gov_null_hash`
in `imt.rs` for delegation-only hashes.

## Inputs

- Public (14 field elements)
   * **nf_signed** (offset 0): the derived nullifier of the keystone note.
   * **rk** (offsets 1–2): the randomized public key for spend authorization (x, y coordinates).
   * **cmx_new** (offset 3): the extracted note commitment (`ExtractP(cm_new)`) of the output note.
   * **van_comm** (offset 4): the governance commitment — a Pallas base field element identifying the governance context.
   * **vote_round_id** (offset 5): the vote round identifier — prevents cross-round replay.
   * **nc_root** (offset 6): ledger-state anchor for the Ironwood note commitment tree at the verifier-pinned snapshot height; real-note Merkle paths must resolve to this root.
   * **nf_imt_root** (offset 7): ledger-state anchor for the alternate-nullifier Indexed Merkle Tree at the same snapshot height as `nc_root`; non-membership proofs must resolve to this root.
   * **gov_null_1..5** (offsets 8–12): per-note alternate nullifiers, one per note slot.
   * **dom** (offset 13): the nullifier domain — Poseidon("governance authorization", vote_round_id). Exposed as a public input for API compatibility; the circuit constrains it against vote_round_id rather than trusting an arbitrary value.

- Private (keystone note)
   * **rho_signed** ("rho"): the nullifier of the note that was spent to create the signed note.
   * **psi_signed** ("psi"): a pseudorandom field element derived from the note's `rseed` and rho.
   * **cm_signed**: the note commitment, witnessed as an ECC point.
   * **nk**: nullifier deriving key.
   * **ak**: spend validating key (the long-lived public key for spend authorization).
   * **alpha**: a fresh random scalar used to rerandomize the spend authorization key for each transaction.
   * **rivk**: the randomness (blinding factor) for the CommitIvk Sinsemilla commitment (external scope).
   * **rivk_internal**: the randomness (blinding factor) for the CommitIvk Sinsemilla commitment (internal/change scope).
   * **rcm_signed**: the note commitment trapdoor (randomness).
   * **g_d_signed**: the diversified generator from the note recipient's address.
   * **pk_d_signed**: the diversified transmission key from the note recipient's address.

- Private (output note — condition 6)
   * **g_d_new**: the diversified generator from the output note recipient's address.
   * **pk_d_new**: the diversified transmission key from the output note recipient's address.
   * **psi_new**: pseudorandom field element for the output note.
   * **rcm_new**: the output note commitment trapdoor.

- Private (per-note slot ×5 — conditions 9–15)
   * **g_d**: diversified generator from the note recipient's address.
   * **pk_d**: diversified transmission key from the note recipient's address.
   * **v**: the note value (in zatoshi).
   * **rho**: the nullifier of the note that was spent to create this note.
   * **psi**: pseudorandom field element derived from the note's `rseed` and rho.
   * **rcm**: note commitment trapdoor.
   * **cm**: note commitment, witnessed as an ECC point.
   * **path**: Sinsemilla-based Merkle authentication path (32 siblings).
   * **pos**: leaf position in the note commitment tree.
   * **is_internal**: boolean flag — 1 for internal (change) scope notes, 0 for external scope notes.
   * **imt_nf_bounds**: three sorted nullifier boundaries `[nf_lo, nf_mid, nf_hi]` from the K=2 punctured-range leaf.
   * **imt_leaf_pos**: position of the leaf in the IMT.
   * **imt_path**: Poseidon-based IMT Merkle authentication path (29 pure siblings).

- Private (governance — condition 7)
   * **van_comm_rand**: a random blinding factor for the governance commitment.

- Private (ballot scaling — condition 8)
   * **num_ballots**: the ballot count — honestly `floor(v_total / 12,500,000)` — witnessed as a free advice. The circuit constrains this approximately (see §8 "Soundness scope").
   * **remainder**: honestly `v_total mod 12,500,000`, witnessed as a free advice. Range-checked to `[0, 2^24)` rather than `[0, BALLOT_DIVISOR)` (see §8 "Soundness scope").

- Internal wires (not public inputs, not free witnesses)
   * **ivk**: derived in condition 5 (CommitIvk with external `rivk`), shared with condition 11.
   * **ivk_internal**: derived in condition 5 (CommitIvk with internal `rivk_internal`), shared with condition 11.
   * **cmx_1..5**: produced by per-note condition 9, consumed by condition 3.
   * **v_1..5**: produced by per-note condition 9, consumed by conditions 7 and 8.

## Public Input Provenance

The circuit proves statements relative to the public inputs supplied by the
verifier; it does not authenticate where those inputs came from.

**Ledger-state anchors:** `nc_root` and `nf_imt_root` must come from the
chain's state at the same verifier-pinned snapshot height. `nc_root` is the
Ironwood note commitment tree root used by condition 10. `nf_imt_root` is the
alternate-nullifier IMT root used by condition 13. A prover bundle may carry
copies of these values for convenience, but a verifier must not trust the
bundle as their source of truth.

The high-level bundle builder rejects non-V3 notes, but the circuit does not
carry a note-version bit or derive `psi` and `rcm` from `rseed`. It proves that
the witnessed values open commitments contained in the supplied tree. Pool
identity therefore comes from the verifier-authenticated `nc_root`; accepting
an Orchard root would also accept commitments from the Orchard pool.

**Session parameters:** `vote_round_id` is pinned by the governance session.
`dom` is exposed for API compatibility but constrained in-circuit to
`Poseidon("governance authorization", vote_round_id)`. `van_comm` is
proof-attested by condition 7 and then consumed by the voting flow.

## Integration: the Keystone (signed) note is synthetic

The keystone (signed) note **does not exist on any chain**. It is a
synthetic Ironwood spend shape locally constructed by the voting client
so that:

1. The Keystone hardware wallet — which signs Orchard protocol Actions,
   including Ironwood Actions —
   can be coerced into producing a spend-auth signature under `rk` over
   the wrapping Ironwood Action's ZIP-244 sighash. The voting protocol
   piggybacks on the shared Orchard protocol wallet UX surface rather than
   adding a new signing protocol.
2. The wallet's UI renders the builder's `v_signed = 1` zatoshi for user
   approval. Zero-value spends are not rendered, so the circuit requires
   `v_signed != 0` (see condition 1 below).
3. The proof's public-input layout `(nf_signed, rk, cmx_new, ...)`
   matches what the wrapping Ironwood Action expects.

Unlike each of the five real-note slots, which carry a Sinsemilla
Merkle membership proof against the public `nc_root` anchor (gated by
`v * (root - nc_root) = 0` so that `v = 0` padding slots can skip the
check), the keystone branch witnesses **no Merkle path** and performs
**no anchor check**. `cm_signed` is recomputed from witnessed
`(g_d_signed, pk_d_signed, v_signed, rho_signed, psi_signed, rcm_signed)`
and constrained equal to a separately-witnessed `cm_signed` ECC point;
this is a "the prover knows the opening" check, not a membership check.
The voter does not own a 1-zatoshi keystone note that the proof spends.

The keystone branch's in-circuit conditions therefore sort into three
buckets, useful for auditing future refactors:

- **Protocol-relevant.** Condition 5 (the prover knows `(ak, nk, rivk)`
  consistent with the real-note slots' `ivk`); condition 4
  (`rk = [α]·SpendAuthG + ak`).
- **Wallet-authorization binding (load-bearing as a chain).** Condition
  1 requires `v_signed != 0`, ensuring the wallet renders the spend.
  Condition 3 (`rho_signed = Poseidon(cmx_1..5, van_comm, vote_round_id)`) and
  condition 2 (`nf_signed = DeriveNullifier(nk, rho_signed, psi_signed, cm_signed)`,
  exposed as a public input). Together with the verifier-enforced
  `proof.nf_signed == action.nullifier` check, this is the chain by
  which the wallet's spend-auth signature cryptographically commits to
  `van_comm` and `vote_round_id`. Remove any link in this chain and the
  binding breaks.
- **Ironwood Action shape mimicry.** Condition 1 (`cm_signed`
  well-formed under the keystone diversified address) and condition 6
  (`cmx_new` derivation from the synthetic output note). These exist to
  make the wrapping Action's public-input layout look like a standard
  Ironwood spend; the voting protocol does not consume `cm_signed` or
  `cmx_new` for any of its own invariants.

## 1. Signed Note Commitment Integrity

Purpose: ensure that the signed note commitment is correctly constructed. Establishes the binding link between spending authority, nullifier key, and the note itself.

```
v_signed != 0
NoteCommit_rcm_signed(repr(g_d_signed), repr(pk_d_signed), v_signed, rho_signed, psi_signed) = cm_signed
```

Where:
- **rcm_signed**: the note commitment randomness (trapdoor). A scalar derived from the note's `rseed` and `rho`. Blinds the commitment.
- **repr(g_d_signed)**: the diversified base point from the recipient's payment address.
- **repr(pk_d_signed)**: the diversified transmission key.
- **v_signed**: the keystone note value. The circuit requires it to be nonzero;
  the honest builder chooses 1 zatoshi as the minimum value. Hardware wallets
  such as Keystone do not render zero-value spends, so this constraint ensures
  the spend is surfaced for user approval. Contrast condition 6, which uses
  value 0 for the *output* (change) note; the asymmetry is intentional.
- **rho_signed**: the nullifier of the note that was spent to create this note. Bound by condition 3.
- **psi_signed**: pseudorandom field element from `rseed` and `rho`.
- **cm_signed**: the witnessed note commitment. The circuit recomputes NoteCommit and enforces strict equality.

The commitment binds together: **who the note belongs to** (g_d, pk_d), **how
much it's worth** (the nonzero `v_signed`), **where it came from** (rho),
**random uniqueness** (psi), **all blinded by randomness** (rcm).

**Constructions:** `SinsemillaChip` (config 1), `EccChip`, `NoteCommitChip` (signed).

## 2. Nullifier Integrity

Purpose: derive the standard Orchard protocol nullifier used by Ironwood deterministically from the note's secret components. Validate it against the one used in the exclusion proof.

```
nf_signed = DeriveNullifier_nk(rho_signed, psi_signed, cm_signed)
```

Where:
- **nk**: The nullifier deriving key associated with the note.
- **rho_signed** ("rho"): The nullifier of the note that was spent to create the signed note. A Pallas base field element that serves as a unique, per-note domain separator. rho ensures that even if two notes have identical contents, they will produce different nullifiers because they were created by spending different input notes.
- **psi_signed** ("psi"): A pseudorandom field element derived from the note's random seed `rseed` and its nullifier domain separator rho. Adds randomness to the nullifier so that even if two notes share the same rho and nk, they produce different nullifiers. Provided as a witness (not derived in-circuit) since derivation would require an expensive Blake2b.
- **cm_signed**: The note commitment, witnessed as an ECC point (the form `DeriveNullifier` expects).

**Function:** `DeriveNullifier`

```
DeriveNullifier_nk(rho, psi, cm) = ExtractP(
    [ (PRF_nf_Orchard_nk(rho) + psi) mod q_P ] * K_Orchard + cm
)
```

- `ExtractP` extracts the base field element from the resulting group element.
- `K_Orchard` is a fixed generator. Input to the `EccChip`.
- `PRF_nf_Orchard_nk(rho)` is the nullifier pseudorandom function. Uses Poseidon hash for PRF.

**Constructions:** `PoseidonChip`, `AddChip`, `EccChip`.

- **Why do we take PRF of rho?**
   * The primary reason is unlinkability. Rho is the nullifier of the note that was spent to create this note. In the standard Orchard protocol flow used by Ironwood, nullifiers are published onchain. The PRF destroys the link.
- **Why not expose nf_old publicly?**
   * In the standard Orchard protocol flow used by Ironwood, the nullifier is published to prevent double-spending. In this delegation circuit, nf_old is not directly exposed as a public input. Instead, it is checked against the exclusion interval and a domain nullifier is published instead. The standard nullifier stays hidden.

## 3. Rho Binding

Purpose: the signed note's rho is bound to the exact notes being delegated, the governance commitment, and the round. This makes the keystone signature non-replayable and scoped.

```
rho_signed = Poseidon(cmx_1, cmx_2, cmx_3, cmx_4, cmx_5, van_comm, vote_round_id)
```

Where:
- **cmx_1..5**: the extracted note commitments (`ExtractP(cm_i)`) of the five delegated notes. **These are internal wires** — produced by per-note condition 9 (note commitment integrity), not free witnesses. By hashing all five commitments into rho, the keystone signature is bound to the exact set of notes the delegator chose.
- **van_comm**: the governance commitment (public input).
- **vote_round_id**: the vote round identifier (public input).

**Function:** `Poseidon` with `ConstantLength<7>`. Uses `Pow5Chip` / `P128Pow5T3` with rate 2 (4 absorption rounds for 7 inputs).

**Constraint:** The circuit computes `derived_rho = Poseidon(cmx_1, cmx_2, cmx_3, cmx_4, cmx_5, van_comm, vote_round_id)` and enforces strict equality `derived_rho == rho_signed`. Since `rho_signed` is the same value used in both note commitment integrity (condition 1) and nullifier integrity (condition 2), this creates a three-way binding: the nullifier, the note commitment, and the delegation scope are all tied to the same rho.

**Constructions:** `PoseidonChip`.

## 4. Spend Authority

Purpose: proves spending authority while preserving unlinkability. Links to the Keystone spend-auth signature verified out-of-circuit.

```
rk = [alpha] * SpendAuthG + ak_P
```

Where:
- **ak** — the authorizing key, the long-lived public key for spend authorization.
- **alpha** — fresh randomness. If rk were the same across transactions, an observer could link them to the same spender.
- **SpendAuthG** — the fixed base generator point on the Pallas curve dedicated to spend authorization.

**Constructions:** Shared `circuit::spend_authority::prove_spend_authority` gadget (which computes fixed-base `[alpha]*SpendAuthG`, adds `ak_P`, and constrains `rk` to the instance), plus `EccChip` operations internally.

## 5. CommitIvk & Diversified Address Integrity

Purpose: proves the signed note's address belongs to the same key material `(ak, nk)`. Derives `ivk` (external) and `ivk_internal` — used by condition 11 for per-note ownership checks with scope selection.

```
ivk = CommitIvk_rivk(ExtractP(ak_P), nk)
ivk_internal = CommitIvk_rivk_internal(ExtractP(ak_P), nk)
pk_d_signed = [ivk] * g_d_signed
```

Without address integrity, the nullifier integrity proves "I know (nk, rho, psi, cm) that produce this nullifier" and "I know ak such that rk = ak + [alpha] * G", but nothing ties ak to nk. A malicious prover could supply their own `ak` and someone else's `nk`.

`CommitIvk(ExtractP(ak), nk)` forces `ak` and `nk` to come from the same key tree. `pk_d_signed = [ivk] * g_d_signed` proves the note's destination address was derived from this specific ivk.

The `ivk = ⊥` case is handled internally by `CommitDomain::short_commit`: incomplete addition allows the identity to occur, and synthesis detects this edge case and aborts proof creation. No explicit conditional is needed in the circuit.

**Both ivk values are internal wires** — they are NOT constrained to public inputs. They flow directly to condition 11 (per-note address checks) via cell reuse. `ivk_internal` is derived identically to `ivk` but uses `rivk_internal` (the change-scope blinding factor) instead of `rivk`.

Where:
- **ak_P** — the spend validating key (shared with condition 4). `ExtractP(ak_P)` extracts its x-coordinate.
- **nk** — the nullifier deriving key (shared with conditions 2, 12, 14).
- **rivk** — the CommitIvk randomness for external scope, extracted from the full viewing key via `fvk.rivk(Scope::External)`.
- **rivk_internal** — the CommitIvk randomness for internal (change) scope, extracted via `fvk.rivk(Scope::Internal)`.
- **g_d_signed** — the diversified generator from the note recipient's address.
- **pk_d_signed** — the diversified transmission key from the note recipient's address.

**Constructions:** Shared `circuit::address_ownership::prove_address_ownership` (CommitIvk + `[ivk]*g_d` + constrain to pk_d). Uses `CommitIvkChip`, `SinsemillaChip` (config 1), `EccChip` internally. A second `commit_ivk` call derives `ivk_internal`.

## 6. Output Note Commitment Integrity

Purpose: prove the output note's commitment is correctly constructed, with its `rho` chained from the signed note's nullifier. Creates a cryptographic link between spending the signed note and creating the output note.

```
ExtractP(NoteCommit_rcm_new(repr(g_d_new), repr(pk_d_new), 0, rho_new, psi_new)) in {cmx_new, bottom}
where rho_new = nf_signed mod q_P
```

Where:
- **rcm_new**: the output note commitment trapdoor.
- **repr(g_d_new)**: the diversified base point from the output note recipient's address.
- **repr(pk_d_new)**: the diversified transmission key from the output note recipient's address.
- **0**: the note value is hardcoded to zero.
- **rho_new**: set to `nf_signed` — the nullifier derived in condition 2. Enforced in-circuit by reusing the same cell.
- **psi_new**: pseudorandom field element derived from `rseed_new` and `rho_new`.
- **cmx_new**: the public input. `ExtractP` extracts the x-coordinate of the commitment point.

**Chain from condition 2**: The `nf_signed` cell computed in condition 2 is reused directly as `rho_new`. Since that cell is also constrained to the `NF_SIGNED_PUBLIC_OFFSET` public input, the chain is: `nf_signed` (public) = `DeriveNullifier(nk, rho_signed, psi_signed, cm_signed)` = `rho_new` (input to output NoteCommit).

**Constructions:** `SinsemillaChip` (config 2), `EccChip`, `NoteCommitChip` (new).

## 7. Gov Commitment Integrity

Purpose: prove that the governance commitment (a public input) is correctly derived from the domain tag, the output note's diversified address x-coordinates (`g_d_new_x`, `pk_d_new_x`), the ballot count, the vote round identifier, a blinding factor, and the proposal authority bitmask. Binds the delegated weight, output address coordinates, and authority scope into a single public commitment that the vote proof can open. The output address is hashed as data here; this circuit does not separately enforce output-address ownership against `ivk` in condition 6. The domain tag provides domain separation from Vote Commitments in the shared vote commitment tree.

```
van_comm_core = Poseidon(DOMAIN_VAN, g_d_new_x, pk_d_new_x, num_ballots, vote_round_id, MAX_PROPOSAL_AUTHORITY)
van_comm = Poseidon(van_comm_core, van_comm_rand)
```

Address encoding note: `g_d_new_x` and `pk_d_new_x` are x-coordinates only. Negating both output address points preserves this VAN hash, so `van_comm` is not a standalone commitment to unique full points. The current protocol still carries the needed full-point information because condition 6 feeds `g_d_new` and `pk_d_new` into NoteCommit's point encoding, which distinguishes the y-sign in `cmx_new`, and the vote proof spends the VAN through its full-point address ownership check and per-VAN nullifier.

Where:
- **DOMAIN_VAN**: `0`. Domain separation tag for Vote Authority Notes (vs `DOMAIN_VC = 1` for Vote Commitments). Assigned via `assign_constant` so the value is baked into the verification key.
- **g_d_new_x**: the x-coordinate of the output note's diversified generator. Reuses the ECC point from condition 6.
- **pk_d_new_x**: the x-coordinate of the output note's diversified transmission key. Reuses the ECC point from condition 6.
- **num_ballots**: the ballot count (honestly `floor(v_total / 12,500,000)`; the precise proven relation is defined by condition 8 — see §8 "Soundness scope"). `v_total` is the sum `v_1 + v_2 + v_3 + v_4 + v_5`, computed in-circuit via four `AddChip` additions. **Each `v_i` is an internal wire** — produced by per-note condition 9 (note commitment integrity), not a free witness. The value is bound to the actual note commitment.
- **vote_round_id**: the vote round identifier (public input, same cell as condition 3).
- **MAX_PROPOSAL_AUTHORITY**: `2^16 - 1 = 65535`. A 16-bit bitmask authorizing voting on all 16 proposals. Assigned via `assign_constant` so the value is baked into the verification key.
- **van_comm_rand**: a random blinding factor. Prevents observers from brute-forcing the address or weight from the public `van_comm`.

**Function layout:** Two Poseidon hashes via the shared `circuit::van_integrity` gadget:
- `van_comm_core` uses `ConstantLength<6>` over the structural fields.
- `van_comm` finalizes with `ConstantLength<2>` over `(van_comm_core, van_comm_rand)`.

The same two-layer hash structure is used by ZKP #2 (vote proof, conditions 2 and 7) for cross-circuit interoperability — a VAN created here can be opened by the vote proof circuit.

**Out-of-circuit helper:** `van_commitment_hash()` delegates to `van_integrity::van_integrity_hash()` with `MAX_PROPOSAL_AUTHORITY` as the proposal authority.

**Constructions:** `van_integrity::van_integrity_poseidon` (shared gadget from `circuit::van_integrity`), `AddChip`, `MulChip`.

## 8. Ballot Scaling

Purpose: convert the total delegated value into a ballot count (1 ballot = 0.125 ZEC) and simultaneously enforce a minimum voting weight (at least 1 ballot). The Euclidean-style decomposition below is one constraint short of pinning `num_ballots` to exact floor-division — see *Soundness scope* below for the precise statement and the documented under-claim window.

```
num_ballots = floor(v_total / 12,500,000)
```

**Approach:** The circuit witnesses `num_ballots` and `remainder` as free advice, then constrains:

1. **Euclidean division**: `num_ballots * BALLOT_DIVISOR + remainder == v_total` — via `MulChip` (product) and `AddChip` (sum), then equality constraint against `v_total`.

2. **Remainder range**: `remainder < 2^24` — since 24 is not a multiple of 10 (the lookup table word size), the circuit multiplies by `2^6` and range-checks the shifted value to 30 bits (3 words × 10 bits). If `remainder >= 2^24`, the shifted value `>= 2^30`, failing the check. The bound `2^24` is the tightest power-of-two-aligned bound that does not break completeness — every honest `remainder < BALLOT_DIVISOR = 12,500,000` fits in `[0, 2^24)`. The bound is *looser* than `BALLOT_DIVISOR`, however, and that slack admits an alternative witness; see *Soundness scope* below.

3. **Non-zero and upper bound**: `0 < num_ballots <= 2^30` — via `nb_minus_one = num_ballots - 1`, constrained by `nb_minus_one + 1 == num_ballots`, with a 30-bit range check on `nb_minus_one` (3 words × 10 bits, no shift needed). This single check enforces both bounds: if `nb_minus_one < 2^30` then `num_ballots ∈ [1, 2^30]`. If `num_ballots = 0`, `nb_minus_one` wraps to `p - 1 ≈ 2^254`, which fails the range check. `2^30` ballots × 0.125 ZEC ≈ 134M ZEC, well above the 21M ZEC supply.

**Constructions:** `MulChip`, `AddChip`, `LookupRangeCheckConfig`.

### Soundness scope: one-ballot under-claim window

The three constraints above do not pin `(num_ballots, remainder)` to the unique pair returned by Euclidean floor-division. Given the honest witness `(q, r) = (floor(v_total / D), v_total mod D)` with `D = BALLOT_DIVISOR = 12,500,000`, the alternative `(q - 1, r + D)` also satisfies every constraint whenever:

- `r + D < 2^24`, i.e. `r < 2^24 - D = 4,277,216`, **and**
- `q - 1 >= 1`, i.e. honest `q >= 2`.

Modeling `r` as uniform over `[0, D)`, the window covers `4,277,216 / 12,500,000 ≈ 34.22%` of `v_total` values. The opposite direction (`(q + 1, r - D)`) requires `r - D` as a near-`p` field element, which fails the 30-bit shifted range check — so **over-claim is impossible**.

**Proof sketch.** Suppose `(q_1, r_1) ≠ (q_2, r_2)` both satisfy the constraints for the same `v_total`. Then `(q_1 - q_2) * D ≡ r_2 - r_1 (mod p)`. The range checks give `|q_1 - q_2| * D ≤ (2^30 - 1) * D < 2^54 << p`, so the equation holds in integers. Combined with `|r_2 - r_1| < 2^24`, that forces `|q_1 - q_2| < 2^24 / D < 2`, so `|q_1 - q_2| ∈ {0, 1}`. The `= 1` case yields the under-claim above; the field representation of `r - D` for an honest `r ∈ [0, D)` is `≥ p - D`, which fails the 30-bit range check, ruling out over-claim.

**Impact.** The deviation is self-harming: a voter can choose to commit to one fewer ballot than their stake entitles them to, losing voting power. The VAN binding (condition 7) hashes whatever `num_ballots` the prover chose, and ZKP #2 (vote proof) opens the VAN with that same value — no downstream check re-derives `num_ballots` from `v_total`. There is no over-claim path.

### Tightening options (not implemented)

Either of the following constraints, added alongside the existing remainder range check, would pin `remainder` to `[0, BALLOT_DIVISOR)` exactly and close the under-claim window:

- Range-check `delta = (BALLOT_DIVISOR - 1) - remainder` to 24 bits (via the same `× 2^6 → 30-bit` shift), with an equality constraint `delta + remainder == BALLOT_DIVISOR - 1`. If `remainder ≥ BALLOT_DIVISOR`, `delta` wraps to a near-`p` element and fails the check.
- Range-check `remainder + (2^24 - BALLOT_DIVISOR)` to 24 bits directly. If `remainder ≥ BALLOT_DIVISOR`, the value exceeds `2^24`.

Either approach adds ~1 `MulChip` + 1 `AddChip` + 1 30-bit range check (≈3-4 rows). The current circuit ships without this tightening; the self-harming under-claim window is documented as a known limitation rather than fixed.

## 9. Note Commitment Integrity (x5)

Purpose: recompute each note's commitment in-circuit and extract `cmx` and `v` as internal wires for conditions 3 and 7.

```
NoteCommit_rcm(repr(g_d), repr(pk_d), v, rho, psi) = cm
cmx = ExtractP(cm)
```

The circuit recomputes NoteCommit from the per-note witness data and enforces strict equality against the witnessed `cm`. The resulting `cmx` (x-coordinate of the commitment point) flows to condition 3 (rho binding) and `v` flows into the shared `v_total` path used by conditions 7 and 8 as internal wires — eliminating the need for `cmx_1..5` and `v_1..5` as free witnesses.

The `v` cell is a `NoteValue` inside NoteCommit. A separate `v_base` cell (as `pallas::Base`) is constrained equal to `v` and returned for the `AddChip` sum used by conditions 7 and 8.

**Constructions:** `SinsemillaChip` (config 1), `EccChip`, `NoteCommitChip` (signed config, reused from condition 1).

## 10. Merkle Path Validity (x5)

Purpose: prove that the note's commitment exists in the note commitment tree. Uses Orchard's standard dummy note mechanism: the check is gated by value (ZIP §Note Padding).

```
root = MerklePath(cmx, pos, path)
v * (root - nc_root) = 0
```

The `GadgetMerklePath` gadget computes the Merkle root from the leaf (`cmx`) and the 32-level authentication path using Sinsemilla hashing. The `q_per_note` custom gate then enforces that either the computed root equals the public `nc_root` anchor or `v = 0` (dummy note — root mismatch is allowed).

For padding (dummy) notes (v=0), the auth path is unconstrained — the builder supplies a `MerklePath::dummy(...)` so the Merkle computation still runs (the gadget always evaluates), but the `q_per_note` root-check gate is gated off by `v = 0`. See the padding FAQ entry below for how padding addresses are synthesized.

**Constructions:** `MerkleChip` (configs 1+2), `SinsemillaChip` (configs 1+2 via MerkleChip), `q_per_note`.

## 11. Diversified Address Integrity (x5)

Purpose: prove each note's address was derived from the correct scope-specific `ivk` established in condition 5.

```
selected_ivk = ivk + is_internal * (ivk_internal - ivk)
pk_d = [selected_ivk] * g_d
```

The `q_scope_select` custom gate muxes between `ivk` (external scope) and `ivk_internal` (internal/change scope) based on the per-note `is_internal` flag. External notes use `ivk` derived from `rivk`; internal (change) notes use `ivk_internal` derived from `rivk_internal`. The gate constrains `is_internal` to be boolean.

Where:
- **ivk** — the external-scope internal wire from condition 5.
- **ivk_internal** — the internal-scope internal wire from condition 5.
- **is_internal** — per-note boolean flag used directly by the `q_scope_select` gate for the mux.
- **selected_ivk** — the muxed result, converted to `ScalarVar` for ECC multiplication.
- **g_d** — the diversified generator from the note's address.
- **pk_d** — the diversified transmission key from the note's address.

This ensures all five delegated notes belong to the same wallet (same key material `(ak, nk)`), while allowing both external and internal (change) notes.

**Constructions:** `q_scope_select`, `EccChip` (ScalarVar, variable-base mul).

## 12. Private Nullifier Derivation (x5)

Purpose: derive each Ironwood note's true nullifier in-circuit. This nullifier is NOT published — it feeds into condition 13 (IMT non-membership) and condition 14 (governance nullifier).

```
real_nf = DeriveNullifier_nk(rho, psi, cm)
```

Same `DeriveNullifier` construction as condition 2, but applied to each delegated note. Uses the shared `nk` cell (witnessed once, reused across all slots).

**Constructions:** `PoseidonChip`, `AddChip`, `EccChip`.

## 13. IMT Non-Membership (x5)

Purpose: prove the note's nullifier has NOT been spent, using a Poseidon-based Indexed Merkle Tree with K=2 punctured-range leaves. Each leaf stores three sorted nullifier boundaries `[nf_lo, nf_mid, nf_hi]` and covers the punctured interval `(nf_lo, nf_hi) \ {nf_mid}`.

**Approach:**

1. **Leaf hash**: `leaf_hash = Poseidon3(nf_lo, nf_mid, nf_hi)` — a `ConstantLength<3>` Poseidon hash (two permutations) that authenticates all three boundaries via the Merkle root.

2. **Merkle path** (29 levels, starting from `leaf_hash`): At each level, a `q_imt_swap` gate conditionally swaps `(current, sibling)` into `(left, right)` based on the position bit, then `Poseidon(left, right)` computes the parent. The swap gate constrains:
   - `left = current + pos_bit * (sibling - current)`
   - `left + right = current + sibling`
   - `bool_check(pos_bit)`

3. **Root check**: The `q_per_note` gate constrains `imt_root = nf_imt_root` (the public input). This is unconditional — dummy notes must also provide a valid IMT non-membership proof.

4. **Punctured interval check** (`q_interval` + `q_neq` gates): Proves `nf_lo < real_nf < nf_hi` AND `real_nf != nf_mid` using 3 constraints:
   - `x_lo = real_nf - nf_lo - 1` (strict lower bound offset)
   - `x_hi = nf_hi - real_nf - 1` (strict upper bound offset)
   - `(real_nf - nf_mid) * diff_inv = 1` (non-equality via inverse witness)
   - `x_lo` is range-checked to `[0, 2^250)` → `real_nf > nf_lo`
   - `x_hi` is range-checked to `[0, 2^250)` → `real_nf < nf_hi`

**Leaf authentication**: All three boundaries are authenticated via `Poseidon3(nf_lo, nf_mid, nf_hi) → Merkle root` — forging any boundary produces the wrong root. The strict ordering `nf_lo < nf_mid < nf_hi` is enforced by tree construction (sorted nullifier input) and locked by the Merkle commitment.

**Tree construction assumption:** The circuit checks only the selected authenticated leaf. Soundness also requires the committed leaves to form the canonical K=2 punctured-range tree for the nullifier set: sorted, deduplicated, padded to odd count, and non-overlapping except for shared boundary nullifiers. A nullifier must not appear as `nf_mid` in one leaf while lying inside another leaf, because the per-leaf gate cannot detect that global overlap.

**250-bit range bound assumption:** The 250-bit range check requires `nf_hi - nf_lo <= 2^250` in canonical Pallas base-field ordering. Since the Pallas base field is just above `2^254`, the IMT operator must pre-populate sentinel nullifiers at intervals of at most `2^249` (so each K=2 leaf spans at most `2 * 2^249 = 2^250`). With 33 evenly-spaced sentinels at multiples of `2^249` plus `p-1` to close the remaining tail, the entire field is covered. The `SpacedLeafImtProvider` implements this strategy.

**Why outer-interval + non-equality instead of two sub-interval checks:** An alternative design would check `nf_lo < value < nf_mid` OR `nf_mid < value < nf_hi` separately, which would bound each sub-span independently and allow coarser sentinel spacing. However, OR requires a witness selector bit and two MUX constraints in-circuit (5 custom constraints total), whereas the current approach checks the single outer interval `(nf_lo, nf_hi)` and patches the hole with a cheap inverse-witness non-equality `(value - nf_mid) * inv = 1` (3 custom constraints total). The trade-off is 16 extra off-chain sentinels (33 vs 17) in exchange for a simpler gate and 2 fewer constraints per note slot (10 fewer across all 5 slots).

**Constructions:** `PoseidonChip`, `LookupRangeCheckConfig`, `q_imt_swap`, `q_interval`, `q_neq`, `q_per_note`.

## 14. Alternate Nullifier Integrity (x5)

Purpose: derive an alternate nullifier (ZIP §Alternate Nullifier Derivation) that is published as a public input. This prevents double-delegation without revealing the true Ironwood note nullifier.

```
nf_dom = Poseidon(nk, dom, real_nf)
```

Single Poseidon hash (`ConstantLength<3>`, 2 permutations at rate 2):
- **nk** — the nullifier deriving key, making the result unforgeable.
- **dom** — the nullifier domain (public input at offset 13), constrained in-circuit to `Poseidon("governance authorization", vote_round_id)`. Scopes the alternate nullifier to this application instance and voting round.
- **real_nf** — the note's true nullifier from condition 12.

The result is constrained to the public input offsets in `GOV_NULL_PUBLIC_OFFSETS`.

**Constructions:** `PoseidonChip`.

## FAQ

- **"Why is cm_signed witnessed as a Point but ak_P as a NonIdentityPoint?"** — ak_P being identity would be a degenerate key (any signature verifies). cm_signed being identity is cryptographically negligible and caught by the equality constraint with the recomputed commitment.

- **"What if the same proof is submitted twice?"** — The nullifier nf_signed is a public input. The consuming protocol must track spent nullifiers. The circuit itself is stateless.

- **"Why are psi and rcm witnessed, not derived in-circuit?"** — Both are derived from `rseed` using Blake2b out-of-circuit and provided as private inputs. If either is incorrect, the recomputed commitment will not match, and the proof will fail.

- **"Why two Sinsemilla configs (and two NoteCommitChips)?"** — This mirrors the audited Orchard action circuit, which uses two Sinsemilla configs (one for spend-side NoteCommit, one for output-side NoteCommit) with column assignments `advices[..5]` and `advices[5..]`. Each `SinsemillaChip::configure` call creates its own selectors and gates, and each `NoteCommitChip::configure` creates decomposition/canonicity gates tied to the Sinsemilla config it receives — so two Sinsemilla configs require two NoteCommitChips. We replicate this exact layout so the delegation circuit inherits the audited chip wiring without modification. It may be possible to collapse to a single config (condition 9 already runs 5 NoteCommits on config 1 without conflict), but reusing the known-correct pattern avoids the need for a separate audit of the chip interaction.

- **"Why are padding (dummy) notes bound to the real ivk, and why not a real Orchard address?"** — Padding slots must still pass condition 11 (`pk_d = [selected_ivk] * g_d`) using the ivk derived in condition 5; otherwise the circuit would need a per-slot bypass that real notes could also exploit. Instead of constructing a real Orchard address with `fvk.address_at(...)` (which would burn an otherwise-spendable diversifier index on every proof and tie padding to a real receive-capable address), the builder synthesizes a padding pair outside the Orchard address API:
   - `g_d_pad = hash_to_curve("shielded-vote/padding-v1")(slot_index_le_bytes)` — domain-separated from Orchard's `DiversifyHash` and not derived through Orchard diversifier selection, so `(g_d_pad, pk_d_pad)` is intentionally not an `orchard::Address`.
   - `pk_d_pad = [ivk_external] * g_d_pad` — satisfies condition 11 by construction. Padding always uses the **external** ivk (`is_internal = false`), independent of the real notes' scopes, so the `q_scope_select` mux selects `ivk_external` and the equality check holds.
   - `rho`/`rseed` are sampled fresh (or replayed from `PrecomputedRandomness.padded_notes` for ZIP-244 sighash determinism), `psi`/`rcm` are derived from `rseed` exactly as Orchard does, and the note commitment / real nullifier are computed off-circuit by builder helpers that mirror Orchard's `NoteCommit` and `DeriveNullifier` bit-for-bit.

   The padding note still has `v = 0`, so condition 10 skips the Merkle root check via `v * (root - nc_root) = 0` and the auth path can be `MerklePath::dummy(...)`. Conditions 9 (note commitment integrity), 11 (address ownership against `ivk_external`), 12 (real nullifier), 13 (IMT non-membership against `nf_imt_root`), and 14 (alternate nullifier publication) all run unconditionally on the synthesized values. The published `gov_null` for a padding slot is harmless — the consuming protocol can ignore alternate nullifiers from zero-value slots or treat them as no-ops.

   Because the in-circuit `NoteCommit` and `DeriveNullifier` consume the witnessed `(g_d, pk_d, v, rho, psi, rcm)` directly and re-derive the commitment / nullifier from those bits, the synthetic `g_d_pad` is opaque to the circuit — it is just a domain-separated point chosen outside Orchard's address derivation flow. The `NoteSlotWitness` keeps `g_d`/`pk_d` typed as `NonIdentityPallasPoint` (carrying Orchard's non-identity invariant so an accidental identity fails at the construction site rather than only at proof time via `NonIdentityPoint::new`), but `cm` is relaxed to plain `pallas::Point` because Orchard's `NoteCommitment` newtype has a private constructor and synthetic padding commitments cannot be wrapped — condition 9 re-derives and constrain-equals `cm`, so a malformed point would still be rejected at proof time.
