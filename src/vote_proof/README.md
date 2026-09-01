# Vote Proof Circuit (ZKP 2, compact cast circuit)

Proves that a registered voter is casting a valid vote, without revealing
which VAN they hold. The structure follows the delegation circuit's pattern
(ZKP 1). The ElGamal encryption work lives in the encrypt-choice circuit
(ZKP 1.5, `src/encrypt_choice/`): a vote is a two-proof bundle whose halves
share one public **bridge** commitment. Numbering matches Gov Steps V1
(ZKP #2): 12 conditions total; conditions 1–9 are unchanged, and conditions
10–12 are the weighted replacements 10′–12′ (condition 4 enforces spend
authority `r_vpk = vsk.ak + [alpha_v]*G` in-circuit; the vote signature is
verified out-of-circuit under `r_vpk`).

**Public inputs:** 11 field elements.
**Current K:** 11 (2,048 rows) — with the two former El Gamal tracks removed,
`CircuitCost` reports a 1,662-row high-water mark (18.8% headroom) with 24
advice columns. Conditions 10′–12′ run on the dedicated four-column Poseidon
track; proposal-authority decrement and the condition-10′ witnesses use a
dedicated 10-column aux lane.

**Authoritative hash sources:** this README is explanatory. Reusable hash
preimages are owned by `crate::circuit::van_integrity` (VAN integrity),
`crate::circuit::vote_commitment` (vote commitment v2), `crate::bridge`
(selected-share commitments and the bridge), and `crate::shares_hash`
(the aggregate `shares_hash`). Domain-tag encoding is owned by
`crate::domain_tags`.

## Inputs

- Public (11 field elements)
   * **van_nullifier** (offset 0): the nullifier of the old VAN being spent (prevents double-vote).
   * **r_vpk_x** (offset 1): x-coordinate of the rerandomized voting key `r_vpk = vsk.ak + [alpha_v]*G` (condition 4).
   * **r_vpk_y** (offset 2): y-coordinate of `r_vpk`. Links to the vote signature verified out-of-circuit.
   * **vote_authority_note_new** (offset 3): the new VAN commitment with decremented proposal authority.
   * **vote_commitment** (offset 4): the vote commitment hash `H(DOMAIN_VC_V2, voting_round_id, shares_hash, proposal_id, decision_bucket_count)`.
   * **vote_comm_tree_root** (offset 5): root of the Poseidon-based vote commitment tree at anchor height.
   * **vote_comm_tree_anchor_height** (offset 6): caller-authenticated chain height used to source `vote_comm_tree_root`. This slot is transcript-bound but not constrained to a circuit witness.
   * **proposal_id** (offset 7): governance session parameter identifying which proposal this vote is for. The circuit constrains it to `[1, 50]`; the verifier must check it is active for `voting_round_id`.
   * **voting_round_id** (offset 8): governance session parameter identifying the active voting round — prevents cross-round replay when pinned by the verifier.
   * **bridge** (offset 9): the compact bridge commitment shared with the encrypt-choice proof. The verifier must check it equals the encrypt-choice instance's bridge; `vote_proof::verify_vote_bundle` performs the bundle checks.
   * **decision_bucket_count** (offset 10): the proposal's public option count `D` (2–16). Must be authenticated from governance and must match the encrypt-choice instance.

- Private (VAN ownership — conditions 1–4, 5)
   * **vpk_g_d**: voting public key — diversified base point (full affine point from DiversifyHash(d)). Witnessed as `NonIdentityPoint`; x-coordinate extracted for Poseidon hashing (conditions 2, 7). This is the `vpk_d` component of the voting hotkey address. Matches ZKP 1 (delegation) VAN structure.
   * **vpk_pk_d**: voting public key — diversified transmission key (full affine point, pk_d = [ivk_v] * g_d). Witnessed as `NonIdentityPoint`; x-coordinate extracted for Poseidon hashing (conditions 2, 7). Condition 3 (Diversified Address Integrity) constrains this to equal `[ivk_v] * vpk_g_d`. Matches ZKP 1 VAN structure.
   * **total_note_value**: the voter's total delegated weight, denominated in ballots (1 ballot = 0.125 ZEC; converted from zatoshi by ZKP #1 condition 8 — see the delegation README §8 for the proven relation).
   * **proposal_authority_old**: remaining proposal authority bitmask in the old VAN.
   * **van_comm_rand**: blinding randomness for the VAN commitment.
   * **vote_authority_note_old**: the old VAN commitment (two-layer Poseidon hash, same structure as ZKP 1 van_comm).
   * **vote_comm_tree_path**: Poseidon-based Merkle authentication path (24 sibling hashes).
   * **vote_comm_tree_position**: leaf position in the vote commitment tree.
   * **vsk**: voting spending key (scalar for ECC multiplication). Used in condition 3 for `[vsk] * SpendAuthG`.
   * **rivk_v**: CommitIvk randomness (scalar). Blinding factor for `CommitIvk(ak, nk, rivk_v)` in condition 3.
   * **vsk_nk**: nullifier deriving key. Concretely `fvk.nk().inner()` — the standard Orchard `NullifierDerivingKey` derived from the spending key via `PRF_expand_nk(sk)`. The "vsk" prefix reflects its role in the voting key hierarchy (shared between condition 3's CommitIvk and condition 5's nullifier), not distinct key material. It is structurally identical to the `nk` used in ZKP 1's governance nullifier; cross-circuit uniqueness is ensured by condition 5's direct VAN nullifier tag and ZKP 1's derived `dom`.

- Private (vote commitment — conditions 8–12′)
   * **shares_1..16**: the voting share vector (each in `[0, 2^30)`), identical to the shares encrypted by the encrypt-choice proof.
   * **selected_commitments[0..15]**: the per-share weighted selected commitments from the encrypt-choice bundle. Each commits one share's blind and all 16 bucket ciphertext coordinates (`crate::bridge`); ZKP 1.5 proves they commit real ElGamal encryptions.
   * The voter's decision never appears in this circuit — it is bound only inside the encrypt-choice proof's committed one-hot ciphertext vectors.

- Internal wires (not public inputs, not free witnesses)
   * **voting_round_id cell**: copied from the instance column, used in condition 2 Poseidon hash and condition 5 inner hash.
   * **domain_van_nullifier cell**: constant encoding of `"vote authority spend"` (condition 5).
   * **proposal_authority_new**: derived as `proposal_authority_old - (1 << proposal_id)` (condition 6).
   * **shares_hash**: two-level Poseidon hash over 16 blinded share commitments (condition 10). Internal wire consumed by condition 12. See `crate::shares_hash` for the authoritative preimage shape; y-coordinates defend against ciphertext sign-malleability.
   * **bridge / decision_bucket_count cells**: copied from the instance column (conditions 10′ and 12′).
   * **DOMAIN_VC constant**: `1`. Domain separation tag for Vote Commitments (condition 12). Baked into the verification key.
   * **proposal_id cell**: copied from the instance column (condition 12). Used in the vote commitment Poseidon hash.

## Public Input Provenance

The circuit proves statements relative to the public inputs supplied by the
verifier; it does not authenticate where those inputs came from.

**Ledger-state anchor:** `vote_comm_tree_root` must be looked up from chain
state at `vote_comm_tree_anchor_height`; the height is transcript-bound
metadata, not a circuit-derived witness.

**Governance session parameters:** `voting_round_id` must come from the active
governance session. `proposal_id` must be in that round's active proposal set.
The circuit only proves internal consistency: `proposal_id` is an authority
bit index in `[1, 50]`, the corresponding authority bit is decremented, and the
same value is folded into the vote commitment. It does not authenticate the
active-proposal registry.

**Weighted-vote parameters:** `decision_bucket_count` must equal the option
count `D` declared by governance for the proposal (also reject `D < 2`), and
`bridge` must equal the accompanying encrypt-choice instance's public bridge.
The election-authority key is authenticated on the encrypt-choice instance,
which carries the ElGamal ciphertext constraints.

## Condition 2: VAN Integrity ✅

Purpose: prove that the old VAN commitment is correctly constructed from its components. Uses the **same two-layer hash structure as ZKP 1 (delegation)** so that a VAN created by the delegation circuit can be spent (opened) by the vote proof circuit.

```
van_comm_core = Poseidon(DOMAIN_VAN, vpk_g_d_x, vpk_pk_d_x, total_note_value,
                         voting_round_id, proposal_authority_old)
vote_authority_note_old = Poseidon(van_comm_core, van_comm_rand)
```

Where:
- **DOMAIN_VAN**: `0`. Domain separation tag for Vote Authority Notes (vs `DOMAIN_VC = 1` for Vote Commitments). Assigned via `assign_advice_from_constant` so the value is baked into the verification key.
- **vpk_g_d**, **vpk_pk_d**: voting public key address components (diversified base and transmission key x-coordinates). Same encoding as in ZKP 1 condition 7, so a VAN created by delegation has the same commitment structure.
- **total_note_value**: the voter's total delegated weight. Shared with condition 8 (shares sum check).
- **voting_round_id**: the vote round identifier (public input `VOTING_ROUND_ID_PUBLIC_OFFSET`, offset 8). Copied from the instance column via `assign_advice_from_instance`, ensuring the in-circuit value matches the verifier's public input.
- **proposal_authority_old**: remaining proposal authority bitmask. Shared with condition 6 (decrement check).
- **van_comm_rand**: random blinding factor. Prevents observers from brute-forcing the address or weight from the public VAN commitment.
- **vote_authority_note_old**: the witnessed VAN commitment. Constrained to equal the two-layer Poseidon output via `constrain_equal`.

Address encoding note: the VAN hash binds only the x-coordinates of `vpk_g_d` and `vpk_pk_d`. A coordinated negation of both address points leaves the condition 2 and condition 7 VAN hash inputs unchanged. The vote proof remains sound because condition 3 constrains the witnessed full points to the voting key hierarchy, and condition 5 derives the public nullifier from the same `vsk_nk`, `voting_round_id`, and `vote_authority_note_old`, so both y-sign variants converge to the same spend nullifier. Future code must not use `van_integrity_hash` as a standalone full-address commitment unless it adds equivalent full-point and nullifier constraints or changes the hash preimage.

**Function:** Two Poseidon invocations: first `ConstantLength<6>` (core), then `ConstantLength<2>` (core, van_comm_rand). Uses `Pow5Chip` / `P128Pow5T3` with rate 2. Matches delegation circuit condition 7 (van_comm) structure.

**Constraint:** The circuit computes the two-layer hash and enforces strict equality with `vote_authority_note_old`. Since `vote_authority_note_old` will also be used as the Merkle leaf in condition 1, this creates a binding: the VAN membership proof and the VAN integrity check are tied to the same commitment.

**Condition 4: Spend Authority** — enforced in-circuit. The spec requires `r_vpk = vsk.ak + [alpha_v] * G`. The circuit witnesses `alpha_v`, computes `[alpha_v]*SpendAuthG` via fixed-base mul, adds it to `vsk_ak_point` (from condition 3), and constrains the result to the instance column at `R_VPK_X_PUBLIC_OFFSET` and `R_VPK_Y_PUBLIC_OFFSET` (public input offsets 1 and 2). The vote signature is verified out-of-circuit under `r_vpk` over the transaction sighash.

**Out-of-circuit helper:** `van_integrity::van_integrity_hash(vpk_g_d_x, vpk_pk_d_x, total_note_value, voting_round_id, proposal_authority_old, van_comm_rand)` from the shared `circuit::van_integrity` module computes the same two-layer hash outside the circuit for builder and test use.

**Constructions:** `van_integrity::van_integrity_poseidon` (shared gadget from `circuit::van_integrity`).

## Condition 1: VAN Membership ✅

Purpose: prove the voter's VAN is registered in the vote commitment tree, without revealing which one.

```
MerklePath(vote_authority_note_old, vote_comm_tree_position, vote_comm_tree_path) = vote_comm_tree_root
```

Where:
- **vote_authority_note_old**: the Merkle leaf. Cell-equality-linked to condition 2's derived VAN hash, binding the membership proof to the same commitment.
- **vote_comm_tree_position**: leaf position in the tree (private witness). At each level, the position bit determines child ordering.
- **vote_comm_tree_path**: 24 sibling hashes along the authentication path (private witness).
- **vote_comm_tree_root**: the public tree anchor (public input `VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET`, offset 5).
- **vote_comm_tree_anchor_height**: the chain height used by the verifier's caller to look up `vote_comm_tree_root` (public input offset 6). The circuit does not constrain this value to the Merkle path witness.

**Function:** Poseidon-based Merkle path verification (24 levels). At each level, a conditional swap gate orders (current, sibling) into (left, right) based on the position bit, then `Poseidon(left, right)` computes the parent. The hash function matches `vote_commitment_tree::MerkleHashVote::combine` — `Poseidon(left, right)` with no level tag.

**Structure:** 24 swap regions (1 row each) + 24 Poseidon `ConstantLength<2>` hashes (~1,560 total rows). The swap gate (`q_merkle_swap`) constrains:
- `left = current + pos_bit * (sibling - current)` — selects current or sibling
- `left + right = current + sibling` — conservation
- `pos_bit ∈ {0, 1}` — boolean check

Identical to the delegation circuit's `q_imt_swap` gate.

**Constraint:** The circuit computes the Merkle root from the leaf and path, then enforces `constrain_instance(computed_root, VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET)`, binding the derived root to the `VOTE_COMM_TREE_ROOT_PUBLIC_OFFSET` public input (offset 5). The verifier's caller enforces the binding between the height and root when it looks up the root at `vote_comm_tree_anchor_height`.

**Out-of-circuit helper:** `poseidon_hash_2()` computes `Poseidon(a, b)` outside the circuit for builder and test use.

**Constructions:** `PoseidonChip`, `q_merkle_swap` selector.

## Condition 3: Diversified Address Integrity ✅

Purpose: prove the voter controls the voting hotkey address delegated to in Phase 1–2 (spec: Diversified Address Integrity). Uses the same CommitIvk chain as ZKP 1 (delegation) condition 5, implemented via the shared **`circuit::address_ownership`** gadget (ZKP 1 and ZKP 2 both call `spend_auth_g_mul` and `prove_address_ownership`).

```
vsk_ak      = [vsk] * SpendAuthG               (fixed-base ECC mul)
ak          = ExtractP(vsk_ak)                  (x-coordinate)
ivk_v       = CommitIvk_rivk_v(ak, vsk.nk)     (Sinsemilla commitment)
vpk_pk_d    = [ivk_v] * vpk_g_d                (variable-base ECC mul)
```

Where:
- **vsk**: voting spending key (private witness, `pallas::Scalar`). The secret key that authorizes vote casting.
- **SpendAuthG**: fixed generator point on the Pallas curve, reused from the Zcash Orchard protocol. Used both here (condition 3) and by the encrypt-choice circuit (El Gamal generator).
- **ak**: the spend validating key's x-coordinate, derived in-circuit from `[vsk] * SpendAuthG` then `ExtractP`. Not a separate witness — it's an internal wire.
- **vsk_nk**: nullifier deriving key (private witness, `pallas::Base`). The same cell is shared with condition 5 (VAN nullifier keying). Witnessed before condition 3 in the synthesize flow.
- **rivk_v**: CommitIvk randomness (private witness, `pallas::Scalar`). Blinding factor for the Sinsemilla commitment.
- **ivk_v**: the incoming viewing key, derived in-circuit via `CommitIvk(ak, nk, rivk_v)`. Internal wire — flows from CommitIvk output to variable-base ECC mul input via `ScalarVar::from_base`.
- **vpk_g_d**: diversified base point (private witness, full affine point). Witnessed as `NonIdentityPoint`. The x-coordinate is extracted for Poseidon hashing in conditions 2 and 7.
- **vpk_pk_d**: diversified transmission key (private witness, full affine point). Witnessed as `NonIdentityPoint`. Constrained to equal the derived `[ivk_v] * vpk_g_d` via `Point::constrain_equal`.

**Structure:** Uses the shared `circuit::address_ownership` gadget:
1. `spend_auth_g_mul(ecc_chip, layouter, "cond3: [vsk] SpendAuthG", vsk_scalar)` → `vsk_ak_point` (fixed-base scalar mul)
2. `vsk_ak_point.extract_p()` → `ak` (x-coordinate extraction)
3. `prove_address_ownership(..., ak, vsk_nk, rivk_v_scalar, &vpk_g_d_point, &vpk_pk_d_point)` — CommitIvk, `[ivk_v]*vpk_g_d`, and constrain to vpk_pk_d (same flow as ZKP 1 condition 5)

**Chip dependencies:** `SinsemillaChip`, `CommitIvkChip`, `EccChip` (used inside the shared gadget). The Sinsemilla chip also loads the 10-bit lookup table used by conditions 6 and 9.

**Constraint:** The circuit derives vpk_pk_d from vsk → ak → ivk_v → [ivk_v] * vpk_g_d and enforces full point equality with the witnessed vpk_pk_d. Since vpk_pk_d's x-coordinate flows into conditions 2 and 7 (VAN integrity hashes), and vpk_g_d's x-coordinate flows into the same hashes, any mismatch in the key hierarchy would break conditions 2/3/7 simultaneously.

**Security properties:**
- **Key binding:** The CommitIvk chain cryptographically binds vsk to the VAN address (vpk_g_d, vpk_pk_d). A prover who doesn't know vsk cannot produce a valid ivk_v that maps vpk_g_d to vpk_pk_d.
- **Canonicity:** The CommitIvk gadget enforces canonical decomposition of ak and nk, preventing malleability attacks where different bit representations produce the same commitment.
- **Non-identity:** Both vpk_g_d and vpk_pk_d are witnessed as `NonIdentityPoint`, ensuring they are not the curve identity (which would trivially satisfy the constraint for any ivk_v).
- **Shared nk:** Using the same vsk_nk cell for both CommitIvk (condition 3) and the VAN nullifier (condition 5) ensures the nullifier is bound to the same key hierarchy that authorizes the vote.

**Out-of-circuit helper:** `derive_voting_address(vsk, nk, rivk_v)` in tests performs the same computation: `[vsk] * SpendAuthG → ExtractP → CommitIvk → [ivk_v] * g_d`. Uses `CommitDomain::short_commit` from `halo2_gadgets::sinsemilla::primitives`.

**Constructions:** Shared `circuit::address_ownership::spend_auth_g_mul` and `circuit::address_ownership::prove_address_ownership`; `SinsemillaChip`, `CommitIvkChip`, `EccChip`, `ScalarFixed`, `NonIdentityPoint`, `Point::constrain_equal`.

## Condition 4: Spend Authority ✅

Purpose: bind the rerandomized voting public key `r_vpk` to the spending key and a randomizer so the verifier can check the vote signature out-of-circuit under `r_vpk`.

```
vsk_ak_point   = [vsk] * SpendAuthG        (from condition 3)
alpha_v_commit = [alpha_v] * SpendAuthG    (fixed-base ECC mul)
r_vpk_derived  = alpha_v_commit + vsk_ak_point
constrain_instance(r_vpk_derived, R_VPK_X_PUBLIC_OFFSET), constrain_instance(r_vpk_derived.y(), R_VPK_Y_PUBLIC_OFFSET)
```

Where:
- **vsk_ak_point**: same point as in condition 3 (`[vsk]*SpendAuthG`), reused via the existing fixed-base mul.
- **alpha_v**: spend auth randomizer (private witness, `pallas::Scalar`). The
  circuit matches upstream Orchard and does not constrain this value non-zero;
  an honest wallet samples it freshly, while a zero witness is a documented
  self-linking/coercion surface rather than a soundness break.
- **r_vpk_derived**: in-circuit result constrained to the instance column at offsets 1 (x) and 2 (y).

**Constraint:** The circuit computes `r_vpk = vsk.ak + [alpha_v]*G` and constrains it to the public inputs `r_vpk_x`, `r_vpk_y`. The vote signature is verified out-of-circuit under `r_vpk` over the transaction sighash.

**Constructions:** `spend_auth_g_mul` (same as condition 3), `Point::add`, `constrain_instance`.

## Condition 5: VAN Nullifier Integrity ✅

Purpose: derive a nullifier that prevents double-voting without revealing the VAN.

```
van_nullifier = Poseidon(vsk_nk, domain_van_nullifier, voting_round_id, vote_authority_note_old)
```

Single `ConstantLength<4>` call. Unlike ZKP 1 condition 14's three-input
alternate nullifier over the precomputed `dom`, the VAN nullifier includes its
domain tag and voting round ID directly:

- **`vsk_nk`**: nullifier deriving key (private witness, base field element). Concretely `fvk.nk().inner()` — structurally the same value as the `nk` used in ZKP 1. The same cell is shared with condition 3 (CommitIvk), binding the nullifier to the authenticated key hierarchy.
- **`domain_van_nullifier`**: `"vote authority spend"` (20 bytes) zero-padded to 32 and interpreted as a little-endian Pallas field element per `crate::domain_tags`. Assigned via `assign_advice_from_constant` so the value is **baked into the verification key** — a prover cannot substitute a different value. This tag is the sole cross-circuit separator between this nullifier and ZKP 1's governance nullifier, which uses `"governance authorization"` under the same key. The registry test asserts the tags are distinct field elements.
- **`voting_round_id`**: cell-equality-linked to condition 2's instance copy, scoping the nullifier to this round.
- **`vote_authority_note_old`**: cell-equality-linked to condition 2's derived VAN hash, binding conditions 2 and 5 together.

**Structure:** Single `ConstantLength<4>` Poseidon hash (2 permutations at rate 2, ~130 rows).

**Constraint:** The circuit computes the nested hash and enforces `constrain_instance(result, VAN_NULLIFIER_PUBLIC_OFFSET)` — binding the derived value to the public input at offset 0. This is the first `constrain_instance` call in the circuit.

**Out-of-circuit helper:** `van_nullifier_hash()` computes the same nested Poseidon hash outside the circuit. `domain_van_nullifier()` returns the domain separator constant.

**Constructions:** `PoseidonChip` (reused from condition 2), `constrain_instance`.

## Condition 6: Proposal Authority Decrement ✅

Purpose: ensure the voter has authority for the voted proposal and correctly clears that bit in the authority bitmask.

**Baseline spec (Gov Steps V1 §3.5 Step 2, ZKP #2 Condition 6):** `proposal_authority` is a 16-bit bitmask. This circuit widens it to 51 bits so IDs 1–50 are usable while ID 0 remains reserved. One vote consumes the chosen bit: `proposal_authority_new = proposal_authority_old - (1 << proposal_id)`, and the `proposal_id`-th bit of `proposal_authority_old` must be 1.

**Implementation (bit decomposition):**

1. **Decompose** `proposal_authority_old` into 51 bits `b_i` (each boolean), with recomposition `sum(b_i * 2^i) = proposal_authority_old`.
2. **Selector** `sel_i = 1` iff `proposal_id == i` (exactly one active); constrain `run_selected = sum(sel_i * b_i) = 1` so the selected bit is set (voter has authority).
3. **Clear and recompose**: `b_new_i = b_i*(1-sel_i)`; then `sum(b_new_i * 2^i) = proposal_authority_new`. Constrain this to equal the witnessed `proposal_authority_new` (and thus the new VAN in condition 7).

No diff/gap or strict range-check chip; the 51-bit decomposition implies `proposal_authority_old` and `proposal_authority_new` are in `[0, 2^51)`. The `(proposal_id, one_shifted)` lookup constrains `proposal_id in [0, 50]` and `one_shifted = 2^proposal_id`; a separate non-zero gate (`q_cond_6 * (1 - proposal_id * proposal_id_inv) = 0`) rejects `proposal_id = 0`, making the effective circuit range `[1, 50]`. Bit 0 is permanently reserved as the sentinel/unset value. A voting round therefore supports at most 50 proposals. The builder provides `one_shifted` and `proposal_authority_new = old - one_shifted`.

**Structure:** One 52-row region: row 0 has `proposal_id`, `one_shifted` (lookup); rows 1..51 have bits, selectors, and running sums; gates cover initialization, recurrence, and `run_selected = 1` at the last bit row. Equality constraints bind recomposed `run_old` to `proposal_authority_old` and `run_new` to `proposal_authority_new`.

**Constructions:** Custom `AuthorityDecrementChip` (see `src/vote_proof/authority_decrement.rs`) — a dedicated 52-row bit-decomposition chip with its own `(proposal_id, 2^proposal_id)` lookup table covering `proposal_id ∈ [0, 50]`. Range enforcement comes from the bit decomposition itself (51 boolean cells recompose into `proposal_authority_old`).

## Condition 7: New VAN Integrity ✅

Purpose: the new VAN has the same structure as the old (ZKP 1–compatible two-layer hash) except with decremented authority.

Same x-coordinate-only two-layer formula as condition 2: `van_comm_core = Poseidon(DOMAIN_VAN, vpk_g_d_x, vpk_pk_d_x, total_note_value, voting_round_id, proposal_authority_new)` then `vote_authority_note_new = Poseidon(van_comm_core, van_comm_rand)`.

Where:
- **vpk_g_d**, **vpk_pk_d**, **total_note_value**, **voting_round_id**, **van_comm_rand** are cell-equality-linked to the same witness cells used in condition 2.
- **proposal_authority_new**: flows from condition 6's output. This is the only difference between the condition 2 and condition 7 hashes.

**Constraint:** The circuit computes the two-layer hash and enforces `constrain_instance(derived_van_new, VOTE_AUTHORITY_NOTE_NEW_PUBLIC_OFFSET)` — binding the result to the public input at offset 3.

**Out-of-circuit helper:** Reuses `van_integrity::van_integrity_hash(vpk_g_d, vpk_pk_d, total_note_value, voting_round_id, proposal_authority_new, van_comm_rand)` with `proposal_authority_new = proposal_authority_old - (1 << proposal_id)`. (Note: the shared module's parameter names are `g_d_x`/`pk_d_x`.)

Clients that need to plan several ordered votes before proving can call
`vote_proof::derive_vote_authority_transition`. It derives the old and new VAN
with the same native implementation used by the proof builder, while leaving
the circuit and its proving and verifying keys unchanged.

**Constructions:** `van_integrity::van_integrity_poseidon` (shared gadget from `circuit::van_integrity`).

## Share Decomposition (builder)

The builder decomposes `num_ballots` into 16 shares using a three-phase algorithm that maximizes per-share anonymity while remaining compatible with the circuit constraints (conditions 8–9).

**Phase 1 — Greedy fill (up to 9 slots).** Place the largest standard denominations from `[10M, 1M, 100K, 10K, 1K, 100, 10, 1]` (in ballots) that fit into the remaining balance. At most `MAX_DENOM_SHARES = 9` slots are consumed, leaving at least 7 free.

**Why 9 denomination slots?** The greedy algorithm can repeat denominations (e.g. 20M ballots uses two 10M slots), so the worst case exceeds the 8 distinct denominations. Capping at 9 accommodates realistic balances while reserving at least 7 slots for remainder dispersion — fewer remainder slots would concentrate non-standard values into larger, more fingerprintable amounts.

**Phase 2 — Remainder distribution (free slots).** If a non-zero remainder exists after greedy fill, spread it across all remaining slots using PRF-derived weights (`DOMAIN_REMAINDER = 0x03`). Each slot receives a weighted proportion of the remainder (with at least 1 ballot per slot when the remainder allows), preventing any single non-standard value from fingerprinting the voter's exact balance.

**Phase 3 — Deterministic shuffle.** A [Fisher-Yates shuffle](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle) (iterating from the last index down, swapping each position with a uniformly random earlier position) seeded by the PRF (`DOMAIN_SHUFFLE = 0x02`) randomizes all 16 slot positions. Without this, share indices would encode denomination rank (e.g. index 0 = largest denomination), leaking balance magnitude to any adversary that decrypts a single share.

All PRF derivations are keyed by the spending key and bound to `(voting_round_id, proposal_id, van_commitment)`. The resulting `r_i` values are deterministically remapped away from the exact zero field element before encryption, and the circuit rejects any zero witness. This means:
- Two voters with the **same balance** produce different remainder weights and shuffle orders (different `sk`).
- The same voter with **multiple VANs** produces different patterns per VAN (different `van_commitment`).
- Secrets are deterministically re-derivable after a crash without persisting them.

The denomination slots use values common across all voters at a given balance tier, so they blend into the anonymity set. The remainder slots use unique PRF-derived splits that prevent exact-balance fingerprinting. After shuffling, the position of each value is uniformly random.

**Invariants enforced by the circuit:**
- Sum of all 16 shares equals `num_ballots` (condition 8).
- Each share is in `[0, 2^30)` (condition 9).

## Condition 8: Shares Sum Correctness ✅

Purpose: voting shares decomposition is consistent with the total delegated weight (in ballots).

```
sum(share_0, ..., share_15) = total_note_value
```

Where:
- **share_0..share_15**: the 16 plaintext voting shares (private witnesses). Each share is produced by the denomination-based decomposition described above. These cells will also be used by condition 9 (range check) and condition 10′ (bridge re-opening).
- **total_note_value**: the voter's total delegated weight in ballots (1 ballot = 0.125 ZEC). Cell-equality-linked to the same witness cell used in condition 2 (VAN integrity), binding the shares decomposition to the authenticated VAN.

**Structure:** Fifteen chained `AddChip` additions (15 rows):
- `partial_1  = share_0  + share_1`
- `partial_2  = partial_1  + share_2`
- `...`
- `partial_14 = partial_13 + share_14`
- `shares_sum = partial_14 + share_15`

**Constraint:** `constrain_equal(shares_sum, total_note_value)` — the sum of all 16 shares must exactly equal the voter's total delegated weight. This prevents the voter from creating or destroying voting power during the share decomposition.

**Constructions:** `AddChip` (reused from condition 6).

## Condition 9: Shares Range ✅

Purpose: prevent overflow by ensuring each share fits in a bounded range.

```
Each share_i in [0, 2^30)
```

Where:
- **share_0..share_15**: the 16 plaintext voting shares from condition 8.

**Motivation:** The sum constraint (condition 8) holds in the base field F_p, but El Gamal encryption (in the encrypt-choice circuit) operates in the scalar field F_q via `share_i * G`. For Pallas, p ≠ q — a large base-field element (e.g. `p − 50`) reduces to a different value mod q, breaking the correspondence between the constrained sum and the encrypted values. Bounding each share to `[0, 2^30)` guarantees both representations agree (no modular reduction in either field), so the homomorphic tally faithfully reflects condition 8's sum. The encrypt-choice circuit proves the same 30-bit range through its fixed-base decomposition; this re-check is cheap defense in depth for the condition-8 no-wraparound argument.

Secondary benefit: after accumulation the EA decrypts to `total_value * G` and must solve a bounded DLOG (baby-step giant-step, O(√n)) to recover `total_value`. Bounded shares keep the per-decision aggregate small enough for efficient recovery.

**Bound:** `[0, 2^30)` via 3 × 10-bit words with strict mode. Shares are denominated in ballots (1 ballot = 0.125 ZEC, converted from zatoshi by ZKP #1's condition 8; see the delegation README §8 for the proven relation). halo2_gadgets v0.5's `short_range_check` is `pub(crate)` (private to the gadget crate), so exact non-10-bit-aligned bounds (e.g. 24-bit) are unavailable. 2^30 ballots ≈ 134M ZEC — well above the 21M ZEC supply — so the bound is never binding in practice.

**Structure:** For each share, one `copy_check` call (16 calls total, ~64 rows):
- `copy_check(share_i, 3, true)` — decomposes the share into 3 × 10-bit lookup words. Each word is range-checked via the 10-bit lookup table. The `true` (strict) flag constrains the final running sum `z_3` to equal 0, enforcing `share < 2^30`.

If a share exceeds `2^30` or is a wrapped large field element (e.g. `p - k` from underflow), the 3-word decomposition produces a non-zero `z_3`, which fails the strict check.

**Note:** Share cells are cloned for `copy_check` (which takes ownership). The original cells remain available for condition 10′ (bridge re-opening).

**Constructions:** `LookupRangeCheckConfig` (reused from condition 6).

## Condition 10′: Bridge Re-Opening ✅

Purpose: bind the pre-encrypted weights and ciphertext commitments proven by
the encrypt-choice proof (ZKP 1.5) to this cast proof.

```
bridge = Poseidon(ENCRYPT_CHOICE_BRIDGE_DOMAIN, voting_round_id, proposal_id,
                  decision_bucket_count,
                  w_0, selected_comm_0, ..., w_15, selected_comm_15)
```

Where:
- **w_i**: the same share cells constrained by conditions 8 (sum) and 9
  (range) — cell reuse guarantees the weights ZKP 1.5 encrypted are exactly
  the shares this VAN authorizes.
- **selected_comm_i**: witnessed from the encrypt-choice bundle. Each is a
  66-input Poseidon commitment over the share's blind and all 16 bucket
  ciphertext coordinates; `crate::bridge` owns the preimage layout and
  ZKP 1.5 constrains the committed points to real ElGamal encryptions.
- **voting_round_id / proposal_id / decision_bucket_count**: copied from the
  instance column, so a bridge assembled under another voting context cannot
  re-verify here.

**Constraint:** the derived bridge is bound to `BRIDGE_PUBLIC_OFFSET`
(offset 9). The verifier must additionally check the value equals the
encrypt-choice instance's public bridge (`verify_vote_bundle`).

**Out-of-circuit helper:** `crate::bridge::bridge_commitment`.

**Constructions:** shared `crate::bridge::compute_bridge_in_circuit` on the
dedicated Poseidon hash track.

## Condition 11′: Shares Hash Integrity ✅

Purpose: aggregate the 16 selected commitments for the vote commitment.

```
shares_hash = Poseidon(selected_comm_0, ..., selected_comm_15)
```

`shares_hash` is an internal wire consumed by condition 12′; ZKP #3
recomputes the same hash from private witnesses when revealing a share. The
authoritative shape lives in `crate::shares_hash`.

**Constructions:** `crate::shares_hash::compute_shares_hash_from_comms_in_circuit`
(`PoseidonChip` on the dedicated hash track).

## Condition 12′: Vote Commitment Integrity (v2) ✅

Purpose: the public vote commitment is correctly constructed from the shares
hash, the proposal, and the public bucket count. This is the value posted
on-chain, inserted into the vote commitment tree, and later opened by ZKP #3.

```
vote_commitment = Poseidon(DOMAIN_VC_V2, voting_round_id, shares_hash,
                           proposal_id, decision_bucket_count)
```

Where:
- **DOMAIN_VC_V2**: `2`. Distinct from `DOMAIN_VAN = 0` and the retired v1
  `DOMAIN_VC = 1`, so v2 commitments can never collide with VANs or
  historical v1 commitments in the shared tree. Assigned via
  `assign_advice_from_constant` so the value is baked into the verification
  key.
- **shares_hash**: internal wire from condition 11′.
- **proposal_id / decision_bucket_count**: instance copies. Binding `D`
  prevents replaying a commitment under a proposal with a different option
  count. The plaintext `vote_decision` of v1 is gone: the decision is bound
  only through the committed one-hot ciphertext vectors inside
  `shares_hash`.

**Constraint:** `constrain_instance(vote_commitment,
VOTE_COMMITMENT_PUBLIC_OFFSET)` (offset 4).

**Out-of-circuit helper:** `vote_commitment_hash_v2()`.

**Constructions:** shared `crate::gadgets::vote_commitment` gadget on the
dedicated hash track.

**Data flow (conditions 8–12′):**
```
shares (8: sum, 9: range) ────────────┐
selected_comms (from ZKP 1.5) ────────┼─ bridge (10′) ──→ BRIDGE_PUBLIC_OFFSET
                                      │
selected_comms ─ shares_hash (11′) ───┼─ vote_commitment (12′) ──→ VOTE_COMMITMENT_PUBLIC_OFFSET
proposal_id / round / D ──────────────┘
```

## Vote bundle

A complete vote is `VoteBundle { encrypt_choice, cast }`. The verifier must:

1. Verify the encrypt-choice proof (which authenticates `ea_pk` and the
   ElGamal ciphertexts) and this cast proof.
2. Check `bridge`, `voting_round_id`, `proposal_id`, and
   `decision_bucket_count` are equal across the two instances
   (`check_vote_bundle_consistency`).
3. Authenticate the governance-sourced fields of both instances.

`vote_proof::verify_vote_bundle` performs steps 1–2. The cast builder
independently re-derives the shares and bridge from its own inputs and
rejects a stale or cross-context encrypt-choice bundle before proving.

## Column Layout

| Columns | Use |
|---------|-----|
| `advices[0..5]` | General witness assignment, ECC (cond 3, 4), Sinsemilla/CommitIvk (cond 3) |
| `advices[5]` | Poseidon partial S-box |
| `advices[6]` | Poseidon state + AddChip output (c) |
| `advices[7]` | Poseidon state + AddChip input (a) |
| `advices[8]` | Poseidon state + AddChip input (b) |
| `advices[9]` | Range check running sum |
| `aux_advices[0..10]` | Condition 6 authority decrement; condition 10′ witness cells |
| `hash_advices[0..4]` | Conditions 10′–12′: bridge, shares hash, vote commitment |
| `constants` | Constant assignments (DOMAIN_VAN, DOMAIN_VC_V2, domain tags, ONE) |
| `lagrange_coeffs[0..2]` | Primary ECC Lagrange coefficients |
| `lagrange_coeffs[2..5]` | Poseidon rc_a |
| `lagrange_coeffs[5..8]` | Poseidon rc_b |
| `hash_round_constants[0..6]` | Dedicated hash-track Poseidon round constants |
| `table_idx` (+ additional lookup columns) | 10-bit lookup table [0, 1024), Sinsemilla lookup (loaded by `SinsemillaChip`); (proposal_id, 2^proposal_id) table for condition 6 |
| `primary` | 11 public inputs |

## Chip Summary

| Chip | Conditions | Role |
|------|-----------|------|
| `PoseidonChip` (Pow5) | 1, 2, 5, 7, 10′, 11′, 12′ | Poseidon hashing (Merkle paths, VAN integrity, nullifiers, bridge, shares hash, vote commitment) |
| `EccChip` | 3, 4 | Fixed-base and variable-base scalar multiplication, point addition, ExtractP (cond 4: [alpha_v]*G + vsk_ak) |
| `SinsemillaChip` | 3 | Sinsemilla hash inside CommitIvk |
| `CommitIvkChip` | 3 | Canonicity gate for ak/nk decomposition in CommitIvk |
| `AddChip` | 8 | Field element addition (shares sum) |
| `LookupRangeCheckConfig` | 9 | 10-bit running-sum range checks |
| `AuthorityDecrementChip` (custom) | 6 | Bit-decomposition + `(proposal_id, 2^proposal_id)` lookup; see `authority_decrement.rs` |
