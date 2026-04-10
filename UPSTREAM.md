# Orchard fork: upstream checklist

This document lists every remaining change in `orchard/` (our fork of
[zcash/orchard](https://github.com/zcash/orchard) v0.11.0) that would need to
be accepted upstream before we can delete the fork and depend on the published
crate.

Once all items below are merged upstream, replace
`orchard = { path = "../orchard" }` in `voting-circuits/Cargo.toml` with
`orchard = "<version>"` and delete the `orchard/` directory.

## Visibility widenings (`pub(crate)` to `pub`)

These items are `pub(crate)` in upstream v0.11.0. Our fork widens them to `pub`
because voting-circuits imports them directly.

### `circuit.rs` / `circuit/gadget.rs`

- [ ] `pub mod commit_ivk` (circuit.rs) — expose the CommitIvk sub-circuit
- [ ] `pub mod note_commit` (circuit.rs) — expose the NoteCommit sub-circuit
- [ ] `pub mod gadget` (circuit.rs) — expose the gadget module
- [ ] `pub trait AddInstruction` (gadget.rs) — addition trait
- [ ] `pub fn assign_free_advice` (gadget.rs) — cell assignment helper
- [ ] `pub fn derive_nullifier` (gadget.rs) — nullifier derivation gadget
- [ ] `pub use commit_ivk`, `pub use note_commit` (gadget.rs) — re-exports

### `circuit/gadget/add_chip.rs`

- [ ] `pub struct AddConfig` / `pub struct AddChip`
- [ ] `pub fn configure` / `pub fn construct`

### `circuit/commit_ivk.rs`

- [ ] `pub fn configure` / `pub fn construct`
- [ ] `pub mod gadgets` / `pub fn commit_ivk`

### `circuit/note_commit.rs`

- [ ] `pub fn configure` / `pub fn construct`
- [ ] `pub mod gadgets` / `pub fn note_commit`

### `constants.rs`

- [ ] `pub const L_ORCHARD_BASE`

### `constants/fixed_bases.rs`

- [ ] `pub` re-exports: `OrchardFixedBases`, `OrchardFixedBasesFull`

### `keys.rs`

- [ ] `pub struct NullifierDerivingKey` + `pub fn inner()`
- [ ] `pub struct CommitIvkRandomness` + `pub fn inner()`
- [ ] `pub fn SpendingKey::random()`
- [ ] `pub fn SpendAuthorizingKey::derive_inner()`
- [ ] `pub fn FullViewingKey::nk()` / `pub fn FullViewingKey::rivk()`
- [ ] `pub fn DiversifiedTransmissionKey::inner()` / `pub fn DiversifiedTransmissionKey::to_bytes()`

### `spec.rs`

- [ ] `pub struct NonIdentityPallasPoint` + `pub fn from_bytes()`

### `note.rs` / `note/commitment.rs` / `note/nullifier.rs`

- [ ] `pub mod commitment` / `pub mod nullifier` (note.rs)
- [ ] `pub fn Note::new` / `pub fn Note::dummy` / `pub fn Note::from_nf_old`
- [ ] `pub fn Note::into_inner` / `pub fn Note::psi` / `pub fn Note::rcm`
- [ ] `pub use NoteCommitTrapdoor` re-export (note.rs)
- [ ] `pub struct NoteCommitTrapdoor` + `pub fn inner()` (commitment.rs)
- [ ] `pub fn NoteCommitment::inner()`
- [ ] `pub fn ExtractedNoteCommitment::inner()`
- [ ] `pub` field on `Nullifier` (nullifier.rs)

### `tree.rs`

- [ ] `pub fn MerkleHashOrchard::inner()`
- [ ] `pub fn MerklePath::dummy()`

### `value.rs`

- [ ] `pub fn NoteValue::zero()`

### `address.rs`

- [ ] `pub fn Address::g_d()`
- [ ] `pub fn Address::pk_d()`

## Structural additions (new code, not just visibility)

These items don't exist in upstream v0.11.0 at all. They would need to be
proposed as new functionality.

### `constants/fixed_bases.rs`

- [ ] `OrchardBaseFieldBases` enum — new enum routing base-field fixed-base
  multiplication (variants: `NullifierK`, `SpendAuthGBase`)
- [ ] `OrchardShortScalarBases` enum — new enum routing short-scalar fixed-base
  multiplication (variants: `ValueCommitV`, `SpendAuthGShort`)
- [ ] Expanded `OrchardFixedBases` enum — new `Base(OrchardBaseFieldBases)` and
  `Short(OrchardShortScalarBases)` variants alongside the existing
  `Full(OrchardFixedBasesFull)`
- [ ] `From` trait implementations and `FixedPoint` impls for the new variants
- [ ] Test additions for the new variants

### `constants/fixed_bases/spend_auth_g.rs`

- [ ] `Z_SHORT` / `U_SHORT` precomputed tables for `SpendAuthGShort` — enables
  short-scalar multiplication on the spend authorization generator

### `circuit/gadget.rs`

- [ ] `pub fn assign_constant()` — helper for assigning a constant value
  constrained by the verification key (counterpart to the existing
  `assign_free_advice`)
