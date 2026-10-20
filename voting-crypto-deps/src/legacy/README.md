# Voting-owned Orchard circuit compatibility

This private module pins the legacy Halo2 composition implementation used by
voting proofs. Common may change its payment backend without maintaining a
public composition API for these circuits. This prototype retains Pasta,
Halo2, and their arithmetic traits; replacing those libraries is separate work.

## Provenance

The chips, fixed-base tables, and protocol domains are inherited from
`zakura-core/common` revision `0b19c5cfff9fcf8d78bbd0424ea884aca63a921e`:

- `crates/orchard/src/circuit/halo2/{commit_ivk,note_commit,gadget}.rs`
- `crates/orchard/src/circuit/halo2/gadget/add_chip.rs`
- `crates/orchard/src/backend/halo2/constants.rs` and its fixed-base/domain files

`key_vectors.rs` preserves the owner's frozen protocol vectors from
`crates/orchard/src/test_vectors/keys.rs` at the same revision.

They retain Common/Orchard's MIT or Apache-2.0 licensing. The source was adapted
only for import paths, visibility, feature selection, and removal of payment
configuration/value-commitment helpers and owner-specific test scaffolding.
No circuit arithmetic, tables, selectors, or column assignments are intentionally
changed. The three immutable voting verification-key fingerprints are the
compatibility gate; never regenerate them to accept this move.

## Witness boundary

`witness.rs` adapts public protocol values through Orchard's documented raw FVK,
address, note-seed, rho, and signing encodings. FVK components are decoded in one
place. Scoped viewing randomness, diversification, psi, and V2/V3 trapdoor
computations inherit Common's `keys.rs`, `note.rs`, and `backend/halo2/spec.rs`
from the same revision and use their existing cryptographic primitive libraries.
The complete note commitment is recomputed from note components; an extracted
x-coordinate alone cannot recover its sign. Native Orchard commitments and
nullifiers remain independent test oracles. Signing-key correction stays in
Orchard: a zero randomizer obtains its existing canonical signing-key encoding.

All viewing and spending components remain private material. Spending scalars
permit spending and stay inside the trusted signer/prover; they must never be
logged, persisted unprotected, or sent to an untrusted helper. Synthetic
components do not establish membership, ownership, or session authenticity.

## Removal contract

The paired Common branch removes `circuit-composition`, not merely its default
activation. Acceptance requires the external consumer to compile against that
revision, native/circuit agreement and malicious-witness tests to pass, and all
three verification-key fingerprints to remain unchanged on both supported
backends. Existing payment proof fixtures in Common must remain unchanged.
