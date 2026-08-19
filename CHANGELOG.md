# Changelog

## Unreleased

## v0.10.2

### Added

- Added fine-grained upstream and Zakura cryptography facade features,
  including minimal `upstream-vct` and `zakura-vct` selections that avoid
  enabling Orchard and validator-only dependencies.

## v0.10.1

### Added

- Added an upstream-default, Zakura-opt-in cryptography dependency facade so
  downstream applications can select one coherent package family without
  maintaining separate voting-circuits releases. Validator-only extensions
  also select coherent RedPallas and transaction-primitives packages without
  adding them to lighter consumers. The validator extensions and Zakura backend
  require Rust 1.88; the default upstream backend retains Rust 1.86 support.

## v0.10.0

### Changed

- Released the exact `v0.10.0-rc.1` implementation as `v0.10.0` without
  circuit or verifying-key changes.

## v0.10.0-rc.1

### Changed

- Required the synthetic keystone note's signed value to be exactly one
  zatoshi in the delegation circuit, ensuring hardware-wallet authorization
  cannot be hidden as a zero-value spend. This changes the delegation proving
  and verifying keys.
- Reduced the vote-proof circuit domain from K=13 to K=11 by splitting its El
  Gamal operations across two ten-column tracks and its shares hash across the
  primary and a four-column Poseidon track. This changes the vote-proof
  verification key; downstream verifiers must deploy the new key before
  accepting proofs from this circuit.
- Reduced the delegation circuit to K=12 by distributing its Merkle and IMT
  paths across four shared column lanes. This changes the delegation proving
  and verifying keys and increases the proof size to 11,328 bytes; downstream
  verifiers must deploy the new key and accept proofs up to at least 11,328
  bytes.
- Used separate El Gamal randomness domains for standard and single-share vote
  layouts while preserving deterministic recovery within each layout.
- Reduced the share reveal circuit to K=10 by distributing its vote-commitment
  Merkle path across two Poseidon column configurations. This changes the share
  reveal proving and verifying keys and increases the proof size from 4,000 to
  4,992 bytes, which remains below vote-sdk's 15 KiB proof limit.

## v0.9.0

### Changed

- Released the exact `v0.9.0-rc.3` implementation as `v0.9.0` without circuit
  or verifying-key changes.

## v0.9.0-rc.3

### Changed

- Allowed compatible `orchard 0.15` and `halo2_gadgets 0.5` patch releases so
  downstream workspaces can share one Orchard dependency with librustzcash.

## v0.9.0-rc.2

### Fixed

- Updated the delegation benchmark and positive circuit fixtures to construct
  Ironwood V3 notes. Pull request CI now smoke tests the benchmark path.

### Changed

- Made `DelegationBuildError` non-exhaustive so downstream callers can handle
  future builder errors without exhaustive matching.

### Documented

- Clarified that the bundle builder rejects non-V3 notes, while the circuit
  proves membership relative to the root supplied by the verifier. An
  Ironwood-only verifier must independently authenticate `nc_root` as an
  Ironwood note commitment tree root and authenticate `nf_imt_root` at the
  same snapshot height.

## v0.9.0-rc.1

### Added

- Added `DelegationBuildError::UnsupportedNoteVersion`.

### Changed

- Updated to upstream `orchard 0.15.0`. The delegation bundle builder now
  requires Ironwood V3 delegated notes and constructs its synthetic signed and
  output notes as V3.
- Updated the delegation and vote-proof verifying keys. Downstream verifiers
  must deploy the keys from this release before accepting its proofs.

## v0.8.0

### Changed

- Updated the Orchard circuit dependency line to `orchard 0.14` and
  `halo2_gadgets =0.5.0`.
- Updated the IMT reference dependency to published `imt-tree 0.2.0`.

## v0.7.0

### Documented

- Clarified that the delegation circuit's condition 8 ("Ballot Scaling")
  proves `num_ballots ∈ { floor(v_total / BALLOT_DIVISOR), floor(v_total /
  BALLOT_DIVISOR) − 1 }` in a documented under-claim window, rather than
  exact floor-division. The remainder range check is `< 2^24` rather than
  `< BALLOT_DIVISOR`, admitting one-sided under-claim of one ballot in ~34%
  of `v_total` values. Over-claim is impossible, and the deviation is
  self-harming (the voter under-claims their own voting power). The circuit
  itself is unchanged in this release; see `src/delegation/README.md` §8
  ("Soundness scope") for the analysis and the available tightening
  approaches.

### Added

- Added a `Default` impl for `delegation::SpacedLeafImtProvider`.
- Added an `unstable-internal-api` Cargo feature that re-exposes
  `delegation::build_nullifier_list` for the in-tree IMT integration test.
  Downstream consumers should not enable this feature; the item it gates is
  not covered by the stable public API.

### Changed

- **Breaking:** Moved shared protocol items out of `vote_proof` to the crate
  root so they no longer appear to belong to a single proof module.
  - `voting_circuits::vote_proof::spend_auth_g_affine` → `voting_circuits::spend_auth_g_affine`
  - `voting_circuits::vote_proof::shares_hash` → `voting_circuits::shares_hash`
  - `voting_circuits::vote_proof::share_commitment` → `voting_circuits::share_commitment`
  - `voting_circuits::vote_proof::VOTE_COMM_TREE_DEPTH` → `voting_circuits::VOTE_COMM_TREE_DEPTH`

### Removed

- **Breaking:** `voting_circuits::vote_proof::poseidon_hash_2` is no longer
  part of the public API.
- **Breaking:** Several builder-style methods on `delegation::Circuit`
  (`from_note_unchecked`, `with_output_note`, `with_notes`,
  `with_van_comm_rand`, `with_ballot_scaling`) are no longer public.
  Construct delegation circuits through `delegation::build_delegation_bundle`
  instead.
- **Breaking:** Trimmed the public API to the surface actually consumed by
  known downstream clients. The following items remain implemented in-tree
  but are no longer reachable from outside the crate. They can be re-exposed
  in a future release if a consumer needs them.
  - Halo2 `Config` associated types: `vote_proof::Config` and
    `share_reveal::Config`. External code that drives Halo2 through the
    bundle's `circuit` field never needs to name `Config` directly — halo2
    resolves the type through the `Circuit` trait impl.
  - Bundle-builder error enum: `vote_proof::VoteProofBuildError`.
    `build_vote_proof_from_delegation` is still public; callers should
    propagate via `Display` / `?` instead of matching on variants.
  - Returned-bundle helper type: `delegation::SyntheticPaddingNoteParts`.
    The `synthetic_padding_note_parts` function remains; callers should let
    type inference name its return value.
  - `vote_proof::create_vote_proof`. The high-level wrapper
    `build_vote_proof_from_delegation` is the supported entry point;
    delegation- and share-reveal-side `create_*_proof` are still exported.
  - `share_reveal::verify_share_reveal_proof`. (The other share-reveal
    prove/verify helpers remain.)
  - Every `*_PUBLIC_OFFSET` constant across the three modules, plus the
    grouped `delegation::GOV_NULL_PUBLIC_OFFSETS` array. Assemble public
    inputs through `Instance::to_halo2_instance` instead of indexing by
    offset.
  - Delegation IMT sentinel helper `delegation::build_nullifier_list` (still
    reachable under the `unstable-internal-api` Cargo feature for the
    in-tree integration test).

### Security

- Reject Halo2 proofs that verify but leave trailing unread transcript bytes
  after delegation, vote-proof, or share-reveal verification.

### Migration

- Drop named imports of removed identifiers
  (`vote_proof::Config`, `share_reveal::Config`,
  `vote_proof::VoteProofBuildError`,
  `delegation::SyntheticPaddingNoteParts`, `vote_proof::create_vote_proof`,
  `share_reveal::verify_share_reveal_proof`, and every `*_PUBLIC_OFFSET` /
  `GOV_NULL_PUBLIC_OFFSETS` constant). Build public inputs via
  `Instance::to_halo2_instance`, propagate errors with `?` or `Display`,
  let type inference name `synthetic_padding_note_parts`'s return value,
  and prove votes via `build_vote_proof_from_delegation`.
- If your in-tree tests need `delegation::build_nullifier_list`, enable the
  `unstable-internal-api` Cargo feature on the `voting-circuits` dependency.

## v0.6.0

### Added

- Exported delegation hash helpers `rho_binding_hash`,
  `van_commitment_hash`, and `gov_null_hash` so client crates can derive the
  same signed-note rho, VAN commitment, and governance nullifiers constrained
  by the delegation circuit.

### Changed

- Revert "Domain-separate delegation governance nullifiers from IMT leaf hashes by adding
  a dedicated governance nullifier tag to the Poseidon preimage."
- Revert "Add a protocol domain tag to the delegation rho binding hash."
- Revert "Domain separate share commitments", restoring the prior per-share
  Poseidon commitment preimage shape.

## v0.5.0

### Added

- Declared Rust 1.86 as the minimum supported Rust version.
- Added `vote_proof_cached_keys()` so callers can warm and reuse the vote proof
  params, proving key, and verifying key together.
- Added `ProveError` for typed Halo2 proof creation failures.
- Exported delegation, vote-proof, and share-reveal public input offsets from
  their module roots for clients that construct or inspect proof instances.
- Exported `delegation::synthetic_padding_note_parts` and
  `delegation::SyntheticPaddingNoteParts` so off-circuit consumers (PCZT
  metadata, PIR precompute, IMT non-membership lookups) can derive the
  `(cmx, nullifier)` of a synthetic padding slot from the same construction
  the delegation builder uses in-circuit, without re-implementing it.

### Changed

- `create_delegation_proof`, `create_vote_proof`, and `create_share_reveal_proof` now return `Result<Vec<u8>, ProveError>` instead of panicking when Halo2 proof creation fails.
- Share-reveal APIs are now always available; the `share-reveal` Cargo feature was removed.
- Public input counts are exposed on each instance type for downstream
  FFI/wire decoders.

### Removed

- Removed the `mock-prover-checks` Cargo feature and its optional vote-proof builder diagnostics.
- Removed the raw field-byte verifier APIs for delegation, vote proof, and share reveal proofs.

### Security

- Domain-separate delegation governance nullifiers from IMT leaf hashes by adding
  a dedicated governance nullifier tag to the Poseidon preimage.
- Bind the delegation nullifier domain to `vote_round_id` in-circuit, preventing `dom` from being used as a free public input.
- Reject identity delegation `rk` values during public input construction instead of panicking while preparing Halo2 inputs.
- Return typed vote-proof builder errors for identity election authority keys,
  identity randomized voting public keys, and identity encrypted share points.
- Return a typed vote-proof builder error for proposal IDs outside the supported
  `[1, 15]` range instead of shifting by an unchecked caller value.
- Reject exact zero El Gamal share randomness in vote proofs.

### Changed

- Narrowed internal module, gadget, and helper visibility so only curated circuit/prover APIs remain public.
- Update delegation padding notes to use synthetic, IVK-bound padding points with custom derivation, avoiding reuse of ordinary Zcash mainnet diversified-address indices.

### Migration

- Regenerate delegation proving and verifying keys, and update downstream code
  that recomputes `gov_null_1..5`. The governance nullifier values change, but
  the IMT leaf and root format does not.
- Replace `delegation::builder::*` and `delegation::imt::*` imports with named `delegation::*` root exports.
- Replace `vote_proof::builder::*` and `vote_proof::circuit::*` imports with named `vote_proof::*` root exports.
- Replace `share_reveal::builder::*` imports with named `share_reveal::*` root exports.
- Remove `features = ["share-reveal"]` from `voting-circuits` dependency entries and refresh downstream lockfiles.
- Handle or propagate errors from all `create_*_proof` calls.
- Account for the new `VoteProofBuildError::Prove` variant in exhaustive matches.
- Shared gadget helpers such as `vote_proof::elgamal_encrypt` are no longer
  public API. `vote_proof::spend_auth_g_affine` remains public for downstream
  encryption code.
- Regenerate vote-proof proving/verifying keys after the El Gamal randomness
  hardening constraint.

## 0.4.2 - 2026-05-11

- Added an explicit `mock-prover-checks` feature for vote-proof builder diagnostics.
- Disabled runtime `MockProver` validation by default so importer builds do not pay the extra circuit synthesis and constraint-checking cost unless they opt in.

## 0.4.1 and Earlier

Release history was not tracked in this changelog before this point.
