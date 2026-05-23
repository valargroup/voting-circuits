# Changelog

## Unreleased

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

- Re-exported `delegation::InstanceError` at the `delegation` module root
  so callers can name the error type returned by `delegation::Instance::from_parts`
  and carried by `DelegationBuildError::Instance`.
- Re-exported `delegation::Config` at the `delegation` module root for
  consistency with `vote_proof::Config` and `share_reveal::Config`.
- Added a `Default` impl for `delegation::SpacedLeafImtProvider`.

### Changed

- **Breaking:** Moved shared protocol items out of `vote_proof` to the crate
  root so they no longer appear to belong to a single proof module.
  - `voting_circuits::vote_proof::spend_auth_g_affine` → `voting_circuits::spend_auth_g_affine`
  - `voting_circuits::vote_proof::shares_hash` → `voting_circuits::shares_hash`
  - `voting_circuits::vote_proof::VOTE_COMM_TREE_DEPTH` → `voting_circuits::VOTE_COMM_TREE_DEPTH`

### Removed

- **Breaking:** `voting_circuits::vote_proof::poseidon_hash_2` and
  `voting_circuits::vote_proof::share_commitment` are no longer part of the
  public API.
- **Breaking:** Several builder-style methods on `delegation::Circuit`
  (`from_note_unchecked`, `with_output_note`, `with_van_comm_rand`,
  `with_ballot_scaling`) are no longer public. Construct delegation circuits
  through `delegation::build_delegation_bundle` instead.

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
