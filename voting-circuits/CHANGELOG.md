# Changelog

## Unreleased

### Added

- Declared Rust 1.86 as the minimum supported Rust version.
- Added `vote_proof_cached_keys()` so callers can warm and reuse the vote proof
  params, proving key, and verifying key together.
- Added `ProveError` for typed Halo2 proof creation failures.
- Exported delegation, vote-proof, and share-reveal public input offsets from
  their module roots for clients that construct or inspect proof instances.

### Changed

- `create_delegation_proof`, `create_vote_proof`, and `create_share_reveal_proof` now return `Result<Vec<u8>, ProveError>` instead of panicking when Halo2 proof creation fails.
- Share-reveal APIs are now always available; the `share-reveal` Cargo feature was removed.
- Public input counts are exposed on each instance type for downstream
  FFI/wire decoders.
- Bumped the vote-proof circuit from K=13 to K=14 after adding indexed share commitments.

### Removed

- Removed the `mock-prover-checks` Cargo feature and its optional vote-proof builder diagnostics.
- Removed the raw field-byte verifier APIs for delegation, vote proof, and share reveal proofs.

### Security

- Add a protocol domain tag to the delegation rho binding hash.
- Domain-separate delegation governance nullifiers from IMT leaf hashes by adding
  a dedicated governance nullifier tag to the Poseidon preimage.
- Bind the delegation nullifier domain to `vote_round_id` in-circuit, preventing `dom` from being used as a free public input.
- Bind `share_index` and a share-commitment domain tag into each per-share encrypted-share commitment.
- Bind `share_index` as a public input in share reveal proofs, so verifiers authenticate the externally declared reveal slot.
- Reject identity delegation `rk` values during public input construction instead of panicking while preparing Halo2 inputs.
- Return typed vote-proof builder errors for identity election authority keys,
  identity randomized voting public keys, and identity encrypted share points.
- Return a typed vote-proof builder error for proposal IDs outside the supported
  `[1, 15]` range instead of shifting by an unchecked caller value.

### Changed

- Narrowed internal module, gadget, and helper visibility so only curated circuit/prover APIs remain public.
- Update delegation padding notes to use synthetic, IVK-bound padding points with custom derivation, avoiding reuse of ordinary Zcash mainnet diversified-address indices.

### Migration

- Regenerate delegation, vote-proof, and share-reveal proving and verifying
  keys. Delegation governance nullifier values change, vote-proof keys change
  because K is now 14, and share-reveal keys change because the public input
  layout now includes `share_index`.
- Treat old vote proofs, share reveal proofs, persisted `shares_hash` values,
  and in-flight vote commitment leaves as incompatible with the indexed,
  domain-separated share commitment shape.
- Replace `delegation::builder::*` and `delegation::imt::*` imports with named `delegation::*` root exports.
- Replace `vote_proof::builder::*` and `vote_proof::circuit::*` imports with named `vote_proof::*` root exports.
- Replace `share_reveal::builder::*` imports with named `share_reveal::*` root exports.
- Remove `features = ["share-reveal"]` from `voting-circuits` dependency entries and refresh downstream lockfiles.
- Handle or propagate errors from all `create_*_proof` calls.
- Account for the new `VoteProofBuildError::Prove` variant in exhaustive matches.
- Replace raw field-byte verifier calls with typed `Instance` construction and
  `verify_delegation_proof`, `verify_vote_proof`, or
  `verify_share_reveal_proof`. Share reveal instances now carry 10 public
  fields, with `share_index` last.
- Update direct `share_commitment(blind, c1_x, c2_x, c1_y, c2_y)` calls to
  `share_commitment(share_index, blind, c1_x, c2_x, c1_y, c2_y)`.
- Include `share_index` as the final share-reveal public input when constructing
  `share_reveal::Instance` values. The `Instance::from_parts` argument order
  now matches the serialized public input order:
  `share_nullifier`, `enc_share_c1_x`, `enc_share_c1_y`, `enc_share_c2_x`,
  `enc_share_c2_y`, `proposal_id`, `vote_decision`, `vote_comm_tree_root`,
  `voting_round_id`, `share_index`.
- Shared gadget helpers such as `vote_proof::elgamal_encrypt` are no longer
  public API. `vote_proof::spend_auth_g_affine` remains public for downstream
  encryption code.

## 0.4.2 - 2026-05-11

- Added an explicit `mock-prover-checks` feature for vote-proof builder diagnostics.
- Disabled runtime `MockProver` validation by default so importer builds do not pay the extra circuit synthesis and constraint-checking cost unless they opt in.

## 0.4.1 and Earlier

Release history was not tracked in this changelog before this point.
