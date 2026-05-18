# Changelog

## Unreleased

### Added

- Declared Rust 1.85.1 as the minimum supported Rust version.

### Changed

- Share-reveal APIs are now always available; the `share-reveal` Cargo feature was removed.

### Removed

- Removed the `mock-prover-checks` Cargo feature and its optional vote-proof builder diagnostics.

### Security

- Bind the delegation nullifier domain to `vote_round_id` in-circuit, preventing `dom` from being used as a free public input.
- Fix `verify_vote_proof_raw` to accept the vote proof circuit's 11 public inputs instead of rejecting well-formed raw verification payloads as 9-input payloads.
- Reject identity delegation `rk` values during public input construction instead of panicking while preparing Halo2 inputs.

### Changed

- Narrowed internal module, gadget, and helper visibility so only curated circuit/prover APIs remain public.

### Migration

- Replace `delegation::builder::*` and `delegation::imt::*` imports with named `delegation::*` root exports.
- Replace `vote_proof::builder::*` and `vote_proof::circuit::*` imports with named `vote_proof::*` root exports.
- Replace `share_reveal::builder::*` imports with named `share_reveal::*` root exports.
- Shared gadget helpers such as `vote_proof::spend_auth_g_affine` and `vote_proof::elgamal_encrypt` are no longer public API.

## 0.4.2 - 2026-05-11

- Added an explicit `mock-prover-checks` feature for vote-proof builder diagnostics.
- Disabled runtime `MockProver` validation by default so importer builds do not pay the extra circuit synthesis and constraint-checking cost unless they opt in.

## 0.4.1 and Earlier

Release history was not tracked in this changelog before this point.
