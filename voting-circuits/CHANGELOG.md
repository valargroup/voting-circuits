# Changelog

## Unreleased

### Security

- Bind the delegation nullifier domain to `vote_round_id` in-circuit, preventing `dom` from being used as a free public input.
- Fix `verify_vote_proof_raw` to accept the vote proof circuit's 11 public inputs instead of rejecting well-formed raw verification payloads as 9-input payloads.

## 0.4.2 - 2026-05-11

- Added an explicit `mock-prover-checks` feature for vote-proof builder diagnostics.
- Disabled runtime `MockProver` validation by default so importer builds do not pay the extra circuit synthesis and constraint-checking cost unless they opt in.

## 0.4.1 and Earlier

Release history was not tracked in this changelog before this point.
