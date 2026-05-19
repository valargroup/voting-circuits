# voting-circuits

Governance ZKP circuits for the Zcash shielded-voting protocol.

Contains the three Halo 2 circuits wallets use to participate in a vote round without revealing which notes back their ballot:

- **ZKP1 / delegation** — prove ownership of unspent Orchard notes at the snapshot height and bind the delegated weight to a fresh voting hotkey (creates a VAN on the vote commitment tree).
- **ZKP2 / vote commitment** — consume a VAN and produce a `(proposal_id, vote_decision)` commitment leaf.
- **ZKP3 / share reveal** — open one of the 16 encrypted shares of a vote in a way that's verifiable without revealing the whole ballot.

## Usage

This crate is the circuit-only side. Wallets typically don't call it directly; they consume the higher-level [`zcash_voting`](https://github.com/valargroup/zcash_voting) crate which wraps proof generation, hotkey derivation, share construction, and the HTTP wire format.

If you do want the raw gadgets for a custom prover:

```rust
use voting_circuits::vote_proof::Circuit as VoteProofCircuit;
// ... assemble public/private inputs and run halo2_proofs
```

Minimum supported Rust version: 1.86, as declared by the crate manifest.

Protocol domain-separation tags are registered in
[`src/domain_tags.rs`](src/domain_tags.rs). Hash-owning modules document their
own preimage layout, but new tags should be added to the registry first so the
encoding rule and distinctness test stay centralized.

## Dependency on `orchard`

This crate now depends on upstream [`zcash/orchard`](https://github.com/zcash/orchard) `0.13`, with the `unstable-voting-circuits` feature enabled to expose the governance-visibility APIs the voting circuits rely on (the `valar-orchard` fork has been retired).

While the upstream PRs are landing it is pinned via `[patch.crates-io]` to the `valargroup/orchard` `valar/0.13-spend-auth-g` branch (tracked by [valargroup/orchard PR #19](https://github.com/valargroup/orchard/pull/19)), which carries `orchard 0.13.0` plus cherry-picks of [zcash/orchard #489](https://github.com/zcash/orchard/pull/489) (SpendAuthG fixed-base multiplication) and [zcash/orchard #495](https://github.com/zcash/orchard/pull/495) (`NoteValue::ZERO` public associated constant). Once both upstream PRs land and an `orchard 0.14` ships, the pin will collapse to the published crate.

## Testing

Short-running tests are the default:

```sh
cargo test
```

Long-running tests are explicitly ignored and can be run when circuit-level
coverage is needed. Skip the row-budget diagnostics for a normal regression
pass:

```sh
cargo test -- --ignored --skip row_budget --skip cost_breakdown
```

To inspect circuit size diagnostics, keep `--nocapture` so the row-budget
output is printed:

```sh
cargo test row_budget -- --ignored --nocapture
cargo test cost_breakdown -- --ignored --nocapture
```

The long tests are slow because they synthesize Halo 2 circuits and run
`MockProver` verification over the configured `K` domain (`delegation` uses
K=14, `vote_proof` K=13, and `share_reveal` K=11). Some gadget stress tests are
also long-running because they run larger configured `K` domains. The real proof
roundtrip also performs proving-key/proof generation and verification, so it is
intentionally outside the default unit-test path.

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE-MIT](../LICENSE-MIT) and [LICENSE-APACHE](../LICENSE-APACHE).
