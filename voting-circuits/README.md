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
use voting_circuits::vote_proof::circuit::VoteProofCircuit;
// ... assemble public/private inputs and run halo2_proofs
```

## Dependency on the Valar `orchard` fork

This crate depends on `orchard` from a Valar Group fork that adds the governance-visibility methods the voting circuits rely on. The fork lives alongside this crate in the [`valargroup/voting-circuits`](https://github.com/valargroup/voting-circuits) repository at the `orchard/` path and is consumed via git tag (not crates.io, since the name is taken by upstream `zcash/orchard`). When the relevant changes land in upstream `zcash/orchard`, this crate's dependency will flip back to the real upstream. See [the shielded-voting plan](https://github.com/valargroup/vote-sdk) for status.

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE-MIT](../LICENSE-MIT) and [LICENSE-APACHE](../LICENSE-APACHE).
