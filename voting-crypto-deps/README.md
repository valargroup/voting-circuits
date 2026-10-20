# voting-crypto-deps

This crate provides one stable import surface for the cryptography packages
used by Valar's shielded-voting crates.

The default features select the Zakura packages:

```toml
voting-crypto-deps = "0.3"
```

LRZ consumers select the crates.io Zcash packages explicitly:

```toml
voting-crypto-deps = { version = "0.3", default-features = false, features = ["lrz"] }
```

Consumers that only need the Vote Commitment Tree (VCT) dependency graph can
select just Pasta and the Halo2 gadgets facade from the default Zakura family:

```toml
voting-crypto-deps = { version = "0.3", default-features = false, features = ["vct"] }
```

The equivalent `lrz-vct` feature selects the LRZ packages. Individual default
packages can also be selected with the clean `pasta`, `gadgets`, `poseidon`,
`proofs`, `orchard`, `sinsemilla`, and `rand` features or their `lrz-*`
counterparts. The default feature set and `lrz` aggregate include the matching
RNG crate (`rand` 0.10) so consumers share one coherent
RNG trait family with the selected backend. Features from the two package
families cannot be mixed.

Both the default Zakura package family and the LRZ family require Rust 1.97.

Validator crates that also parse transactions and verify RedPallas signatures
select `validator` or `lrz-validator`. These extensions include the
corresponding full backend and additionally reexport `reddsa` and
`zcash_primitives` without adding them to lighter circuit, tree, or VCT graphs.
Both validator extensions require Rust 1.97.

Selecting `orchard` or `lrz-orchard` also exposes the matching
`incrementalmerkletree`, `zip32`, and `zcash_note_encryption` APIs. Use these
exports when passing witnesses, address indexes, or encrypted notes to Orchard
so those types belong to the selected backend.

### Voting circuit composition

Selecting `orchard` exposes Common's opt-in `circuit-composition` interface through
`voting_crypto_deps::composition`; `lrz-orchard` supplies the corresponding existing
LRZ exports at the same path. `CircuitValue` decodes protocol-owned canonical
encodings only at the Halo2 witness boundary. It does not replace Orchard's
protocol codecs. The note helpers delegate commitment randomness and
sign-corrected spending-key derivations to the selected Orchard backend.
Spending and viewing witnesses remain secret material and must not be logged.

Helpers that sample keys, notes, or padding paths additionally require `rand`
or `lrz-rand`. Padding data does not establish membership in an on-chain tree.
