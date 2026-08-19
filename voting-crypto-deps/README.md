# voting-crypto-deps

This crate provides one stable import surface for the cryptography packages
used by Valar's shielded-voting crates.

The default `upstream` feature selects the crates.io Zcash packages. Zakura
consumers select the renamed Zakura packages with:

```toml
voting-crypto-deps = { version = "0.1", default-features = false, features = ["zakura"] }
```

Consumers that only need the Vote Commitment Tree (VCT) dependency graph can
select just Pasta and the Halo2 gadgets facade:

```toml
voting-crypto-deps = { version = "0.1", default-features = false, features = ["upstream-vct"] }
```

The equivalent `zakura-vct` feature selects the Zakura packages. Individual
packages can also be selected with the `upstream-pasta`, `upstream-gadgets`,
`upstream-poseidon`, `upstream-proofs`, `upstream-orchard`, and
`upstream-sinsemilla` features or their `zakura-*` counterparts. Features from
the two package families cannot be mixed.

The Zakura package family currently requires Rust 1.88; the upstream family
supports Rust 1.86.

Validator crates that also parse transactions and verify RedPallas signatures
select `upstream-validator` or `zakura-validator`. These extensions include the
corresponding base backend and additionally reexport `reddsa` and
`zcash_primitives` without adding them to lighter circuit, tree, or VCT graphs.
Both validator extensions require Rust 1.88.
