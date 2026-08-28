# voting-crypto-deps

This crate provides one stable import surface for the cryptography packages
used by Valar's shielded-voting crates.

The default features select the Zakura packages:

```toml
voting-crypto-deps = "0.2"
```

LRZ consumers select the crates.io Zcash packages explicitly:

```toml
voting-crypto-deps = { version = "0.2", default-features = false, features = ["lrz"] }
```

Consumers that only need the Vote Commitment Tree (VCT) dependency graph can
select just Pasta and the Halo2 gadgets facade from the default Zakura family:

```toml
voting-crypto-deps = { version = "0.2", default-features = false, features = ["vct"] }
```

The equivalent `lrz-vct` feature selects the LRZ packages. Individual default
packages can also be selected with the clean `pasta`, `gadgets`, `poseidon`,
`proofs`, `orchard`, `sinsemilla`, and `rand` features or their `lrz-*`
counterparts. The default feature set and `lrz` aggregate include the matching
RNG crate (`rand` 0.10 Zakura, `rand` 0.8 LRZ) so consumers share one coherent
RNG trait family with the selected backend. Features from the two package
families cannot be mixed.

The default Zakura package family currently requires Rust 1.88; the LRZ family
supports Rust 1.86.

Validator crates that also parse transactions and verify RedPallas signatures
select `validator` or `lrz-validator`. These extensions include the
corresponding full backend and additionally reexport `reddsa` and
`zcash_primitives` without adding them to lighter circuit, tree, or VCT graphs.
Both validator extensions require Rust 1.88.
