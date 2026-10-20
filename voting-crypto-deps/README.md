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

The default Orchard adapter uses `orchard::circuit::composition` for shared
components and `orchard::circuit::composition::witness` for canonical private
witness encodings. This facade decodes those encodings at the circuit boundary;
application voting policy remains in voting-circuits. LRZ retains its own native
accessors behind the same facade functions.
