# voting-crypto-deps

This crate provides one stable import surface for the cryptography packages
used by Valar's shielded-voting crates.

The absence of the `zakura` feature selects the crates.io Zcash packages,
including when default features are disabled. Zakura consumers select the
renamed packages with:

```toml
voting-crypto-deps = { version = "0.1", features = ["zakura"] }
```

The Zakura backend currently requires Rust 1.88; the upstream mode supports
Rust 1.86. Cargo still builds the unconditional upstream dependencies in
Zakura mode, but the facade reexports only the selected Zakura types.
