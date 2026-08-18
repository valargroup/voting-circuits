# voting-crypto-deps

This crate provides one stable import surface for the cryptography packages
used by Valar's shielded-voting crates.

The default `upstream` feature selects the crates.io Zcash packages. Zakura
consumers select the renamed Zakura packages with:

```toml
voting-crypto-deps = { version = "0.1", default-features = false, features = ["zakura"] }
```

Exactly one backend must be enabled. The Zakura backend currently requires
Rust 1.88; the default upstream backend supports Rust 1.86.
