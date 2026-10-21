# Changelog

## Unreleased

## v0.4.0

### Changed

- Select the published Common 4.0.0 cryptography family for the default backend.
  The existing LRZ backend remains available with default features disabled.
- Expose shared Orchard circuit components through `composition` and adapt
  protocol-owned canonical witness encodings at the circuit boundary.
- Key and note sampling helpers require cryptographic randomness on both backends.
- Default-backend public types now belong to Common 4 and cannot be mixed
  with the Common 3 types from facade 0.3. Wallet integrations must select
  the same Common family; `zakura-wallet-lib 0.1.0-rc9.2` supports it.
