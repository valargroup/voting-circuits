# Changelog

`imt-tree` is versioned independently of `voting-circuits`. Releases up to and
including `0.5.4` were published from the
[vote-nullifier-pir](https://github.com/valargroup/vote-nullifier-pir)
repository.

## Unreleased

### Changed

- The crate's source now lives in the
  [voting-circuits](https://github.com/valargroup/voting-circuits) repository,
  next to `voting-crypto-deps`. Consumers depend on the published crate as
  before.
- **Breaking:** The `upstream` (LRZ) backend moves to the `orchard` `0.16`
  generation of the librustzcash crates: `pasta_curves` `0.6` and
  `halo2_gadgets` `0.6`. Under `upstream`, `Fp` values exchanged with these
  crates are `pasta_curves` `0.6` types.
- `rust-version` is now 1.88, the toolchain the `upstream` backend supports.
  The default `zakura` backend still requires Rust 1.91.

## v0.5.4

- Publish the Zakura `2.0.0` cryptography backend through
  `voting-crypto-deps` `0.2.4`.

## v0.5.3

- Publish the Zakura `1.2.0` cryptography backend through
  `voting-crypto-deps` `0.2.3`.

## v0.5.2

- Publish the stable Zakura `1.0.0` cryptography backend through
  `voting-crypto-deps` `0.2.2`.

## v0.5.1

- Publish the Rust 1.91-compatible crate against `voting-crypto-deps` `0.2.1`.

## v0.5.0

- Update to `voting-crypto-deps` `0.2.0`, preserving the existing `zakura` and
  `upstream` feature names over the new `vct` and `lrz-vct` backend features.

## v0.4.0

- Depend on `voting-crypto-deps` `0.1.2` (Zakura RC.3) and import field traits
  via the selected pasta backend instead of a direct `ff` crate pin.

## v0.3.0

- Add mutually exclusive Zakura and upstream crypto backends, with Zakura as
  the default and minimal VCT-only dependency sets in both modes.
