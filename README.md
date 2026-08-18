# voting-circuits

Governance ZKP circuits (delegation, vote proof, share reveal) for the Zcash shielded-voting protocol.

Built with [halo2](https://github.com/zcash/halo2) on top of the upstream
[`orchard`](https://github.com/zcash/orchard) implementation shared by Orchard
and Ironwood. The crate requires `std`.

## Proof flow

```
Ironwood Notes ──► Delegation (ZKP 1) ──► Vote Authority Notes (VANs)
                                              │
                                              ▼
                  Vote Proof  (ZKP 2) ──► Vote Commitments + encrypted shares
                                              │
                                              ▼
                  Share Reveal (ZKP 3) ──► Revealed shares for tally
```

1. [**Delegation**](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp1-delegation-proof) spends Ironwood V3 notes and mints VANs that carry delegated voting weight.
2. [**Vote Proof**](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp2-vote-proof) spends a VAN to cast a vote, producing El Gamal-encrypted shares and a vote commitment.
3. [**Share Reveal**](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp3-vote-reveal-proof) opens a single encrypted share and proves it belongs to a registered vote commitment.

## Usage

This crate is the circuit-only side. Wallets typically don't call it directly; they consume the higher-level [`zcash_voting`](https://github.com/valargroup/zcash_voting) crate which wraps proof generation, hotkey derivation, share construction, and the HTTP wire format.

If you do want the raw gadgets for a custom prover:

```rust
use voting_circuits::vote_proof::Circuit as VoteProofCircuit;
// ... assemble public/private inputs and run halo2_proofs
```

The default upstream backend supports Rust 1.86. The Zakura backend currently
requires Rust 1.88.

### Cryptography backend

The default `upstream` feature uses the crates.io Orchard, Halo2, Pasta, and
Sinsemilla packages, so existing consumers do not need to change their
dependency declaration.

Zakura consumers select the renamed package family explicitly:

```toml
voting-circuits = { version = "0.10", default-features = false, features = ["zakura"] }
```

The `upstream` and `zakura` features are mutually exclusive. Disabling default
features without selecting `zakura` also fails to compile, so a build cannot
silently omit or combine backends.

Protocol domain-separation tags are registered in [`src/domain_tags.rs`](src/domain_tags.rs). Hash-owning modules document their own preimage layout, but new tags should be added to the registry first so the encoding rule and distinctness test stay centralized.

## Package layout

```
src/
├── lib.rs                        # Crate root — re-exports the three circuits
├── circuit/                      # Shared gadgets used across circuits
│   ├── address_ownership.rs      #   CommitIvk + diversified-address integrity
│   ├── elgamal.rs                #   El Gamal encryption (vote proof condition 11)
│   ├── poseidon_merkle.rs        #   Poseidon-based Merkle path verification
│   ├── van_integrity.rs          #   VAN commitment hash (two-layer Poseidon)
│   └── vote_commitment.rs        #   Vote commitment hash
├── shares_hash.rs                # Shares-hash gadget (shared by ZKP 2 & 3)
│
├── delegation/                   # ZKP #1 — Delegation circuit (K=12)
│   ├── circuit.rs                #   15-condition halo2 circuit
│   ├── builder.rs                #   Multi-note bundle builder (up to 5 notes)
│   ├── prove.rs                  #   Prove / verify helpers
│   ├── imt.rs                    #   Indexed Merkle Tree (data structure)
│   ├── imt_circuit.rs            #   IMT non-membership proof gadget
│   └── README.md                 #   Detailed specification
│
├── vote_proof/                   # ZKP #2 — Vote Proof circuit (K=11)
│   ├── circuit.rs                #   12-condition halo2 circuit
│   ├── builder.rs                #   Builder producing VoteProofBundle
│   ├── prove.rs                  #   Prove / verify helpers
│   ├── authority_decrement.rs    #   Proposal-authority decrement gadget
│   └── README.md                 #   Detailed specification
│
└── share_reveal/                 # ZKP #3 — Share Reveal circuit (K=10)
    ├── circuit.rs                #   5-condition halo2 circuit
    ├── builder.rs                #   Builder
    └── prove.rs                  #   Prove / verify helpers

benches/
└── delegation.rs                 # Criterion benchmarks for delegation proving
```

### Shared gadgets (`circuit/`)

Reusable halo2 gadgets that appear in more than one circuit:

| Gadget | Used by | Purpose |
|--------|---------|---------|
| `address_ownership` | Delegation, Vote Proof | CommitIvk + diversified-address binding |
| `elgamal` | Vote Proof | El Gamal encryption of vote shares |
| `poseidon_merkle` | All three | Poseidon Merkle-path membership proofs |
| `van_integrity` | Delegation, Vote Proof | Two-layer Poseidon hash for VAN commitments |
| `vote_commitment` | Vote Proof, Share Reveal | Hash of `(domain, round_id, shares_hash, proposal_id, decision)` |

`shares_hash` (at crate root) computes a two-level Poseidon hash over 16 blinded share commitments and is shared by ZKP 2 and ZKP 3.

### Circuit details

| Circuit | K | Rows | Conditions | Spec |
|---------|---|------|------------|------|
| Delegation | 12 | 4 096 | 15 | [ZKP #1](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp1-delegation-proof) |
| Vote Proof | 11 | 2 048 | 12 | [ZKP #2](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp2-vote-proof) |
| Share Reveal | 10 | 1 024 | 5 | [ZKP #3](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp3-vote-reveal-proof) |

## Dependency on Orchard

The default backend uses the upstream `orchard 0.15` release. The opt-in Zakura
backend uses its API-compatible renamed package. Both enable the `circuit` and
`unstable-voting-circuits` features required by the governance proofs. The
delegation bundle builder requires Ironwood V3 notes and constructs its
synthetic signed and output notes as V3.

The V3 check is a bundle-construction policy, not a note-version bit in the
Halo2 statement. An Ironwood-only verifier must independently authenticate
`nc_root` as an Ironwood note commitment tree root and must not accept an
Orchard root supplied by the prover. It must authenticate `nf_imt_root` at the
same snapshot height.

## Building

```bash
cargo build
# Zakura backend
cargo build --no-default-features --features zakura
```

## Testing

Short-running tests are the default:

```bash
cargo test
# Zakura backend
cargo test --no-default-features --features zakura
```

Long-running tests are explicitly ignored and can be run when circuit-level coverage is needed. Skip the row-budget and cost-breakdown diagnostics for a normal regression pass:

```bash
cargo test -- --ignored --skip row_budget --skip cost_breakdown
```

To inspect circuit size diagnostics, keep `--nocapture` so the output is printed:

```bash
cargo test row_budget -- --ignored --nocapture
cargo test cost_breakdown -- --ignored --nocapture
```

The long tests are slow because they synthesize Halo 2 circuits and run
`MockProver` verification over the configured `K` domain (`delegation` uses
K=12, `vote_proof` uses K=11, and `share_reveal` uses K=10). Some gadget
stress tests are also long-running because they repeat many `MockProver`
checks, for example one K=12 shares-hash test runs 16 separate prover checks.
The real proof roundtrip also performs proving-key/proof generation and
verification, so it is intentionally outside the default unit-test path.

## Benchmarks

```bash
cargo bench   # runs delegation proving benchmarks via Criterion
```

## Key dependencies

| Crate | Role |
|-------|------|
| `halo2_proofs` | Proof system (with batch verification) |
| `halo2_gadgets` | Standard gadgets (Poseidon, Sinsemilla, ECC) |
| `pasta_curves` | Pallas / Vesta curve arithmetic |
| `orchard` | Orchard protocol primitives used by Ironwood, including note commitments, nullifiers, and CommitIvk |
| `halo2_poseidon` | Poseidon hash for Merkle trees and commitments |
| `incrementalmerkletree` | Incremental Merkle tree data structure |
| `sinsemilla` | Sinsemilla hash (used via Orchard) |

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
