# voting-circuits

Governance ZKP circuits (delegation, vote proof, share reveal) for the Zcash shielded-voting protocol.

Built with [halo2](https://github.com/zcash/halo2) on top of the upstream [Orchard](https://github.com/zcash/orchard) shielded protocol. The crate requires `std`.

## Proof flow

```
Orchard Notes ──► Delegation (ZKP 1) ──► Vote Authority Notes (VANs)
                                              │
                                              ▼
                  Vote Proof  (ZKP 2) ──► Vote Commitments + encrypted shares
                                              │
                                              ▼
                  Share Reveal (ZKP 3) ──► Revealed shares for tally
```

1. [**Delegation**](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp1-delegation-proof) spends Orchard notes and mints VANs that carry delegated voting weight.
2. [**Vote Proof**](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp2-vote-proof) spends a VAN to cast a vote, producing El Gamal-encrypted shares and a vote commitment.
3. [**Share Reveal**](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp3-vote-reveal-proof) opens a single encrypted share and proves it belongs to a registered vote commitment.

## Usage

This crate is the circuit-only side. Wallets typically don't call it directly; they consume the higher-level [`zcash_voting`](https://github.com/valargroup/zcash_voting) crate which wraps proof generation, hotkey derivation, share construction, and the HTTP wire format.

If you do want the raw gadgets for a custom prover:

```rust
use voting_circuits::vote_proof::Circuit as VoteProofCircuit;
// ... assemble public/private inputs and run halo2_proofs
```

Minimum supported Rust version: 1.86, as declared by the crate manifest.

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
├── delegation/                   # ZKP #1 — Delegation circuit (K=14)
│   ├── circuit.rs                #   15-condition halo2 circuit
│   ├── builder.rs                #   Multi-note bundle builder (up to 5 notes)
│   ├── prove.rs                  #   Prove / verify helpers
│   ├── imt.rs                    #   Indexed Merkle Tree (data structure)
│   ├── imt_circuit.rs            #   IMT non-membership proof gadget
│   └── README.md                 #   Detailed specification
│
├── vote_proof/                   # ZKP #2 — Vote Proof circuit (K=13)
│   ├── circuit.rs                #   12-condition halo2 circuit
│   ├── builder.rs                #   Builder producing VoteProofBundle
│   ├── prove.rs                  #   Prove / verify helpers
│   ├── authority_decrement.rs    #   Proposal-authority decrement gadget
│   └── README.md                 #   Detailed specification
│
└── share_reveal/                 # ZKP #3 — Share Reveal circuit (K=11)
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
| Delegation | 14 | 16 384 | 15 | [ZKP #1](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp1-delegation-proof) |
| Vote Proof | 13 | 8 192 | 12 | [ZKP #2](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp2-vote-proof) |
| Share Reveal | 11 | 2 048 | 5 | [ZKP #3](https://valargroup.gitbook.io/shielded-vote-docs/zkp-specifications/zkp3-vote-reveal-proof) |

## Dependency on `orchard`

This crate depends on upstream [`zcash/orchard`](https://github.com/zcash/orchard) `0.13.1` from crates.io, allowing compatible patch releases, with the `unstable-voting-circuits` feature enabled to expose the governance-visibility APIs the voting circuits rely on. The previous `valar-orchard` fork has been retired.

## Building

```bash
cargo build
```

## Testing

Short-running tests are the default:

```bash
cargo test
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

The long tests are slow because they synthesize Halo 2 circuits and run `MockProver` verification over the configured `K` domain (`delegation` uses K=14, `vote_proof` K=13, and `share_reveal` K=11). Some gadget stress tests are also long-running because they repeat many `MockProver` checks, for example one K=12 shares-hash test runs 16 separate prover checks. The real proof roundtrip also performs proving-key/proof generation and verification, so it is intentionally outside the default unit-test path.

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
| `orchard` | Orchard note commitment, nullifier, CommitIvk |
| `halo2_poseidon` | Poseidon hash for Merkle trees and commitments |
| `incrementalmerkletree` | Incremental Merkle tree data structure |
| `sinsemilla` | Sinsemilla hash (used via Orchard) |

## License

Dual-licensed under MIT or Apache-2.0. See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE).
