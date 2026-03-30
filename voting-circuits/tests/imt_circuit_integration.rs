//! Integration test: prove non-membership with K=2 punctured-range IMT providers
//! and verify through the delegation circuit using MockProver.
//!
//! Two provider implementations are tested:
//! - `SpacedLeafImtProvider`: simplified fixture provider (test-only sentinel layout)
//! - `ProductionSentinelImtAdapter`: replicates the production `prepare_nullifiers`
//!   sentinel injection path using `imt-tree` functions directly

use ff::{Field, PrimeField};
use halo2_proofs::dev::MockProver;
use incrementalmerkletree::{Hashable, Level};
use pasta_curves::pallas;
use rand::rngs::OsRng;

use orchard::{
    keys::{FullViewingKey, Scope, SpendingKey},
    note::{ExtractedNoteCommitment, Note, Rho},
    tree::{MerkleHashOrchard, MerklePath},
    value::NoteValue,
    NOTE_COMMITMENT_TREE_DEPTH,
};
use voting_circuits::delegation::{
    builder::{build_delegation_bundle, RealNoteInput},
    imt::{ImtError, ImtProofData, ImtProvider, SpacedLeafImtProvider},
};

use imt_tree::tree::{
    build_levels, build_punctured_ranges, commit_punctured_ranges, find_punctured_range_for_value,
    precompute_empty_hashes, verify_punctured_range_spans, PuncturedRange, TREE_DEPTH,
};

/// Merged circuit K value (must match the delegation circuit).
const K: u32 = 14;

/// Build a note commitment tree with up to 4 notes, returning
/// `(inputs, nc_root)` suitable for `build_delegation_bundle`.
fn make_real_note_inputs(
    fvk: &FullViewingKey,
    values: &[u64],
    scopes: &[Scope],
    imt_provider: &impl ImtProvider,
    rng: &mut impl rand::RngCore,
) -> (Vec<RealNoteInput>, pallas::Base) {
    let n = values.len();
    assert!(n >= 1 && n <= 4);
    assert_eq!(n, scopes.len());

    let mut notes = Vec::with_capacity(n);
    for (idx, &v) in values.iter().enumerate() {
        let recipient = fvk.address_at(0u32, scopes[idx]);
        let note_value = NoteValue::from_raw(v);
        let (_, _, dummy_parent) = Note::dummy(&mut *rng, None);
        let note = Note::new(
            recipient,
            note_value,
            Rho::from_nf_old(dummy_parent.nullifier(fvk)),
            &mut *rng,
        );
        notes.push(note);
    }

    // Pad leaf hashes to 4 with empty leaves.
    let empty_leaf = MerkleHashOrchard::empty_leaf();
    let mut leaves = [empty_leaf; 4];
    for (i, note) in notes.iter().enumerate() {
        let cmx = ExtractedNoteCommitment::from(note.commitment());
        leaves[i] = MerkleHashOrchard::from_cmx(&cmx);
    }

    // Build the bottom two levels of the shared tree.
    let l1_0 = MerkleHashOrchard::combine(Level::from(0), &leaves[0], &leaves[1]);
    let l1_1 = MerkleHashOrchard::combine(Level::from(0), &leaves[2], &leaves[3]);
    let l2_0 = MerkleHashOrchard::combine(Level::from(1), &l1_0, &l1_1);

    // Hash up through the remaining levels with empty subtree siblings.
    let mut current = l2_0;
    for level in 2..NOTE_COMMITMENT_TREE_DEPTH {
        let sibling = MerkleHashOrchard::empty_root(Level::from(level as u8));
        current = MerkleHashOrchard::combine(Level::from(level as u8), &current, &sibling);
    }
    // nc_root is the full 32-level Orchard note commitment tree root.
    let nc_root = pallas::Base::from_repr_vartime(current.to_bytes()).unwrap();

    let l1 = [l1_0, l1_1];
    let mut inputs = Vec::with_capacity(n);
    for (i, note) in notes.into_iter().enumerate() {
        let mut auth_path = [MerkleHashOrchard::empty_leaf(); NOTE_COMMITMENT_TREE_DEPTH];
        auth_path[0] = leaves[i ^ 1]; // sibling leaf in the same pair
        auth_path[1] = l1[1 - (i >> 1)]; // sibling pair hash
        // Levels 2..31: empty subtree roots.
        for level in 2..NOTE_COMMITMENT_TREE_DEPTH {
            auth_path[level] = MerkleHashOrchard::empty_root(Level::from(level as u8));
        }
        let merkle_path = MerklePath::from_parts(i as u32, auth_path);

        let real_nf = note.nullifier(fvk);
        let nf_base = pallas::Base::from_repr_vartime(real_nf.to_bytes()).unwrap();
        let imt_proof = imt_provider
            .non_membership_proof(nf_base)
            .expect("nullifier should be in a gap range");

        inputs.push(RealNoteInput {
            note,
            fvk: fvk.clone(),
            merkle_path,
            imt_proof,
            scope: scopes[i],
        });
    }

    (inputs, nc_root)
}

/// End-to-end test: build a SpacedLeafImtProvider (K=2 punctured ranges),
/// construct a delegation bundle with a single real note, and verify the
/// merged circuit with MockProver.
#[test]
fn k2_punctured_range_single_note_verifies_in_circuit() {
    let mut rng = OsRng;
    let imt = SpacedLeafImtProvider::new();

    let sk = SpendingKey::random(&mut rng);
    let fvk: FullViewingKey = (&sk).into();
    let output_recipient = fvk.address_at(1u32, Scope::External);
    let vote_round_id = pallas::Base::random(&mut rng);
    let van_comm_rand = pallas::Base::random(&mut rng);
    let alpha = pallas::Scalar::random(&mut rng);

    // Single note with value >= 12,500,000 (the min weight).
    let (inputs, nc_root) =
        make_real_note_inputs(&fvk, &[13_000_000], &[Scope::External], &imt, &mut rng);

    let bundle = build_delegation_bundle(
        inputs,
        &fvk,
        alpha,
        output_recipient,
        vote_round_id,
        nc_root,
        van_comm_rand,
        &imt,
        &mut rng,
        None,
    )
    .expect("build_delegation_bundle should succeed");

    let pi = bundle.instance.to_halo2_instance();
    let prover = MockProver::run(K, &bundle.circuit, vec![pi]).unwrap();
    assert_eq!(
        prover.verify(),
        Ok(()),
        "K=2 punctured-range delegation circuit should verify (single note)"
    );
}

/// Same test with 4 real notes and mixed scopes.
#[test]
fn k2_punctured_range_four_notes_verify_in_circuit() {
    let mut rng = OsRng;
    let imt = SpacedLeafImtProvider::new();

    let sk = SpendingKey::random(&mut rng);
    let fvk: FullViewingKey = (&sk).into();
    let output_recipient = fvk.address_at(1u32, Scope::External);
    let vote_round_id = pallas::Base::random(&mut rng);
    let van_comm_rand = pallas::Base::random(&mut rng);
    let alpha = pallas::Scalar::random(&mut rng);

    // 4 notes × 3,200,000 = 12,800,000 >= 12,500,000 (the min weight).
    // Mix External and Internal scopes to exercise the scope mux gate.
    let (inputs, nc_root) = make_real_note_inputs(
        &fvk,
        &[3_200_000, 3_200_000, 3_200_000, 3_200_000],
        &[
            Scope::External,
            Scope::Internal,
            Scope::Internal,
            Scope::External,
        ],
        &imt,
        &mut rng,
    );

    let bundle = build_delegation_bundle(
        inputs,
        &fvk,
        alpha,
        output_recipient,
        vote_round_id,
        nc_root,
        van_comm_rand,
        &imt,
        &mut rng,
        None,
    )
    .expect("build_delegation_bundle should succeed");

    let pi = bundle.instance.to_halo2_instance();
    let prover = MockProver::run(K, &bundle.circuit, vec![pi]).unwrap();
    assert_eq!(
        prover.verify(),
        Ok(()),
        "K=2 punctured-range delegation circuit should verify (4 notes)"
    );
}

// ── ProductionSentinelImtAdapter ───────────────────────────────────────

/// Adapter that builds a full depth-29 IMT using the same sentinel
/// construction as the production `prepare_nullifiers` path in pir-export.
///
/// This exercises the real `imt-tree` functions (`build_punctured_ranges`,
/// `commit_punctured_ranges`, `build_levels`, `verify_punctured_range_spans`)
/// end-to-end through the delegation circuit.
struct ProductionSentinelImtAdapter {
    root: pallas::Base,
    ranges: Vec<PuncturedRange>,
    levels: Vec<Vec<pallas::Base>>,
}

impl ProductionSentinelImtAdapter {
    /// Replicates the production sentinel injection from `pir-export::prepare_nullifiers`:
    /// sentinels at `k * 2^249` for k in 0..=32, plus `p-1`, padded to odd count.
    fn new(extra_nfs: &[pallas::Base]) -> Self {
        let step = pallas::Base::from(2u64).pow([249, 0, 0, 0]);
        let mut all_nfs: Vec<pallas::Base> =
            (0u64..=32).map(|k| step * pallas::Base::from(k)).collect();
        all_nfs.push(-pallas::Base::one()); // p - 1
        all_nfs.extend_from_slice(extra_nfs);
        all_nfs.sort();
        all_nfs.dedup();
        if all_nfs.len() % 2 == 0 {
            debug_assert_eq!(all_nfs[0], pallas::Base::zero());
            all_nfs.insert(1, pallas::Base::from(2u64));
        }

        let ranges = build_punctured_ranges(&all_nfs);
        verify_punctured_range_spans(&ranges).expect("all spans must be ≤ 2^250");
        let leaf_hashes = commit_punctured_ranges(&ranges);
        let empty = precompute_empty_hashes();
        let (root, levels) = build_levels(leaf_hashes, &empty, TREE_DEPTH);

        Self {
            root,
            ranges,
            levels,
        }
    }
}

impl ImtProvider for ProductionSentinelImtAdapter {
    fn root(&self) -> pallas::Base {
        self.root
    }

    fn non_membership_proof(&self, nf: pallas::Base) -> Result<ImtProofData, ImtError> {
        let idx = find_punctured_range_for_value(&self.ranges, nf)
            .ok_or_else(|| ImtError("nullifier not in any punctured range".into()))?;

        let empty = precompute_empty_hashes();
        let mut path = [pallas::Base::zero(); TREE_DEPTH];
        let mut pos = idx;
        for level in 0..TREE_DEPTH {
            let sibling_idx = pos ^ 1;
            path[level] = if sibling_idx < self.levels[level].len() {
                self.levels[level][sibling_idx]
            } else {
                empty[level]
            };
            pos >>= 1;
        }

        Ok(ImtProofData {
            root: self.root,
            nf_bounds: self.ranges[idx],
            leaf_pos: idx as u32,
            path,
        })
    }
}

/// End-to-end test using the production sentinel construction path.
///
/// Verifies that `build_punctured_ranges`, `commit_punctured_ranges`,
/// `build_levels`, and `verify_punctured_range_spans` from `imt-tree`
/// produce a valid tree whose proofs pass the delegation circuit.
#[test]
fn production_sentinel_path_verifies_in_circuit() {
    let mut rng = OsRng;

    let extra_nfs = [
        pallas::Base::from(12345u64),
        pallas::Base::from(67890u64),
    ];
    let imt = ProductionSentinelImtAdapter::new(&extra_nfs);

    let sk = SpendingKey::random(&mut rng);
    let fvk: FullViewingKey = (&sk).into();
    let output_recipient = fvk.address_at(1u32, Scope::External);
    let vote_round_id = pallas::Base::random(&mut rng);
    let van_comm_rand = pallas::Base::random(&mut rng);
    let alpha = pallas::Scalar::random(&mut rng);

    let (inputs, nc_root) =
        make_real_note_inputs(&fvk, &[13_000_000], &[Scope::External], &imt, &mut rng);

    let bundle = build_delegation_bundle(
        inputs,
        &fvk,
        alpha,
        output_recipient,
        vote_round_id,
        nc_root,
        van_comm_rand,
        &imt,
        &mut rng,
        None,
    )
    .expect("build_delegation_bundle should succeed");

    let pi = bundle.instance.to_halo2_instance();
    let prover = MockProver::run(K, &bundle.circuit, vec![pi]).unwrap();
    assert_eq!(
        prover.verify(),
        Ok(()),
        "production sentinel path should verify in delegation circuit"
    );
}

// ── PoisonedNfMidProvider ──────────────────────────────────────────────

/// IMT provider that places the target nullifier as `nf_mid` in a leaf,
/// producing a structurally valid tree (correct leaf hash and Merkle path)
/// but an unsatisfiable `q_neq` constraint: `(real_nf - nf_mid) * inv = 1`
/// becomes `0 * inv = 1` when `real_nf == nf_mid`.
struct PoisonedNfMidProvider {
    root: pallas::Base,
    ranges: Vec<PuncturedRange>,
    levels: Vec<Vec<pallas::Base>>,
    target_nf: pallas::Base,
    poisoned_leaf_idx: usize,
}

impl PoisonedNfMidProvider {
    fn new(target_nf: pallas::Base) -> Self {
        let step = pallas::Base::from(2u64).pow([249, 0, 0, 0]);
        let mut all_nfs: Vec<pallas::Base> =
            (0u64..=32).map(|k| step * pallas::Base::from(k)).collect();
        all_nfs.push(-pallas::Base::one());
        all_nfs.sort();
        all_nfs.dedup();
        if all_nfs.len() % 2 == 0 {
            debug_assert_eq!(all_nfs[0], pallas::Base::zero());
            all_nfs.insert(1, pallas::Base::from(2u64));
        }

        let mut ranges = build_punctured_ranges(&all_nfs);

        // Find the range that contains target_nf and poison its nf_mid.
        // The outer boundaries (nf_lo, nf_hi) stay the same, so the interval
        // check still passes. Only the q_neq constraint becomes unsatisfiable.
        let idx = find_punctured_range_for_value(&ranges, target_nf)
            .expect("target_nf should fall within some punctured range");
        ranges[idx][1] = target_nf;

        let leaf_hashes = commit_punctured_ranges(&ranges);
        let empty = precompute_empty_hashes();
        let (root, levels) = build_levels(leaf_hashes, &empty, TREE_DEPTH);

        Self {
            root,
            ranges,
            levels,
            target_nf,
            poisoned_leaf_idx: idx,
        }
    }

    fn proof_for_leaf(&self, idx: usize) -> ImtProofData {
        let empty = precompute_empty_hashes();
        let mut path = [pallas::Base::zero(); TREE_DEPTH];
        let mut pos = idx;
        for level in 0..TREE_DEPTH {
            let sibling_idx = pos ^ 1;
            path[level] = if sibling_idx < self.levels[level].len() {
                self.levels[level][sibling_idx]
            } else {
                empty[level]
            };
            pos >>= 1;
        }
        ImtProofData {
            root: self.root,
            nf_bounds: self.ranges[idx],
            leaf_pos: idx as u32,
            path,
        }
    }
}

impl ImtProvider for PoisonedNfMidProvider {
    fn root(&self) -> pallas::Base {
        self.root
    }

    fn non_membership_proof(&self, nf: pallas::Base) -> Result<ImtProofData, ImtError> {
        let idx = if nf == self.target_nf {
            self.poisoned_leaf_idx
        } else {
            find_punctured_range_for_value(&self.ranges, nf)
                .ok_or_else(|| ImtError("not in any punctured range".into()))?
        };
        Ok(self.proof_for_leaf(idx))
    }
}

/// Negative test: verify the circuit rejects a proof where `real_nf == nf_mid`.
///
/// The `q_neq` gate constrains `(real_nf - nf_mid) * diff_inv = 1`, which is
/// unsatisfiable when `real_nf == nf_mid` (since `0 * anything ≠ 1`). This test
/// constructs a structurally valid tree with a poisoned leaf (correct leaf hash,
/// valid Merkle path) where the note's real nullifier IS `nf_mid`, and asserts
/// the circuit rejects the proof.
///
/// In debug builds the circuit's `debug_assert!` catches the violation during
/// synthesis (panic). In release builds MockProver catches the `q_neq`
/// constraint failure. Both are valid rejections.
#[test]
fn circuit_rejects_nf_mid_equal_to_real_nf() {
    let mut rng = OsRng;

    let sk = SpendingKey::random(&mut rng);
    let fvk: FullViewingKey = (&sk).into();

    let normal_imt = SpacedLeafImtProvider::new();
    let (mut inputs, nc_root) =
        make_real_note_inputs(&fvk, &[13_000_000], &[Scope::External], &normal_imt, &mut rng);

    let nf_base =
        pallas::Base::from_repr_vartime(inputs[0].note.nullifier(&fvk).to_bytes()).unwrap();

    let poisoned_imt = PoisonedNfMidProvider::new(nf_base);
    inputs[0].imt_proof = poisoned_imt
        .non_membership_proof(nf_base)
        .expect("poisoned proof should be constructable");

    let output_recipient = fvk.address_at(1u32, Scope::External);
    let vote_round_id = pallas::Base::random(&mut rng);
    let van_comm_rand = pallas::Base::random(&mut rng);
    let alpha = pallas::Scalar::random(&mut rng);

    let bundle = build_delegation_bundle(
        inputs,
        &fvk,
        alpha,
        output_recipient,
        vote_round_id,
        nc_root,
        van_comm_rand,
        &poisoned_imt,
        &mut rng,
        None,
    )
    .expect("bundle construction should succeed");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let pi = bundle.instance.to_halo2_instance();
        let prover = MockProver::run(K, &bundle.circuit, vec![pi]).unwrap();
        prover.verify()
    }));

    match result {
        Err(_) => {} // debug_assert! caught the violation during synthesis
        Ok(verify_result) => {
            assert!(
                verify_result.is_err(),
                "circuit must reject proof where real_nf == nf_mid (q_neq unsatisfiable)"
            );
        }
    }
}
