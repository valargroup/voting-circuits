use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicUsize, Ordering},
};

use criterion::{criterion_group, criterion_main, Criterion};
use incrementalmerkletree::{Hashable, Level};
use voting_circuits::delegation::{
    build_delegation_bundle, DelegationBundle, ImtProvider, RealNoteInput, SpacedLeafImtProvider, K,
};
use voting_circuits::ff::{Field, PrimeField};
use voting_circuits::rand::{rngs::OsRng, Rng};
use voting_crypto_deps::halo2_proofs::{
    plonk::{self, SingleVerifier},
    transcript::{Blake2bRead, Blake2bWrite},
};
use voting_crypto_deps::orchard::{
    constants::MERKLE_DEPTH_ORCHARD as MERKLE_DEPTH,
    keys::{FullViewingKey, Scope, SpendingKey},
    note::{ExtractedNoteCommitment, Note, NoteVersion, RandomSeed, Rho},
    tree::{MerkleHashOrchard, MerklePath},
    value::NoteValue,
};
use voting_crypto_deps::pasta_curves::{pallas, vesta};

struct TrackingAllocator;

static LIVE_ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE_ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_ALLOCATED_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                LIVE_ALLOCATED_BYTES.fetch_add(new_size - layout.size(), Ordering::Relaxed);
            } else {
                LIVE_ALLOCATED_BYTES.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}

fn live_allocated_bytes() -> usize {
    LIVE_ALLOCATED_BYTES.load(Ordering::Relaxed)
}

fn measured_heap_usage_via_clone<T: Clone>(value: &T) -> usize {
    let cloned = value.clone();
    let after_clone = live_allocated_bytes();
    drop(cloned);
    let after_drop = live_allocated_bytes();
    after_clone.saturating_sub(after_drop)
}

/// The pool width halo2 will see, which decides whether it routes through the
/// prepared tables at all (at most eight effective threads, ten for `k = 11`
/// on AArch64 macOS).
fn rayon_threads() -> usize {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
        })
}

/// Zakura only: LRZ is on rand 0.8, and has no prepared tables to compare
/// against in the first place.
#[cfg(not(feature = "lrz"))]
/// A deterministic RNG, so the prepared and unprepared provers can be driven
/// with byte-identical randomness. SplitMix64; not cryptographic, and used
/// only to make one comparison reproducible.
struct FixedRng(u64);

#[cfg(not(feature = "lrz"))]
impl rand_core::TryRng for FixedRng {
    type Error = rand_core::Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.try_next_u64()? as u32)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Ok(z ^ (z >> 31))
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in dst.chunks_mut(8) {
            let word = self.try_next_u64()?.to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        Ok(())
    }
}

fn format_bytes(bytes: usize) -> String {
    format!("{bytes} bytes ({:.2} KiB)", bytes as f64 / 1024.0)
}

/// Create an Ironwood V3 note using only public APIs.
fn make_note(
    recipient: voting_crypto_deps::orchard::Address,
    value: NoteValue,
    rng: &mut impl Rng,
) -> Note {
    // Generate a random nullifier for rho.
    loop {
        let mut rho_bytes = [0u8; 32];
        rng.fill_bytes(&mut rho_bytes);
        let rho = Rho::from_bytes(&rho_bytes);
        if bool::from(rho.is_none()) {
            continue;
        }
        let rho = rho.unwrap();
        let mut rseed_bytes = [0u8; 32];
        rng.fill_bytes(&mut rseed_bytes);
        let rseed = RandomSeed::from_bytes(rseed_bytes, &rho);
        if bool::from(rseed.is_none()) {
            continue;
        }
        let note = Note::from_parts(recipient, value, rho, rseed.unwrap(), NoteVersion::V3);
        if bool::from(note.is_some()) {
            return note.unwrap();
        }
    }
}

/// Get the nullifier of a note as a pallas::Base field element.
fn nullifier_base(note: &Note, fvk: &FullViewingKey) -> pallas::Base {
    pallas::Base::from_repr(note.nullifier(fvk).to_bytes()).unwrap()
}

/// Helper: create 1-4 real note inputs with a shared Merkle tree and anchor.
fn make_real_note_inputs(
    fvk: &FullViewingKey,
    values: &[u64],
    imt_provider: &impl ImtProvider,
    rng: &mut impl Rng,
) -> (Vec<RealNoteInput>, pallas::Base) {
    let n = values.len();
    assert!((1..=4).contains(&n));

    let recipient = fvk.address_at(0u32, Scope::External);
    let mut notes = Vec::with_capacity(n);
    for &v in values {
        notes.push(make_note(recipient, NoteValue::from_raw(v), rng));
    }

    let empty_leaf = MerkleHashOrchard::empty_leaf();
    let mut leaves = [empty_leaf; 4];
    for (i, note) in notes.iter().enumerate() {
        let cmx = ExtractedNoteCommitment::from(note.commitment());
        leaves[i] = MerkleHashOrchard::from_cmx(&cmx);
    }

    let l1_0 = MerkleHashOrchard::combine(Level::from(0), &leaves[0], &leaves[1]);
    let l1_1 = MerkleHashOrchard::combine(Level::from(0), &leaves[2], &leaves[3]);
    let l2_0 = MerkleHashOrchard::combine(Level::from(1), &l1_0, &l1_1);

    let mut current = l2_0;
    for level in 2..MERKLE_DEPTH {
        let sibling = MerkleHashOrchard::empty_root(Level::from(level as u8));
        current = MerkleHashOrchard::combine(Level::from(level as u8), &current, &sibling);
    }
    let nc_root = pallas::Base::from_repr(current.to_bytes())
        .expect("MerkleHashOrchard always contains a valid field element");

    let l1 = [l1_0, l1_1];
    let mut inputs = Vec::with_capacity(n);
    for (i, note) in notes.into_iter().enumerate() {
        let mut auth_path = [MerkleHashOrchard::empty_leaf(); MERKLE_DEPTH];
        auth_path[0] = leaves[i ^ 1];
        auth_path[1] = l1[1 - (i >> 1)];
        for level in 2..MERKLE_DEPTH {
            auth_path[level] = MerkleHashOrchard::empty_root(Level::from(level as u8));
        }
        let merkle_path = MerklePath::from_parts(i as u32, auth_path);

        let nf = nullifier_base(&note, fvk);
        let imt_proof = imt_provider
            .non_membership_proof(nf)
            .expect("non_membership_proof");

        inputs.push(RealNoteInput {
            note,
            fvk: fvk.clone(),
            merkle_path,
            imt_proof,
            scope: Scope::External,
        });
    }

    (inputs, nc_root)
}

/// Build a delegation bundle with the given note values.
fn build_test_bundle(values: &[u64]) -> DelegationBundle {
    let mut rng = OsRng;
    let sk = SpendingKey::from_bytes([7; 32]).unwrap();
    let fvk: FullViewingKey = (&sk).into();
    let output_recipient = fvk.address_at(1u32, Scope::External);
    let vote_round_id = pallas::Base::random(&mut rng);
    let van_comm_rand = pallas::Base::random(&mut rng);
    let alpha = pallas::Scalar::random(&mut rng);

    let imt = SpacedLeafImtProvider::new();
    let (inputs, nc_root) = make_real_note_inputs(&fvk, values, &imt, &mut rng);

    build_delegation_bundle(
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
    .unwrap()
}

fn criterion_benchmark(c: &mut Criterion) {
    // Build a valid bundle (1 real note + 3 padded).
    let bundle = build_test_bundle(&[13_000_000]);
    let pi = bundle.instance.to_halo2_instance();
    let instance_column = pi.clone();
    let instance_columns = [&instance_column[..]];
    let instances = [&instance_columns[..]];

    // Generate params and keys.
    let params =
        voting_crypto_deps::halo2_proofs::poly::commitment::Params::<vesta::Affine>::new(K);
    let keygen_circuit = bundle.circuit.clone();
    let vk = plonk::keygen_vk(&params, &keygen_circuit).unwrap();
    let pk = plonk::keygen_pk(&params, vk.clone(), &keygen_circuit).unwrap();

    // This halo2 version does not expose key serialization APIs, so we report in-memory
    // size as: stack footprint + retained heap bytes observed for a cloned key.
    let vk_stack_bytes = std::mem::size_of_val(&vk);
    let pk_stack_bytes = std::mem::size_of_val(&pk);
    let vk_heap_bytes = measured_heap_usage_via_clone(&vk);
    let pk_heap_bytes = measured_heap_usage_via_clone(&pk);
    let vk_size_bytes = vk_stack_bytes + vk_heap_bytes;
    let pk_size_bytes = pk_stack_bytes + pk_heap_bytes;
    eprintln!(
        "delegation key sizes (in-memory) -> vk: {} [stack {}, heap {}], pk: {} [stack {}, heap {}]",
        format_bytes(vk_size_bytes),
        format_bytes(vk_stack_bytes),
        format_bytes(vk_heap_bytes),
        format_bytes(pk_size_bytes),
        format_bytes(pk_stack_bytes),
        format_bytes(pk_heap_bytes),
    );

    // Sanity-check with MockProver.
    let mock = voting_crypto_deps::halo2_proofs::dev::MockProver::run(
        K,
        &bundle.circuit,
        vec![pi.clone()],
    )
    .unwrap();
    mock.verify().expect("MockProver failed");

    // Generate one proof up-front for verify benchmarks.
    let proof_bytes = {
        let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
        plonk::create_proof(
            &params,
            &pk,
            std::slice::from_ref(&bundle.circuit),
            &instances,
            &mut OsRng,
            &mut transcript,
        )
        .unwrap();
        transcript.finalize()
    };
    eprintln!("delegation proof bytes: {}", proof_bytes.len());

    // A second SRS instance, armed with the prepared commitment tables. The
    // caches are shared with every clone of a `Params`, so this has to be an
    // independent `Params::new(K)` rather than a clone of the one above —
    // otherwise the "unprepared" arm would be silently prepared too.
    //
    // `Params::new` is deterministic for a given `K`, so the proving key built
    // over the first SRS is equally valid over this one.
    let params_prepared =
        voting_crypto_deps::halo2_proofs::poly::commitment::Params::<vesta::Affine>::new(K);
    let before_prepare = live_allocated_bytes();
    #[cfg(not(feature = "lrz"))]
    let armed = params_prepared.prepare_commitments();
    #[cfg(feature = "lrz")]
    let armed = false;
    let prepared_retained_bytes = live_allocated_bytes().saturating_sub(before_prepare);
    eprintln!(
        "delegation prepared commitments (K={K}): armed={armed}, retained {}, pool threads {}",
        format_bytes(prepared_retained_bytes),
        rayon_threads(),
    );
    if !armed {
        eprintln!(
            "delegation prepared commitments: arming declined \u{2014} the prepared arm below \
             measures the same unprepared path as the control"
        );
    }

    // Preparation must be a pure performance change: same proving key, same
    // randomness, same proof bytes. Only the route by which the commitment
    // MSMs are evaluated differs, so an unmodified verifier — and any already
    // published verifying key — stays compatible.
    //
    // Only meaningful on a pool narrow enough for halo2 to actually route
    // through the prepared tables (`RAYON_NUM_THREADS=6`). On a wide pool both
    // arms take the identical unprepared path and this passes vacuously.
    #[cfg(not(feature = "lrz"))]
    {
        let prove_with =
            |srs: &voting_crypto_deps::halo2_proofs::poly::commitment::Params<vesta::Affine>| {
                let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
                plonk::create_proof(
                    srs,
                    &pk,
                    std::slice::from_ref(&bundle.circuit),
                    &instances,
                    &mut FixedRng(0x5EED),
                    &mut transcript,
                )
                .unwrap();
                transcript.finalize()
            };
        assert_eq!(
            prove_with(&params),
            prove_with(&params_prepared),
            "preparation changed the proof bytes; it must only change how the \
             commitment MSMs are evaluated"
        );
        eprintln!("delegation prepared/unprepared proof bytes: identical under a fixed RNG");
    }

    // A prepared proof must still verify under the ordinary verifier.
    {
        let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
        plonk::create_proof(
            &params_prepared,
            &pk,
            std::slice::from_ref(&bundle.circuit),
            &instances,
            &mut OsRng,
            &mut transcript,
        )
        .unwrap();
        let prepared_proof = transcript.finalize();
        let strategy = SingleVerifier::new(&params);
        let mut transcript = Blake2bRead::init(&prepared_proof[..]);
        plonk::verify_proof(&params, &vk, strategy, &instances, &mut transcript)
            .expect("a proof built over prepared params must verify");
    }

    {
        let mut group = c.benchmark_group("delegation-keygen");
        group.sample_size(10);
        let keygen_circuit = bundle.circuit.clone();
        group.bench_function("keygen", |b| {
            b.iter(|| {
                let vk = plonk::keygen_vk(&params, &keygen_circuit).unwrap();
                let _pk = plonk::keygen_pk(&params, vk, &keygen_circuit).unwrap();
            });
        });
    }

    {
        let mut group = c.benchmark_group("delegation-proving");
        group.sample_size(10);
        let circuit = bundle.circuit.clone();
        group.bench_function("prove", |b| {
            b.iter(|| {
                let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
                plonk::create_proof(
                    &params,
                    &pk,
                    std::slice::from_ref(&circuit),
                    &instances,
                    &mut OsRng,
                    &mut transcript,
                )
                .unwrap();
                transcript.finalize()
            });
        });

        // Same circuit and key, prepared SRS. halo2 only routes through the
        // prepared tables on narrow pools, so run this with a phone-shaped
        // `RAYON_NUM_THREADS` (6) as well as the host default.
        group.bench_function("prove-prepared", |b| {
            b.iter(|| {
                let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
                plonk::create_proof(
                    &params_prepared,
                    &pk,
                    std::slice::from_ref(&circuit),
                    &instances,
                    &mut OsRng,
                    &mut transcript,
                )
                .unwrap();
                transcript.finalize()
            });
        });
    }

    {
        let mut group = c.benchmark_group("delegation-verifying");
        group.bench_function("verify", |b| {
            b.iter(|| {
                let strategy = SingleVerifier::new(&params);
                let mut transcript = Blake2bRead::init(&proof_bytes[..]);
                plonk::verify_proof(&params, &vk, strategy, &instances, &mut transcript).unwrap();
            });
        });
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = criterion_benchmark
}
criterion_main!(benches);
