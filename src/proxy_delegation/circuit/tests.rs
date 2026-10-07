use super::*;
use crate::ff::{Field, PrimeField, PrimeFieldBits};
#[allow(unused_imports)]
use crate::ff::PrimeField as _PF;
use crate::gadgets::elgamal::{base_to_scalar, elgamal_encrypt, spend_auth_g_affine};
use crate::gadgets::van_integrity::van_integrity_hash;
use crate::group::{Curve, Group};
use crate::protocol_hash::poseidon_hash_2;
use crate::rand::rngs::OsRng;
use crate::shares_hash::{share_commitment, shares_hash};
use core::iter;
use std::println;
use voting_crypto_deps::halo2_gadgets::sinsemilla::primitives::CommitDomain;
use voting_crypto_deps::halo2_proofs::dev::MockProver;
use voting_crypto_deps::orchard::constants::{
    fixed_bases::COMMIT_IVK_PERSONALIZATION, L_ORCHARD_BASE,
};
use voting_crypto_deps::pasta_curves::arithmetic::CurveAffine;

fn empty_path() -> [pallas::Base; VOTE_COMM_TREE_DEPTH] {
    let mut r = [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH];
    r[0] = poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero());
    for i in 1..VOTE_COMM_TREE_DEPTH {
        r[i] = poseidon_hash_2(r[i - 1], r[i - 1]);
    }
    r
}

fn single_leaf_root(leaf: pallas::Base) -> pallas::Base {
    let p = empty_path();
    let mut c = leaf;
    for s in p.iter() {
        c = poseidon_hash_2(c, *s);
    }
    c
}

fn derive_voting_address(
    vsk: pallas::Scalar,
    nk: pallas::Base,
    rivk_v: pallas::Scalar,
) -> (pallas::Affine, pallas::Affine) {
    let ak_x = *(spend_auth_g_affine() * vsk)
        .to_affine()
        .coordinates()
        .unwrap()
        .x();
    let ivk_v = CommitDomain::new(COMMIT_IVK_PERSONALIZATION)
        .short_commit(
            iter::empty()
                .chain(ak_x.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE))
                .chain(nk.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE)),
            &rivk_v,
        )
        .unwrap();
    let g_d = pallas::Point::generator() * pallas::Scalar::from(12345u64);
    let pk_d = g_d * base_to_scalar(ivk_v).unwrap();
    (g_d.to_affine(), pk_d.to_affine())
}

/// Knobs for negative tests.
#[derive(Clone)]
pub(super) struct Fx {
    pub total: u64,
    pub shares: [u64; 16],
    /// Override for W_new (field element); default W - sum(shares).
    pub remaining_override: Option<pallas::Base>,
    pub authority_old: u64,
    pub delegate_index: u64,
    pub instance_delegate_index: Option<u64>,
    pub instance_dc_proposal: u64,
    pub dc_slot: u64,
}

impl Default for Fx {
    fn default() -> Self {
        let mut shares = [0u64; 16];
        shares[0] = 1000;
        shares[3] = 100;
        shares[7] = 7;
        Fx {
            total: 10_000,
            shares,
            remaining_override: None,
            authority_old: MAX_PROPOSAL_AUTHORITY,
            delegate_index: 1234,
            instance_delegate_index: None,
            instance_dc_proposal: 0,
            dc_slot: 3,
        }
    }
}

pub(super) struct Built {
    pub circuit: Circuit,
    pub instance: Instance,
    pub share_comms: [pallas::Base; 16],
    pub blinds: [pallas::Base; 16],
    pub c1: [(pallas::Base, pallas::Base); 16],
    pub c2: [(pallas::Base, pallas::Base); 16],
    pub round: pallas::Base,
    pub dc: pallas::Base,
}

pub(super) fn build(fx: &Fx) -> Built {
    let mut rng = OsRng;
    let vsk = pallas::Scalar::random(&mut rng);
    let nk = pallas::Base::random(&mut rng);
    let rivk = pallas::Scalar::random(&mut rng);
    let alpha = pallas::Scalar::random(&mut rng);
    let (g_d, pk_d) = derive_voting_address(vsk, nk, rivk);
    let g_d_x = *g_d.coordinates().unwrap().x();
    let pk_d_x = *pk_d.coordinates().unwrap().x();
    let round = pallas::Base::random(&mut rng);
    let rand = pallas::Base::random(&mut rng);
    let w: u64 = fx.shares.iter().sum();
    let total = pallas::Base::from(fx.total);
    let remaining = fx
        .remaining_override
        .unwrap_or(pallas::Base::from(fx.total) - pallas::Base::from(w));

    let van_old = van_integrity_hash(
        g_d_x,
        pk_d_x,
        total,
        round,
        pallas::Base::from(fx.authority_old),
        rand,
    );
    let van_new = van_integrity_hash(
        g_d_x,
        pk_d_x,
        remaining,
        round,
        pallas::Base::from(MAX_PROPOSAL_AUTHORITY),
        rand,
    );
    let root = single_leaf_root(van_old);
    let nf = poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<4>, 3, 2>::init().hash([
        nk,
        domain_tags::vote_authority_spend(),
        round,
        van_old,
    ]);
    let r_vpk = (spend_auth_g_affine() * (vsk + alpha)).to_affine();
    let ea_pk = (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine();

    let mut c1x = [pallas::Base::zero(); 16];
    let mut c1y = [pallas::Base::zero(); 16];
    let mut c2x = [pallas::Base::zero(); 16];
    let mut c2y = [pallas::Base::zero(); 16];
    let randomness: [pallas::Base; 16] =
        core::array::from_fn(|i| pallas::Base::from((i as u64 + 1) * 101));
    let blinds: [pallas::Base; 16] = core::array::from_fn(|_| pallas::Base::random(&mut rng));
    for i in 0..16 {
        let (a, b, c, d) =
            elgamal_encrypt(pallas::Base::from(fx.shares[i]), randomness[i], ea_pk).unwrap();
        c1x[i] = a;
        c2x[i] = b;
        c1y[i] = c;
        c2y[i] = d;
    }
    let sh = shares_hash(blinds, c1x, c2x, c1y, c2y);
    let share_comms: [pallas::Base; 16] =
        core::array::from_fn(|i| share_commitment(blinds[i], c1x[i], c2x[i], c1y[i], c2y[i]));
    let dc_inst = vote_commitment::vote_commitment_hash(
        round,
        sh,
        pallas::Base::from(fx.instance_dc_proposal),
        pallas::Base::from(fx.instance_delegate_index.unwrap_or(fx.delegate_index)),
    );
    let dc_true = delegation_commitment_hash(round, sh, fx.delegate_index as u32);

    let circuit = Circuit {
        vote_comm_tree_path: Value::known(empty_path()),
        vote_comm_tree_position: Value::known(0),
        vpk_g_d: Value::known(g_d),
        vpk_pk_d: Value::known(pk_d),
        total_note_value: Value::known(total),
        remaining_note_value: Value::known(remaining),
        van_comm_rand: Value::known(rand),
        vote_authority_note_old: Value::known(van_old),
        vsk: Value::known(vsk),
        rivk_v: Value::known(rivk),
        alpha_v: Value::known(alpha),
        vsk_nk: Value::known(nk),
        shares: fx.shares.map(|s| Value::known(pallas::Base::from(s))),
        enc_share_c1_x: c1x.map(Value::known),
        enc_share_c2_x: c2x.map(Value::known),
        enc_share_c1_y: c1y.map(Value::known),
        enc_share_c2_y: c2y.map(Value::known),
        share_blinds: blinds.map(Value::known),
        share_randomness: randomness.map(Value::known),
        ea_pk: Value::known(ea_pk),
        delegate_index: Value::known(pallas::Base::from(fx.delegate_index)),
        dc_slot: Value::known(pallas::Base::from(fx.dc_slot)),
    };
    let instance = Instance {
        van_nullifier: nf,
        r_vpk_x: *r_vpk.coordinates().unwrap().x(),
        r_vpk_y: *r_vpk.coordinates().unwrap().y(),
        vote_authority_note_new: van_new,
        delegation_commitment: dc_inst,
        vote_comm_tree_root: root,
        vote_comm_tree_anchor_height: pallas::Base::from(77u64),
        voting_round_id: round,
        ea_pk_x: *ea_pk.coordinates().unwrap().x(),
        ea_pk_y: *ea_pk.coordinates().unwrap().y(),
        dc_slot_nullifier: dc_slot_nullifier_hash(nk, round, rand, fx.dc_slot as u8),
    };
    Built {
        circuit,
        instance,
        share_comms,
        blinds,
        c1: core::array::from_fn(|i| (c1x[i], c1y[i])),
        c2: core::array::from_fn(|i| (c2x[i], c2y[i])),
        round,
        dc: dc_true,
    }
}

fn verify(fx: &Fx) -> bool {
    let b = build(fx);
    match MockProver::run(K, &b.circuit, vec![b.instance.to_halo2_instance()]) {
        Ok(p) => p.verify().is_ok(),
        Err(_) => false,
    }
}

#[test]
fn zkp4_valid_partial() {
    assert!(verify(&Fx::default()));
}

#[test]
fn zkp4_valid_full_delegation_successor_zero() {
    let mut fx = Fx::default();
    fx.total = 1107; // == sum(shares) -> W_new = 0
    assert!(verify(&fx));
}

#[test]
fn zkp4_valid_max_weight() {
    let mut fx = Fx::default();
    fx.total = 1 << 30; // ZKP1 upper bound
    fx.shares = [(1 << 30) / 16; 16];
    assert!(verify(&fx));
}

#[test]
fn zkp4_rejects_overdelegation_w_gt_w() {
    let mut fx = Fx::default();
    fx.total = 1000; // w = 1107 > W
    assert!(!verify(&fx));
}

#[test]
fn zkp4_rejects_zero_weight() {
    let mut fx = Fx::default();
    fx.shares = [0; 16];
    assert!(!verify(&fx));
}

#[test]
fn zkp4_rejects_non_max_authority() {
    let mut fx = Fx::default();
    fx.authority_old = MAX_PROPOSAL_AUTHORITY - 2; // voted on proposal 1 already
    assert!(!verify(&fx));
}

#[test]
fn zkp4_rejects_nonzero_pool_proposal() {
    let mut fx = Fx::default();
    fx.instance_dc_proposal = 1;
    assert!(!verify(&fx));
}

#[test]
fn zkp4_rejects_wrong_delegate_in_dc() {
    let mut fx = Fx::default();
    fx.instance_delegate_index = Some(1235);
    assert!(!verify(&fx));
}

#[test]
fn zkp4_rejects_inflated_successor() {
    let mut fx = Fx::default();
    fx.remaining_override = Some(pallas::Base::from(10_000u64)); // W_new = W, w > 0
    assert!(!verify(&fx));
}

#[test]
fn zkp4_rejects_share_overflow() {
    let mut fx = Fx::default();
    fx.shares[0] = 1 << 30;
    fx.total = (1 << 30) + 2000;
    assert!(!verify(&fx));
}

/// ZKP3 (unchanged) reveals a DC share with proposal_id = 0 and decision = delegate_index.
#[test]
fn zkp3_reveals_dc_share_with_pool_proposal_and_large_index() {
    let fx = Fx::default();
    let b = build(&fx);
    for idx in [0usize, 7] {
        let rb = crate::share_reveal::build_share_reveal(
            empty_path(),
            0,
            b.share_comms,
            b.blinds[idx],
            b.c1[idx].0,
            b.c2[idx].0,
            b.c1[idx].1,
            b.c2[idx].1,
            idx as u32,
            pallas::Base::zero(),
            pallas::Base::from(fx.delegate_index),
            b.round,
        );
        assert_eq!(rb.instance.vote_comm_tree_root, single_leaf_root(b.dc));
        let p = MockProver::run(
            crate::share_reveal::K,
            &rb.circuit,
            vec![rb.instance.to_halo2_instance()],
        )
        .unwrap();
        assert_eq!(p.verify(), Ok(()));

        // Same reveal, but claiming a different delegate, must fail against the DC root.
        let mut inst = rb.instance.clone();
        inst.vote_decision = pallas::Base::from(fx.delegate_index + 1);
        let p = MockProver::run(
            crate::share_reveal::K,
            &rb.circuit,
            vec![inst.to_halo2_instance()],
        )
        .unwrap();
        assert!(p.verify().is_err());
        // ... or claiming proposal 1.
        let mut inst = rb.instance.clone();
        inst.proposal_id = pallas::Base::one();
        let p = MockProver::run(
            crate::share_reveal::K,
            &rb.circuit,
            vec![inst.to_halo2_instance()],
        )
        .unwrap();
        assert!(p.verify().is_err());
    }
    // u32::MAX index is also accepted by ZKP3 (no range constraint on decision).
    let mut fx2 = Fx::default();
    fx2.delegate_index = u32::MAX as u64;
    let b2 = build(&fx2);
    let rb = crate::share_reveal::build_share_reveal(
        empty_path(),
        0,
        b2.share_comms,
        b2.blinds[0],
        b2.c1[0].0,
        b2.c2[0].0,
        b2.c1[0].1,
        b2.c2[0].1,
        0,
        pallas::Base::zero(),
        pallas::Base::from(u32::MAX as u64),
        b2.round,
    );
    let p = MockProver::run(
        crate::share_reveal::K,
        &rb.circuit,
        vec![rb.instance.to_halo2_instance()],
    )
    .unwrap();
    assert_eq!(p.verify(), Ok(()));
}

#[test]
#[ignore]
fn row_budget() {
    use voting_crypto_deps::halo2_proofs::dev::CircuitCost;
    let b = build(&Fx::default());
    let cost = CircuitCost::<vesta::Point, _>::measure(K, &b.circuit);
    println!("ZKP4 filled: {cost:?}");
    let cost_d = CircuitCost::<vesta::Point, _>::measure(K, &Circuit::default());
    println!("ZKP4 default: {cost_d:?}");
}

#[test]
#[ignore]
fn prove_verify_timing() {
    use std::time::Instant;
    use voting_crypto_deps::halo2_proofs::{
        pasta::EqAffine,
        plonk::{keygen_pk, keygen_vk},
        poly::commitment::Params,
    };
    let t = Instant::now();
    let params: Params<EqAffine> = Params::new(K);
    let vk = keygen_vk(&params, &Circuit::default()).unwrap();
    let pk = keygen_pk(&params, vk.clone(), &Circuit::default()).unwrap();
    println!("ZKP4 keygen: {:?}", t.elapsed());
    let pinned = format!("{:?}", vk.pinned());
    let fp = blake2b_simd::Params::new().hash_length(32).hash(pinned.as_bytes());
    println!("ZKP4 VK fingerprint (prototype): {:02x?}", fp.as_bytes());
    let mut times = vec![];
    for i in 0..8 {
        let b = build(&Fx::default());
        let pi = b.instance.to_halo2_instance();
        let t = Instant::now();
        let proof = crate::prove_error::create_proof_bytes(&params, &pk, b.circuit, &pi).unwrap();
        let dt = t.elapsed();
        times.push(dt);
        let t = Instant::now();
        crate::prove_error::verify_proof_bytes("zkp4", &params, &vk, &proof, &pi).unwrap();
        println!(
            "ZKP4 prove #{i}: {:?}  verify: {:?}  proof bytes: {}",
            dt,
            t.elapsed(),
            proof.len()
        );
    }
    times.sort();
    println!("ZKP4 median prove: {:?}", times[times.len() / 2]);
}

#[test]
#[ignore]
fn explain_negative_failures() {
    let mut cases: Vec<(&str, Fx)> = vec![];
    let mut f = Fx::default(); f.total = 1000; cases.push(("w > W", f));
    let mut f = Fx::default(); f.shares = [0; 16]; cases.push(("w = 0", f));
    let mut f = Fx::default(); f.authority_old = MAX_PROPOSAL_AUTHORITY - 2; cases.push(("auth != MAX", f));
    let mut f = Fx::default(); f.instance_dc_proposal = 1; cases.push(("DC proposal 1", f));
    let mut f = Fx::default(); f.instance_delegate_index = Some(1235); cases.push(("wrong delegate", f));
    let mut f = Fx::default(); f.remaining_override = Some(pallas::Base::from(10_000u64)); cases.push(("inflated successor", f));
    let mut f = Fx::default(); f.dc_slot = 8; cases.push(("slot 8", f));
    for (name, fx) in cases {
        let b = build(&fx);
        let p = MockProver::run(K, &b.circuit, vec![b.instance.to_halo2_instance()]).unwrap();
        match p.verify() {
            Ok(()) => println!("{name}: UNEXPECTED PASS"),
            Err(errs) => {
                let mut names: Vec<String> = errs.iter().map(|e| {
                    let s = format!("{e}");
                    s.lines().next().unwrap_or("").chars().take(140).collect()
                }).collect();
                names.sort(); names.dedup();
                println!("{name}: {} failures; e.g. {:?}", errs.len(), &names[..names.len().min(3)]);
            }
        }
    }
}

#[test]
#[ignore]
fn compare_zkp2_zkp4_timing() {
    use std::time::Instant;
    use voting_crypto_deps::halo2_proofs::{
        pasta::EqAffine,
        plonk::{keygen_pk, keygen_vk},
        poly::commitment::Params,
    };
    use voting_crypto_deps::orchard::keys::SpendingKey;
    let params: Params<EqAffine> = Params::new(K);
    let vk = keygen_vk(&params, &Circuit::default()).unwrap();
    let pk = keygen_pk(&params, vk.clone(), &Circuit::default()).unwrap();
    crate::vote_proof::vote_proof_cached_keys().unwrap();
    let sk = SpendingKey::from_bytes([7u8; 32]).unwrap();
    let ea_pk = (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine();
    let (mut t2, mut t4) = (vec![], vec![]);
    for i in 0..10 {
        let t = Instant::now();
        let b2 = crate::vote_proof::build_vote_proof_from_delegation(
            &sk, 1, 1_000 * 12_500_000, pallas::Base::from(5u64), pallas::Base::from(9u64),
            empty_path(), 0, 1, 1 + (i % 50) as u64, 1, ea_pk,
            pallas::Scalar::random(&mut OsRng), MAX_PROPOSAL_AUTHORITY, false,
        ).unwrap();
        t2.push(t.elapsed());
        let _ = b2;
        let b = build(&Fx::default());
        let pi = b.instance.to_halo2_instance();
        let t = Instant::now();
        let _p = crate::prove_error::create_proof_bytes(&params, &pk, b.circuit, &pi).unwrap();
        t4.push(t.elapsed());
    }
    t2.sort(); t4.sort();
    println!("CMP ZKP2 build+prove median {:?} (min {:?}); ZKP4 prove median {:?} (min {:?})", t2[5], t2[0], t4[5], t4[0]);
}

#[test]
#[ignore]
fn keygen_only_zkp4() {
    use voting_crypto_deps::halo2_proofs::{pasta::EqAffine, plonk::{keygen_pk, keygen_vk}, poly::commitment::Params};
    let params: Params<EqAffine> = Params::new(K);
    let vk = keygen_vk(&params, &Circuit::default()).unwrap();
    let pk = keygen_pk(&params, vk, &Circuit::default()).unwrap();
    std::hint::black_box(&pk);
}

#[test]
#[ignore]
fn keygen_only_zkp2() {
    let k = crate::vote_proof::vote_proof_cached_keys().unwrap();
    std::hint::black_box(k);
}

#[test]
#[ignore]
fn keygen_only_none() {}

#[test]
#[ignore]
fn print_frozen_vectors() {
    let dc = delegation_commitment_hash(pallas::Base::from(42u64), pallas::Base::from(100u64), 7);
    println!("VEC dc(42,100,7) = {:?}", dc.to_repr());
    let vc = vote_commitment::vote_commitment_hash(pallas::Base::from(42u64), pallas::Base::from(100u64), pallas::Base::from(0u64), pallas::Base::from(7u64));
    assert_eq!(dc, vc);
    let nf = crate::share_reveal::share_nullifier_hash(dc, pallas::Base::from(3u64), pallas::Base::from(1001u64));
    println!("VEC share_nf(dc,3,1001) = {:?}", nf.to_repr());
}

/// Runs MockProver with `dc_slot = slot` and returns the rendered failures.
fn slot_failures(slot: u64) -> Vec<String> {
    let mut fx = Fx::default();
    fx.dc_slot = slot;
    let b = build(&fx);
    let p = MockProver::run(K, &b.circuit, vec![b.instance.to_halo2_instance()]).unwrap();
    match p.verify() {
        Ok(()) => vec![],
        Err(errs) => errs.iter().map(|e| format!("{e}")).collect(),
    }
}

/// Asserts `slot` fails only at C14's 3-bit short range check.
fn assert_rejected_by_slot_range_check(slot: u64) {
    let fails = slot_failures(slot);
    assert!(!fails.is_empty(), "slot {slot} must be rejected");
    for f in &fails {
        assert!(
            f.starts_with("Lookup ") && f.contains("'Range check 3 bits'"),
            "slot {slot}: unexpected failure outside the slot range check: {f}"
        );
    }
}

#[test]
fn zkp4_valid_slot_7() {
    for slot in [0, 7] {
        assert!(slot_failures(slot).is_empty(), "slot {slot} must verify");
    }
}

#[test]
fn zkp4_rejects_slot_8_at_range_check() {
    assert_rejected_by_slot_range_check(8);
}

#[test]
fn zkp4_rejects_slot_15() {
    assert_rejected_by_slot_range_check(15);
    assert_rejected_by_slot_range_check(16);
}

/// `TAG + s` for every valid slot stays distinct from each other and from the
/// crate's other domain tags, and only moves byte 0 (no carry for s < 8).
#[test]
fn slot_tags_are_distinct() {
    let n = 1usize << DC_SLOT_BITS;
    assert_eq!(n, 8);
    let tag = proxy_delegation_slot_tag();
    let slot_tags: Vec<pallas::Base> = (0..n as u64).map(|s| tag + pallas::Base::from(s)).collect();
    for (s, t) in slot_tags.iter().enumerate() {
        let r = t.to_repr();
        assert_eq!(r[0], b'p' + s as u8);
        assert_eq!(&r[1..21], &b"proxy delegation slot"[1..]);
        assert!(r[21..].iter().all(|&x| x == 0));
    }
    let others = [
        pallas::Base::from(domain_tags::DOMAIN_VAN),
        pallas::Base::from(domain_tags::DOMAIN_VC),
        domain_tags::vote_authority_spend(),
        domain_tags::governance_authorization(),
        domain_tags::share_spend(),
    ];
    for (i, a) in slot_tags.iter().enumerate() {
        for b in &slot_tags[i + 1..] {
            assert_ne!(a, b);
        }
        assert!(!others.contains(a), "slot tag {i} collides with a protocol tag");
    }
    let (nk, round, rand) = (pallas::Base::from(1u64), pallas::Base::from(42u64), pallas::Base::from(5u64));
    let nfs: Vec<_> = (0..n as u8).map(|s| dc_slot_nullifier_hash(nk, round, rand, s)).collect();
    for (i, a) in nfs.iter().enumerate() {
        for b in &nfs[i + 1..] {
            assert_ne!(a, b);
        }
    }
}

#[test]
#[ignore]
fn debug_valid() {
    let b = build(&Fx::default());
    match MockProver::run(K, &b.circuit, vec![b.instance.to_halo2_instance()]) {
        Err(e) => println!("DBG run error: {e:?}"),
        Ok(p) => match p.verify() { Ok(()) => println!("DBG ok"), Err(es) => for e in es.iter().take(4) { println!("DBG {e}"); } },
    }
}

#[test]
#[ignore]
fn print_slot_vector() {
    let nf = dc_slot_nullifier_hash(pallas::Base::from(1u64), pallas::Base::from(42u64), pallas::Base::from(5u64), 3);
    println!("VEC slot_nf(nk=1,round=42,rand=5,slot=3) = {}", nf.to_repr().iter().map(|b| format!("{b:02x}")).collect::<String>());
}


/// Last-moment single-share DC layout: all weight in share 0, shares 1..15 zero.
#[test]
fn zkp4_single_share_layout_accepts_and_reveals() {
    // Typical partial delegation in single-share layout.
    let mut fx = Fx::default();
    fx.shares = [0u64; 16];
    fx.shares[0] = 4_000;
    assert!(verify(&fx), "single-share partial DC must verify");

    // Delegate everything at the 2^30 - 1 weight ceiling: W_new = 0.
    let mut fx = Fx::default();
    fx.total = (1u64 << 30) - 1;
    fx.shares = [0u64; 16];
    fx.shares[0] = (1u64 << 30) - 1;
    assert!(verify(&fx), "single-share full DC at max weight must verify");

    // ZKP3 reveals share 0 as (proposal 0, decision d) against the DC leaf.
    let b = build(&fx);
    let rb = crate::share_reveal::build_share_reveal(
        empty_path(),
        0,
        b.share_comms,
        b.blinds[0],
        b.c1[0].0,
        b.c2[0].0,
        b.c1[0].1,
        b.c2[0].1,
        0,
        pallas::Base::zero(),
        pallas::Base::from(fx.delegate_index),
        b.round,
    );
    assert_eq!(rb.instance.vote_comm_tree_root, single_leaf_root(b.dc));
    let p = MockProver::run(
        crate::share_reveal::K,
        &rb.circuit,
        vec![rb.instance.to_halo2_instance()],
    )
    .unwrap();
    assert_eq!(p.verify(), Ok(()));

    // Share 0 alone above 2^30 is still rejected (C9).
    let mut fx = Fx::default();
    fx.total = 1u64 << 31;
    fx.shares = [0u64; 16];
    fx.shares[0] = 1u64 << 30;
    assert!(!verify(&fx), "share 0 at 2^30 must fail the range check");
}
