use super::*;
use crate::ff::{Field, PrimeFieldBits};
use crate::gadgets::elgamal::{base_to_scalar, elgamal_encrypt, spend_auth_g_affine};
use crate::gadgets::van_integrity::van_integrity_hash;
use crate::group::{Curve, Group};
use crate::protocol_hash::poseidon_hash_2;
use crate::rand::rngs::OsRng;
use core::iter;
use std::println;
use voting_crypto_deps::halo2_gadgets::sinsemilla::primitives::CommitDomain;
use voting_crypto_deps::halo2_proofs::dev::MockProver;
use voting_crypto_deps::orchard::constants::{fixed_bases::COMMIT_IVK_PERSONALIZATION, L_ORCHARD_BASE};
use voting_crypto_deps::pasta_curves::arithmetic::CurveAffine;

fn empty_path() -> [pallas::Base; VOTE_COMM_TREE_DEPTH] {
    let mut r = [pallas::Base::zero(); VOTE_COMM_TREE_DEPTH];
    r[0] = poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero());
    for i in 1..VOTE_COMM_TREE_DEPTH { r[i] = poseidon_hash_2(r[i - 1], r[i - 1]); }
    r
}

fn build() -> (Circuit, Vec<pallas::Base>) {
    let mut rng = OsRng;
    let vsk = pallas::Scalar::random(&mut rng);
    let nk = pallas::Base::random(&mut rng);
    let rivk = pallas::Scalar::random(&mut rng);
    let alpha = pallas::Scalar::random(&mut rng);
    let ak_x = *(spend_auth_g_affine() * vsk).to_affine().coordinates().unwrap().x();
    let ivk = CommitDomain::new(COMMIT_IVK_PERSONALIZATION).short_commit(
        iter::empty().chain(ak_x.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE))
            .chain(nk.to_le_bits().iter().by_vals().take(L_ORCHARD_BASE)), &rivk).unwrap();
    let g_d = (pallas::Point::generator() * pallas::Scalar::from(12345u64)).to_affine();
    let pk_d = (g_d * base_to_scalar(ivk).unwrap()).to_affine();
    let (gx, px) = (*g_d.coordinates().unwrap().x(), *pk_d.coordinates().unwrap().x());
    let round = pallas::Base::random(&mut rng);
    let rand = pallas::Base::random(&mut rng);
    let mut w = [0u64; 16];
    for j in 0..NSLOTS { w[j] = 100 + j as u64; }
    let wsum: u64 = w.iter().sum();
    let total = 10_000u64;
    let rem = pallas::Base::from(total - wsum);
    let max = pallas::Base::from(MAX_PROPOSAL_AUTHORITY);
    let van_old = van_integrity_hash(gx, px, pallas::Base::from(total), round, max, rand);
    let van_new = van_integrity_hash(gx, px, rem, round, max, rand);
    let mut root = van_old;
    for s in empty_path() { root = poseidon_hash_2(root, s); }
    let nf = poseidon::Hash::<_, poseidon::P128Pow5T3, ConstantLength<4>, 3, 2>::init()
        .hash([nk, domain_tags::vote_authority_spend(), round, van_old]);
    let r_vpk = (spend_auth_g_affine() * (vsk + alpha)).to_affine();
    let ea_pk = (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine();
    let mut c = [[pallas::Base::zero(); 4]; 16];
    let randomness: [pallas::Base; 16] = core::array::from_fn(|i| pallas::Base::from((i as u64 + 1) * 101));
    for i in 0..16 {
        let (a, b, cc, d) = elgamal_encrypt(pallas::Base::from(w[i]), randomness[i], ea_pk).unwrap();
        c[i] = [a, b, cc, d];
    }
    let circuit = Circuit {
        vote_comm_tree_path: Value::known(empty_path()),
        vote_comm_tree_position: Value::known(0),
        vpk_g_d: Value::known(g_d), vpk_pk_d: Value::known(pk_d),
        total_note_value: Value::known(pallas::Base::from(total)),
        remaining_note_value: Value::known(rem),
        van_comm_rand: Value::known(rand),
        vote_authority_note_old: Value::known(van_old),
        vsk: Value::known(vsk), rivk_v: Value::known(rivk), alpha_v: Value::known(alpha), vsk_nk: Value::known(nk),
        shares: w.map(|s| Value::known(pallas::Base::from(s))),
        enc_share_c1_x: core::array::from_fn(|i| Value::known(c[i][0])),
        enc_share_c2_x: core::array::from_fn(|i| Value::known(c[i][1])),
        enc_share_c1_y: core::array::from_fn(|i| Value::known(c[i][2])),
        enc_share_c2_y: core::array::from_fn(|i| Value::known(c[i][3])),
        share_blinds: [Value::known(pallas::Base::zero()); 16],
        share_randomness: randomness.map(Value::known),
        ea_pk: Value::known(ea_pk),
        delegate_index: Value::known(pallas::Base::zero()),
    };
    let mut pi = vec![nf, *r_vpk.coordinates().unwrap().x(), *r_vpk.coordinates().unwrap().y(), van_new,
        pallas::Base::zero(), root, pallas::Base::from(5u64), round,
        *ea_pk.coordinates().unwrap().x(), *ea_pk.coordinates().unwrap().y()];
    for j in 0..NSLOTS {
        pi.push(pallas::Base::from(1000 + j as u64));
        pi.extend([c[j][0], c[j][2], c[j][1], c[j][3]]);
    }
    (circuit, pi)
}

#[test]
#[ignore]
fn b1_mock_rows_and_timing() {
    use std::time::Instant;
    use voting_crypto_deps::halo2_proofs::{dev::CircuitCost, pasta::EqAffine, plonk::{keygen_pk, keygen_vk}, poly::commitment::Params};
    let (c, pi) = build();
    let p = MockProver::run(K, &c, vec![pi.clone()]).unwrap();
    assert_eq!(p.verify(), Ok(()));
    println!("B1 cost: {:?}", CircuitCost::<vesta::Point, _>::measure(K, &c));
    let params: Params<EqAffine> = Params::new(K);
    let vk = keygen_vk(&params, &Circuit::default()).unwrap();
    let pk = keygen_pk(&params, vk.clone(), &Circuit::default()).unwrap();
    let mut ts = vec![];
    for _ in 0..6 {
        let (c, pi) = build();
        let t = Instant::now();
        let proof = crate::prove_error::create_proof_bytes(&params, &pk, c, &pi).unwrap();
        ts.push(t.elapsed());
        crate::prove_error::verify_proof_bytes("b1", &params, &vk, &proof, &pi).unwrap();
        println!("B1 proof bytes {} public inputs {}", proof.len(), pi.len());
    }
    ts.sort();
    println!("B1 median prove {:?}", ts[3]);
}
