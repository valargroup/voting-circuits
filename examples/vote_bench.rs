//! Production-API performance benchmark for the vote circuits.
//!
//! Measures keygen, warm proving, and verification of the encrypt-choice
//! (ZKP 1.5), cast (ZKP #2), and share-reveal (ZKP #3) proofs at M = 8.
//! Run with:
//!   RAYON_NUM_THREADS=8 /usr/bin/time -l cargo run --release --example vote_bench

use std::time::Instant;

use voting_circuits::encrypt_choice::{
    build_encrypt_choice, verify_encrypt_choice_proof, warm_encrypt_choice_keys,
};
use voting_circuits::share_reveal::{
    build_share_reveal, create_share_reveal_proof, verify_share_reveal_proof,
    warm_share_reveal_keys,
};
use voting_circuits::vote_proof::{
    build_vote_proof_from_delegation, derive_vote_authority_transition, verify_vote_bundle,
    warm_vote_proof_keys,
};
use voting_circuits::{spend_auth_g_affine, VOTE_COMM_TREE_DEPTH};
use voting_crypto_deps::orchard::keys::SpendingKey;
use voting_crypto_deps::pasta_curves::pallas;

const BALLOT_DIVISOR: u64 = 12_500_000;
const SAMPLES: usize = 5;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn main() {
    let sk = SpendingKey::from_bytes([0x42; 32]).expect("valid test spending key");
    let ea_pk = {
        use voting_circuits::group::Curve;
        (spend_auth_g_affine() * pallas::Scalar::from(42u64)).to_affine()
    };
    let total_note_value = 12_345 * BALLOT_DIVISOR;
    let van_comm_rand = pallas::Base::from(0xDEAD_u64);
    let round = pallas::Base::from(0xCAFE_u64);
    let proposal_id = 3u64;
    let decision = 3u64;
    let bucket_count = 8u64;
    let authority = voting_circuits::MAX_PROPOSAL_AUTHORITY;

    // ---- Keygen (cold, once per process) ----
    let t = Instant::now();
    warm_encrypt_choice_keys().expect("encrypt-choice keygen");
    println!(
        "keygen encrypt-choice: {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    let t = Instant::now();
    warm_vote_proof_keys().expect("vote-proof keygen");
    println!(
        "keygen cast:           {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    let t = Instant::now();
    warm_share_reveal_keys().expect("share-reveal keygen");
    println!(
        "keygen share-reveal:   {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    let transition = derive_vote_authority_transition(
        &sk,
        1,
        total_note_value,
        van_comm_rand,
        round,
        proposal_id,
        authority,
    )
    .expect("transition");

    let build_aux = || {
        build_encrypt_choice(
            &sk,
            total_note_value,
            transition.vote_authority_note_old,
            round,
            proposal_id,
            decision,
            bucket_count,
            ea_pk,
            false,
        )
        .expect("encrypt-choice build")
    };
    let build_cast = |aux: &voting_circuits::encrypt_choice::EncryptChoiceBundle| {
        build_vote_proof_from_delegation(
            &sk,
            1,
            total_note_value,
            van_comm_rand,
            round,
            [pallas::Base::from(0u64); VOTE_COMM_TREE_DEPTH],
            0,
            123,
            proposal_id,
            pallas::Scalar::from(7u64),
            authority,
            aux,
        )
        .expect("cast build")
    };

    // Warm-up proofs.
    let aux = build_aux();
    let cast = build_cast(&aux);
    println!(
        "proof sizes: encrypt-choice {:.1} KiB, cast {:.1} KiB, bundle {:.1} KiB",
        aux.proof.len() as f64 / 1024.0,
        cast.proof.len() as f64 / 1024.0,
        (aux.proof.len() + cast.proof.len()) as f64 / 1024.0
    );

    // ---- Warm proving ----
    let mut aux_times = Vec::new();
    let mut cast_times = Vec::new();
    let mut bundle_times = Vec::new();
    for _ in 0..SAMPLES {
        let t = Instant::now();
        let a = build_aux();
        aux_times.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        let c = build_cast(&a);
        cast_times.push(t.elapsed().as_secs_f64() * 1e3);
        bundle_times.push(aux_times.last().unwrap() + cast_times.last().unwrap());
        drop(c);
    }
    println!("prove encrypt-choice median: {:.0} ms", median(aux_times));
    println!("prove cast median:           {:.0} ms", median(cast_times));
    println!(
        "prove bundle (sequential):   {:.0} ms",
        median(bundle_times)
    );

    // ---- Verification ----
    let mut aux_verify = Vec::new();
    let mut bundle_verify = Vec::new();
    for _ in 0..10 {
        let t = Instant::now();
        verify_encrypt_choice_proof(&aux.proof, &aux.instance).expect("aux verify");
        aux_verify.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        verify_vote_bundle(&aux.proof, &aux.instance, &cast.proof, &cast.instance)
            .expect("bundle verify");
        bundle_verify.push(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("verify encrypt-choice median: {:.2} ms", median(aux_verify));
    println!(
        "verify bundle median:         {:.2} ms",
        median(bundle_verify)
    );

    // ---- Share reveal ----
    let reveal_path = [pallas::Base::from(0u64); VOTE_COMM_TREE_DEPTH];
    let reveal = build_share_reveal(
        reveal_path,
        0,
        aux.selected_commitments,
        aux.share_blinds[0],
        &aux.encrypted_shares[0].ciphertexts,
        0,
        cast.instance.proposal_id,
        cast.instance.voting_round_id,
        cast.instance.decision_bucket_count,
    );
    let proof = create_share_reveal_proof(reveal.circuit.clone(), &reveal.instance)
        .expect("reveal warm-up proof");
    println!(
        "proof size share-reveal: {:.1} KiB",
        proof.len() as f64 / 1024.0
    );

    let mut reveal_prove = Vec::new();
    let mut reveal_verify = Vec::new();
    for _ in 0..SAMPLES {
        let t = Instant::now();
        let p = create_share_reveal_proof(reveal.circuit.clone(), &reveal.instance)
            .expect("reveal proof");
        reveal_prove.push(t.elapsed().as_secs_f64() * 1e3);
        let t = Instant::now();
        verify_share_reveal_proof(&p, &reveal.instance).expect("reveal verify");
        reveal_verify.push(t.elapsed().as_secs_f64() * 1e3);
    }
    println!("prove share-reveal median:  {:.0} ms", median(reveal_prove));
    println!(
        "verify share-reveal median: {:.2} ms",
        median(reveal_verify)
    );
}
