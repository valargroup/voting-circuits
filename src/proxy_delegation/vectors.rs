//! Golden vectors for ZKP #4, `test-vectors/proxy_delegation_circuit_v1.json`
//! (proxy delegation contracts v1, §13).
//!
//! `zkp4_instance` pins how [`Instance`] packs the public inputs, and
//! `share_secrets` pins the share secrets that [`build_proxy_delegation_proof`]
//! derives from `dc_seed` (contracts v1, §10). [`generate`] builds every case
//! from fixed seeds. The tests check the committed file case by case against
//! the code and byte for byte against the generator. Rewriting the file is a
//! contract change:
//!
//! ```text
//! VOTING_CIRCUITS_WRITE_VECTORS=1 cargo test --locked --lib write_circuit_vectors -- --ignored
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use voting_crypto_deps::orchard::keys::SpendingKey;
use voting_crypto_deps::pasta_curves::{arithmetic::CurveAffine, pallas};

use super::{
    build_proxy_delegation_proof, builder::proxy_share_prf, Instance, ProxyDelegationBundle,
    ShareLayout,
};
use crate::{
    domain_tags,
    ff::{FromUniformBytes, PrimeField},
    gadgets::elgamal::spend_auth_g_affine,
    group::Curve,
    params::{SHARE_VALUE_LIMIT, VOTE_COMM_TREE_DEPTH},
    protocol_hash::poseidon_hash_2,
};

/// The committed vector file.
const COMMITTED_VECTORS: &str = include_str!("../../test-vectors/proxy_delegation_circuit_v1.json");

/// The committed file's SHA-256, frozen by contracts v1, §13.
const FROZEN_SHA256: &str = "6293658ba866a7738f5d34ae5364559938b8f3b3b8175bbd77ac6b3431e5b883";

/// Where [`write_circuit_vectors`] writes the vector file.
const VECTOR_FILE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test-vectors/proxy_delegation_circuit_v1.json"
);

/// Setting this to `1` lets [`write_circuit_vectors`] rewrite the file.
const WRITE_VECTORS_ENV: &str = "VOTING_CIRCUITS_WRITE_VECTORS";

/// BLAKE2b personalization of the seeded inputs (see [`seeded`]).
const SEED_PERSONALIZATION: &[u8; 16] = b"zvc-vectors-v1__";

/// The `(domain, index)` pairs whose PRF outputs every `share_secrets` case
/// records: the first and last share of the El Gamal, blind and single-share
/// El Gamal domains, the shuffle seed, and the remainder weights of slots 9
/// and 15.
const PRF_INPUTS: [(u8, u8); 9] = [
    (domain_tags::PROXY_PRF_DOMAIN_ELGAMAL, 0),
    (domain_tags::PROXY_PRF_DOMAIN_ELGAMAL, 15),
    (domain_tags::PROXY_PRF_DOMAIN_BLIND, 0),
    (domain_tags::PROXY_PRF_DOMAIN_BLIND, 15),
    (domain_tags::PROXY_PRF_DOMAIN_SHUFFLE, 0),
    (domain_tags::PROXY_PRF_DOMAIN_REMAINDER, 9),
    (domain_tags::PROXY_PRF_DOMAIN_REMAINDER, 15),
    (domain_tags::PROXY_PRF_DOMAIN_ELGAMAL_SINGLE_SHARE, 0),
    (domain_tags::PROXY_PRF_DOMAIN_ELGAMAL_SINGLE_SHARE, 15),
];

/// The vector file.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VectorFile {
    format: String,
    contract: String,
    generator: String,
    zkp4_instance: Vec<Case<InstanceInputs, InstanceExpected>>,
    share_secrets: Vec<Case<ShareSecretsInputs, ShareSecretsExpected>>,
}

/// One named case.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case<I, E> {
    name: String,
    inputs: I,
    expected: E,
}

/// The [`Instance::from_parts`] arguments, in its order.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InstanceInputs {
    van_nullifier: String,
    r_vpk_x: String,
    r_vpk_y: String,
    vote_authority_note_new: String,
    delegation_commitment: String,
    vote_comm_tree_root: String,
    vote_comm_tree_anchor_height: String,
    voting_round_id: String,
    ea_pk_x: String,
    ea_pk_y: String,
    dc_slot_nullifier: String,
}

/// The output of [`Instance::to_halo2_instance`].
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InstanceExpected {
    public_inputs: [String; Instance::NUM_PUBLIC_INPUTS],
}

/// A DC's share secret inputs, and the PRF inputs to record.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShareSecretsInputs {
    dc_seed: String,
    weight_ballots: u64,
    layout: u8,
    prf_inputs: Vec<PrfInput>,
}

/// A PRF domain byte and share index.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrfInput {
    domain: u8,
    index: u8,
}

/// A DC's share secrets, in share order, and the recorded PRF outputs.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShareSecretsExpected {
    prf_outputs: Vec<String>,
    shares: [u64; 16],
    elgamal_randomness: [String; 16],
    blinds: [String; 16],
}

impl InstanceInputs {
    /// Records `instance`'s fields.
    fn of(instance: &Instance) -> Self {
        InstanceInputs {
            van_nullifier: base_hex(instance.van_nullifier),
            r_vpk_x: base_hex(instance.r_vpk_x),
            r_vpk_y: base_hex(instance.r_vpk_y),
            vote_authority_note_new: base_hex(instance.vote_authority_note_new),
            delegation_commitment: base_hex(instance.delegation_commitment),
            vote_comm_tree_root: base_hex(instance.vote_comm_tree_root),
            vote_comm_tree_anchor_height: base_hex(instance.vote_comm_tree_anchor_height),
            voting_round_id: base_hex(instance.voting_round_id),
            ea_pk_x: base_hex(instance.ea_pk_x),
            ea_pk_y: base_hex(instance.ea_pk_y),
            dc_slot_nullifier: base_hex(instance.dc_slot_nullifier),
        }
    }

    /// Constructs the instance through [`Instance::from_parts`].
    fn to_instance(&self) -> Instance {
        Instance::from_parts(
            parse_base(&self.van_nullifier),
            parse_base(&self.r_vpk_x),
            parse_base(&self.r_vpk_y),
            parse_base(&self.vote_authority_note_new),
            parse_base(&self.delegation_commitment),
            parse_base(&self.vote_comm_tree_root),
            parse_base(&self.vote_comm_tree_anchor_height),
            parse_base(&self.voting_round_id),
            parse_base(&self.ea_pk_x),
            parse_base(&self.ea_pk_y),
            parse_base(&self.dc_slot_nullifier),
        )
    }
}

/// Where the spent VAN sits in the vote commitment tree.
#[derive(Clone, Copy)]
enum Anchor {
    /// Alone at position 0, at anchor height 0, as the chain anchors a batch
    /// with a registration (contracts v1, §5.1).
    Registration,
    /// At `position` among seeded siblings, at anchor height `height`.
    Height { position: u32, height: u32 },
}

/// The delegation inputs of one builder run.
struct Delegation {
    van_ballots: u64,
    delegated_ballots: u64,
    delegate_index: u32,
    dc_slot: u8,
    dc_seed: [u8; 32],
    layout: ShareLayout,
    anchor: Anchor,
}

/// Builds the vector file from its fixed seeds.
fn generate() -> VectorFile {
    VectorFile {
        format: "proxy_delegation_circuit_v1".to_owned(),
        contract: "Proxy delegation contracts, version 1, section 13".to_owned(),
        generator: "voting-circuits: src/proxy_delegation/vectors.rs, written by \
                    `VOTING_CIRCUITS_WRITE_VECTORS=1 cargo test --locked --lib \
                    write_circuit_vectors -- --ignored`. Seeded bytes are BLAKE2b with an \
                    N-byte output (personal = \"zvc-vectors-v1__\") of a label naming the \
                    section, case and field; a seeded field element or scalar is \
                    from_uniform_bytes of 64 such bytes. The rest are literals in that file. \
                    Field elements are canonical 32-byte little-endian encodings. \
                    zkp4_instance inputs are the 11 Instance::from_parts arguments and \
                    expected.public_inputs is Instance::to_halo2_instance. The \
                    *_from_builder cases are build_proxy_delegation_proof outputs; \
                    registration_from_builder anchors at height 0 to the tree holding only \
                    its VAN, as a batch with a registration does. \
                    no_registration_extreme_values uses the largest canonical encodings and \
                    anchor height 2^64 - 1. share_secrets come from \
                    build_proxy_delegation_proof, whose other inputs do not affect them: the \
                    shares after the split and shuffle, their El Gamal randomness and their \
                    commitment blinds, in share order. prf_outputs[i] is the contracts v1 \
                    section 10 PRF at prf_inputs[i]; domains 16 to 20 are 0x10 to 0x14."
            .to_owned(),
        zkp4_instance: zkp4_instance_cases(),
        share_secrets: share_secrets_cases(),
    }
}

/// The packing cases: builder outputs with and without a registration, and
/// [`extreme_instance`].
fn zkp4_instance_cases() -> Vec<Case<InstanceInputs, InstanceExpected>> {
    let from_builder = |name: &str, delegation: Delegation| {
        let instance = build(&format!("zkp4_instance/{name}"), &delegation).instance;
        case(name, InstanceInputs::of(&instance), pack)
    };
    vec![
        from_builder(
            "registration_from_builder",
            Delegation {
                van_ballots: 25_000,
                delegated_ballots: 12_345,
                delegate_index: 17,
                dc_slot: 0,
                dc_seed: seeded("zkp4_instance/registration_from_builder/dc_seed"),
                layout: ShareLayout::Standard,
                anchor: Anchor::Registration,
            },
        ),
        // A full delegation from the last slot, so the successor VAN keeps 0.
        from_builder(
            "no_registration_from_builder",
            Delegation {
                van_ballots: 2_000,
                delegated_ballots: 2_000,
                delegate_index: 1,
                dc_slot: 7,
                dc_seed: seeded("zkp4_instance/no_registration_from_builder/dc_seed"),
                layout: ShareLayout::SingleShare,
                anchor: Anchor::Height {
                    position: 48_211,
                    height: 3_284_117,
                },
            },
        ),
        case(
            "no_registration_extreme_values",
            InstanceInputs::of(&extreme_instance()),
            pack,
        ),
    ]
}

/// An instance at the top of the canonical encodings: `r_vpk = (-1, 2)` and
/// `ea_pk = (-1, -2)`, the two points with `x = p - 1`; the other field
/// elements `p - 3` to `p - 8` in order; and anchor height `2^64 - 1`, the
/// largest the chain's `uint64` carries.
fn extreme_instance() -> Instance {
    let minus = |k: u64| -pallas::Base::from(k);
    let (x, y) = (minus(1), pallas::Base::from(2));
    for point_y in [y, -y] {
        assert!(bool::from(pallas::Affine::from_xy(x, point_y).is_some()));
    }
    Instance::from_parts(
        minus(3),
        x,
        y,
        minus(4),
        minus(5),
        minus(6),
        pallas::Base::from(u64::MAX),
        minus(7),
        x,
        -y,
        minus(8),
    )
}

/// The share secret cases: the edges of the standard split, and both edges
/// of the single-share layout.
fn share_secrets_cases() -> Vec<Case<ShareSecretsInputs, ShareSecretsExpected>> {
    [
        // The least weight, one share of 1.
        ("standard_minimum_weight", 1, ShareLayout::Standard),
        // Nine 10s fill the denomination slots, and the remainder of 3 is
        // below the 7 remainder slots, so it takes no PRF weights.
        ("standard_small_remainder", 93, ShareLayout::Standard),
        // Four 1,000s and five 100s, then 300 weighted over slots 9 to 15.
        ("standard_weighted_remainder", 4_800, ShareLayout::Standard),
        // 2^30, the most a VAN holds: nine 10,000,000s and a weighted remainder.
        (
            "standard_max_weight",
            SHARE_VALUE_LIMIT,
            ShareLayout::Standard,
        ),
        ("single_share", 4_800, ShareLayout::SingleShare),
        // 2^30 - 1, the most one share holds.
        (
            "single_share_max_weight",
            SHARE_VALUE_LIMIT - 1,
            ShareLayout::SingleShare,
        ),
    ]
    .into_iter()
    .map(|(name, weight_ballots, layout)| {
        let inputs = ShareSecretsInputs {
            dc_seed: hex::encode(seeded::<32>(&format!("share_secrets/{name}/dc_seed"))),
            weight_ballots,
            layout: layout as u8,
            prf_inputs: PRF_INPUTS
                .iter()
                .map(|&(domain, index)| PrfInput { domain, index })
                .collect(),
        };
        let vector = case(name, inputs, derive_share_secrets);
        assert_eq!(
            vector.expected.shares.iter().sum::<u64>(),
            weight_ballots,
            "share_secrets/{name}"
        );
        vector
    })
    .collect()
}

/// A case whose expected outputs `compute` derives from its inputs.
fn case<I, E>(name: &str, inputs: I, compute: impl Fn(&I) -> E) -> Case<I, E> {
    let expected = compute(&inputs);
    Case {
        name: name.to_owned(),
        inputs,
        expected,
    }
}

/// Packs a case's instance with [`Instance::to_halo2_instance`].
fn pack(inputs: &InstanceInputs) -> InstanceExpected {
    InstanceExpected {
        public_inputs: inputs
            .to_instance()
            .to_halo2_instance()
            .into_iter()
            .map(base_hex)
            .collect::<Vec<_>>()
            .try_into()
            .expect("an instance packs NUM_PUBLIC_INPUTS field elements"),
    }
}

/// Derives a case's share secrets with the builder, and its PRF outputs with
/// the builder's PRF.
fn derive_share_secrets(inputs: &ShareSecretsInputs) -> ShareSecretsExpected {
    let dc_seed = parse_bytes(&inputs.dc_seed);
    let layout = match inputs.layout {
        0 => ShareLayout::Standard,
        1 => ShareLayout::SingleShare,
        other => panic!("layout {other} is not 0 or 1"),
    };
    let bundle = build(
        "share_secrets",
        &Delegation {
            van_ballots: SHARE_VALUE_LIMIT,
            delegated_ballots: inputs.weight_ballots,
            delegate_index: 1,
            dc_slot: 0,
            dc_seed,
            layout,
            anchor: Anchor::Registration,
        },
    );
    ShareSecretsExpected {
        prf_outputs: inputs
            .prf_inputs
            .iter()
            .map(|input| hex::encode(proxy_share_prf(&dc_seed, input.domain, input.index)))
            .collect(),
        shares: bundle
            .encrypted_shares
            .each_ref()
            .map(|share| share.plaintext_value),
        elgamal_randomness: bundle
            .encrypted_shares
            .each_ref()
            .map(|share| hex::encode(share.randomness)),
        blinds: bundle.share_blinds.map(base_hex),
    }
}

/// Runs the builder on `delegation`, with the spending key, VAN randomness,
/// round, `ea_pk`, `alpha_v` and any tree siblings seeded under `label`.
fn build(label: &str, delegation: &Delegation) -> ProxyDelegationBundle {
    let seed_label = |field: &str| format!("{label}/{field}");
    let sk =
        SpendingKey::from_bytes(seeded(&seed_label("sk"))).expect("a seeded spending key is valid");
    let (path, position, anchor_height) = match delegation.anchor {
        Anchor::Registration => (single_leaf_path(), 0, 0),
        Anchor::Height { position, height } => (
            core::array::from_fn(|level| seeded_base(&seed_label(&format!("path[{level}]")))),
            position,
            height,
        ),
    };
    build_proxy_delegation_proof(
        &sk,
        1,
        delegation.van_ballots,
        seeded_base(&seed_label("van_comm_rand")),
        seeded_base(&seed_label("voting_round_id")),
        path,
        position,
        anchor_height,
        (spend_auth_g_affine() * seeded_scalar(&seed_label("ea_sk"))).to_affine(),
        seeded_scalar(&seed_label("alpha_v")),
        delegation.delegate_index,
        delegation.delegated_ballots,
        delegation.dc_slot,
        delegation.dc_seed,
        delegation.layout,
    )
    .expect("the vector inputs are valid")
}

/// Siblings of position 0 in an otherwise empty vote commitment tree, whose
/// empty leaf is `Poseidon(0, 0)`.
fn single_leaf_path() -> [pallas::Base; VOTE_COMM_TREE_DEPTH] {
    let mut path =
        [poseidon_hash_2(pallas::Base::zero(), pallas::Base::zero()); VOTE_COMM_TREE_DEPTH];
    for level in 1..VOTE_COMM_TREE_DEPTH {
        path[level] = poseidon_hash_2(path[level - 1], path[level - 1]);
    }
    path
}

/// `N` bytes seeded by `label`: BLAKE2b with an `N`-byte output and
/// [`SEED_PERSONALIZATION`], over the label. `N` is at most 64.
fn seeded<const N: usize>(label: &str) -> [u8; N] {
    blake2b_simd::Params::new()
        .hash_length(N)
        .personal(SEED_PERSONALIZATION)
        .hash(label.as_bytes())
        .as_bytes()
        .try_into()
        .expect("the hash has N bytes")
}

/// A field element seeded by `label`.
fn seeded_base(label: &str) -> pallas::Base {
    pallas::Base::from_uniform_bytes(&seeded::<64>(label))
}

/// A scalar seeded by `label`.
fn seeded_scalar(label: &str) -> pallas::Scalar {
    pallas::Scalar::from_uniform_bytes(&seeded::<64>(label))
}

/// The canonical encoding of `value`, as hex.
fn base_hex(value: pallas::Base) -> String {
    hex::encode(value.to_repr())
}

/// Decodes lowercase hex of exactly `N` bytes.
fn parse_bytes<const N: usize>(hex: &str) -> [u8; N] {
    let bytes = hex::decode(hex).unwrap_or_else(|error| panic!("invalid hex {hex:?}: {error}"));
    assert_eq!(hex::encode(&bytes), hex, "hex must be lowercase");
    bytes
        .try_into()
        .unwrap_or_else(|bytes: Vec<u8>| panic!("expected {N} bytes, got {}", bytes.len()))
}

/// Decodes the canonical encoding of a field element.
fn parse_base(hex: &str) -> pallas::Base {
    let value: Option<pallas::Base> = pallas::Base::from_repr(parse_bytes(hex)).into();
    value.unwrap_or_else(|| panic!("{hex} is not a canonical field element"))
}

/// The committed file, parsed.
fn committed() -> VectorFile {
    serde_json::from_str(COMMITTED_VECTORS).expect("the committed vectors parse")
}

/// The file's text: pretty-printed JSON with a trailing newline.
fn render(file: &VectorFile) -> String {
    let mut text = serde_json::to_string_pretty(file).expect("the vectors serialize");
    text.push('\n');
    text
}

/// Every case's expected outputs are what the code computes from its inputs,
/// so the file cannot drift from the crate.
#[test]
fn committed_vectors_recompute_from_inputs() {
    let file = committed();
    assert!(!file.zkp4_instance.is_empty() && !file.share_secrets.is_empty());
    for case in &file.zkp4_instance {
        assert_eq!(
            pack(&case.inputs),
            case.expected,
            "zkp4_instance/{}",
            case.name
        );
    }
    for case in &file.share_secrets {
        assert_eq!(
            derive_share_secrets(&case.inputs),
            case.expected,
            "share_secrets/{}",
            case.name
        );
    }
}

/// The committed file is exactly what [`generate`] writes from its seeds.
#[test]
fn committed_vectors_match_generator() {
    let generated = generate();
    assert_eq!(
        generated,
        committed(),
        "regenerate only as a contract change"
    );
    assert!(
        render(&generated) == COMMITTED_VECTORS,
        "the committed file is not formatted as the generator writes it"
    );
}

/// The committed file is byte for byte the one contracts v1 froze.
#[test]
fn committed_vectors_match_frozen_hash() {
    assert_eq!(
        hex::encode(Sha256::digest(COMMITTED_VECTORS)),
        FROZEN_SHA256,
        "the vector file changed; any change is a new contract version"
    );
}

/// Rewrites the vector file when `VOTING_CIRCUITS_WRITE_VECTORS=1`;
/// otherwise runs [`committed_vectors_match_generator`], so an ignored-test
/// run still passes.
#[test]
#[ignore = "rewrites test-vectors/proxy_delegation_circuit_v1.json when VOTING_CIRCUITS_WRITE_VECTORS=1"]
fn write_circuit_vectors() {
    if std::env::var(WRITE_VECTORS_ENV).as_deref() == Ok("1") {
        std::fs::write(VECTOR_FILE_PATH, render(&generate())).expect("write the vector file");
    } else {
        committed_vectors_match_generator();
    }
}
