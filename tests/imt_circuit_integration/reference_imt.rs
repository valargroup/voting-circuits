//! Compare canonical field encodings with the published, independent IMT.
//!
//! The reference retains its old Pallas crate. Convert only at this test boundary;
//! tree construction, range validation, and hashing still run in the reference.

use imt_tree::tree as reference;
use legacy_pasta_curves::{group::ff::PrimeField as LegacyPrimeField, pallas::Base as LegacyBase};
use voting_circuits::ff::PrimeField;
use voting_crypto_deps::pasta_curves::pallas::Base;

pub const TREE_DEPTH: usize = reference::TREE_DEPTH;
pub type PuncturedRange = [Base; 3];

fn legacy(value: Base) -> LegacyBase {
    LegacyBase::from_repr(value.to_repr()).expect("Pallas field encodings agree")
}

fn current(value: LegacyBase) -> Base {
    Base::from_repr(value.to_repr()).expect("Pallas field encodings agree")
}

fn legacy_ranges(ranges: &[PuncturedRange]) -> Vec<reference::PuncturedRange> {
    ranges.iter().map(|range| range.map(legacy)).collect()
}

pub fn build_punctured_ranges(nullifiers: &[Base]) -> Vec<PuncturedRange> {
    reference::build_punctured_ranges(&nullifiers.iter().copied().map(legacy).collect::<Vec<_>>())
        .into_iter()
        .map(|range| range.map(current))
        .collect()
}

pub fn verify_punctured_range_spans(ranges: &[PuncturedRange]) -> Result<(), String> {
    reference::verify_punctured_range_spans(&legacy_ranges(ranges))
        .map_err(|error| error.to_string())
}

pub fn commit_punctured_ranges(ranges: &[PuncturedRange]) -> Vec<Base> {
    reference::commit_punctured_ranges(&legacy_ranges(ranges))
        .into_iter()
        .map(current)
        .collect()
}

pub fn precompute_empty_hashes() -> [Base; TREE_DEPTH] {
    reference::precompute_empty_hashes().map(current)
}

pub fn build_levels(
    leaves: Vec<Base>,
    empty: &[Base; TREE_DEPTH],
    depth: usize,
) -> (Base, Vec<Vec<Base>>) {
    let (root, levels) = reference::build_levels(
        leaves.into_iter().map(legacy).collect(),
        &empty.map(legacy),
        depth,
    );
    (
        current(root),
        levels
            .into_iter()
            .map(|level| level.into_iter().map(current).collect())
            .collect(),
    )
}

pub fn find_punctured_range_for_value(ranges: &[PuncturedRange], value: Base) -> Option<usize> {
    reference::find_punctured_range_for_value(&legacy_ranges(ranges), legacy(value))
}
