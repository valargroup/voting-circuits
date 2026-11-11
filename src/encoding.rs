//! Bit ordering used by Orchard commitments.

/// Expands a canonical field or compressed-point encoding, least-significant bit first.
/// Callers select the protocol's bit length from the full 256-bit encoding.
pub(crate) fn byte_bits(bytes: [u8; 32]) -> impl Iterator<Item = bool> {
    bytes
        .into_iter()
        .flat_map(|byte| (0..8).map(move |bit| ((byte >> bit) & 1) == 1))
}

#[cfg(test)]
mod tests {
    use super::byte_bits;
    use crate::ff::PrimeField;
    use voting_crypto_deps::pasta_curves::pallas;

    #[test]
    fn canonical_field_bits_preserve_byte_boundaries() {
        for (value, positions) in [(0, vec![]), (1, vec![0]), (256, vec![8]), (257, vec![0, 8])] {
            let bits: Vec<_> = byte_bits(pallas::Base::from(value).to_repr()).collect();
            assert_eq!(bits.len(), 256);
            assert_eq!(
                bits.iter()
                    .enumerate()
                    .filter_map(|(index, bit)| bit.then_some(index))
                    .collect::<Vec<_>>(),
                positions
            );
        }
    }
}
