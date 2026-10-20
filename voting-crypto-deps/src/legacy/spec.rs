//! Bit encoding inherited from Orchard's private Halo2 specification helpers.
pub(crate) fn i2lebsp<const NUM_BITS: usize>(int: u64) -> [bool; NUM_BITS] {
    assert!(NUM_BITS <= u64::BITS as usize);
    core::array::from_fn(|i| i < u64::BITS as usize && (int >> i) & 1 == 1)
}
