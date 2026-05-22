//! Gadgets shared by two or more of the governance ZKP circuits.
//!
//! Gadgets used by only a single circuit live in that circuit's own
//! `gadgets` submodule (e.g. `crate::delegation::gadgets`).

pub(crate) mod address_ownership;
pub(crate) mod elgamal;
pub(crate) mod nonzero;
pub(crate) mod poseidon_merkle;
pub(crate) mod spend_authority;
pub(crate) mod van_integrity;
pub(crate) mod vote_commitment;
