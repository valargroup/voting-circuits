use halo2_proofs::{
    pasta::EqAffine,
    plonk::{self, create_proof},
    poly::commitment::Params,
    transcript::{Blake2bWrite, Challenge255},
};
use pasta_curves::vesta;
use rand::rngs::OsRng;
use std::vec::Vec;

pub(crate) fn create_proof_bytes<ConcreteCircuit>(
    params: &Params<EqAffine>,
    pk: &plonk::ProvingKey<EqAffine>,
    circuit: ConcreteCircuit,
    public_inputs: &[vesta::Scalar],
) -> Result<Vec<u8>, ProveError>
where
    ConcreteCircuit: plonk::Circuit<vesta::Scalar>,
{
    let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
    create_proof(
        params,
        pk,
        &[circuit],
        &[&[public_inputs]],
        OsRng,
        &mut transcript,
    )?;
    Ok(transcript.finalize())
}

/// Error returned when Halo2 proof creation fails.
#[derive(Debug)]
#[non_exhaustive]
pub enum ProveError {
    /// Halo2 rejected the proof inputs or failed during synthesis.
    Halo2(plonk::Error),
}

impl From<plonk::Error> for ProveError {
    fn from(error: plonk::Error) -> Self {
        ProveError::Halo2(error)
    }
}

impl core::fmt::Display for ProveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ProveError::Halo2(error) => write!(f, "Halo2 proof creation failed: {error}"),
        }
    }
}

impl std::error::Error for ProveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProveError::Halo2(error) => Some(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Advice, Column, ConstraintSystem, Error as PlonkError, Instance},
    };

    #[derive(Clone, Debug)]
    struct TinyCircuit {
        witness: Value<vesta::Scalar>,
    }

    #[derive(Clone, Debug)]
    struct TinyConfig {
        advice: Column<Advice>,
        instance: Column<Instance>,
    }

    impl plonk::Circuit<vesta::Scalar> for TinyCircuit {
        type Config = TinyConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self {
                witness: Value::unknown(),
            }
        }

        fn configure(meta: &mut ConstraintSystem<vesta::Scalar>) -> Self::Config {
            let advice = meta.advice_column();
            let instance = meta.instance_column();
            meta.enable_equality(advice);
            meta.enable_equality(instance);
            TinyConfig { advice, instance }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<vesta::Scalar>,
        ) -> Result<(), PlonkError> {
            let cell = layouter.assign_region(
                || "witness",
                |mut region| region.assign_advice(|| "witness", config.advice, 0, || self.witness),
            )?;

            layouter.constrain_instance(cell.cell(), config.instance, 0)
        }
    }

    #[test]
    fn create_proof_bytes_returns_err_for_missing_witness() {
        let params = Params::<EqAffine>::new(4);
        let empty_circuit = TinyCircuit {
            witness: Value::unknown(),
        };
        let vk = plonk::keygen_vk(&params, &empty_circuit).expect("tiny keygen_vk should succeed");
        let pk =
            plonk::keygen_pk(&params, vk, &empty_circuit).expect("tiny keygen_pk should succeed");
        let public_inputs = [vesta::Scalar::from(1)];

        let err = create_proof_bytes(&params, &pk, empty_circuit, &public_inputs).unwrap_err();

        assert!(matches!(err, ProveError::Halo2(plonk::Error::Synthesis)));
    }
}
