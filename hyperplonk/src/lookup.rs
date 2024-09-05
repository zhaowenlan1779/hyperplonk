use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use ark_std::log2;
use std::{iter::zip, marker::PhantomData, sync::Arc};
use subroutines::{
    pcs::prelude::PolynomialCommitmentScheme, Commitment, JoltInstruction, LookupCheck, PolyIOP, PolyIOPErrors
};
use transcript::IOPTranscript;

pub struct HyperPlonkLookupProverOpeningPoints<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub regular_openings: Vec<(PCS::Polynomial, PCS::Commitment, Vec<E::ScalarField>)>,
    pub witness_openings: Vec<Vec<E::ScalarField>>,
}

pub struct HyperPlonkLookupVerifierOpeningPoints<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub regular_openings: Vec<(PCS::Commitment, Vec<E::ScalarField>)>,
    pub witness_openings: Vec<Vec<E::ScalarField>>,
}

pub trait HyperPlonkLookupPlugin<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ops;
    type Preprocessing: Clone + Sync;
    type Transcript;
    type Proof;

    fn preprocess() -> Self::Preprocessing;
    fn construct_witnesses(ops: &Self::Ops) -> Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>;
    fn num_witness_columns() -> Vec<usize>;
    fn max_num_variables() -> usize;
    fn prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        ops: &Self::Ops,
        transcript: &mut Self::Transcript,
    ) -> (Self::Proof, HyperPlonkLookupProverOpeningPoints<E, PCS>);
    fn num_regular_openings(proof: &Self::Proof) -> usize;
    fn verify(
        proof: &Self::Proof,
        witness_openings: &[E::ScalarField],
        regular_openings: &[E::ScalarField],
        transcript: &mut Self::Transcript,
    ) -> Result<HyperPlonkLookupVerifierOpeningPoints<E, PCS>, PolyIOPErrors>;
}

pub struct HyperPlonkLookupPluginSingle<
    Instruction: JoltInstruction + Default,
    const C: usize,
    const M: usize,
> {
    marker: PhantomData<Instruction>,
}

impl<E, PCS, Instruction, const C: usize, const M: usize> HyperPlonkLookupPlugin<E, PCS>
    for HyperPlonkLookupPluginSingle<Instruction, C, M>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Commitment = Commitment<E>,
    >,
    Instruction: JoltInstruction + Default,
{
    type Ops = Vec<Instruction>;
    type Preprocessing =
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::Preprocessing;
    type Proof =
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::LookupCheckProof;
    type Transcript = IOPTranscript<E::ScalarField>;

    fn preprocess() -> Self::Preprocessing {
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::preprocess()
    }
    fn construct_witnesses(ops: &Self::Ops) -> Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> {
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::construct_witnesses(
            ops,
        )
    }
    fn num_witness_columns() -> Vec<usize> {
        vec![3]
    }
    fn max_num_variables() -> usize {
        log2(M) as usize
    }

    fn prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        ops: &Self::Ops,
        transcript: &mut Self::Transcript,
    ) -> (Self::Proof, HyperPlonkLookupProverOpeningPoints<E, PCS>) {
        let alpha = transcript
            .get_and_append_challenge(b"lookup_alpha")
            .unwrap();
        let tau = transcript.get_and_append_challenge(b"lookup_tau").unwrap();

        let mut polys =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::construct_polys(
                preprocessing,
                ops,
                &alpha,
            );

        let (proof, r_f, r_g, r_z, r_primary_sumcheck) =
            <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::prove(
                preprocessing,
                pcs_param,
                &mut polys,
                &alpha,
                &tau,
                transcript,
            )
            .unwrap();

        let mut regular_openings =
            Vec::with_capacity(polys.dim.len() + polys.m.len() + 2 * polys.E_polys.len());
        for (poly, comm) in zip(polys.m.iter(), proof.commitment.m_commitment.iter()) {
            regular_openings.push((poly.clone(), *comm, r_f.clone()));
        }
        for (poly, comm) in zip(polys.dim.iter(), proof.commitment.dim_commitment.iter()) {
            regular_openings.push((poly.clone(), *comm, r_g.clone()));
        }
        for (poly, comm) in zip(polys.E_polys.iter(), proof.commitment.E_commitment.iter()) {
            regular_openings.push((poly.clone(), *comm, r_g.clone()));
        }
        for (poly, comm) in zip(polys.E_polys.iter(), proof.commitment.E_commitment.iter()) {
            regular_openings.push((poly.clone(), *comm, r_z.clone()));
        }

        (
            proof,
            HyperPlonkLookupProverOpeningPoints {
                regular_openings,
                witness_openings: vec![r_primary_sumcheck.clone(); 3],
            },
        )
    }
    fn num_regular_openings(proof: &Self::Proof) -> usize {
        proof.commitment.dim_commitment.len()
            + proof.commitment.m_commitment.len()
            + 2 * proof.commitment.E_commitment.len()
    }
    fn verify(
        proof: &Self::Proof,
        witness_openings: &[E::ScalarField],
        regular_openings: &[E::ScalarField],
        transcript: &mut Self::Transcript,
    ) -> Result<HyperPlonkLookupVerifierOpeningPoints<E, PCS>, PolyIOPErrors> {
        let alpha = transcript
            .get_and_append_challenge(b"lookup_alpha")
            .unwrap();
        let tau = transcript.get_and_append_challenge(b"lookup_tau").unwrap();

        let subclaim = <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::verify(
            proof, transcript,
        )?;
        <PolyIOP<E::ScalarField> as LookupCheck<E, PCS, Instruction, C, M>>::check_openings(
            &subclaim,
            &regular_openings[proof.commitment.m_commitment.len()
                ..proof.commitment.dim_commitment.len() + proof.commitment.m_commitment.len()],
            &regular_openings
                [proof.commitment.dim_commitment.len() + proof.commitment.m_commitment.len()..],
            &regular_openings[..proof.commitment.m_commitment.len()],
            &witness_openings,
            &alpha,
            &tau,
        )?;

        let mut regular_openings = Vec::with_capacity(
            proof.commitment.dim_commitment.len()
                + proof.commitment.m_commitment.len()
                + 2 * proof.commitment.E_commitment.len(),
        );
        for comm in proof.commitment.m_commitment.iter() {
            regular_openings.push((*comm, subclaim.logup_checking.point_f.clone()));
        }
        for comm in proof.commitment.dim_commitment.iter() {
            regular_openings.push((*comm, subclaim.logup_checking.point_g.clone()));
        }
        for comm in proof.commitment.E_commitment.iter() {
            regular_openings.push((*comm, subclaim.logup_checking.point_g.clone()));
        }
        for comm in proof.commitment.E_commitment.iter() {
            regular_openings.push((*comm, subclaim.r_z.clone()));
        }

        Ok(HyperPlonkLookupVerifierOpeningPoints {
            regular_openings,
            witness_openings: vec![subclaim.r_primary_sumcheck.clone(); 3],
        })
    }
}

#[macro_export]
macro_rules! combine_lookup_plugins {
    ($name:ident : $($plugin:ty),*) => {
        pub struct $name {}

        impl<E, PCS> $crate::lookup::HyperPlonkLookupPlugin<E, PCS> for $name
            where E: ark_ec::pairing::Pairing,
            PCS: subroutines::pcs::prelude::PolynomialCommitmentScheme<E,
                Polynomial = std::sync::Arc<ark_poly::DenseMultilinearExtension<E::ScalarField>>,
                Point = Vec<E::ScalarField>,
                Evaluation = E::ScalarField,
                Commitment = subroutines::Commitment<E>,
                BatchProof = subroutines::BatchProof<E, PCS>,
            > {
            type Ops = ($(Option<<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::Ops>,)*);
            type Preprocessing = ($(<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::Preprocessing,)*);
            type Proof = ($(Option<<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::Proof>,)*);
            type Transcript = transcript::IOPTranscript<E::ScalarField>;

            fn preprocess() -> Self::Preprocessing {
                ($(<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::preprocess(),)*)
            }
            fn construct_witnesses(ops: &Self::Ops) -> Vec<std::sync::Arc<ark_poly::DenseMultilinearExtension<E::ScalarField>>> {
                let witness_vecs : Vec<Vec<std::sync::Arc<ark_poly::DenseMultilinearExtension<E::ScalarField>>>> = vec![$(if let Some(ops) = &ops.${index()} {
                    <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::construct_witnesses(ops)
                } else {
                    vec![]
                }),*];
                witness_vecs.concat()
            }
            fn num_witness_columns() -> Vec<usize> {
                let witness_columns : Vec<Vec<usize>> = 
                    vec![$(<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::num_witness_columns()),*];
                witness_columns.concat()
            }
            fn max_num_variables() -> usize {
                *vec![0, $(<$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::max_num_variables()),*].iter().max().unwrap()
            }
            fn prove(
                preprocessing: &Self::Preprocessing,
                pcs_param: &PCS::ProverParam,
                ops: &Self::Ops,
                transcript: &mut Self::Transcript,
            ) -> (Self::Proof, $crate::lookup::HyperPlonkLookupProverOpeningPoints<E, PCS>) {
                let mut all_openings = $crate::lookup::HyperPlonkLookupProverOpeningPoints {
                    regular_openings: vec![],
                    witness_openings: vec![],
                };

                (($(
                    if let Some(ops) = &ops.${index()} {
                        let (proof, mut openings) = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::prove(&preprocessing.${index()}, pcs_param, ops, transcript);
                        all_openings.regular_openings.append(&mut openings.regular_openings);
                        all_openings.witness_openings.append(&mut openings.witness_openings);
                        Some(proof)
                    } else {
                        None
                    }
                ,)*), all_openings)
            }
            fn num_regular_openings(
                proof: &Self::Proof,
            ) -> usize {
                vec![
                    $(if let Some(proof) = &proof.${index()} {
                        <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::num_regular_openings(&proof)
                    } else {
                        0
                    }),*
                ].iter().sum()
            }
            fn verify(
                proof: &Self::Proof,
                witness_openings: &[E::ScalarField],
                regular_openings: &[E::ScalarField],
                transcript: &mut Self::Transcript,
            ) -> Result<$crate::lookup::HyperPlonkLookupVerifierOpeningPoints<E, PCS>, subroutines::PolyIOPErrors> {
                let mut witness_index = 0;
                let mut regular_index = 0;
                let mut all_openings = $crate::lookup::HyperPlonkLookupVerifierOpeningPoints {
                    regular_openings: vec![],
                    witness_openings: vec![],
                };

                $(if let Some(proof) = &proof.${index()} {
                    let num_witnesses : usize = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::num_witness_columns().iter().sum();
                    let num_regular_openings = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::num_regular_openings(proof);
                    
                    let mut openings = <$plugin as $crate::lookup::HyperPlonkLookupPlugin<E, PCS>>::verify(proof,
                        &witness_openings[witness_index..witness_index + num_witnesses],
                        &regular_openings[regular_index..regular_index + num_regular_openings],
                        transcript)?;
                    all_openings.regular_openings.append(&mut openings.regular_openings);
                    all_openings.witness_openings.append(&mut openings.witness_openings);

                    witness_index += num_witnesses;
                    regular_index += num_regular_openings;
                })*

                Ok(all_openings)
            }
        }
    };
}

combine_lookup_plugins! { HyperPlonkLookupPluginNull : }

#[macro_export]
macro_rules! jolt_lookup {
    ($name:ident, $C:expr, $M:expr; $($inst:ty),*) => {
        $crate::combine_lookup_plugins! { $name : $($crate::lookup::HyperPlonkLookupPluginSingle<$inst, $C, $M>),* }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use subroutines::instruction::xor::XORInstruction;
    
    combine_lookup_plugins! { HyperPlonkLookupPluginTest1 : HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>}
    combine_lookup_plugins! { HyperPlonkLookupPluginTest2 : HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>,
        HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>}
    combine_lookup_plugins! { HyperPlonkLookupPluginTest4 : HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>,
            HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>,
            HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>,
            HyperPlonkLookupPluginSingle<XORInstruction, 4, 65536>}
    jolt_lookup! { JoltLookupTest1, 4, 65536; XORInstruction }
    jolt_lookup! { JoltLookupTest3, 4, 65536; XORInstruction, XORInstruction, XORInstruction }
}
