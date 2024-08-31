use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use ark_std::log2;
use std::iter::zip;
use std::{marker::PhantomData, sync::Arc};
use subroutines::{
    pcs::prelude::PolynomialCommitmentScheme, Commitment, JoltInstruction, LookupCheck, PolyIOP,
    PolyIOPErrors,
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
    fn verify(
        proof: &Self::Proof,
        witness_openings: &[E::ScalarField],
        regular_openings: &[E::ScalarField],
        transcript: &mut Self::Transcript,
    ) -> Result<HyperPlonkLookupVerifierOpeningPoints<E, PCS>, PolyIOPErrors>;
}

pub struct HyperPlonkLookupPluginNull {}

impl<E, PCS> HyperPlonkLookupPlugin<E, PCS> for HyperPlonkLookupPluginNull
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type Ops = ();
    type Preprocessing = ();
    type Proof = ();
    type Transcript = IOPTranscript<E::ScalarField>;

    fn preprocess() -> Self::Preprocessing {}
    fn construct_witnesses(
        _ops: &Self::Ops,
    ) -> Vec<Arc<DenseMultilinearExtension<E::ScalarField>>> {
        vec![]
    }
    fn num_witness_columns() -> Vec<usize> {
        vec![]
    }
    fn max_num_variables() -> usize {
        0
    }
    fn prove(
        _preprocessing: &Self::Preprocessing,
        _pcs_param: &PCS::ProverParam,
        _ops: &Self::Ops,
        _transcript: &mut Self::Transcript,
    ) -> (Self::Proof, HyperPlonkLookupProverOpeningPoints<E, PCS>) {
        (
            (),
            HyperPlonkLookupProverOpeningPoints {
                regular_openings: vec![],
                witness_openings: vec![],
            },
        )
    }
    fn verify(
        _proof: &Self::Proof,
        _witness_openings: &[E::ScalarField],
        _regular_openings: &[E::ScalarField],
        _transcript: &mut Self::Transcript,
    ) -> Result<HyperPlonkLookupVerifierOpeningPoints<E, PCS>, PolyIOPErrors> {
        Ok(HyperPlonkLookupVerifierOpeningPoints {
            regular_openings: vec![],
            witness_openings: vec![],
        })
    }
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
