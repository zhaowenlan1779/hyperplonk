use crate::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::{
        errors::PolyIOPErrors,
        rational_sumcheck::{
            RationalSumcheckProof, RationalSumcheckSlow, RationalSumcheckSubClaim,
        },
        PolyIOP,
    },
    Commitment,
};
use arithmetic::{math::Math, OptimizedMul, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_ff::{batch_inversion, One, PrimeField};
use ark_poly::DenseMultilinearExtension;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::sync::Arc;
use transcript::IOPTranscript;

use super::{
    instruction::JoltInstruction, util::SurgeCommons, SurgePolysPrimary, SurgePreprocessing,
};

#[derive(Clone, Debug, PartialEq)]
pub struct LogupCheckingProof<
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
> {
    pub f_proof: RationalSumcheckProof<E::ScalarField>,
    pub g_proof: RationalSumcheckProof<E::ScalarField>,
    pub f_inv_comm: Vec<PCS::Commitment>,
    pub g_inv_comm: Vec<PCS::Commitment>,
}

/// A permutation subclaim consists of
/// - the SubClaim from the ProductCheck
/// - Challenges beta and gamma
#[derive(Clone, Debug, PartialEq)]
pub struct LogupCheckingSubclaim<F: PrimeField>
where
    F: PrimeField,
{
    pub f_subclaims: RationalSumcheckSubClaim<F>,
    pub g_subclaims: RationalSumcheckSubClaim<F>,

    /// Challenges beta and gamma
    pub challenges: (F, F),
}

pub(super) trait LogupChecking<E, PCS, Instruction, const C: usize, const M: usize>:
    RationalSumcheckSlow<E>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    Instruction: JoltInstruction + Default,
{
    type LogupCheckingProof;
    type LogupCheckingSubclaim;
    type Preprocessing;
    type Polys;

    fn compute_leaves(
        preprocessing: &Self::Preprocessing,
        polynomials: &mut Self::Polys,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
        alpha: &E::ScalarField,
    ) -> (
        (
            Vec<Self::VirtualPolynomial>,
            Vec<Self::MultilinearExtension>,
            Vec<Self::MultilinearExtension>,
        ),
        (
            Vec<Self::VirtualPolynomial>,
            Vec<Self::MultilinearExtension>,
            Vec<Self::MultilinearExtension>,
        ),
    );

    fn prove_logup_checking(
        pcs_param: &PCS::ProverParam,
        preprocessing: &Self::Preprocessing,
        polynomials_primary: &mut Self::Polys,
        alpha: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<(Self::LogupCheckingProof,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>), PolyIOPErrors>;

    fn verify_logup_checking(
        proof: &Self::LogupCheckingProof,
        aux_info_f: &Self::VPAuxInfo,
        aux_info_g: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LogupCheckingSubclaim, PolyIOPErrors>;
}

impl<E, PCS, Instruction, const C: usize, const M: usize> LogupChecking<E, PCS, Instruction, C, M>
    for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Commitment = Commitment<E>,
    >,
    Instruction: JoltInstruction + Default,
{
    type LogupCheckingProof = LogupCheckingProof<E, PCS>;
    type LogupCheckingSubclaim = LogupCheckingSubclaim<E::ScalarField>;
    type Preprocessing = SurgePreprocessing<E::ScalarField>;
    type Polys = SurgePolysPrimary<E>;

    fn compute_leaves(
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials: &mut SurgePolysPrimary<E>,
        beta: &E::ScalarField,
        gamma: &E::ScalarField,
        alpha: &E::ScalarField,
    ) -> (
        (
            Vec<Self::VirtualPolynomial>,
            Vec<Self::MultilinearExtension>,
            Vec<Self::MultilinearExtension>,
        ),
        (
            Vec<Self::VirtualPolynomial>,
            Vec<Self::MultilinearExtension>,
            Vec<Self::MultilinearExtension>,
        ),
    ) {
        let num_vars_g = polynomials.dim[0].num_vars;
        let num_lookups = polynomials.dim[0].evaluations.len();
        let (g_leaves_q, f_leaves_q) = rayon::join(
            || {
                (0..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
                    .into_par_iter()
                    .map(|memory_index| {
                        let dim_index = <Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::memory_to_dimension_index(memory_index);

                        let q = DenseMultilinearExtension::from_evaluations_vec(
                            num_vars_g,
                            (0..num_lookups)
                                .map(|i| {
                                    polynomials.E_polys[memory_index][i].mul_0_optimized(*gamma)
                                        + polynomials.dim[dim_index][i]
                                        + *beta
                                })
                                .collect(),
                        );

                        let mut q_inv = DenseMultilinearExtension::clone(&q);
                        batch_inversion(&mut q_inv.evaluations);

                        (Arc::new(q), Arc::new(q_inv))
                    })
                    .unzip::<_, _, Vec<_>, Vec<_>>()
            },
            || {
                let mut f_leaves_q = preprocessing
                    .materialized_subtables
                    .par_iter()
                    .map(|subtable| {
                        let q = DenseMultilinearExtension::from_evaluations_vec(
                            subtable.len().log_2(),
                            subtable
                                .iter()
                                .enumerate()
                                .map(|(i, t_eval)| {
                                    t_eval.mul_0_optimized(*gamma)
                                        + E::ScalarField::from_u64(i as u64).unwrap()
                                        + *beta
                                })
                                .collect(),
                        );

                        let mut q_inv = DenseMultilinearExtension::clone(&q);
                        batch_inversion(&mut q_inv.evaluations);

                        (Arc::new(q), Arc::new(q_inv))
                    })
                    .unzip::<_, _, Vec<_>, Vec<_>>();

                let bits_per_operand = (ark_std::log2(M) / 2) as usize;
                let sqrtM = bits_per_operand.pow2() as u64;
                let mut dechunk_evals = Vec::with_capacity(M);
                for x in 0..sqrtM {
                    for y in 0..sqrtM {
                        dechunk_evals.push(
                            (E::ScalarField::from_u64(x).unwrap()
                                + E::ScalarField::from_u64(y).unwrap() * *alpha)
                                * *gamma
                                + E::ScalarField::from_u64(((x << bits_per_operand) | y) as u64)
                                    .unwrap()
                                + *beta,
                        )
                    }
                }
                let mut dechunk_evals_inv = dechunk_evals.clone();
                batch_inversion(&mut dechunk_evals_inv);

                f_leaves_q.0.push(
                    Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                        ark_std::log2(M) as usize,
                        dechunk_evals,
                    )));
                f_leaves_q.1.push(
                    Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                        ark_std::log2(M) as usize,
                        dechunk_evals_inv,
                    )));
                f_leaves_q
            },
        );

        let mut g_leaves_p = VirtualPolynomial::new(num_vars_g);
        g_leaves_p.add_mle_list([], E::ScalarField::one()).unwrap();

        let f_leaves_p = polynomials
            .m
            .iter()
            .map(|m_poly| VirtualPolynomial::new_from_mle(m_poly, E::ScalarField::one()))
            .collect();

        (
            (f_leaves_p, f_leaves_q.0, f_leaves_q.1),
            (vec![g_leaves_p.clone(); g_leaves_q.0.len()], g_leaves_q.0, g_leaves_q.1),
        )
    }

    // #[tracing::instrument(skip_all, name =
    // "LogupCheckingProof::prove_logup_checking")]
    fn prove_logup_checking(
        pcs_param: &PCS::ProverParam,
        preprocessing: &SurgePreprocessing<E::ScalarField>,
        polynomials_primary: &mut SurgePolysPrimary<E>,
        alpha: &E::ScalarField,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(Self::LogupCheckingProof,
        Vec<Self::MultilinearExtension>,
        Vec<Self::MultilinearExtension>), PolyIOPErrors> {
        // We assume that primary commitments are already appended to the transcript
        let beta = transcript.get_and_append_challenge(b"logup_beta")?;
        let gamma = transcript.get_and_append_challenge(b"logup_gamma")?;

        let (f_leaves, g_leaves) =
            <Self as LogupChecking<E, PCS, Instruction, C, M>>::compute_leaves(preprocessing, polynomials_primary, &beta, &gamma, alpha);

        let (f_inv_comm, g_inv_comm) = rayon::join(
            || {
                f_leaves
                    .2
                    .iter()
                    .map(|inv| PCS::commit(pcs_param, inv).unwrap())
                    .collect()
            },
            || {
                g_leaves
                    .2
                    .iter()
                    .map(|inv| PCS::commit(pcs_param, inv).unwrap())
                    .collect()
            },
        );

        let claimed_sums = (0..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
        .map(|memory_index| 
            g_leaves.2[memory_index].evaluations.iter().sum()
        )
        .collect::<Vec<_>>();

        transcript.append_serializable_element(b"f_inv_comm", &f_inv_comm)?;
        transcript.append_serializable_element(b"g_inv_comm", &g_inv_comm)?;

        let f_p = (0..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
            .map(|memory_index| {
                let dim_index = <Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::memory_to_dimension_index(memory_index);
                f_leaves.0[dim_index].clone()
            }).collect::<Vec<_>>();
        let (f_q, f_q_inv) = (0..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories())
            .map(|memory_index| {
                let subtable_index = <Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::memory_to_subtable_index(memory_index);
                (f_leaves.1[subtable_index].clone(), f_leaves.2[subtable_index].clone())
            }).unzip::<_, _, Vec<_>, Vec<_>>();

        let f_proof = <Self as RationalSumcheckSlow<E>>::prove(
            &f_p,
            &f_q,
            &f_q_inv,
            claimed_sums.clone(),
            transcript,
        )?;

        let g_proof = <Self as RationalSumcheckSlow<E>>::prove(&g_leaves.0, &g_leaves.1, &g_leaves.2, 
            claimed_sums, transcript)?;

        Ok((LogupCheckingProof {
            f_proof,
            f_inv_comm,
            g_proof,
            g_inv_comm,
        }, f_leaves.2, g_leaves.2))
    }

    // #[tracing::instrument(skip_all, name =
    // "LogupCheckingProof::verify_logup_checking")]
    fn verify_logup_checking(
        proof: &LogupCheckingProof<E, PCS>,
        aux_info_f: &Self::VPAuxInfo,
        aux_info_g: &Self::VPAuxInfo,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<LogupCheckingSubclaim<E::ScalarField>, PolyIOPErrors> {
        // Check that the final claims are equal
        rayon::join(
            || {
                (0..<Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories()).into_par_iter().for_each(|i| {
                    assert_eq!(
                        proof.f_proof.claimed_sums[i],
                        proof.g_proof.claimed_sums[i],
                        "Final claims are inconsistent"
                    );
                });
            },
            || {
                // Assumes that primary commitments have been added to transcript
                let beta = transcript.get_and_append_challenge(b"logup_beta")?;
                let gamma = transcript.get_and_append_challenge(b"logup_gamma")?;

                transcript.append_serializable_element(b"f_inv_comm", &proof.f_inv_comm)?;
                transcript.append_serializable_element(b"g_inv_comm", &proof.g_inv_comm)?;

                let f_subclaims = 
                    <Self as RationalSumcheckSlow<E>>::verify(
                        &proof.f_proof,
                        aux_info_f,
                        transcript,
                    )?;

                let g_subclaims =
                    <Self as RationalSumcheckSlow<E>>::verify(
                        &proof.g_proof,
                        aux_info_g,
                        transcript,
                    )?;

                Ok(LogupCheckingSubclaim {
                    f_subclaims,
                    g_subclaims,
                    challenges: (beta, gamma),
                })
            },
        )
        .1
    }
}
