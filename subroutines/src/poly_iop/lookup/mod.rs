// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the Lookup Check protocol

#![allow(non_snake_case)]
use crate::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::{
        errors::PolyIOPErrors, sum_check::generic_sumcheck::SumcheckInstanceProof, PolyIOP,
    },
    split_bits,
};
use arithmetic::{eq_eval, eq_poly::EqPolynomial, math::Math, OptimizedMul};
use ark_ec::pairing::Pairing;
use ark_ff::{One, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{end_timer, start_timer};
use instruction::JoltInstruction;
use instruction::{concatenate_lookups, evaluate_mle_dechunk_operands};
use logup_checking::{LogupChecking, LogupCheckingProof, LogupCheckingSubclaim};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use std::sync::Arc;
use transcript::IOPTranscript;
use util::{polys_from_evals, polys_from_evals_usize, SurgeCommons};

pub mod instruction;
mod logup_checking;
mod subtable;
mod util;
pub trait LookupCheck<E, PCS, Instruction, const C: usize, const M: usize>:
    SurgeCommons<E::ScalarField, Instruction, C, M>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    Instruction: JoltInstruction + Default,
{
    type LookupCheckSubClaim;
    type LookupCheckProof;

    type Preprocessing;
    type WitnessPolys;
    type Polys;
    type MultilinearExtension;
    type Transcript;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a LookupCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// LookupCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    fn preprocess() -> Self::Preprocessing;

    fn construct_witnesses(ops: &[Instruction]) -> Self::WitnessPolys;

    fn construct_polys(
        preprocessing: &Self::Preprocessing,
        ops: &[Instruction],
        alpha: &E::ScalarField,
    ) -> Self::Polys;

    // Returns (proof, r_f, r_g, r_z, r_primary_sumcheck)
    fn prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        poly: &mut Self::Polys,
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<
        (
            Self::LookupCheckProof,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
        ),
        PolyIOPErrors,
    >;

    /// verify the claimed sum using the proof
    fn verify(
        proof: &Self::LookupCheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors>;

    fn check_openings(
        subclaim: &Self::LookupCheckSubClaim,
        dim_openings: &[E::ScalarField],
        E_openings: &[E::ScalarField],
        m_openings: &[E::ScalarField],
        witness_openings: &[E::ScalarField],
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
    ) -> Result<(), PolyIOPErrors>;
}

#[derive(Clone)]
pub struct SurgePreprocessing<F>
where
    F: PrimeField,
{
    pub materialized_subtables: Vec<Vec<F>>,
}

pub struct SurgePolysPrimary<E>
where
    E: Pairing,
{
    // These polynomials are constructed as big-endian
    // Since everything else treats polynomials as little-endian, r_f, r_g, and r_z
    // all have their indices reversed
    pub dim: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>, // Size C
    pub E_polys: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>, // Size NUM_MEMORIES
    pub m: Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>,   // Size C

    // Sparse representation of m
    pub m_indices: Vec<Vec<usize>>,
    pub m_values: Vec<Vec<E::ScalarField>>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgeCommitmentPrimary<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    pub dim_commitment: Vec<PCS::Commitment>, // Size C
    pub E_commitment: Vec<PCS::Commitment>,   // Size NUM_MEMORIES
    pub m_commitment: Vec<PCS::Commitment>,   // Size C
}

impl<E> SurgePolysPrimary<E>
where
    E: Pairing,  
{
    // #[tracing::instrument(skip_all, name = "SurgePolysPrimary::commit")]
    fn commit<PCS>(&self, pcs_params: &PCS::ProverParam) -> SurgeCommitmentPrimary<E, PCS>
    where PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
    PCS::Commitment: Send {
        let dim_commitment = self
            .dim
            .par_iter()
            .map(|poly| PCS::commit(pcs_params, poly).unwrap())
            .collect();

        let E_commitment = self
            .E_polys
            .par_iter()
            .map(|poly| PCS::commit(pcs_params, poly).unwrap())
            .collect();

        let m_commitment = self
            .m
            .par_iter()
            .map(|poly| PCS::commit(pcs_params, poly).unwrap())
            .collect();

        SurgeCommitmentPrimary {
            dim_commitment,
            E_commitment,
            m_commitment,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct SurgePrimarySumcheck<F>
where
    F: PrimeField,
{
    sumcheck_proof: SumcheckInstanceProof<F>,
    num_rounds: usize,
    claimed_evaluation: F,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LookupCheckProof<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    pub primary_sumcheck: SurgePrimarySumcheck<E::ScalarField>,
    pub logup_checking: LogupCheckingProof<E::ScalarField>,
    pub commitment: SurgeCommitmentPrimary<E, PCS>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookupCheckSubClaim<F: PrimeField> {
    pub r_primary_sumcheck: Vec<F>, // Challenge used as eq polynomial parameter
    pub primary_sumcheck_claim: F,

    // Primary sumcheck subclaim. Note this differs from r_primary_sumcheck
    pub r_z: Vec<F>,
    pub primary_sumcheck_expected_evaluation: F,

    pub logup_checking: LogupCheckingSubclaim<F>,
}

impl<E, PCS, Instruction, const C: usize, const M: usize> LookupCheck<E, PCS, Instruction, C, M>
    for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
    PCS::Commitment: Send,
    Instruction: JoltInstruction + Default,
{
    type LookupCheckSubClaim = LookupCheckSubClaim<E::ScalarField>;
    type LookupCheckProof = LookupCheckProof<E, PCS>;

    type Preprocessing = SurgePreprocessing<E::ScalarField>;
    type WitnessPolys = Vec<Arc<DenseMultilinearExtension<E::ScalarField>>>;
    type Polys = SurgePolysPrimary<E>;
    type MultilinearExtension = Arc<DenseMultilinearExtension<E::ScalarField>>;
    type Transcript = IOPTranscript<E::ScalarField>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing LookupCheck transcript")
    }

    fn preprocess() -> Self::Preprocessing {
        let instruction = Instruction::default();

        let materialized_subtables = instruction
            .subtables(C, M)
            .par_iter()
            .map(|(subtable, _)| subtable.materialize(M))
            .collect();

        Self::Preprocessing {
            materialized_subtables,
        }
    }

    fn construct_witnesses(ops: &[Instruction]) -> Self::WitnessPolys {
        let num_lookups = ops.len().next_power_of_two();
        let log_m = ark_std::log2(num_lookups) as usize;

        // aka operands
        let mut witness_evals = vec![vec![0usize; num_lookups]; 3];
        for (op_index, op) in ops.iter().enumerate() {
            let (operand_x, operand_y) = op.operands();
            witness_evals[0][op_index] = operand_x as usize;
            witness_evals[1][op_index] = operand_y as usize;
            witness_evals[2][op_index] = op.lookup_entry() as usize;
        }

        polys_from_evals_usize(log_m, &witness_evals)
    }

    // #[tracing::instrument(skip_all, name = "Surge::construct_polys")]
    fn construct_polys(
        preprocessing: &Self::Preprocessing,
        ops: &[Instruction],
        alpha: &E::ScalarField,
    ) -> Self::Polys {
        let num_memories =
            <Self as SurgeCommons<E::ScalarField, Instruction, C, M>>::num_memories();
        let num_lookups = ops.len().next_power_of_two();
        let log_m = ark_std::log2(num_lookups) as usize;

        // Construct dim, m
        let mut dim_usize: Vec<Vec<usize>> = vec![vec![0; num_lookups]; C];

        let mut m_evals = vec![vec![0usize; M]; C];
        let log_M = ark_std::log2(M) as usize;
        let bits_per_operand = log_M / 2;

        for (op_index, op) in ops.iter().enumerate() {
            let access_sequence = op.to_indices(C, log_M);
            assert_eq!(access_sequence.len(), C);

            for dimension_index in 0..C {
                let memory_address = access_sequence[dimension_index];
                debug_assert!(memory_address < M);

                dim_usize[dimension_index][op_index] = memory_address;
                m_evals[dimension_index][memory_address] += 1;
            }
        }

        // num_ops is padded to the nearest power of 2 for the usage of DensePolynomial. We cannot just fill
        // in zeros for m_evals as this implicitly specifies a read at address 0.
        for _fake_ops_index in ops.len()..num_lookups {
            for dimension_index in 0..C {
                let memory_address = 0;
                m_evals[dimension_index][memory_address] += 1;
            }
        }

        let mut m_indices = vec![];
        let mut m_values = vec![];
        let mut dim_poly = vec![];
        let mut m_poly = vec![];
        let mut E_poly = vec![];
        rayon::scope(|s| {
            s.spawn(|_| {
                (m_indices, m_values) = m_evals
                    .iter()
                    .map(|m_evals_it| {
                        let mut indices = vec![];
                        let mut values = vec![];
                        for (i, m) in m_evals_it.iter().enumerate() {
                            if *m != 0 {
                                indices.push(i);
                                values.push(E::ScalarField::from_u64(*m as u64).unwrap());
                            }
                        }
                        (indices, values)
                    })
                    .unzip();
            });
            s.spawn(|_| {
                dim_poly = polys_from_evals_usize(log_m, &dim_usize);
            });
            s.spawn(|_| {
                m_poly = polys_from_evals_usize(log_M, &m_evals);
            });
            s.spawn(|_| {
                // Construct E
                let mut E_i_evals = Vec::with_capacity(num_memories);
                for E_index in 0..num_memories {
                    let mut E_evals = Vec::with_capacity(num_lookups);
                    for op_index in 0..num_lookups {
                        let dimension_index = <Self as SurgeCommons<
                            E::ScalarField,
                            Instruction,
                            C,
                            M,
                        >>::memory_to_dimension_index(
                            E_index
                        );
                        let subtable_index = <Self as SurgeCommons<
                            E::ScalarField,
                            Instruction,
                            C,
                            M,
                        >>::memory_to_subtable_index(
                            E_index
                        );

                        let eval_index = dim_usize[dimension_index][op_index];
                        let E_eval = if subtable_index >= preprocessing.materialized_subtables.len()
                        {
                            let (x, y) = split_bits(eval_index, bits_per_operand);
                            E::ScalarField::from_u64(x as u64).unwrap()
                                + *alpha * E::ScalarField::from_u64(y as u64).unwrap()
                        } else {
                            preprocessing.materialized_subtables[subtable_index][eval_index]
                        };
                        E_evals.push(E_eval);
                    }
                    E_i_evals.push(E_evals);
                }
                E_poly = polys_from_evals(log_m, &E_i_evals);
            });
        });

        SurgePolysPrimary {
            dim: dim_poly,
            E_polys: E_poly,
            m: m_poly,
            m_indices,
            m_values,
        }
    }

    fn prove(
        preprocessing: &Self::Preprocessing,
        pcs_param: &PCS::ProverParam,
        poly: &mut Self::Polys,
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
        transcript: &mut Self::Transcript,
    ) -> Result<
        (
            Self::LookupCheckProof,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
            Vec<E::ScalarField>,
        ),
        PolyIOPErrors,
    > {
        let start = start_timer!(|| "lookup_check prove");

        let commitment = poly.commit(pcs_param);
        transcript.append_serializable_element(b"primary_commitment", &commitment)?;

        let num_rounds = poly.dim[0].num_vars;
        let instruction = Instruction::default();

        // TODO(sragss): Commit some of this stuff to transcript?

        // Primary sumcheck
        let mut r_primary_sumcheck =
            transcript.get_and_append_challenge_vectors(b"primary_sumcheck", num_rounds)?;
        let eq = DenseMultilinearExtension::from_evaluations_vec(
            r_primary_sumcheck.len(),
            EqPolynomial::evals(&r_primary_sumcheck),
        );
        r_primary_sumcheck.reverse();

        let log_M = ark_std::log2(M) as usize;

        let sumcheck_claim = {
            let hypercube_size = poly.E_polys[0].evaluations.len();
            poly.E_polys
                .iter()
                .for_each(|operand| assert_eq!(operand.evaluations.len(), hypercube_size));

            let instruction = Instruction::default();

            (0..hypercube_size)
                .into_par_iter()
                .map(|eval_index| {
                    let g_operands: Vec<E::ScalarField> = (0..<Self as SurgeCommons<
                        E::ScalarField,
                        Instruction,
                        C,
                        M,
                    >>::num_memories(
                    ))
                        .map(|memory_index| poly.E_polys[memory_index][eval_index])
                        .collect();

                    let vals: &[E::ScalarField] = &g_operands[0..(g_operands.len() - C)];
                    let fingerprints = &g_operands[g_operands.len() - C..];
                    eq[eval_index]
                        * (instruction.combine_lookups(vals, C, M)
                            + *alpha * concatenate_lookups(fingerprints, C, log_M / 2)
                            + *tau)
                })
                .sum()
        };
        transcript.append_field_element(b"sumcheck_claim", &sumcheck_claim)?;

        let mut combined_sumcheck_polys = poly
            .E_polys
            .iter()
            .map(|it| Arc::new((**it).clone()))
            .collect::<Vec<_>>();
        combined_sumcheck_polys.push(Arc::new(eq));

        let combine_lookups_eq = |vals: &[E::ScalarField]| -> E::ScalarField {
            let vals_no_eq: &[E::ScalarField] = &vals[0..(vals.len() - 1 - C)];
            let fingerprints = &vals[vals.len() - 1 - C..vals.len() - 1];
            let eq = vals[vals.len() - 1];
            (instruction.combine_lookups(vals_no_eq, C, M)
                + *alpha * concatenate_lookups(fingerprints, C, log_M / 2)
                + *tau)
                * eq
        };

        let (primary_sumcheck_proof, mut r_z, _) =
            SumcheckInstanceProof::<E::ScalarField>::prove_arbitrary::<_>(
                &sumcheck_claim,
                num_rounds,
                &mut combined_sumcheck_polys,
                combine_lookups_eq,
                instruction.g_poly_degree(C) + 1, // combined degree + eq term
                transcript,
            );
        r_z.reverse();

        let (logup_checking, r_f, r_g) =
            <Self as LogupChecking<E, PCS, Instruction, C, M>>::prove_logup_checking(
                preprocessing,
               poly,
                alpha,
                transcript,
            )?;

        end_timer!(start);

        Ok((
            LookupCheckProof {
                primary_sumcheck: SurgePrimarySumcheck {
                    sumcheck_proof: primary_sumcheck_proof,
                    num_rounds,
                    claimed_evaluation: sumcheck_claim,
                },
                logup_checking,
                commitment,
            },
            r_f,
            r_g,
            r_z,
            r_primary_sumcheck,
        ))
    }

    fn verify(
        proof: &Self::LookupCheckProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::LookupCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "lookup_check verify");

        transcript.append_serializable_element(b"primary_commitment", &proof.commitment)?;

        let instruction = Instruction::default();

        let mut r_primary_sumcheck = transcript.get_and_append_challenge_vectors(
            b"primary_sumcheck",
            proof.primary_sumcheck.num_rounds,
        )?;
        r_primary_sumcheck.reverse();

        transcript.append_field_element(
            b"sumcheck_claim",
            &proof.primary_sumcheck.claimed_evaluation,
        )?;
        let primary_sumcheck_poly_degree = instruction.g_poly_degree(C) + 1;
        let (claim_last, mut r_z) = proof.primary_sumcheck.sumcheck_proof.verify(
            proof.primary_sumcheck.claimed_evaluation,
            proof.primary_sumcheck.num_rounds,
            primary_sumcheck_poly_degree,
            transcript,
        )?;
        r_z.reverse();

        let logup_checking =
            <Self as LogupChecking<E, PCS, Instruction, C, M>>::verify_logup_checking(
                &proof.logup_checking,
                transcript,
            )?;

        end_timer!(start);
        Ok(LookupCheckSubClaim {
            r_primary_sumcheck,
            primary_sumcheck_claim: proof.primary_sumcheck.claimed_evaluation,
            r_z,
            primary_sumcheck_expected_evaluation: claim_last,
            logup_checking,
        })
    }

    // m opened to r_f
    // dim opened to r_g
    // E opened to r_g & r_z
    // witness opened to r_primary_sumcheck
    fn check_openings(
        subclaim: &Self::LookupCheckSubClaim,
        dim_openings: &[E::ScalarField],
        E_openings: &[E::ScalarField],
        m_openings: &[E::ScalarField],
        witness_openings: &[E::ScalarField],
        alpha: &E::ScalarField,
        tau: &E::ScalarField,
    ) -> Result<(), PolyIOPErrors> {
        let (beta, gamma) = subclaim.logup_checking.challenges;

        let mut f_ok = false;
        let mut g_ok = false;
        let mut primary_ok = false;
        let mut witness_ok = false;
        rayon::scope(|s| {
            s.spawn(|_| {
                let mut r_f = subclaim.logup_checking.point_f.clone();
                r_f.reverse();

                let sid: E::ScalarField = (0..r_f.len())
                    .map(|i| {
                        E::ScalarField::from_u64((r_f.len() - i - 1).pow2() as u64).unwrap()
                            * r_f[i]
                    })
                    .sum();
                let mut t = Instruction::default()
                    .subtables(C, M)
                    .par_iter()
                    .map(|(subtable, _)| subtable.evaluate_mle(&r_f))
                    .collect::<Vec<_>>();
                t.push(evaluate_mle_dechunk_operands(&r_f, *alpha));

                f_ok = subclaim
                    .logup_checking
                    .expected_evaluations_f
                    .par_iter()
                    .enumerate()
                    .all(|(i, claim)| {
                        let dim_idx = <Self as SurgeCommons<
                            E::ScalarField,
                            Instruction,
                            C,
                            M,
                        >>::memory_to_dimension_index(i);
                        let subtable_idx = <Self as SurgeCommons<
                            E::ScalarField,
                            Instruction,
                            C,
                            M,
                        >>::memory_to_subtable_index(i);

                        if claim.p != m_openings[dim_idx] {
                            return false;
                        }
                        claim.q == beta + sid + t[subtable_idx].mul_01_optimized(gamma)
                    });
            });
            s.spawn(|_| {
                g_ok = subclaim
                    .logup_checking
                    .expected_evaluations_g
                    .par_iter()
                    .enumerate()
                    .all(|(i, claim)| {
                        let dim_idx = <Self as SurgeCommons<
                            E::ScalarField,
                            Instruction,
                            C,
                            M,
                        >>::memory_to_dimension_index(i);

                        claim.p == E::ScalarField::one()
                            && claim.q == beta + dim_openings[dim_idx] + E_openings[i] * gamma
                    });
            });
            s.spawn(|_| {
                let instruction = Instruction::default();

                // Remaining part is for primary opening
                let vals = &E_openings[subclaim.logup_checking.expected_evaluations_g.len()..];
                let log_M = ark_std::log2(M) as usize;

                let vals_no_eq: &[E::ScalarField] = &vals[0..(vals.len() - C)];
                let fingerprints = &vals[vals.len() - C..];

                let eq = eq_eval(&subclaim.r_primary_sumcheck, &subclaim.r_z);
                primary_ok = if let Ok(eq_val) = eq {
                    subclaim.primary_sumcheck_expected_evaluation
                        == (instruction.combine_lookups(vals_no_eq, C, M)
                            + *alpha * concatenate_lookups(fingerprints, C, log_M / 2)
                            + *tau)
                            * eq_val
                } else {
                    false
                }
            });
            s.spawn(|_| {
                witness_ok = subclaim.primary_sumcheck_claim
                    == witness_openings[2]
                        + *alpha * (witness_openings[0] + *alpha * witness_openings[1])
                        + *tau;
            })
        });
        if f_ok && g_ok && primary_ok && witness_ok {
            Ok(())
        } else {
            Err(PolyIOPErrors::InvalidProof(format!(
                "wrong subclaim w/ check openings"
            )))
        }
    }
}

#[cfg(test)]
mod test {
    use arithmetic::evaluate_opt;
    use ark_bls12_381::Bls12_381;
    use ark_ff::UniformRand;
    use transcript::IOPTranscript;

    use super::instruction::xor::XORInstruction;
    use super::instruction::JoltInstruction;
    use super::LookupCheck;
    use crate::pcs::PolynomialCommitmentScheme;
    use crate::MultilinearKzgPCS;
    use crate::PolyIOP;
    use ark_ec::pairing::Pairing;
    use ark_std::test_rng;

    fn test_helper<
        E: Pairing,
        Instruction: JoltInstruction + Default,
        const C: usize,
        const M: usize,
    >(
        ops: &[Instruction],
    ) {
        let mut transcript = IOPTranscript::new(b"test_transcript");
        let preprocessing = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::preprocess();

        let mut rng = test_rng();
        let srs = MultilinearKzgPCS::<E>::gen_srs_for_testing(&mut rng, 10).unwrap();
        let (pcs_param, _) = MultilinearKzgPCS::<E>::trim(&srs, None, Some(10)).unwrap();

        let alpha = E::ScalarField::rand(&mut rng);
        let tau = E::ScalarField::rand(&mut rng);

        let witnesses = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::construct_witnesses(ops);
        let mut poly = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::construct_polys(&preprocessing, ops, &alpha);
        let (proof, r_f, r_g, r_z, r_primary_sumcheck) = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::prove(
            &preprocessing,
            &pcs_param,
            &mut poly,
            &alpha,
            &tau,
            &mut transcript,
        )
        .unwrap();

        let mut transcript = IOPTranscript::new(b"test_transcript");
        let subclaim = <PolyIOP<E::ScalarField> as LookupCheck<
            E,
            MultilinearKzgPCS<E>,
            Instruction,
            C,
            M,
        >>::verify(&proof, &mut transcript)
        .unwrap();
    
        assert_eq!(subclaim.r_primary_sumcheck, r_primary_sumcheck);
        assert_eq!(subclaim.r_z, r_z);
        assert_eq!(subclaim.logup_checking.point_f, r_f);
        assert_eq!(subclaim.logup_checking.point_g, r_g);

        let m_openings = poly
            .m
            .iter()
            .map(|poly| evaluate_opt(poly, &r_f))
            .collect::<Vec<_>>();
        let dim_openings = poly
            .dim
            .iter()
            .map(|poly| evaluate_opt(poly, &r_g))
            .collect::<Vec<_>>();
        let E_openings = poly
            .E_polys
            .iter()
            .map(|poly| evaluate_opt(poly, &r_g))
            .chain(poly.E_polys.iter().map(|poly| evaluate_opt(poly, &r_z)))
            .collect::<Vec<_>>();
        let witness_openings = witnesses
            .iter()
            .map(|poly| evaluate_opt(poly, &r_primary_sumcheck))
            .collect::<Vec<_>>();

        <PolyIOP<E::ScalarField> as LookupCheck<E, MultilinearKzgPCS<E>, Instruction, C, M>>::check_openings(&subclaim, &dim_openings, &E_openings, &m_openings, &witness_openings, &alpha, &tau).unwrap();
    }

    #[test]
    fn e2e() {
        let ops = vec![
            XORInstruction(12, 12),
            XORInstruction(12, 82),
            XORInstruction(12, 12),
            XORInstruction(25, 12),
        ];
        const C: usize = 8;
        const M: usize = 1 << 8;
        test_helper::<Bls12_381, XORInstruction, C, M>(&ops);
    }

    #[test]
    fn e2e_non_pow_2() {
        let ops = vec![
            XORInstruction(0, 1),
            XORInstruction(101, 101),
            XORInstruction(202, 1),
            XORInstruction(220, 1),
            XORInstruction(220, 1),
        ];
        const C: usize = 2;
        const M: usize = 1 << 8;
        test_helper::<Bls12_381, XORInstruction, C, M>(&ops);
    }
}
