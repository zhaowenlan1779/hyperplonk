// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Main module for the Permutation Check protocol

use crate::poly_iop::{
    errors::PolyIOPErrors,
    rational_sumcheck::layered_circuit::{
        BatchedDenseRationalSum, BatchedRationalSum, BatchedRationalSumProof,
    },
    PolyIOP,
};
use arithmetic::{math::Math, Fraction};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use itertools::izip;
use rayon::iter::{IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::iter::zip;
use std::mem::take;
use std::sync::Arc;
use transcript::IOPTranscript;
use util::compute_leaves;

pub struct PermutationCheckProofSingle<F>
where
    F: PrimeField,
{
    pub proof: BatchedRationalSumProof<F>,
    pub f_claims: Vec<Fraction<F>>,
    pub g_claims: Vec<Fraction<F>>,
}

pub struct PermutationCheckProof<F>
where
    F: PrimeField,
{
    pub proofs: Vec<PermutationCheckProofSingle<F>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PermutationCheckSubClaimSingle<F>
where
    F: PrimeField,
{
    pub point: Vec<F>,
    pub expected_evaluations: Vec<F>,
    pub len: usize,
}

/// A permutation subclaim consists of
/// - the SubClaim from the ProductCheck
/// - Challenges beta and gamma
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PermutationCheckSubClaim<F>
where
    F: PrimeField,
{
    pub subclaims: Vec<PermutationCheckSubClaimSingle<F>>,

    /// Challenges beta and gamma
    pub challenges: (F, F),
}

pub mod util;

/// A PermutationCheck w.r.t. `(fs, gs, perms)`
/// proves that (g1, ..., gk) is a permutation of (f1, ..., fk) under
/// permutation `(p1, ..., pk)`
/// It is derived from ProductCheck.
///
/// A Permutation Check IOP takes the following steps:
///
/// Inputs:
/// - fs = (f1, ..., fk)
/// - gs = (g1, ..., gk)
/// - permutation oracles = (p1, ..., pk)
pub trait PermutationCheck<F>
where
    F: PrimeField,
{
    type PermutationCheckSubClaim;
    type PermutationProof;

    type MultilinearExtension;
    type Transcript;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a PermutationCheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// PermutationCheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    /// Inputs:
    /// - fs = (f1, ..., fk)
    /// - gs = (g1, ..., gk)
    /// - permutation oracles = (p1, ..., pk)
    /// Outputs:
    /// - a permutation check proof proving that gs is a permutation of fs under
    ///   permutation
    ///
    /// Cost: O(N)
    #[allow(clippy::type_complexity)]
    fn prove(
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(Self::PermutationProof, Vec<Vec<F>>), PolyIOPErrors>;

    /// Verify that (g1, ..., gk) is a permutation of
    /// (f1, ..., fk) over the permutation oracles (perm1, ..., permk)
    fn verify(
        proof: &Self::PermutationProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors>;

    fn check_openings(
        subclaim: &Self::PermutationCheckSubClaim,
        f_openings: &[F],
        g_openings: &[F],
        perm_openings: &[F],
    ) -> Result<(), PolyIOPErrors>;
}

impl<F> PermutationCheck<F> for PolyIOP<F>
where
    F: PrimeField,
{
    type PermutationCheckSubClaim = PermutationCheckSubClaim<F>;
    type PermutationProof = PermutationCheckProof<F>;
    type MultilinearExtension = Arc<DenseMultilinearExtension<F>>;
    type Transcript = IOPTranscript<F>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<F>::new(b"Initializing PermutationCheck transcript")
    }

    fn prove(
        fxs: &[Self::MultilinearExtension],
        gxs: &[Self::MultilinearExtension],
        perms: &[Self::MultilinearExtension],
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(Self::PermutationProof, Vec<Vec<F>>), PolyIOPErrors> {
        let start = start_timer!(|| "Permutation check prove");
        if fxs.is_empty() {
            return Err(PolyIOPErrors::InvalidParameters("fxs is empty".to_string()));
        }
        if (fxs.len() != gxs.len()) || (fxs.len() != perms.len()) {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "fxs.len() = {}, gxs.len() = {}, perms.len() = {}",
                fxs.len(),
                gxs.len(),
                perms.len(),
            )));
        }

        // generate challenge `beta` and `gamma` from current transcript
        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;
        let mut leaves = compute_leaves(&beta, &gamma, fxs, gxs, perms)?;

        let mut to_prove = leaves
            .par_iter_mut()
            .map(|mut leave| {
                let leave_len = leave.len();
                let batched_circuit =
                    <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F>>::construct((
                        vec![vec![F::one(); leave[0].len()]; leave.len()],
                        take(&mut leave),
                    ));
                let mut f_claims = <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F>>::claims(
                    &batched_circuit,
                );
                let g_claims = f_claims.split_off(leave_len / 2);
                (batched_circuit, f_claims, g_claims)
            })
            .collect::<Vec<_>>();

        let mut proofs = Vec::with_capacity(to_prove.len());
        let mut points = Vec::with_capacity(to_prove.len());
        for (batched_circuit, f_claims, g_claims) in to_prove.iter_mut() {
            let (proof, point) =
                <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F>>::prove_rational_sum(
                    batched_circuit,
                    transcript,
                );
            proofs.push(PermutationCheckProofSingle {
                proof,
                f_claims: take(f_claims),
                g_claims: take(g_claims),
            });
            points.push(point);
        }

        end_timer!(start);
        Ok((Self::PermutationProof { proofs }, points))
    }

    fn verify(
        proof: &Self::PermutationProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "Permutation check verify");

        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        let (sum_f, sum_g) = proof
            .proofs
            .par_iter()
            .map(|proof| {
                let sum_f = proof
                    .f_claims
                    .iter()
                    .fold(Fraction::zero(), |acc, x| Fraction::rational_add(acc, *x));
                let sum_g = proof
                    .g_claims
                    .iter()
                    .fold(Fraction::zero(), |acc, x| Fraction::rational_add(acc, *x));
                (sum_f, sum_g)
            })
            .reduce(
                || (Fraction::zero(), Fraction::zero()),
                |(acc_f, acc_g), (f, g)| {
                    (
                        Fraction::rational_add(acc_f, f),
                        Fraction::rational_add(acc_g, g),
                    )
                },
            );

        if sum_f.p * sum_g.q != sum_g.p * sum_f.q {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "Permutation check claims are inconsistent"
            )));
        }

        let mut subclaims = Vec::with_capacity(proof.proofs.len());
        for proof in proof.proofs.iter() {
            let (claims, point) =
                <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F>>::verify_rational_sum(
                    &proof.proof,
                    &[&proof.f_claims[..], &proof.g_claims[..]].concat(),
                    transcript,
                );

            if claims.iter().any(|claim| claim.p != F::one()) {
                return Err(PolyIOPErrors::InvalidProof(format!(
                    "Permutation check claim opened to non-1 value on numerator"
                )));
            }
            subclaims.push(PermutationCheckSubClaimSingle {
                point,
                expected_evaluations: claims.iter().map(|claim| claim.q).collect(),
                len: proof.f_claims.len(),
            });
        }

        end_timer!(start);
        Ok(PermutationCheckSubClaim {
            subclaims,
            challenges: (beta, gamma),
        })
    }

    fn check_openings(
        subclaim: &Self::PermutationCheckSubClaim,
        f_openings: &[F],
        g_openings: &[F],
        perm_openings: &[F],
    ) -> Result<(), PolyIOPErrors> {
        let (beta, gamma) = subclaim.challenges;

        let mut shift = 0;
        let mut offset = 0;
        for subclaim in subclaim.subclaims.iter() {
            let num_vars = subclaim.point.len();
            let sid: F = (0..num_vars)
                .map(|i| F::from_u64(i.pow2() as u64).unwrap() * subclaim.point[i])
                .sum::<F>()
                + F::from_u64(shift as u64).unwrap();

            // check subclaim
            if subclaim.len * 2 != subclaim.expected_evaluations.len() {
                return Err(PolyIOPErrors::InvalidVerifier(
                    "wrong subclaim lengthes".to_string(),
                ));
            }

            let subclaim_valid = zip(
                f_openings[offset..offset + subclaim.len].iter(),
                subclaim.expected_evaluations[..subclaim.len].iter(),
            )
            .enumerate()
            .all(|(i, (f_eval, expected_evaluation))| {
                *f_eval + beta * (sid + F::from((i * (1 << num_vars)) as u64)) + gamma
                    == *expected_evaluation
            }) && izip!(
                g_openings[offset..offset + subclaim.len].iter(),
                perm_openings[offset..offset + subclaim.len].iter(),
                subclaim.expected_evaluations[subclaim.len..].iter(),
            )
            .all(|(g_eval, perm_eval, expected_evaluation)| {
                *g_eval + beta * perm_eval + gamma == *expected_evaluation
            });

            if !subclaim_valid {
                return Err(PolyIOPErrors::InvalidVerifier("wrong subclaim".to_string()));
            }

            shift += subclaim.len * num_vars.pow2();
            offset += subclaim.len;
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::PermutationCheck;
    use crate::poly_iop::{errors::PolyIOPErrors, PolyIOP};
    use arithmetic::{
        evaluate_opt, identity_permutation_mle, identity_permutation_mles, math::Math,
        random_permutation_u64,
    };
    use ark_bls12_381::Fr;
    use ark_ff::PrimeField;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use rand_core::RngCore;
    use std::sync::Arc;

    fn test_permutation_check_helper<F>(
        fxs: &[Arc<DenseMultilinearExtension<F>>],
        gxs: &[Arc<DenseMultilinearExtension<F>>],
        perms: &[Arc<DenseMultilinearExtension<F>>],
    ) -> Result<(), PolyIOPErrors>
    where
        F: PrimeField,
    {
        // prover
        let mut transcript = <PolyIOP<F> as PermutationCheck<F>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let (proof, _) =
            <PolyIOP<F> as PermutationCheck<F>>::prove(fxs, gxs, perms, &mut transcript)?;

        // verifier
        let mut transcript = <PolyIOP<F> as PermutationCheck<F>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let perm_check_sub_claim =
            <PolyIOP<F> as PermutationCheck<F>>::verify(&proof, &mut transcript)?;

        let mut f_openings = vec![];
        let mut g_openings = vec![];
        let mut perm_openings = vec![];
        let mut offset = 0;
        for subclaim in perm_check_sub_claim.subclaims.iter() {
            let mut f_evals = fxs[offset..offset + subclaim.len]
                .iter()
                .map(|f| evaluate_opt(f, &subclaim.point))
                .collect::<Vec<_>>();
            let mut g_evals = gxs[offset..offset + subclaim.len]
                .iter()
                .map(|g| evaluate_opt(g, &subclaim.point))
                .collect::<Vec<_>>();
            let mut perm_evals = perms[offset..offset + subclaim.len]
                .iter()
                .map(|perm| evaluate_opt(perm, &subclaim.point))
                .collect::<Vec<_>>();

            f_openings.append(&mut f_evals);
            g_openings.append(&mut g_evals);
            perm_openings.append(&mut perm_evals);
            offset += subclaim.len;
        }

        <PolyIOP<F> as PermutationCheck<F>>::check_openings(
            &perm_check_sub_claim,
            &f_openings,
            &g_openings,
            &perm_openings,
        )
    }

    fn generate_polys<R: RngCore>(
        nv: &[usize],
        rng: &mut R,
    ) -> Vec<Arc<DenseMultilinearExtension<Fr>>> {
        nv.iter()
            .map(|x| Arc::new(DenseMultilinearExtension::rand(*x, rng)))
            .collect()
    }

    fn test_permutation_check(
        nv: Vec<usize>,
        id_perms: Vec<Arc<DenseMultilinearExtension<Fr>>>,
    ) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();

        {
            // good path: (w1, w2) is a permutation of (w1, w2) itself under the identify
            // map
            let ws = generate_polys(&nv, &mut rng);
            // perms is the identity map
            test_permutation_check_helper::<Fr>(&ws, &ws, &id_perms)?;
        }

        {
            let fs = generate_polys(&nv, &mut rng);

            let size0 = nv[0].pow2();

            let perm = random_permutation_u64(nv[0].pow2() + nv[1].pow2(), &mut rng);
            let perms = vec![
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv[0],
                    perm[..size0]
                        .iter()
                        .map(|x| Fr::from_u64(*x).unwrap())
                        .collect(),
                )),
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv[1],
                    perm[size0..]
                        .iter()
                        .map(|x| Fr::from_u64(*x).unwrap())
                        .collect(),
                )),
            ];

            let get_f = |index: usize| {
                if index < size0 {
                    fs[0].evaluations[index]
                } else {
                    fs[1].evaluations[index - size0]
                }
            };

            let g_evals = (
                (0..size0)
                    .map(|x| get_f(perm[x] as usize))
                    .collect::<Vec<_>>(),
                (size0..size0 + nv[1].pow2())
                    .map(|x| get_f(perm[x] as usize))
                    .collect::<Vec<_>>(),
            );
            let gs = vec![
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv[0], g_evals.0,
                )),
                Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                    nv[1], g_evals.1,
                )),
            ];
            test_permutation_check_helper::<Fr>(&fs, &gs, &perms)?;
        }

        {
            // bad path 1: w is a not permutation of w itself under a random map
            let ws = generate_polys(&nv, &mut rng);
            // perms is a random map
            let perms = id_perms
                .iter()
                .map(|perm| {
                    let mut evals = perm.evaluations.clone();
                    evals.reverse();
                    Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                        perm.num_vars(),
                        evals,
                    ))
                })
                .collect::<Vec<_>>();

            assert!(test_permutation_check_helper::<Fr>(&ws, &ws, &perms).is_err());
        }

        {
            // bad path 2: f is a not permutation of g under a identity map
            let fs = generate_polys(&nv, &mut rng);
            let gs = generate_polys(&nv, &mut rng);
            // s_perm is the identity map

            assert!(test_permutation_check_helper::<Fr>(&fs, &gs, &id_perms).is_err());
        }

        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        let id_perms = identity_permutation_mles(1, 2);
        test_permutation_check(vec![1, 1], id_perms)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        let id_perms = identity_permutation_mles(5, 2);
        test_permutation_check(vec![5, 5], id_perms)
    }

    #[test]
    fn test_different_lengths() -> Result<(), PolyIOPErrors> {
        let id_perms = vec![
            identity_permutation_mle(0, 5),
            identity_permutation_mle(32, 4),
        ];
        test_permutation_check(vec![5, 4], id_perms)
    }
}
