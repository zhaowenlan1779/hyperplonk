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
use arithmetic::Fraction;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use std::sync::Arc;
use transcript::IOPTranscript;
use util::compute_leaves;

pub struct PermutationCheckProof<F>
where
    F: PrimeField,
{
    pub proof: BatchedRationalSumProof<F>,
    pub f_claims: Vec<Fraction<F>>,
    pub g_claims: Vec<Fraction<F>>,
}

/// A permutation subclaim consists of
/// - the SubClaim from the ProductCheck
/// - Challenges beta and gamma
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PermutationCheckSubClaim<F>
where
    F: PrimeField,
{
    pub point: Vec<F>,
    pub expected_evaluations: Vec<F>,

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
    ) -> Result<(Self::PermutationProof, Vec<F>), PolyIOPErrors>;

    /// Verify that (g1, ..., gk) is a permutation of
    /// (f1, ..., fk) over the permutation oracles (perm1, ..., permk)
    fn verify(
        proof: &Self::PermutationProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors>;
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
    ) -> Result<(Self::PermutationProof, Vec<F>), PolyIOPErrors> {
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

        let num_vars = fxs[0].num_vars;
        for ((fx, gx), perm) in fxs.iter().zip(gxs.iter()).zip(perms.iter()) {
            if (fx.num_vars != num_vars) || (gx.num_vars != num_vars) || (perm.num_vars != num_vars)
            {
                return Err(PolyIOPErrors::InvalidParameters(
                    "number of variables unmatched".to_string(),
                ));
            }
        }

        // generate challenge `beta` and `gamma` from current transcript
        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;
        let (mut f_leaves, mut g_leaves) = compute_leaves(&beta, &gamma, fxs, gxs, perms)?;
        f_leaves.append(&mut g_leaves);

        let mut batched_circuit =
            <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F>>::construct((
                vec![vec![F::one(); f_leaves[0].len()]; f_leaves.len()],
                f_leaves,
            ));
        let mut f_claims =
            <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F>>::claims(&batched_circuit);
        let g_claims = f_claims.split_off(fxs.len());
    
        let (proof, point) =
            <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F>>::prove_rational_sum(
                &mut batched_circuit,
                transcript,
            );

        end_timer!(start);
        Ok((Self::PermutationProof {
            proof,
            f_claims,
            g_claims,
        }, point))
    }

    fn verify(
        proof: &Self::PermutationProof,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::PermutationCheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "Permutation check verify");

        let beta = transcript.get_and_append_challenge(b"beta")?;
        let gamma = transcript.get_and_append_challenge(b"gamma")?;

        let sum_f = proof.f_claims.iter().fold(Fraction::zero(), |acc, x| Fraction::rational_add(acc, *x));
        let sum_g = proof.g_claims.iter().fold(Fraction::zero(), |acc, x| Fraction::rational_add(acc, *x));
        if sum_f.p * sum_g.q != sum_g.p * sum_f.q {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "Permutation check claims are inconsistent"
            )));
        }

        let (claims, point) =
            <BatchedDenseRationalSum<F, 1> as BatchedRationalSum<F>>::verify_rational_sum(
                &proof.proof,
                &[&proof.f_claims[..], &proof.g_claims[..]].concat(),
                transcript,
            );

        if claims.iter().any(|claim| claim.p != F::one())
        {
            return Err(PolyIOPErrors::InvalidProof(format!(
                "Permutation check claim opened to non-1 value on numerator"
            )));
        }

        end_timer!(start);
        Ok(PermutationCheckSubClaim {
            point,
            expected_evaluations: claims.iter().map(|claim| claim.q).collect(),
            challenges: (beta, gamma),
        })
    }
}

#[cfg(test)]
mod test {
    use super::PermutationCheck;
    use crate::poly_iop::{errors::PolyIOPErrors, PolyIOP};
    use arithmetic::{
        evaluate_opt, identity_permutation_mles, math::Math, random_permutation_mles,
    };
    use ark_bls12_381::Fr;
    use ark_ff::PrimeField;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::test_rng;
    use itertools::izip;
    use std::iter::zip;
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
        let (proof, _) = <PolyIOP<F> as PermutationCheck<F>>::prove(fxs, gxs, perms, &mut transcript)?;

        // verifier
        let mut transcript = <PolyIOP<F> as PermutationCheck<F>>::init_transcript();
        transcript.append_message(b"testing", b"initializing transcript for testing")?;
        let perm_check_sub_claim =
            <PolyIOP<F> as PermutationCheck<F>>::verify(&proof, &mut transcript)?;

        let (beta, gamma) = perm_check_sub_claim.challenges;

        let num_vars = fxs[0].num_vars();
        let sid: F = (0..num_vars)
            .map(|i| {
                F::from_u64(i.pow2() as u64).unwrap() * perm_check_sub_claim.point[i]
            })
            .sum();

        // check subclaim
        if fxs.len() + gxs.len() != perm_check_sub_claim.expected_evaluations.len()
        {
            return Err(PolyIOPErrors::InvalidVerifier(
                "wrong subclaim lengthes".to_string(),
            ));
        }

        let subclaim_valid =
        zip(
            fxs.iter(),
            perm_check_sub_claim.expected_evaluations[..fxs.len()].iter(),
        )
        .enumerate()
        .all(|(i, (poly, expected_evaluation))| {
            evaluate_opt(poly, &perm_check_sub_claim.point)
                + beta * (sid + F::from((i * (1 << num_vars)) as u64))
                + gamma
                == *expected_evaluation
        })
         && izip!(
            gxs.iter(),
            perms.iter(),
            perm_check_sub_claim.expected_evaluations[fxs.len()..].iter(),
        )
        .all(|(poly, perm, expected_evaluation)| {
            evaluate_opt(poly, &perm_check_sub_claim.point)
                + beta * evaluate_opt(perm, &perm_check_sub_claim.point)
                + gamma
                == *expected_evaluation
        });

        if !subclaim_valid {
            return Err(PolyIOPErrors::InvalidVerifier("wrong subclaim".to_string()));
        }

        Ok(())
    }

    fn test_permutation_check(nv: usize) -> Result<(), PolyIOPErrors> {
        let mut rng = test_rng();
        let id_perms = identity_permutation_mles(nv, 2);

        {
            // good path: (w1, w2) is a permutation of (w1, w2) itself under the identify
            // map
            let ws = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            // perms is the identity map
            test_permutation_check_helper::<Fr>(&ws, &ws, &id_perms)?;
        }

        {
            // good path: f = (w1, w2) is a permutation of g = (w2, w1) itself under a map
            let mut fs = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            let gs = fs.clone();
            fs.reverse();
            // perms is the reverse identity map
            let mut perms = id_perms.clone();
            perms.reverse();
            test_permutation_check_helper::<Fr>(&fs, &gs, &perms)?;
        }

        {
            // bad path 1: w is a not permutation of w itself under a random map
            let ws = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            // perms is a random map
            let perms = random_permutation_mles(nv, 2, &mut rng);

            assert!(test_permutation_check_helper::<Fr>(&ws, &ws, &perms).is_err());
        }

        {
            // bad path 2: f is a not permutation of g under a identity map
            let fs = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            let gs = vec![
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
                Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)),
            ];
            // s_perm is the identity map

            assert!(test_permutation_check_helper::<Fr>(&fs, &gs, &id_perms).is_err());
        }

        Ok(())
    }

    #[test]
    fn test_trivial_polynomial() -> Result<(), PolyIOPErrors> {
        test_permutation_check(1)
    }
    #[test]
    fn test_normal_polynomial() -> Result<(), PolyIOPErrors> {
        test_permutation_check(5)
    }
}
