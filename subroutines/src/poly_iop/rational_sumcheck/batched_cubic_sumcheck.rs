// Copyright (c) Jolt Project
// Copyright (c) 2023 HyperPlonk Project
// Copyright (c) 2024 Pengfei Zhu (HyperPianist Project)

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements the sum check protocol.

use crate::poly_iop::errors::PolyIOPErrors;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::fmt::Debug;
use transcript::IOPTranscript;
use arithmetic::unipoly::{CompressedUniPoly, UniPoly};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug)]
pub struct SumcheckInstanceProof<F: PrimeField> {
    compressed_polys: Vec<CompressedUniPoly<F>>,
}

impl<F: PrimeField> SumcheckInstanceProof<F> {
    pub fn new(compressed_polys: Vec<CompressedUniPoly<F>>) -> SumcheckInstanceProof<F> {
        SumcheckInstanceProof { compressed_polys }
    }

    /// Verify this sumcheck proof.
    /// Note: Verification does not execute the final check of sumcheck protocol: g_v(r_v) = oracle_g(r),
    /// as the oracle is not passed in. Expected that the caller will implement.
    ///
    /// Params
    /// - `claim`: Claimed evaluation
    /// - `num_rounds`: Number of rounds of sumcheck, or number of variables to bind
    /// - `degree_bound`: Maximum allowed degree of the combined univariate polynomial
    /// - `transcript`: Fiat-shamir transcript
    ///
    /// Returns (e, r)
    /// - `e`: Claimed evaluation at random point
    /// - `r`: Evaluation point
    /// #[tracing::instrument(skip_all, name = "Sumcheck::verify")]
    pub fn verify(
        &self,
        claim: F,
        num_rounds: usize,
        degree_bound: usize,
        transcript: &mut IOPTranscript<F>,
    ) -> Result<(F, Vec<F>), PolyIOPErrors> {
        let mut e = claim;
        let mut r: Vec<F> = Vec::new();

        // verify that there is a univariate polynomial for each round
        assert_eq!(self.compressed_polys.len(), num_rounds);
        for i in 0..self.compressed_polys.len() {
            let poly = self.compressed_polys[i].decompress(&e);

            // verify degree bound
            if poly.degree() != degree_bound {
                return Err(PolyIOPErrors::InvalidProof(
                    format!("degree_bound = {}, poly.degree() = {}",
                    degree_bound,
                    poly.degree(),
                )
                ));
            }

            // check if G_k(0) + G_k(1) = e
            assert_eq!(poly.eval_at_zero() + poly.eval_at_one(), e);

            // append the prover's message to the transcript
            transcript.append_serializable_element(b"poly", &poly)?;

            //derive the verifier's challenge for the next round
            let r_i = transcript.get_and_append_challenge(b"challenge_nextround")?;

            r.push(r_i);

            // evaluate the claimed degree-ell polynomial at r_i
            e = poly.evaluate(&r_i);
        }

        Ok((e, r))
    }
}

// A cubic sumcheck instance that is not represented as virtual polynomials.
// Instead the struct itself can hold arbitrary state as long as it can bind varaibles
// and produce a cubic polynomial on demand.
// Used by the layered circuit implementation for rational sumcheck
pub trait BatchedCubicSumcheckInstance<F: PrimeField, FinalClaimType> : Sync {
    fn num_rounds(&self) -> usize;
    fn bind(&mut self, eq_poly: &mut DenseMultilinearExtension<F>, r: &F);
    fn compute_cubic(
        &self,
        coeffs: &[F],
        eq_poly: &DenseMultilinearExtension<F>,
        previous_round_claim: F,
        lambda: &F,
    ) -> UniPoly<F>;
    fn final_claims(&self) -> (Vec<FinalClaimType>, Vec<FinalClaimType>);

    // #[tracing::instrument(skip_all, name = "BatchedCubicSumcheck::prove_sumcheck")]
    fn prove_sumcheck(
        &mut self,
        claim: &F,
        coeffs: &[F],
        eq_poly: &mut DenseMultilinearExtension<F>,
        transcript: &mut IOPTranscript<F>,
        lambda: &F,
    ) -> (SumcheckInstanceProof<F>, Vec<F>, (Vec<FinalClaimType>, Vec<FinalClaimType>)) {
        debug_assert_eq!(eq_poly.num_vars, self.num_rounds());

        let mut previous_claim = *claim;
        let mut r: Vec<F> = Vec::new();
        let mut cubic_polys: Vec<CompressedUniPoly<F>> = Vec::new();

        for _round in 0..self.num_rounds() {
            let cubic_poly = self.compute_cubic(coeffs, eq_poly, previous_claim, lambda);
            // append the prover's message to the transcript
            transcript.append_serializable_element(b"poly", &cubic_poly).unwrap();
            //derive the verifier's challenge for the next round
            let r_j = transcript.get_and_append_challenge(b"challenge_nextround").unwrap();

            r.push(r_j);
            // bind polynomials to verifier's challenge
            self.bind(eq_poly, &r_j);

            previous_claim = cubic_poly.evaluate(&r_j);
            cubic_polys.push(cubic_poly.compress());
        }

        debug_assert_eq!(eq_poly.evaluations.len(), 1);

        (
            SumcheckInstanceProof::new(cubic_polys),
            r,
            self.final_claims(),
        )
    }
}
