// Copyright (c) Jolt Project
// Copyright (c) 2023 HyperPlonk Project
// Copyright (c) 2024 Pengfei Zhu (HyperPianist Project)

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! This module implements batched cubic sumchecks

use crate::poly_iop::sum_check::generic_sumcheck::SumcheckInstanceProof;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use transcript::IOPTranscript;
use arithmetic::unipoly::{CompressedUniPoly, UniPoly};

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
