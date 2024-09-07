//! This module implements the rational sum check protocol with or without
//! layered circuits

use crate::{
    pcs::PolynomialCommitmentScheme,
    poly_iop::{errors::PolyIOPErrors, zero_check::ZeroCheck, PolyIOP},
    SumCheck,
};
use arithmetic::VirtualPolynomial;
use ark_ec::pairing::Pairing;
use ark_ff::{batch_inversion, One, PrimeField};
use ark_poly::DenseMultilinearExtension;
use ark_std::{end_timer, start_timer};
use std::sync::Arc;
use transcript::IOPTranscript;

pub mod layered_circuit;

/// Non-layered-circuit version of RationalSumcheck.
/// Proves that \sum p(x) / q(x) = v.
pub trait RationalSumcheckSlow<E, PCS>: ZeroCheck<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    type RationalSumcheckSubClaim;
    type RationalSumcheckProof;

    /// Initialize the system with a transcript
    ///
    /// This function is optional -- in the case where a RationalSumcheck is
    /// an building block for a more complex protocol, the transcript
    /// may be initialized by this complex protocol, and passed to the
    /// RationalSumcheck prover/verifier.
    fn init_transcript() -> Self::Transcript;

    fn extract_sum(proof: &Self::RationalSumcheckProof) -> E::ScalarField;

    /// Returns (proof, inv_g) for testing
    #[allow(clippy::type_complexity)]
    fn prove(
        pcs_param: &PCS::ProverParam,
        fx: &Self::MultilinearExtension,
        gx: &Self::MultilinearExtension,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(Self::RationalSumcheckProof, Self::MultilinearExtension), PolyIOPErrors>;

    /// Verify that for witness multilinear polynomials (f1, ..., fk, g1, ...,
    /// gk) it holds that
    ///      `\prod_{x \in {0,1}^n} f1(x) * ... * fk(x)
    ///     = \prod_{x \in {0,1}^n} g1(x) * ... * gk(x)`
    fn verify(
        claimed_sum: E::ScalarField,
        proof: &Self::RationalSumcheckProof,
        aux_info_zero_check: &Self::VPAuxInfo,
        aux_info_sum_check: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::RationalSumcheckSubClaim, PolyIOPErrors>;
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RationalSumcheckSubClaim<F: PrimeField, ZC: ZeroCheck<F>, SC: SumCheck<F>> {
    // the SubClaim from the ZeroCheck
    pub zero_check_sub_claim: ZC::ZeroCheckSubClaim,

    // the SubClaim from the SumCheck
    pub sum_check_sub_claim: SC::SumCheckSubClaim,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RationalSumcheckProof<
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
    ZC: ZeroCheck<E::ScalarField>,
    SC: SumCheck<E::ScalarField>,
> {
    pub zero_check_proof: ZC::ZeroCheckProof,
    pub sum_check_proof: SC::SumCheckProof,
    pub inv_comm: PCS::Commitment,
}

impl<E, PCS> RationalSumcheckSlow<E, PCS> for PolyIOP<E::ScalarField>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>>,
{
    type RationalSumcheckSubClaim = RationalSumcheckSubClaim<E::ScalarField, Self, Self>;
    type RationalSumcheckProof = RationalSumcheckProof<E, PCS, Self, Self>;

    fn init_transcript() -> Self::Transcript {
        IOPTranscript::<E::ScalarField>::new(b"Initializing RationalSumcheck transcript")
    }

    fn extract_sum(proof: &Self::RationalSumcheckProof) -> E::ScalarField {
        <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::extract_sum(&proof.sum_check_proof)
    }

    fn prove(
        pcs_param: &PCS::ProverParam,
        fx: &Self::MultilinearExtension,
        gx: &Self::MultilinearExtension,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<(Self::RationalSumcheckProof, Self::MultilinearExtension), PolyIOPErrors> {
        let start = start_timer!(|| "rational_sumcheck prove");

        let mut g_inv = Arc::new(DenseMultilinearExtension::clone(gx));
        batch_inversion(&mut Arc::get_mut(&mut g_inv).unwrap().evaluations);

        let inv_comm = PCS::commit(pcs_param, &g_inv)?;
        transcript.append_serializable_element(b"inv(g(x))", &inv_comm)?;

        let mut prod = VirtualPolynomial::new(gx.num_vars);
        prod.add_mle_list([gx.clone(), g_inv.clone()], E::ScalarField::one())?;
        prod.add_mle_list([], -E::ScalarField::one())?;

        let zero_check_proof =
            <PolyIOP<E::ScalarField> as ZeroCheck<E::ScalarField>>::prove(&prod, transcript)?;

        let mut quot = VirtualPolynomial::new(gx.num_vars);
        quot.add_mle_list([fx.clone(), g_inv.clone()], E::ScalarField::one())?;

        let sum_check_proof =
            <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(&quot, transcript)?;

        end_timer!(start);

        Ok((
            RationalSumcheckProof {
                zero_check_proof,
                sum_check_proof,
                inv_comm,
            },
            g_inv,
        ))
    }

    fn verify(
        claimed_sum: E::ScalarField,
        proof: &Self::RationalSumcheckProof,
        aux_info_zero_check: &Self::VPAuxInfo,
        aux_info_sum_check: &Self::VPAuxInfo,
        transcript: &mut Self::Transcript,
    ) -> Result<Self::RationalSumcheckSubClaim, PolyIOPErrors> {
        let start = start_timer!(|| "rational_sumcheck verify");

        // update transcript and generate challenge
        transcript.append_serializable_element(b"inv(g(x))", &proof.inv_comm)?;

        // invoke the zero check on the iop_proof
        // the virtual poly info for Q(x)
        let zero_check_sub_claim = <Self as ZeroCheck<E::ScalarField>>::verify(
            &proof.zero_check_proof,
            aux_info_zero_check,
            transcript,
        )?;

        let sum_check_sub_claim = <Self as SumCheck<E::ScalarField>>::verify(
            claimed_sum,
            &proof.sum_check_proof,
            aux_info_sum_check,
            transcript,
        )?;

        end_timer!(start);

        Ok(RationalSumcheckSubClaim {
            zero_check_sub_claim,
            sum_check_sub_claim,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::pcs::prelude::MultilinearKzgPCS;
    use arithmetic::VPAuxInfo;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly::MultilinearExtension;
    use ark_std::{test_rng, UniformRand, Zero};
    use std::{iter::zip, marker::PhantomData};

    #[test]
    fn test_rational_sumcheck() -> Result<(), PolyIOPErrors> {
        let num_vars = 5;

        let mut rng = test_rng();
        let srs = MultilinearKzgPCS::<Bls12_381>::gen_srs_for_testing(&mut rng, num_vars)?;
        let (pcs_param, _) = MultilinearKzgPCS::<Bls12_381>::trim(&srs, None, Some(num_vars))?;

        let evals_p = std::iter::repeat_with(|| Fr::rand(&mut rng))
            .take(1 << num_vars)
            .collect();
        let p = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals_p,
        ));

        let evals_q = std::iter::repeat_with(|| {
            let mut val = Fr::zero();
            while val == Fr::zero() {
                val = Fr::rand(&mut rng);
            }
            val
        })
        .take(1 << num_vars)
        .collect();
        let q = Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            num_vars, evals_q,
        ));

        let expected_sum = zip(p.evaluations.iter(), q.evaluations.iter())
            .map(|(p, q)| p / q)
            .sum::<Fr>();

        let mut transcript = <PolyIOP<Fr> as RationalSumcheckSlow<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::init_transcript();
        let (proof, inv_g) = <PolyIOP<Fr> as RationalSumcheckSlow<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::prove(&pcs_param, &p, &q, &mut transcript)?;

        assert_eq!(<PolyIOP<Fr> as RationalSumcheckSlow<Bls12_381, MultilinearKzgPCS<Bls12_381>>>::extract_sum(&proof), expected_sum);

        // Apparently no one knows what's this for?
        let aux_info = VPAuxInfo {
            max_degree: 2,
            num_variables: num_vars,
            phantom: PhantomData::default(),
        };
        let mut transcript = <PolyIOP<Fr> as RationalSumcheckSlow<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::init_transcript();
        let subclaim = <PolyIOP<Fr> as RationalSumcheckSlow<
            Bls12_381,
            MultilinearKzgPCS<Bls12_381>,
        >>::verify(
            expected_sum, &proof, &aux_info, &aux_info, &mut transcript
        )?;

        // Zerocheck subclaim
        assert_eq!(
            q.evaluate(&subclaim.zero_check_sub_claim.point).unwrap()
                * inv_g
                    .evaluate(&subclaim.zero_check_sub_claim.point)
                    .unwrap()
                - Fr::one(),
            subclaim.zero_check_sub_claim.expected_evaluation
        );

        // Sumcheck subclaim
        assert_eq!(
            p.evaluate(&subclaim.sum_check_sub_claim.point).unwrap()
                * inv_g.evaluate(&subclaim.sum_check_sub_claim.point).unwrap(),
            subclaim.sum_check_sub_claim.expected_evaluation
        );

        Ok(())
    }
}
