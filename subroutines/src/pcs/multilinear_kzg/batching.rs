// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

//! Sumcheck based batch opening and verify commitment.
// TODO: refactoring this code to somewhere else
// currently IOP depends on PCS because perm check requires commitment.
// The sumcheck based batch opening therefore cannot stay in the PCS repo --
// which creates a cyclic dependency.

use crate::{
    pcs::{
        multilinear_kzg::util::eq_eval,
        prelude::{Commitment, PCSError},
        PolynomialCommitmentScheme,
    },
    poly_iop::{prelude::SumCheck, PolyIOP},
    IOPProof,
};
use arithmetic::{build_eq_x_r_vec, DenseMultilinearExtension, VPAuxInfo, VirtualPolynomial};
use ark_ec::{pairing::Pairing, scalar_mul::variable_base::VariableBaseMSM, CurveGroup};

use ark_std::{end_timer, log2, start_timer, One, Zero};
use std::{collections::BTreeMap, iter, marker::PhantomData, ops::Deref, sync::Arc};
use transcript::IOPTranscript;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
use rayon::prelude::*;

#[derive(Clone, Debug, Default, PartialEq, Eq, CanonicalSerialize, CanonicalDeserialize)]
pub struct BatchProof<E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E>,
{
    /// A sum check proof proving tilde g's sum
    pub(crate) sum_check_proof: IOPProof<E::ScalarField>,
    /// f_i(point_i)
    pub f_i_eval_at_point_i: Vec<E::ScalarField>,
    /// proof for g'(a_2)
    pub(crate) g_prime_proof: PCS::Proof,
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build eq(t,i) for i in [0..k]
/// 3. build \tilde g_i(b) = eq(t, i) * f_i(b)
/// 4. compute \tilde eq_i(b) = eq(b, point_i)
/// 5. run sumcheck on \sum_i=1..k \tilde eq_i * \tilde g_i
/// 6. build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is
///    the sumcheck's point 7. open g'(X) at point (a2)
pub(crate) fn multi_open_internal<E, PCS>(
    prover_param: &PCS::ProverParam,
    polynomials: &[PCS::Polynomial],
    points: &[PCS::Point],
    evals: &[PCS::Evaluation],
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<BatchProof<E, PCS>, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        ProverCommitmentAdvice = (),
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
    >,
{
    let open_timer = start_timer!(|| format!("multi open {} points", points.len()));

    // TODO: sanity checks
    let num_var = polynomials[0].num_vars;
    let k = polynomials.len();
    let ell = log2(k) as usize;

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // eq(t, i) for i in [0..k]
    let eq_t_i_list = build_eq_x_r_vec(t.as_ref())?;

    // \tilde g_i(b) = eq(t, i) * f_i(b)
    let timer = start_timer!(|| format!("compute tilde g for {} points", points.len()));
    // combine the polynomials that have same opening point first to reduce the
    // cost of sum check later.
    let point_indices = points
        .iter()
        .fold(BTreeMap::<_, _>::new(), |mut indices, point| {
            let idx = indices.len();
            indices.entry(point).or_insert(idx);
            indices
        });
    let deduped_points =
        BTreeMap::from_iter(point_indices.iter().map(|(point, idx)| (*idx, *point)))
            .into_values()
            .collect::<Vec<_>>();

    let mut point_ids = vec![vec![]; point_indices.len()];
    for (i, point) in points.iter().enumerate() {
        point_ids[point_indices[point]].push(i);
    }

    let tilde_gs = polynomials
        .par_iter()
        .zip(&eq_t_i_list)
        .map(|(poly, coeff)| {
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                poly.num_vars,
                poly.evaluations.iter().map(|eval| *eval * coeff).collect(),
            ))
        })
        .collect::<Vec<_>>();

    let mut merged_tilde_gs = point_ids
        .par_iter()
        .map(|point_ids| {
            let mut merged_tilde_g = Arc::new(DenseMultilinearExtension::zero());
            for &idx in point_ids {
                let poly = &tilde_gs[idx];
                *Arc::get_mut(&mut merged_tilde_g).unwrap() += poly.deref();
            }
            merged_tilde_g
        })
        .collect::<Vec<_>>();
    end_timer!(timer);

    let timer = start_timer!(|| format!("compute tilde eq for {} points", points.len()));
    let tilde_eqs: Vec<_> = deduped_points
        .iter()
        .map(|point| {
            let eq_b_zi = build_eq_x_r_vec(point).unwrap();
            Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_var, eq_b_zi,
            ))
        })
        .collect();
    end_timer!(timer);

    // built the virtual polynomial for SumCheck
    let timer = start_timer!(|| format!("sum check prove of {} variables", num_var));

    let proof = {
    let step = start_timer!(|| "add mle");
    let mut sum_check_vp = VirtualPolynomial::new(num_var);
    for (merged_tilde_g, tilde_eq) in merged_tilde_gs.iter().zip(tilde_eqs.into_iter()) {
        sum_check_vp.add_mle_list([merged_tilde_g.clone(), tilde_eq], E::ScalarField::one())?;
    }
    end_timer!(step);

    let proof = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::prove(
        &sum_check_vp,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch proving Failed".to_string(),
            ));
        },
    };
    proof
};

    end_timer!(timer);

    // a2 := sumcheck's point
    let a2 = &proof.point[..num_var];

    // build g'(X) = \sum_i=1..k \tilde eq_i(a2) * \tilde g_i(X) where (a2) is the
    // sumcheck's point \tilde eq_i(a2) = eq(a2, point_i)
    let step = start_timer!(|| "evaluate at a2");
    (&mut merged_tilde_gs, &deduped_points)
        .into_par_iter()
        .for_each(|(merged_tilde_g, point)| {
            let eq_i_a2 = eq_eval(&a2, point).unwrap();
            Arc::get_mut(merged_tilde_g)
                .unwrap()
                .evaluations
                .par_iter_mut()
                .for_each(|x| *x *= eq_i_a2)
        });
    let g_prime = merged_tilde_gs.into_par_iter().reduce(
        || Arc::new(DenseMultilinearExtension::zero()),
        |mut a, b| {
            if a.is_zero() {
                return b;
            }
            if b.is_zero() {
                return a;
            }
            *Arc::get_mut(&mut a).unwrap() += &*b;
            a
        },
    );
    end_timer!(step);

    let step = start_timer!(|| "pcs open");
    let g_prime_proof = PCS::open(prover_param, &g_prime, &(), a2.to_vec().as_ref())?;
    // assert_eq!(g_prime_eval, tilde_g_eval);
    end_timer!(step);

    let step = start_timer!(|| "evaluate fi(pi)");
    end_timer!(step);
    end_timer!(open_timer);

    Ok(BatchProof {
        sum_check_proof: proof,
        f_i_eval_at_point_i: evals.to_vec(),
        g_prime_proof,
    })
}

/// Steps:
/// 1. get challenge point t from transcript
/// 2. build g' commitment
/// 3. ensure \sum_i eq(a2, point_i) * eq(t, <i>) * f_i_evals matches the sum
///    via SumCheck verification 4. verify commitment
pub(crate) fn batch_verify_internal<E, PCS>(
    verifier_param: &PCS::VerifierParam,
    f_i_commitments: &[Commitment<E>],
    points: &[PCS::Point],
    proof: &BatchProof<E, PCS>,
    transcript: &mut IOPTranscript<E::ScalarField>,
) -> Result<bool, PCSError>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<
        E,
        Polynomial = Arc<DenseMultilinearExtension<E::ScalarField>>,
        Point = Vec<E::ScalarField>,
        Evaluation = E::ScalarField,
        Commitment = Commitment<E>,
    >,
{
    let open_timer = start_timer!(|| "batch verification");

    // TODO: sanity checks

    let k = f_i_commitments.len();
    let ell = log2(k) as usize;
    let num_var = proof.sum_check_proof.point.len();

    // challenge point t
    let t = transcript.get_and_append_challenge_vectors("t".as_ref(), ell)?;

    // sum check point (a2)
    let a2 = &proof.sum_check_proof.point[..num_var];

    // build g' commitment
    let step = start_timer!(|| "build homomorphic commitment");
    let eq_t_list = build_eq_x_r_vec(t.as_ref())?;

    let mut scalars = vec![];
    let mut bases = vec![];

    for (i, point) in points.iter().enumerate() {
        let eq_i_a2 = eq_eval(a2, point)?;
        scalars.push(eq_i_a2 * eq_t_list[i]);
        bases.push(f_i_commitments[i].0);
    }
    let g_prime_commit = E::G1MSM::msm_unchecked_par_auto(&bases, &scalars);
    end_timer!(step);

    // ensure \sum_i eq(t, <i>) * f_i_evals matches the sum via SumCheck
    let mut sum = E::ScalarField::zero();
    for (i, &e) in eq_t_list.iter().enumerate().take(k) {
        sum += e * proof.f_i_eval_at_point_i[i];
    }
    let aux_info = VPAuxInfo {
        max_degree: 2,
        num_variables: num_var,
        phantom: PhantomData,
    };
    let subclaim = match <PolyIOP<E::ScalarField> as SumCheck<E::ScalarField>>::verify(
        sum,
        &proof.sum_check_proof,
        &aux_info,
        transcript,
    ) {
        Ok(p) => p,
        Err(_e) => {
            // cannot wrap IOPError with PCSError due to cyclic dependency
            return Err(PCSError::InvalidProver(
                "Sumcheck in batch verification failed".to_string(),
            ));
        },
    };
    let tilde_g_eval = subclaim.expected_evaluation;

    // verify commitment
    let res = PCS::verify(
        verifier_param,
        &Commitment(g_prime_commit.into()),
        a2.to_vec().as_ref(),
        &tilde_g_eval,
        &proof.g_prime_proof,
    )?;

    end_timer!(open_timer);
    Ok(res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::{
        prelude::{MultilinearKzgPCS, MultilinearUniversalParams},
        StructuredReferenceString,
    };
    use arithmetic::get_batched_nv;
    use ark_bls12_381::Bls12_381 as E;
    use ark_ec::pairing::Pairing;
    use ark_poly::{DenseMultilinearExtension, MultilinearExtension};
    use ark_std::{rand::Rng, test_rng, vec::Vec, UniformRand};

    type Fr = <E as Pairing>::ScalarField;

    fn test_multi_open_helper<R: Rng>(
        ml_params: &MultilinearUniversalParams<E>,
        polys: &[Arc<DenseMultilinearExtension<Fr>>],
        rng: &mut R,
    ) -> Result<(), PCSError> {
        let merged_nv = get_batched_nv(polys[0].num_vars(), polys.len());
        let (ml_ck, ml_vk) = ml_params.trim(merged_nv)?;

        let mut points = Vec::new();
        for poly in polys.iter() {
            let point = (0..poly.num_vars())
                .map(|_| Fr::rand(rng))
                .collect::<Vec<Fr>>();
            points.push(point);
        }

        let evals = polys
            .iter()
            .zip(points.iter())
            .map(|(f, p)| f.evaluate(p).unwrap())
            .collect::<Vec<_>>();

        let commitments = polys
            .iter()
            .map(|poly| MultilinearKzgPCS::commit(ml_ck.clone(), poly).unwrap().0)
            .collect::<Vec<_>>();

        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;

        let batch_proof = multi_open_internal::<E, MultilinearKzgPCS<E>>(
            &ml_ck,
            polys,
            &points,
            &evals,
            &mut transcript,
        )?;

        // good path
        let mut transcript = IOPTranscript::new("test transcript".as_ref());
        transcript.append_field_element("init".as_ref(), &Fr::zero())?;
        assert!(batch_verify_internal::<E, MultilinearKzgPCS<E>>(
            &ml_vk,
            &commitments,
            &points,
            &batch_proof,
            &mut transcript
        )?);

        Ok(())
    }

    #[test]
    fn test_multi_open_internal() -> Result<(), PCSError> {
        let mut rng = test_rng();

        let ml_params = MultilinearUniversalParams::<E>::gen_srs_for_testing(&mut rng, 20)?;
        for num_poly in 5..6 {
            for nv in 15..16 {
                let polys1: Vec<_> = (0..num_poly)
                    .map(|_| Arc::new(DenseMultilinearExtension::rand(nv, &mut rng)))
                    .collect();
                test_multi_open_helper(&ml_params, &polys1, &mut rng)?;
            }
        }

        Ok(())
    }
}
