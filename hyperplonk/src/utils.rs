// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use crate::{
    custom_gate::CustomizedGates, errors::HyperPlonkErrors, structs::HyperPlonkParams,
    witness::WitnessColumn,
};
use arithmetic::{evaluate_opt, VirtualPolynomial};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use itertools::Itertools;
use std::iter::zip;
use std::{borrow::Borrow, sync::Arc};
use subroutines::pcs::{prelude::Commitment, PolynomialCommitmentScheme};
use subroutines::BatchProof;
use transcript::IOPTranscript;

/// An accumulator structure that holds a polynomial and
/// its opening points
#[derive(Debug)]
pub(super) struct PcsAccumulator<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    // sequence:
    // - prod(x) at 5 points
    // - w_merged at perm check point
    // - w_merged at zero check points (#witness points)
    // - selector_merged at zero check points (#selector points)
    // - w[0] at r_pi
    pub(crate) num_var: usize,
    pub(crate) polynomials: Vec<PCS::Polynomial>,
    pub(crate) commitments: Vec<PCS::Commitment>,
    pub(crate) points: Vec<PCS::Point>,
    pub(crate) evals: Vec<PCS::Evaluation>,
}

impl<E, PCS> PcsAccumulator<E, PCS>
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
    /// Create an empty accumulator.
    pub(super) fn new(num_var: usize) -> Self {
        Self {
            num_var,
            polynomials: vec![],
            commitments: vec![],
            points: vec![],
            evals: vec![],
        }
    }

    /// Push a new evaluation point into the accumulator
    pub(super) fn insert_poly_and_points(
        &mut self,
        poly: &PCS::Polynomial,
        commit: &PCS::Commitment,
        point: &PCS::Point,
    ) {
        assert!(poly.num_vars == point.len());
        assert!(poly.num_vars == self.num_var);

        let eval = evaluate_opt(poly, point);

        self.evals.push(eval);
        self.polynomials.push(poly.clone());
        self.points.push(point.clone());
        self.commitments.push(*commit);
    }

    /// Batch open all the points over a merged polynomial.
    /// A simple wrapper of PCS::multi_open
    pub(super) fn multi_open(
        &self,
        prover_param: impl Borrow<PCS::ProverParam>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<PCS::BatchProof, HyperPlonkErrors> {
        Ok(PCS::multi_open(
            prover_param.borrow(),
            self.polynomials.as_ref(),
            self.points.as_ref(),
            self.evals.as_ref(),
            transcript,
        )?)
    }
}

/// An accumulator structure that automatically creates accumulators
/// for different num_vars
#[derive(Debug)]
pub(super) struct PcsDynamicAccumulator<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    pub(crate) accumulators: Vec<(usize, PcsAccumulator<E, PCS>)>,
    pub(crate) indices: Vec<usize>,
}

impl<E, PCS> PcsDynamicAccumulator<E, PCS>
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
    pub(super) fn new() -> Self {
        Self {
            accumulators: vec![],
            indices: vec![],
        }
    }

    /// Push a new evaluation point into the accumulator
    pub(super) fn insert_poly_and_points(
        &mut self,
        poly: &PCS::Polynomial,
        commit: &PCS::Commitment,
        point: &PCS::Point,
    ) {
        let acc_opt = self
            .accumulators
            .iter_mut()
            .find_position(|(num_vars, _)| *num_vars == poly.num_vars);
        if let Some((index, acc)) = acc_opt {
            acc.1.insert_poly_and_points(poly, commit, point);
            self.indices.push(index);
        } else {
            self.accumulators
                .push((poly.num_vars, PcsAccumulator::new(poly.num_vars)));
            self.accumulators
                .last_mut()
                .unwrap()
                .1
                .insert_poly_and_points(poly, commit, point);
            self.indices.push(self.accumulators.len() - 1);
        }
    }

    /// Batch open all the points over a merged polynomial.
    /// A simple wrapper of PCS::multi_open
    pub(super) fn multi_open(
        &self,
        prover_param: impl Borrow<PCS::ProverParam>,
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<PcsDynamicProof<E, PCS>, HyperPlonkErrors> {
        let mut proofs = Vec::with_capacity(self.accumulators.len());
        for (_, accumulator) in self.accumulators.iter() {
            let proof = accumulator.multi_open(prover_param.borrow(), transcript)?;
            proofs.push(proof);
        }
        Ok(PcsDynamicProof {
            proofs,
            indices: self.indices.clone(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcsDynamicProof<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    pub(crate) proofs: Vec<PCS::BatchProof>,
    pub(crate) indices: Vec<usize>,
}

pub(super) struct PcsDynamicOpenings<'a, E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    pub(crate) proof: &'a PcsDynamicProof<E, PCS>,

    // temporary state
    pub(crate) proof_offsets: Vec<usize>,
    pub(crate) offset: usize,
}

impl<'a, E, PCS> PcsDynamicOpenings<'a, E, PCS>
where
    E: Pairing,
    PCS: PolynomialCommitmentScheme<E, BatchProof = BatchProof<E, PCS>>,
{
    pub(super) fn new(proof: &'a PcsDynamicProof<E, PCS>) -> Self {
        Self {
            proof,
            proof_offsets: vec![0; proof.proofs.len()],
            offset: 0,
        }
    }

    pub(super) fn next_openings(&mut self, len: usize) -> Vec<E::ScalarField> {
        let indices = if len == usize::MAX {
            &self.proof.indices[self.offset..]
        } else {
            &self.proof.indices[self.offset..self.offset + len]
        };
        let result = indices
            .iter()
            .map(|index| {
                let result = self.proof.proofs[*index].f_i_eval_at_point_i[self.proof_offsets[*index]];
                self.proof_offsets[*index] += 1;
                result
            })
            .collect();
        if len == usize::MAX {
            self.offset = usize::MAX;
        } else {
            self.offset += len;
        }
        result
    }
}

/// An accumulator structure that automatically creates accumulators
/// for different num_vars
#[derive(Debug)]
pub(super) struct PcsDynamicVerifier<E: Pairing, PCS: PolynomialCommitmentScheme<E>> {
    pub(crate) items: Vec<(usize, Vec<PCS::Commitment>, Vec<Vec<E::ScalarField>>)>,
}

impl<E, PCS> PcsDynamicVerifier<E, PCS>
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
    pub(super) fn new() -> Self {
        Self { items: vec![] }
    }

    /// Push a new evaluation point into the accumulator
    pub(super) fn insert_comm_and_points(&mut self, commit: PCS::Commitment, point: PCS::Point) {
        if let Some((_, comms, points)) =
            self.items.iter_mut().find(|(nv, _, _)| *nv == point.len())
        {
            comms.push(commit);
            points.push(point);
        } else {
            self.items.push((point.len(), vec![commit], vec![point]));
        }
    }

    /// Batch open all the points over a merged polynomial.
    /// A simple wrapper of PCS::multi_open
    pub(super) fn batch_verify(
        &self,
        verifier_param: impl Borrow<PCS::VerifierParam>,
        batch_proof: &[PCS::BatchProof],
        transcript: &mut IOPTranscript<E::ScalarField>,
    ) -> Result<bool, HyperPlonkErrors> {
        for (proof, (_, comms, points)) in zip(batch_proof.iter(), self.items.iter()) {
            let res =
                PCS::batch_verify(verifier_param.borrow(), comms, points, &proof, transcript)?;
            if !res {
                return Ok(res);
            }
        }
        Ok(true)
    }
}

/// Build MLE from matrix of witnesses.
///
/// Given a matrix := [row1, row2, ...] where
/// row1:= (a1, a2, ...)
/// row2:= (b1, b2, ...)
/// row3:= (c1, c2, ...)
///
/// output mle(a1,b1,c1, ...), mle(a2,b2,c2, ...), ...
#[macro_export]
macro_rules! build_mle {
    ($rows:expr) => {{
        let mut res = Vec::with_capacity($rows.len());
        let num_vars = log2($rows.len()) as usize;
        let num_mles = $rows[0].0.len();

        for i in 0..num_mles {
            let mut cur_coeffs = Vec::new();
            for row in $rows.iter() {
                cur_coeffs.push(row.0[i])
            }
            res.push(Arc::new(DenseMultilinearExtension::from_evaluations_vec(
                num_vars, cur_coeffs,
            )))
        }

        Ok(res)
    }};
}

/// Sanity-check for HyperPlonk SNARK proving
pub(crate) fn prover_sanity_check<F: PrimeField>(
    params: &HyperPlonkParams,
    pub_input: &[F],
    witnesses: &[WitnessColumn<F>],
) -> Result<(), HyperPlonkErrors> {
    // public input length must be no greater than num_constraints

    if pub_input.len() > params.num_constraints {
        return Err(HyperPlonkErrors::InvalidProver(format!(
            "Public input length {} is greater than num constraits {}",
            pub_input.len(),
            params.num_pub_input
        )));
    }

    // public input length
    if pub_input.len() != params.num_pub_input {
        return Err(HyperPlonkErrors::InvalidProver(format!(
            "Public input length is not correct: got {}, expect {}",
            pub_input.len(),
            params.num_pub_input
        )));
    }
    if !pub_input.len().is_power_of_two() {
        return Err(HyperPlonkErrors::InvalidProver(format!(
            "Public input length is not power of two: got {}",
            pub_input.len(),
        )));
    }

    // witnesses length
    for (i, w) in witnesses.iter().enumerate() {
        if w.0.len() != params.num_constraints {
            return Err(HyperPlonkErrors::InvalidProver(format!(
                "{}-th witness length is not correct: got {}, expect {}",
                i,
                w.0.len(),
                params.num_constraints
            )));
        }
    }
    // check public input matches witness[0]'s first 2^ell elements
    for (i, (&pi, &w)) in pub_input
        .iter()
        .zip(witnesses[0].0.iter().take(pub_input.len()))
        .enumerate()
    {
        if pi != w {
            return Err(HyperPlonkErrors::InvalidProver(format!(
                "The {:?}-th public input {:?} does not match witness[0] {:?}",
                i, pi, w
            )));
        }
    }

    Ok(())
}

/// build `f(w_0(x),...w_d(x))` where `f` is the constraint polynomial
/// i.e., `f(a, b, c) = q_l a(x) + q_r b(x) + q_m a(x)b(x) - q_o c(x)` in
/// vanilla plonk
pub(crate) fn build_f<F: PrimeField>(
    gates: &CustomizedGates,
    num_vars: usize,
    selector_mles: &[Arc<DenseMultilinearExtension<F>>],
    witness_mles: &[Arc<DenseMultilinearExtension<F>>],
) -> Result<VirtualPolynomial<F>, HyperPlonkErrors> {
    // TODO: check that selector and witness lengths match what is in
    // the gate definition

    for selector_mle in selector_mles.iter() {
        if selector_mle.num_vars != num_vars {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "selector has different number of vars: {} vs {}",
                selector_mle.num_vars, num_vars
            )));
        }
    }

    for witness_mle in witness_mles.iter() {
        if witness_mle.num_vars != num_vars {
            return Err(HyperPlonkErrors::InvalidParameters(format!(
                "selector has different number of vars: {} vs {}",
                witness_mle.num_vars, num_vars
            )));
        }
    }

    let mut res = VirtualPolynomial::<F>::new(num_vars);

    for (coeff, selector, witnesses) in gates.gates.iter() {
        let coeff_fr = if *coeff < 0 {
            -F::from(-*coeff as u64)
        } else {
            F::from(*coeff as u64)
        };
        let mut mle_list = vec![];
        if let Some(s) = *selector {
            mle_list.push(selector_mles[s].clone())
        }
        for &witness in witnesses.iter() {
            mle_list.push(witness_mles[witness].clone())
        }
        res.add_mle_list(mle_list, coeff_fr)?;
    }

    Ok(res)
}

pub(crate) fn eval_f<F: PrimeField>(
    gates: &CustomizedGates,
    selector_evals: &[F],
    witness_evals: &[F],
) -> Result<F, HyperPlonkErrors> {
    let mut res = F::zero();
    for (coeff, selector, witnesses) in gates.gates.iter() {
        let mut cur_value = if *coeff < 0 {
            -F::from(-*coeff as u64)
        } else {
            F::from(*coeff as u64)
        };
        cur_value *= match selector {
            Some(s) => selector_evals[*s],
            None => F::one(),
        };
        for &witness in witnesses.iter() {
            cur_value *= witness_evals[witness]
        }
        res += cur_value;
    }
    Ok(res)
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::PrimeField;
    use ark_poly::MultilinearExtension;
    #[test]
    fn test_build_gate() -> Result<(), HyperPlonkErrors> {
        test_build_gate_helper::<Fr>()
    }

    fn test_build_gate_helper<F: PrimeField>() -> Result<(), HyperPlonkErrors> {
        let num_vars = 2;

        // ql = 3x1x2 + 2x2 whose evaluations are
        // 0, 0 |-> 0
        // 0, 1 |-> 2
        // 1, 0 |-> 0
        // 1, 1 |-> 5
        let ql_eval = vec![F::zero(), F::from(2u64), F::zero(), F::from(5u64)];
        let ql = Arc::new(DenseMultilinearExtension::from_evaluations_vec(2, ql_eval));

        // W1 = x1x2 + x1 whose evaluations are
        // 0, 0 |-> 0
        // 0, 1 |-> 0
        // 1, 0 |-> 1
        // 1, 1 |-> 2
        let w_eval = vec![F::zero(), F::zero(), F::from(1u64), F::from(2u64)];
        let w1 = Arc::new(DenseMultilinearExtension::from_evaluations_vec(2, w_eval));

        // W2 = x1 + x2 whose evaluations are
        // 0, 0 |-> 0
        // 0, 1 |-> 1
        // 1, 0 |-> 1
        // 1, 1 |-> 2
        let w_eval = vec![F::zero(), F::one(), F::from(1u64), F::from(2u64)];
        let w2 = Arc::new(DenseMultilinearExtension::from_evaluations_vec(2, w_eval));

        // Example:
        //     q_L(X) * W_1(X)^5 - W_2(X)
        // is represented as
        // vec![
        //     ( 1,    Some(id_qL),    vec![id_W1, id_W1, id_W1, id_W1, id_W1]),
        //     (-1,    None,           vec![id_W2])
        // ]
        let gates = CustomizedGates {
            gates: vec![(1, Some(0), vec![0, 0, 0, 0, 0]), (-1, None, vec![1])],
        };
        let f = build_f(&gates, num_vars, &[ql.clone()], &[w1.clone(), w2.clone()])?;

        // Sanity check on build_f
        // f(0, 0) = 0
        assert_eq!(f.evaluate(&[F::zero(), F::zero()])?, F::zero());
        // f(0, 1) = 2 * 0^5 + (-1) * 1 = -1
        assert_eq!(f.evaluate(&[F::zero(), F::one()])?, -F::one());
        // f(1, 0) = 0 * 1^5 + (-1) * 1 = -1
        assert_eq!(f.evaluate(&[F::one(), F::zero()])?, -F::one());
        // f(1, 1) = 5 * 2^5 + (-1) * 2 = 158
        assert_eq!(f.evaluate(&[F::one(), F::one()])?, F::from(158u64));

        // test eval_f
        {
            let point = [F::zero(), F::zero()];
            let selector_evals = ql.evaluate(&point).unwrap();
            let witness_evals = [w1.evaluate(&point).unwrap(), w2.evaluate(&point).unwrap()];
            let eval_f = eval_f(&gates, &[selector_evals], &witness_evals)?;
            // f(0, 0) = 0
            assert_eq!(eval_f, F::zero());
        }
        {
            let point = [F::zero(), F::one()];
            let selector_evals = ql.evaluate(&point).unwrap();
            let witness_evals = [w1.evaluate(&point).unwrap(), w2.evaluate(&point).unwrap()];
            let eval_f = eval_f(&gates, &[selector_evals], &witness_evals)?;
            // f(0, 1) = 2 * 0^5 + (-1) * 1 = -1
            assert_eq!(eval_f, -F::one());
        }
        {
            let point = [F::one(), F::zero()];
            let selector_evals = ql.evaluate(&point).unwrap();
            let witness_evals = [w1.evaluate(&point).unwrap(), w2.evaluate(&point).unwrap()];
            let eval_f = eval_f(&gates, &[selector_evals], &witness_evals)?;
            // f(1, 0) = 0 * 1^5 + (-1) * 1 = -1
            assert_eq!(eval_f, -F::one());
        }
        {
            let point = [F::one(), F::one()];
            let selector_evals = ql.evaluate(&point).unwrap();
            let witness_evals = [w1.evaluate(&point).unwrap(), w2.evaluate(&point).unwrap()];
            let eval_f = eval_f(&gates, &[selector_evals], &witness_evals)?;
            // f(1, 1) = 5 * 2^5 + (-1) * 2 = 158
            assert_eq!(eval_f, F::from(158u64));
        }
        Ok(())
    }
}
