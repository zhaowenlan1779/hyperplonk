// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use std::{fs::File, io, time::Instant};

use ark_bn254::{Bn254, Fr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Write};
use ark_std::test_rng;
use deDory::PublicParameters;
use hyperplonk::{
    prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
    HyperPlonkSNARK,
};
use subroutines::{
    pcs::{dory::Dory, PolynomialCommitmentScheme}, poly_iop::PolyIOP, MultilinearKzgPCS, MultilinearUniversalParams
};
use ark_poly::DenseMultilinearExtension;
use subroutines::BatchProof;
use std::sync::Arc;

const SUPPORTED_SIZE: usize = 13;

fn main() -> Result<(), HyperPlonkErrors> {
    let thread = rayon::current_num_threads();
    println!("start benchmark with #{} threads", thread);
    let mut rng = test_rng();
    
    let args: Vec<_> = std::env::args().collect();
    assert!(args.len() >= 2);
    if args[1] == "--doryjellyfish" {
        let pcs_srs = {
            match read_dory_srs() {
                Ok(p) => p,
                Err(_e) => {
                    let srs = Dory::<Bn254>::gen_srs_for_testing(&mut rng, SUPPORTED_SIZE)?;
                    write_dory_srs(&srs);
                    srs
                },
            }
        };

        let nv = str::parse(&args[2]).unwrap();
        bench_jellyfish_plonk::<Dory<Bn254>>(&pcs_srs, nv)?;
    } else if args[1] == "--dory" {
        let pcs_srs = {
            match read_dory_srs() {
                Ok(p) => p,
                Err(_e) => {
                    let srs = Dory::<Bn254>::gen_srs_for_testing(&mut rng, SUPPORTED_SIZE)?;
                    write_dory_srs(&srs);
                    srs
                },
            }
        };

        let nv = str::parse(&args[2]).unwrap();
        bench_vanilla_plonk::<Dory<Bn254>>(&pcs_srs, nv)?;
    } else if args[1] == "--jellyfish" {
        let pcs_srs = {
            match read_mkzg_srs() {
                Ok(p) => p,
                Err(_e) => {
                    let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, 26)?;
                    write_mkzg_srs(&srs);
                    srs
                },
            }
        };

        let nv = str::parse(&args[2]).unwrap();
        bench_jellyfish_plonk::<MultilinearKzgPCS<Bn254>>(&pcs_srs, nv)?;
    } else {
        let pcs_srs = {
            match read_mkzg_srs() {
                Ok(p) => p,
                Err(_e) => {
                    let srs = MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, 26)?;
                    write_mkzg_srs(&srs);
                    srs
                },
            }
        };

        let nv = str::parse(&args[1]).unwrap();
        bench_vanilla_plonk::<MultilinearKzgPCS<Bn254>>(&pcs_srs, nv)?;
    }

    Ok(())
}

fn read_dory_srs() -> Result<PublicParameters<Bn254>, io::Error> {
    let mut f = File::open("dory.params")?;
    Ok(PublicParameters::<Bn254>::deserialize_uncompressed_unchecked(&mut f).unwrap())
}

fn write_dory_srs(pcs_srs: &PublicParameters<Bn254>) {
    let mut f = File::create("dory.params").unwrap();
    pcs_srs.serialize_uncompressed(&mut f).unwrap();
}

fn read_mkzg_srs() -> Result<MultilinearUniversalParams<Bn254>, io::Error> {
    let mut f = File::open("mkzg.params")?;
    Ok(MultilinearUniversalParams::<Bn254>::deserialize_uncompressed_unchecked(&mut f).unwrap())
}

fn write_mkzg_srs(pcs_srs: &MultilinearUniversalParams<Bn254>) {
    let mut f = File::create("mkzg.params").unwrap();
    pcs_srs.serialize_uncompressed(&mut f).unwrap();
}

fn bench_vanilla_plonk<
    PCS: PolynomialCommitmentScheme<
        Bn254,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
        BatchProof = BatchProof<Bn254, PCS>,
    >,
>(
    pcs_srs: &PCS::SRS,
    nv: usize,
) -> Result<(), HyperPlonkErrors> {
    let vanilla_gate = CustomizedGates::vanilla_plonk_gate();
    bench_mock_circuit_zkp_helper::<PCS>(nv, &vanilla_gate, pcs_srs)?;

    Ok(())
}

fn bench_jellyfish_plonk<
    PCS: PolynomialCommitmentScheme<
        Bn254,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
        BatchProof = BatchProof<Bn254, PCS>,
    >,
>(
    pcs_srs: &PCS::SRS,
    nv: usize,
) -> Result<(), HyperPlonkErrors> {
    let jf_gate = CustomizedGates::jellyfish_turbo_plonk_gate();
    bench_mock_circuit_zkp_helper::<PCS>(nv, &jf_gate, pcs_srs)?;

    Ok(())
}

fn bench_mock_circuit_zkp_helper<
    PCS: PolynomialCommitmentScheme<
        Bn254,
        Polynomial = Arc<DenseMultilinearExtension<Fr>>,
        Point = Vec<Fr>,
        Evaluation = Fr,
        BatchProof = BatchProof<Bn254, PCS>,
    >,
>(
    nv: usize,
    gate: &CustomizedGates,
    pcs_srs: &PCS::SRS,
) -> Result<(), HyperPlonkErrors> {
    let repetition = if nv <= 20 {
        10
    } else if nv <= 22 {
        5
    } else {
        3
    };

    //==========================================================
    let circuit = MockCircuit::<Fr>::new(1 << nv, gate);
    assert!(circuit.is_satisfied());
    let index = circuit.index;
    //==========================================================
    // generate pk and vks
    let (pk, vk) = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::preprocess(&index, pcs_srs)?;
    //==========================================================
    // generate a proof
    let start = Instant::now();
    for _ in 0..repetition {
        let _proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::prove(
            &pk,
            &circuit.public_inputs,
            &circuit.witnesses,
        )?;
    }
    println!(
        "proving for {} variables: {} us",
        nv,
        start.elapsed().as_micros() / repetition as u128
    );

    let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::prove(
        &pk,
        &circuit.public_inputs,
        &circuit.witnesses,
    )?;

    let mut bytes = Vec::with_capacity(CanonicalSerialize::compressed_size(&proof));
    CanonicalSerialize::serialize_compressed(&proof, &mut bytes).unwrap();
    println!(
        "proof size for {} variables compressed: {} bytes",
        nv,
        bytes.len()
    );

    let mut bytes = Vec::with_capacity(CanonicalSerialize::uncompressed_size(&proof));
    CanonicalSerialize::serialize_uncompressed(&proof, &mut bytes).unwrap();
    println!(
        "proof size for {} variables uncompressed: {} bytes",
        nv,
        bytes.len()
    );

    //==========================================================
    // verify a proof
    let start = Instant::now();
    for _ in 0..(repetition * 5) {
        let verify = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, PCS>>::verify(
            &vk,
            &circuit.public_inputs,
            &proof,
        )?;
        assert!(verify);
    }
    println!(
        "verifying for {} variables: {} us",
        nv,
        start.elapsed().as_micros() / (repetition * 5) as u128
    );
    Ok(())
}
