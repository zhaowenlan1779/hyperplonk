// Copyright (c) 2023 Espresso Systems (espressosys.com)
// This file is part of the HyperPlonk library.

// You should have received a copy of the MIT License
// along with the HyperPlonk library. If not, see <https://mit-license.org/>.

use std::{fs::File, io, time::Instant};

use ark_bn254::{Bn254, Fr};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Write};
use ark_std::test_rng;
use hyperplonk::{
    prelude::{CustomizedGates, HyperPlonkErrors, MockCircuit},
    HyperPlonkSNARK,
};
use subroutines::{
    pcs::{
        prelude::{MultilinearKzgPCS, MultilinearUniversalParams},
        PolynomialCommitmentScheme,
    },
    poly_iop::PolyIOP,
};

const SUPPORTED_SIZE: usize = 26;

fn main() -> Result<(), HyperPlonkErrors> {
    let thread = rayon::current_num_threads();
    println!("start benchmark with #{} threads", thread);
    let mut rng = test_rng();
    let pcs_srs = {
        match read_srs() {
            Ok(p) => p,
            Err(_e) => {
                let srs =
                    MultilinearKzgPCS::<Bn254>::gen_srs_for_testing(&mut rng, SUPPORTED_SIZE)?;
                write_srs(&srs);
                srs
            },
        }
    };
    let args: Vec<_> = std::env::args().collect();
    assert!(args.len() >= 2);
    if args[1] == "--jellyfish" {
        let nv = str::parse(&args[2]).unwrap();
        bench_jellyfish_plonk(&pcs_srs, nv)?;
    } else {
        let nv = str::parse(&args[1]).unwrap();
        bench_vanilla_plonk(&pcs_srs, nv)?;
    }

    Ok(())
}

fn read_srs() -> Result<MultilinearUniversalParams<Bn254>, io::Error> {
    let mut f = File::open("srs.params")?;
    Ok(MultilinearUniversalParams::<Bn254>::deserialize_uncompressed_unchecked(&mut f).unwrap())
}

fn write_srs(pcs_srs: &MultilinearUniversalParams<Bn254>) {
    let mut f = File::create("srs.params").unwrap();
    pcs_srs.serialize_uncompressed(&mut f).unwrap();
}

fn bench_vanilla_plonk(
    pcs_srs: &MultilinearUniversalParams<Bn254>,
    nv: usize
) -> Result<(), HyperPlonkErrors> {
        let vanilla_gate = CustomizedGates::vanilla_plonk_gate();
        bench_mock_circuit_zkp_helper(nv, &vanilla_gate, pcs_srs)?;

    Ok(())
}

fn bench_jellyfish_plonk(
    pcs_srs: &MultilinearUniversalParams<Bn254>,
    nv: usize,
) -> Result<(), HyperPlonkErrors> {
        let jf_gate = CustomizedGates::jellyfish_turbo_plonk_gate();
        bench_mock_circuit_zkp_helper( nv, &jf_gate, pcs_srs)?;

    Ok(())
}

fn bench_mock_circuit_zkp_helper(
    nv: usize,
    gate: &CustomizedGates,
    pcs_srs: &MultilinearUniversalParams<Bn254>,
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
    let (pk, vk) =
        <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, MultilinearKzgPCS<Bn254>>>::preprocess(
            &index, pcs_srs,
        )?;
    //==========================================================
    // generate a proof
    let start = Instant::now();
    for _ in 0..repetition {
        let _proof =
            <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, MultilinearKzgPCS<Bn254>>>::prove(
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

    let proof = <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, MultilinearKzgPCS<Bn254>>>::prove(
        &pk,
        &circuit.public_inputs,
        &circuit.witnesses,
    )?;

    let mut bytes = Vec::with_capacity(CanonicalSerialize::compressed_size(&proof));
    CanonicalSerialize::serialize_compressed(&proof, &mut bytes).unwrap();
    println!("proof size for {} variables compressed: {} bytes", nv, bytes.len());

    let mut bytes = Vec::with_capacity(CanonicalSerialize::uncompressed_size(&proof));
    CanonicalSerialize::serialize_uncompressed(&proof, &mut bytes).unwrap();
    println!("proof size for {} variables uncompressed: {} bytes", nv, bytes.len());

    //==========================================================
    // verify a proof
    let start = Instant::now();
    for _ in 0..(repetition * 5) {
        let verify =
            <PolyIOP<Fr> as HyperPlonkSNARK<Bn254, MultilinearKzgPCS<Bn254>>>::verify(
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
