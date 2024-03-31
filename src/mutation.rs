use serde::{Deserialize, Serialize};
use vmmc::InputParams;
use equationx::Expr;
use std::str::FromStr;
use vmmc::protocol::SynthesisProtocol;
use rand::rngs::SmallRng;

use crate::nn::NueralNet;

// pub struct Polynomial {
//     terms: Vec<f64>,
// }

pub fn terms_to_polynomial(terms: &[f64]) -> String {
    let mut s = String::new();
    for (order, term) in terms.iter().enumerate() {
        if order == terms.len() - 1{
            s.push_str(&format!("{}t^{}", term, order))
        }
        else {
            s.push_str(&format!("{}t^{} + ", term, order))
        }
    }
    s
    // Expr::from_str(&s).unwrap()
}



#[derive(Clone, Serialize, Deserialize)]
pub enum MutationFunc {
    UniformRandomCoefficients(Vec<f64>, Vec<f64>, f64), // terms, order, percentage of range that any one coefficient can contribute
    LearningToGrowClassic(NueralNet),
}

// MutationFunc + Synthesis Protocol need to be combined

// TODO: synthesis protocol just needs to accept things that can eval
// TODO: want polynomial class? ==> form factor of mutation func and synthesis protocol are tied together (?)
// need a function that is SynthesisProtocol +  


// synthesisprotocol
// 

// fn mutate_polynomial(SynthesisProtoc)


// let T be an eval thing

// TODO: figure out a rationale to how we scale random values
fn perturb_poly(rng: &mut SmallRng, terms: &[f64], lower: f64, upper: f64, mag: f64) -> Vec<f64> {
    // generate random value in range and scale by mag
    let perturbation = rng::gen_range::<f64>(lower * mag, upper * mag);
    for (order, term) in terms.iter().enumerate(){

    }
    // 
    // add to term
    // terms_to_polynomial
    unimplemented!()
}

impl MutationFunc {
    pub fn mutate(&mut self, rng: &mut SmallRng, ip: &InputParams) -> InputParams {
        use MutationFunc::*;
        match self {
            UniformRandomCoefficients(interaction_energy_poly, chemical_potential_poly, mag) => {
                let new_interaction_energy_poly = perturb_poly(rng, interaction_energy_poly, *mag, 0.0, 20.0);
                let new_chemical_potential_poly = perturb_poly(rng, chemical_potential_poly, *mag, -10.0, 10.0);
                *interaction_energy_poly = new_interaction_energy_poly.clone();
                *chemical_potential_poly = new_chemical_potential_poly.clone();
                let mut new_ip = ip.clone();
                let new_protocol = SynthesisProtocol::new(&terms_to_polynomial(&new_interaction_energy_poly), &terms_to_polynomial(&chemical_potential_poly), ip.protocol.num_megasteps());
                new_ip.protocol = new_protocol;
                new_ip
            },
            LearningToGrowClassic(nn) => {unimplemented!()},
        }
    }
}
