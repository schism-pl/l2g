use rand::{rngs::SmallRng, Rng};
use serde::{Deserialize, Serialize};
use vmmc::protocol::SynthesisProtocol;

use crate::nn::{NnConfig, NueralNet};

// pub fn terms_to_polynomial(terms: &[f64]) -> String {
//     let mut s = String::new();
//     for (order, term) in terms.iter().enumerate() {
//         if order == terms.len() - 1 {
//             s.push_str(&format!("{}t^{}", term, order))
//         } else {
//             s.push_str(&format!("{}t^{} + ", term, order))
//         }
//     }
//     s
// }

#[derive(Clone, Serialize, Deserialize)]
pub enum MutationFunc {
    UniformRandomCoefficients(SynthesisProtocol, Vec<f64>, Vec<f64>, f64), // terms, order, percentage of range that any one coefficient can contribute
    LearningToGrowClassic(NnConfig, SynthesisProtocol), // argument 2 = # of steps for each protocol
}

// TODO: figure out a rationale to how we scale random values
// write out equation
// fn perturb_poly(rng: &mut SmallRng, terms: &[f64], lower: f64, upper: f64, mag: f64) -> Vec<f64> {
//     // generate random value in range and scale by mag
//     let mut new_coefficients = Vec::new();
//     let perturbation = rng.gen_range(lower * mag..upper * mag);
//     for (order, coefficient) in terms.iter().enumerate() {
//         new_coefficients.push((*coefficient + perturbation * 10_f64.powi(-(order as i32))) / 2.0);
//     }
//     new_coefficients
// }

// TODO: current_protocol?
impl MutationFunc {
    pub fn mutate(&mut self, rng: &mut SmallRng) {
        use MutationFunc::*;
        match self {
            UniformRandomCoefficients(..) => {
                unimplemented!()
                // let new_interaction_energy_poly =
                //     perturb_poly(rng, interaction_energy_poly, *mag, 0.0, 20.0);
                // let new_chemical_potential_poly =
                //     perturb_poly(rng, chemical_potential_poly, *mag, -10.0, 10.0);
                // *interaction_energy_poly = new_interaction_energy_poly.clone();
                // *chemical_potential_poly = new_chemical_potential_poly.clone();

                // let new_protocol = SynthesisProtocol::new(
                //     &terms_to_polynomial(interaction_energy_poly),
                //     &terms_to_polynomial(chemical_potential_poly),
                //     protocol.num_megasteps(),
                // );
                // *protocol = new_protocol
            }
            LearningToGrowClassic(nn, _protocol) => {
                nn.increment_child_id();
            }
        }
    }
}
