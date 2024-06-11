use rand::{rngs::SmallRng, SeedableRng};
/// Implement nueral net implementation from original paper
use rand_distr::{Distribution, Normal};
use runnt::{activation::ActivationType, nn::NN, regularization::Regularization};
use serde::{Deserialize, Serialize};
use vmmc::protocol::{self, Peekable, ProtocolStep, SynthesisProtocol};

// TODO: fixed at 10,10,10 architecture
// TODO: currently does not incorporate targeted protocol
// will ony

const NUM_PERIODS: usize = 10;
const PERIOD_LENGTH: usize = 100;
// const Times: [f32; 10] = [0., 1., 2., 3., 4. ,5., 6.,7.,8.,9.];

#[derive(Clone, Serialize, Deserialize)]
// Uniquely IDs a nueral net
// NN can be reconstructed by building nn from original set and
// mutating it by `child_id` times
pub struct FllConfig {
    nn: runnt::nn::NN,
}

impl FllConfig {
    pub fn new(orig_seed: u32, child_id: u32) -> Self {
        let nn = runnt::nn::NN::new(&[10, 10, 10])
            .with_learning_rate(0.1)
            .with_hidden_type(ActivationType::Sigmoid)
            .with_output_type(ActivationType::Linear)
            .with_regularization(Regularization::None);
        Self { nn }
    }
}

// pub struct FLLMegastepIter {
//     ep_slopes: Vec<f32>,
//     mu_slopes: Vec<f32>,
//     length: usize,
//     t: f64,
//     ep_accum: f64,
//     mu_accum: f64,
// }

// impl FLLMegastepIter {
//     fn new(nn: &NN, protocol: &SynthesisProtocol) -> Self {
//         let times = Vec::from_iter((0 ..NUM_PERIODS).map(|i| i as f32));
//         let mut slopes = nn.forward(&times);
//         let mu_slopes = slopes.split_off(10);
//         let length = protocol.num_megasteps();

//         Self {
//             ep_slopes: slopes,
//             mu_slopes,
//             t: 0.0,
//             ep_accum: 0.0,
//             mu_accum: 0.0,
//         }
//     }
// }

// impl Iterator for FLLMegastepIter {
//     type Item = ProtocolStep;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.t >= 1.0 {
//             return None;
//         }

//         // let (epsilon, mu) = self.nn.eval(self.t);
//         // let orig_epsilon = self.protocol.interaction_energy(self.t as usize);
//         // let orig_mu = self.protocol.chemical_potential(self.t as usize);
//         let curr_period = t * self.length / PERIOD_LENGTH;
//         let epsilon = self.ep_slopes[(t * self.length / PERIOD_LENGTH) as usize];

//         self.ep_accum += epsilon;
//         self.mu_accum += mu;
//         let chemical_potential = (orig_mu + self.mu_accum).clamp(-20.0, 20.0);
//         let interaction_energy = (orig_epsilon + self.ep_accum).clamp(0.0, 20.0);
//         let step = ProtocolStep::new(chemical_potential, interaction_energy);
//         self.t += 1.0 / self.protocol.num_megasteps() as f64;
//         Some(step)
//     }
// }

// impl ExactSizeIterator for FLLMegastepIter {
//     // We can easily calculate the remaining number of iterations.
//     fn len(&self) -> usize {
//         self.length - (self.t * self.length as f64) as usize    }
// }

// impl Peekable for FLLMegastepIter {
//     type Output = ProtocolStep;

//     fn peek(&self) -> Self::Output {
//         let (epsilon, mu) = self.nn.eval(self.t);
//         let orig_epsilon = self.protocol.interaction_energy(self.t as usize);
//         let orig_mu = self.protocol.chemical_potential(self.t as usize);
//         let chemical_potential = (orig_mu + mu + self.mu_accum).clamp(-20.0, 20.0);
//         let interaction_energy = (orig_epsilon + epsilon + self.ep_accum).clamp(0.0, 20.0);
//         ProtocolStep::new(chemical_potential, interaction_energy)
//     }
// }
