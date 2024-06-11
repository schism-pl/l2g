use rand::{rngs::SmallRng, SeedableRng};
/// Implement nueral net implementation from original paper
use rand_distr::{Distribution, Normal};
use runnt::{activation::ActivationType, nn::NN, regularization::Regularization};
use serde::{Deserialize, Serialize};
use vmmc::protocol::{self, Peekable, ProtocolStep, SynthesisProtocol};

// TODO: fixed at self.num_phases,self.num_phases,self.num_phases*2 architecture
// TODO: currently does not incorporate targeted protocol
// will ony

// const NUM_PHASES: usize = 10;
// const PERIOD_LENGTH: usize = 100;
// const Times: [f32; 10] = [0., 1., 2., 3., 4. ,5., 6.,7.,8.,9.];

#[derive(Clone, Serialize, Deserialize)]
// Uniquely IDs a nueral net
// NN can be reconstructed by building nn from original set and
// mutating it by `child_id` times
pub struct FLLConfig {
    nn: runnt::nn::NN,
    num_phases: usize,
    phase_length: usize,
}

impl FLLConfig {
    pub fn new(num_phases: usize, phase_length: usize) -> Self {
        let nn = runnt::nn::NN::new(&[num_phases, num_phases, num_phases * 2])
            .with_learning_rate(0.1)
            .with_hidden_type(ActivationType::Sigmoid)
            .with_output_type(ActivationType::Linear)
            .with_regularization(Regularization::None);
        Self {
            nn,
            num_phases,
            phase_length,
        }
    }

    // TODO: only uses protocol for initial info
    pub fn proto_vec(&self, proto: &SynthesisProtocol) -> Vec<ProtocolStep> {
        let times =
            Vec::from_iter((0..self.num_phases).map(|phase| phase as f32 / self.num_phases as f32));
        let mut epsilon_slopes = self.nn.forward(&times);
        let mu_slopes = epsilon_slopes.split_off(self.num_phases);

        let mut epsilon = proto.interaction_energy(0);
        let mut mu = proto.chemical_potential(0);

        let step = ProtocolStep::new(mu, epsilon);
        let mut steps = vec![step];

        for phase in 0..self.num_phases {
            let epsilon_delta = epsilon_slopes[phase] as f64 / self.phase_length as f64;
            let mu_delta = mu_slopes[phase] as f64 / self.phase_length as f64;
            for t in 0..self.phase_length {
                epsilon = epsilon + epsilon_delta;
                mu = mu + mu_delta;
                let step = ProtocolStep::new(mu, epsilon);
                steps.push(step);
            }
        }

        assert_eq!(steps.len(), self.num_phases * self.phase_length);
        steps
        // let nn = NueralNet::from_config(self);
        // nn.current_protocol(proto).collect()
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
