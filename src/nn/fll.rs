// use rand::{rngs::SmallRng, SeedableRng};
/// Implement nueral net implementation from original paper
// use rand_distr::{Distribution, Normal};
use runnt::{activation::ActivationType, nn::NN, regularization::Regularization};
use serde::{Deserialize, Serialize};
use vmmc::protocol::{ProtocolStep, SynthesisProtocol};

// TODO: fixed at self.num_phases,self.num_phases,self.num_phases*2 architecture
// TODO: only uses protocol for initial info
// TODO: randomness here is not reproducible

#[derive(Clone, Serialize, Deserialize)]
pub struct FLLConfig {
    nn: NN,
    num_phases: usize,
    mutation_factor: f32,
}

impl FLLConfig {
    pub fn new(num_phases: usize, mutation_factor: f32) -> Self {
        let nn = NN::new(&[num_phases, num_phases, num_phases * 2])
            .with_learning_rate(0.1)
            .with_hidden_type(ActivationType::Sigmoid)
            .with_output_type(ActivationType::Linear)
            .with_regularization(Regularization::None);
        Self {
            nn,
            num_phases,
            mutation_factor,
        }
    }

    pub fn mutate(&mut self) {
        let mut weights = self.nn.get_weights();
        for w in &mut weights {
            *w = (*w + (fastrand::f32() * 2.0 - 1.0) * self.mutation_factor).clamp(-1.0, 1.0);
        }

        self.nn.set_weights(&weights);
    }

    pub fn proto_vec(&self, proto: &SynthesisProtocol) -> Vec<ProtocolStep> {
        let times =
            Vec::from_iter((0..self.num_phases).map(|phase| phase as f32 / self.num_phases as f32));
        let mut epsilon_slopes = self.nn.forward(&times);
        let mu_slopes = epsilon_slopes.split_off(self.num_phases);

        let mut epsilon = proto.interaction_energy(0);
        let mut mu = proto.chemical_potential(0);

        let mut steps = Vec::new();
        let phase_len = proto.len() / self.num_phases;
        assert_eq!(phase_len * self.num_phases, proto.len());

        for phase in 0..self.num_phases {
            let epsilon_delta = epsilon_slopes[phase] as f64 / phase_len as f64;
            let mu_delta = mu_slopes[phase] as f64 / phase_len as f64;
            for _ in 0..phase_len {
                let step = ProtocolStep::new(mu, epsilon);
                steps.push(step);
                epsilon += epsilon_delta;
                mu += mu_delta;
            }
        }

        assert_eq!(steps.len(), self.num_phases * phase_len);
        steps
    }
}
