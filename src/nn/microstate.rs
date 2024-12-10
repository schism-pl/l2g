/// Implement microstate NN + TimeNet from original paper
use runnt::{activation::ActivationType, nn::NN, regularization::Regularization};
use serde::{Deserialize, Serialize};
use vmmc::protocol::{ProtocolStep, SynthesisProtocol};

// TODO: probably don't need this many nodes in hidden layer

#[derive(Clone, Serialize, Deserialize)]
pub struct MicrostateConfig {
    nn: NN,
    num_patches: usize,
    mutation_factor: f32,
    len: usize,
}

impl MicrostateConfig {
    // Reusable so long as you take shape vector as argument
    pub fn new(num_patches: usize, len: usize, mutation_factor: f32) -> Self {
        let nn = NN::new(&[num_patches + 1, 1000, 2])
            .with_learning_rate(0.1)
            .with_hidden_type(ActivationType::Sigmoid)
            .with_output_type(ActivationType::Linear)
            .with_regularization(Regularization::None);
        Self {
            nn,
            num_patches,
            mutation_factor,
            len,
        }
    }

    // Reusable across architectures
    pub fn mutate(&mut self) {
        let mut weights = self.nn.get_weights();
        for w in &mut weights {
            *w = (*w + (fastrand::f32() * 2.0 - 1.0) * self.mutation_factor).clamp(-1.0, 1.0);
        }
        self.nn.set_weights(&weights);
    }


    pub fn proto_vec(&self, patch_bonds: &[usize], proto: &SynthesisProtocol) -> Vec<ProtocolStep> {
        let mut steps = Vec::new();
        for t in 0..self.len {
            let time = t as f32 / self.len as f32;
            let patches = patch_bonds.iter().map(|&c| c as f32 / 1000.0).collect::<Vec<_>>();
            let mut inputs = patches.to_vec();
            inputs.push(time);

            let outputs = self.nn.forward(&inputs);
            
            let orig_epsilon = proto.interaction_energy(t);
            let orig_mu = proto.chemical_potential(t);

            let chemical_potential = (orig_mu + outputs[1] as f64).clamp(-20.0, 20.0);
            let interaction_energy = (orig_epsilon + outputs[0] as f64).clamp(0.0, 20.0);

            let step = ProtocolStep::new(chemical_potential, interaction_energy);
            steps.push(step);
        }
        steps
    }
}
