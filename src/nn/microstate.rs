use runnt::initialization::InitializationType;
/// Implement microstate NN + TimeNet from original paper
use runnt::{activation::ActivationType, nn::NN, regularization::Regularization};
use serde::{Deserialize, Serialize};
use vmmc::morphology::Morphology;
use vmmc::{
    polygons::calc_bond_distribution,
    protocol::{ProtocolIter, ProtocolStep, SynthesisProtocol},
    vmmc::Vmmc,
};

// TODO: probably don't need this many nodes in hidden layer
// TODO: runnt fastrand => SmallRng
// TODO: fix initialization so first protocol is always the one requested?
// maybe try xavier to see if its reasonable?

#[derive(Clone, Serialize, Deserialize)]
pub struct MicrostateConfig {
    nn: NN,
    // num_patches: usize,
    mutation_factor: f32,
    // len: usize,
}

// TODO: how to initilize network to something sane?
// Why is fixed 0.0 weights not just the original protocol???
// the nueral net is indeed being initialized to 0
// where is other results coming from?
impl MicrostateConfig {
    // Reusable so long as you take shape vector as argument
    // num_patches + 2 bc 0..n (n+1), and time t
    pub fn new(shapes: &[Morphology], mutation_factor: f32) -> Self {
        // for each shape, get num patches + 1 and sum them
        let num_patches = shapes.iter().map(|s| s.patches().len() + 1).sum::<usize>() + 1;
        let nn = NN::new(&[num_patches, 1000, 2])
            .with_learning_rate(0.1)
            .with_hidden_type(ActivationType::Tanh)
            .with_initialization(InitializationType::Fixed(0.0))
            //.with_output_type(ActivationType::Linear)
            .with_regularization(Regularization::None);

        Self {
            nn,
            // num_patches,
            mutation_factor,
            // len,
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

    pub fn proto_iter(&self, proto: &SynthesisProtocol) -> impl ProtocolIter {
        MicroStateIter::new(self.nn.clone(), proto)
    }
}

fn format_inputs_for_one(patch_distr: &[usize]) -> Vec<f32> {
    let total_particles: usize = patch_distr.iter().sum();
    patch_distr
        .iter()
        .map(|&x| {
            if total_particles == 0 {
                0.0
            } else {
                x as f32 / total_particles as f32
            }
        })
        .collect()
}

fn format_inputs(patch_distrs: &[Vec<usize>], t: f32) -> Vec<f32> {
    let mut inputs = patch_distrs
        .iter()
        .flat_map(|d| format_inputs_for_one(d))
        .collect::<Vec<_>>();
    inputs.push(t);
    inputs
}

pub struct MicroStateIter {
    nn: NN,
    t: f64,
    protocol: SynthesisProtocol,
}

impl MicroStateIter {
    fn new(nn: NN, protocol: &SynthesisProtocol) -> Self {
        Self {
            nn,
            t: 0.0,
            protocol: protocol.clone(),
        }
    }

    // TODO: solve hidden randomness
    // inputs should be [t, p0..p_n]
    fn eval(&self, patch_distrs: &[Vec<usize>], t: f32) -> (f64, f64) {
        // Get particle distribution
        let inputs = format_inputs(patch_distrs, t);
        let outputs = self.nn.forward(&inputs);
        //println!("{:?} -> {:?}", inputs, slopes);
        assert_eq!(outputs.len(), 2); // interaction energy, chemical potential
        (outputs[1] as f64, outputs[0] as f64) // TODO: scale?
    }
}

impl ProtocolIter for MicroStateIter {
    fn next(&mut self, vmmc: &Vmmc) -> Option<ProtocolStep> {
        if self.t >= 1.0 {
            return None;
        }

        let patch_distrs = calc_bond_distribution(vmmc);

        let (epsilon, mu) = self.eval(&patch_distrs, self.t as f32);

        let orig_epsilon = self
            .protocol
            .interaction_energy((self.t * self.protocol.num_megasteps() as f64) as usize);
        let orig_mu = self
            .protocol
            .chemical_potential((self.t * self.protocol.num_megasteps() as f64) as usize);
        // println!("{:?}: {epsilon} {mu} {orig_epsilon} {orig_mu}", (self.t * self.protocol.num_megasteps() as f64) as usize);
        let chemical_potential = (orig_mu + mu).clamp(-20.0, 20.0);
        let interaction_energy = (orig_epsilon + epsilon).clamp(0.0, 20.0);
        // println!("({chemical_potential} --- {interaction_energy})");
        let step = ProtocolStep::new(chemical_potential, interaction_energy);
        self.t += 1.0 / self.protocol.num_megasteps() as f64;
        Some(step)
    }

    // TODO: which patch distribution to use for multi-morphology systems?
    fn peek(&self, vmmc: &Vmmc) -> ProtocolStep {
        let patch_distrs = calc_bond_distribution(vmmc);

        let (epsilon, mu) = self.eval(&patch_distrs, self.t as f32);
        let orig_epsilon = self
            .protocol
            .interaction_energy((self.t * self.protocol.num_megasteps() as f64) as usize);
        let orig_mu = self
            .protocol
            .chemical_potential((self.t * self.protocol.num_megasteps() as f64) as usize);
        let chemical_potential = (orig_mu + mu).clamp(-20.0, 20.0);
        let interaction_energy = (orig_epsilon + epsilon).clamp(0.0, 20.0);
        ProtocolStep::new(chemical_potential, interaction_energy)
    }

    fn start(&self) -> ProtocolStep {
        let orig_epsilon = self.protocol.interaction_energy(0);
        let orig_mu = self.protocol.chemical_potential(0);
        let chemical_potential = (orig_mu).clamp(-20.0, 20.0);
        let interaction_energy = (orig_epsilon).clamp(0.0, 20.0);
        ProtocolStep::new(chemical_potential, interaction_energy)
    }

    fn len(&self) -> usize {
        self.protocol.num_megasteps() - (self.t * self.protocol.num_megasteps() as f64) as usize
    }
}
