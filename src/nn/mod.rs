pub mod dna;
pub mod new_nn;
use rand::{rngs::SmallRng, SeedableRng};
/// Implement nueral net implementation from original paper
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use vmmc::protocol::{Peekable, ProtocolStep, SynthesisProtocol};

#[derive(Clone, Serialize, Deserialize, Debug)]
// Uniquely IDs a nueral net
// NN can be reconstructed by building nn from original set and
// mutating it by `child_id` times
pub struct NnConfig {
    orig_seed: u32,
    child_id: u32,
    num_layers: u32,
    mutation_factor: f64,
}

impl NnConfig {
    pub fn new(orig_seed: u32, child_id: u32, num_layers: u32, mutation_factor: f64) -> Self {
        Self {
            orig_seed,
            child_id,
            num_layers,
            mutation_factor,
        }
    }

    pub fn set_child_id(&mut self, child_id: u32) {
        self.child_id = child_id
    }

    pub fn child_id(&self) -> u32 {
        self.child_id
    }
}

// architecture = 1 input, 1000 hidden layers of 1 node each, and 2 outputs (mu and epsilon)
#[derive(Clone, Debug)]
pub struct HiddenLayer {
    input_weight: f64,
    epsilon_weight: f64,
    mu_weight: f64,
    bias: f64,
}

impl HiddenLayer {
    pub fn new(input_weight: f64, epsilon_weight: f64, mu_weight: f64, bias: f64) -> Self {
        Self {
            input_weight,
            epsilon_weight,
            mu_weight,
            bias,
        }
    }

    pub fn eval(&self, t: f64) -> f64 {
        (t * self.input_weight + self.bias).tanh()
    }

    pub fn mutate(&mut self, mag: f64, rng: &mut SmallRng) {
        let normal = Normal::new(0.0, mag).unwrap();

        self.input_weight += normal.sample(rng);
        self.epsilon_weight += normal.sample(rng);
        self.mu_weight += normal.sample(rng);
        self.bias += normal.sample(rng);
    }
}

#[derive(Clone, Debug)]
pub struct NueralNet {
    layers: Vec<HiddenLayer>,
    mutation_factor: f64,
    rng: SmallRng,
}

impl NueralNet {
    pub fn from_config(config: &NnConfig) -> Self {
        let mut rng = SmallRng::seed_from_u64(config.orig_seed as u64);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut layers = Vec::new();

        for _ in 0..config.num_layers {
            let iw = normal.sample(&mut rng);
            let epsilon = normal.sample(&mut rng);
            let mu = normal.sample(&mut rng);
            let bias = normal.sample(&mut rng);
            let layer = HiddenLayer::new(iw, epsilon, mu, bias);
            layers.push(layer);
        }

        let mut nn = NueralNet {
            layers,
            mutation_factor: config.mutation_factor,
            rng,
        };

        for _ in 0..config.child_id {
            nn.mutate();
        }

        nn
    }

    // t is in [0; 1.0]. Represents current time / total time of simulation
    pub fn eval(&self, t: f64) -> (f64, f64) {
        let epsilon = self
            .layers
            .iter()
            .map(|l| l.eval(t) * l.epsilon_weight)
            .sum::<f64>()
            / self.layers.len() as f64;
        let mu = self
            .layers
            .iter()
            .map(|l| l.eval(t) * l.mu_weight)
            .sum::<f64>()
            / self.layers.len() as f64;
        // println!("nn eval {epsilon} {mu}");
        (epsilon, mu)
    }

    pub fn mutate(&mut self) {
        self.layers
            .iter_mut()
            .for_each(|l| l.mutate(self.mutation_factor, &mut self.rng))
    }

    pub fn current_protocol(self, protocol: &SynthesisProtocol) -> NnMegastepIter {
        NnMegastepIter::new(self, protocol)
    }
}

pub struct NnMegastepIter<'a> {
    nn: NueralNet,
    t: f64,
    protocol: &'a SynthesisProtocol,
    ep_accum: f64,
    mu_accum: f64,
}

impl<'a> NnMegastepIter<'a> {
    fn new(nn: NueralNet, protocol: &'a SynthesisProtocol) -> Self {
        Self {
            nn,
            t: 0.0,
            protocol,
            ep_accum: 0.0,
            mu_accum: 0.0,
        }
    }
}

impl<'a> Iterator for NnMegastepIter<'a> {
    type Item = ProtocolStep;

    fn next(&mut self) -> Option<Self::Item> {
        if self.t >= 1.0 {
            return None;
        }

        let (epsilon, mu) = self.nn.eval(self.t);
        let orig_epsilon = self.protocol.interaction_energy(self.t as usize);
        let orig_mu = self.protocol.chemical_potential(self.t as usize);
        self.ep_accum += epsilon;
        self.mu_accum += mu;
        let chemical_potential = (orig_mu + self.mu_accum).clamp(-20.0, 20.0);
        let interaction_energy = (orig_epsilon + self.ep_accum).clamp(0.0, 20.0);
        let step = ProtocolStep::new(chemical_potential, interaction_energy);
        self.t += 1.0 / self.protocol.num_megasteps() as f64;
        Some(step)
    }
}

impl<'a> ExactSizeIterator for NnMegastepIter<'a> {
    // We can easily calculate the remaining number of iterations.
    fn len(&self) -> usize {
        self.protocol.num_megasteps() - (self.t * self.protocol.num_megasteps() as f64) as usize
    }
}

impl<'a> Peekable for NnMegastepIter<'a> {
    type Output = ProtocolStep;

    fn peek(&self) -> Self::Output {
        let (epsilon, mu) = self.nn.eval(self.t);
        let orig_epsilon = self.protocol.interaction_energy(self.t as usize);
        let orig_mu = self.protocol.chemical_potential(self.t as usize);
        let chemical_potential = (orig_mu + mu + self.mu_accum).clamp(-20.0, 20.0);
        let interaction_energy = (orig_epsilon + epsilon + self.ep_accum).clamp(0.0, 20.0);
        ProtocolStep::new(chemical_potential, interaction_energy)
    }
}
