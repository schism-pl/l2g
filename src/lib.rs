use dna::Dna;
use engine::EvoEngine;
use fitness::FitnessFunc;
use nn::NnConfig;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use vmmc::protocol::SynthesisProtocol;

pub mod dna;
pub mod engine;
pub mod fitness;
pub mod nn;
pub mod pruning;

impl Default for EvoEngine {
    fn default() -> Self {
        let sim_params = Default::default();
        let seed = SmallRng::from_entropy().gen::<u32>();
        let nn_seed = SmallRng::from_entropy().gen::<u32>();

        let num_generations = 3;
        let children_per_survivor = 3;
        let survivors_per_generation = 1;

        let protocol = SynthesisProtocol::flat_protocol(0.0, 10.0, 10);

        let nn_config = NnConfig::new(nn_seed, 0, 1000, 0.1);

        Self {
            sim_params,
            fitness_func: FitnessFunc::PolygonSum,
            init_dna: Dna::TimeNet(nn_config, protocol),
            seed,
            num_generations,
            children_per_survivor,
            survivors_per_generation,
        }
    }
}
