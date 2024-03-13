use rand::{rngs::SmallRng, Rng, SeedableRng};
use vmmc::InputParams;

use serde::{Deserialize, Serialize};

use crate::{fitness::FitnessFunc, mutation::MutationFunc};

#[derive(Clone, Serialize, Deserialize)]
// #[serde(default)]
pub struct L2GInputParams {
    // TODO: change this to u64
    pub seed: i64, // toml crashes when I try to store as u64?
    ip: InputParams,

    pub fitness_func: FitnessFunc,
    pub mutation_func: MutationFunc,

    // L2G parameters
    num_generations: usize,
    children_per_generation: usize,
    survivors_per_generation: usize, // # of children post-pruning
}

impl L2GInputParams {
    pub fn ip(&self) -> &InputParams {
        &self.ip
    }

    pub fn num_generations(&self) -> usize {
        self.num_generations
    }

    pub fn children_per_generation(&self) -> usize {
        self.children_per_generation
    }

    pub fn survivors_per_generation(&self) -> usize {
        self.survivors_per_generation
    }
}

impl Default for L2GInputParams {
    fn default() -> Self {
        let ip = InputParams::default();
        let seed = SmallRng::from_entropy().gen::<i64>();

        let num_generations = 3;
        let children_per_generation = 3;
        let survivors_per_generation = 1;

        Self {
            ip,
            fitness_func: FitnessFunc::PolygonSum,
            mutation_func: MutationFunc::UniformRandom(0.1),
            seed,
            num_generations,
            children_per_generation,
            survivors_per_generation,
        }
    }
}
