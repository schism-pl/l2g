use evo_vmmc::EvoVmmc;
use fitness::FitnessFunc;
use mutation::MutationFunc;
use nn::NnConfig;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use vmmc::InputParams;

pub mod evo_vmmc;
pub mod fitness;
pub mod mutation;
pub mod nn;
pub mod pruning;

impl Default for EvoVmmc {
    fn default() -> Self {
        let initial_ip = InputParams::default();
        let seed = SmallRng::from_entropy().gen::<i64>();
        let nn_seed = SmallRng::from_entropy().gen::<i64>();

        let num_generations = 3;
        let children_per_survivor = 3;
        let survivors_per_generation = 1;

        let protocol = initial_ip.protocol.clone();

        let nn_config = NnConfig::new(nn_seed, 0, 1000, 0.1);

        Self {
            initial_ip,
            fitness_func: FitnessFunc::PolygonSum,
            initial_mutation_func: MutationFunc::LearningToGrowClassic(nn_config, protocol),
            seed,
            num_generations,
            children_per_survivor,
            survivors_per_generation,
        }
    }
}
