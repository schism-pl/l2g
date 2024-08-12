use anyhow::Result;
use engine::EvoEngine;
use fitness::FitnessFunc;
use nn::fll_fixed_particle::FLLFixedParticleConfig;
use nn::Dna;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use vmmc::{
    protocol::{ProtocolIter, SynthesisProtocol},
    run_vmmc,
    vmmc::Vmmc,
    vmmc_from_simparams, SimParams,
};

pub mod engine;
pub mod fitness;
pub mod io;
pub mod nn;
pub mod pruning;

impl Default for EvoEngine {
    fn default() -> Self {
        let sim_params = Default::default();
        let seed = SmallRng::from_entropy().gen::<u32>();

        let num_generations =3;
        let children_per_survivor = 3;
        let survivors_per_generation = 3;

        let protocol = SynthesisProtocol::flat_protocol(0.0, 10.0, 50);

        // let nn_config = NnConfig::new(nn_seed, 0, 1000, 0.1);
        let config = FLLFixedParticleConfig::new(10, 5, 0.5);

        // let init_dna = Dna::new(0, DnaInner::TimeNet(nn_config, protocol));
        let init_dna = Dna::fresh_fll_fixed_particle(config, protocol);

        Self {
            sim_params,
            fitness_func: FitnessFunc::PolygonSum,
            init_dna,
            seed,
            num_generations,
            children_per_survivor,
            survivors_per_generation,
            child_ctr: 0,
            history: Vec::new(),
            fitnesses: Vec::new(),
        }
    }
}

pub fn run_fresh_vmmc(
    sim_params: &SimParams,
    protocol_iter: impl ProtocolIter,
    rng: &mut SmallRng,
) -> Vmmc {
    let initial_interaction_energy = protocol_iter.peek().interaction_energy();
    let mut vmmc = vmmc_from_simparams(sim_params, initial_interaction_energy, rng);
    let _: Result<()> = run_vmmc(&mut vmmc, protocol_iter, vmmc::no_callback(), rng);
    vmmc
}
