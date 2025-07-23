use anyhow::Result;
use engine::EvoEngine;
use fitness::FitnessFunc;
use nn::fll::FLLConfig;
use nn::{microstate::MicrostateConfig, timenet::TimeNetConfig};
// use nn::{fll_temp_only::FLLTempOnlyConfig, l2g_nn::NnConfig};
use nn::{Dna, LearningStrategy};
use rand::{Rng, SeedableRng};
use vmmc::{
    protocol::{ProtocolIter, ProtocolStep, SynthesisProtocol},
    run_vmmc,
    vmmc::Vmmc,
    vmmc_from_simparams, Prng, SimParams,
};
// use rand_core::SeedableRng;

pub mod engine;
pub mod fitness;
pub mod io;
pub mod nn;
pub mod pruning;

impl EvoEngine {
    /// Default dna currently set to microstate
    pub fn init_dna(&self) -> Dna {
        let proto = self.init_protocol.clone();
        // TODO: make into `from_strategy` method
        match self.learning_strategy {
            // TODO: parameterize by num_layers?
            LearningStrategy::Timenet => {
                let config = TimeNetConfig::new(self.seed, 0, 1000, self.mutation_factor as f64);
                Dna::fresh_time_net(config, proto)
            }
            LearningStrategy::Fll => {
                // TODO: parameterize by number of phases?
                let num_phases = 10;
                let config = FLLConfig::new(num_phases, self.mutation_factor);
                Dna::fresh_fll(config, proto)
            }
            LearningStrategy::MicroState => {
                let config = MicrostateConfig::new(&self.sim_params.shapes, self.mutation_factor);
                Dna::fresh_microstate(config, proto)
            }
        }
    }
}

impl Default for EvoEngine {
    fn default() -> Self {
        let sim_params: SimParams = Default::default();
        let seed = Prng::from_os_rng().random::<u32>();

        let num_generations = 3;
        let children_per_survivor = 3;
        let survivors_per_generation = 1;

        let init_protocol = SynthesisProtocol::flat_protocol(0.0, 10.0, 100);

        // let nn_config = NnConfig::new(nn_seed, 0, 1000, 0.1);
        // let config = NnConfig::new(seed, 0, 1000, 0.1);

        // let config = FLLConfig::new(10, 0.5);

        // let init_dna = Dna::new(0, DnaInner::TimeNet(nn_config, protocol));
        // let init_dna = Dna::fresh_time_net(config, protocol);
        // let init_dna = Dna::fresh_fll(config, protocol);
        // let init_dna = Dna::fresh_microstate(config, protocol);

        Self {
            sim_params,
            learning_strategy: LearningStrategy::MicroState,
            fitness_func: FitnessFunc::PolygonSum,
            init_protocol,
            mutation_factor: 0.5,
            // init_dna,
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
    protocol_iter: Box<dyn ProtocolIter>,
    rng: &mut Prng,
) -> Result<(Vec<ProtocolStep>, Vmmc)> {
    let initial_interaction_energy = protocol_iter.start().interaction_energy();
    let mut vmmc = vmmc_from_simparams(sim_params, initial_interaction_energy, rng);
    let (proto, _) = run_vmmc(&mut vmmc, protocol_iter, vmmc::no_callback(), rng)?;
    Ok((proto, vmmc))
}

pub fn run_fresh_vmmc_to_console(
    sim_params: &SimParams,
    protocol_iter: Box<dyn ProtocolIter>,
    rng: &mut Prng,
) -> Result<(Vec<ProtocolStep>, Vmmc)> {
    let initial_interaction_energy = protocol_iter.start().interaction_energy();
    let mut vmmc = vmmc_from_simparams(sim_params, initial_interaction_energy, rng);
    let (proto, _) = run_vmmc(
        &mut vmmc,
        protocol_iter,
        Box::new(vmmc::StdCallback::new()),
        rng,
    )?;
    Ok((proto, vmmc))
}
