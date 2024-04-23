use std::fs::create_dir_all;

use crate::fitness::FitnessFunc;
use crate::mutation::Mutator;
use crate::nn::NueralNet;
use crate::pruning::prune;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use vmmc::io::{write_geometry_png, write_protocols_png, write_stats};
use vmmc::protocol::ProtocolIter;
use vmmc::{run_vmmc, vmmc::Vmmc, vmmc_from_config, InputParams};
use Mutator::*;

pub fn run_fresh_vmmc(
    ip: &InputParams,
    protocol_iter: impl ProtocolIter,
    rng: &mut SmallRng,
) -> Vmmc {
    let mut vmmc = vmmc_from_config(ip, rng);
    let _: Result<()> = run_vmmc(&mut vmmc, protocol_iter, vmmc::no_callback(), rng);
    vmmc
}

// TODO: add diagram to markdown
#[derive(Clone)]
pub struct EvoState {
    pub ip: InputParams,
    pub mutator: Mutator,
}

impl EvoState {
    pub fn new(ip: InputParams, mutator: Mutator) -> Self {
        Self {
            ip: ip.clone(),
            mutator: mutator.clone(),
        }
    }

    pub fn mutate(&mut self) -> Self {
        self.mutator.mutate();
        let f = self.mutator.clone();
        Self {
            ip: self.ip.clone(),
            mutator: f,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EvoVmmc {
    // TODO: change this to u64
    pub seed: i64, // toml crashes when I try to store as u64?
    // pub initial_state: EvoState,
    pub initial_ip: InputParams,
    pub initial_mutator: Mutator,

    pub fitness_func: FitnessFunc,

    pub num_generations: usize,
    pub survivors_per_generation: usize,
    pub children_per_survivor: usize,
}

impl EvoVmmc {
    pub fn initial_ip(&self) -> &InputParams {
        &self.initial_ip
    }

    pub fn num_generations(&self) -> usize {
        self.num_generations
    }

    pub fn children_per_survivor(&self) -> usize {
        self.children_per_survivor
    }

    pub fn survivors_per_generation(&self) -> usize {
        self.survivors_per_generation
    }

    pub fn generation_size(&self) -> usize {
        self.survivors_per_generation * self.children_per_survivor
    }

    pub fn initial_evo_state(&self) -> EvoState {
        EvoState::new(self.initial_ip.clone(), self.initial_mutator.clone())
    }

    pub fn initial_evo_states(&self) -> Vec<EvoState> {
        vec![self.initial_evo_state(); self.generation_size()]
    }

    // Executes a generation
    fn step_generation(&mut self, states: &[EvoState], rng: &mut SmallRng) -> Vec<Vmmc> {
        use Mutator::*;
        let seeds: Vec<u64> = (0..states.len()).map(|_| rng.gen()).collect();
        states
            .par_iter()
            .enumerate()
            .map(|(idx, s)| {
                let thread_seed = seeds[idx];
                let mut thread_rng = SmallRng::seed_from_u64(thread_seed);
                match &s.mutator {
                    TimeParticleNet(nn_config, protocol) => {
                        let nn = NueralNet::from_config(nn_config);
                        run_fresh_vmmc(&s.ip, nn.current_protocol(protocol), &mut thread_rng)
                    }
                    TimeNet(nn_config, protocol) => {
                        let nn = NueralNet::from_config(nn_config);
                        run_fresh_vmmc(&s.ip, nn.current_protocol(protocol), &mut thread_rng)
                    }
                }
            })
            .collect()
    }

    pub fn step_all(&mut self, rng: &mut SmallRng) {
        println!("Creating threadpool of {} workers", self.generation_size());

        // Create initial generation
        let mut evo_states = self.initial_evo_states();
        for idx in 0..self.num_generations {
            println!("Starting generation {:?}", idx);

            // 1.) Execute a generations worth of sims
            let children = self.step_generation(&evo_states, rng);
            // Dump outputs
            for (child_idx, child) in children.iter().enumerate() {
                let p_str = format!("./out/{idx}/{child_idx}");
                let out_path = std::path::Path::new(&p_str);
                create_dir_all(out_path).unwrap();
                let ip = &evo_states[idx].ip;
                let mutator = &evo_states[idx].mutator;

                let toml = toml::to_string(&ip).unwrap();
                std::fs::write(format!("{p_str}/config.toml"), toml).expect("Unable to write file");
                write_geometry_png(child, &format!("{p_str}/geometry.png"));
                match mutator {
                    TimeParticleNet(nn_config, protocol) => write_protocols_png(
                        NueralNet::from_config(nn_config).current_protocol(protocol),
                        &format!("{p_str}/protocols.png"),
                    ),
                    TimeNet(nn_config, protocol) => write_protocols_png(
                        NueralNet::from_config(nn_config).current_protocol(protocol),
                        &format!("{p_str}/protocols.png"),
                    ),
                };
                write_stats(child, &format!("{p_str}/stats.txt"))
            }

            println!(
                "Children executed: fitnesses = {:?}",
                children
                    .iter()
                    .map(|c| self.fitness_func.eval(c))
                    .collect::<Vec<f64>>()
            );
            // 2.) Use Fitness function to trim down to the survivors
            println!("Pruning survivors");
            let survivors = prune(
                &evo_states,
                children,
                self.survivors_per_generation(),
                &self.fitness_func,
            );
            // 3.) Use Mutation function to get back to normal number of sims
            println!("Generating children");
            evo_states = self.spawn_children(survivors);
        }
    }

    // generate children from survivors of previously generations
    pub fn spawn_children(&self, survivors: Vec<EvoState>) -> Vec<EvoState> {
        let mut children = Vec::new();
        for mut s in survivors.into_iter() {
            for _ in 0..self.children_per_survivor {
                let child_state = s.mutate();
                children.push(child_state);
            }
        }
        children
    }
}
