use std::fs::create_dir_all;

use crate::dna::Dna;
use crate::fitness::FitnessFunc;
use crate::nn::NueralNet;
use crate::pruning::prune;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use vmmc::io::{write_geometry_png, write_protocols_png, write_stats};
use vmmc::protocol::ProtocolIter;
use vmmc::SimParams;
use vmmc::{run_vmmc, vmmc::Vmmc, vmmc_from_simparams};
use Dna::*;

pub fn run_fresh_vmmc(
    sim_params: &SimParams,
    initial_interaction_energy: f64,
    protocol_iter: impl ProtocolIter,
    rng: &mut SmallRng,
) -> Vmmc {
    let mut vmmc = vmmc_from_simparams(sim_params, initial_interaction_energy, rng);
    let _: Result<()> = run_vmmc(&mut vmmc, protocol_iter, vmmc::no_callback(), rng);
    vmmc
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EvoEngine {
    pub seed: u32,
    pub sim_params: SimParams,
    pub init_dna: Dna,

    pub fitness_func: FitnessFunc,

    pub num_generations: usize,
    pub survivors_per_generation: usize,
    pub children_per_survivor: usize,
}

impl EvoEngine {
    pub fn sim_params(&self) -> &SimParams {
        &self.sim_params
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

    fn prune(&self, candidates: &[Dna], vmmcs: Vec<Vmmc>) -> Vec<Dna> {
        prune(
            candidates,
            vmmcs,
            self.survivors_per_generation(),
            &self.fitness_func,
        )
    }

    pub fn initial_candidates(&self) -> Vec<Dna> {
        let mut r = Vec::new();
        let mut progenitor = self.init_dna.clone();
        for _ in 0..self.generation_size() {
            r.push(progenitor.clone());
            progenitor.mutate();
        }
        r
    }

    fn step_one(&self, dna: &Dna, rng: &mut SmallRng) -> Vmmc {
        match dna {
            TimeParticleNet(nn_config, protocol) | TimeNet(nn_config, protocol) => {
                let nn = NueralNet::from_config(nn_config);
                let initial_interaction_energy = protocol.initial_interaction_energy();
                let protocol_iter = nn.current_protocol(protocol);
                run_fresh_vmmc(
                    self.sim_params(),
                    initial_interaction_energy,
                    protocol_iter,
                    rng,
                )
            }
        }
    }

    // Executes a generation
    fn step_generation(&mut self, states: &[Dna], rng: &mut SmallRng) -> Vec<Vmmc> {
        let seeds: Vec<u64> = (0..states.len()).map(|_| rng.gen()).collect();
        states
            .par_iter()
            .enumerate()
            .map(|(idx, s)| {
                let thread_seed = seeds[idx];
                let mut thread_rng = SmallRng::seed_from_u64(thread_seed);
                self.step_one(s, &mut thread_rng)
            })
            .collect()
    }

    pub fn step_all(&mut self, rng: &mut SmallRng) {
        println!("Creating threadpool of {} workers", self.generation_size());

        // Create initial generation
        let mut candidates = self.initial_candidates();
        for idx in 0..self.num_generations {
            println!("Starting generation {:?}", idx);

            // 1.) Execute a generations worth of sims
            let children = self.step_generation(&candidates, rng);
            // Dump outputs
            for (child_idx, child) in children.iter().enumerate() {
                let p_str = format!("./out/{idx}/{child_idx}");
                let out_path = std::path::Path::new(&p_str);
                create_dir_all(out_path).unwrap();
                // let toml = toml::to_string(&ip).unwrap();
                // std::fs::write(format!("{p_str}/config.toml"), toml).expect("Unable to write file");
                write_geometry_png(child, &format!("{p_str}/geometry.png"));
                match &candidates[idx] {
                    TimeParticleNet(nn_config, protocol) | TimeNet(nn_config, protocol) => {
                        write_protocols_png(
                            NueralNet::from_config(nn_config).current_protocol(protocol),
                            &format!("{p_str}/protocols.png"),
                        )
                    }
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
            let survivors = self.prune(&candidates, children);
            // 3.) Use Mutation function to get back to normal number of sims
            println!("Generating children");
            candidates = self.spawn_children(survivors);
        }
    }

    // generate children from survivors of previously generations
    pub fn spawn_children(&self, survivors: Vec<Dna>) -> Vec<Dna> {
        let mut children = Vec::new();
        for mut s in survivors.into_iter() {
            for _ in 0..self.children_per_survivor {
                s.mutate();
                children.push(s.clone());
            }
        }
        children
    }
}
