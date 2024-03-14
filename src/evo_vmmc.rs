// use crate::config::L2GInputParams;
use crate::fitness::FitnessFunc;
use crate::mutation::MutationFunc;
use crate::pruning::prune;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use vmmc::{run_vmmc, vmmc::Vmmc, vmmc_from_config, InputParams};

pub fn run_fresh_vmmc(ip: &InputParams, rng: &mut SmallRng) -> Vmmc {
    let mut vmmc = vmmc_from_config(&ip, rng);
    let _: Result<()> = run_vmmc(&mut vmmc, ip.protocol.clone(), vmmc::no_callback(), rng);
    vmmc
}

// TODO: add diagram to markdown
// 0. load config using serde (has embedded inputconfig?)
// 1. Spin up first generation from InputConfig
// 2.

#[derive(Clone)]

pub struct EvoState {
    pub ip: InputParams,
    pub mutation_func: MutationFunc,
}

impl EvoState {
    pub fn mutate(&mut self) -> Self {
        let ip = self.mutation_func.mutate(&self.ip);
        let f = self.mutation_func.clone();
        Self {
            ip,
            mutation_func: f,
        }

    }
}

// fn get_initial_evostates(n: usize) -> Vec<>

#[derive(Clone, Serialize, Deserialize)]
// #[serde(default)]
pub struct EvoVmmc {
    // TODO: change this to u64
    pub seed: i64, // toml crashes when I try to store as u64?
    ip: InputParams,

    pub fitness_func: FitnessFunc,
    pub initial_mutation_func: MutationFunc,

    num_generations: usize,
    survivors_per_generation: usize, // # of children post-pruning
    children_per_survivor: usize,
}

impl EvoVmmc {
    pub fn ip(&self) -> &InputParams {
        &self.ip
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

    // Executes a generation
    fn step_generation(&mut self, ips: &[InputParams], rng: &mut SmallRng) -> Vec<Vmmc> {
        let mut sims = Vec::new();
        for ip in ips.iter() {
            let vmmc = run_fresh_vmmc(ip, rng);
            sims.push(vmmc);
        }
        sims
    }

    // TODO: how to derive new generation from old?
    pub fn step_all(&mut self, rng: &mut SmallRng) {
        // Create initial generation
        let mut active_states: Vec<EvoState> = vec![
            InputParams::default();
            self.survivors_per_generation
                * self.children_per_survivor
        ];
        for idx in 0..self.num_generations {
            println!("Starting generation {:?}", idx);
            // 1.) Execute a generations worth of sims
            // [IP] -> [Vmmc]
            let children = self.step_generation(&active_ips, rng);
            // 2.) Use Fitness function to trim down to the survivors
            // [Ips] -> [Vmmc] -> [Ip]
            let survivor_states = prune(
                &active_ips,
                children,
                self.survivors_per_generation(),
                &self.fitness_func,
            );
            // 3.) Use Mutation function to get back to normal number of sims
            // [(Ip)] -> [Ip]
            // [Sim] -> [Sim]
            active_ips = self.spawn_children(survivor_states);
        }
    }

    // generate children from survivors of previously generations
    // survivors: [survivors_per_generation; Vmmc] -> p[]
    pub fn spawn_children(&self, survivors: Vec<EvoState>) -> Vec<EvoState> {
        let mut children = Vec::new();
        for mut s in survivors.into_iter() {
            for _ in 0..self.children_per_survivor {
                let child_state = s.mutate();
                // let child_ip = self.mutation_func.mutate(ip);
                children.push(child_state);
            }
        }
        children
    }
}

impl Default for EvoVmmc {
    fn default() -> Self {
        let ip = InputParams::default();
        let seed = SmallRng::from_entropy().gen::<i64>();

        let num_generations = 3;
        let children_per_survivor = 3;
        let survivors_per_generation = 1;

        Self {
            ip,
            fitness_func: FitnessFunc::PolygonSum,
            initial_mutation_func: MutationFunc::UniformRandom(0.1),
            seed,
            num_generations,
            children_per_survivor,
            survivors_per_generation,
        }
    }
}
