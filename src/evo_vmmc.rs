use std::fs::create_dir_all;

// use crate::config::L2GInputParams;
use crate::fitness::FitnessFunc;
use crate::mutation::MutationFunc;
use crate::nn::NueralNet;
use crate::pruning::prune;
use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use vmmc::io::write_geometry_png;
use vmmc::protocol::SynthesisProtocol;
use vmmc::{run_vmmc, vmmc::Vmmc, vmmc_from_config, InputParams};

// TODO: parallelize executing simulations

pub fn run_fresh_vmmc(ip: &InputParams, rng: &mut SmallRng) -> Vmmc {
    let mut vmmc = vmmc_from_config(&ip, rng);
    let _: Result<()> = run_vmmc(
        &mut vmmc,
        ip.protocol.megastep_iter(),
        vmmc::no_callback(),
        rng,
    );
    vmmc
}

// as polynomial, to polynomial

// TODO: add diagram to markdown
// 0. load config using serde (has embedded inputconfig?)
// 1. Spin up first generation from InputConfig
// 2.

// Current strat: just ignore the synthesis protocol in the ip

#[derive(Clone)]
pub struct EvoState {
    pub ip: InputParams,
    pub mutation_func: MutationFunc,
}

impl EvoState {
    pub fn new(ip: InputParams, mutation_func: MutationFunc) -> Self {
        Self {
            ip: ip.clone(),
            mutation_func: mutation_func.clone(),
        }
    }

    pub fn mutate(&mut self, rng: &mut SmallRng) -> Self {
        self.mutation_func.mutate(rng);
        let f = self.mutation_func.clone();
        Self {
            ip: self.ip.clone(),
            mutation_func: f,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EvoVmmc {
    // TODO: change this to u64
    pub seed: i64, // toml crashes when I try to store as u64?
    initial_ip: InputParams,

    pub fitness_func: FitnessFunc,
    pub initial_mutation_func: MutationFunc,

    num_generations: usize,
    survivors_per_generation: usize,
    children_per_survivor: usize,
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
        EvoState::new(self.initial_ip.clone(), self.initial_mutation_func.clone())
    }

    pub fn initial_evo_states(&self) -> Vec<EvoState> {
        vec![self.initial_evo_state(); self.generation_size()]
    }

    // Executes a generation
    fn step_generation(&mut self, states: &[EvoState], rng: &mut SmallRng) -> Vec<Vmmc> {
        let mut sims = Vec::new();
        for state in states.iter() {
            let vmmc = run_fresh_vmmc(&state.ip, rng);
            sims.push(vmmc);
        }
        sims
    }

    pub fn step_all(&mut self, rng: &mut SmallRng) {
        // Create initial generation
        let mut evo_states = self.initial_evo_states();
        for idx in 0..self.num_generations {
            println!("Starting generation {:?}", idx);

            // 1.) Execute a generations worth of sims
            let children = self.step_generation(&evo_states, rng);
            // Dump outputs
            // TODO: need to get actual output path from config object
            for (child_idx, child) in children.iter().enumerate() {
                let p_str = format!("./out/{idx}/{child_idx}");
                let out_path = std::path::Path::new(&p_str);
                create_dir_all(out_path).unwrap();
                let ip = &evo_states[idx].ip;
                // dump full config toml to output directory
                // TODO: dump final stats, both here and in the original vmmc
                let toml = toml::to_string(&ip).unwrap();
                std::fs::write(format!("{p_str}/config.toml"), toml).expect("Unable to write file");
                write_geometry_png(&child, &format!("{p_str}/geometry.png"));
                // write_protocols_png(&ip, &config.protocols());
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
            let survivor_states = prune(
                &evo_states,
                children,
                self.survivors_per_generation(),
                &self.fitness_func,
            );
            // 3.) Use Mutation function to get back to normal number of sims
            println!("Generating children");
            evo_states = self.spawn_children(rng, survivor_states);
        }
    }

    // generate children from survivors of previously generations
    // survivors: [survivors_per_generation; Vmmc] -> p[]
    pub fn spawn_children(&self, rng: &mut SmallRng, survivors: Vec<EvoState>) -> Vec<EvoState> {
        let mut children = Vec::new();
        for mut s in survivors.into_iter() {
            for _ in 0..self.children_per_survivor {
                let child_state = s.mutate(rng);
                // let child_ip = self.mutation_func.mutate(ip);
                children.push(child_state);
            }
        }
        children
    }
}

impl Default for EvoVmmc {
    fn default() -> Self {
        let initial_ip = InputParams::default();
        let seed = SmallRng::from_entropy().gen::<i64>();

        let num_generations = 3;
        let children_per_survivor = 3;
        let survivors_per_generation = 1;

        // let num_megasteps = initial_ip.protocol.num_megasteps();

        Self {
            initial_ip,
            fitness_func: FitnessFunc::PolygonSum,
            initial_mutation_func: MutationFunc::LearningToGrowClassic(NueralNet::new()),
            // initial_mutation_func: MutationFunc::UniformRandomCoefficients(
            //     SynthesisProtocol::new("8.0", "0.0", num_megasteps),
            //     [8.0, 0.0, 0.0, 0.0].to_vec(),
            //     [0.0, 0.0, 0.0, 0.0].to_vec(),
            //     0.1,
            // ),
            seed,
            num_generations,
            children_per_survivor,
            survivors_per_generation,
        }
    }
}
