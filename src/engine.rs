use std::time::Instant;

use crate::fitness::FitnessFunc;
use crate::io::{record_child, record_child_config};
use crate::nn::{Dna, LearningStrategy};
use crate::pruning::prune;
use crate::run_fresh_vmmc;
use anyhow::Result;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use vmmc::polygons::calc_polygon_count;
use vmmc::protocol::{ProtocolStep, SynthesisProtocol};
use vmmc::vmmc::Vmmc;
use vmmc::Prng;
use vmmc::SimParams;

// DNA, fitness, usize
type GenePool = [(Dna, f64, usize)];

fn default_mutation_factor() -> f32 {
    0.2
}

#[derive(Clone, Serialize, Deserialize)]
pub struct EvoEngine {
    pub seed: u32,
    pub sim_params: SimParams,
    pub init_protocol: SynthesisProtocol,

    // pub init_dna: Dna,
    pub learning_strategy: LearningStrategy,
    pub fitness_func: FitnessFunc,

    pub num_generations: usize,
    pub survivors_per_generation: usize,
    pub children_per_survivor: usize,

    #[serde(default = "default_mutation_factor")]
    pub mutation_factor: f32,

    // runtime state
    #[serde(default)]
    #[serde(skip_serializing)]
    pub child_ctr: usize,
    #[serde(default)]
    #[serde(skip_serializing)]
    pub history: Vec<(usize, usize)>, // edge-list of parent-child relationships
    #[serde(default)]
    #[serde(skip_serializing)]
    pub fitnesses: Vec<f64>,
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

    fn prune(
        &self,
        genepool: &mut GenePool,
        candidates: &[Dna],
        fitnesses: Vec<f64>,
        polygons: Vec<usize>,
    ) {
        prune(
            genepool, candidates, fitnesses, polygons,
            // &self.fitness_func,
        );
    }

    fn mutate(&mut self, dna: &mut Dna) {
        // use DnaInner::*;
        dna.mutate(self.child_ctr);
        self.child_ctr += 1;
    }

    pub fn initial_candidates(&mut self) -> Vec<Dna> {
        let init_dna = self.init_dna();
        self.spawn_children(&[(init_dna.clone(), 0., 0)], self.generation_size())
    }

    fn step_one(&self, dna: &Dna, rng: &mut Prng) -> Result<(Vec<ProtocolStep>, Vmmc)> {
        let protocol_iter = dna.protocol_iter();
        run_fresh_vmmc(self.sim_params(), protocol_iter, rng)
    }

    fn step_generation_to(
        &mut self,
        states: &[Dna],
        rng: &mut Prng,
        output_dir: &str,
    ) -> Vec<Vmmc> {
        let seeds: Vec<u64> = (0..states.len()).map(|_| rng.random()).collect();
        states
            .par_iter()
            .enumerate()
            .map(|(idx, s)| {
                let thread_seed = seeds[idx];
                let mut thread_rng = Prng::seed_from_u64(thread_seed);
                let p_str = format!("./{output_dir}/{:0>3}", idx);

                record_child_config(&p_str, s);
                let (proto, child) = self
                    .step_one(s, &mut thread_rng)
                    .expect("Simulation failed");
                record_child(&p_str, &child, proto);
                child
            })
            .collect()
    }

    fn get_fitnesses(&mut self, children: &[Vmmc], rng: &mut Prng) -> Vec<f64> {
        let fitnesses: Vec<f64> = children
            .iter()
            .map(|c| self.fitness_func.eval(c, rng))
            .collect();
        self.fitnesses.extend_from_slice(&fitnesses);
        let avg_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        log::info!(
            "Children executed: fitnesses = {:?} avg = {avg_fitness}",
            fitnesses,
        );
        fitnesses
    }

    fn get_polygons(&mut self, children: &[Vmmc]) -> Vec<usize> {
        let polygons: Vec<usize> = children.iter().map(|c| calc_polygon_count(c, 12)).collect();
        let avg_polygons = polygons.iter().sum::<usize>() / polygons.len();
        log::info!(
            "Children executed: polygons = {:?} avg = {avg_polygons}",
            polygons,
        );
        polygons
    }

    fn record_genepool(&self, genepool: &GenePool) {
        let ids: Vec<(usize, f64)> = genepool.iter().map(|(c, fit, _)| (c.id(), *fit)).collect();
        let polygons: Vec<(usize, usize)> = genepool.iter().map(|(c, _, p)| (c.id(), *p)).collect();
        log::info!("Updated genepool: {:?}", ids);
        log::info!("Updated genepool polygons: {:?}\n", polygons);
    }

    pub fn step_all_and_save(&mut self, output_dir: &str, rng: &mut Prng) {
        log::info!(
            "Creating threadpool of {} workers\n",
            self.generation_size()
        );

        // Create initial generation
        let mut candidates = self.initial_candidates();
        let mut genepool: Vec<(Dna, f64, usize)> = candidates[0..self.survivors_per_generation]
            .iter()
            .map(|c| (c.clone(), 0., 0))
            .collect();

        for gen_idx in 0..self.num_generations {
            log::info!("Starting generation {gen_idx}: ");
            let ids: Vec<usize> = candidates.iter().map(|c| c.id()).collect();
            log::info!("Candidates: {:?}", ids);

            // 1.) Execute a generations worth of sims
            let start = Instant::now();
            let gen_dir = format!("./{output_dir}/{:0>3}", gen_idx);
            let children = self.step_generation_to(&candidates, rng, &gen_dir);

            let generation_end = Instant::now();
            log::info!("Generation execution time: {:?}", generation_end - start);
            // record_children(output_dir, &candidates, &children, );

            let fitnesses = self.get_fitnesses(&children, rng);
            let polygons = self.get_polygons(&children);

            // 2.) Prune survivors based on fitness function
            self.prune(&mut genepool, &candidates, fitnesses, polygons);
            self.record_genepool(&genepool);
            // self.record_genepool_polygons(&polygons);

            // 3.) Use Mutation function to get back to normal number of sims
            candidates = self.spawn_children(&genepool, self.children_per_survivor);
            // let end = Instant::now();
            // log::info!("Time to prep next generation: {:?}", end - generation_end);
        }
    }

    // generate children from survivors of previously generations
    pub fn spawn_children(&mut self, genepool: &GenePool, num_children: usize) -> Vec<Dna> {
        let mut children = Vec::new();
        for (dna, _, _) in genepool.iter() {
            for _ in 0..num_children {
                // record parent-child relationship
                self.history.push((dna.id(), self.child_ctr));
                // create new child
                let mut child_dna = dna.clone();
                self.mutate(&mut child_dna);
                children.push(child_dna);
            }
        }

        children
    }
}
