use std::time::Instant;

use crate::fitness::FitnessFunc;
use crate::io::record_children;
use crate::nn::Dna;
use crate::pruning::prune;
use crate::run_fresh_vmmc;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use vmmc::vmmc::Vmmc;
use vmmc::SimParams;

#[derive(Clone, Serialize, Deserialize)]
pub struct EvoEngine {
    pub seed: u32,
    pub sim_params: SimParams,
    pub init_dna: Dna,

    pub fitness_func: FitnessFunc,

    pub num_generations: usize,
    pub survivors_per_generation: usize,
    pub children_per_survivor: usize,

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

    fn prune(&self, genepool: &mut [(Dna, f64)], candidates: &[Dna], fitnesses: Vec<f64>) {
        prune(
            genepool, candidates, fitnesses,
            // &self.fitness_func,
        );
    }

    fn mutate(&mut self, dna: &mut Dna) {
        // use DnaInner::*;
        dna.mutate(self.child_ctr);
        self.child_ctr += 1;
    }

    pub fn initial_candidates(&mut self) -> Vec<Dna> {
        self.spawn_children(&vec![(self.init_dna.clone(), 0.)], self.generation_size())
    }

    fn step_one(&self, dna: &Dna, rng: &mut SmallRng) -> Vmmc {
        let protocol_iter = dna.protocol_iter();
        run_fresh_vmmc(self.sim_params(), protocol_iter, rng)
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

    fn get_fitnesses(&mut self, children: &[Vmmc]) -> Vec<f64> {
        let fitnesses: Vec<f64> = children.iter().map(|c| self.fitness_func.eval(c)).collect();
        self.fitnesses.extend_from_slice(&fitnesses);
        let avg_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        log::info!(
            "Children executed: fitnesses = {:?} avg = {avg_fitness}",
            fitnesses,
        );
        fitnesses
    }

    fn record_genepool(&self, genepool: &[(Dna, f64)]) {
        let ids: Vec<(usize, f64)> = genepool.iter().map(|(c, fit)| (c.id(), *fit)).collect();
        log::info!("Updated genepool: {:?}\n", ids);
    }

    pub fn step_all(&mut self, rng: &mut SmallRng) {
        log::info!(
            "Creating threadpool of {} workers\n",
            self.generation_size()
        );

        // Create initial generation
        let mut candidates = self.initial_candidates();
        let mut genepool: Vec<(Dna, f64)> = candidates[0..self.survivors_per_generation]
            .iter()
            .map(|c| (c.clone(), 0.))
            .collect();

        for gen_idx in 0..self.num_generations {
            log::info!("Starting generation {gen_idx}: ");
            let ids: Vec<usize> = candidates.iter().map(|c| c.id()).collect();
            log::info!("Candidates: {:?}", ids);

            // 1.) Execute a generations worth of sims
            let start = Instant::now();
            let children = self.step_generation(&candidates, rng);
            let generation_end = Instant::now();
            log::info!("Generation execution time: {:?}", generation_end - start);
            record_children(&candidates, &children, gen_idx);

            let fitnesses = self.get_fitnesses(&children);

            // 2.) Prune survivors based on fitness function
            self.prune(&mut genepool, &candidates, fitnesses);
            self.record_genepool(&genepool);

            // 3.) Use Mutation function to get back to normal number of sims
            candidates = self.spawn_children(&genepool, self.children_per_survivor);
            let end = Instant::now();
            log::info!("Time to prep next generation: {:?}", end - generation_end);
        }
    }

    // generate children from survivors of previously generations
    pub fn spawn_children(&mut self, genepool: &[(Dna, f64)], num_children: usize) -> Vec<Dna> {
        let mut children = Vec::new();
        for (dna, _) in genepool.iter() {
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
