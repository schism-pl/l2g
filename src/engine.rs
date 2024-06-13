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

    fn prune(&self, candidates: &[Dna], fitnesses: Vec<f64>) -> Vec<Dna> {
        prune(
            candidates,
            fitnesses,
            self.survivors_per_generation(),
            // &self.fitness_func,
        )
    }

    fn mutate(&mut self, dna: &mut Dna) {
        // use DnaInner::*;
        dna.mutate(self.child_ctr);
        self.child_ctr += 1;
    }

    pub fn initial_candidates(&mut self) -> Vec<Dna> {
        self.spawn_children(vec![self.init_dna.clone()], self.generation_size())
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

    fn get_fitnesses(&self, children: &[Vmmc]) -> Vec<f64> {
        let fitnesses: Vec<f64> = children.iter().map(|c| self.fitness_func.eval(c)).collect();
        let avg_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        log::info!(
            "Children executed: fitnesses = {:?} avg = {avg_fitness}",
            fitnesses,
        );
        fitnesses
    }

    fn record_survivors(&self, survivors: &[Dna]) {
        let ids: Vec<usize> = survivors.iter().map(|c| c.id()).collect();
        log::info!("Pruned survivors to: {:?}\n", ids);
    }

    pub fn step_all(&mut self, rng: &mut SmallRng) {
        log::info!(
            "Creating threadpool of {} workers\n",
            self.generation_size()
        );

        // Create initial generation
        let mut candidates = self.initial_candidates();
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
            // TODO: to make this a proper hill climbing algorithm, we just need to concatenate hall of fame results
            let survivors = self.prune(&candidates, fitnesses);
            self.record_survivors(&survivors);

            // 3.) Use Mutation function to get back to normal number of sims
            candidates = self.spawn_children(survivors, self.children_per_survivor);
            let end = Instant::now();
            log::info!("Time to prep next generation: {:?}", end - generation_end);
        }
    }

    // generate children from survivors of previously generations
    pub fn spawn_children(&mut self, survivors: Vec<Dna>, num_children: usize) -> Vec<Dna> {
        let mut children = Vec::new();
        for mut s in survivors.into_iter() {
            for _ in 0..num_children {
                // record parent-child relationship
                self.history.push((s.id(), self.child_ctr));
                // create new child
                self.mutate(&mut s);
                children.push(s.clone());
            }
        }

        children
    }
}
