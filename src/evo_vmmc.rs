use anyhow::Result;
use rand::rngs::SmallRng;
use vmmc::{run_vmmc, vmmc::Vmmc, vmmc_from_config, InputParams};
use crate::config::L2GInputParams;
use crate::fitness::FitnessFunc;
use crate::pruning::prune;


// pub enum MutationFunc {
//     NumParticles,
//     Null,
// }

pub fn run_fresh_vmmc(ip: InputParams, rng: &mut SmallRng) -> Vmmc {
    let mut vmmc = vmmc_from_config(&ip, rng);
    let _: Result<()> = run_vmmc(&mut vmmc, ip.protocol, vmmc::no_callback(), rng);
    vmmc
}

pub struct EvoVmmc {
    fitness_func: FitnessFunc,
    params: L2GInputParams,
}

impl EvoVmmc {
    pub fn new(fitness_func: FitnessFunc) -> Self {
        let params = L2GInputParams::default();
        Self {
            fitness_func,
            params,
        }
    }

    fn step_generation(&mut self, rng: &mut SmallRng) -> Vec<Vmmc> {
        let mut children = Vec::new();
        for jdx in 0..self.params.children_per_generation() {
            // let r_vmmc = self.run_child(self.params.ip.clone(), rng);
            let r_vmmc = run_fresh_vmmc(self.params.ip().clone(), rng);
            let avg_energy = r_vmmc.get_average_energy();
            println!(
                "average energy of child-sim {:?} = {:?} kBT\n",
                jdx, avg_energy
            );
            children.push(r_vmmc);
        }
        children
    }

    // TODO: how to derive new generation from old?
    pub fn step_generation_n(&mut self, n: usize, rng: &mut SmallRng) {
        for idx in 0..n {
            println!("Starting generation {:?}", idx);
            let children = self.step_generation(rng);
            let survivors = prune(children, self.params.survivors_per_generation(), &self.fitness_func);
            let survivor_ids: Vec<usize> = survivors.iter().map(|v| v.0).collect();
            println!("{:?} survive", survivor_ids);
        }
    }

    pub fn step_all(&mut self, rng: &mut SmallRng) {
        self.step_generation_n(self.params.num_generations(), rng)
    }

    // generate children from survivors of previously generations
    pub fn spawn_children(&self, _vmmcs: Vec<Vmmc>) -> Vec<Vmmc> {
        panic!("TODO")
    }


}


