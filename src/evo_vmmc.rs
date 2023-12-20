use rand::rngs::SmallRng;
use vmmc::{run_vmmc, vmmc::Vmmc, vmmc_from_config, InputParams};

pub struct L2GInputParams {
    ip: InputParams,

    // L2G parameters
    num_generations: usize,
    children_per_generation: usize,
    survivors_per_generation: usize, // # of children post-pruning
}

impl L2GInputParams {
    pub fn ip(&self) -> &InputParams {
        &self.ip
    }
}

impl Default for L2GInputParams {
    fn default() -> Self {
        // TODO: build a toml file for steve' stuff and populate it with this
        // TODO: read the toml here (this shouldn't be default)
        let ip = InputParams::default();

        let num_generations = 3;
        let children_per_generation = 3;
        let survivors_per_generation = 1;

        Self {
            ip,
            num_generations,
            children_per_generation,
            survivors_per_generation,
        }
    }
}

pub enum FitnessFunc {
    Random,
    AvgEnergy,
    OverlappingPolygonSum,
}

// pub enum MutationFunc {
//     NumParticles,
//     Null,
// }

pub fn run_fresh_vmmc(ip: InputParams, rng: &mut SmallRng) -> Vmmc {
    let mut vmmc = vmmc_from_config(&ip, rng);
    run_vmmc(&mut vmmc, ip.protocol, None, rng);
    vmmc
}

pub struct EvoVmmc {
    fitness_func: FitnessFunc,
    params: L2GInputParams,
    // Target Tiling
}

impl EvoVmmc {
    pub fn new(fitness_func: FitnessFunc) -> Self {
        let params = L2GInputParams::default();
        Self {
            fitness_func,
            params,
        }
    }

    /// returns a number between 0.0 and 1.0
    pub fn fitness(&self, vmmc: &Vmmc) -> f64 {
        match self.fitness_func {
            FitnessFunc::AvgEnergy => 1.0 - 1.0 / vmmc.get_average_energy().abs(),
            FitnessFunc::Random => 0.5, // we don't differentiate between
            FitnessFunc::OverlappingPolygonSum => panic!("Unimplemented"), //1.0 - 1.0 / overlapping_polygon_sum(vmmc),
        }
    }

    fn step_generation(&mut self, rng: &mut SmallRng) -> Vec<Vmmc> {
        let mut children = Vec::new();
        for jdx in 0..self.params.children_per_generation {
            // let r_vmmc = self.run_child(self.params.ip.clone(), rng);
            let r_vmmc = run_fresh_vmmc(self.params.ip.clone(), rng);
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
            let survivors = self.prune(children);
            let survivor_ids: Vec<usize> = survivors.iter().map(|v| v.0).collect();
            println!("{:?} survive", survivor_ids);
        }
    }

    pub fn step_all(&mut self, rng: &mut SmallRng) {
        self.step_generation_n(self.params.num_generations, rng)
    }

    fn get_index_of_least_fit(&self, values: &[(usize, Vmmc)]) -> usize {
        let mut lowest = f64::MIN;
        let mut index = 0;
        for (idx, v) in values.iter().enumerate() {
            if self.fitness(&v.1) < lowest {
                lowest = self.fitness(&v.1);
                index = idx;
            }
        }
        index
    }

    // generate children from survivors of previously generations
    pub fn spawn_children(&self, _vmmcs: Vec<Vmmc>) -> Vec<Vmmc> {
        panic!("TODO")
    }

    // Note: this function is implemented assuming that computing fitness is cheap
    // pretty easy to optimize if that isn't the case
    pub fn prune(&self, vmmcs: Vec<Vmmc>) -> Vec<(usize, Vmmc)> {
        let n = self.params.survivors_per_generation;
        let mut survivors = Vec::new();

        for (idx, vmmc) in vmmcs.into_iter().enumerate() {
            // the first n children get to start as survivors
            if idx < n {
                survivors.push((idx, vmmc));
                continue;
            }
            // attempt to replace one of the survivors
            let fitness = self.fitness(&vmmc);
            let index = self.get_index_of_least_fit(&survivors);
            if fitness > self.fitness(&survivors[index].1) {
                survivors[index] = (idx, vmmc);
            }
        }
        survivors
    }
}
