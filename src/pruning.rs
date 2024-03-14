use crate::fitness::FitnessFunc;
use vmmc::{vmmc::Vmmc, InputParams};

fn get_index_of_least_fit(fitness: &FitnessFunc, values: &[Vmmc]) -> usize {
    let mut lowest = f64::MIN;
    let mut index = 0;
    for (idx, v) in values.iter().enumerate() {
        if fitness.eval(&v) < lowest {
            lowest = fitness.eval(&v);
            index = idx;
        }
    }
    index
}

// Note: this function is implemented assuming that computing fitness is cheap
// pretty easy to optimize if that isn't the case
pub fn prune(
    ips: &[InputParams],
    vmmcs: Vec<Vmmc>,
    prune_to: usize,
    fitness: &FitnessFunc,
) -> Vec<InputParams> {
    // let n = self.params.survivors_per_generation();
    let mut survivors = Vec::new();
    let mut survivor_ips = Vec::new();

    for (idx, vmmc) in vmmcs.into_iter().enumerate() {
        let ip = ips[idx].clone();
        // the first n children get to start as survivors
        if idx < prune_to {
            survivors.push(vmmc);
            survivor_ips.push(ip);
            continue;
        }
        // attempt to replace one of the survivors
        let fit = fitness.eval(&vmmc);
        let index = get_index_of_least_fit(fitness, &survivors);
        if fit > fitness.eval(&survivors[index]) {
            survivors[index] = vmmc;
            survivor_ips[index] = ip;
        }
    }
    survivor_ips
}
