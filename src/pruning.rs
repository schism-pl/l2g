use crate::fitness::FitnessFunc;
use vmmc::vmmc::Vmmc;

fn get_index_of_least_fit(fitness: &FitnessFunc, values: &[(usize, Vmmc)]) -> usize {
    let mut lowest = f64::MIN;
    let mut index = 0;
    for (idx, v) in values.iter().enumerate() {
        if fitness.eval(&v.1) < lowest {
            lowest = fitness.eval(&v.1);
            index = idx;
        }
    }
    index
}

// Note: this function is implemented assuming that computing fitness is cheap
// pretty easy to optimize if that isn't the case
pub fn prune(vmmcs: Vec<Vmmc>, prune_to: usize, fitness: &FitnessFunc) -> Vec<(usize, Vmmc)> {
    // let n = self.params.survivors_per_generation();
    let mut survivors = Vec::new();

    for (idx, vmmc) in vmmcs.into_iter().enumerate() {
        // the first n children get to start as survivors
        if idx < prune_to {
            survivors.push((idx, vmmc));
            continue;
        }
        // attempt to replace one of the survivors
        let fit = fitness.eval(&vmmc);
        let index = get_index_of_least_fit(&fitness, &survivors);
        if fit > fitness.eval(&survivors[index].1) {
            survivors[index] = (idx, vmmc);
        }
    }
    survivors
}
