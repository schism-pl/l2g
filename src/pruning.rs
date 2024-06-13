use crate::nn::Dna;

fn get_index_of_least_fit(fitnesses: &[f64]) -> usize {
    let mut lowest = f64::MIN;
    let mut index = 0;
    for (idx, v) in fitnesses.iter().enumerate() {
        if v < &lowest {
            lowest = *v;
            index = idx;
        }
    }
    index
}

// Note: this function is implemented assuming that computing fitness is cheap
// pretty easy to optimize if that isn't the case
pub fn prune(candidates: &[Dna], fitnesses: Vec<f64>, prune_to: usize) -> Vec<Dna> {
    let mut survivor_fitnesses = Vec::new();
    let mut survivors = Vec::new();

    for (idx, dna) in candidates.iter().enumerate() {
        let fitness = fitnesses[idx];
        // the first n children get to start as survivors
        if idx < prune_to {
            survivor_fitnesses.push(fitness);
            survivors.push(dna.clone());
            continue;
        }
        // attempt to replace one of the survivors
        // let fit = fitness.eval(&vmmc);
        let index = get_index_of_least_fit(&survivor_fitnesses);
        if fitness > fitnesses[index] {
            survivor_fitnesses[index] = fitness;
            survivors[index] = dna.clone();
        }
    }
    survivors
}
