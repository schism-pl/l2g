use crate::nn::Dna;

fn get_index_of_least_fit(genepool: &[(Dna, f64)]) -> (usize, f64) {
    let mut lowest = genepool[0].1;
    let mut index = 0;
    for (idx, (_, v)) in genepool.iter().enumerate() {
        if v < &lowest {
            lowest = *v;
            index = idx;
        }
    }
    (index, lowest)
}

// Note: this function is implemented assuming that computing fitness is cheap
// pretty easy to optimize if that isn't the case
pub fn prune(genepool: &mut [(Dna, f64)], candidates: &[Dna], fitnesses: Vec<f64>) {
    for (idx, dna) in candidates.iter().enumerate() {
        let fitness = fitnesses[idx];
        let (index, lowest_fit) = get_index_of_least_fit(genepool);
        // println!("Assessing: {:?} {fitness} vs {:?} {:?}", dna.id(), genepool[index].0.id(), lowest_fit);
        if fitness > lowest_fit {
            genepool[index] = (dna.clone(), fitness)
        }
    }
}
