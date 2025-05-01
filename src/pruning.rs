use crate::nn::Dna;

fn get_index_of_least_fit(genepool: &[(Dna, f64, usize)]) -> (usize, f64) {
    let mut lowest = genepool[0].1;
    let mut index = 0;
    for (idx, (_, v, _)) in genepool.iter().enumerate() {
        if v < &lowest {
            lowest = *v;
            index = idx;
        }
    }
    (index, lowest)
}

// Note: this function is implemented assuming that computing fitness is cheap
// pretty easy to optimize if that isn't the case
pub fn prune(
    genepool: &mut [(Dna, f64, usize)],
    candidates: &[Dna],
    fitnesses: Vec<f64>,
    polygons: Vec<usize>,
) {
    for (idx, dna) in candidates.iter().enumerate() {
        let fitness = fitnesses[idx];
        let polygon = polygons[idx];
        let (index, lowest_fit) = get_index_of_least_fit(genepool);
        if fitness > lowest_fit {
            genepool[index] = (dna.clone(), fitness, polygon)
        }
    }
}
