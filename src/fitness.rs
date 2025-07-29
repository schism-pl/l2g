use rand::Rng;
use serde::{Deserialize, Serialize};
use vmmc::particle::IsParticle;
use vmmc::polygons::calc_polygon_count;
use vmmc::polygons::calc_polygon_distribution;
use vmmc::vmmc::Vmmc;
use vmmc::Prng;

type BondOrderMatrix = ((f64, f64), (f64, f64));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FitnessFunc {
    Random,
    PolygonSum,
    ShapeDist,
    // ideal row-normalized matrix
    BondOrder(BondOrderMatrix), // Bond order for 2 vertex
}

impl FitnessFunc {
    pub fn eval(&self, vmmc: &Vmmc, rng: &mut Prng) -> f64 {
        match self {
            FitnessFunc::Random => rng.random(), // we don't differentiate between different simulations
            FitnessFunc::PolygonSum => calc_polygon_count(vmmc, 12) as f64,
            FitnessFunc::ShapeDist => {
                //we are assigning dist to be the values of the vector "polygon_dist" from calc_polygon_distribution
                let dist = calc_polygon_distribution(vmmc, 12);
                let squares = dist[3] as f64;
                squares
            }
            FitnessFunc::BondOrder(ideal_matrix) => {
                let actual_matrix = calculate_bond_order_matrix(vmmc);
                calculate_manhattan_distance(actual_matrix, *ideal_matrix)
            }
        }
    }
}

// Calculate manhattan distance between two 2x2 matrices
fn calculate_manhattan_distance(actual: BondOrderMatrix, ideal: BondOrderMatrix) -> f64 {
    (actual.0 .0 - ideal.0 .0).abs()
        + (actual.0 .1 - ideal.0 .1).abs()
        + (actual.1 .0 - ideal.1 .0).abs()
        + (actual.1 .1 - ideal.1 .1).abs()
}

// Calculate row-normalized 2x2 matrix of bond orders for 2 vertex system
fn calculate_bond_order_matrix(vmmc: &Vmmc) -> BondOrderMatrix {
    let mut counts = [[0; 2]; 2];

    // Count interactions between particles
    for particle in vmmc.particles().iter() {
        let row = particle.shape_id() as usize;
        for neighbor_id in vmmc.determine_interactions(particle) {
            let neighbor = vmmc.particle(neighbor_id);
            let col = neighbor.shape_id() as usize;
            counts[row][col] += 1;
        }
    }

    // Normalize each row to get row-normalized matrix
    let row_sums: [usize; 2] = [counts[0][0] + counts[0][1], counts[1][0] + counts[1][1]];

    let normalize = |count: usize, sum: usize| -> f64 {
        if sum > 0 {
            count as f64 / sum as f64
        } else {
            0.0
        }
    };

    let p1_p1 = normalize(counts[0][0], row_sums[0]);
    let p1_p2 = normalize(counts[0][1], row_sums[0]);
    let p2_p1 = normalize(counts[1][0], row_sums[1]);
    let p2_p2 = normalize(counts[1][1], row_sums[1]);

    ((p1_p1, p1_p2), (p2_p1, p2_p2))
}
