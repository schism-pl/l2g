use rand::Rng;
use serde::{Deserialize, Serialize};
use vmmc::particle::IsParticle;
use vmmc::polygons::calc_polygon_count;
use vmmc::polygons::calc_polygon_distribution;
use vmmc::polygons::calc_unitcells;
use vmmc::tilings::tiling_from_str;
use vmmc::tilings::UnitCell;
use vmmc::vmmc::Vmmc;
use vmmc::Prng;

type BondOrderMatrix = ((u32, u32), (u32, u32));
type NormalizedBondOrderMatrix = ((f64, f64), (f64, f64));

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FitnessFunc {
    Random,
    PolygonSum,
    ShapeDist,
    // ideal matrix for 2-vertex bond orders
    BondOrder(BondOrderMatrix),
    // target unit cell structure
    Unitcell(String),
}

impl FitnessFunc {
    pub fn eval(&self, vmmc: &Vmmc, rng: &mut Prng) -> f64 {
        match self {
            FitnessFunc::Random => rng.random(), // we don't differentiate between different simulations
            FitnessFunc::PolygonSum => calc_polygon_count(vmmc, 12) as f64,
            FitnessFunc::ShapeDist => {
                //we are assigning dist to be the values of the vector "polygon_dist" from calc_polygon_distribution
                let dist = calc_polygon_distribution(vmmc, 12);
                dist[3] as f64 // number of squares
            }
            FitnessFunc::BondOrder(ideal_matrix) => {
                let actual_matrix = calculate_bond_order_matrix(vmmc);
                let normalized_ideal_matrix = normalize_bond_order_matrix(*ideal_matrix);
                let dist = calculate_manhattan_distance(actual_matrix, normalized_ideal_matrix); // max 2
                if dist.is_nan() {
                    // This means there are no particles in the system, and we return worst fitness
                    0.0
                } else if dist == 0.0 {
                    2.0
                } else {
                    2.0 - dist // 2.0 is the max distance, so we subtract it from 2.0 so smaller dist = better fitness
                }
            }
            FitnessFunc::Unitcell(tiling_string) => {
                // Convert string to UnitCell and count instances
                let unitcell = tiling_from_str(tiling_string).expect("Invalid tiling string");
                let calced_unitcells = calc_unitcells(vmmc, 12, &unitcell);
                calced_unitcells.len() as f64
            }
        }
    }
}

fn normalize_bond_order_matrix(matrix: BondOrderMatrix) -> NormalizedBondOrderMatrix {
    let sum = matrix.0 .0 + matrix.0 .1 + matrix.1 .0 + matrix.1 .1;
    (
        (
            matrix.0 .0 as f64 / sum as f64,
            matrix.0 .1 as f64 / sum as f64,
        ),
        (
            matrix.1 .0 as f64 / sum as f64,
            matrix.1 .1 as f64 / sum as f64,
        ),
    )
}

// Calculate manhattan distance between two 2x2 matrices
fn calculate_manhattan_distance(
    actual: NormalizedBondOrderMatrix,
    ideal: NormalizedBondOrderMatrix,
) -> f64 {
    (actual.0 .0 - ideal.0 .0).abs()
        + (actual.0 .1 - ideal.0 .1).abs()
        + (actual.1 .0 - ideal.1 .0).abs()
        + (actual.1 .1 - ideal.1 .1).abs()
}

// Calculate normalized 2x2 matrix of bond orders for 2 vertex system
fn calculate_bond_order_matrix(vmmc: &Vmmc) -> NormalizedBondOrderMatrix {
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

    // Convert to BondOrderMatrix format and normalize the whole matrix
    let bond_order_matrix = (
        (counts[0][0] as u32, counts[0][1] as u32),
        (counts[1][0] as u32, counts[1][1] as u32),
    );

    normalize_bond_order_matrix(bond_order_matrix)
}
