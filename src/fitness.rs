use serde::{Deserialize, Serialize};
use vmmc::polygons::calc_polygon_count;
use vmmc::polygons::calc_polygon_distribution;
use vmmc::vmmc::Vmmc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FitnessFunc {
    Random,
    PolygonSum,
    ShapeDist,
}

impl FitnessFunc {
    pub fn eval(&self, vmmc: &Vmmc) -> f64 {
        match self {
            FitnessFunc::Random => 0.5, // we don't differentiate between different simulations
            FitnessFunc::PolygonSum => calc_polygon_count(vmmc, 12) as f64,
            FitnessFunc::ShapeDist => {
                let dist = calc_polygon_distribution(vmmc, 12); //we are assigning dist
                                                                //to be the values of
                                                                //the vector
                                                                //"polygon_dist" from
                                                                //calc_polygon_distribution
                let squares = dist[3] as f64;
                squares
            }
        }
    }
}
