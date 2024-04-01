use serde::{Deserialize, Serialize};
use vmmc::polygons::calc_polygon_count;
use vmmc::vmmc::Vmmc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FitnessFunc {
    Random,
    PolygonSum,
}

impl FitnessFunc {
    pub fn eval(&self, vmmc: &Vmmc) -> f64 {
        match self {
            FitnessFunc::Random => 0.5, // we don't differentiate between different simulations
            FitnessFunc::PolygonSum => calc_polygon_count(vmmc, 12) as f64,
        }
    }
}
