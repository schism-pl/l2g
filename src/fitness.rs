use vmmc::vmmc::Vmmc;
use vmmc::polygons::calc_polygon_count;

pub enum FitnessFunc {
    Random,
    // NumBonds,
    PolygonSum,
}

impl FitnessFunc {
    /// returns a number between 0.0 and 1.0
    pub fn eval(&self, vmmc: &Vmmc) -> f64 {
        match self {
            // FitnessFunc::NumBonds => 1.0 - 1.0 / calc_bond_distribution(vmmc).iter().sum::<usize>() as f64,
            FitnessFunc::Random => 0.5, // we don't differentiate between different simulations
            FitnessFunc::PolygonSum => 1.0 - 1.0 / calc_polygon_count(vmmc, 12) as f64, 
        }
    }
}

