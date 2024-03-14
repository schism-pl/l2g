use serde::{Deserialize, Serialize};
use vmmc::InputParams;

use crate::nn::NueralNet;

#[derive(Clone, Serialize, Deserialize)]
pub enum MutationFunc {
    UniformRandom(f64), // magnitude
    LearningToGrowClassic(NueralNet),
}

impl MutationFunc {
    pub fn mutate(&self, ip: &InputParams) -> InputParams {
        use MutationFunc::*;
        match self {
            UniformRandom(mag) => unimplemented!(),
            LearningToGrowClassic(nn) => unimplemented!(),
        }
    }
}
