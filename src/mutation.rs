use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use vmmc::protocol::SynthesisProtocol;

use crate::nn::NnConfig;

#[derive(Clone, Serialize, Deserialize)]
pub enum Mutator {
    TimeParticleNet(NnConfig, SynthesisProtocol), // terms, order, percentage of range that any one coefficient can contribute
    TimeNet(NnConfig, SynthesisProtocol),         // argument 2 = # of steps for each protocol
}

// TODO: current_protocol?
impl Mutator {
    pub fn mutate(&mut self, rng: &mut SmallRng) {
        use Mutator::*;
        match self {
            TimeParticleNet(nn, ..) => {
                nn.increment_child_id();
            }
            TimeNet(nn, ..) => {
                nn.increment_child_id();
            }
        }
    }
}
