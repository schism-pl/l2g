use serde::{Deserialize, Serialize};
use vmmc::protocol::SynthesisProtocol;

use crate::nn::NnConfig;

#[derive(Clone, Serialize, Deserialize)]
pub enum Dna {
    TimeParticleNet(NnConfig, SynthesisProtocol),
    TimeNet(NnConfig, SynthesisProtocol),
}

// TODO: current_protocol?
impl Dna {
    pub fn mutate(&mut self) {
        match self {
            Self::TimeParticleNet(nn, ..) => {
                nn.increment_child_id();
            }
            Self::TimeNet(nn, ..) => {
                nn.increment_child_id();
            }
        }
    }
}
