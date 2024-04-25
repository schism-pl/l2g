use serde::{Deserialize, Serialize};
use vmmc::protocol::SynthesisProtocol;

use crate::nn::NnConfig;

#[derive(Clone, Serialize, Deserialize)]
pub enum Dna {
    TimeParticleNet(NnConfig, SynthesisProtocol),
    TimeNet(NnConfig, SynthesisProtocol),
}

impl Dna {
    pub fn type_str(&self) -> &str {
        match self {
            Self::TimeNet(..) => "Time Network",
            Self::TimeParticleNet(..) => "Time + Particle Network",
        }
    }

    pub fn nn_config(&self) -> &NnConfig {
        match self {
            Self::TimeNet(nn, ..) | Dna::TimeParticleNet(nn, ..) => nn,
        }
    }
}
