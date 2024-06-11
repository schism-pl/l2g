pub mod l2g_nn;
pub mod new_nn;

use crate::nn::l2g_nn::NnConfig;
use serde::{Deserialize, Serialize};
use vmmc::protocol::SynthesisProtocol;

use l2g_nn::{NnMegastepIter, NueralNet};

#[derive(Clone, Serialize, Deserialize)]
pub enum Dna {
    TimeParticleNet(NnConfig, SynthesisProtocol),
    TimeNet(NnConfig, SynthesisProtocol),
    Fll(NnConfig, SynthesisProtocol),
}

impl Dna {
    pub fn type_str(&self) -> &str {
        match self {
            Self::TimeNet(..) => "Time Network",
            Self::TimeParticleNet(..) => "Time + Particle Network",
            Self::Fll(..) => "FLL (fixed-length linear)",
        }
    }

    pub fn nn_config(&self) -> &NnConfig {
        match self {
            Self::TimeNet(nn, ..) | Dna::TimeParticleNet(nn, ..) | Dna::Fll(nn, ..) => nn,
        }
    }

    // TODO: expand
    pub fn protocol_iter(&self) -> NnMegastepIter {
        match self {
            Self::TimeNet(nn, proto) | Dna::TimeParticleNet(nn, proto) | Dna::Fll(nn, proto) => {
                NueralNet::from_config(nn).current_protocol(proto)
            }
        }
    }
}
