pub mod fll;
pub mod microstate;
pub mod timenet;

use crate::nn::timenet::TimeNetConfig;
use fll::FLLConfig;
use microstate::MicrostateConfig;
use serde::{Deserialize, Serialize};
use vmmc::protocol::{ProtocolIter, SynthesisProtocol};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningStrategy {
    Timenet,
    Fll,
    MicroState,
}

#[derive(Clone, Serialize, Deserialize)]
enum DnaInner {
    TimeNet(TimeNetConfig, SynthesisProtocol),
    Fll(FLLConfig, SynthesisProtocol),
    MicroState(MicrostateConfig, SynthesisProtocol),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Dna {
    id: usize,
    inner: DnaInner,
}

impl Dna {
    fn new(id: usize, inner: DnaInner) -> Self {
        Self { id, inner }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    // TODO: combine all 3 options into a single parameterized model
    // 1. layer architecture
    // 2. smoothness constraint
    // 3. linear batching

    pub fn protocol_iter(&self) -> Box<dyn ProtocolIter> {
        use DnaInner::*;
        match &self.inner {
            TimeNet(nn, proto) => Box::new(nn.proto_iter(proto)),
            Fll(nn, proto) => Box::new(nn.proto_iter(proto)),
            MicroState(config, proto) => Box::new(config.proto_iter(proto)),
        }
    }

    pub fn type_str(&self) -> &str {
        use DnaInner::*;
        match self.inner {
            TimeNet(..) => "Time Network (Steve's code)",
            Fll(..) => "FLL (fixed-length linear)",
            MicroState(..) => "Microstate + Time Network (Steve's code)",
        }
    }

    pub fn fresh_time_net(nn_config: TimeNetConfig, proto: SynthesisProtocol) -> Self {
        Dna::new(0, DnaInner::TimeNet(nn_config, proto))
    }

    pub fn fresh_fll(config: FLLConfig, proto: SynthesisProtocol) -> Self {
        Dna::new(0, DnaInner::Fll(config, proto))
    }

    pub fn fresh_microstate(config: MicrostateConfig, proto: SynthesisProtocol) -> Self {
        Dna::new(0, DnaInner::MicroState(config, proto))
    }

    pub fn mutate(&mut self, new_id: usize) {
        use DnaInner::*;
        match &mut self.inner {
            TimeNet(nn, ..) => {
                nn.set_child_id(new_id as u32);
            }
            Fll(nn, ..) => nn.mutate(),
            MicroState(nn, ..) => nn.mutate(),
        }
        self.id = new_id;
    }
}
