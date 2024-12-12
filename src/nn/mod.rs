pub mod fll;
pub mod timenet;
pub mod microstate;

use crate::nn::timenet::TimeNetConfig;
use fll::FLLConfig;
use microstate::MicrostateConfig;
use serde::{Deserialize, Serialize};
use vmmc::protocol::{Peekable, ProtocolStep, SynthesisProtocol};


#[derive(Clone, Serialize, Deserialize)]
enum DnaInner {
    TimeNet(TimeNetConfig, SynthesisProtocol),
    Fll(FLLConfig, SynthesisProtocol),
    MicroState(MicrostateConfig, SynthesisProtocol)
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

    // TODO: needs the protocol to have access to the vmmc    
    pub fn protocol_iter(&self) -> StaticMegastepIter {
        use DnaInner::*;
        let proto_vec = match &self.inner {
            TimeNet(nn, proto) => nn.proto_vec(proto),
            Fll(nn, proto) => nn.proto_vec(proto),
            MicroState(_config, _proto) => unimplemented!(), 
        };

        StaticMegastepIter {
            inner: proto_vec,
            t: 0,
        }
    }

    pub fn type_str(&self) -> &str {
        use DnaInner::*;
        match self.inner {
            TimeNet(..) => "Time Network (Steve's code)",
            Fll(..) => "FLL (fixed-length linear)",
            MicroState(..) => "Microstate + Time Network (Steve's code)"
        }
    }

    pub fn fresh_time_net(nn_config: TimeNetConfig, proto: SynthesisProtocol) -> Self {
        Dna::new(0, DnaInner::TimeNet(nn_config, proto))
    }

    pub fn fresh_fll(config: FLLConfig, proto: SynthesisProtocol) -> Self {
        Dna::new(0, DnaInner::Fll(config, proto))
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

pub struct StaticMegastepIter {
    inner: Vec<ProtocolStep>,
    t: usize,
}

impl Iterator for StaticMegastepIter {
    type Item = ProtocolStep;

    fn next(&mut self) -> Option<Self::Item> {
        if self.t >= self.inner.len() {
            return None;
        }
        let r = self.inner[self.t].clone();
        self.t += 1;
        Some(r)
    }
}

impl ExactSizeIterator for StaticMegastepIter {
    // We can easily calculate the remaining number of iterations.
    fn len(&self) -> usize {
        self.inner.len() - self.t
    }
}

impl Peekable for StaticMegastepIter {
    type Output = ProtocolStep;

    fn peek(&self) -> Self::Output {
        self.inner[self.t].clone()
    }
}
