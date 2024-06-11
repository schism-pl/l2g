pub mod fll;
pub mod l2g_nn;

use crate::nn::l2g_nn::NnConfig;
use fll::FLLConfig;
use serde::{Deserialize, Serialize};
use vmmc::protocol::{Peekable, ProtocolStep, SynthesisProtocol};

#[derive(Clone, Serialize, Deserialize)]
enum DnaInner {
    TimeParticleNet(NnConfig, SynthesisProtocol),
    TimeNet(NnConfig, SynthesisProtocol),
    Fll(FLLConfig, SynthesisProtocol),
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

    pub fn protocol_iter(&self) -> StaticMegastepIter {
        use DnaInner::*;
        let proto_vec = match &self.inner {
            TimeNet(nn, proto) => nn.proto_vec(proto),
            TimeParticleNet(_nn, _proto) => unimplemented!(),
            Fll(nn, proto) => nn.proto_vec(proto),
        };

        StaticMegastepIter {
            inner: proto_vec,
            t: 0,
        }
    }

    pub fn type_str(&self) -> &str {
        use DnaInner::*;
        match self.inner {
            TimeNet(..) => "Time Network",
            TimeParticleNet(..) => "Time + Particle Network",
            Fll(..) => "FLL (fixed-length linear)",
        }
    }

    pub fn fresh_time_net(nn_config: NnConfig, proto: SynthesisProtocol) -> Self {
        Dna::new(0, DnaInner::TimeNet(nn_config, proto))
    }

    pub fn fresh_fll(config: FLLConfig, proto: SynthesisProtocol) -> Self {
        Dna::new(0, DnaInner::Fll(config, proto))
    }

    pub fn mutate(&mut self, new_id: usize) {
        use DnaInner::*;
        match &mut self.inner {
            TimeParticleNet(nn, ..) | TimeNet(nn, ..) => {
                nn.set_child_id(new_id as u32);
            }
            Fll(nn, ..) => nn.mutate(),
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
