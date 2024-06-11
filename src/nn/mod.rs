pub mod l2g_nn;
pub mod new_nn;

use crate::nn::l2g_nn::NnConfig;
use new_nn::FLLConfig;
use serde::{Deserialize, Serialize};
use vmmc::protocol::{Peekable, ProtocolStep, SynthesisProtocol};

use l2g_nn::{NnMegastepIter, NueralNet};

#[derive(Clone, Serialize, Deserialize)]
pub enum Dna {
    TimeParticleNet(NnConfig, SynthesisProtocol),
    TimeNet(NnConfig, SynthesisProtocol),
    Fll(FLLConfig, SynthesisProtocol),
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
            Self::TimeNet(nn, ..) | Dna::TimeParticleNet(nn, ..) => nn,
            Dna::Fll(nn, proto) => unimplemented!(),
        }
    }

    pub fn protocol_iter(&self) -> StaticMegastepIter {
        let proto_vec = match self {
            Self::TimeNet(nn, proto) => nn.proto_vec(proto),
            Self::TimeParticleNet(_nn, _proto) => unimplemented!(),
            Self::Fll(nn, proto) => nn.proto_vec(proto),
        };

        StaticMegastepIter {
            inner: proto_vec,
            t: 0,
        }
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

impl<'a> Peekable for StaticMegastepIter {
    type Output = ProtocolStep;

    fn peek(&self) -> Self::Output {
        self.inner[self.t].clone()
    }
}
