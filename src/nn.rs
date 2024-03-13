/// Implement nueral net implementation from original paper
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

// architecture = 1 input, 1000 hidden layers of 1 node each, and 2 outputs (mu and epsilon)
#[derive(Clone, Serialize, Deserialize, Debug)]
struct HiddenLayer {
    input_weight: f64,
    epsilon_weight: f64,
    mu_weight: f64,
    bias: f64,
    val: f64,
}

impl HiddenLayer {
    fn new(input_weight: f64, epsilon_weight: f64, mu_weight: f64, bias: f64, val: f64) -> Self {
        Self {
            input_weight,
            epsilon_weight,
            mu_weight,
            bias,
            val,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct NueralNet {
    layers: Vec<HiddenLayer>,
}

impl NueralNet {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut layers = Vec::new();

        for _ in 0..1000 {
            let iw = normal.sample(&mut rng);
            let epsilon = normal.sample(&mut rng);
            let mu = normal.sample(&mut rng);
            let bias = normal.sample(&mut rng);
            let val = normal.sample(&mut rng);
            let layer = HiddenLayer::new(iw, epsilon, mu, bias, val);
            layers.push(layer);
        }

        NueralNet { layers }
    }

    // t is in [0; 1.0]. Represents current time / total time of simulation
    pub fn exec(&mut self, t: f64) -> (f64, f64) {
        self.layers
            .iter_mut()
            .for_each(|l| l.val = t * l.input_weight + l.bias);
        let epsilon = self
            .layers
            .iter()
            .map(|l| l.val * l.epsilon_weight)
            .sum::<f64>()
            / self.layers.len() as f64;
        let mu =
            self.layers.iter().map(|l| l.val * l.mu_weight).sum::<f64>() / self.layers.len() as f64;
        (epsilon, mu)
    }
}
