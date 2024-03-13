use vmmc::InputParams;

pub struct L2GInputParams {
    ip: InputParams,

    // L2G parameters
    num_generations: usize,
    children_per_generation: usize,
    survivors_per_generation: usize, // # of children post-pruning
}

impl L2GInputParams {
    pub fn ip(&self) -> &InputParams {
        &self.ip
    }

    pub fn num_generations(&self) -> usize {
        self.num_generations
    }

    pub fn children_per_generation(&self) -> usize {
        self.children_per_generation
    }

    pub fn survivors_per_generation(&self) -> usize {
        self.survivors_per_generation
    }



}

impl Default for L2GInputParams {
    fn default() -> Self {
        // TODO: build a toml file for steve' stuff and populate it with this
        // TODO: read the toml here (this shouldn't be default)
        let ip = InputParams::default();

        let num_generations = 3;
        let children_per_generation = 3;
        let survivors_per_generation = 1;

        Self {
            ip,
            num_generations,
            children_per_generation,
            survivors_per_generation,
        }
    }
}