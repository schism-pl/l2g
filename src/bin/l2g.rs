use clap::Parser;
use l2g::config::L2GInputParams;
use l2g::evo_vmmc::{self, EvoVmmc};
use l2g::fitness::FitnessFunc;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use vmmc::cli::VmmcConfig;
use std::fs;

// correctness criteria:
// 1. average energy monotonically increases (decreases?)
// 2. particles visibly stick together in visualization
// 3. values match other impls (approximately)

// TODO: builder pattern
fn main() -> anyhow::Result<()> {
    env_logger::init();
    // Get commandline arguments
    let config = VmmcConfig::parse();

    let l2gip = if config.input() != "" {
        let contents = fs::read_to_string(config.input())?;
        toml::from_str(&contents)?
    } else {
        L2GInputParams::default()
    };

    // Get default params
    // Seed the rng
    let seed = l2gip.ip().seed;
    println!("Using seed = {:x?}", seed);
    let mut rng = SmallRng::seed_from_u64(seed as u64);

    let mut evo_vmmc = EvoVmmc::new(l2gip);
    evo_vmmc.step_all(&mut rng);

    Ok(())
}
