use clap::Parser;
use l2g::evo_vmmc::EvoVmmc;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fs;
use vmmc::cli::VmmcConfig;

// correctness criteria:
// 1. average energy monotonically increases (decreases?)
// 2. particles visibly stick together in visualization
// 3. values match other impls (approximately)

// TODO: builder pattern
fn main() -> anyhow::Result<()> {
    env_logger::init();
    // Get commandline arguments
    let config = VmmcConfig::parse();

    let mut evo_vmmc = if config.input() != "" {
        let contents = fs::read_to_string(config.input())?;
        toml::from_str(&contents)?
    } else {
        EvoVmmc::default()
    };

    // Get default params
    // Seed the rng
    let seed = evo_vmmc.ip().seed;
    println!("Using seed = {:x?}", seed);
    let mut rng = SmallRng::seed_from_u64(seed as u64);

    evo_vmmc.step_all(&mut rng);

    Ok(())
}
