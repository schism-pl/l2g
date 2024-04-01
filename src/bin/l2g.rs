use clap::Parser;
use l2g::evo_vmmc::{self, EvoVmmc};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fs::{self, create_dir_all};
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
    let seed = evo_vmmc.initial_ip().seed;
    println!("Using seed = {:x?}", seed);
    let mut rng = SmallRng::seed_from_u64(seed as u64);

    println!(
        "Executing {} generations, with {} survivors per generation and {} children per survivor",
        evo_vmmc.num_generations(),
        evo_vmmc.survivors_per_generation(),
        evo_vmmc.children_per_survivor()
    );
    println!("Fitness Function: {:?}", evo_vmmc.fitness_func);
    // TODO: print mutation function
    let ip = evo_vmmc.initial_ip();
    println!(
        "Simbox: {}x{} with {} initial particles",
        ip.box_width, ip.box_height, ip.initial_particles
    );
    println!(
        "Each simulation runs for {} megasteps",
        ip.protocol.num_megasteps()
    );
    // pub protocol: SynthesisProtocol,
    // pub shapes: Vec<Morphology>,

    // Init I/O
    println!("Writing output to {}\n", config.output_dir());
    let out_path = std::path::Path::new(config.output_dir());
    create_dir_all(out_path).unwrap();
    // clear_out_files(&config)?;

    // dump full config toml to output directory
    let toml = toml::to_string(&evo_vmmc).unwrap();
    fs::write(config.toml(), toml).expect("Unable to write file");

    evo_vmmc.step_all(&mut rng);

    Ok(())
}
