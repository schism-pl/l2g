use clap::Parser;
use l2g::engine::EvoEngine;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::fs::{self, create_dir_all};
use vmmc::cli::VmmcConfig;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    // Get commandline arguments
    let config = VmmcConfig::parse();

    let mut engine = if config.input() != "" {
        println!("Reading configuration from {}", config.input());
        let contents = fs::read_to_string(config.input())?;
        toml::from_str(&contents)?
    } else {
        println!("No configuration provided, using default config");
        EvoEngine::default()
    };

    // Get default params
    // Seed the rng
    let seed = engine.seed;
    println!("Using seed = {:x?}", seed);
    let mut rng = SmallRng::seed_from_u64(seed as u64);

    println!(
        "Executing {} generations, with {} survivors per generation and {} children per survivor",
        engine.num_generations(),
        engine.survivors_per_generation(),
        engine.children_per_survivor()
    );
    println!("Fitness Function: {:?}", engine.fitness_func);
    // TODO: print mutation function
    let ip = engine.sim_params();
    println!(
        "Simbox: {}x{} with {} initial particles",
        ip.box_width, ip.box_height, ip.initial_particles
    );
    // println!(
    //     "Each simulation runs for {} megasteps",
    //     ip.protocol.num_megasteps()
    // );

    // Init I/O
    println!("Writing output to {}\n", config.output_dir());
    let out_path = std::path::Path::new(config.output_dir());
    create_dir_all(out_path).unwrap();

    // dump full config toml to output directory
    let toml = toml::to_string(&engine).unwrap();
    fs::write(config.toml(), toml).expect("Unable to write file");

    engine.step_all(&mut rng);

    Ok(())
}
