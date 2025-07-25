use l2g::engine::EvoEngine;
use l2g::nn::Dna;
use l2g::run_fresh_vmmc_to_console;
use rand::SeedableRng;
use std::fs;
use vmmc::Prng;

// fn init_logging(output_dir: &str) -> anyhow::Result<()> {
//     // env_logger::init();
//     // log::set_logger(&L2G_LOGGER)?;
//     // log::set_max_level(LevelFilter::Info);
//     let logfile = FileAppender::builder()
//         .encoder(Box::new(PatternEncoder::new("{m}\n")))
//         .build(format!("{output_dir}/output.log"))?;

//     let stdout = ConsoleAppender::builder()
//         .encoder(Box::new(PatternEncoder::new("{m}\n")))
//         .target(Target::Stdout)
//         .build();
//     let logfile_appender = Appender::builder().build("logfile", Box::new(logfile));
//     let stdout_appender = Appender::builder().build("stdout", Box::new(stdout));

//     let config = Config::builder()
//         .appender(logfile_appender)
//         .appender(stdout_appender)
//         .build(
//             Root::builder()
//                 .appender("logfile")
//                 .appender("stdout")
//                 .build(LevelFilter::Info),
//         )?;

//     log4rs::init_config(config)?;
//     Ok(())
// }

fn main() -> anyhow::Result<()> {
    // Get commandline arguments
    // 1) sim params
    // 2)
    let args: Vec<String> = std::env::args().collect();
    let config_path = args[1].clone();
    let dna_path = args[2].clone();

    let contents = fs::read_to_string(config_path)?;
    let engine: EvoEngine = toml::from_str(&contents)?;

    let contents = fs::read_to_string(dna_path)?;
    let dna: Dna = toml::from_str(&contents)?;

    let seed = engine.seed;
    let mut rng = Prng::seed_from_u64(seed as u64);

    let protocol_iter = dna.protocol_iter();
    let _ = run_fresh_vmmc_to_console(engine.sim_params(), protocol_iter, &mut rng);

    // let config = VmmcConfig::parse();
    // init_logging(config.output_dir())?;

    // let mut engine = if config.input() != "" {
    //     log::info!("Reading configuration from {}", config.input());
    //     let contents = fs::read_to_string(config.input())?;
    //     toml::from_str(&contents)?
    // } else {
    //     log::info!("No configuration provided, using default config");
    //     EvoEngine::default()
    // };

    // // Get default params
    // // Seed the rng
    // let seed = engine.seed;
    // log::info!("Using seed = {:x?}", seed);
    // let mut rng = SmallRng::seed_from_u64(seed as u64);

    // log::info!(
    //     "Executing {} generations, with {} survivors per generation and {} children per survivor",
    //     engine.num_generations(),
    //     engine.survivors_per_generation(),
    //     engine.children_per_survivor()
    // );
    // log::info!("Fitness Function: {:?}", engine.fitness_func);
    // log::info!("Mutation Method: {}", engine.init_dna.type_str());
    // let ip = engine.sim_params();
    // log::info!(
    //     "Simbox: {}x{} with {} initial particles",
    //     ip.box_width,
    //     ip.box_height,
    //     ip.initial_particles
    // );

    // // Init I/O
    // log::info!("Writing output to {}", config.output_dir());
    // let out_path = std::path::Path::new(config.output_dir());
    // create_dir_all(out_path).unwrap();

    // // dump full config toml to output directory
    // let toml = toml::to_string(&engine).unwrap();
    // fs::write(config.toml(), toml).expect("Unable to write file");

    // engine.step_all_and_save(config.output_dir(), &mut rng);

    // let fit_path = format!("{}/fitnesses.txt", config.output_dir());
    // fs::write(fit_path, format!("{:?}", &engine.fitnesses)).expect("Unable to write file");

    // let progress_path = format!("{}/progress.png", config.output_dir());
    // write_progress_png(&engine.fitnesses, &progress_path);
    Ok(())
}
