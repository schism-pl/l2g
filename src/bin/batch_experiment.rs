use std::fs::create_dir_all;

use clap::Parser;
use rand::{rngs::SmallRng, SeedableRng, Rng};
use vmmc::{
    cli::VmmcConfig, morphology::Morphology, protocol::FixedProtocol, run_vmmc, vmmc_from_config,
    InputParams, io::{write_geometry_png, clear_out_files},
};
use threadpool::ThreadPool;

fn experiment_ip() -> InputParams {
    let num_particles = 500;
    let box_width = 75.0;
    let box_height = 75.0;

    let prob_translate = 0.5;
    let max_translation = 0.15;
    let max_rotation = 0.2;

    let protocol = FixedProtocol::flat_protocol(0.0, 10.0, 10);

    let shapes = vec![Morphology::regular_3patch(0.1)];

    InputParams {
        num_particles,
        protocol,
        shapes,

        box_width,
        box_height,

        prob_translate,
        max_translation,
        max_rotation,
    }
}



fn main() {
    env_logger::init();
    // Get commandline arguments
    let config = VmmcConfig::parse();

    let num_threads = 8;
    let num_jobs = 8;
    let pool = ThreadPool::new(num_threads);

    // Seed the rng
    let seed = config.seed();
    println!("Using seed = {:?}", seed);
    let mut rng = SmallRng::seed_from_u64(seed);
    let seeds: Vec<u64> = (0..num_jobs).map(|_| rng.gen()).collect();

    for jdx in 0..num_jobs {
        // create job rng
        let thread_seed = seeds[jdx];
        let mut thread_rng = SmallRng::seed_from_u64(thread_seed);
        let ip = experiment_ip();

        // TODO: remove outdated files
        // set up output directory
        let mut thread_config = config.clone();
        thread_config.set_output_dir(&format!("{}/{:?}", thread_config.output_dir(), jdx));

        // set up thread output directory
        let out_path = std::path::Path::new(thread_config.output_dir());
        create_dir_all(out_path).unwrap();
        clear_out_files(&thread_config).unwrap();

        // write config to output directory
        let toml = toml::to_string(&ip).unwrap();
        std::fs::write(thread_config.toml(), toml).expect("Unable to write file");

        pool.execute(move || {

            let mut vmmc = vmmc_from_config(&ip, &mut thread_rng);
            run_vmmc(&mut vmmc, ip.protocol, None, &mut thread_rng);
    
            let avg_energy = vmmc.get_average_energy();
            println!("average energy of job_{:?} = {:?} kBT\n", jdx, avg_energy);

            write_geometry_png(&vmmc, &thread_config.geometry()); 
        });
       
    }
    pool.join()
}
