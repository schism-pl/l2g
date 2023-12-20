use clap::Parser;
use l2g::evo_vmmc::run_fresh_vmmc;
use rand::{rngs::SmallRng, SeedableRng};
use vmmc::{
    cli::VmmcConfig, morphology::Morphology, protocol::FixedProtocol, run_vmmc, vmmc_from_config,
    InputParams,
};

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
    // Get default params

    let num_threads = 1;
    let num_jobs = 2;

    // Seed the rng
    let seed = config.seed();
    println!("Using seed = {:?}", seed);
    let mut rng = SmallRng::seed_from_u64(seed);

    for jdx in 0..num_jobs {
        let ip = experiment_ip();
        let mut vmmc = vmmc_from_config(&ip, &mut rng);
        run_vmmc(&mut vmmc, ip.protocol, None, &mut rng);

        let avg_energy = vmmc.get_average_energy();
        println!("average energy of job_{:?} = {:?} kBT\n", jdx, avg_energy);
    }
}
