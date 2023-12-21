use std::fs::create_dir_all;

use clap::Parser;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use threadpool::ThreadPool;
use vmmc::{
    cli::VmmcConfig,
    io::{clear_out_files, write_geometry_png},
    morphology::Morphology,
    polygons::{calc_bond_distribution, calc_polygons, Polygon},
    protocol::{FixedProtocol, ProtocolStep},
    run_vmmc,
    stats::RunStats,
    vmmc::Vmmc,
    vmmc_from_config, InputParams, VmmcCallback,
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

#[derive(Serialize, Deserialize, Clone)]
struct ExperimentStats {
    num_particles: usize,
    interaction_energy: f64,
    chemical_potential: f64,
    polygons: Vec<Polygon>,
    bond_distributions: Vec<Vec<usize>>, // shape -> {}
}

impl ExperimentStats {
    fn polygon_count(&self) -> usize {
        self.polygons.len()
    }

    fn average_bonds(&self) -> Vec<f64> {
        let mut v = Vec::new();
        for shape_stats in self.bond_distributions.iter() {
            let weighted_sum: usize = shape_stats.iter().enumerate().map(|(i, c)| i * c).sum();
            let avg_bonds = weighted_sum as f64 / self.num_particles as f64;
            v.push(avg_bonds);
        }
        v
    }
}
#[derive(Serialize, Deserialize)]
struct ExperimentCallback {
    stats: Vec<ExperimentStats>,
}
impl VmmcCallback for ExperimentCallback {
    type CbResult = Vec<ExperimentStats>;
    // runs after every million steps
    fn run(&mut self, vmmc: &Vmmc, step: &ProtocolStep, _: usize, _: &RunStats) {
        let num_particles = vmmc.particles().num_particles();
        let interaction_energy = step.interaction_energy();
        let chemical_potential = step.chemical_potential();
        let polygons = calc_polygons(vmmc, 6);
        let bond_distributions = calc_bond_distribution(vmmc);
        let stats = ExperimentStats {
            num_particles,
            interaction_energy,
            chemical_potential,
            polygons,
            bond_distributions,
        };
        self.stats.push(stats);
    }

    fn state(&self) -> Vec<ExperimentStats> {
        self.stats.clone()
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
            let cb = Box::new(ExperimentCallback { stats: Vec::new() });

            let r = run_vmmc(&mut vmmc, ip.protocol, Some(cb), &mut thread_rng);

            let avg_energy = vmmc.get_average_energy();
            println!("average energy of job_{:?} = {:?} kBT\n", jdx, avg_energy);

            write_geometry_png(&vmmc, &thread_config.geometry());

            let toml = toml::to_string(&r.unwrap()).unwrap();
            std::fs::write(thread_config.stats(), toml).expect("Unable to write file");
        });
    }
    pool.join()
}
