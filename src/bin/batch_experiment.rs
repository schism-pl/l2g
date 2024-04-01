// use std::fs::create_dir_all;

// use clap::Parser;
// use rand::{rngs::SmallRng, Rng, SeedableRng};
// use serde::{Deserialize, Serialize};
// use threadpool::ThreadPool;
// use vmmc::{
//     cli::VmmcConfig,
//     io::{clear_out_files, write_geometry_png},
//     polygons::{calc_bond_distribution, calc_polygons, Polygon},
//     protocol::ProtocolStep,
//     run_vmmc,
//     stats::RunStats,
//     vmmc::Vmmc,
//     vmmc_from_config, InputParams, VmmcCallback,
// };

// #[derive(Serialize, Deserialize, Clone)]
// struct ExperimentStats {
//     num_particles: usize,
//     interaction_energy: f64,
//     chemical_potential: f64,
//     polygons: Vec<Polygon>,
//     bond_distributions: Vec<Vec<usize>>, // shape -> {}
// }

// #[derive(Serialize, Deserialize)]
// struct StatState {
//     stats: Vec<ExperimentStats>,
// }

// struct ExperimentCallback {
//     stats: Vec<ExperimentStats>,
// }
// impl VmmcCallback for ExperimentCallback {
//     type CbResult = StatState;
//     // runs after every million steps
//     fn run(&mut self, vmmc: &Vmmc, step: &ProtocolStep, _: usize, _: &RunStats) {
//         assert!(vmmc.well_formed());
//         let num_particles = vmmc.particles().num_particles();
//         let interaction_energy = step.interaction_energy();
//         let chemical_potential = step.chemical_potential();
//         let polygons = calc_polygons(vmmc, 6);
//         let bond_distributions = calc_bond_distribution(vmmc);
//         let stats = ExperimentStats {
//             num_particles,
//             interaction_energy,
//             chemical_potential,
//             polygons,
//             bond_distributions,
//         };
//         self.stats.push(stats);
//     }

//     fn state(&self) -> StatState {
//         StatState {
//             stats: self.stats.clone(),
//         }
//     }
// }

// fn exec_job(config: VmmcConfig, jdx: usize, thread_seed: u64) {
//     let mut thread_rng = SmallRng::seed_from_u64(thread_seed);
//     let mut ip = InputParams::default();
//     ip.seed = thread_seed as i64;

//     // TODO: remove outdated files
//     // set up output directory
//     let mut thread_config = config.clone();

//     thread_config.set_output_dir(&format!("{}/{:?}", thread_config.output_dir(), jdx));

//     // set up thread output directory
//     let out_path = std::path::Path::new(thread_config.output_dir());
//     create_dir_all(out_path).unwrap();
//     clear_out_files(&thread_config).unwrap();

//     // write config to output directory
//     let toml = toml::to_string(&ip).unwrap();
//     std::fs::write(thread_config.toml(), toml).expect("Unable to write file");

//     let mut vmmc = vmmc_from_config(&ip, &mut thread_rng);
//     let cb = Box::new(ExperimentCallback { stats: Vec::new() });

//     let r = run_vmmc(&mut vmmc, ip.protocol, cb, &mut thread_rng);

//     write_geometry_png(&vmmc, &thread_config.geometry());

//     let toml = toml::to_string(&r.unwrap()).unwrap();
//     std::fs::write(thread_config.stats(), toml).expect("Unable to write file");
// }

fn main() {}

// fn main() {
//     env_logger::init();
//     // Get commandline arguments
//     let config = VmmcConfig::parse();

//     let num_threads = 10;
//     let num_jobs = 10;
//     let pool = ThreadPool::new(num_threads);

//     // Seed the rng
//     let seed = InputParams::default().seed as u64;
//     println!("Using seed = {:?}", seed);
//     let mut rng = SmallRng::seed_from_u64(seed);
//     let seeds: Vec<u64> = (0..num_jobs).map(|_| rng.gen()).collect();

//     for jdx in 0..num_jobs {
//         let thread_config = config.clone();
//         let thread_seed = seeds[jdx];
//         pool.execute(move || {
//             if let Err(e) = std::panic::catch_unwind(|| {
//                 exec_job(thread_config, jdx, thread_seed);
//             }) {
//                 println!("Job {:?} encountered a panic: {:?}", jdx, e)
//             }
//         })
//     }

//     pool.join()
// }
