use std::fs::create_dir_all;

use l2g::nn::*;
use vmmc::{io::write_protocols_png, protocol::SynthesisProtocol};

// fn nn_from_vec(iws: Vec<f64>, ews: Vec<f64>, mws: Vec<f64>, biases: Vec<f64>) -> NueralNet {
//     let mut layers = Vec::new();
//     for idx in 0..iws.len() {
//         let iw = iws[idx];
//         let ew = ews[idx];
//         let mw = mws[idx];
//         let bias = biases[idx];
//         let layer = HiddenLayer::new(iw,ew,mw,bias);
//         layers.push(layer);
//     }
//     NueralNet::new(layers, 0.1)
// }

// fn dummy_nn() -> NueralNet {
//     let iws = vec![1.68386, 0.455614, -2.51552, 1.49307, -0.764242, 2.00576, 2.84198, 0.712714, -1.62817, 0.235705];
//     let ews = vec![-0.692959, -0.0310458, 0.385992, -1.32482, -0.67391, 0.228362, -0.959666, 0.957226, -1.38234, 0.164757];
//     let mws = vec![2.16823, 0.722991, -0.433223, -0.617527, -1.08172, 0.0628167, -1.1174, -0.0688782, -0.77532, 1.03813];
//     let biases = vec![-0.115804, -1.22997, 0.132718, -0.19448, 0.387762, 1.2796, -1.12286, 0.330479, -0.982813, -0.124925];
//     nn_from_vec(iws, ews, mws, biases)
// }

// fn print_samples(mu: f64, epsilon: f64) {
//     let nn = dummy_nn();
//     let mut curr_mu = mu;
//     let mut curr_epsilon = epsilon;
//     let (delta_ep,delta_mu) = nn.eval(0.0); 
//     println!("dMu/dEpsilon: {delta_mu} {delta_ep}");
//     for idx in 0..1000 {
//         let input = idx as f64 / 1000.0;
//         println!("Input = {input}");
//         let (delta_ep,delta_mu) = nn.eval(input); 
//         curr_mu += delta_mu;
//         curr_epsilon += delta_ep;
//         curr_mu = curr_mu.clamp(-20.0, 20.0);
//         curr_epsilon = curr_epsilon.clamp(0.0, 20.0);
//         println!("{idx}: {curr_mu}({delta_mu}) {curr_epsilon}({delta_ep})");
//     }
// }

// fn nn_protocol_to_png(protocol: &SynthesisProtocol, pathname: &str) {
//     // protocol.megastep_iter(), &format!("{p_str}/protocols.png"
//     write_protocols_png(protocol.megastep_iter(), pathname)
// }

fn main() -> anyhow::Result<()> {
    // env_logger::init();
    // // Get commandline arguments
    // let config = VmmcConfig::parse();

    // let mut evo_vmmc = if config.input() != "" {
    //     println!("Reading configuration from {}", config.input());
    //     let contents = fs::read_to_string(config.input())?;
    //     toml::from_str(&contents)?
    // } else {
    //     println!("No configuration provided, using default config");
    //     EvoVmmc::default()
    // };



    // // Init I/O
    let outdir = "mutation_viz";
    println!("Writing output to {}\n", outdir);
    let out_path = std::path::Path::new(outdir);
    create_dir_all(out_path).unwrap();

    // let mut example_nn = dummy_nn();
    // let mut nn = NueralNet::new_rand(10);
    // let base_protocol = SynthesisProtocol::flat_protocol(0.0, 10.0, 1000);
    // print_samples(2.0, 5.0);


    // let mut curr_mu = 2.0;
    // let mut curr_epsilon = 5.0;
    // for step in example_nn.current_protocol(&base_protocol) {
    //     println!("mu = {} epsilon = {}", step.chemical_potential(), step.interaction_energy());
    // }
    // Need function that takes iter and prints protocol to path
    
    // let seed = 1337;
    // nn_protocol_to_png(curr_proto, pathname);
    for idx in 0..50 {
        println!("Mutation {idx}");
        let nn_config = NnConfig::new(1337, 1000, 0.1);
        let mut nn = NueralNet::from_config(&nn_config);
        let base_protocol = SynthesisProtocol::flat_protocol(0.0, 10.0, 1000);
        let curr_proto = nn.current_protocol(&base_protocol);
        write_protocols_png(curr_proto, &format!("{outdir}/{:0>3}.png", idx));
        nn.mutate();
    }


    Ok(())
}


// TODO: compare print samples