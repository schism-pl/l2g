use std::fs::create_dir_all;

use fll::FLLConfig;
use l2g::nn::*;
use vmmc::{io::write_protocols_png, protocol::SynthesisProtocol};

fn main() -> anyhow::Result<()> {
    // Init I/O
    let outdir = "mutation_viz";
    println!("Writing output to {}\n", outdir);
    let out_path = std::path::Path::new(outdir);
    create_dir_all(out_path).unwrap();

    let proto = SynthesisProtocol::flat_protocol(0.0, 10.0, 20);

    // let nn_config = NnConfig::new(nn_seed, 0, 1000, 0.1);
    let config = FLLConfig::new(10, 100, 0.5);

    let mut dna = Dna::fresh_fll(config, proto);
    for idx in 0..50 {
        println!("Mutation {idx}");
        // let nn_config = NnConfig::new(1337, 0, 1000, 0.1);
        // // let mut nn = NueralNet::from_config(&nn_config);
        // let base_protocol = SynthesisProtocol::flat_protocol(0.0, 10.0, 1000);

        // let curr_proto = nn.current_protocol(&base_protocol);
        // let curr_proto = nn_config.current_protocol();
        let curr_proto = dna.protocol_iter();
        write_protocols_png(curr_proto, &format!("{outdir}/{:0>3}.png", idx));
        dna.mutate(0);
    }

    Ok(())
}
