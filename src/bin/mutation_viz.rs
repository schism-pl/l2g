use std::fs::create_dir_all;

use l2g::nn::*;
use vmmc::{io::write_protocols_png, protocol::SynthesisProtocol};

fn main() -> anyhow::Result<()> {
    // Init I/O
    let outdir = "mutation_viz";
    println!("Writing output to {}\n", outdir);
    let out_path = std::path::Path::new(outdir);
    create_dir_all(out_path).unwrap();

    for idx in 0..50 {
        println!("Mutation {idx}");
        let nn_config = NnConfig::new(1337, 0, 1000, 0.1);
        let mut nn = NueralNet::from_config(&nn_config);
        let base_protocol = SynthesisProtocol::flat_protocol(0.0, 10.0, 1000);
        let curr_proto = nn.current_protocol(&base_protocol);
        write_protocols_png(curr_proto, &format!("{outdir}/{:0>3}.png", idx));
        nn.mutate();
    }

    Ok(())
}
