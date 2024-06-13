use std::fs::create_dir_all;

use vmmc::{
    io::{write_geometry_png, write_protocols_png, write_stats},
    vmmc::Vmmc,
};

use crate::nn::Dna;

pub fn record_children(candidates: &[Dna], children: &[Vmmc], gen_idx: usize) {
    // Dump outputs
    for (child_idx, child) in children.iter().enumerate() {
        let p_str = format!("./out/{:0>3}/{:0>3}", gen_idx, child_idx);
        let out_path = std::path::Path::new(&p_str);
        create_dir_all(out_path).unwrap();
        write_geometry_png(child, &format!("{p_str}/geometry.png"));
        // let nn_config = candidates[idx].nn_config();
        let dna = &candidates[child_idx];
        let protocol_iter = candidates[child_idx].protocol_iter();
        let toml = toml::to_string(dna).unwrap();
        std::fs::write(format!("{p_str}/dna.toml"), toml).expect("Unable to write file");
        write_protocols_png(protocol_iter, &format!("{p_str}/protocols.png"));
        write_stats(child, &format!("{p_str}/stats.txt"))
    }
}
