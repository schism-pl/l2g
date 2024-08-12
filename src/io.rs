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


pub fn write_progress_png(fitnesses: &[f64], pathname: &str) {
    use plotters::prelude::*;

    let num_candidates = fitnesses.len() as i32;

    let root_area = BitMapBackend::new(pathname, (1000, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let min_fitness = fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 10.0;
    let max_fitness = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 10.0;

    let mut ctx = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 32)
        .set_label_area_size(LabelAreaPosition::Bottom, 32)
        .caption("Interaction Energy", ("sans-serif", 32))
        .build_cartesian_2d(0..num_candidates - 1, min_fitness..max_fitness)
        .unwrap();


    ctx.configure_mesh().draw().unwrap();

    let score_line_style = RGBColor(0xf3, 0x70, 0x21).stroke_width(2);
    let best_score_line_style = RGBColor(0x09, 0x61, 0x91).stroke_width(2);

    let mut score_line = Vec::new();
    let mut best_score_line: Vec<(i32, f64)> = Vec::new();
    for (idx, fit) in fitnesses.iter().enumerate() {
        score_line.push((idx as i32, *fit));
        let last_score = if idx == 0 {0.0} else {best_score_line[idx-1].1};
        best_score_line.push((idx as i32, fit.max(last_score)));
    }

    ctx
        .draw_series(LineSeries::new(score_line, score_line_style))
        .unwrap();
    ctx
        .draw_series(LineSeries::new(best_score_line, best_score_line_style))
        .unwrap();
}