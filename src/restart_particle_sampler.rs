#![allow(dead_code)]
#![allow(non_snake_case)]
mod vlsv_reader;
use crate::mod_vlsv_reader::VlsvFile;
use crate::vlsv_reader::*;
use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::s;
use rand::Rng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::rng;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum SamplingScheme {
    In,
    Out,
    Full,
}

#[derive(Parser, Debug)]
#[command(author, version, about = "VLSV Particle Sampler", long_about = None)]
struct Cli {
    /// VLSV file
    restart_file: String,

    /// Output file
    #[arg(short, long, default_value = "lucky_particles.txt")]
    output: String,

    /// Initial time
    #[arg(long, default_value_t = 305.0)]
    init_time: f32,

    /// Particles per cell (PPC)
    #[arg(long, default_value_t = 1024)]
    ppc: usize,

    /// Stride for cells
    #[arg(long, default_value_t = 32)]
    stride: usize,

    /// Sparsity to be used
    #[arg(long, default_value_t = 1e-16)]
    sparse: f32,

    // Bounding Box Parameters
    #[arg(long, default_value_t = -1.0e23)]
    xmin: f64,
    #[arg(long, default_value_t = 1.0e23)]
    xmax: f64,
    #[arg(long, default_value_t = -1.0e23)]
    ymin: f64,
    #[arg(long, default_value_t = 1.0e23)]
    ymax: f64,
    #[arg(long, default_value_t = -1.0e23)]
    zmin: f64,
    #[arg(long, default_value_t = 1.0e23)]
    zmax: f64,

    /// Sampling scheme (based on thermal radius)
    #[arg(long, value_enum, default_value_t = SamplingScheme::Full)]
    sample_scheme: SamplingScheme,

    /// Thermal radius in m/s
    #[arg(long, default_value_t = 75000.0)]
    thermal_radius: f64,

    /// Enable VDF shift (based on bulk velocity)
    #[arg(long, default_value_t = false)]
    vdf_shift: bool,

    /// Target a specific CELLID only
    #[arg(long)]
    target_cell: Option<usize>,
}

// Physical constants
const RE: f64 = 6378137.0;

struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
}

fn main() {
    let cli = Cli::parse();
    let f = VlsvFile::new(&cli.restart_file).unwrap();
    let spatial_mesh = f
        .get_spatial_mesh_bbox()
        .unwrap_or_else(|| panic!("Could not read spatial mesh from {}", cli.restart_file));
    let velocity_mesh = f
        .get_vspace_mesh_extents("proton")
        .expect("Could not read vspace mesh extents");
    let ncells = spatial_mesh.0 * spatial_mesh.1 * spatial_mesh.2;

    let cells_to_process: Vec<usize> = if let Some(target) = cli.target_cell {
        vec![target]
    } else {
        (1..=ncells)
            .filter(|&cid| (cid - 1) % cli.stride == 0)
            .collect()
    };

    println!(
        "Sampling particles from {} [PPC={} N={}].",
        cli.restart_file,
        cli.ppc,
        cells_to_process.len() * cli.ppc
    );

    let (nvx, nvy, nvz) = f
        .get_vspace_mesh_bbox("proton")
        .expect("Could not read vspace mesh size");
    let (vxmin, vymin, vzmin, vxmax, vymax, vzmax) = velocity_mesh;
    let dvx = (vxmax - vxmin) / nvx as f64;
    let dvy = (vymax - vymin) / nvy as f64;
    let dvz = (vzmax - vzmin) / nvz as f64;

    let pb = ProgressBar::new((cells_to_process.len() * cli.ppc).try_into().unwrap());
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let particles: Vec<Option<Particle>> = cells_to_process
        .into_par_iter()
        .flat_map(|cid| {
            let mut rng = rng();
            let coords = f
                .get_cell_coordinate(cid as u64)
                .unwrap_or_else(|| panic!("Could not read in coordinates for cid {cid}"));

            let is_in_box = {
                let x = coords[0] / RE;
                let y = coords[1] / RE;
                let z = coords[2] / RE;
                x > cli.xmin
                    && x < cli.xmax
                    && y > cli.ymin
                    && y < cli.ymax
                    && z > cli.zmin
                    && z < cli.zmax
            };

            if !is_in_box {
                return Vec::<Option<Particle>>::new();
            }

            let vdf = f
                .read_vdf::<f32>(cid, "proton")
                .expect("Could not read VDF from cid {cid}");
            let v3 = vdf.slice(s![.., .., .., 0]);
            let mut weights: Vec<f64> = Vec::with_capacity(nvx * nvy * nvz);
            let mut sum = 0.0f64;

            for k in 0..nvz {
                for j in 0..nvy {
                    for i in 0..nvx {
                        let w = v3[(i, j, k)] as f64;
                        let w = if w > cli.sparse as f64 { w } else { 0.0 };
                        weights.push(w);
                        sum += w;
                    }
                }
            }

            if sum <= 0.0 {
                return Vec::<Option<Particle>>::new();
            }
            let moments_r = f
                .read_vg_variable_at::<f64>("moments_r", &[cid])
                .expect("Could not read moments");

            let vg_v = [moments_r[1], moments_r[2], moments_r[3]];
            let dist = WeightedIndex::new(&weights).unwrap();

            (0..cli.ppc)
                .map(|_| {
                    let flat = dist.sample(&mut rng);

                    pb.inc(1);
                    let pl = nvx * nvy;
                    let k = flat / pl;
                    let rem = flat % pl;
                    let j = rem / nvx;
                    let i = rem % nvx;
                    let mut urand = || -> f64 { rng.random::<f64>() - 0.5 };
                    let (vx, vy, vz) = if cli.vdf_shift {
                        (
                            vxmin + (i as f64 + 0.5 + urand()) * dvx - vg_v[0],
                            vymin + (j as f64 + 0.5 + urand()) * dvy - vg_v[1],
                            vzmin + (k as f64 + 0.5 + urand()) * dvz - vg_v[2],
                        )
                    } else {
                        (
                            vxmin + (i as f64 + 0.5 + urand()) * dvx,
                            vymin + (j as f64 + 0.5 + urand()) * dvy,
                            vzmin + (k as f64 + 0.5 + urand()) * dvz,
                        )
                    };

                    let thermal_circle_dist: f64 = if !cli.vdf_shift {
                        ((vx - vg_v[0]).powi(2) + (vy - vg_v[1]).powi(2) + (vz - vg_v[2]).powi(2))
                            .sqrt()
                    } else {
                        (vx.powi(2) + vy.powi(2) + vz.powi(2)).sqrt()
                    };
                    let keep = match cli.sample_scheme {
                        SamplingScheme::In => thermal_circle_dist < cli.thermal_radius,
                        SamplingScheme::Out => thermal_circle_dist > cli.thermal_radius,
                        SamplingScheme::Full => true,
                    };

                    if keep {
                        Some(Particle {
                            x: coords[0] as f32,
                            y: coords[1] as f32,
                            z: coords[2] as f32,
                            vx: vx as f32,
                            vy: vy as f32,
                            vz: vz as f32,
                        })
                    } else {
                        None
                    }
                })
                .collect::<Vec<Option<Particle>>>()
        })
        .collect();

    let file = File::create(&cli.output).expect("Could not open output file");
    let mut writer = BufWriter::new(file);
    let mut nparticles = 0;
    for particle in &particles {
        if let Some(p) = particle {
            nparticles += 1;
            writeln!(
                writer,
                "{},{},{},{},{},{},{}",
                cli.init_time, p.x, p.y, p.z, p.vx, p.vy, p.vz
            )
            .expect("Could not write the particle file!");
        }
    }
    pb.finish();
    println!("Sampled {} particles", nparticles);
}
