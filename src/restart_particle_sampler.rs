#![allow(dead_code)]
#![allow(non_snake_case)]
mod vlsv_reader;
use crate::mod_vlsv_reader::VlsvFile;
use crate::vlsv_reader::*;
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

pub enum SAMPLING {
    IN,
    OUT,
    FULL,
}

//Some configuration params
const INIT_TIME: f32 = 305.0;
const PPC: usize = 1024;
const STRIDE: usize = 32;
const SPARSE: f32 = 1e-16;
const LSHELL_MIN: f64 = 8.0;
const LSHELL_MAX: f64 = 20.0;
const RE: f64 = 6378137.0;
const XMIN: f64 = -10.0;
const XMAX: f64 = 1000.0;
const YMIN: f64 = -10.0;
const YMAX: f64 = 1000.0;
const ZMIN: f64 = -1000.0;
const ZMAX: f64 = 1000.0;
const SAMPLE_SCHEME: SAMPLING = SAMPLING::IN;
const THERMAL_RADIOUS: f64 = 75000.0;
const VDF_SHIFT: bool = false;

struct Particle {
    x: f32,
    y: f32,
    z: f32,
    vx: f32,
    vy: f32,
    vz: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let restart_file: &String = args.get(1).expect("No file provided!");
    let f = VlsvFile::new(restart_file).unwrap();
    let spatial_mesh = f
        .get_spatial_mesh_bbox()
        .expect("Could not read spatial mesh from {restart_file}");
    let velocity_mesh = f
        .get_vspace_mesh_extents("proton")
        .expect("Could not read vspace mesh extents");
    let ncells = 1; //spatial_mesh.0 * spatial_mesh.1 * spatial_mesh.2;

    println!(
        "Sampling particles from {restart_file} [PPC={PPC} N={}].",
        PPC * ncells / STRIDE
    );
    let (nvx, nvy, nvz) = f
        .get_vspace_mesh_bbox("proton")
        .expect("Could not read vspace mesh size");
    let (vxmin, vymin, vzmin, vxmax, vymax, vzmax) = velocity_mesh;
    let dvx = (vxmax - vxmin) / nvx as f64;
    let dvy = (vymax - vymin) / nvy as f64;
    let dvz = (vzmax - vzmin) / nvz as f64;
    let pb = ProgressBar::new(((ncells / STRIDE) * PPC).try_into().unwrap());
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let target_cid = 356829833;
    let particles: Vec<Option<Particle>> = (1..=ncells)
        .into_par_iter()
        .filter(|&cid| (cid - 1) % STRIDE == 0)
        .flat_map(|cid| {
            let mut rng = rng();
            let coords = f
                .get_cell_coordinate(target_cid as u64)
                .expect("Could not read in coordinates for target_cid {target_cid}");
            let x = coords[0] / RE;
            let y = coords[1] / RE;
            let z = coords[2] / RE;
            let vdf = f
                .read_vdf::<f32>(target_cid, "proton")
                .expect("Could not read VDF from target_cid {target_cid}");
            let v3 = vdf.slice(s![.., .., .., 0]);
            let mut weights: Vec<f64> = Vec::with_capacity(nvx * nvy * nvz);
            let mut sum = 0.0f64;
            for k in 0..nvz {
                for j in 0..nvy {
                    for i in 0..nvx {
                        let w = v3[(i, j, k)] as f64;
                        let w = if w > SPARSE as f64 { w } else { 0.0 };
                        weights.push(w);
                        sum += w;
                    }
                }
            }

            if sum <= 0.0 {
                return Vec::<Option<Particle>>::new();
            }
            let dist = WeightedIndex::new(&weights).unwrap();
            (0..PPC)
                .map(|_| {
                    let flat = dist.sample(&mut rng);

                    pb.inc(1);
                    let pl = nvx * nvy;
                    let k = flat / pl;
                    let rem = flat % pl;
                    let j = rem / nvx;
                    let i = rem % nvx;
                    let mut urand = || -> f64 { rng.random::<f64>() - 0.5 };
                    let (vx, vy, vz) = (
                        vxmin + (i as f64 + 0.5 + urand()) * dvx,
                        vymin + (j as f64 + 0.5 + urand()) * dvy,
                        vzmin + (k as f64 + 0.5 + urand()) * dvz,
                    );

                    let keep = true;
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
    let file = File::create("lucky_particles.txt").expect("Could not open output file");
    let mut writer = BufWriter::new(file);
    let mut nparticles = 0;
    for particle in &particles {
        if let Some(p) = particle {
            nparticles += 1;
            writeln!(
                writer,
                "{},{},{},{},{},{},{}",
                INIT_TIME, p.x, p.y, p.z, p.vx, p.vy, p.vz
            )
            .expect("Could not write the particle file!");
        }
    }
    println!("Sampled {} particles", nparticles);
}
