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
const THERMAL_RADIOUS: f64 = 550.0;
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
    let ncells = spatial_mesh.0 * spatial_mesh.1 * spatial_mesh.2;

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

    let particles: Vec<Option<Particle>> = (1..=ncells)
        .into_par_iter()
        .filter(|&cid| (cid - 1) % STRIDE == 0)
        .flat_map(|cid| {
            let mut rng = rng();
            let coords = f
                .get_cell_coordinate(cid as u64)
                .expect("Could not read in coordinates for cid {cid}");
            // let rho = ((coords[0] * coords[0] + coords[1] * coords[1] + coords[2] * coords[2])
            //     .sqrt())
            //     / RE;
            // if rho < LSHELL_MIN || rho > LSHELL_MAX {
            //     return Vec::<Particle>::new();
            // }
            let x = coords[0] / RE;
            let y = coords[1] / RE;
            let z = coords[2] / RE;
            let is_in_box = x > XMIN && x < XMAX && y > YMIN && y < YMAX && z > ZMIN && z < ZMAX;
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
                        let w = if w > SPARSE as f64 { w } else { 0.0 };
                        weights.push(w);
                        sum += w;
                    }
                }
            }

            if sum <= 0.0 {
                return Vec::<Option<Particle>>::new();
            }
            let vg_v_x = f
                .read_vg_variable_at::<f64>("moments_r", Some(0), &[cid])
                .expect("Could not read moments");
            let vg_v_y = f
                .read_vg_variable_at::<f64>("moments_r", Some(1), &[cid])
                .expect("Could not read moments");
            let vg_v_z = f
                .read_vg_variable_at::<f64>("moments_r", Some(2), &[cid])
                .expect("Could not read moments");
            let vg_v = [vg_v_x[0], vg_v_y[0], vg_v_z[0]];

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
                    let (vx, vy, vz) = if VDF_SHIFT {
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
                    let thermal_circle_dist: f64 = if !VDF_SHIFT {
                        ((vx - vg_v[0]).powi(2) + (vy - vg_v[1]).powi(2) + (vz - vg_v[2]).powi(2))
                            .sqrt()
                    } else {
                        (vx.powi(2) + vy.powi(2) + vz.powi(2)).sqrt()
                    };
                    let retval = match SAMPLE_SCHEME {
                        SAMPLING::IN => {
                            if THERMAL_RADIOUS > thermal_circle_dist {
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
                        }
                        SAMPLING::OUT => {
                            if THERMAL_RADIOUS < thermal_circle_dist {
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
                        }
                        SAMPLING::FULL => Some(Particle {
                            x: coords[0] as f32,
                            y: coords[1] as f32,
                            z: coords[2] as f32,
                            vx: vx as f32,
                            vy: vy as f32,
                            vz: vz as f32,
                        }),
                    };
                    retval
                })
                .collect::<Vec<Option<Particle>>>()
        })
        .collect();
    println!("Sampled {} particles", particles.len());
    let file = File::create("lucky_particles.txt").expect("Could not open output file");
    let mut writer = BufWriter::new(file);
    for particle in &particles {
        if let Some(p) = particle {
            writeln!(
                writer,
                "{},{},{},{},{},{},{}",
                INIT_TIME, p.x, p.y, p.z, p.vx, p.vy, p.vz
            )
            .expect("Could not write the particle file!");
        }
    }
}
