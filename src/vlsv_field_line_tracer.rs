#![allow(dead_code)]
#![allow(non_snake_case)]
mod vlsv_reader;

use crate::mod_vlsv_tracing::*;
use crate::vlsv_reader::*;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::Rng;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

const FWD: f64 = 1.0_f64;
const BWD: f64 = -1.0_f64;

#[derive(Parser, Debug)]
#[command(
    name = "vlsv_field_line_tracer",
    about = "Forward and backward field line tracer"
)]
struct Args {
    /// VLSV file (static field) or a directory (dynamic field)
    #[arg(short, long)]
    vlsv: Option<String>,

    /// Periodic boundary in X
    #[arg(long, default_value_t = false)]
    periodic_x: bool,

    /// Periodic boundary in Y
    #[arg(long, default_value_t = false)]
    periodic_y: bool,

    /// Periodic boundary in Z
    #[arg(long, default_value_t = false)]
    periodic_z: bool,

    /// Number of seed points to generate when no --input is provided
    #[arg(short, long, default_value_t = 1)]
    num_seeds: usize,

    /// Input txt file with seed points
    #[arg(short, long)]
    input: Option<String>,

    /// Number of steps to do
    #[arg(short, long, default_value_t = 10)]
    steps: usize,

    /// L-shell
    #[arg(short, long, default_value_t = 10.0)]
    lshell: f64,

    /// Save particle positions every N steps
    #[arg(long, default_value_t = 10000000000)]
    dump_every: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct Point<T: num_traits::Zero> {
    x: T,
    y: T,
    z: T,
}

impl<T: num_traits::Zero> Default for Point<T> {
    fn default() -> Self {
        Self {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }
}

pub fn step_euler<T, F>(optional_point: Option<Point<T>>, f: &F, dir: T) -> Option<Point<T>>
where
    T: PtrTrait + Sync + Send + Copy,
    F: Field<T> + Sync,
{
    const RE: f64 = physical_constants::f64::EARTH_RE;
    let step = T::from(1e3)?;
    let p1 = optional_point?;

    let rho = (p1.x * p1.x + p1.y * p1.y + p1.z * p1.z).sqrt();
    if rho < T::from(RE)? {
        return None;
    }

    if let Some(fields) = f.get_fields_at(T::zero(), p1.x, p1.y, p1.z) {
        let (bx, by, bz) = (fields[0], fields[1], fields[2]);
        let bmag = (bx * bx + by * by + bz * bz).sqrt();
        let (bunit_x, bunit_y, bunit_z) = (bx / bmag, by / bmag, bz / bmag);

        let p2 = Point::<T> {
            x: p1.x + dir * bunit_x * step,
            y: p1.y + dir * bunit_y * step,
            z: p1.z + dir * bunit_z * step,
        };
        Some(p2)
    } else {
        None
    }
}

pub fn step_bs<T, F>(optional_point: Option<Point<T>>, f: &F, dir: T) -> Option<Point<T>>
where
    T: PtrTrait + Sync + Send + Copy,
    F: Field<T> + Sync,
{
    fn get_bunit<T, F>(f: &F, p: Point<T>) -> Option<Point<T>>
    where
        T: PtrTrait,
        F: Field<T>,
    {
        let fields = f.get_fields_at(T::zero(), p.x, p.y, p.z)?;
        let bmag = (fields[0] * fields[0] + fields[1] * fields[1] + fields[2] * fields[2]).sqrt();
        if bmag == T::zero() {
            return None;
        }
        Some(Point {
            x: fields[0] / bmag,
            y: fields[1] / bmag,
            z: fields[2] / bmag,
        })
    }

    let p1 = optional_point?;
    let h_total = T::from(1e5)?;
    let k_max = 8;

    //Modified midpoint method
    let modified_midpoint = |total_step: T, n_substeps: usize| -> Option<Point<T>> {
        let h = total_step / T::from(n_substeps as f64)?;
        let mut y0 = p1;

        let b0 = get_bunit(f, y0)?;
        let mut y1 = Point {
            x: y0.x + dir * b0.x * h,
            y: y0.y + dir * b0.y * h,
            z: y0.z + dir * b0.z * h,
        };

        for _ in 1..n_substeps {
            let bi = get_bunit(f, y1)?;
            let y_next = Point {
                x: y0.x + dir * bi.x * T::from(2.0)? * h,
                y: y0.y + dir * bi.y * T::from(2.0)? * h,
                z: y0.z + dir * bi.z * T::from(2.0)? * h,
            };
            y0 = y1;
            y1 = y_next;
        }

        let bn = get_bunit(f, y1)?;
        Some(Point {
            x: T::from(0.5)? * (y0.x + y1.x + dir * bn.x * h),
            y: T::from(0.5)? * (y0.y + y1.y + dir * bn.y * h),
            z: T::from(0.5)? * (y0.z + y1.z + dir * bn.z * h),
        })
    };

    //Richardson Extrapolation
    let mut table = Vec::with_capacity(k_max);
    let n_seq = [2, 4, 6, 8, 10, 12, 14, 16];

    for k in 0..k_max {
        let current_estimate = modified_midpoint(h_total, n_seq[k])?;
        table.push(current_estimate);
    }
    let final_p = table.last()?;
    let rho = (final_p.x * final_p.x + final_p.y * final_p.y + final_p.z * final_p.z).sqrt();
    if rho < T::from(physical_constants::f64::EARTH_RE)? {
        None
    } else {
        Some(*final_p)
    }
}

pub fn step_lines<T, F>(p: &mut [Option<Point<T>>], f: &F, dir: T)
where
    T: PtrTrait + Sync + Send + Copy + num_traits::Zero,
    F: Field<T> + Sync,
{
    p.par_iter_mut().for_each(|point| {
        *point = step_euler(*point, f, dir);
        *point = step_bs(*point, f, dir);
    });
}

fn dump_snapshot(
    step: usize,
    seeds_positive: &[Option<Point<f64>>],
    seeds_negative: &[Option<Point<f64>>],
) -> std::io::Result<()> {
    let n = seeds_positive.len() + seeds_negative.len();
    let mut pop = ParticlePopulation::<f64>::new(n, 0.0_f64, 0.0_f64);
    for p in seeds_positive.iter() {
        if let Some(v) = p {
            pop.add_particle([v.x, v.y, v.z, 0.0_f64, 0.0_f64, 0.0_f64], true);
        }
    }
    for p in seeds_negative.iter() {
        if let Some(v) = p {
            pop.add_particle([v.x, v.y, v.z, 0.0_f64, 0.0_f64, 0.0_f64], true);
        }
    }
    let fname = format!("field_lines.{:07}.ptr", step);
    pop.save(&fname);
    Ok(())
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let args = Args::parse();

    if args.dump_every == 0 {
        eprintln!("--dump-every must be > 0");
        return Err(std::process::ExitCode::FAILURE);
    }

    let mut seeds_positive: Vec<Option<Point<f64>>> = if let Some(filename) = &args.input {
        let file = File::open(filename).expect("Failed to open input file");
        let reader = BufReader::new(file);
        reader
            .lines()
            .map(|line| line.expect("Could not read line from file"))
            .map(|line| {
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    let sub: Vec<f64> = trimmed
                        .split(',')
                        .map(|s| s.trim().parse::<f64>().expect("Parse error"))
                        .collect();

                    if sub.len() == 3 {
                        return Some(Point::<f64> {
                            x: sub[0],
                            y: sub[1],
                            z: sub[2],
                        });
                    }
                }
                None
            })
            .collect()
    } else {
        let mut rng = rand::rng();
        const RE: f64 = physical_constants::f64::EARTH_RE;
        (0..args.num_seeds)
            .map(|_| {
                let phi: f64 = rng.random_range(0.0..360.0);
                Some(Point::<f64> {
                    x: args.lshell * f64::cos(phi.to_radians()) * RE,
                    y: args.lshell * f64::sin(phi.to_radians()) * RE,
                    z: 0.0_f64,
                })
            })
            .collect()
    };

    let fields: Box<dyn Field<f64> + Sync + Send> = if args.vlsv.is_some() {
        Box::new(VlsvStaticField::<f64>::new(
            &args.vlsv.clone().unwrap(),
            [args.periodic_x, args.periodic_y, args.periodic_z],
        ))
    } else {
        Box::new(DipoleField::<f64>::new(8e15_f64))
    };

    let pb = ProgressBar::new(args.steps as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} steps ({eta})",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let mut seeds_negative = seeds_positive.clone();
    if let Err(e) = dump_snapshot(0, &seeds_positive, &seeds_negative) {
        eprintln!("Failed to dump initial snapshot: {}", e);
        return Err(std::process::ExitCode::FAILURE);
    }

    for step in 1..=args.steps {
        let keep_going_fwd = seeds_positive.iter().all(|x| x.is_some());
        let keep_going_bwd = seeds_negative.iter().all(|x| x.is_some());
        let stop = !keep_going_bwd && !keep_going_fwd;
        if stop {
            break;
        }
        if keep_going_fwd {
            step_lines(&mut seeds_positive, &fields, FWD);
        }
        if keep_going_bwd {
            step_lines(&mut seeds_negative, &fields, BWD);
        }
        if step % args.dump_every == 0 || step == args.steps {
            if let Err(e) = dump_snapshot(step, &seeds_positive, &seeds_negative) {
                eprintln!("Failed to dump snapshot at step {}: {}", step, e);
                return Err(std::process::ExitCode::FAILURE);
            }
        }
        pb.inc(1);
    }
    pb.finish_with_message("Done");
    Ok(std::process::ExitCode::SUCCESS)
}
