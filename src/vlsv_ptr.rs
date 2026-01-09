#![allow(dead_code)]
#![allow(non_snake_case)]
mod vlsv_reader;
use crate::mod_vlsv_tracing::*;
use crate::vlsv_reader::*;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};

const TOUT: f64 = 0.25;
const TMAX: f64 = 25.0;
const DEFAULT_VLSV: &str = "/home/kstppd/Desktop/bulk_with_fg_10.0000109.vlsv";

pub fn push_population_cpu_adpt<T: PtrTrait, F: Field<T> + Sync>(
    pop: &mut Arc<Mutex<ParticlePopulation<T>>>,
    f: &F,
    time_span: T,
    actual_time: &mut T,
) {
    let n = pop.lock().unwrap().size();
    let mass = pop.lock().unwrap().mass;
    let charge = pop.lock().unwrap().charge;

    (0..n).into_par_iter().for_each(|i| {
        let pr = Arc::clone(&pop);
        let mut particle = {
            let pop_ref = pr.lock().unwrap();
            pop_ref.get_temp_particle(i)
        };

        let mut dt_val = T::from(1e-4).unwrap();
        borris_adaptive(
            &mut particle,
            f,
            &mut dt_val,
            *actual_time,
            *actual_time + time_span,
            mass,
            charge,
        );

        let mut pop_ref = pr.lock().unwrap();
        pop_ref.take_temp_particle(&particle, i);
    });

    *actual_time = *actual_time + time_span;
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let args: Vec<String> = env::args().collect();
    let fields = VlsvStaticField::<f64>::new(&String::from(DEFAULT_VLSV), [false, false, false]);
    let mass = physical_constants::f64::PROTON_MASS;
    let charge = physical_constants::f64::PROTON_CHARGE;
    let mut actual_time: f64 = 0.0;

    let mut pop = ParticlePopulation::<f64>::new(1024, mass, charge);

    if args.len() > 1 {
        let filename = &args[1];
        let file = File::open(filename).expect("Failed to open input file");
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            let sub: Vec<f64> = line
                .split(',')
                .map(|s| s.trim().parse::<f64>().expect("Parse error"))
                .collect();

            if sub.len() == 7 {
                actual_time = sub[0];
                pop.add_particle([sub[1], sub[2], sub[3], sub[4], sub[5], sub[6]], true);
            }
        }
    } else {
        pop = ParticlePopulation::<f64>::new_with_energy_at_Lshell(
            1,
            mass,
            charge,
            512_f64,
            10_f64 * physical_constants::f64::EARTH_RE,
        );
    }

    let num_particles = pop.size();
    let mut pop_arc = Arc::new(Mutex::new(pop));
    let mut out_count = 0;

    while actual_time < TMAX {
        println!(
            "Tracing {} particles at t= {:.3} s",
            num_particles, actual_time
        );

        push_population_cpu_adpt(&mut pop_arc, &fields, TOUT, &mut actual_time);

        let fname = format!("state.{:07}.ptr", out_count);
        let locked = pop_arc.lock().unwrap();
        locked.save(&fname);
        out_count += 1;
    }

    Ok(std::process::ExitCode::SUCCESS)
}
