#![allow(dead_code)]
#![allow(non_snake_case)]
mod constants;
mod vlsv_reader;
use crate::constants::physical_constants;
use crate::vlsv_reader::*;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

fn push_population_cpu<T: PtrTrait, F: Field<T> + std::marker::Sync>(
    pop: &mut Arc<Mutex<ParticlePopulation<T>>>,
    f: &F,
    time_span: T,
) {
    let n = pop.try_lock().unwrap().size();
    let mass = pop.try_lock().unwrap().mass;
    let charge = pop.try_lock().unwrap().charge;
    (0..n).into_par_iter().for_each(|i| {
        let pr = Arc::clone(&pop);
        let mut time: T = T::zero();
        let dt = T::from(5e-7).unwrap();
        while time < time_span {
            let mut particle = pr.lock().unwrap().get_temp_particle(i);
            let fields = f
                .get_fields_at(time, particle.x, particle.y, particle.z)
                .unwrap();
            time = time + dt;
            boris(
                &mut particle,
                &fields[3..6],
                &fields[0..3],
                dt,
                mass,
                charge,
            );
            pr.lock().unwrap().take_temp_particle(&particle, i);
        }
    });
}

fn push_population_cpu_adpt<T: PtrTrait, F: Field<T> + std::marker::Sync>(
    pop: &mut Arc<Mutex<ParticlePopulation<T>>>,
    f: &F,
    time_span: T,
) {
    let n = pop.try_lock().unwrap().size();
    let mass = pop.try_lock().unwrap().mass;
    let charge = pop.try_lock().unwrap().charge;
    (0..n).into_par_iter().for_each(|i| {
        let pr = Arc::clone(&pop);
        let dt = T::from(5e-7).unwrap();
        let mut particle = pr.lock().unwrap().get_temp_particle(i);
        let mut _dt = dt;
        borris_adaptive(
            &mut particle,
            f,
            &mut _dt,
            T::from(0.0).unwrap(),
            time_span,
            mass,
            charge,
        );
        pr.lock().unwrap().take_temp_particle(&particle, i);
    });
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let _file = String::from("/home/kstppd/Desktop/bulk_with_fg_10.0000109.vlsv");
    let fields1 = DipoleField::<f64>::new(8e15_f64);
    // let fields1 = VlsvStaticField::<f64>::new(&file);
    let mut p = Arc::new(Mutex::new(
        ParticlePopulation::<f64>::new_with_energy_at_Lshell(
            1,
            physical_constants::f64::PROTON_MASS,
            physical_constants::f64::PROTON_CHARGE,
            512_f64,
            10_f64 * physical_constants::f64::EARTH_RE,
        ),
    ));
    let t = p.try_lock().unwrap().get_temp_particle(0);
    println!("{:?}", t);
    for i in 0..100 {
        // push_population_cpu(&mut p, &fields1, 0.25_f64);
        push_population_cpu_adpt(&mut p, &fields1, 0.25_f64);
        let name = format!("state.{:07}.ptr", i);
        p.try_lock().unwrap().save(name.as_str());
    }
    let t = p.try_lock().unwrap().get_temp_particle(0);
    println!("{:?}", t);
    Ok(std::process::ExitCode::SUCCESS)
}
