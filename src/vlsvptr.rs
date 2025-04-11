mod constants;
mod tracer_fields;
mod tracer_particles;
mod vlsv_reader;
use crate::constants::physical_constants;
use crate::tracer_particles::tracer_particles::ParticlePopulation;
use core::panic;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use tracer_fields::vlsv_reader::{DipoleField, Field};
use tracer_fields::vlsv_reader::{PtrTrait, VlsvStaticField};

fn boris<T: PtrTrait>(
    p: &mut tracer_particles::tracer_particles::Particle<T>,
    e: &[T],
    b: &[T],
    dt: T,
    m: T,
    c: T,
) {
    // println!("b={:?},e={:?}", b, e);
    // panic!();
    let mut v_minus: [T; 3] = [T::zero(); 3];
    let mut v_prime: [T; 3] = [T::zero(); 3];
    let mut v_plus: [T; 3] = [T::zero(); 3];
    let mut t: [T; 3] = [T::zero(); 3];
    let mut s: [T; 3] = [T::zero(); 3];
    let g = tracer_particles::tracer_particles::gamma(p.vx, p.vy, p.vz);
    let cm = c / m;
    t[0] = cm * b[0] * T::from(0.5).unwrap() * dt / g;
    t[1] = cm * b[1] * T::from(0.5).unwrap() * dt / g;
    t[2] = cm * b[2] * T::from(0.5).unwrap() * dt / g;

    let t_mag2 = t[0].powi(2) + t[1].powi(2) + t[2].powi(2);

    s[0] = T::from(2.0).unwrap() * t[0] / (T::one() + t_mag2);
    s[1] = T::from(2.0).unwrap() * t[1] / (T::one() + t_mag2);
    s[2] = T::from(2.0).unwrap() * t[2] / (T::one() + t_mag2);

    v_minus[0] = p.vx + cm * e[0] * T::from(0.5).unwrap() * dt;
    v_minus[1] = p.vy + cm * e[1] * T::from(0.5).unwrap() * dt;
    v_minus[2] = p.vz + cm * e[2] * T::from(0.5).unwrap() * dt;

    v_prime[0] = v_minus[0] + v_minus[1] * t[2] - v_minus[2] * t[1];
    v_prime[1] = v_minus[1] - v_minus[0] * t[2] + v_minus[2] * t[0];
    v_prime[2] = v_minus[2] + v_minus[0] * t[1] - v_minus[1] * t[0];

    v_plus[0] = v_minus[0] + v_prime[1] * s[2] - v_prime[2] * s[1];
    v_plus[1] = v_minus[1] - v_prime[0] * s[2] + v_prime[2] * s[0];
    v_plus[2] = v_minus[2] + v_prime[0] * s[1] - v_prime[1] * s[0];

    p.vx = v_plus[0] + cm * e[0] * T::from(0.5).unwrap() * dt;
    p.vy = v_plus[1] + cm * e[1] * T::from(0.5).unwrap() * dt;
    p.vz = v_plus[2] + cm * e[2] * T::from(0.5).unwrap() * dt;

    p.x = p.x + p.vx * dt;
    p.y = p.y + p.vy * dt;
    p.z = p.z + p.vz * dt;
}

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

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let file = String::from("/home/kstppd/Desktop/bulk_with_fg_10.0000109.vlsv");
    // let fields1 = DipoleField::<f32>::new(8e15_f32);
    let fields1 = VlsvStaticField::<f32>::new(&file);
    let mut p = Arc::new(Mutex::new(
        ParticlePopulation::<f32>::new_with_energy_at_Lshell(
            1,
            physical_constants::f32::PROTON_MASS,
            physical_constants::f32::PROTON_CHARGE,
            512_f32,
            10_f32 * physical_constants::f32::EARTH_RE,
        ),
    ));
    let t = p.try_lock().unwrap().get_temp_particle(0);
    println!("{:?}", t);
    for i in 0..10000 {
        push_population_cpu(&mut p, &fields1, 0.25_f32);
        let name = format!("state.{:07}.ptr", i);
        p.try_lock().unwrap().save(name.as_str());
    }
    let t = p.try_lock().unwrap().get_temp_particle(0);
    println!("{:?}", t);
    Ok(std::process::ExitCode::SUCCESS)
}
