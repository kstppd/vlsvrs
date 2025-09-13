#![allow(dead_code)]
#![allow(non_snake_case)]
mod vlsv_reader;
use crate::mod_vlsv_tracing::*;
use crate::vlsv_reader::*;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

pub fn push_gc_population_cpu_adpt<T: PtrTrait, F: Field<T> + Sync>(
    pop: &mut Arc<Mutex<GCPopulation<T>>>,
    f: &F,
    time_span: T,
    actual_time: &mut T,
) {
    let n = pop.lock().unwrap().size();
    let mass = pop.lock().unwrap().mass;
    let charge = pop.lock().unwrap().charge;

    (0..n).into_par_iter().for_each(|i| {
        let pr = Arc::clone(&pop);
        let mut gc = {
            let pop_ref = pr.lock().unwrap();
            pop_ref.particles[i].clone()
        };

        let mut dt = gc.dt;
        gc_adaptive(
            &mut gc,
            f,
            &mut dt,
            *actual_time,
            *actual_time + time_span,
            mass,
            charge,
        );
        gc.dt = dt;
        let mut pop_ref = pr.lock().unwrap();
        pop_ref.particles[i] = gc;
    });
    *actual_time = *actual_time + time_span;
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let fields = VlsvDynamicField::<f64>::new(&String::from(
        "/wrk-vakka/group/spacephysics/vlasiator/2D/AID/bulk/",
    ));
    let mass = physical_constants::f64::PROTON_MASS;
    let charge = physical_constants::f64::PROTON_CHARGE;
    let energy_kev = 112.0;
    let pitch_angle_deg: f64 = 90.0;

    let ke_joules = energy_kev * 1.0e3 * physical_constants::f64::EV_TO_JOULE;
    let v = (2.0 * ke_joules / mass).sqrt();
    let pitch = pitch_angle_deg.to_radians();
    let vpar = v * pitch.cos();
    let vperp = v * pitch.sin();
    let mut actual_time: f64 = 0.0;

    let mut pop = GCPopulation::new(mass, charge);
    for L in 8..=10 {
        let r = (L as f64) * physical_constants::f64::EARTH_RE;
        let fields_here = fields.get_fields_at(actual_time, r, 0.0, 0.0).unwrap();
        let bmag = mag(fields_here[0], fields_here[1], fields_here[2]);
        let mu = 0.5 * mass * vperp * vperp / bmag;

        pop.add(GuidingCenter2D {
            x: r,
            y: 0.0,
            vpar,
            mu,
            alive: true,
            dt: 1e-2,
        });
    }
    let mut pop = Arc::new(Mutex::new(pop));
    for out in 0..500 {
        push_gc_population_cpu_adpt(&mut pop, &fields, 5.0, &mut actual_time);
        let fname = format!("state.{:07}.ptr", out);
        let locked = pop.lock().unwrap();
        locked.save(&fname);
    }
    Ok(std::process::ExitCode::SUCCESS)
}
