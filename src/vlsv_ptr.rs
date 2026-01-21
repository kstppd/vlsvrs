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
const TMAX: f64 = 300.0;
const DEFAULT_VLSV: &str = "/home/kstppd/Desktop/bulk.0000601.vlsv";

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
        boris_adaptive(
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
    let fields = VlsvStaticField::<f64>::new(&String::from(DEFAULT_VLSV), [false, false, true]);
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_constants::f64::*;
    use std::f64::consts::PI;

    const ANGLE_TOL_DEG: f64 = 0.5; // Degree tolerance for a full rotation
    const PERCENT_TOLERANCE: f64 = 0.1;

    fn get_analytical_values(vperp: f64, bmag: f64, q: f64, m: f64) -> (f64, f64, f64) {
        let omega_c = (q.abs() * bmag) / m;
        let period = 2.0 * PI / omega_c;
        let radius = vperp / omega_c;
        (omega_c, period, radius)
    }

    #[test]
    fn test_batch_uniform_field_accuracy() {
        let b_strength = 50e-9;
        let mass = PROTON_MASS;
        let charge = PROTON_CHARGE;
        let field = UniformField::new(b_strength, 2);

        println!(
            "\n{:>10} | {:>15} | {:>15} | {:>15}",
            "Energy(keV)", "Angle Err(deg)", "Energy Err%", "G.Center Err%"
        );
        println!("{}", "-".repeat(65));

        for i in [1, 10, 50, 100, 256, 512, 1024] {
            let energy_kev = i as f64;
            let ke_j = energy_kev * 1.0e3 * EV_TO_JOULE;
            let v_mag = (2.0 * ke_j / mass).sqrt();

            let p0 = [EARTH_RE, 0.0, 0.0];
            let v0 = [0.0, v_mag, 0.0];

            let mut pop = ParticlePopulation::<f64>::new(1, mass, charge);
            pop.add_particle([p0[0], p0[1], p0[2], v0[0], v0[1], v0[2]], true);

            let (omega_c, t_gyro, r_larmor) =
                get_analytical_values(v_mag, b_strength, charge, mass);
            let mut actual_time: f64 = 0.0;
            let mut pop_arc = Arc::new(Mutex::new(pop));

            let num_steps = 100;
            let dt_sub = t_gyro / (num_steps as f64);

            for _ in 0..num_steps {
                push_population_cpu_adpt(&mut pop_arc, &field, dt_sub, &mut actual_time);
            }

            let locked = pop_arc.lock().unwrap();
            let pf = [locked.x[0], locked.y[0], locked.z[0]];
            let vf = [locked.vx[0], locked.vy[0], locked.vz[0]];
            let center_x = p0[0] + r_larmor;
            let center_y = 0.0;
            let dx_start = p0[0] - center_x;
            let dy_start = p0[1] - center_y;
            let phi_start = dy_start.atan2(dx_start).to_degrees();
            let dx_final = pf[0] - center_x;
            let dy_final = pf[1] - center_y;
            let phi_final = dy_final.atan2(dx_final).to_degrees();
            let mut delta_phi = (phi_final - phi_start).abs();
            if delta_phi < 180.0 {
                delta_phi = 360.0 - delta_phi;
            }
            let angle_err = (delta_phi - 360.0).abs();

            let v_mag_final = (vf[0].powi(2) + vf[1].powi(2) + vf[2].powi(2)).sqrt();
            let energy_err_pct = ((v_mag_final - v_mag).abs() / v_mag) * 100.0;
            let omega_sign = (charge * b_strength) / mass;
            let calc_center_x = pf[0] + (vf[1] / omega_sign);
            let center_err_pct = ((calc_center_x - center_x).abs() / r_larmor) * 100.0;
            println!(
                "{:>10.1} | {:>15.2e} | {:>15.2e} | {:>15.2e}",
                energy_kev, angle_err, energy_err_pct, center_err_pct
            );
            assert!(
                angle_err < ANGLE_TOL_DEG,
                "Angle error too high at {} keV",
                energy_kev
            );
            assert!(
                energy_err_pct < 1e-10,
                "Energy conservation failed at {} keV",
                energy_kev
            );
            assert!(
                center_err_pct < PERCENT_TOLERANCE,
                "G.Center drift failed at {} keV",
                energy_kev
            );
        }
    }
}
