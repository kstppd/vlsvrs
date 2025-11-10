#![allow(dead_code)]
#![allow(non_snake_case)]
mod vlsv_reader;
use crate::mod_vlsv_tracing::*;
use crate::physical_constants::f64::*;
use crate::vlsv_reader::*;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};
//Configure these
const TOUT: f64 = 5.0; //output file cadence in seconds
const TMAX: f64 = 700.0; // run until we hit this
const VLSV_DIR: &str = "/wrk-vakka/group/spacephysics/vlasiator/2D/AID/bulk/";

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
    let args: Vec<String> = env::args().collect();
    let fields = VlsvDynamicField::<f64>::new(VLSV_DIR);
    let mass = physical_constants::f64::PROTON_MASS;
    let charge = physical_constants::f64::PROTON_CHARGE;
    let mut actual_time: f64 = 0.0;
    let mut pop = GCPopulation::new(mass, charge);

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
                .map(|s| s.trim().parse::<f64>().expect("Nan ???"))
                .collect();

            if sub.len() != 7 {
                panic!("Wrong format: {}", line);
            }
            let (time, x, y, z, vx, vy, vz) =
                (sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6]);
            actual_time = time;
            let fields_here = fields
                .get_fields_at(actual_time, x, y, z)
                .expect("Could not get fields during initialization");
            let bmag = mag(fields_here[0], fields_here[1], fields_here[2]);
            let vperp2 = vx * vx + vy * vy;
            let mu = 0.5 * mass * vperp2 / bmag;
            pop.add(GuidingCenter2D {
                x,
                y,
                vpar: vz,
                vperp: vperp2.sqrt(),
                mu,
                alive: true,
                dt: 1e-2,
            });
        }
    } else {
        //Test-Default
        let energy_kev = 112.0;
        let pitch_angle_deg: f64 = 90.0;
        let ke_joules = energy_kev * 1.0e3 * physical_constants::f64::EV_TO_JOULE;
        let v = (2.0 * ke_joules / mass).sqrt();
        let pitch = pitch_angle_deg.to_radians();
        let vpar = v * pitch.cos();
        let vperp = v * pitch.sin();

        for L in 8..=10 {
            let r = (L as f64) * physical_constants::f64::EARTH_RE;
            let fields_here = fields.get_fields_at(actual_time, r, 0.0, 0.0).unwrap();
            let bmag = mag(fields_here[0], fields_here[1], fields_here[2]);
            let mu = 0.5 * mass * vperp * vperp / bmag;

            pop.add(GuidingCenter2D {
                x: r,
                y: 0.0,
                vpar,
                vperp,
                mu,
                alive: true,
                dt: 1e-2,
            });
        }
    }

    let len = pop.particles.len();
    let mut pop = Arc::new(Mutex::new(pop));
    let mut out = 0;
    while actual_time < TMAX {
        println!("Tracing {} particles at t= {:02} s", len, actual_time);
        push_gc_population_cpu_adpt(&mut pop, &fields, TOUT, &mut actual_time);
        let fname = format!("state.{:07}.ptr", out);
        let locked = pop.lock().unwrap();
        locked.save(&fname);
        out = out + 1;
    }
    Ok(std::process::ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    use std::sync::{Arc, Mutex};

    const PERIOD_TOL_REL: f64 = 0.01;
    const PHASE_TOL_DEG: f64 = 10.0;

    #[inline]
    fn wrap_deg(a: f64) -> f64 {
        let mut x = a % 360.0;
        if x < 0.0 {
            x += 360.0;
        }
        x
    }

    #[inline]
    fn drift_period(vpar: f64, vperp: f64, r: f64, b: f64, q: f64, m: f64) -> (f64, f64) {
        let omega_d = m * (1.5 * vperp * vperp + vpar * vpar) / (q * b * r * r);
        let td = 2.0 * PI / omega_d.abs();
        (omega_d, td)
    }

    fn get_case(fields: &DipoleField<f64>, energy_kev: f64, L: f64) -> (GCPopulation<f64>, f64) {
        let mass = PROTON_MASS;
        let charge = PROTON_CHARGE;

        let ke_j = energy_kev * 1.0e3 * EV_TO_JOULE;
        let v = (2.0 * ke_j / mass).sqrt();
        let vpar = 0.0;
        let vperp = v;

        let r = L * EARTH_RE;
        let f = fields.get_fields_at(0.0, r, 0.0, 0.0).unwrap();
        let bmag = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
        let mu = 0.5 * mass * vperp * vperp / bmag;

        let mut pop = GCPopulation::new(mass, charge);
        pop.add(GuidingCenter2D {
            x: r,
            y: 0.0,
            vpar,
            vperp,
            mu,
            alive: true,
            dt: 1e-2,
        });
        (pop, bmag)
    }

    fn drift_test(L: f64) -> bool {
        let fields = DipoleField::<f64>::new(8e15_f64);
        let energies_kev: [f64; _] = [75.0, 90.0, 100.0, 200.0, 300.0, 400.0];
        struct Row {
            e_kev: f64,
            td_ana: f64,
            t_sim: f64,
            rel_err: f64,
            pass: bool,
        }
        let mut rows: Vec<Row> = Vec::with_capacity(energies_kev.len());
        let mut all_ok = true;
        for &e_kev in &energies_kev {
            let (pop0, bmag) = get_case(&fields, e_kev, L);
            let p0 = pop0.particles.first().unwrap().clone();
            let r = (p0.x.hypot(p0.y)).abs();
            let (omega_d, td) =
                drift_period(p0.vpar, p0.vperp, r, bmag, PROTON_CHARGE, PROTON_MASS);
            let mut actual_time: f64 = 0.0;
            let mut pop_arc = Arc::new(Mutex::new(pop0));
            while actual_time < td {
                push_gc_population_cpu_adpt(&mut pop_arc, &fields, TOUT, &mut actual_time);
            }
            let p_end = pop_arc.lock().unwrap().particles.first().unwrap().clone();
            let t_sim = actual_time;
            let rel_err = (t_sim - td).abs() / td.max(1.0);
            let phi_sim_deg = wrap_deg(p_end.y.atan2(p_end.x).to_degrees());
            let phi_ana_deg = wrap_deg((omega_d * t_sim).to_degrees());
            let mut dphi = (phi_sim_deg - phi_ana_deg).abs();
            if dphi > 180.0 {
                dphi = 360.0 - dphi;
            }

            let pass = rel_err <= PERIOD_TOL_REL && dphi <= PHASE_TOL_DEG;
            all_ok &= pass;

            rows.push(Row {
                e_kev: e_kev,
                td_ana: td,
                t_sim,
                rel_err,
                pass,
            });
        }

        eprintln!("\nDRIFTS  (Dipole, R={} Re, pitch=90°):", L);
        eprintln!(
            "{:>7} | {:>10} | {:>10} | {:>9} | {}",
            "keV", "T_theory", "T_simulated", "error %", "Status"
        );
        eprintln!("{}", "-".repeat(64));
        for r in &rows {
            eprintln!(
                "{:7.1} | {:10.3} | {:10.3} | {:9.3} | {}",
                r.e_kev,
                r.td_ana,
                r.t_sim,
                100.0 * r.rel_err,
                if r.pass { "yes" } else { "NO " }
            );
        }
        return all_ok;
    }
    #[test]
    fn test() {
        [4.0, 8.0, 12.0, 30.0].par_iter().for_each(|&L| {
            eprintln!("Testing L={}", L);
            assert!(drift_test(L), "FATAL::TEST AT L {} FAILED", L);
        });
    }
}
