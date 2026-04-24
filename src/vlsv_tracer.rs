#![allow(dead_code)]
#![allow(non_snake_case)]

mod vlsv_reader;

use crate::mod_vlsv_tracing::*;
use crate::vlsv_reader::*;
use clap::Parser;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{Arc, Mutex};

#[derive(Parser, Debug)]
#[command(name = "vlsv_tracer", about = "Forward and backward particle tracer")]
struct Args {
    /// VLSV file (static field) or a directory (dynamic field)
    #[arg(short, long)]
    vlsv: Option<String>,

    /// Simulation start time in seconds
    #[arg(long)]
    tstart: Option<f64>,

    /// Minimum simulation time in seconds
    #[arg(long, default_value_t = 300.0)]
    tmin: f64,

    /// Maximum simulation time in seconds
    #[arg(long, default_value_t = 500.0)]
    tmax: f64,

    /// Output cadence in seconds
    #[arg(long, default_value_t = 1.0)]
    tout: f64,

    /// Trace backward in time
    #[arg(short, long, default_value_t = false)]
    backward: bool,

    /// Periodic boundary in X
    #[arg(long, default_value_t = false)]
    periodic_x: bool,

    /// Periodic boundary in Y
    #[arg(long, default_value_t = false)]
    periodic_y: bool,

    /// Periodic boundary in Z
    #[arg(long, default_value_t = false)]
    periodic_z: bool,

    /// Input txt file with particles
    #[arg(short, long)]
    input: Option<String>,

    /// Output file prefix
    #[arg(short, long, default_value = "state")]
    output: String,

    /// Number of particles to generate when no --input is provided
    #[arg(short, long, default_value_t = 1)]
    num_particles: usize,

    /// Initial kinetic energy in keV (used when generating particles)
    #[arg(short, long, default_value_t = 512.0)]
    energy_kev: f64,

    /// L-shell
    #[arg(short, long, default_value_t = 10.0)]
    lshell: f64,

    /// Skip saving files as in DRY RUN
    #[arg(short, long, default_value_t = false)]
    dry: bool,

    /// File buffer size in seconds for dynamic VLSV windowing
    #[arg(short, long, default_value_t = 10.0)]
    buffer_size: f64,
}

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
        let pr = Arc::clone(pop);
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

pub fn backtrace_population_cpu_adpt<T: PtrTrait, F: Field<T> + Sync>(
    pop: &mut Arc<Mutex<ParticlePopulation<T>>>,
    f: &F,
    time_span: T,
    actual_time: &mut T,
) {
    let n = pop.lock().unwrap().size();
    let mass = pop.lock().unwrap().mass;
    let charge = pop.lock().unwrap().charge;

    (0..n).into_par_iter().for_each(|i| {
        let pr = Arc::clone(pop);
        let mut particle = {
            let pop_ref = pr.lock().unwrap();
            pop_ref.get_temp_particle(i)
        };

        let mut dt_val = T::from(-1e-4).unwrap();
        boris_backtracing_adaptive(
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

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum SimulationKind {
    Other,
    Static,
    Dynamic,
}

fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

fn make_window(
    actual_time: f64,
    tout_signed: f64,
    buffer_size: f64,
    tmin: f64,
    tmax: f64,
    backward: bool,
) -> (f64, f64) {
    let eps = tout_signed.abs().max(1e-9);

    let (mut wmin, mut wmax) = if backward {
        (actual_time - buffer_size, actual_time + eps)
    } else {
        (actual_time - eps, actual_time + buffer_size)
    };

    wmin = clamp(wmin, tmin, tmax);
    wmax = clamp(wmax, tmin, tmax);

    if wmax <= wmin {
        wmax = (wmin + eps).min(tmax);
    }

    (wmin, wmax)
}

fn save_population(
    pop_arc: &Arc<Mutex<ParticlePopulation<f64>>>,
    output: &str,
    out_count: usize,
    dry: bool,
) {
    if !dry {
        let fname = format!("{}.{:07}.ptr", output, out_count);
        pop_arc.lock().unwrap().save(&fname);
    }
}

fn compute_requested_step(actual_time: f64, args: &Args) -> f64 {
    let tout_abs = args.tout.abs();

    if args.backward {
        let remaining = args.tmin - actual_time;
        remaining.max(-tout_abs)
    } else {
        let remaining = args.tmax - actual_time;
        remaining.min(tout_abs)
    }
}

fn clamp_dynamic_start_time(actual_time: &mut f64, loaded_tmin: f64, loaded_tmax: f64) {
    if *actual_time < loaded_tmin {
        println!(
            "Requested start time {:.12} is before loaded field range; shifting to {:.12}",
            *actual_time, loaded_tmin
        );
        *actual_time = loaded_tmin;
    }
    if *actual_time > loaded_tmax {
        println!(
            "Requested start time {:.12} is after loaded field range; shifting to {:.12}",
            *actual_time, loaded_tmax
        );
        *actual_time = loaded_tmax;
    }
}

fn clamp_step_to_loaded_range(
    actual_time: f64,
    dt: f64,
    loaded_tmin: f64,
    loaded_tmax: f64,
    backward: bool,
) -> f64 {
    if backward {
        let min_dt = loaded_tmin - actual_time;
        dt.max(min_dt)
    } else {
        let max_dt = loaded_tmax - actual_time;
        dt.min(max_dt)
    }
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let args = Args::parse();
    let periodic = [args.periodic_x, args.periodic_y, args.periodic_z];

    let sim_kind = if let Some(path) = &args.vlsv {
        let meta = std::fs::metadata(path).unwrap();
        if meta.file_type().is_file() {
            SimulationKind::Static
        } else {
            SimulationKind::Dynamic
        }
    } else {
        SimulationKind::Other
    };

    let mass = physical_constants::f64::PROTON_MASS;
    let charge = physical_constants::f64::PROTON_CHARGE;

    let default_start = if args.backward { args.tmax } else { args.tmin };
    let mut actual_time: f64 = args.tstart.unwrap_or(default_start);

    let mut pop = ParticlePopulation::<f64>::new(1024, mass, charge);

    if let Some(filename) = &args.input {
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
            args.num_particles,
            mass,
            charge,
            args.energy_kev,
            args.lshell * physical_constants::f64::EARTH_RE,
        );
    }

    let num_particles = pop.size();
    let mut pop_arc = Arc::new(Mutex::new(pop));
    let mut out_count: usize = 0;

    match sim_kind {
        SimulationKind::Dynamic => {
            let vlsv_dir = args
                .vlsv
                .as_ref()
                .expect("Dynamic simulation requires --vlsv directory");

            let initial_tout_signed = if args.backward {
                -args.tout.abs()
            } else {
                args.tout.abs()
            };

            let (mut win_tmin, mut win_tmax) = make_window(
                actual_time,
                initial_tout_signed,
                args.buffer_size,
                args.tmin,
                args.tmax,
                args.backward,
            );

            let mut fields =
                VlsvDynamicField::<f64>::new_partial(vlsv_dir, periodic, win_tmin, win_tmax);

            let (mut loaded_tmin, mut loaded_tmax) = fields.temporal_range();
            clamp_dynamic_start_time(&mut actual_time, loaded_tmin, loaded_tmax);

            if args.backward {
                while actual_time > args.tmin {
                    let mut dt = compute_requested_step(actual_time, &args);
                    dt =
                        clamp_step_to_loaded_range(actual_time, dt, loaded_tmin, loaded_tmax, true);

                    if dt >= 0.0 || (actual_time + dt) >= actual_time {
                        let (new_wmin, new_wmax) = make_window(
                            actual_time,
                            -args.tout.abs(),
                            args.buffer_size,
                            args.tmin,
                            args.tmax,
                            true,
                        );

                        if new_wmin == win_tmin && new_wmax == win_tmax {
                            break;
                        }

                        println!(
                            "Reloading dynamic field window -> [{:.12}, {:.12}]",
                            new_wmin, new_wmax
                        );

                        fields = VlsvDynamicField::<f64>::new_partial(
                            vlsv_dir, periodic, new_wmin, new_wmax,
                        );

                        win_tmin = new_wmin;
                        win_tmax = new_wmax;

                        (loaded_tmin, loaded_tmax) = fields.temporal_range();
                        clamp_dynamic_start_time(&mut actual_time, loaded_tmin, loaded_tmax);

                        dt = compute_requested_step(actual_time, &args);
                        dt = clamp_step_to_loaded_range(
                            actual_time,
                            dt,
                            loaded_tmin,
                            loaded_tmax,
                            true,
                        );

                        if dt >= 0.0 || (actual_time + dt) >= actual_time {
                            break;
                        }
                    }

                    let n_alive = pop_arc.lock().unwrap().count_alive();
                    println!(
                        "Tracing {} particles [{} alive] at t= {:.12} s, loaded range [{:.12}, {:.12}], request window [{:.12}, {:.12}]",
                        num_particles,
                        n_alive,
                        actual_time,
                        loaded_tmin,
                        loaded_tmax,
                        win_tmin,
                        win_tmax
                    );

                    backtrace_population_cpu_adpt(&mut pop_arc, &fields, dt, &mut actual_time);
                    save_population(&pop_arc, &args.output, out_count, args.dry);
                    out_count += 1;

                    if actual_time <= loaded_tmin + 1e-12 {
                        let (new_wmin, new_wmax) = make_window(
                            actual_time,
                            -args.tout.abs(),
                            args.buffer_size,
                            args.tmin,
                            args.tmax,
                            true,
                        );

                        if new_wmin != win_tmin || new_wmax != win_tmax {
                            println!(
                                "Reloading dynamic field window -> [{:.12}, {:.12}]",
                                new_wmin, new_wmax
                            );

                            fields = VlsvDynamicField::<f64>::new_partial(
                                vlsv_dir, periodic, new_wmin, new_wmax,
                            );
                            win_tmin = new_wmin;
                            win_tmax = new_wmax;
                            (loaded_tmin, loaded_tmax) = fields.temporal_range();
                            clamp_dynamic_start_time(&mut actual_time, loaded_tmin, loaded_tmax);
                        }
                    }
                }
            } else {
                while actual_time < args.tmax {
                    let mut dt = compute_requested_step(actual_time, &args);
                    dt = clamp_step_to_loaded_range(
                        actual_time,
                        dt,
                        loaded_tmin,
                        loaded_tmax,
                        false,
                    );

                    if dt <= 0.0 || (actual_time + dt) <= actual_time {
                        let (new_wmin, new_wmax) = make_window(
                            actual_time,
                            args.tout.abs(),
                            args.buffer_size,
                            args.tmin,
                            args.tmax,
                            false,
                        );

                        if new_wmin == win_tmin && new_wmax == win_tmax {
                            break;
                        }

                        println!(
                            "Reloading dynamic field window -> [{:.12}, {:.12}]",
                            new_wmin, new_wmax
                        );

                        fields = VlsvDynamicField::<f64>::new_partial(
                            vlsv_dir, periodic, new_wmin, new_wmax,
                        );

                        win_tmin = new_wmin;
                        win_tmax = new_wmax;

                        (loaded_tmin, loaded_tmax) = fields.temporal_range();
                        clamp_dynamic_start_time(&mut actual_time, loaded_tmin, loaded_tmax);

                        dt = compute_requested_step(actual_time, &args);
                        dt = clamp_step_to_loaded_range(
                            actual_time,
                            dt,
                            loaded_tmin,
                            loaded_tmax,
                            false,
                        );

                        if dt <= 0.0 || (actual_time + dt) <= actual_time {
                            break;
                        }
                    }

                    let n_alive = pop_arc.lock().unwrap().count_alive();
                    println!(
                        "Tracing {} particles [{} alive] at t= {:.12} s, loaded range [{:.12}, {:.12}], request window [{:.12}, {:.12}]",
                        num_particles,
                        n_alive,
                        actual_time,
                        loaded_tmin,
                        loaded_tmax,
                        win_tmin,
                        win_tmax
                    );

                    push_population_cpu_adpt(&mut pop_arc, &fields, dt, &mut actual_time);
                    save_population(&pop_arc, &args.output, out_count, args.dry);
                    out_count += 1;

                    if actual_time >= loaded_tmax - 1e-12 {
                        let (new_wmin, new_wmax) = make_window(
                            actual_time,
                            args.tout.abs(),
                            args.buffer_size,
                            args.tmin,
                            args.tmax,
                            false,
                        );

                        if new_wmin != win_tmin || new_wmax != win_tmax {
                            println!(
                                "Reloading dynamic field window -> [{:.12}, {:.12}]",
                                new_wmin, new_wmax
                            );

                            fields = VlsvDynamicField::<f64>::new_partial(
                                vlsv_dir, periodic, new_wmin, new_wmax,
                            );
                            win_tmin = new_wmin;
                            win_tmax = new_wmax;
                            (loaded_tmin, loaded_tmax) = fields.temporal_range();
                            clamp_dynamic_start_time(&mut actual_time, loaded_tmin, loaded_tmax);
                        }
                    }
                }
            }
        }

        SimulationKind::Static => {
            let vlsv_file = args
                .vlsv
                .as_ref()
                .expect("Static simulation requires --vlsv file");

            let fields = VlsvStaticField::<f64>::new(vlsv_file, periodic);

            if args.backward {
                while actual_time > args.tmin {
                    let dt = compute_requested_step(actual_time, &args);
                    if dt >= 0.0 {
                        break;
                    }

                    let n_alive = pop_arc.lock().unwrap().count_alive();
                    println!(
                        "Tracing {} particles [{} alive] at t= {:.12} s",
                        num_particles, n_alive, actual_time
                    );

                    backtrace_population_cpu_adpt(&mut pop_arc, &fields, dt, &mut actual_time);
                    save_population(&pop_arc, &args.output, out_count, args.dry);
                    out_count += 1;
                }
            } else {
                while actual_time < args.tmax {
                    let dt = compute_requested_step(actual_time, &args);
                    if dt <= 0.0 {
                        break;
                    }

                    let n_alive = pop_arc.lock().unwrap().count_alive();
                    println!(
                        "Tracing {} particles [{} alive] at t= {:.12} s",
                        num_particles, n_alive, actual_time
                    );

                    push_population_cpu_adpt(&mut pop_arc, &fields, dt, &mut actual_time);
                    save_population(&pop_arc, &args.output, out_count, args.dry);
                    out_count += 1;
                }
            }
        }

        SimulationKind::Other => {
            let fields = DipoleField::<f64>::new(8e15_f64);

            if args.backward {
                while actual_time > args.tmin {
                    let dt = compute_requested_step(actual_time, &args);
                    if dt >= 0.0 {
                        break;
                    }

                    let n_alive = pop_arc.lock().unwrap().count_alive();
                    println!(
                        "Tracing {} particles [{} alive] at t= {:.12} s",
                        num_particles, n_alive, actual_time
                    );

                    backtrace_population_cpu_adpt(&mut pop_arc, &fields, dt, &mut actual_time);
                    save_population(&pop_arc, &args.output, out_count, args.dry);
                    out_count += 1;
                }
            } else {
                while actual_time < args.tmax {
                    let dt = compute_requested_step(actual_time, &args);
                    if dt <= 0.0 {
                        break;
                    }

                    let n_alive = pop_arc.lock().unwrap().count_alive();
                    println!(
                        "Tracing {} particles [{} alive] at t= {:.12} s",
                        num_particles, n_alive, actual_time
                    );

                    push_population_cpu_adpt(&mut pop_arc, &fields, dt, &mut actual_time);
                    save_population(&pop_arc, &args.output, out_count, args.dry);
                    out_count += 1;
                }
            }
        }
    }

    Ok(std::process::ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical_constants::f64::*;
    use std::f64::consts::PI;

    const ANGLE_TOL_DEG: f64 = 0.5;
    const PERCENT_TOLERANCE: f64 = 0.1;

    fn get_analytical_values(vperp: f64, bmag: f64, q: f64, m: f64) -> (f64, f64, f64) {
        let omega_c = (q.abs() * bmag) / m;
        let period = 2.0 * PI / omega_c;
        let radius = vperp / omega_c;
        (omega_c, period, radius)
    }

    #[test]
    fn test_forward_uniform_field_accuracy() {
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

            let (_omega_c, t_gyro, r_larmor) =
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

    #[test]
    fn test_backward_uniform_field_accuracy() {
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

            let (_omega_c, t_gyro, r_larmor) =
                get_analytical_values(v_mag, b_strength, charge, mass);
            let mut actual_time: f64 = 0.0;
            let mut pop_arc = Arc::new(Mutex::new(pop));

            let num_steps = 100;
            let dt_sub = t_gyro / (num_steps as f64);

            for _ in 0..num_steps {
                backtrace_population_cpu_adpt(&mut pop_arc, &field, -dt_sub, &mut actual_time);
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
