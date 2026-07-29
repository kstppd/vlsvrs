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

const TIME_EPS: f64 = 1.0e-10;

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
    #[arg(long, default_value_t = 10.0)]
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

fn validate_args(args: &Args) -> Result<(), String> {
    if !args.tmin.is_finite() || !args.tmax.is_finite() || args.tmin >= args.tmax {
        return Err(format!(
            "Invalid time interval: tmin={} and tmax={}",
            args.tmin, args.tmax
        ));
    }

    if !args.tout.is_finite() || args.tout <= 0.0 {
        return Err(format!(
            "tout must be finite and positive, got {}",
            args.tout
        ));
    }

    if !args.buffer_size.is_finite() || args.buffer_size <= 0.0 {
        return Err(format!(
            "buffer_size must be finite and positive, got {}",
            args.buffer_size
        ));
    }

    Ok(())
}

#[inline]
fn trace_finished(actual_time: f64, args: &Args) -> bool {
    if args.backward {
        actual_time <= args.tmin + TIME_EPS
    } else {
        actual_time >= args.tmax - TIME_EPS
    }
}

fn scheduled_target(schedule_origin: f64, schedule_step: u64, args: &Args) -> f64 {
    let offset = schedule_step as f64 * args.tout.abs();
    if args.backward {
        (schedule_origin - offset).max(args.tmin)
    } else {
        (schedule_origin + offset).min(args.tmax)
    }
}

fn make_dynamic_window(
    current_time: f64,
    target_time: f64,
    tout: f64,
    buffer_size: f64,
    backward: bool,
) -> (f64, f64) {
    let guard = tout.abs().max(1.0e-9);
    let step_size = (target_time - current_time).abs();
    let span = buffer_size.max(step_size + guard);

    if backward {
        (current_time - span, current_time + guard)
    } else {
        (current_time - guard, current_time + span)
    }
}

#[inline]
fn range_contains_step(
    loaded_tmin: f64,
    loaded_tmax: f64,
    current_time: f64,
    target_time: f64,
) -> bool {
    let step_tmin = current_time.min(target_time);
    let step_tmax = current_time.max(target_time);

    loaded_tmin <= step_tmin + TIME_EPS && loaded_tmax + TIME_EPS >= step_tmax
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

fn print_trace_status(
    pop_arc: &Arc<Mutex<ParticlePopulation<f64>>>,
    num_particles: usize,
    actual_time: f64,
    loaded_range: Option<(f64, f64)>,
    request_window: Option<(f64, f64)>,
) {
    let n_alive = pop_arc.lock().unwrap().count_alive();

    match (loaded_range, request_window) {
        (Some((loaded_tmin, loaded_tmax)), Some((win_tmin, win_tmax))) => println!(
            "Tracing {} particles [{} alive] at t= {:.12} s, loaded range [{:.12}, {:.12}], request window [{:.12}, {:.12}]",
            num_particles, n_alive, actual_time, loaded_tmin, loaded_tmax, win_tmin, win_tmax
        ),
        _ => println!(
            "Tracing {} particles [{} alive] at t= {:.12} s",
            num_particles, n_alive, actual_time
        ),
    }
}

fn advance_population<F: Field<f64> + Sync>(
    pop_arc: &mut Arc<Mutex<ParticlePopulation<f64>>>,
    fields: &F,
    actual_time: &mut f64,
    target_time: f64,
    backward: bool,
) {
    let dt = target_time - *actual_time;

    if backward {
        debug_assert!(dt < 0.0);
        backtrace_population_cpu_adpt(pop_arc, fields, dt, actual_time);
    } else {
        debug_assert!(dt > 0.0);
        push_population_cpu_adpt(pop_arc, fields, dt, actual_time);
    }

    *actual_time = target_time;
}

fn run_fixed_field<F: Field<f64> + Sync>(
    args: &Args,
    fields: &F,
    pop_arc: &mut Arc<Mutex<ParticlePopulation<f64>>>,
    num_particles: usize,
    actual_time: &mut f64,
    out_count: &mut usize,
) {
    let schedule_origin = *actual_time;
    let mut schedule_step: u64 = 1;

    while !trace_finished(*actual_time, args) {
        let target_time = scheduled_target(schedule_origin, schedule_step, args);
        let dt = target_time - *actual_time;

        if (args.backward && dt >= 0.0) || (!args.backward && dt <= 0.0) {
            break;
        }

        print_trace_status(pop_arc, num_particles, *actual_time, None, None);
        advance_population(pop_arc, fields, actual_time, target_time, args.backward);
        save_population(pop_arc, &args.output, *out_count, args.dry);

        *out_count += 1;
        schedule_step += 1;
    }
}

fn run_dynamic_field(
    args: &Args,
    vlsv_dir: &str,
    periodic: [bool; 3],
    pop_arc: &mut Arc<Mutex<ParticlePopulation<f64>>>,
    num_particles: usize,
    actual_time: &mut f64,
    out_count: &mut usize,
) {
    if trace_finished(*actual_time, args) {
        return;
    }

    let schedule_origin = *actual_time;
    let mut schedule_step: u64 = 1;
    let first_target = scheduled_target(schedule_origin, schedule_step, args);

    let (mut win_tmin, mut win_tmax) = make_dynamic_window(
        *actual_time,
        first_target,
        args.tout,
        args.buffer_size,
        args.backward,
    );

    let mut fields = VlsvDynamicField::<f64>::new_partial(vlsv_dir, periodic, win_tmin, win_tmax);
    let (mut loaded_tmin, mut loaded_tmax) = fields.temporal_range();

    if !range_contains_step(loaded_tmin, loaded_tmax, *actual_time, first_target) {
        panic!(
            "Loaded VLSV range [{loaded_tmin:.12}, {loaded_tmax:.12}] does not bracket the first exact tracer step [{:.12}, {first_target:.12}] requested through [{win_tmin:.12}, {win_tmax:.12}]. Do not shift actual_time to a file timestamp; load bracketing snapshots instead.",
            *actual_time
        );
    }

    while !trace_finished(*actual_time, args) {
        let target_time = scheduled_target(schedule_origin, schedule_step, args);
        let dt = target_time - *actual_time;

        if (args.backward && dt >= 0.0) || (!args.backward && dt <= 0.0) {
            break;
        }

        if !range_contains_step(loaded_tmin, loaded_tmax, *actual_time, target_time) {
            (win_tmin, win_tmax) = make_dynamic_window(
                *actual_time,
                target_time,
                args.tout,
                args.buffer_size,
                args.backward,
            );

            println!(
                "Reloading dynamic field window -> [{:.12}, {:.12}]",
                win_tmin, win_tmax
            );

            fields = VlsvDynamicField::<f64>::new_partial(vlsv_dir, periodic, win_tmin, win_tmax);
            (loaded_tmin, loaded_tmax) = fields.temporal_range();

            if !range_contains_step(loaded_tmin, loaded_tmax, *actual_time, target_time) {
                panic!(
                    "Loaded VLSV range [{loaded_tmin:.12}, {loaded_tmax:.12}] does not bracket exact tracer step [{:.12}, {target_time:.12}] requested through [{win_tmin:.12}, {win_tmax:.12}]. The tracer clock was left unchanged.",
                    *actual_time
                );
            }
        }

        print_trace_status(
            pop_arc,
            num_particles,
            *actual_time,
            Some((loaded_tmin, loaded_tmax)),
            Some((win_tmin, win_tmax)),
        );

        advance_population(pop_arc, &fields, actual_time, target_time, args.backward);
        save_population(pop_arc, &args.output, *out_count, args.dry);

        *out_count += 1;
        schedule_step += 1;
    }
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let args = Args::parse();

    if let Err(message) = validate_args(&args) {
        eprintln!("ERROR: {message}");
        return Err(std::process::ExitCode::from(2));
    }

    let periodic = [args.periodic_x, args.periodic_y, args.periodic_z];

    let sim_kind = if let Some(path) = &args.vlsv {
        let meta = std::fs::metadata(path)
            .unwrap_or_else(|error| panic!("Failed to inspect VLSV path '{path}': {error}"));
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
    let mut actual_time = args.tstart.unwrap_or(default_start);
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
                .map(|value| value.trim().parse::<f64>().expect("Parse error"))
                .collect();

            if sub.len() != 7 {
                panic!("Expected seven comma-separated values, got: {line}");
            }

            actual_time = sub[0];
            pop.add_particle([sub[1], sub[2], sub[3], sub[4], sub[5], sub[6]], true);
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

    if !actual_time.is_finite()
        || actual_time < args.tmin - TIME_EPS
        || actual_time > args.tmax + TIME_EPS
    {
        eprintln!(
            "ERROR: start time {:.12} is outside [{:.12}, {:.12}]",
            actual_time, args.tmin, args.tmax
        );
        return Err(std::process::ExitCode::from(2));
    }

    let num_particles = pop.size();
    let mut pop_arc = Arc::new(Mutex::new(pop));
    let mut out_count: usize = 0;

    match sim_kind {
        SimulationKind::Dynamic => {
            let vlsv_dir = args
                .vlsv
                .as_deref()
                .expect("Dynamic simulation requires --vlsv directory");
            run_dynamic_field(
                &args,
                vlsv_dir,
                periodic,
                &mut pop_arc,
                num_particles,
                &mut actual_time,
                &mut out_count,
            );
        }
        SimulationKind::Static => {
            let vlsv_file = args
                .vlsv
                .as_deref()
                .expect("Static simulation requires --vlsv file");
            let fields = VlsvStaticField::<f64>::new(&String::from(vlsv_file), periodic);
            run_fixed_field(
                &args,
                &fields,
                &mut pop_arc,
                num_particles,
                &mut actual_time,
                &mut out_count,
            );
        }
        SimulationKind::Other => {
            let fields = DipoleField::<f64>::new(8e15_f64);
            run_fixed_field(
                &args,
                &fields,
                &mut pop_arc,
                num_particles,
                &mut actual_time,
                &mut out_count,
            );
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

    fn scheduling_args(backward: bool) -> Args {
        Args {
            vlsv: None,
            tstart: Some(1418.0),
            tmin: 1300.0,
            tmax: 1419.0,
            tout: 1.0,
            backward,
            periodic_x: false,
            periodic_y: false,
            periodic_z: false,
            input: None,
            output: "state".to_string(),
            num_particles: 1,
            energy_kev: 512.0,
            lshell: 10.0,
            dry: true,
            buffer_size: 10.0,
        }
    }

    #[test]
    fn test_backward_output_schedule_remains_anchored() {
        let args = scheduling_args(true);
        let origin = 1418.0;

        assert_eq!(scheduled_target(origin, 1, &args), 1417.0);
        assert_eq!(scheduled_target(origin, 10, &args), 1408.0);
        assert_eq!(scheduled_target(origin, 69, &args), 1349.0);
        assert_eq!(scheduled_target(origin, 70, &args), 1348.0);
    }

    #[test]
    fn test_dynamic_window_is_based_on_logical_time() {
        let (wmin, wmax) = make_dynamic_window(1408.0, 1407.0, 1.0, 10.0, true);
        assert_eq!(wmin, 1398.0);
        assert_eq!(wmax, 1409.0);
    }

    #[test]
    fn test_jittered_loaded_edge_triggers_reload_without_shortening_step() {
        assert!(!range_contains_step(
            1408.001598113233,
            1418.012292310861,
            1409.0,
            1408.0,
        ));

        assert!(range_contains_step(
            1398.003607842251,
            1409.004562765789,
            1409.0,
            1408.0,
        ));
    }

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
