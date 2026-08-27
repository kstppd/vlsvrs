#![allow(dead_code)]
#![allow(non_snake_case)]

mod vlsv_reader;
use crate::mod_vlsv_tracing::*;
use crate::vlsv_reader::*;
use clap::Parser;
use rayon::prelude::*;
use std::process::ExitCode;
const TIME_EPS: f64 = 1.0e-10;
const DT_SEED: f64 = 1.0e-4;

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

enum Fields {
    Dipole(DipoleField<f64>),
    Static(VlsvStaticField<f64>),
    Dynamic(VlsvDynamicField<f64>),
}

impl Field<f64> for Fields {
    fn get_fields_at(&self, t: f64, x: f64, y: f64, z: f64) -> Option<[f64; 6]> {
        match self {
            Fields::Dipole(f) => f.get_fields_at(t, x, y, z),
            Fields::Static(f) => f.get_fields_at(t, x, y, z),
            Fields::Dynamic(f) => f.get_fields_at(t, x, y, z),
        }
    }

    fn ds(&self) -> f64 {
        match self {
            Fields::Dipole(f) => f.ds(),
            Fields::Static(f) => f.ds(),
            Fields::Dynamic(f) => f.ds(),
        }
    }
}

fn validate(args: &Args) -> Result<(), String> {
    if !(args.tmin.is_finite() && args.tmax.is_finite() && args.tmin < args.tmax) {
        return Err(format!(
            "invalid time interval: tmin={} and tmax={}",
            args.tmin, args.tmax
        ));
    }
    if !(args.tout.is_finite() && args.tout > 0.0) {
        return Err(format!(
            "tout must be finite and positive, got {}",
            args.tout
        ));
    }
    if !(args.buffer_size.is_finite() && args.buffer_size > 0.0) {
        return Err(format!(
            "buffer_size must be finite and positive, got {}",
            args.buffer_size
        ));
    }
    if args.tstart.is_some_and(|t| !t.is_finite()) {
        return Err("tstart must be finite".to_string());
    }
    Ok(())
}

fn periodic(args: &Args) -> [bool; 3] {
    [args.periodic_x, args.periodic_y, args.periodic_z]
}

fn sign(args: &Args) -> f64 {
    if args.backward { -1.0 } else { 1.0 }
}

fn target_time(origin: f64, step: u64, args: &Args) -> f64 {
    let t = origin + sign(args) * step as f64 * args.tout.abs();
    if args.backward {
        t.max(args.tmin)
    } else {
        t.min(args.tmax)
    }
}

fn finished(current: f64, args: &Args) -> bool {
    if args.backward {
        current <= args.tmin + TIME_EPS
    } else {
        current >= args.tmax - TIME_EPS
    }
}

fn window(current: f64, target: f64, args: &Args) -> (f64, f64) {
    let guard = args.tout.abs().max(1.0e-9);
    let span = args.buffer_size.max((target - current).abs() + guard);
    if args.backward {
        (current - span, current + guard)
    } else {
        (current - guard, current + span)
    }
}

fn contains(loaded: (f64, f64), current: f64, target: f64) -> bool {
    loaded.0 <= current.min(target) && loaded.1 >= current.max(target)
}

fn advance<F: Field<f64> + Sync>(
    pop: &mut ParticlePopulation<f64>,
    fields: &F,
    from: f64,
    to: f64,
) {
    let (mass, charge, backward) = (pop.mass, pop.charge, to < from);
    let mut particles: Vec<Particle<f64>> =
        (0..pop.size()).map(|i| pop.get_temp_particle(i)).collect();

    particles.par_iter_mut().for_each(|p| {
        if !p.alive {
            return;
        }
        let mut dt = if backward { -DT_SEED } else { DT_SEED };
        if backward {
            boris_backtracing_adaptive(p, fields, &mut dt, from, to, mass, charge);
        } else {
            boris_adaptive(p, fields, &mut dt, from, to, mass, charge);
        }
    });

    for (i, p) in particles.iter().enumerate() {
        pop.take_temp_particle(p, i);
    }
}

fn trace(
    args: &Args,
    mut fields: Option<Fields>,
    vlsv_dir: Option<&str>,
    pop: &mut ParticlePopulation<f64>,
    start: f64,
) -> Result<f64, String> {
    let mut current = start;
    let mut loaded = (f64::NEG_INFINITY, f64::INFINITY);
    let fname = format!("{}.{:07}.ptr", "state", 0);
    pop.save(&fname, current as f64);
    if finished(current, args) {
        println!(
            "Start time {current:.12} s is already at the far end of [{:.12}, {:.12}], nothing to trace",
            args.tmin, args.tmax
        );
    }

    for step in 1.. {
        if finished(current, args) {
            break;
        }
        let target = target_time(start, step, args);
        if (target - current) * sign(args) <= 0.0 {
            break;
        }

        if let Some(dir) = vlsv_dir {
            if !contains(loaded, current, target) {
                let request = window(current, target, args);
                println!(
                    "Loading dynamic field window -> [{:.12}, {:.12}]",
                    request.0, request.1
                );
                drop(fields.take());
                let loading =
                    VlsvDynamicField::<f64>::new_partial(dir, periodic(args), request.0, request.1);
                loaded = loading.temporal_range();
                fields = Some(Fields::Dynamic(loading));

                if !contains(loaded, current, target) {
                    return Err(format!(
                        "loaded VLSV snapshots [{:.12}, {:.12}] do not contain the tracer step [{:.12}, {:.12}] requested through [{:.12}, {:.12}]; the tracer clock is never moved onto a file timestamp, so containing snapshots must exist in {dir}",
                        loaded.0, loaded.1, current, target, request.0, request.1
                    ));
                }
            }
        }

        let range = match vlsv_dir {
            Some(_) => format!(", loaded range [{:.12}, {:.12}]", loaded.0, loaded.1),
            None => String::new(),
        };
        println!(
            "Tracing {} particles [{} alive] at t= {current:.12} s{range}",
            pop.size(),
            pop.count_alive()
        );

        advance(
            pop,
            fields.as_ref().expect("no field loaded"),
            current,
            target,
        );
        current = target;
        let fname = format!("{}.{:07}.ptr", "state", step);
        pop.save(&fname, current as f64);
    }

    Ok(current)
}

fn load_particles(
    path: &str,
    mass: f64,
    charge: f64,
) -> Result<(ParticlePopulation<f64>, f64), String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("failed to read {path}: {error}"))?;
    let mut pop = ParticlePopulation::<f64>::new(1024, mass, charge);
    let mut start: Option<f64> = None;

    for (index, line) in text.lines().enumerate() {
        let (line, lineno) = (line.trim(), index + 1);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let v: Vec<f64> = line
            .split(',')
            .map(|field| field.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|error| format!("{path}:{lineno}: {error} in '{line}'"))?;

        if v.len() != 7 || v.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "{path}:{lineno}: expected seven finite values (time,x,y,z,vx,vy,vz), got '{line}'"
            ));
        }
        if (v[0] - *start.get_or_insert(v[0])).abs() > TIME_EPS {
            return Err(format!(
                "{path}:{lineno}: start time {:.12} differs from {:.12} used by the earlier rows; all particles must share one start time",
                v[0],
                start.unwrap()
            ));
        }

        pop.add_particle([v[1], v[2], v[3], v[4], v[5], v[6]], true);
    }

    start
        .map(|time| (pop, time))
        .ok_or_else(|| format!("input file {path} contains no particles"))
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    validate(&args)?;

    let mass = physical_constants::f64::PROTON_MASS;
    let charge = physical_constants::f64::PROTON_CHARGE;

    let (mut pop, start) = match &args.input {
        Some(path) => {
            let (pop, time) = load_particles(path, mass, charge)?;
            if args.tstart.is_some_and(|t| (t - time).abs() > TIME_EPS) {
                eprintln!("WARNING: --tstart ignored, using start time {time:.12} from {path}");
            }
            (pop, time)
        }
        None => {
            let pop = ParticlePopulation::<f64>::new_with_energy_at_Lshell(
                args.num_particles,
                mass,
                charge,
                args.energy_kev,
                args.lshell * physical_constants::f64::EARTH_RE,
            );
            let default_start = if args.backward { args.tmax } else { args.tmin };
            (pop, args.tstart.unwrap_or(default_start))
        }
    };

    if pop.size() == 0 {
        return Err("no particles to trace".to_string());
    }
    if !start.is_finite() || start < args.tmin - TIME_EPS || start > args.tmax + TIME_EPS {
        return Err(format!(
            "start time {start:.12} is outside [{:.12}, {:.12}]",
            args.tmin, args.tmax
        ));
    }

    let (fields, vlsv_dir) = match &args.vlsv {
        None => (
            Some(Fields::Dipole(DipoleField::new(
                physical_constants::f64::DIPOLE_MOMENT,
            ))),
            None,
        ),
        Some(path) => {
            let meta = std::fs::metadata(path)
                .map_err(|error| format!("failed to inspect VLSV path '{path}': {error}"))?;
            if meta.is_file() {
                let fields = VlsvStaticField::new(path, periodic(&args));
                (Some(Fields::Static(fields)), None)
            } else {
                (None, Some(path.as_str()))
            }
        }
    };

    let reached = trace(&args, fields, vlsv_dir, &mut pop, start)?;

    println!(
        "Done: {}, final time {reached:.12} s, {} of {} particles alive",
        if args.dry {
            " (dry run, nothing written)"
        } else {
            ""
        },
        pop.count_alive(),
        pop.size()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("ERROR: {message}");
            ExitCode::from(2)
        }
    }
}
