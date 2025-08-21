mod vlsv_reader;
use clap::Parser;
use vlsv_reader::mod_vlsv_reader::VlsvFile;
/// Simple CLI tool for reading .vlsv files
#[derive(Parser, Debug)]
#[command(
    name = "vlsvdump",
    version,
    about = "Reads and displays variables or config from .vlsv files",
    long_about = r#"This tool allows you to read and display the configuration
and variables from a .vlsv file.
Author:
    Kostis Papadakis <kpapadakis@protonmail.com> (2025) "#
)]
struct Args {
    /// Path to the .vlsv file
    #[arg()]
    file: String,

    /// Prints config
    #[arg(short = 'c', long = "config")]
    print_config: bool,

    /// Prints variables
    #[arg(short = 'v', long = "vars")]
    print_vars: bool,

    /// Prints git status of run
    #[arg(short = 'g', long = "git")]
    print_git: bool,
}

fn print_variables(f: &VlsvFile) -> Result<(), Box<dyn std::error::Error>> {
    let wid = f.get_wid().unwrap();
    println!("WID: {:?}", wid);
    println!("Max AMR level: {:?}", f.get_max_amr_refinement().unwrap());
    println!(
        "Real Space Geometry: {:?} cells",
        f.get_spatial_mesh_bbox().unwrap()
    );
    println!(
        "Real Space Extents: {:?} [m]",
        f.get_spatial_mesh_extents().unwrap()
    );
    let pops = f.get_all_populations().unwrap();
    println!("Populations: {:?} ", pops);
    pops.iter().for_each(|p| {
        println!(
            "\t {} Velocity Space Geometry: {:?} [blocks (WID={wid})]",
            p.to_uppercase(),
            TryInto::<[usize; 3]>::try_into(f.get_vspace_mesh_bbox(p).unwrap())
                .unwrap()
                .map(|x| x / wid)
        );
        println!(
            "\t {} Velocity Space Extents: {:?} [km/s]",
            p.to_uppercase(),
            f.get_vspace_mesh_extents(p).unwrap()
        );
    });
    print!("Variables :\n[");
    for (name, _meta) in f.variables.iter() {
        print!("{}, ", name);
    }
    println!("]\n");
    print!("Parameters :\n[");
    for (name, _meta) in f.parameters.iter() {
        let ds = f.get_dataset(name).unwrap();
        if ds.arraysize < 3 && ds.vectorsize < 3 {
            print!("{}[{:.2?}], ", name, f.read_scalar_parameter(name).unwrap());
        } else {
            print!("{}, ", name);
        }
    }
    println!("]");
    Ok(())
}

fn print_config(f: &VlsvFile) -> Result<(), Box<dyn std::error::Error>> {
    println!("{},", f.read_config().expect("Config not found"));
    Ok(())
}

fn print_version(f: &VlsvFile) -> Result<(), Box<dyn std::error::Error>> {
    println!("{},", f.read_version().expect("Version not found"));
    Ok(())
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let args = Args::parse();

    if !args.print_config && !args.print_vars && !args.print_git {
        eprintln!("Error: At least one of -c, -v or -g  must be specified.");
        return Err(std::process::ExitCode::FAILURE);
    }
    let f = VlsvFile::new(&args.file).unwrap();

    if args.print_config {
        if let Err(e) = print_config(&f) {
            eprintln!("Failed to print config: {}", e);
            return Err(std::process::ExitCode::FAILURE);
        }
    }

    if args.print_vars {
        if let Err(e) = print_variables(&f) {
            eprintln!("Failed to print variables: {}", e);
            return Err(std::process::ExitCode::FAILURE);
        }
    }

    if args.print_git {
        if let Err(e) = print_version(&f) {
            eprintln!("Failed to print version: {}", e);
            return Err(std::process::ExitCode::FAILURE);
        }
    }

    Ok(std::process::ExitCode::SUCCESS)
}
