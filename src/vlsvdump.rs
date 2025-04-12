mod vlsv_reader;
use clap::Parser;
use vlsv_reader::vlsv_reader::VlsvFile;

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
    print!("[");
    for v in f.data.iter() {
        if v.1.arraysize.is_some() {
            if f.get_data_info(v.0).unwrap().arraysize < 3
                && f.get_data_info(v.0).unwrap().vectorsize < 3
            {
                print!("{}[{:.2?}], ", &v.0, f.read_parameter(v.0).unwrap());
            } else {
                print!("{}, ", &v.0);
            }
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
    println!("{},", f.read_parameter("time").unwrap());
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
