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
}

fn print_variables(fname: &String) -> Result<(), Box<dyn std::error::Error>> {
    let f = VlsvFile::new(fname)?;
    print!("[");
    for v in f.data.iter() {
        if v.1.arraysize.is_some() {}
        print!("{}, ", &v.0);
    }
    println!("]");
    Ok(())
}

fn print_config(fname: &String) -> Result<(), Box<dyn std::error::Error>> {
    let f = VlsvFile::new(fname)?;
    println!("{},", f.read_config().expect("Config not found"));
    Ok(())
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let args = Args::parse();

    if !args.print_config && !args.print_vars {
        eprintln!("Error: At least one of -c or -v must be specified.");
        return Err(std::process::ExitCode::FAILURE);
    }

    if args.print_config {
        if let Err(e) = print_config(&args.file) {
            eprintln!("Failed to print config: {}", e);
            return Err(std::process::ExitCode::FAILURE);
        }
    }

    if args.print_vars {
        if let Err(e) = print_variables(&args.file) {
            eprintln!("Failed to print variables: {}", e);
            return Err(std::process::ExitCode::FAILURE);
        }
    }

    Ok(std::process::ExitCode::SUCCESS)
}
