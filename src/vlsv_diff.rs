mod vlsv_reader;
use crate::mod_vlsv_reader::VlsvFile;
use crate::vlsv_reader::*;
use clap::{Parser, Subcommand};
use ndarray::Array4;
use std::collections::{HashMap, HashSet};
use std::process::ExitCode;

/// Simple CLI tool for diffing .vlsv files
#[derive(Parser, Debug)]
#[command(
    name = "vlsv_diff",
    version,
    about = "Diffs .vlsv files",
    long_about = r#"This tool allows you to diff .vlsv files.

Author:
    Kostis Papadakis <kpapadakis@protonmail.com> (2025)"#
)]
struct Args {
    /// Path to the first .vlsv file
    file1: String,

    /// Path to the second .vlsv file
    file2: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Diff a variable
    Var {
        /// Variable name
        #[arg(short, long)]
        variable: String,
    },

    /// Diff VDFs
    Vdf {
        /// CELLID
        #[arg(long)]
        cid: usize,
    },
}

fn diff_arrays(var1: &Array4<f32>, var2: &Array4<f32>) {
    assert_eq!(
        var1.shape(),
        var2.shape(),
        "Variable shapes differ: {:?} vs {:?}",
        var1.shape(),
        var2.shape()
    );

    let var1_mean = var1.mean().expect("Could not get mean of var1");
    let var2_mean = var2.mean().expect("Could not get mean of var2");

    let var1_std = var1.std(0.0f32);
    let var2_std = var2.std(0.0f32);

    let var1_min = var1.fold(f32::INFINITY, |a, &b| a.min(b));
    let var2_min = var2.fold(f32::INFINITY, |a, &b| a.min(b));

    let var1_max = var1.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let var2_max = var2.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    let n = var1.len() as f32;
    let mut l1 = 0.0f32;
    let mut l2_sq = 0.0f32;
    let mut linf = 0.0f32;
    let mut rel_l1_denom = 0.0f32;
    let mut rel_l2_denom = 0.0f32;
    let mut rel_linf_denom = 0.0f32;

    for (&a, &b) in var1.iter().zip(var2.iter()) {
        let diff = a - b;
        let abs_diff = diff.abs();
        l1 += abs_diff;
        l2_sq += diff * diff;
        linf = linf.max(abs_diff);
        rel_l1_denom += a.abs();
        rel_l2_denom += a * a;
        rel_linf_denom = rel_linf_denom.max(a.abs());
    }

    let l2 = l2_sq.sqrt();
    let mae = l1 / n;
    let rmse = (l2_sq / n).sqrt();

    let rel_l1 = l1 / rel_l1_denom.max(f32::EPSILON);
    let rel_l2 = l2 / rel_l2_denom.sqrt().max(f32::EPSILON);
    let rel_linf = linf / rel_linf_denom.max(f32::EPSILON);

    println!("------|----------------------|----------------------|");
    println!(
        "Dims: | {:<20} | {:<20} |",
        format!("{:?}", var1.shape()),
        format!("{:?}", var2.shape())
    );
    println!("Mean: | {:<20.6e} | {:<20.6e} |", var1_mean, var2_mean);
    println!("Std : | {:<20.6e} | {:<20.6e} |", var1_std, var2_std);
    println!("Min : | {:<20.6e} | {:<20.6e} |", var1_min, var2_min);
    println!("Max : | {:<20.6e} | {:<20.6e} |", var1_max, var2_max);

    println!();
    println!("Error Statistics");
    println!("L1   : {:<20.6e}", l1);
    println!("L2   : {:<20.6e}", l2);
    println!("Linf : {:<20.6e}", linf);
    println!("MAE  : {:<20.6e}", mae);
    println!("RMSE : {:<20.6e}", rmse);
    println!("rL1  : {:<20.6e}", rel_l1);
    println!("rL2  : {:<20.6e}", rel_l2);
    println!("rLinf: {:<20.6e}", rel_linf);
}

fn diff_vdfs(var1: &HashMap<usize, f32>, var2: &HashMap<usize, f32>) {
    let keys: HashSet<usize> = var1.keys().chain(var2.keys()).copied().collect();
    let n = keys.len();
    let n_f = n.max(1) as f32;
    let only_in_1 = var1.keys().filter(|k| !var2.contains_key(k)).count();
    let only_in_2 = var2.keys().filter(|k| !var1.contains_key(k)).count();
    let common = var1.keys().filter(|k| var2.contains_key(k)).count();
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;

    let mut sumsq1 = 0.0f32;
    let mut sumsq2 = 0.0f32;

    let mut min1 = f32::INFINITY;
    let mut min2 = f32::INFINITY;

    let mut max1 = f32::NEG_INFINITY;
    let mut max2 = f32::NEG_INFINITY;

    let mut l1 = 0.0f32;
    let mut l2_sq = 0.0f32;
    let mut linf = 0.0f32;

    let mut rel_l1_denom = 0.0f32;
    let mut rel_l2_denom = 0.0f32;
    let mut rel_linf_denom = 0.0f32;

    let mut max_diff_key = None;
    let mut max_diff_a = 0.0f32;
    let mut max_diff_b = 0.0f32;

    for k in keys {
        let a = *var1.get(&k).unwrap_or(&0.0);
        let b = *var2.get(&k).unwrap_or(&0.0);

        sum1 += a;
        sum2 += b;

        sumsq1 += a * a;
        sumsq2 += b * b;

        min1 = min1.min(a);
        min2 = min2.min(b);

        max1 = max1.max(a);
        max2 = max2.max(b);

        let diff = a - b;
        let abs_diff = diff.abs();

        l1 += abs_diff;
        l2_sq += diff * diff;
        linf = linf.max(abs_diff);

        if abs_diff >= linf {
            max_diff_key = Some(k);
            max_diff_a = a;
            max_diff_b = b;
        }

        rel_l1_denom += a.abs();
        rel_l2_denom += a * a;
        rel_linf_denom = rel_linf_denom.max(a.abs());
    }

    let mean1 = sum1 / n_f;
    let mean2 = sum2 / n_f;

    let std1 = (sumsq1 / n_f - mean1 * mean1).max(0.0).sqrt();
    let std2 = (sumsq2 / n_f - mean2 * mean2).max(0.0).sqrt();

    let l2 = l2_sq.sqrt();
    let mae = l1 / n_f;
    let rmse = (l2_sq / n_f).sqrt();

    let rel_l1 = l1 / rel_l1_denom.max(f32::EPSILON);
    let rel_l2 = l2 / rel_l2_denom.sqrt().max(f32::EPSILON);
    let rel_linf = linf / rel_linf_denom.max(f32::EPSILON);

    let mass_diff = sum2 - sum1;
    let rel_mass_diff = mass_diff / sum1.abs().max(f32::EPSILON);

    println!("      | {:<20} | {:<20} |", "VDF 1", "VDF 2");
    println!("------|----------------------|----------------------|");
    println!("N   : | {:<20} | {:<20} |", var1.len(), var2.len());
    println!("Mean: | {:<20.6e} | {:<20.6e} |", mean1, mean2);
    println!("Std : | {:<20.6e} | {:<20.6e} |", std1, std2);
    println!("Min : | {:<20.6e} | {:<20.6e} |", min1, min2);
    println!("Max : | {:<20.6e} | {:<20.6e} |", max1, max2);
    println!("Mass: | {:<20.6e} | {:<20.6e} |", sum1, sum2);

    println!();
    println!("Sparsity Statistics");
    println!("Union entries : {}", n);
    println!("Common entries: {}", common);
    println!("Only in VDF 1 : {}", only_in_1);
    println!("Only in VDF 2 : {}", only_in_2);

    println!();
    println!("Error Statistics");
    println!("L1    : {:<20.6e}", l1);
    println!("L2    : {:<20.6e}", l2);
    println!("Linf  : {:<20.6e}", linf);
    println!("MAE   : {:<20.6e}", mae);
    println!("RMSE  : {:<20.6e}", rmse);
    println!("rL1   : {:<20.6e}", rel_l1);
    println!("rL2   : {:<20.6e}", rel_l2);
    println!("rLinf : {:<20.6e}", rel_linf);

    println!();
    println!("Physical / Shape Statistics");
    println!("Mass diff        : {:<20.6e}", mass_diff);
    println!("Relative mass diff: {:<20.6e}", rel_mass_diff);

    if let Some(k) = max_diff_key {
        println!();
        println!("Largest local difference");
        println!("Key : {}", k);
        println!("VDF1: {:<20.6e}", max_diff_a);
        println!("VDF2: {:<20.6e}", max_diff_b);
        println!("Diff: {:<20.6e}", max_diff_a - max_diff_b);
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let f1 = VlsvFile::new(&args.file1).expect("Could not open first .vlsv file");
    let f2 = VlsvFile::new(&args.file2).expect("Could not open second .vlsv file");

    match args.command {
        Command::Var { variable } => {
            let var1 = f1
                .read_variable::<f32>(&variable, None)
                .expect("Could not read variable from first vlsv file!");

            let var2 = f2
                .read_variable::<f32>(&variable, None)
                .expect("Could not read variable from second vlsv file!");
            diff_arrays(&var1, &var2);
        }
        Command::Vdf { cid } => {
            let var1 = f1
                .read_vdf_dict::<f32>(cid, "proton")
                .expect("Could not read VDF from first vlsv file!");
            let var2 = f2
                .read_vdf_dict::<f32>(cid, "proton")
                .expect("Could not read VDF from second vlsv file!");
            diff_vdfs(&var1, &var2);
        }
    }
    ExitCode::SUCCESS
}
