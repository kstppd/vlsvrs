mod tracer_fields;
mod tracer_particles;
mod vlsv_reader;
use crate::tracer_particles::tracer_particles::ParticlePopulation;
use tracer_fields::vlsv_reader::VlsvStaticField;
use tracer_fields::{
    vlsv_reader::{DipoleField, Field},
    *,
};
use tracer_particles::*;
use vlsv_reader::vlsv_reader::VlsvFile;

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let file = std::env::args()
        .skip(1)
        .collect::<Vec<String>>()
        .first()
        .cloned()
        .expect("No file provided!");

    let fields = VlsvStaticField::<f32>::new(&file);
    println!(
        "{:?}",
        fields.get_fields_at(0.0, 6378137.0_f32 * 8_f32, 0.0, 0.0)
    );
    let p = ParticlePopulation::<f32>::new_with_energy_at_Lshell(
        100,
        1_f32,
        2_f32,
        1024_f32,
        10_f32 * 6378137_f32,
    );
    Ok(std::process::ExitCode::SUCCESS)
}
