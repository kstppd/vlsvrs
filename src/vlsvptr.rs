mod constants;
mod tracer_fields;
mod tracer_particles;
mod vlsv_reader;
use crate::constants::physical_constants;
use crate::tracer_particles::tracer_particles::ParticlePopulation;
use tracer_fields::vlsv_reader::VlsvStaticField;
use tracer_fields::{
    vlsv_reader::{DipoleField, Field},
    *,
};
use tracer_particles::*;
use vlsv_reader::vlsv_reader::VlsvFile;

// fn push_population_cpu<T>(p: &ParticlePopulation<T>) {
//     todo!();
// }

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let field = DipoleField::<f32>::new(physical_constants::f32::DIPOLE_MOMENT);
    let p = ParticlePopulation::<f32>::new_with_energy_at_Lshell(
        1024,
        physical_constants::f32::PROTON_MASS,
        physical_constants::f32::PROTON_CHARGE,
        512_f32,
        10_f32 * physical_constants::f32::EARTH_RE,
    );
    p.save("state.0000000.ptr");
    Ok(std::process::ExitCode::SUCCESS)
}
