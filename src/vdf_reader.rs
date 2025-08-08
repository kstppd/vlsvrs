mod vlsv_reader;
use vlsv_reader::vlsv_reader::VlsvFile;

use std::fs::File;
use std::io::{self, Write};

fn dump_raw_f32(path: &str, data: &[f32]) -> io::Result<()> {
    let mut f = File::create(path)?;
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<f32>(),
        )
    };
    f.write_all(bytes)
}

fn main() {
    let args = std::env::args()
        .into_iter()
        .skip(1)
        .collect::<Vec<String>>();
    let f = VlsvFile::new(&args[0]).unwrap();
    let vdf = f.read_vdf(1, "proton").expect("No VDF in CellID");
    println!("vdf size = {}", vdf.len());
    dump_raw_f32("vdf.bin", &vdf).unwrap();
    println!("{},", f.read_scalar_parameter("time").unwrap());
}
