mod vlsv_reader;
use ndarray_npy::write_npy;
use vlsv_reader::vlsv_reader::VlsvFile;

fn main() {
    let args = std::env::args()
        .into_iter()
        .skip(1)
        .collect::<Vec<String>>();
    let f = VlsvFile::new(&args[0]).unwrap();
    let vdf = f.read_vdf(1, "antiproton").expect("No VDF in CellID");
    println!("vdf size = {}", vdf.len());
    ndarray_npy::write_npy("vdf.npy", &vdf).unwrap();
    println!("{},", f.read_scalar_parameter("time").unwrap());
}
