mod vlsv_reader;
use vlsv_reader::vlsv_reader::VlsvFile;

fn dump(fname: &String) -> Result<(), Box<dyn std::error::Error>> {
    let f = VlsvFile::new(fname)?;
    println!(
        "File {} contains:",
        std::path::Path::new(&f.filename)
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
    );
    for v in f.data.iter() {
        if v.1.arraysize.is_some() {}
        println!("\t{}", &v.0);
    }
    Ok(())
}

fn main() -> Result<std::process::ExitCode, std::process::ExitCode> {
    let file = std::env::args()
        .skip(1)
        .collect::<Vec<String>>()
        .first()
        .cloned()
        .expect("No file provided!");

    if dump(&file).is_err() {
        return Err(std::process::ExitCode::FAILURE);
    }
    Ok(std::process::ExitCode::SUCCESS)
}
