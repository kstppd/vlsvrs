pub mod vlsv_reader {
    use bytemuck::{Pod, cast_slice_mut};
    use ndarray::{Array4, IntoNdProducer, Order, Shape, ShapeBuilder, s};
    use num_traits;
    use serde::Deserialize;
    use serde_xml_rs::from_str;
    use std::collections::HashMap;
    use std::io::{Read, Seek};
    use std::os::unix::fs::FileExt;
    #[derive(Deserialize, Debug, Clone)]
    #[serde(rename_all = "UPPERCASE")]
    #[allow(dead_code)]
    pub struct Variable {
        #[serde(rename = "arraysize")]
        pub arraysize: Option<String>,
        #[serde(rename = "datasize")]
        datasize: Option<String>,
        #[serde(rename = "datatype")]
        datatype: Option<String>,
        #[serde(rename = "mesh")]
        mesh: Option<String>,
        #[serde(rename = "name")]
        name: Option<String>,
        #[serde(rename = "vectorsize")]
        vectorsize: Option<String>,
        #[serde(rename = "max_refinement_level")]
        max_refinement_level: Option<String>,
        #[serde(rename = "unit")]
        unit: Option<String>,
        #[serde(rename = "unitConversion")]
        unit_conversion: Option<String>,
        #[serde(rename = "unitLaTeX")]
        unit_latex: Option<String>,
        #[serde(rename = "variableLaTeX")]
        variable_latex: Option<String>,
        #[serde(rename = "text")]
        text: Option<String>,
        #[serde(rename = "$value")]
        offset: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub struct VlsvDataset {
        pub offset: usize,
        pub arraysize: usize,
        pub vectorsize: usize,
        pub datasize: usize,
        pub datatype: String,
    }

    #[derive(Debug)]
    pub struct VlsvFile {
        pub filename: String,
        pub data: HashMap<String, Variable>,
    }

    impl VlsvFile {
        pub fn new(filename: &String) -> Result<Self, Box<dyn std::error::Error>> {
            let mut f = std::fs::File::open(filename)?;
            f.seek_relative(8)?;
            let mut footer_offset: [i64; 1] = [0];
            f.read_exact(cast_slice_mut(&mut footer_offset))?;
            f.seek_relative(-8)?;
            f.seek_relative(footer_offset[0] + 1)?;
            let mut footer_bytes: Vec<u8> = vec![];
            let bytes_read = f.read_to_end(&mut footer_bytes)?;
            assert!(bytes_read == footer_bytes.len());
            let mut xml_string = String::from(std::str::from_utf8(footer_bytes.as_slice())?);
            xml_string.truncate(xml_string.len() - 9);
            let result: Vec<Variable> = from_str(&xml_string).unwrap();
            let mut map: HashMap<String, Variable> = HashMap::new();
            for i in result.iter() {
                if let Some(val) = &i.name {
                    map.insert(val.clone(), i.clone());
                }
            }
            let re = regex::Regex::new(r#"arraysize="([^"]+)"[^>]*datasize="([^"]+)"[^>]*datatype="([^"]+)"[^>]*mesh="([^"]+)"[^>]*vectorsize="([^"]+)">([^<]+)</MESH_DECOMPOSITION>"#).unwrap();
            if let Some(caps) = re.captures(xml_string.as_str()) {
                let arraysize = caps.get(1).map(|m| m.as_str().to_string());
                let datasize = caps.get(2).map(|m| m.as_str().to_string());
                let datatype = caps.get(3).map(|m| m.as_str().to_string());
                let mesh = caps.get(4).map(|m| m.as_str().to_string());
                let vectorsize = caps.get(5).map(|m| m.as_str().to_string());
                let offset = caps.get(6).map(|m| m.as_str().to_string());

                let variable = Variable {
                    arraysize,
                    datasize,
                    datatype,
                    mesh,
                    name: Some(String::from("decomposition").clone()),
                    vectorsize,
                    max_refinement_level: None,
                    unit: None,
                    unit_conversion: None,
                    unit_latex: None,
                    variable_latex: None,
                    text: None,
                    offset: Some(offset.expect("FUBAR")),
                };
                map.insert(variable.name.clone().unwrap().clone(), variable.clone());
            } else {
                eprintln!(
                    "WARNING: Domain Decomposition not found in file! FS Grid reading will fail!"
                );
            }
            Ok(Self {
                filename: filename.clone(),
                data: map,
            })
        }

        pub fn print_variables(&self) {
            for key in self.data.keys() {
                println!("{}", key);
            }
        }

        pub fn read_parameter(&self, name: &str) -> Option<f64> {
            let info = self.get_data_info(name)?;
            assert!(info.vectorsize == 1);
            assert!(info.arraysize == 1);
            let f = std::fs::File::open(&self.filename)
                .map_err(|err| {
                    eprintln!("ERROR: could not open file '{}': {:?}", self.filename, err)
                })
                .unwrap();

            let retval = match info.datasize {
                8 => {
                    let mut buffer: [u8; 8] = [0; 8];
                    f.read_exact_at(&mut buffer, info.offset as u64).unwrap();
                    match info.datatype.as_str() {
                        "float" => f64::from_ne_bytes(buffer),
                        "uint" => usize::from_ne_bytes(buffer) as f64,
                        "int" => i64::from_ne_bytes(buffer) as f64,
                        _ => panic!("Only matched against uint and float"),
                    }
                }
                4 => {
                    let mut buffer: [u8; 4] = [0; 4];
                    f.read_exact_at(&mut buffer, info.offset as u64).unwrap();
                    match info.datatype.as_str() {
                        "float" => f32::from_ne_bytes(buffer) as f64,
                        "uint" => u32::from_ne_bytes(buffer) as f64,
                        "int" => i32::from_ne_bytes(buffer) as f64,
                        _ => panic!("Only matched against uint and float"),
                    }
                }
                _ => panic!("Did not expect data size found!"),
            };
            Some(retval)
        }

        pub fn read_config(&self) -> Option<String> {
            let name = "config_file";
            let info = self.get_data_info(name)?;
            let f = std::fs::File::open(&self.filename)
                .map_err(|err| {
                    eprintln!("ERROR: could not open file '{}': {:?}", self.filename, err)
                })
                .unwrap();
            let mut buffer: Vec<u8> = Vec::with_capacity(info.arraysize);
            unsafe {
                buffer.set_len(info.arraysize);
            }
            f.read_exact_at(&mut buffer, info.offset as u64).unwrap();
            let cfgfile = String::from_utf8(buffer).unwrap();
            Some(cfgfile)
        }

        pub fn read_version(&self) -> Option<String> {
            let name = "version_information";
            let info = self.get_data_info(name)?;
            let f = std::fs::File::open(&self.filename)
                .map_err(|err| {
                    eprintln!("ERROR: could not open file '{}': {:?}", self.filename, err)
                })
                .unwrap();
            let mut buffer: Vec<u8> = Vec::with_capacity(info.arraysize);
            unsafe {
                buffer.set_len(info.arraysize);
            }
            f.read_exact_at(&mut buffer, info.offset as u64).unwrap();
            let cfgfile = String::from_utf8(buffer).unwrap();
            Some(cfgfile)
        }

        fn read_variable_into<T: Sized + Pod>(&self, name: &str, dst: &mut [T]) {
            let info = self.get_data_info(name).unwrap();
            let f = std::fs::File::open(&self.filename)
                .map_err(|err| {
                    eprintln!("ERROR: could not open file '{}': {:?}", self.filename, err);
                })
                .unwrap();

            //We now allow mismatch of T and file data only for floating point types. In such case
            // we use unsafed to cast appropriatelly
            if info.datasize != std::mem::size_of::<T>() {
                if info.datatype != "float" {
                    panic!("Casting to T supported only for float types.");
                }
                // 1) T is f64 and file is f32
                if info.datasize == 4 && std::mem::size_of::<T>() == 8 {
                    let size = dst.len();
                    let mut _dst: Vec<f32> = Vec::<f32>::with_capacity(size);
                    unsafe {
                        _dst.set_len(size);
                        f.read_exact_at(cast_slice_mut(_dst.as_mut_slice()), info.offset as u64)
                            .unwrap();
                        for i in 0..size {
                            let value = _dst[i];
                            let valuef64 = value as f64;
                            let bytes = valuef64.to_ne_bytes();
                            let b = dst.as_mut_ptr().add(i);
                            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.cast(), 8);
                        }
                    }
                // 2) T is f32 and file is f64
                } else if info.datasize == 8 && std::mem::size_of::<T>() == 4 {
                    let size = dst.len();
                    let mut _dst: Vec<f64> = Vec::<f64>::with_capacity(size);
                    unsafe {
                        _dst.set_len(size);
                        f.read_exact_at(cast_slice_mut(_dst.as_mut_slice()), info.offset as u64)
                            .unwrap();
                        for i in 0..size {
                            let value = _dst[i];
                            let valuef32 = value as f32;
                            let bytes = valuef32.to_ne_bytes();
                            let b = dst.as_mut_ptr().add(i);
                            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.cast(), 8);
                        }
                    }
                } else {
                    unreachable!("This combination is unhandled");
                }
            } else {
                f.read_exact_at(cast_slice_mut(dst), info.offset as u64)
                    .unwrap();
            }
        }

        pub fn read_fsgrid_variable<T: Sized + Pod + num_traits::identities::Zero + Send + Sync>(
            &self,
            name: &str,
        ) -> Option<Array4<T>> {
            if name[0..3] != *"fg_" {
                panic!("ERROR: Variable {} is not an fs_grid variable!", name);
            }
            let mut info = self.get_data_info(name)?;
            let mut decomp: [u32; 3] = [0; 3];
            self.read_variable_into::<u32>("decomposition", decomp.as_mut_slice());
            let decomp = decomp.iter().map(|x| *x as usize).collect::<Vec<usize>>();
            let ntasks = self.read_parameter("numWritingRanks")? as usize;
            let max_amr = self
                .data
                .get("SpatialGrid")
                .unwrap()
                .max_refinement_level
                .clone()
                .unwrap()
                .parse::<u32>()
                .unwrap();

            let mut nx = self.read_parameter("xcells_ini").unwrap() as usize;
            let mut ny = self.read_parameter("ycells_ini").unwrap() as usize;
            let mut nz = self.read_parameter("zcells_ini").unwrap() as usize;
            nx *= usize::pow(2, max_amr);
            ny *= usize::pow(2, max_amr);
            nz *= usize::pow(2, max_amr);

            fn calc_local_start(global_cells: usize, ntasks: usize, my_n: usize) -> usize {
                let n_per_task = global_cells / ntasks;
                let remainder = global_cells % ntasks;
                if my_n < remainder {
                    return my_n * (n_per_task + 1);
                } else {
                    return my_n * n_per_task + remainder;
                }
            }

            fn calc_local_size(global_cells: usize, ntasks: usize, my_n: usize) -> usize {
                let n_per_task = global_cells / ntasks;
                let remainder = global_cells % ntasks;
                if my_n < remainder {
                    return n_per_task + 1;
                } else {
                    return n_per_task;
                }
            }

            let decomp_len = decomp.len();
            assert!(
                decomp_len == 3,
                "ERROR: fsgrid decomposition should have three elements, but is {:?}",
                decomp
            );
            let mut var = ndarray::Array2::<T>::zeros((nx * ny * nz, info.vectorsize));
            self.read_variable_into::<T>(name, var.as_slice_mut().unwrap());
            let bbox = [nx, ny, nz];
            let mut ordered_var = Array4::<T>::zeros((nx, ny, nz, info.vectorsize));

            let mut current_offset = 0;
            for i in 0..ntasks as usize {
                let x = (i / decomp[2]) / decomp[1];
                let y = (i / decomp[2]) % decomp[1];
                let z = i % decomp[2];

                let task_size = [
                    calc_local_size(bbox[0] as usize, decomp[0], x),
                    calc_local_size(bbox[1] as usize, decomp[1], y),
                    calc_local_size(bbox[2] as usize, decomp[2], z),
                ];

                let task_start = [
                    calc_local_start(bbox[0] as usize, decomp[0], x),
                    calc_local_start(bbox[1] as usize, decomp[1], y),
                    calc_local_start(bbox[2] as usize, decomp[2], z),
                ];

                let task_end = [
                    task_start[0] + task_size[0],
                    task_start[1] + task_size[1],
                    task_start[2] + task_size[2],
                ];

                let total_size = task_size[0] * task_size[1] * task_size[2];
                let _mask = var.slice(s![
                    current_offset..current_offset + total_size,
                    0..info.vectorsize
                ]);
                let mask = _mask
                    .to_shape((
                        (task_size[0], task_size[1], task_size[2], info.vectorsize),
                        Order::F,
                    ))
                    .unwrap();

                let mut subarray = ordered_var.slice_mut(s![
                    task_start[0]..task_end[0],
                    task_start[1]..task_end[1],
                    task_start[2]..task_end[2],
                    0..info.vectorsize
                ]);
                subarray.assign(&mask);
                current_offset += total_size;
            }
            Some(ordered_var)
        }

        #[allow(dead_code, unused_variables, unused_assignments)]
        pub fn read_vggrid_variable<T: Sized + Pod + num_traits::identities::Zero>(
            &self,
            name: &str,
        ) -> Option<Array4<T>> {
            if name[0..3] != *"vg_" {
                panic!("ERROR: Variable {} is not an vg_grid variable!", name);
            }
            let info = self.get_data_info("CellID").unwrap();
            let mut cellids: Vec<u64> = Vec::with_capacity(info.arraysize * info.vectorsize);
            unsafe {
                cellids.set_len(info.arraysize * info.vectorsize);
            }
            self.read_variable_into::<u64>("CellID", cellids.as_mut_slice());
            println!("We have {} cellids", cellids.len());
            let mut decomp: [u32; 3] = [0; 3];
            self.read_variable_into::<u32>("decomposition", decomp.as_mut_slice());
            let decomp = decomp.iter().map(|x| *x as usize).collect::<Vec<usize>>();
            let ntasks = self.read_parameter("numWritingRanks")? as usize;
            let max_amr = self
                .data
                .get("SpatialGrid")
                .unwrap()
                .max_refinement_level
                .clone()
                .unwrap()
                .parse::<u32>()
                .unwrap();

            let mut nx = self.read_parameter("xcells_ini").unwrap() as usize;
            let mut ny = self.read_parameter("ycells_ini").unwrap() as usize;
            let mut nz = self.read_parameter("zcells_ini").unwrap() as usize;
            nx *= usize::pow(2, max_amr);
            ny *= usize::pow(2, max_amr);
            nz *= usize::pow(2, max_amr);
            todo!();
        }

        pub fn get_data_info(&self, name: &str) -> Option<VlsvDataset> {
            let entry = self.data.get(name)?;

            let offset = entry
                .offset
                .clone()?
                .parse::<usize>()
                .map_err(|err| eprintln!("ERROR: could not parse offset for '{}': {:?}", name, err))
                .ok()?;

            let datasize = entry
                .datasize
                .clone()?
                .parse::<usize>()
                .map_err(|err| {
                    eprintln!("ERROR: could not parse datasize for '{}': {:?}", name, err)
                })
                .ok()?;

            let vectorsize = entry
                .vectorsize
                .clone()?
                .parse::<usize>()
                .map_err(|err| {
                    eprintln!(
                        "ERROR: could not parse vectorsize for '{}': {:?}",
                        name, err
                    )
                })
                .ok()?;

            let arraysize = entry
                .arraysize
                .clone()?
                .parse::<usize>()
                .map_err(|err| {
                    eprintln!("ERROR: could not parse arraysize for '{}': {:?}", name, err)
                })
                .ok()?;

            let datatype = entry.datatype.clone()?;

            Some(VlsvDataset {
                offset,
                arraysize,
                vectorsize,
                datasize,
                datatype,
            })
        }
    }
}
