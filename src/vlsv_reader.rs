#[allow(dead_code)]
pub mod vlsv_reader {
    use bytemuck::{Pod, cast_slice};
    use memmap2::Mmap;
    use ndarray::{Array4, Order, s};
    use num_traits::{self, Zero};
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Deserialize, Debug, Clone)]
    #[serde(rename_all = "UPPERCASE")]
    pub struct Variable {
        #[serde(rename = "arraysize")]
        pub arraysize: Option<String>,
        #[serde(rename = "datasize")]
        pub datasize: Option<String>,
        #[serde(rename = "datatype")]
        pub datatype: Option<String>,
        #[serde(rename = "mesh")]
        pub mesh: Option<String>,
        #[serde(rename = "name")]
        pub name: Option<String>,
        #[serde(rename = "vectorsize")]
        pub vectorsize: Option<String>,
        #[serde(rename = "max_refinement_level")]
        pub max_refinement_level: Option<String>,
        #[serde(rename = "unit")]
        pub unit: Option<String>,
        #[serde(rename = "unitConversion")]
        pub unit_conversion: Option<String>,
        #[serde(rename = "unitLaTeX")]
        pub unit_latex: Option<String>,
        #[serde(rename = "variableLaTeX")]
        pub variable_latex: Option<String>,
        #[serde(rename = "$value")]
        pub offset: Option<String>,
    }

    #[derive(Deserialize, Debug)]
    pub struct VlsvRoot {
        #[serde(rename = "VARIABLE")]
        pub variables: Vec<Variable>,

        #[serde(rename = "PARAMETER")]
        pub parameters: Vec<Variable>,

        #[serde(rename = "BLOCKIDS")]
        pub blockids: Option<Vec<Variable>>,

        #[serde(rename = "BLOCKSPERCELL")]
        pub blockspercell: Option<Vec<Variable>>,

        #[serde(rename = "BLOCKVARIABLE")]
        pub blockvariable: Option<Vec<Variable>>,

        #[serde(rename = "CELLSWITHBLOCKS")]
        pub cellswithblocks: Option<Vec<Variable>>,

        #[serde(rename = "CONFIG")]
        pub config: Option<Vec<Variable>>,

        #[serde(rename = "MESH")]
        pub mesh: Option<Vec<Variable>>,

        #[serde(rename = "MESH_BBOX")]
        pub mesh_bbox: Option<Vec<Variable>>,

        #[serde(rename = "MESH_DECOMPOSITION")]
        pub mesh_decomposition: Option<Vec<Variable>>,

        #[serde(rename = "MESH_DOMAIN_SIZES")]
        pub mesh_domain_sizes: Option<Vec<Variable>>,

        #[serde(rename = "MESH_GHOST_DOMAINS")]
        pub mesh_ghost_domains: Option<Vec<Variable>>,

        #[serde(rename = "MESH_GHOST_LOCALIDS")]
        pub mesh_ghost_localids: Option<Vec<Variable>>,

        #[serde(rename = "MESH_NODE_CRDS_X")]
        pub mesh_node_crds_x: Option<Vec<Variable>>,

        #[serde(rename = "MESH_NODE_CRDS_Y")]
        pub mesh_node_crds_y: Option<Vec<Variable>>,

        #[serde(rename = "MESH_NODE_CRDS_Z")]
        pub mesh_node_crds_z: Option<Vec<Variable>>,
    }

    #[derive(Debug, Clone)]
    pub struct VlsvDataset {
        pub offset: usize,
        pub arraysize: usize,
        pub vectorsize: usize,
        pub datasize: usize,
        pub datatype: String,
    }

    impl TryFrom<&Variable> for VlsvDataset {
        type Error = String;

        fn try_from(var: &Variable) -> Result<Self, Self::Error> {
            Ok(Self {
                offset: var
                    .offset
                    .as_deref()
                    .ok_or("Missing offset")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid offset: {}", e))?,

                arraysize: var
                    .arraysize
                    .as_deref()
                    .ok_or("Missing arraysize")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid arraysize: {}", e))?,

                vectorsize: var
                    .vectorsize
                    .as_deref()
                    .unwrap_or("1")
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid vectorsize: {}", e))?,

                datasize: var
                    .datasize
                    .as_deref()
                    .ok_or("Missing datasize")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid datasize: {}", e))?,

                datatype: var.datatype.as_ref().ok_or("Missing datatype")?.clone(),
            })
        }
    }

    impl TryFrom<Variable> for VlsvDataset {
        type Error = String;

        fn try_from(var: Variable) -> Result<Self, Self::Error> {
            Ok(Self {
                offset: var
                    .offset
                    .as_ref()
                    .ok_or("Missing offset")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid offset: {}", e))?,

                arraysize: var
                    .arraysize
                    .as_ref()
                    .ok_or("Missing arraysize")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid arraysize: {}", e))?,

                vectorsize: var
                    .vectorsize
                    .as_ref()
                    .unwrap_or(&"1".to_string())
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid vectorsize: {}", e))?,

                datasize: var
                    .datasize
                    .as_ref()
                    .ok_or("Missing datasize")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid datasize: {}", e))?,

                datatype: var.datatype.clone().ok_or("Missing datatype")?,
            })
        }
    }

    #[derive(Debug)]
    pub struct VlsvFile {
        pub filename: String,
        pub variables: HashMap<String, Variable>,
        pub parameters: HashMap<String, Variable>,
        pub xml: String,
        pub memmap: Mmap,
        pub root: VlsvRoot,
    }

    impl VlsvFile {
        pub fn new(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
            let f = std::fs::File::open(filename)?;
            let mmap = unsafe { Mmap::map(&f)? };

            let footer_offset = i64::from_ne_bytes(mmap[8..16].try_into()?) as usize;
            let xml_string = std::str::from_utf8(&mmap[footer_offset..])?.to_string();

            let root: VlsvRoot = serde_xml_rs::from_str(&xml_string)?;

            let vars = root
                .variables
                .iter()
                .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                .collect::<HashMap<_, _>>();

            let params = root
                .parameters
                .iter()
                .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                .collect::<HashMap<_, _>>();

            Ok(Self {
                filename: filename.to_string(),
                variables: vars,
                parameters: params,
                xml: xml_string,
                memmap: mmap,
                root, // reuse the parsed root
            })
        }

        pub fn print_variables(&self) {
            for key in self.variables.keys() {
                println!("{}", key);
            }
        }

        pub fn read_scalar_parameter(&self, name: &str) -> Option<f64> {
            let info = self.get_dataset(name)?;
            assert!(info.vectorsize == 1);
            assert!(info.arraysize == 1);
            let expected_bytes = info.datasize * info.vectorsize;
            assert!(
                info.offset + expected_bytes <= self.memmap.len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let src_bytes = &self.memmap[info.offset..info.offset + expected_bytes];
            let retval = match info.datasize {
                8 => {
                    let mut buffer: [u8; 8] = [0; 8];
                    buffer.copy_from_slice(cast_slice(src_bytes));

                    match info.datatype.as_str() {
                        "float" => f64::from_ne_bytes(buffer),
                        "uint" => usize::from_ne_bytes(buffer) as f64,
                        "int" => i64::from_ne_bytes(buffer) as f64,
                        _ => panic!("Only matched against uint and float"),
                    }
                }
                4 => {
                    let mut buffer: [u8; 4] = [0; 4];
                    buffer.copy_from_slice(cast_slice(src_bytes));
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
            let info = self.get_dataset(name)?;
            let expected_bytes = info.datasize * info.vectorsize;
            assert!(
                info.offset + expected_bytes <= self.memmap.len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let src_bytes = &self.memmap[info.offset..info.offset + expected_bytes];
            let mut buffer: Vec<u8> = Vec::with_capacity(info.arraysize);
            unsafe {
                buffer.set_len(info.arraysize);
            }
            buffer.copy_from_slice(cast_slice(src_bytes));
            let cfgfile = String::from_utf8(buffer).unwrap();
            Some(cfgfile)
        }

        pub fn read_version(&self) -> Option<String> {
            let name = "version_information";
            let info = self.get_dataset(name)?;
            let expected_bytes = info.datasize * info.vectorsize;
            assert!(
                info.offset + expected_bytes <= self.memmap.len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let src_bytes = &self.memmap[info.offset..info.offset + expected_bytes];
            let mut buffer: Vec<u8> = Vec::with_capacity(info.arraysize);
            unsafe {
                buffer.set_len(info.arraysize);
            }
            buffer.copy_from_slice(cast_slice(src_bytes));
            let cfgfile = String::from_utf8(buffer).unwrap();
            Some(cfgfile)
        }

        fn read_variable_into<T: Sized + Pod>(
            &self,
            name: &str,
            dataset: Option<VlsvDataset>,
            dst: &mut [T],
        ) {
            let info = dataset.unwrap_or_else(|| self.get_dataset(name).unwrap());
            let size = dst.len();
            let expected_bytes = info.datasize * size;
            assert!(
                info.offset + expected_bytes <= self.memmap.len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let src_bytes = &self.memmap[info.offset..info.offset + expected_bytes];

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
                        _dst.copy_from_slice(cast_slice(src_bytes));
                        // f.read_exact_at(cast_slice_mut(_dst.as_mut_slice()), info.offset as u64)
                        //     .unwrap();
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
                        _dst.copy_from_slice(cast_slice(src_bytes));
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
                //Handle alignement here
                let (head, aligned, tail) = unsafe { src_bytes.align_to::<T>() };
                if head.is_empty() && tail.is_empty() {
                    assert_eq!(aligned.len(), size);
                    dst.copy_from_slice(aligned);
                } else {
                    for i in 0..size {
                        let ptr = unsafe { src_bytes.as_ptr().add(i * std::mem::size_of::<T>()) };
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                ptr,
                                dst.as_mut_ptr().add(i).cast::<u8>(),
                                std::mem::size_of::<T>(),
                            );
                        }
                    }
                }
            }
        }

        pub fn get_wid(&self) -> Option<usize> {
            let wid = {
                let dataset: VlsvDataset =
                    self.root.blockvariable.as_ref()?.first()?.try_into().ok()?;
                (dataset.vectorsize as f64).cbrt()
            };
            Some(wid as usize)
        }

        pub fn get_vspace_mesh_bbox(&self, pop: &str) -> Option<(usize, usize, usize)> {
            let nvx = self
                .root
                .mesh_node_crds_x
                .as_ref()
                .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))
                .and_then(|var| TryInto::<VlsvDataset>::try_into(var).ok())
                .map(|ds| ds.arraysize - 1)
                .or_else(|| {
                    eprintln!("Failed to get MESH_NODE_CRDS_X for mesh = {pop}");
                    None
                })?;
            let nvy = self
                .root
                .mesh_node_crds_y
                .as_ref()
                .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))
                .and_then(|var| TryInto::<VlsvDataset>::try_into(var).ok())
                .map(|ds| ds.arraysize - 1)
                .or_else(|| {
                    eprintln!("Failed to get MESH_NODE_CRDS_Y for mesh = {pop}");
                    None
                })?;
            let nvz = self
                .root
                .mesh_node_crds_z
                .as_ref()
                .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))
                .and_then(|var| TryInto::<VlsvDataset>::try_into(var).ok())
                .map(|ds| ds.arraysize - 1)
                .or_else(|| {
                    eprintln!("Failed to get MESH_NODE_CRDS_Z for mesh = {pop}");
                    None
                })?;
            Some((nvx, nvy, nvz))
        }

        pub fn get_vspace_mesh_extents(&self, pop: &str) -> Option<(f64, f64, f64, f64, f64, f64)> {
            let nodes_x = TryInto::<VlsvDataset>::try_into(
                self.root
                    .mesh_node_crds_x
                    .as_ref()
                    .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))?,
            )
            .ok()?;
            let nodes_y = TryInto::<VlsvDataset>::try_into(
                self.root
                    .mesh_node_crds_y
                    .as_ref()
                    .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))?,
            )
            .ok()?;
            let nodes_z = TryInto::<VlsvDataset>::try_into(
                self.root
                    .mesh_node_crds_z
                    .as_ref()
                    .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))?,
            )
            .ok()?;
            assert!(nodes_x.datasize == 8, "Expected f64 for mesh node coords");
            let mut datax: Vec<f64> = vec![0_f64; nodes_x.arraysize];
            let mut datay: Vec<f64> = vec![0_f64; nodes_y.arraysize];
            let mut dataz: Vec<f64> = vec![0_f64; nodes_z.arraysize];
            self.read_variable_into::<f64>("", Some(nodes_x), &mut datax);
            self.read_variable_into::<f64>("", Some(nodes_y), &mut datay);
            self.read_variable_into::<f64>("", Some(nodes_z), &mut dataz);
            Some((
                datax.first().copied()?,
                datay.first().copied()?,
                dataz.first().copied()?,
                datax.last().copied()?,
                datay.last().copied()?,
                dataz.last().copied()?,
            ))
        }

        pub fn get_domain_decomposition(&self) -> Option<[usize; 3]> {
            let mut decomp: [u32; 3] = [0; 3];
            let decomposition: VlsvDataset = self
                .root
                .mesh_decomposition
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .and_then(|v| v.try_into().ok())?;
            self.read_variable_into::<u32>("", Some(decomposition), decomp.as_mut_slice());
            Some([decomp[0] as usize, decomp[1] as usize, decomp[2] as usize])
        }

        pub fn get_max_amr_refinement(&self) -> Option<u32> {
            self.root.mesh.as_ref().and_then(|meshes| {
                meshes
                    .iter()
                    .find_map(|v| v.max_refinement_level.as_ref()?.parse::<u32>().ok())
            })
        }

        pub fn get_writting_tasks(&self) -> Option<usize> {
            Some(self.read_scalar_parameter("numWritingRanks")? as usize)
        }

        pub fn get_spatial_mesh_bbox(&self) -> Option<(usize, usize, usize)> {
            let max_amr = self.get_max_amr_refinement()?;
            let mut nx = self.read_scalar_parameter("xcells_ini")? as usize;
            let mut ny = self.read_scalar_parameter("ycells_ini")? as usize;
            let mut nz = self.read_scalar_parameter("zcells_ini")? as usize;
            nx *= usize::pow(2, max_amr);
            ny *= usize::pow(2, max_amr);
            nz *= usize::pow(2, max_amr);
            Some((nx, ny, nz))
        }

        fn get_dataset(&self, name: &str) -> Option<VlsvDataset> {
            self.variables
                .get(name)
                .or_else(|| self.parameters.get(name))
                .or_else(|| {
                    eprintln!("'{}' not found in VARIABLES or PARAMETERS", name);
                    None
                })?
                .clone()
                .try_into()
                .ok()
        }

        pub fn read_fsgrid_variable<T: Pod + Zero>(&self, name: &str) -> Option<Array4<T>> {
            if name[0..3] != *"fg_" {
                panic!("ERROR: Variable {} is not an fs_grid variable!", name);
            }
            let info = self.get_dataset(name)?;
            let decomp = self.get_domain_decomposition()?;
            let ntasks = self.get_writting_tasks()?;
            let (nx, ny, nz) = self.get_spatial_mesh_bbox()?;

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

            let mut var = ndarray::Array2::<T>::zeros((nx * ny * nz, info.vectorsize));
            self.read_variable_into::<T>(name, None, var.as_slice_mut().unwrap());
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

        pub fn read_vdf(&self, cid: usize, pop: &str) -> Option<Vec<f32>> {
            let blockspercell: VlsvDataset =
                self.root.blockspercell.as_ref()?.first()?.try_into().ok()?;
            let cellswithblocks: VlsvDataset = self
                .root
                .cellswithblocks
                .as_ref()?
                .first()?
                .try_into()
                .ok()?;
            let blockids: VlsvDataset = self.root.blockids.as_ref()?.first()?.try_into().ok()?;
            let blockvariable: VlsvDataset =
                self.root.blockvariable.as_ref()?.first()?.try_into().ok()?;
            assert!(
                blockvariable.datasize == 4,
                "VDF is not f32! This is not handled yet"
            );

            let wid = self.get_wid()?;
            let wid3 = wid.pow(3);
            let (nvx, nvy, nvz) = self.get_vspace_mesh_bbox(pop)?;
            let (mx, my, mz) = (nvx / wid, nvy / wid, nvz / wid);

            let mut cids_with_blocks: Vec<usize> = vec![0; cellswithblocks.arraysize];
            let mut blocks_per_cell: Vec<u32> = vec![0; blockspercell.arraysize];
            self.read_variable_into::<usize>("", Some(cellswithblocks), &mut cids_with_blocks);
            self.read_variable_into::<u32>("", Some(blockspercell), &mut blocks_per_cell);

            let index = cids_with_blocks.iter().position(|v| *v == cid)?;
            let read_size = blocks_per_cell[index] as usize;
            let start_block = blocks_per_cell[..index]
                .iter()
                .map(|&x| x as usize)
                .sum::<usize>();

            let read_chunk =
                |ds: &VlsvDataset, elem_offset: usize, elem_count: usize, dst_bytes: &mut [u8]| {
                    let byte_offset = ds.offset + elem_offset * ds.vectorsize * ds.datasize;
                    let byte_len = elem_count * ds.vectorsize * ds.datasize;
                    assert!(byte_offset + byte_len <= self.memmap.len(), "Out-of-bounds");
                    let src = &self.memmap[byte_offset..byte_offset + byte_len];
                    dst_bytes.copy_from_slice(src);
                };

            let mut block_ids: Vec<u32> = vec![0; read_size];
            {
                let dst_bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut block_ids);
                read_chunk(&blockids, start_block, read_size, dst_bytes);
            }

            let mut blocks: Vec<f32> = vec![0.0; read_size * wid3];
            {
                let dst_bytes = bytemuck::cast_slice_mut::<f32, u8>(&mut blocks);
                read_chunk(&blockvariable, start_block, read_size, dst_bytes);
            }

            let id2ijk = |id: usize| -> (usize, usize, usize) {
                let plane = mx * my;
                assert!(id < plane * mz, "block_id out of bounds");
                let k = id / plane;
                let rem = id % plane;
                let j = rem / mx;
                let i = rem % mx;
                (i, j, k)
            };

            let ijk2id = |i: usize, j: usize, k: usize| -> usize {
                assert!(i < nvx && j < nvy && k < nvz, "out of bounds {i},{j},{k}");
                i + j * nvx + k * (nvx * nvy)
            };

            let mut vdf = vec![0.0f32; nvx * nvy * nvz];

            for (block_idx, &bid_u32) in block_ids.iter().enumerate() {
                let bid = bid_u32 as usize;
                let (bi, bj, bk) = id2ijk(bid);
                let block_buf = &blocks[block_idx * wid3..(block_idx + 1) * wid3];

                for dk in 0..wid {
                    for dj in 0..wid {
                        for di in 0..wid {
                            let local_id = di + dj * wid + dk * wid * wid;
                            let val = block_buf[local_id];
                            let gi = bi * wid + di;
                            let gj = bj * wid + dj;
                            let gk = bk * wid + dk;
                            let gid = ijk2id(gi, gj, gk);
                            vdf[gid] = val;
                        }
                    }
                }
            }
            Some(vdf)
        }
    }
}
