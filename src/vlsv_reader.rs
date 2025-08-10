#[allow(dead_code)]
pub mod vlsv_reader {
    const VLSV_FOOTER_LOC_START: usize = 8;
    const VLSV_FOOTER_LOC_END: usize = 16;
    use bytemuck::{Pod, cast_slice};
    use core::convert::TryInto;
    use memmap2::Mmap;
    use ndarray::{Array3, Array4, Axis, Order, s};
    use num_traits::{Num, NumCast, Zero};
    use regex::Regex;
    use serde::Deserialize;
    use std::{collections::HashMap, str::FromStr};

    #[derive(Debug, Clone)]
    pub struct VlsvDataset {
        pub offset: usize,
        pub arraysize: usize,
        pub vectorsize: usize,
        pub datasize: usize,
        pub datatype: String,
        grid: Option<VlasiatorGrid>,
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
            let mmap = unsafe { Mmap::map(&std::fs::File::open(filename)?)? };
            let xml_string = {
                let footer_offset: usize = usize::from_ne_bytes(
                    mmap[VLSV_FOOTER_LOC_START..VLSV_FOOTER_LOC_END].try_into()?,
                );
                std::str::from_utf8(&mmap[footer_offset..])?.to_string()
            };
            let root: VlsvRoot = serde_xml_rs::from_str(&xml_string)?;

            let vars: HashMap<String, Variable> = root
                .variables
                .iter()
                .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                .chain(
                    [
                        ("CONFIG", "config_file", "Config not available!"),
                        ("VERSION", "version_information", "Version not available!"),
                    ]
                    .into_iter()
                    .filter_map(|(tag, section, warn_msg)| {
                        match read_tag(&xml_string, tag, None, Some(section)) {
                            Some(x) => Some((x.name.clone().unwrap(), x)),
                            None => {
                                eprintln!("{}", warn_msg);
                                None
                            }
                        }
                    }),
                )
                .collect();

            let params: HashMap<String, Variable> = root
                .parameters
                .iter()
                .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                .collect();

            Ok(Self {
                filename: filename.to_string(),
                variables: vars,
                parameters: params,
                xml: xml_string,
                memmap: mmap,
                root,
            })
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
            const NAME: &str = "config_file";
            let info = self.get_dataset(NAME)?;
            let expected_bytes = info.datasize * info.vectorsize * info.arraysize;
            assert!(
                info.offset + expected_bytes <= self.memmap.len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let bytes = &self.memmap[info.offset..info.offset + expected_bytes];
            let cfgfile = std::str::from_utf8(bytes).map(|s| s.to_owned()).ok()?;
            Some(cfgfile)
        }

        pub fn read_version(&self) -> Option<String> {
            const NAME: &str = "version_information";
            let info = self.get_dataset(NAME)?;
            let expected_bytes = info.datasize * info.vectorsize * info.arraysize;
            assert!(
                info.offset + expected_bytes <= self.memmap.len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let bytes = &self.memmap[info.offset..info.offset + expected_bytes];
            let version = std::str::from_utf8(bytes).map(|s| s.to_owned()).ok()?;
            Some(version)
        }

        fn read_variable_into<T: Sized + Pod + TypeTag>(
            &self,
            name: Option<&str>,
            dataset: Option<VlsvDataset>,
            dst: &mut [T],
        ) {
            //Sanity check
            let info = match (name, dataset) {
                (None, None) => {
                    panic!("Tried to call read_variable_into with no Dataset and no Variable name")
                }
                (Some(_), Some(_)) => {
                    panic!("Tried to call read_variable_into with both Name and Dataset specified ")
                }
                (Some(name), None) => self
                    .get_dataset(name)
                    .expect("No data set found for variable: {name}"),
                (None, Some(d)) => d,
            };
            let expected_bytes = info.datasize * info.arraysize * info.vectorsize;
            let end = info.offset + expected_bytes;
            let src_bytes = &self.memmap[info.offset..end];
            let dst_bytes = bytemuck::cast_slice_mut::<T, u8>(dst);

            /*
               === DYNAMIC DISPATCH RULES ===
               Floating point conversions ONLY!!!
               Not doing any int conversions becasue the user can just read the correct type.
               For floats it makes sense as we may need to read f64 fields as f32 for memory savings.
            */
            let type_on_disk = info.datatype;
            let type_of_t = T::type_name();
            //T=>T
            if type_on_disk == type_of_t && info.datasize == std::mem::size_of::<T>() {
                dst_bytes.copy_from_slice(src_bytes);
                return;
            }
            unsafe {
                //f32=>f64
                if type_on_disk == "float"
                    && info.datasize == 4
                    && type_of_t == "float"
                    && std::mem::size_of::<T>() == 8
                {
                    let dst_f64: &mut [f64] =
                        std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f64>(), dst.len());

                    for (i, bytes) in src_bytes.chunks_exact(4).enumerate() {
                        let v64 = f32::from_le_bytes(bytes.try_into().unwrap()) as f64;
                        dst_f64[i] = v64;
                    }
                    return;
                }
                //f64=>f32
                if type_on_disk == "float"
                    && info.datasize == 8
                    && std::mem::size_of::<T>() == 4
                    && type_of_t == "float"
                {
                    let dst_f32: &mut [f32] =
                        std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f32>(), dst.len());

                    for (i, bytes) in src_bytes.chunks_exact(8).enumerate() {
                        let v32 = f64::from_le_bytes(bytes.try_into().unwrap()) as f32;
                        dst_f32[i] = v32;
                    }
                    return;
                }
            }
            //Any other mismatch panics!
            panic!(
                "Incompatible reads: {type_on_disk}({}) => {type_of_t}({}) ",
                info.datasize,
                std::mem::size_of::<T>()
            );
        }

        // #[deprecated(note = "TODO: This reads WID from the first population file. Reconsider!")]
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

        pub fn get_spatial_mesh_extents(&self) -> Option<(f64, f64, f64, f64, f64, f64)> {
            let nodes_x = TryInto::<VlsvDataset>::try_into(
                self.root.mesh_node_crds_x.as_ref().and_then(|meshes| {
                    meshes
                        .iter()
                        .find(|v| v.mesh.as_deref() == Some("SpatialGrid"))
                })?,
            )
            .ok()?;
            let nodes_y = TryInto::<VlsvDataset>::try_into(
                self.root.mesh_node_crds_y.as_ref().and_then(|meshes| {
                    meshes
                        .iter()
                        .find(|v| v.mesh.as_deref() == Some("SpatialGrid"))
                })?,
            )
            .ok()?;
            let nodes_z = TryInto::<VlsvDataset>::try_into(
                self.root.mesh_node_crds_z.as_ref().and_then(|meshes| {
                    meshes
                        .iter()
                        .find(|v| v.mesh.as_deref() == Some("SpatialGrid"))
                })?,
            )
            .ok()?;
            assert!(nodes_x.datasize == 8, "Expected f64 for mesh node coords");
            let mut datax: Vec<f64> = vec![0_f64; nodes_x.arraysize];
            let mut datay: Vec<f64> = vec![0_f64; nodes_y.arraysize];
            let mut dataz: Vec<f64> = vec![0_f64; nodes_z.arraysize];
            self.read_variable_into::<f64>(None, Some(nodes_x), &mut datax);
            self.read_variable_into::<f64>(None, Some(nodes_y), &mut datay);
            self.read_variable_into::<f64>(None, Some(nodes_z), &mut dataz);
            Some((
                datax.first().copied()?,
                datay.first().copied()?,
                dataz.first().copied()?,
                datax.last().copied()?,
                datay.last().copied()?,
                dataz.last().copied()?,
            ))
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
            self.read_variable_into::<f64>(None, Some(nodes_x), &mut datax);
            self.read_variable_into::<f64>(None, Some(nodes_y), &mut datay);
            self.read_variable_into::<f64>(None, Some(nodes_z), &mut dataz);
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
            let mut decomp: [i32; 3] = [0; 3];
            let decomposition: VlsvDataset = self
                .root
                .mesh_decomposition
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .and_then(|v| v.try_into().ok())?;
            self.read_variable_into::<i32>(None, Some(decomposition), decomp.as_mut_slice());
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

        pub fn get_dataset(&self, name: &str) -> Option<VlsvDataset> {
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

        pub fn read_vg_variable_as_fg<
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag,
        >(
            &self,
            name: &str,
            op: Option<i32>,
        ) -> Option<ndarray::Array4<T>> {
            let info = self.get_dataset(name)?;
            if info.grid.clone()? != VlasiatorGrid::SPATIALGRID {
                return None;
            }
            let vecsz = info.vectorsize;
            let x0 = self.read_scalar_parameter("xcells_ini")? as usize;
            let y0 = self.read_scalar_parameter("ycells_ini")? as usize;
            let z0 = self.read_scalar_parameter("zcells_ini")? as usize;
            let lmax = self.get_max_amr_refinement()?;
            let cellid_ds = self.get_dataset("CellID")?;
            let mut cell_ids = vec![0u64; cellid_ds.arraysize];
            self.read_variable_into::<u64>(None, Some(cellid_ds), &mut cell_ids);
            let n_cells = info.arraysize;
            let mut vg_rows = vec![T::default(); n_cells * vecsz];
            self.read_variable_into::<T>(None, Some(info), vg_rows.as_mut_slice());
            let mut ordered_var = vg_variable_to_fg(&cell_ids, &vg_rows, vecsz, x0, y0, z0, lmax);
            apply_op_in_place::<T>(&mut ordered_var, op);
            Some(ordered_var)
        }

        pub fn read_fsgrid_variable<
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag,
        >(
            &self,
            name: &str,
            op: Option<i32>,
        ) -> Option<Array4<T>> {
            let info = self.get_dataset(name)?;
            if info.grid? != VlasiatorGrid::FSGRID {
                return None;
            }
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
            self.read_variable_into::<T>(Some(name), None, var.as_slice_mut().unwrap());
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

            apply_op_in_place::<T>(&mut ordered_var, op);
            Some(ordered_var)
        }

        pub fn read_vdf<T>(&self, cid: usize, pop: &str) -> Option<Array4<T>>
        where
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag,
        {
            let blockspercell = TryInto::<VlsvDataset>::try_into(
                self.root
                    .blockspercell
                    .as_ref()?
                    .iter()
                    .find(|v| v.name.as_deref() == Some(pop))?,
            )
            .ok()?;

            let cellswithblocks = TryInto::<VlsvDataset>::try_into(
                self.root
                    .cellswithblocks
                    .as_ref()?
                    .iter()
                    .find(|v| v.name.as_deref() == Some(pop))?,
            )
            .ok()?;

            let blockids = TryInto::<VlsvDataset>::try_into(
                self.root
                    .blockids
                    .as_ref()?
                    .iter()
                    .find(|v| v.name.as_deref() == Some(pop))?,
            )
            .ok()?;

            let blockvariable = TryInto::<VlsvDataset>::try_into(
                self.root
                    .blockvariable
                    .as_ref()?
                    .iter()
                    .find(|v| v.name.as_deref() == Some(pop))?,
            )
            .ok()?;

            let wid = self.get_wid()?;
            let wid3 = wid.pow(3);
            let (nvx, nvy, nvz) = self.get_vspace_mesh_bbox(pop)?;
            let (mx, my, mz) = (nvx / wid, nvy / wid, nvz / wid);

            let mut cids_with_blocks: Vec<usize> = vec![0; cellswithblocks.arraysize];
            let mut blocks_per_cell: Vec<u32> = vec![0; blockspercell.arraysize];
            self.read_variable_into::<usize>(None, Some(cellswithblocks), &mut cids_with_blocks);
            self.read_variable_into::<u32>(None, Some(blockspercell), &mut blocks_per_cell);

            let index = cids_with_blocks.iter().position(|&v| v == cid)?;
            let read_size = blocks_per_cell[index] as usize;
            let start_block = blocks_per_cell[..index]
                .iter()
                .map(|&x| x as usize)
                .sum::<usize>();

            fn slice_ds(ds: &VlsvDataset, elem_offset: usize, elem_count: usize) -> VlsvDataset {
                let mut sub = ds.clone();
                sub.offset = ds.offset + elem_offset * ds.vectorsize * ds.datasize;
                sub.arraysize = elem_count;
                sub
            }

            // Read block data (T)
            let mut block_ids: Vec<u32> = vec![0; read_size * blockids.vectorsize];
            let blockids_slice = slice_ds(&blockids, start_block, read_size);
            self.read_variable_into::<u32>(None, Some(blockids_slice), &mut block_ids);

            let mut blocks: Vec<T> = vec![T::default(); read_size * blockvariable.vectorsize];
            let blockvar_slice = slice_ds(&blockvariable, start_block, read_size);
            self.read_variable_into::<T>(None, Some(blockvar_slice), &mut blocks);

            let id2ijk = |id: usize| -> (usize, usize, usize) {
                let plane = mx * my;
                assert!(id < plane * mz, "GID out of bounds");
                let k = id / plane;
                let rem = id % plane;
                let j = rem / mx;
                let i = rem % mx;
                (i, j, k)
            };

            let mut vdf = Array4::<T>::zeros((nvx, nvy, nvz, 1));
            for (block_idx, &bid_u32) in block_ids.iter().enumerate() {
                let bid = bid_u32 as usize;
                let (bi, bj, bk) = id2ijk(bid);
                let block_buf = &blocks[block_idx * wid3..(block_idx + 1) * wid3];

                for dk in 0..wid {
                    for dj in 0..wid {
                        for di in 0..wid {
                            let local_id = di + dj * wid + dk * wid * wid;
                            let gi = bi * wid + di;
                            let gj = bj * wid + dj;
                            let gk = bk * wid + dk;
                            vdf[(gi, gj, gk, 0)] = block_buf[local_id];
                        }
                    }
                }
            }
            Some(vdf)
        }

        pub fn read_vdf_into<T>(
            &self,
            cid: usize,
            pop: &str,
            target: &mut Array4<T>,
            target_extent: (f64, f64, f64, f64, f64, f64),
        ) -> Option<()>
        where
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag,
        {
            let vdf: Array4<T> = self.read_vdf::<T>(cid, pop)?;
            let (sx, sy, sz, sc) = vdf.dim();
            let (tx, ty, tz, tc) = target.dim();
            assert!(sc == 1 && tc == 1);
            let (sxmin, symin, szmin, sxmax, symax, szmax) = self.get_vspace_mesh_extents(pop)?;
            let (txmin, tymin, tzmin, txmax, tymax, tzmax) = target_extent;

            let sdx = (sxmax - sxmin) / (sx as f64);
            let sdy = (symax - symin) / (sy as f64);
            let sdz = (szmax - szmin) / (sz as f64);
            let tdx = (txmax - txmin) / (tx as f64);
            let tdy = (tymax - tymin) / (ty as f64);
            let tdz = (tzmax - tzmin) / (tz as f64);

            let in_bounds = |i: isize, n: usize| i >= 0 && (i as usize) < n;
            let tri_sample = |ux: f64, uy: f64, uz: f64, chan: usize| -> f64 {
                let ix0 = ux.floor() as isize;
                let iy0 = uy.floor() as isize;
                let iz0 = uz.floor() as isize;
                let fx = ux - ix0 as f64;
                let fy = uy - iy0 as f64;
                let fz = uz - iz0 as f64;
                if !(in_bounds(ix0, sx)
                    && in_bounds(ix0 + 1, sx)
                    && in_bounds(iy0, sy)
                    && in_bounds(iy0 + 1, sy)
                    && in_bounds(iz0, sz)
                    && in_bounds(iz0 + 1, sz))
                {
                    return 0.0;
                }

                let fetch = |i, j, k| -> f64 { NumCast::from(vdf[(i, j, k, chan)]).unwrap_or(0.0) };
                let c000 = fetch(ix0 as usize, iy0 as usize, iz0 as usize);
                let c100 = fetch((ix0 + 1) as usize, iy0 as usize, iz0 as usize);
                let c010 = fetch(ix0 as usize, (iy0 + 1) as usize, iz0 as usize);
                let c110 = fetch((ix0 + 1) as usize, (iy0 + 1) as usize, iz0 as usize);
                let c001 = fetch(ix0 as usize, iy0 as usize, (iz0 + 1) as usize);
                let c101 = fetch((ix0 + 1) as usize, iy0 as usize, (iz0 + 1) as usize);
                let c011 = fetch(ix0 as usize, (iy0 + 1) as usize, (iz0 + 1) as usize);
                let c111 = fetch((ix0 + 1) as usize, (iy0 + 1) as usize, (iz0 + 1) as usize);
                let c00 = c000 * (1.0 - fx) + c100 * fx;
                let c01 = c001 * (1.0 - fx) + c101 * fx;
                let c10 = c010 * (1.0 - fx) + c110 * fx;
                let c11 = c011 * (1.0 - fx) + c111 * fx;
                let c0 = c00 * (1.0 - fy) + c10 * fy;
                let c1 = c01 * (1.0 - fy) + c11 * fy;

                c0 * (1.0 - fz) + c1 * fz
            };

            let to_src_u = |x_t: f64, xmin_s: f64, sdx: f64| -> f64 { (x_t - xmin_s) / sdx - 0.5 };

            let c_use = sc.min(tc);
            target.fill(T::zero());

            for iz in 0..tz {
                let zc = tzmin + (iz as f64 + 0.5) * tdz;
                let uz = to_src_u(zc, szmin, sdz);

                for iy in 0..ty {
                    let yc = tymin + (iy as f64 + 0.5) * tdy;
                    let uy = to_src_u(yc, symin, sdy);

                    for ix in 0..tx {
                        let xc = txmin + (ix as f64 + 0.5) * tdx;
                        let ux = to_src_u(xc, sxmin, sdx);

                        let mut total_value = 0.0;
                        let mut count = 0;

                        for iz_src in (ux.floor() as isize)..=(ux.ceil() as isize) {
                            for iy_src in (uy.floor() as isize)..=(uy.ceil() as isize) {
                                for ix_src in (uz.floor() as isize)..=(uz.ceil() as isize) {
                                    if in_bounds(ix_src, sx)
                                        && in_bounds(iy_src, sy)
                                        && in_bounds(iz_src, sz)
                                    {
                                        total_value += tri_sample(ux, uy, uz, 0);
                                        count += 1;
                                    }
                                }
                            }
                        }

                        let average_value = if count > 0 {
                            T::from(total_value).unwrap() / T::from(count).unwrap()
                        } else {
                            T::zero()
                        };

                        for c in 0..c_use {
                            target[(ix, iy, iz, c)] = average_value;
                        }
                    }
                }
            }

            Some(())
        }

        pub fn read_variable<T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag>(
            &self,
            name: &str,
            op: Option<i32>,
        ) -> Option<ndarray::Array4<T>> {
            self.read_fsgrid_variable::<T>(name, op)
                .or_else(|| self.read_vg_variable_as_fg::<T>(name, op))
        }
    }

    fn read_tag(xml: &str, tag: &str, mesh: Option<&str>, name: Option<&str>) -> Option<Variable> {
        let re_normal = Regex::new(&format!(
            r#"(?s)<{t}\b([^>]*)>([^<]*)</{t}>"#,
            t = regex::escape(tag)
        ))
        .unwrap();
        let re_self =
            Regex::new(&format!(r#"(?s)<{t}\b([^>]*)/>"#, t = regex::escape(tag))).unwrap();
        let re_attr = Regex::new(r#"(\w+)\s*=\s*"([^"]*)""#).unwrap();
        let parse_match = |attrs_str: &str, inner_text: Option<&str>| -> Option<Variable> {
            let mut attrs: HashMap<&str, &str> = HashMap::new();
            for cap in re_attr.captures_iter(attrs_str) {
                let k = cap.get(1).unwrap().as_str();
                let v = cap.get(2).unwrap().as_str();
                attrs.insert(k, v);
            }

            if let Some(m) = mesh {
                if attrs.get("mesh").copied() != Some(m) {
                    return None;
                }
            }
            if let Some(n) = name {
                if attrs.get("name").copied() != Some(n) {
                    return None;
                }
            }

            let arraysize = attrs.get("arraysize").map(|s| s.to_string());
            let datasize = attrs.get("datasize").map(|s| s.to_string());
            let datatype = attrs.get("datatype").map(|s| s.to_string());
            let mesh_str = attrs.get("mesh").map(|s| s.to_string());
            let name_str = attrs.get("name").map(|s| s.to_string());
            let vectorsize = attrs.get("vectorsize").map(|s| s.to_string());
            let max_refinement_level = attrs.get("max_refinement_level").map(|s| s.to_string());
            let offset = inner_text.map(|s| s.trim().to_string());

            Some(Variable {
                arraysize,
                datasize,
                datatype,
                mesh: mesh_str,
                name: name_str.or_else(|| Some(tag.to_string())),
                vectorsize,
                max_refinement_level,
                unit: attrs.get("unit").map(|s| s.to_string()),
                unit_conversion: attrs.get("unitConversion").map(|s| s.to_string()),
                unit_latex: attrs.get("unitLaTeX").map(|s| s.to_string()),
                variable_latex: attrs.get("variableLaTeX").map(|s| s.to_string()),
                offset,
            })
        };
        for caps in re_normal.captures_iter(xml) {
            let attrs_str = caps.get(1).unwrap().as_str();
            let text = caps.get(2).map(|m| m.as_str());
            if let Some(v) = parse_match(attrs_str, text) {
                return Some(v);
            }
        }

        for caps in re_self.captures_iter(xml) {
            let attrs_str = caps.get(1).unwrap().as_str();
            if let Some(v) = parse_match(attrs_str, None) {
                return Some(v);
            }
        }
        None
    }

    pub fn apply_op_in_place<T>(arr: &mut Array4<T>, op: Option<i32>)
    where
        T: Num + NumCast + Copy + Pod,
    {
        let Some(op) = op else { return };

        match op {
            0 | 1 | 2 | 3 => {}
            4 => {
                for mut lane in arr.lanes_mut(Axis(3)) {
                    // compute in f64 for safety
                    let sum_sq: f64 = lane
                        .iter()
                        .map(|&x| {
                            let xf: f64 = NumCast::from(x).unwrap();
                            xf * xf
                        })
                        .sum();

                    let mag_f64 = sum_sq.sqrt();

                    // cast back to T
                    lane[0] = NumCast::from(mag_f64).unwrap();
                }
            }
            _ => panic!("Unknown operator"),
        }
    }

    fn cid2ijk(cid: u64, nx: usize, ny: usize) -> (usize, usize, usize) {
        let lin = (cid - 1) as usize;
        let i = lin % nx;
        let j = (lin / nx) % ny;
        let k = lin / (nx * ny);
        (i, j, k)
    }

    fn amr_level(cellid: u64, x0: usize, y0: usize, z0: usize, lmax: u32) -> Option<u32> {
        let n0 = (x0 as u64) * (y0 as u64) * (z0 as u64);
        let mut cum = 0u64;
        for lvl in 0..=lmax {
            let count = n0.checked_shl(3 * lvl)?;
            if cellid <= cum + count {
                return Some(lvl);
            }
            cum = cum.checked_add(count)?;
        }
        None
    }

    fn cid2fineijk(
        cellid: u64,
        level: u32,
        lmax: u32,
        x0: usize,
        y0: usize,
        z0: usize,
    ) -> Option<(usize, usize, usize)> {
        let n0 = (x0 as u64) * (y0 as u64) * (z0 as u64);

        let mut cum = 0u64;
        for l in 0..level {
            cum = cum.checked_add(n0.checked_shl(3 * l)?)?;
        }

        let id0 = cellid.checked_sub(cum)?.checked_sub(1)?;
        let nx_l = (x0 as u64) << level;
        let ny_l = (y0 as u64) << level;

        let i_l = (id0 % nx_l) as usize;
        let j_l = ((id0 / nx_l) % ny_l) as usize;
        let k_l = (id0 / (nx_l * ny_l)) as usize;

        let scale = 1usize << ((lmax - level) as usize);
        Some((i_l * scale, j_l * scale, k_l * scale))
    }

    pub fn build_vg_indexes_on_fg(
        cell_ids: &[u64],
        fg_dims: (usize, usize, usize),
        x0: usize,
        y0: usize,
        z0: usize,
        lmax: u32,
    ) -> Array3<usize> {
        let (fx, fy, fz) = fg_dims;
        assert_eq!(fx, x0 << lmax);
        assert_eq!(fy, y0 << lmax);
        assert_eq!(fz, z0 << lmax);

        let mut map = Array3::<usize>::from_elem((fx, fy, fz), usize::MAX);
        for (vg_idx, &cid) in cell_ids.iter().enumerate() {
            let lvl = amr_level(cid, x0, y0, z0, lmax).expect("Invalid CellID or max level");
            let (sx, sy, sz) =
                cid2fineijk(cid, lvl, lmax, x0, y0, z0).expect("Failed to map CellID to fine ijk");
            let scale = 1usize << (lmax - lvl) as usize;
            let ex = sx + scale;
            let ey = sy + scale;
            let ez = sz + scale;
            assert!(ex <= fx && ey <= fy && ez <= fz);
            map.slice_mut(s![sx..ex, sy..ey, sz..ez]).fill(vg_idx);
        }
        map
    }

    pub fn vg_variable_to_fg<T: bytemuck::Pod + Copy + Default>(
        cell_ids: &[u64],
        vg_rows: &[T],
        vecsz: usize,
        x0: usize,
        y0: usize,
        z0: usize,
        lmax: u32,
    ) -> ndarray::Array4<T> {
        use ndarray::{Array4, s};
        let (fx, fy, fz) = (x0 << lmax, y0 << lmax, z0 << lmax);
        let mut fg = Array4::<T>::default((fx, fy, fz, vecsz));
        assert_eq!(vg_rows.len(), cell_ids.len() * vecsz);

        for (idx, &cid) in cell_ids.iter().enumerate() {
            let lvl = amr_level(cid, x0, y0, z0, lmax).expect("bad CellID/levels");
            let (sx, sy, sz) = cid2fineijk(cid, lvl, lmax, x0, y0, z0).unwrap();
            let scale = 1usize << ((lmax - lvl) as usize);
            let (ex, ey, ez) = (sx + scale, sy + scale, sz + scale);

            let row = &vg_rows[idx * vecsz..(idx + 1) * vecsz];
            let mut block = fg.slice_mut(s![sx..ex, sy..ey, sz..ez, ..]);
            for xi in 0..scale {
                for yj in 0..scale {
                    for zk in 0..scale {
                        let mut vox = block.slice_mut(s![xi, yj, zk, ..]);
                        vox.assign(&ndarray::Array1::from(row.to_vec()));
                    }
                }
            }
        }
        fg
    }

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

    impl TryFrom<&Variable> for VlsvDataset {
        type Error = String;

        fn try_from(var: &Variable) -> Result<Self, Self::Error> {
            let g = if let Some(v) = &var.mesh {
                Some(v.parse::<VlasiatorGrid>()?)
            } else {
                None
            };
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
                grid: g,
            })
        }
    }

    impl TryFrom<Variable> for VlsvDataset {
        type Error = String;

        fn try_from(var: Variable) -> Result<Self, Self::Error> {
            let g = if let Some(v) = var.mesh {
                Some(v.parse::<VlasiatorGrid>()?)
            } else {
                None
            };
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
                grid: g,
            })
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    enum VlasiatorGrid {
        FSGRID,
        SPATIALGRID,
        VMESH,
        IONOSPHERE,
    }

    impl FromStr for VlasiatorGrid {
        type Err = String;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s.to_ascii_uppercase().as_str() {
                "FSGRID" => Ok(VlasiatorGrid::FSGRID),
                "SPATIALGRID" => Ok(VlasiatorGrid::SPATIALGRID),
                "IONOSPHERE" => Ok(VlasiatorGrid::IONOSPHERE),
                "PROTON" => Ok(VlasiatorGrid::VMESH),
                other => panic!("Unknown VlasiatorGrid type: {}", other),
            }
        }
    }

    pub trait TypeTag {
        fn type_name() -> &'static str;
    }

    macro_rules! impl_prim_meta {
        ( $( $t:ty => $tag:expr ),* $(,)? ) => {
            $(
                impl TypeTag for $t {
                    fn type_name() -> &'static str { $tag }
                }

            )*
        };
    }

    //Vlasiator vlsv naming convention
    impl_prim_meta! {
        f32   => "float",
        f64   => "float",
        u32   => "uint",
        i32   => "int",
        u64   => "uint",
        i64   => "int",
        u8    => "u8",
        usize => "uint",
    }
}
