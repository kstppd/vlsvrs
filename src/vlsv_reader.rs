#[allow(dead_code)]
pub mod vlsv_reader {
    use bytemuck::{Pod, cast_slice};
    use memmap2::Mmap;
    use ndarray::{Array3, Array4, Order, s};
    use num_traits::{self, Zero};
    use regex::Regex;
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

            let mut vars = root
                .variables
                .iter()
                .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                .collect::<HashMap<_, _>>();

            let version: Option<Variable> =
                read_tag(&xml_string, "VERSION", None, Some("version_information"));

            let config: Option<Variable> =
                read_tag(&xml_string, "CONFIG", None, Some("config_file"));

            if let Some(x) = config {
                vars.insert(x.name.clone().unwrap(), x);
            }
            if let Some(x) = version {
                vars.insert(x.name.clone().unwrap(), x);
            }

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
                root,
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
            let expected_bytes = info.datasize * info.vectorsize * info.arraysize;
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
            let expected_bytes = info.datasize * info.vectorsize * info.arraysize;
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
            //We now allow mismatch of T and file data.
            if info.datasize == 4 && std::mem::size_of::<T>() == 8 {
                // file f32 ->  f64
                let dst_f64: &mut [f64] = bytemuck::cast_slice_mut(dst);
                for i in 0..dst_f64.len() {
                    let off = i * 4;
                    let v = f32::from_ne_bytes(src_bytes[off..off + 4].try_into().unwrap());
                    dst_f64[i] = v as f64;
                }
                return;
            }

            if info.datasize == 8 && std::mem::size_of::<T>() == 4 {
                //file f64 -> f32
                let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
                for i in 0..dst_f32.len() {
                    let off = i * 8;
                    let v = f64::from_ne_bytes(src_bytes[off..off + 8].try_into().unwrap());
                    dst_f32[i] = v as f32;
                }
                return;
            }

            if info.datasize == std::mem::size_of::<T>() {
                let dst_bytes = bytemuck::cast_slice_mut::<T, u8>(dst);
                dst_bytes.copy_from_slice(src_bytes);
                return;
            }

            panic!("Unhandled datasize/element-size combination");
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

        pub fn read_vg_variable_as_fg<T: bytemuck::Pod + Copy + Default>(
            &self,
            name: &str,
        ) -> Option<ndarray::Array4<T>> {
            let ds = self.get_dataset(name)?;
            let vecsz = ds.vectorsize;
            let x0 = self.read_scalar_parameter("xcells_ini")? as usize;
            let y0 = self.read_scalar_parameter("ycells_ini")? as usize;
            let z0 = self.read_scalar_parameter("zcells_ini")? as usize;
            let lmax = self.get_max_amr_refinement()?;
            let cellid_ds = self.get_dataset("CellID")?;
            let mut cell_ids = vec![0u64; cellid_ds.arraysize];
            self.read_variable_into::<u64>("CellID", Some(cellid_ds), &mut cell_ids);
            let n_cells = ds.arraysize;
            let mut vg_rows = vec![T::default(); n_cells * vecsz];
            self.read_variable_into::<T>(name, Some(ds), vg_rows.as_mut_slice());
            Some(vg_variable_to_fg(
                &cell_ids, &vg_rows, vecsz, x0, y0, z0, lmax,
            ))
        }

        pub fn read_fsgrid_variable<T: Pod + Zero>(&self, name: &str) -> Option<Array4<T>> {
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

        pub fn read_vdf(&self, cid: usize, pop: &str) -> Option<Array4<f32>> {
            let blockspercell = TryInto::<VlsvDataset>::try_into(
                self.root
                    .blockspercell
                    .as_ref()
                    .and_then(|items| items.iter().find(|v| v.name.as_deref() == Some(pop)))
                    .or_else(|| {
                        eprintln!("ERROR: blockspercell with name '{pop}' not found in VLSV file");
                        None
                    })?,
            )
            .ok()?;
            let cellswithblocks = TryInto::<VlsvDataset>::try_into(
                self.root
                    .cellswithblocks
                    .as_ref()
                    .and_then(|items| items.iter().find(|v| v.name.as_deref() == Some(pop)))
                    .or_else(|| {
                        eprintln!(
                            "ERROR: cellswithblocks with name '{pop}' not found in VLSV file"
                        );
                        None
                    })?,
            )
            .ok()?;
            let blockids = TryInto::<VlsvDataset>::try_into(
                self.root
                    .blockids
                    .as_ref()
                    .and_then(|items| items.iter().find(|v| v.name.as_deref() == Some(pop)))
                    .or_else(|| {
                        eprintln!("ERROR: blockids with name '{pop}' not found in VLSV file");
                        None
                    })?,
            )
            .ok()?;
            let blockvariable = TryInto::<VlsvDataset>::try_into(
                self.root
                    .blockvariable
                    .as_ref()
                    .and_then(|items| items.iter().find(|v| v.name.as_deref() == Some(pop)))
                    .or_else(|| {
                        eprintln!("ERROR: blockvariable with name '{pop}' not found in VLSV file");
                        None
                    })?,
            )
            .ok()?;
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

            let mut vdf = Array4::<f32>::zeros((nvx, nvy, nvz, 1));
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
                            vdf[(gi, gj, gk, 0)] = val;
                        }
                    }
                }
            }
            Some(vdf)
        }
    }
}
