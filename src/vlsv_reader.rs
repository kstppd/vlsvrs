/*
File: vlsv_reader.rs
Copyright (C) 2025 Kostis Papadakis 2024/2025 (kpapadakis@protonmail.com)
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This is a set of tools written mainly for fun but also for
some projects in Vlasiator (Asterix, Faiser...).
A very very nice thing here is that we can actually read
in a VDF into a dense mesh (we can also remap the VDF to a target mesh)
which is handy for training neural nets.

EXAMPLES:
    let f = VlsvFile::new("bulk.vlsv").unwrap();
    //OP: vec->scalar reduction into first component with  0|1->x(noop) 2->y 3->z 4->magnitude
    let OP = 0;
    let data:Array4<_> = f.read_variable::<f32>(&varname, Some(OP)).unwrap()
    let data:Array4<_> = f.read_vg_variable_as_fg::<f32>(&varname, Some(OP)).unwrap()
    let data:Array4<_> = f.read_fsgrid_variable::<f32>(&varname, Some(OP)).unwrap()
    let data:Array4<_> = f.read_vdf::<f32>(256, "proton")).unwrap();

There are 3 main parts here:
1) MOD_VLSV_READER:
    Reads VLSV files and metadata.
    Can read orderd fsgrid variables.
    Can read vg variables as fg.
    Can read dense vdfs and up/down scale them.
    Has much smaller memory footprint than analysator.

    Keywords:
    read_scalar_parameter, read_config, read_version, read_variable_into, get_wid, get_vspace_mesh_bbox, get_spatial_mesh_extents, get_vspace_mesh_extents, get_domain_decomposition, get_max_amr_refinement, get_writting_tasks, get_spatial_mesh_bbox, get_dataset, read_vg_variable_as_fg, read_fsgrid_variable, read_vdf, read_vdf_into, read_variable, read_tag, vg_variable_to_fg

2) MOD_VLSV_TRACING:
    Particle tracing routines using fields from Vlasiator.

    Keywords:
    get_fields_at, new_with_energy_at_Lshell, boris, larmor_radius, borris_adaptive

3) MOD_VLSV_EXPORTS:
    Creates C and Python interfaces for VLSV_READER.
    Keywords:
    read_variable_f64, read_variable_f32, read_vdf_f32, read_vdf_f64
*/
#![allow(dead_code)]
#![allow(non_snake_case)]

pub mod mod_vlsv_reader {
    pub const VLSV_FOOTER_LOC_START: usize = 8;
    pub const VLSV_FOOTER_LOC_END: usize = 16;
    use bytemuck::{Pod, Zeroable, cast_slice, pod_read_unaligned};
    use core::convert::TryInto;
    use memmap2::Mmap;
    use ndarray::{Array4, ArrayView1};
    use ndarray::{Axis, Order, s};
    use num_traits::{Float, FromPrimitive, Num, NumCast, ToPrimitive, Zero};
    use once_cell::sync::OnceCell;
    use regex::Regex;
    use serde::Deserialize;
    use std::cell::RefCell;
    use std::sync::OnceLock;
    use std::{collections::HashMap, str::FromStr};
    extern crate libc;
    static RE_ATTR: OnceLock<Regex> = OnceLock::new();

    #[repr(C)]
    #[derive(Default, Clone, Copy, Debug, Pod, Zeroable)]
    pub struct Norms {
        pub min: f64,
        pub max: f64,
    }

    #[derive(Debug)]
    pub struct PhaseSpaceUnion<T> {
        pub nrows: usize,
        pub ncols: usize,
        pub center_vdfs: bool,
        pub norms: Vec<Norms>,
        pub cids: Vec<u64>,
        pub vcoords: Vec<[T; 3]>,
        pub vbulks: Vec<[T; 3]>,
        pub vspace: Vec<T>,
        pub map: HashMap<u64, usize>,
        pub network_weights: Vec<T>,
        pub effective_vdf_size: usize,
        pub v_limits: [T; 6],
    }

    impl<T> PhaseSpaceUnion<T>
    where
        T: Pod + Zeroable + Default + PartialOrd + num_traits::Float + std::fmt::Display,
    {
        pub fn new_from_buffer(buffer: &[u8]) -> Result<Self, String> {
            let mut instance = Self::empty();
            instance.deserialize_from(buffer)?;
            Ok(instance)
        }

        fn empty() -> Self {
            Self {
                nrows: 0,
                ncols: 0,
                center_vdfs: false,
                norms: Vec::new(),
                cids: Vec::new(),
                vcoords: Vec::new(),
                vbulks: Vec::new(),
                vspace: Vec::new(),
                map: HashMap::new(),
                network_weights: Vec::new(),
                effective_vdf_size: 0,
                v_limits: [
                    T::max_value(),
                    T::max_value(),
                    T::max_value(),
                    T::min_value(),
                    T::min_value(),
                    T::min_value(),
                ],
            }
        }

        pub fn index_2d(&self, row: usize, col: usize) -> usize {
            row * self.ncols + col
        }

        pub fn unnormalize_and_unscale(&mut self, sparse: T) {
            let two = T::from(2.0).unwrap();
            let one = T::from(1.0).unwrap();
            let ten = T::from(10.0).unwrap();

            for x in self.vcoords.iter_mut() {
                x[0] =
                    ((x[0] + one) / two) * (self.v_limits[3] - self.v_limits[0]) + self.v_limits[0];
                x[1] =
                    ((x[1] + one) / two) * (self.v_limits[4] - self.v_limits[1]) + self.v_limits[1];
                x[2] =
                    ((x[2] + one) / two) * (self.v_limits[5] - self.v_limits[2]) + self.v_limits[2];
            }

            for v in 0..self.ncols {
                let max_val = T::from(self.norms[v].max).unwrap();

                for i in 0..self.nrows {
                    let idx = self.index_2d(i, v);
                    self.vspace[idx] = sparse * ten.powf(self.vspace[idx] * max_val);
                }
            }
        }

        pub fn sparsify(&mut self, sparse: T) {
            let zero = T::zero();
            for x in self.vspace.iter_mut() {
                if *x < sparse {
                    *x = zero;
                }
            }
        }

        pub fn deserialize_from(&mut self, buffer: &[u8]) -> Result<(), String> {
            fn consume<'a>(remaining: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
                if remaining.len() < n {
                    return Err("Buffer underflow".into());
                }
                let (head, tail) = remaining.split_at(n);
                *remaining = tail;
                Ok(head)
            }

            let h_size = std::mem::size_of::<Header>();
            let header: Header = *bytemuck::from_bytes(&buffer[..h_size]);

            if header.key != 42 {
                return Err("Wrong MLP Header KEY".into());
            }

            self.nrows = header.rows;
            self.ncols = header.cols;
            self.vspace.resize(self.nrows * self.ncols, T::default());

            let mut remaining = &buffer[h_size..];

            self.cids = bytemuck::cast_slice(consume(&mut remaining, self.ncols * 8)?).to_vec();
            self.norms = bytemuck::cast_slice(consume(&mut remaining, self.ncols * 16)?).to_vec();

            let vbulk_len = self.ncols * std::mem::size_of::<[T; 3]>();
            self.vbulks = bytemuck::cast_slice(consume(&mut remaining, vbulk_len)?).to_vec();

            let v_lim_bytes = 6 * std::mem::size_of::<T>();
            self.v_limits
                .copy_from_slice(bytemuck::cast_slice(consume(&mut remaining, v_lim_bytes)?));

            let vcoords_len = self.nrows * std::mem::size_of::<[T; 3]>();
            self.vcoords = bytemuck::cast_slice(consume(&mut remaining, vcoords_len)?).to_vec();

            let weights_len = header.n_weights * std::mem::size_of::<T>();
            self.network_weights =
                bytemuck::cast_slice(consume(&mut remaining, weights_len)?).to_vec();

            while !remaining.is_empty() {
                let k: u64 = *bytemuck::from_bytes(consume(&mut remaining, 8)?);
                let v: usize = *bytemuck::from_bytes(consume(&mut remaining, 8)?);
                self.map.insert(k, v);
            }

            Ok(())
        }

        pub fn total_serialized_size_bytes(&self) -> usize {
            std::mem::size_of::<Header>()
                + self.cids.len() * 8
                + self.norms.len() * 16
                + self.vbulks.len() * std::mem::size_of::<[T; 3]>()
                + self.vcoords.len() * std::mem::size_of::<[T; 3]>()
                + 6 * std::mem::size_of::<T>()
                + self.network_weights.len() * std::mem::size_of::<T>()
                + self.map.len() * 16
        }
    }

    impl PhaseSpaceUnion<f32> {
        #[cfg(not(no_nn))]
        pub fn decompress(&mut self, hidden_layers: &[usize], fourier_order: usize) {
            #[link(name = "vlasiator_vdf_compressor_nn")]
            unsafe extern "C" {
                pub fn decompress_phasespace6D_f32_nopool(
                    fin: usize,
                    fout: usize,
                    vcoords_ptr: *mut f32,
                    vspace_ptr: *mut f32,
                    size: usize,
                    fourier_order: usize,
                    hidden_layers_ptr: *const usize,
                    n_hidden_layers: usize,
                    weights_ptr: *mut f32,
                    weight_size: usize,
                    use_input_weights: bool,
                );
            }
            let n_rows = self.nrows;
            let n_weights = self.network_weights.len();

            unsafe {
                decompress_phasespace6D_f32_nopool(
                    3,
                    self.ncols,
                    self.vcoords.as_mut_ptr() as *mut f32,
                    self.vspace.as_mut_ptr(),
                    n_rows,
                    fourier_order,
                    hidden_layers.as_ptr(),
                    hidden_layers.len(),
                    self.network_weights.as_mut_ptr(),
                    n_weights,
                    true,
                );
            }
        }

        pub fn reconstruct_vdf_dense(&self, cid: usize, dv: f32) -> Array4<f32> {
            let target = self
                .cids
                .iter()
                .position(|&x| x == cid as u64)
                .ok_or_else(|| format!("CID {} not found in this PhaseSpaceUnion", cid))
                .unwrap();
            let nvx = ((self.v_limits[3] - self.v_limits[0]) / dv).round() as usize;
            let nvy = ((self.v_limits[4] - self.v_limits[1]) / dv).round() as usize;
            let nvz = ((self.v_limits[5] - self.v_limits[2]) / dv).round() as usize;
            let mut vdf = Array4::<f32>::zeros((nvx, nvy, nvz, 1));
            for i in 0..self.nrows {
                let global_vx = self.vcoords[i][0] + self.vbulks[target][0];
                let global_vy = self.vcoords[i][1] + self.vbulks[target][1];
                let global_vz = self.vcoords[i][2] + self.vbulks[target][2];
                let ix = ((global_vx - self.v_limits[0]) / dv).round() as usize;
                let iy = ((global_vy - self.v_limits[1]) / dv).round() as usize;
                let iz = ((global_vz - self.v_limits[2]) / dv).round() as usize;
                if ix < nvx && iy < nvy && iz < nvz {
                    let value = self.vspace[i * self.ncols + target];
                    vdf[[ix, iy, iz, 0]] = value;
                }
            }
            vdf
        }
    }

    impl PhaseSpaceUnion<f64> {
        #[cfg(not(no_nn))]
        pub fn decompress(&mut self, hidden_layers: &[usize], fourier_order: usize) {
            #[link(name = "vlasiator_vdf_compressor_nn")]
            unsafe extern "C" {
                pub fn decompress_phasespace6D_f64_nopool(
                    fin: usize,
                    fout: usize,
                    vcoords_ptr: *mut f64,
                    vspace_ptr: *mut f64,
                    size: usize,
                    fourier_order: usize,
                    hidden_layers_ptr: *const usize,
                    n_hidden_layers: usize,
                    weights_ptr: *mut f64,
                    weight_size: usize,
                    use_input_weights: bool,
                );
            }
            let n_rows = self.nrows;
            let n_weights = self.network_weights.len();

            unsafe {
                decompress_phasespace6D_f64_nopool(
                    3,
                    self.ncols,
                    self.vcoords.as_mut_ptr() as *mut f64,
                    self.vspace.as_mut_ptr(),
                    n_rows,
                    fourier_order,
                    hidden_layers.as_ptr(),
                    hidden_layers.len(),
                    self.network_weights.as_mut_ptr(),
                    n_weights,
                    true,
                );
            }
        }

        pub fn reconstruct_vdf_dense(&self, cid: usize, dv: f64) -> Array4<f64> {
            let target = self
                .cids
                .iter()
                .position(|&x| x == cid as u64)
                .ok_or_else(|| format!("CID {} not found in this PhaseSpaceUnion", cid))
                .unwrap();
            let nvx = ((self.v_limits[3] - self.v_limits[0]) / dv).round() as usize;
            let nvy = ((self.v_limits[4] - self.v_limits[1]) / dv).round() as usize;
            let nvz = ((self.v_limits[5] - self.v_limits[2]) / dv).round() as usize;
            let mut vdf = Array4::<f64>::zeros((nvx, nvy, nvz, 1));
            for i in 0..self.nrows {
                let global_vx = self.vcoords[i][0] + self.vbulks[target][0];
                let global_vy = self.vcoords[i][1] + self.vbulks[target][1];
                let global_vz = self.vcoords[i][2] + self.vbulks[target][2];
                let ix = ((global_vx - self.v_limits[0]) / dv).round() as usize;
                let iy = ((global_vy - self.v_limits[1]) / dv).round() as usize;
                let iz = ((global_vz - self.v_limits[2]) / dv).round() as usize;
                if ix < nvx && iy < nvy && iz < nvz {
                    let value = self.vspace[i * self.ncols + target];
                    vdf[[ix, iy, iz, 0]] = value;
                }
            }
            vdf
        }
    }

    #[derive(Debug)]
    pub struct VlsvFile {
        pub filename: String,
        pub variables: OnceCell<HashMap<String, Variable>>,
        pub parameters: OnceCell<HashMap<String, Variable>>,
        memmap: OnceCell<Mmap>,
        pub root: OnceCell<VlsvRoot>,
    }

    impl VlsvFile {
        pub fn new(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
            Ok(Self {
                filename: filename.to_string(),
                variables: OnceCell::new(),
                parameters: OnceCell::new(),
                memmap: OnceCell::new(),
                root: OnceCell::new(),
            })
        }

        #[cfg(all(feature = "uring", target_os = "linux"))]
        pub async fn new_uring(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
            // println!("Reading with IO Uring");
            use std::io::Error;
            use tokio_uring::fs::File;
            async fn read_exact_at_uring(
                file: &File,
                offset: u64,
                len: usize,
            ) -> Result<Vec<u8>, Error> {
                let buf = vec![0u8; len];
                let (res, buf) = file.read_at(buf, offset).await;
                let n = res?;
                if n != len {
                    panic!();
                }
                Ok(buf)
            }
            let file = File::open(filename).await?;
            let len_footer = VLSV_FOOTER_LOC_END - VLSV_FOOTER_LOC_START;
            let footer_buf =
                read_exact_at_uring(&file, VLSV_FOOTER_LOC_START as u64, len_footer).await?;
            let footer_offset = usize::from_ne_bytes(footer_buf[..len_footer].try_into()?);

            let file_size = std::fs::metadata(filename)?.len() as usize;
            if footer_offset > file_size {
                panic!();
            }
            let xml_len = file_size - footer_offset;

            let xml_buf = read_exact_at_uring(&file, footer_offset as u64, xml_len).await?;
            let xml_str = std::str::from_utf8(&xml_buf)?;
            let root: VlsvRoot = serde_xml_rs::from_str(xml_str)?;

            let vars: HashMap<String, Variable> = root
                .variables
                .iter()
                .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                .chain(
                    [
                        ("CONFIG", "config_file"),
                        ("VERSION", "version_information"),
                    ]
                    .into_iter()
                    .filter_map(|(tag, section)| {
                        read_tag(xml_str, tag, None, Some(section))
                            .map(|x| (x.name.clone().unwrap(), x))
                    }),
                )
                .collect();

            let params: HashMap<String, Variable> = root
                .parameters
                .iter()
                .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                .collect();

            //Set all once cells here
            let v = {
                let a = OnceCell::new();
                let _ = a.set(vars);
                a
            };

            let p = {
                let a = OnceCell::new();
                let _ = a.set(params);
                a
            };

            let r = {
                let a = OnceCell::new();
                let _ = a.set(root);
                a
            };

            Ok(Self {
                filename: filename.to_string(),
                variables: v,
                parameters: p,
                memmap: OnceCell::new(),
                root: r,
            })
        }

        #[inline]
        fn memorymap(&self) -> &memmap2::Mmap {
            self.memmap.get_or_init(|| {
                let file = std::fs::File::open(&self.filename)
                    .unwrap_or_else(|e| panic!("ERROR:mmap open('{}') failed: {e}", self.filename));
                unsafe {
                    memmap2::MmapOptions::new().map(&file).unwrap_or_else(|e| {
                        panic!("ERROR:mmap map('{}') failed: {e}", self.filename)
                    })
                }
            })
        }

        #[inline]
        fn root(&self) -> &VlsvRoot {
            self.root.get_or_init(|| {
                let footer_offset: usize = usize::from_ne_bytes(
                    self.memorymap()[VLSV_FOOTER_LOC_START..VLSV_FOOTER_LOC_END]
                        .try_into()
                        .unwrap(),
                );
                let xml_str = std::str::from_utf8(&self.memorymap()[footer_offset..]).unwrap();
                serde_xml_rs::from_str(xml_str).unwrap()
            })
        }

        #[inline]
        pub fn variables(&self) -> &HashMap<String, Variable> {
            self.variables.get_or_init(|| {
                let footer_offset: usize = usize::from_ne_bytes(
                    self.memorymap()[VLSV_FOOTER_LOC_START..VLSV_FOOTER_LOC_END]
                        .try_into()
                        .unwrap(),
                );
                let xml_str = std::str::from_utf8(&self.memorymap()[footer_offset..]).unwrap();
                self.root()
                    .variables
                    .iter()
                    .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                    .chain(
                        [
                            ("CONFIG", "config_file", "Config not available!"),
                            ("VERSION", "version_information", "Version not available!"),
                        ]
                        .into_iter()
                        .filter_map(|(tag, section, _warn_msg)| {
                            read_tag(xml_str, tag, None, Some(section))
                                .map(|x| (x.name.clone().unwrap(), x))
                        }),
                    )
                    .collect()
            })
        }

        #[inline]
        pub fn parameters(&self) -> &HashMap<String, Variable> {
            self.parameters.get_or_init(|| {
                self.root()
                    .parameters
                    .iter()
                    .filter_map(|var| var.name.clone().map(|n| (n, var.clone())))
                    .collect()
            })
        }

        pub fn read_compression(&self) -> &str {
            let comrpession = self
                .read_scalar_parameter("COMPRESSION")
                .or(Some(0.0))
                .unwrap() as i32;
            match comrpession {
                0 => "NONE",
                1 => "ZFP",
                2 => "OCTREE",
                3 => "MLP",
                4 => "MLPMULTI",
                5 => "HERMITE",
                _ => "NONE",
            }
        }

        pub fn get_vdf_size(&self, pop: &str) -> Option<usize> {
            let root = self.root();
            let blockids_size = root
                .blockids
                .as_ref()
                .and_then(|list| list.iter().find(|v| v.name.as_deref() == Some(pop)))
                .and_then(|item| TryInto::<VlsvDataset>::try_into(item).ok())
                .map(|v| v.arraysize * v.vectorsize * v.datasize)
                .unwrap_or(0);

            let blockvariable_size = root
                .blockvariable
                .as_ref()
                .and_then(|list| list.iter().find(|v| v.name.as_deref() == Some(pop)))
                .and_then(|item| TryInto::<VlsvDataset>::try_into(item).ok())
                .map(|v| v.arraysize * v.vectorsize * v.datasize)
                .unwrap_or(0);
            Some(blockvariable_size + blockids_size)
        }

        pub fn read_scalar_parameter(&self, name: &str) -> Option<f64> {
            let info = self.get_dataset(name)?;
            debug_assert!(info.vectorsize == 1);
            debug_assert!(info.arraysize == 1);
            let expected_bytes = info.datasize * info.vectorsize;
            debug_assert!(
                info.offset + expected_bytes <= self.memorymap().len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let src_bytes = &self.memorymap()[info.offset..info.offset + expected_bytes];
            let retval = match info.datasize {
                8 => {
                    let mut buffer: [u8; 8] = [0; 8];
                    buffer.copy_from_slice(cast_slice(src_bytes));

                    match info.datatype {
                        DataType::Float => f64::from_ne_bytes(buffer),
                        DataType::Uint => usize::from_ne_bytes(buffer) as f64,
                        DataType::Int => i64::from_ne_bytes(buffer) as f64,
                        _ => panic!("Only matched against uint and float"),
                    }
                }
                4 => {
                    let mut buffer: [u8; 4] = [0; 4];
                    buffer.copy_from_slice(cast_slice(src_bytes));
                    match info.datatype {
                        DataType::Float => f32::from_ne_bytes(buffer) as f64,
                        DataType::Uint => u32::from_ne_bytes(buffer) as f64,
                        DataType::Int => i32::from_ne_bytes(buffer) as f64,
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
            debug_assert!(
                info.offset + expected_bytes <= self.memorymap().len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let bytes = &self.memorymap()[info.offset..info.offset + expected_bytes];
            let cfgfile = std::str::from_utf8(bytes).map(|s| s.to_owned()).ok()?;
            Some(cfgfile)
        }

        pub fn read_version(&self) -> Option<String> {
            const NAME: &str = "version_information";
            let info = self.get_dataset(NAME)?;
            let expected_bytes = info.datasize * info.vectorsize * info.arraysize;
            debug_assert!(
                info.offset + expected_bytes <= self.memorymap().len(),
                "Attempt to read out-of-bounds from memory map"
            );
            let bytes = &self.memorymap()[info.offset..info.offset + expected_bytes];
            let version = std::str::from_utf8(bytes).map(|s| s.to_owned()).ok()?;
            Some(version)
        }

        fn read_variable_into<T: Sized + Pod + TypeTag>(
            &self,
            name: Option<&str>,
            dataset: Option<VlsvDataset>,
            dst: &mut [T],
        ) {
            let info = match (name, dataset) {
                (Some(n), None) => self
                    .get_dataset(n)
                    .unwrap_or_else(|| panic!("No dataset found: {n}")),
                (None, Some(d)) => d,
                _ => panic!("read_variable_into requires exactly one of 'name' or 'dataset'"),
            };

            let expected_bytes = info.datasize * info.arraysize * info.vectorsize;
            let src_bytes = &self.memorymap()[info.offset..info.offset + expected_bytes];
            let type_on_disk = info.datatype;
            let size_on_disk = info.datasize;
            let type_of_t = T::data_type();
            let size_of_t = std::mem::size_of::<T>();

            let types_match = type_on_disk == type_of_t;
            let sizes_match = size_on_disk == size_of_t;
            let is_compressed = info.compression != Some(CompressionMethod::NONE);

            //Straight Copy (this is weird but when vdfs are compreessed we read raw bytes)
            if (types_match && sizes_match) || ((types_match || sizes_match) && is_compressed) {
                let dst_bytes = bytemuck::cast_slice_mut::<T, u8>(dst);
                dst_bytes.copy_from_slice(src_bytes);
                return;
            }

            //Dynamic Dispatchh
            match (type_on_disk, size_on_disk, type_of_t, size_of_t) {
                //f32 => f64
                (DataType::Float, 4, DataType::Float, 8) => {
                    let dst_f64: &mut [f64] = bytemuck::cast_slice_mut(dst);
                    src_bytes
                        .chunks_exact(4)
                        .zip(dst_f64)
                        .for_each(|(src, out)| {
                            *out = f32::from_le_bytes(src.try_into().unwrap()) as f64;
                        });
                }
                //f64 => f32
                (DataType::Float, 8, DataType::Float, 4) => {
                    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
                    src_bytes
                        .chunks_exact(8)
                        .zip(dst_f32)
                        .for_each(|(src, out)| {
                            *out = f64::from_le_bytes(src.try_into().unwrap()) as f32;
                        });
                }
                //i32 => f32
                (DataType::Int, 4, DataType::Float, 4) => {
                    let dst_f32: &mut [f32] = bytemuck::cast_slice_mut(dst);
                    src_bytes
                        .chunks_exact(4)
                        .zip(dst_f32)
                        .for_each(|(src, out)| {
                            *out = i32::from_le_bytes(src.try_into().unwrap()) as f32;
                        });
                }
                _ => panic!(
                    "Incompatible read: {type_on_disk:?}({size_on_disk}) => {type_of_t:?}({size_of_t})"
                ),
            }
        }

        // #[deprecated(note = "TODO: This reads WID from the first population file. Use get_wid(pop:&str)!")]
        pub fn get_global_wid(&self) -> Option<usize> {
            let wid = {
                let dataset: VlsvDataset = self
                    .root()
                    .blockvariable
                    .as_ref()?
                    .first()?
                    .try_into()
                    .ok()?;
                (dataset.vectorsize as f64).cbrt()
            };
            Some(wid as usize)
        }

        pub fn get_wid(&self, _pop: &str) -> Option<usize> {
            Some(
                self.read_scalar_parameter("velocity_block_width")
                    .unwrap_or(4.0_f64) as usize,
            )
        }

        pub fn get_vdf_float_datasize(&self, pop: &str) -> Option<usize> {
            let blockvariable = TryInto::<VlsvDataset>::try_into(
                self.root()
                    .blockvariable
                    .as_ref()?
                    .iter()
                    .find(|v| v.name.as_deref() == Some(pop))?,
            )
            .ok()?;
            let compression_used = blockvariable
                .compression
                .clone()
                .unwrap_or(CompressionMethod::NONE);
            let datasize = match compression_used {
                CompressionMethod::NONE => blockvariable.datasize as usize,
                CompressionMethod::ZFP
                | CompressionMethod::OCTREE
                | CompressionMethod::MLPMULTI
                | CompressionMethod::HERMITE
                | CompressionMethod::MLP => {
                    self.read_scalar_parameter("VDF_BYTE_SIZE").unwrap() as usize
                }
            };
            Some(datasize)
        }

        pub fn get_all_populations(&self) -> Option<Vec<&str>> {
            Some(
                self.root()
                    .mesh_node_crds_x
                    .as_ref()?
                    .iter()
                    .filter_map(|v| {
                        let grid: VlasiatorGrid = v.mesh.as_ref()?.as_str().parse().ok()?;
                        (grid == VlasiatorGrid::VMESH).then(|| v.mesh.as_ref().unwrap().as_str())
                    })
                    .collect::<Vec<&str>>(),
            )
        }

        pub fn get_vspace_mesh_bbox(&self, pop: &str) -> Option<(usize, usize, usize)> {
            let nvx = self
                .root()
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
                .root()
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
                .root()
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
                self.root().mesh_node_crds_x.as_ref().and_then(|meshes| {
                    meshes
                        .iter()
                        .find(|v| v.mesh.as_deref() == Some("SpatialGrid"))
                })?,
            )
            .ok()?;
            let nodes_y = TryInto::<VlsvDataset>::try_into(
                self.root().mesh_node_crds_y.as_ref().and_then(|meshes| {
                    meshes
                        .iter()
                        .find(|v| v.mesh.as_deref() == Some("SpatialGrid"))
                })?,
            )
            .ok()?;
            let nodes_z = TryInto::<VlsvDataset>::try_into(
                self.root().mesh_node_crds_z.as_ref().and_then(|meshes| {
                    meshes
                        .iter()
                        .find(|v| v.mesh.as_deref() == Some("SpatialGrid"))
                })?,
            )
            .ok()?;
            debug_assert!(nodes_x.datasize == 8, "Expected f64 for mesh node coords");
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

        pub fn get_amr_level(&self, mut cellid: u64) -> Option<i32> {
            let x0 = self.read_scalar_parameter("xcells_ini")? as u64;
            let y0 = self.read_scalar_parameter("ycells_ini")? as u64;
            let z0 = self.read_scalar_parameter("zcells_ini")? as u64;
            let max_ref = self.get_max_amr_refinement()? as i32;
            let n0 = x0.saturating_mul(y0).saturating_mul(z0);
            if n0 == 0 {
                return Some(-1);
            }

            let mut count: i32 = 0;
            let mut iters: i32 = 0;
            while cellid > 0 {
                let pow8 = 8u64.saturating_pow(count as u32);
                let sub = n0.saturating_mul(pow8);
                cellid = cellid.saturating_sub(sub);
                count += 1;
                iters += 1;
                if iters > max_ref + 1 {
                    eprintln!("Something broke.");
                    break;
                }
            }
            Some(count - 1)
        }

        pub fn get_amr_level_batch(&self, cellids: &[u64]) -> Option<Vec<i32>> {
            let x0 = self.read_scalar_parameter("xcells_ini")? as u64;
            let y0 = self.read_scalar_parameter("ycells_ini")? as u64;
            let z0 = self.read_scalar_parameter("zcells_ini")? as u64;
            let max_ref = self.get_max_amr_refinement()? as i32;
            let n0 = x0.saturating_mul(y0).saturating_mul(z0);
            if n0 == 0 {
                return Some(vec![-1; cellids.len()]);
            }
            let mut levels = vec![0i32; cellids.len()];
            let mut remaining: Vec<u64> = cellids.to_vec();
            let mut iters: i32 = 0;

            loop {
                let mut any_pos = false;
                for (i, cid) in remaining.iter_mut().enumerate() {
                    if *cid > 0 {
                        any_pos = true;
                        let pow8 = 8u64.saturating_pow(levels[i] as u32);
                        let sub = n0.saturating_mul(pow8);
                        *cid = cid.saturating_sub(sub);
                        levels[i] += 1;
                    }
                }
                if !any_pos {
                    break;
                }
                iters += 1;
                if iters > max_ref + 1 {
                    eprintln!("Can't have that large refinements. Something broke.");
                    break;
                }
            }
            for l in &mut levels {
                *l -= 1;
            }
            Some(levels)
        }

        //Cell centers here
        pub fn get_cell_coordinate(&self, cellid: u64) -> Option<[f64; 3]> {
            self.get_cell_coordinates_batch(&[cellid])
                .and_then(|v| v.into_iter().next())
        }

        pub fn get_cell_coordinates_batch(&self, cellids: &[u64]) -> Option<Vec<[f64; 3]>> {
            let xmin = self.read_scalar_parameter("xmin")? as f64;
            let ymin = self.read_scalar_parameter("ymin")? as f64;
            let zmin = self.read_scalar_parameter("zmin")? as f64;
            let xmax = self.read_scalar_parameter("xmax")? as f64;
            let ymax = self.read_scalar_parameter("ymax")? as f64;
            let zmax = self.read_scalar_parameter("zmax")? as f64;

            let x0 = self.read_scalar_parameter("xcells_ini")? as usize;
            let y0 = self.read_scalar_parameter("ycells_ini")? as usize;
            let z0 = self.read_scalar_parameter("zcells_ini")? as usize;
            let lmax = self.get_max_amr_refinement()? as u32;

            let fx = (x0 as u32) << lmax;
            let fy = (y0 as u32) << lmax;
            let fz = (z0 as u32) << lmax;
            if fx == 0 || fy == 0 || fz == 0 {
                return Some(vec![]);
            }
            let dx_f = (xmax - xmin) / (fx as f64);
            let dy_f = (ymax - ymin) / (fy as f64);
            let dz_f = (zmax - zmin) / (fz as f64);
            let levels = self.get_amr_level_batch(cellids)?;
            let mut coords = Vec::<[f64; 3]>::with_capacity(cellids.len());
            for (cid, lvl_i32) in cellids.iter().copied().zip(levels.into_iter()) {
                let lvl = if lvl_i32 < 0 { 0u32 } else { lvl_i32 as u32 };
                let (i_f, j_f, k_f) = match cid2fineijk(cid, lvl, lmax, x0, y0, z0) {
                    Some(t) => t,
                    None => {
                        coords.push([f64::NAN, f64::NAN, f64::NAN]);
                        continue;
                    }
                };
                let scale = 1usize << ((lmax - lvl) as usize);
                let cx = xmin + ((i_f as f64) + (scale as f64) * 0.5) * dx_f;
                let cy = ymin + ((j_f as f64) + (scale as f64) * 0.5) * dy_f;
                let cz = zmin + ((k_f as f64) + (scale as f64) * 0.5) * dz_f;
                coords.push([cx, cy, cz]);
            }
            Some(coords)
        }

        pub fn get_vspace_mesh_extents(&self, pop: &str) -> Option<(f64, f64, f64, f64, f64, f64)> {
            let nodes_x = TryInto::<VlsvDataset>::try_into(
                self.root()
                    .mesh_node_crds_x
                    .as_ref()
                    .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))?,
            )
            .ok()?;
            let nodes_y = TryInto::<VlsvDataset>::try_into(
                self.root()
                    .mesh_node_crds_y
                    .as_ref()
                    .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))?,
            )
            .ok()?;
            let nodes_z = TryInto::<VlsvDataset>::try_into(
                self.root()
                    .mesh_node_crds_z
                    .as_ref()
                    .and_then(|meshes| meshes.iter().find(|v| v.mesh.as_deref() == Some(pop)))?,
            )
            .ok()?;
            debug_assert!(nodes_x.datasize == 8, "Expected f64 for mesh node coords");
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
                .root()
                .mesh_decomposition
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .and_then(|v| v.try_into().ok())?;
            self.read_variable_into::<i32>(None, Some(decomposition), decomp.as_mut_slice());
            Some([decomp[0] as usize, decomp[1] as usize, decomp[2] as usize])
        }

        pub fn get_max_amr_refinement(&self) -> Option<u32> {
            self.root().mesh.as_ref().and_then(|meshes| {
                meshes
                    .iter()
                    .find_map(|v| v.max_refinement_level.as_ref()?.parse::<u32>().ok())
                    .or(Some(0))
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
            self.variables()
                .get(name)
                .or_else(|| self.parameters().get(name))
                .or_else(|| None)?
                .clone()
                .try_into()
                .ok()
        }

        pub fn read_vg_variable_as_fg<
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag + std::cmp::PartialOrd,
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
            let mut cell_ids = Vec::<u64>::with_capacity(cellid_ds.arraysize);
            unsafe { cell_ids.set_len(cellid_ds.arraysize) };
            self.read_variable_into::<u64>(None, Some(cellid_ds), &mut cell_ids);
            let n_cells = info.arraysize;
            let mut vg_rows = Vec::<T>::with_capacity(n_cells * vecsz);
            unsafe { vg_rows.set_len(n_cells * vecsz) };
            self.read_variable_into::<T>(None, Some(info), vg_rows.as_mut_slice());
            let mut ordered_var = vg_variable_to_fg(&cell_ids, &vg_rows, vecsz, x0, y0, z0, lmax);
            apply_op_in_place::<T>(&mut ordered_var, op);
            Some(ordered_var)
        }

        pub fn read_fsgrid_variable<
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag + std::cmp::PartialOrd,
        >(
            &self,
            name: &str,
            op: Option<i32>,
        ) -> Option<Array4<T>> {
            let info = self.get_dataset(name)?;
            if info.grid? != VlasiatorGrid::FSGRID {
                return None;
            }
            let decomp = self
                .get_domain_decomposition()
                .expect("ERROR: Domain Decomposition could not be recovered from {self.filename}.");
            let ntasks = self.get_writting_tasks()?;
            let (nx, ny, nz) = self.get_spatial_mesh_bbox()?;

            fn calc_local_start(global_cells: usize, ntasks: usize, my_n: usize) -> usize {
                let n_per_task = global_cells / ntasks;
                let remainder = global_cells % ntasks;
                if my_n < remainder {
                    my_n * (n_per_task + 1)
                } else {
                    my_n * n_per_task + remainder
                }
            }

            fn calc_local_size(global_cells: usize, ntasks: usize, my_n: usize) -> usize {
                let n_per_task = global_cells / ntasks;
                let remainder = global_cells % ntasks;
                if my_n < remainder {
                    n_per_task + 1
                } else {
                    n_per_task
                }
            }

            let mut var = ndarray::Array2::<T>::zeros((nx * ny * nz, info.vectorsize));
            self.read_variable_into::<T>(Some(name), None, var.as_slice_mut().unwrap());
            let bbox = [nx, ny, nz];
            let mut ordered_var = Array4::<T>::zeros((nx, ny, nz, info.vectorsize));
            let mut current_offset = 0;
            for i in 0..ntasks {
                let x = (i / decomp[2]) / decomp[1];
                let y = (i / decomp[2]) % decomp[1];
                let z = i % decomp[2];

                let task_size = [
                    calc_local_size(bbox[0], decomp[0], x),
                    calc_local_size(bbox[1], decomp[1], y),
                    calc_local_size(bbox[2], decomp[2], z),
                ];

                let task_start = [
                    calc_local_start(bbox[0], decomp[0], x),
                    calc_local_start(bbox[1], decomp[1], y),
                    calc_local_start(bbox[2], decomp[2], z),
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

        fn get_blocks_per_cell_dataset(&self, pop: &str) -> Option<VlsvDataset> {
            TryInto::<VlsvDataset>::try_into(
                self.root()
                    .blockspercell
                    .as_ref()?
                    .iter()
                    .find(|v| v.name.as_deref() == Some(pop))?,
            )
            .ok()
        }

        fn get_cells_with_blocks_dataset(&self, pop: &str) -> Option<VlsvDataset> {
            TryInto::<VlsvDataset>::try_into(
                self.root()
                    .cellswithblocks
                    .as_ref()?
                    .iter()
                    .find(|v| v.name.as_deref() == Some(pop))?,
            )
            .ok()
        }

        fn get_block_variable_dataset(&self, pop: &str) -> Option<VlsvDataset> {
            TryInto::<VlsvDataset>::try_into(
                self.root()
                    .blockvariable
                    .as_ref()?
                    .iter()
                    .find(|v| v.name.as_deref() == Some(pop))?,
            )
            .ok()
        }

        pub fn read_vdf_dict<T>(&self, cid: usize, pop: &str) -> Option<HashMap<usize, T>>
        where
            T: Pod
                + Zero
                + Num
                + NumCast
                + std::iter::Sum
                + Default
                + TypeTag
                + Float
                + ndarray::ScalarOperand
                + std::cmp::PartialOrd
                + std::fmt::Debug,
        {
            let blockspercell = self.get_blocks_per_cell_dataset(pop)?;
            let cellswithblocks = self.get_cells_with_blocks_dataset(pop)?;
            let blockvariable = self.get_block_variable_dataset(pop)?;
            let wid = self.get_wid(pop)?;
            let wid3 = wid.pow(3);
            let mut cids_with_blocks: Vec<usize> = vec![0; cellswithblocks.arraysize];
            let mut blocks_per_cell: Vec<u32> = vec![0; blockspercell.arraysize];
            self.read_variable_into::<usize>(
                None,
                Some(cellswithblocks.clone()),
                &mut cids_with_blocks,
            );
            self.read_variable_into::<u32>(None, Some(blockspercell), &mut blocks_per_cell);

            let index = cids_with_blocks
                .iter()
                .position(|&v| v == cid)
                .expect("CID DOES NOT CONTAIN VDF!");

            let read_size = blocks_per_cell[index] as usize;
            let start_block = blocks_per_cell[..index]
                .iter()
                .map(|&x| x as usize)
                .sum::<usize>();

            let slice_ds =
                |ds: &VlsvDataset, elem_offset: usize, elem_count: usize| -> VlsvDataset {
                    let mut sub = ds.clone();
                    sub.offset = ds.offset + elem_offset * ds.vectorsize * ds.datasize;
                    sub.arraysize = elem_count;
                    sub
                };

            let compression_used = blockvariable
                .compression
                .clone()
                .unwrap_or(CompressionMethod::NONE);

            let mut vdf_map = HashMap::with_capacity(1 << 20);
            let sparse: T = self
                .read_sparsity(pop, cid)
                .unwrap_or(T::from(1e-16).unwrap());

            match compression_used {
                CompressionMethod::HERMITE => {
                    let vdf_byte_size =
                        self.read_scalar_parameter("VDF_BYTE_SIZE").unwrap() as usize;
                    if vdf_byte_size != std::mem::size_of::<T>() {
                        panic!(
                            "This reader will not work for this combo of T and compressed VDF BYTE SIZE"
                        );
                    }
                    let bytespercell = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .bytespercell
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;
                    let mut bytes_per_cell: Vec<u64> = vec![0; bytespercell.arraysize];
                    self.read_variable_into::<u64>(None, Some(bytespercell), &mut bytes_per_cell);
                    let index = cids_with_blocks.iter().position(|&v| v == cid)?;
                    let read_size = bytes_per_cell[index] as usize;
                    let read_offset = bytes_per_cell[..index]
                        .iter()
                        .map(|&x| x as usize)
                        .sum::<usize>();
                    let mut hermite_bytes: Vec<u8> = vec![0_u8; read_size];
                    let blockvar_slice = slice_ds(&blockvariable, read_offset, read_size);
                    self.read_variable_into::<u8>(None, Some(blockvar_slice), &mut hermite_bytes);
                    let hermite_state = parse_hermite_state(&hermite_bytes);
                    let vdf = reconstruct_vdf::<f32>(&hermite_state);
                    let mut vdf_t = vdf.mapv(|val| T::from(val).unwrap());
                    let scale = T::from(0.1).unwrap();
                    let factor = scale * sparse;
                    let base = T::from(10.0).unwrap();
                    let max_exponent = T::from(38.0).unwrap();
                    vdf_t.map_inplace(|x| {
                        let clamped_x = (*x).min(max_exponent);
                        *x = factor * base.powf(clamped_x);
                        if *x < sparse {
                            *x = T::zero();
                        }
                    });
                    let (nvx, nvy, nvz) = self.get_vspace_mesh_bbox(pop)?;
                    let extents = self.get_vspace_mesh_extents(pop)?;

                    let dvx = (extents.3 - extents.0) / nvx as f64;
                    let dvy = (extents.4 - extents.1) / nvy as f64;
                    let dvz = (extents.5 - extents.2) / nvz as f64;

                    let start_i = ((hermite_state.v_limits[0] - extents.0) / dvx).round() as usize;
                    let start_j = ((hermite_state.v_limits[1] - extents.1) / dvy).round() as usize;
                    let start_k = ((hermite_state.v_limits[2] - extents.2) / dvz).round() as usize;
                    let blocks_per_dim_x = nvx / wid;
                    let blocks_per_dim_y = nvy / wid;

                    for (coord, &val) in vdf_t.indexed_iter() {
                        let (i, j, k, _) = coord;
                        let gi = i + start_i;
                        let gj = j + start_j;
                        let gk = k + start_k;
                        let block_i = gi / wid;
                        let block_j = gj / wid;
                        let block_k = gk / wid;
                        let li = gi % wid;
                        let lj = gj % wid;
                        let lk = gk % wid;
                        let linear_block_idx =
                            block_i + blocks_per_dim_x * (block_j + blocks_per_dim_y * block_k);
                        let local_id = li + wid * (lj + wid * lk);
                        vdf_map.insert(local_id + (linear_block_idx * wid3), val);
                    }
                }

                CompressionMethod::NONE | CompressionMethod::ZFP => {
                    let mut blocks: Vec<T> = vec![];
                    let vsamples = read_size * wid3;

                    match compression_used {
                        CompressionMethod::NONE => {
                            let blockvar_slice = slice_ds(&blockvariable, start_block, read_size);
                            blocks.resize(vsamples, T::default());
                            self.read_variable_into::<T>(None, Some(blockvar_slice), &mut blocks);
                        }
                        CompressionMethod::ZFP => {
                            let vdf_byte_size =
                                self.read_scalar_parameter("VDF_BYTE_SIZE").unwrap() as usize;
                            if vdf_byte_size != std::mem::size_of::<T>() {
                                panic!("Reader mismatch for T and compressed VDF BYTE SIZE");
                            }
                            let bytespercell = TryInto::<VlsvDataset>::try_into(
                                self.root()
                                    .bytespercell
                                    .as_ref()?
                                    .iter()
                                    .find(|v| v.name.as_deref() == Some(pop))?,
                            )
                            .ok()?;

                            let mut bytes_per_cell: Vec<u64> = vec![0; bytespercell.arraysize];
                            self.read_variable_into::<u64>(
                                None,
                                Some(bytespercell),
                                &mut bytes_per_cell,
                            );

                            let zread_size = bytes_per_cell[index] as usize;
                            let zstart_block = bytes_per_cell[..index]
                                .iter()
                                .map(|&x| x as usize)
                                .sum::<usize>();

                            let mut zblocks: Vec<u8> = vec![0_u8; zread_size];
                            let zblockvar_slice =
                                slice_ds(&blockvariable, zstart_block, zread_size);
                            self.read_variable_into::<u8>(
                                None,
                                Some(zblockvar_slice),
                                &mut zblocks,
                            );

                            let retval = unsafe {
                                zfp_decompress_1d_f32(&zblocks, vsamples, sparse.to_f64().unwrap())
                                    .unwrap()
                            };

                            blocks.resize(retval.len(), T::zero());
                            blocks.copy_from_slice(bytemuck::cast_slice(retval.as_slice()));
                        }
                        _ => unreachable!(),
                    }

                    let blockids = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .blockids
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mut block_ids: Vec<u32> = vec![0; read_size * blockids.vectorsize];
                    let blockids_slice = slice_ds(&blockids, start_block, read_size);
                    self.read_variable_into::<u32>(None, Some(blockids_slice), &mut block_ids);
                    for (block_idx, &bid_u32) in block_ids.iter().enumerate() {
                        let bid = bid_u32 as usize;
                        let block_buf = &blocks[block_idx * wid3..(block_idx + 1) * wid3];
                        for local_id in 0..wid3 {
                            let val = block_buf[local_id];
                            vdf_map.insert(local_id + (bid * wid3), val);
                        }
                    }
                }

                #[cfg(no_octree)]
                CompressionMethod::OCTREE => {
                    panic!("Octree compression not supported");
                }
                #[cfg(not(no_octree))]
                CompressionMethod::OCTREE => {
                    use std::os::raw::c_uchar;
                    #[cfg(not(no_octree))]
                    #[link(name = "toctree_compressor")]
                    #[cfg(not(no_octree))]
                    unsafe extern "C" {
                        pub fn uncompress_with_toctree_method(
                            buffer: *mut f32,
                            Nx: usize,
                            Ny: usize,
                            Nz: usize,
                            serialized_buffer: *mut c_uchar,
                            serialized_buffer_size: u64,
                        );
                    }

                    let bytespercell = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .bytespercell
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;
                    let mut bytes_per_cell: Vec<u64> = vec![0; bytespercell.arraysize];
                    self.read_variable_into::<u64>(None, Some(bytespercell), &mut bytes_per_cell);

                    let zread_size = bytes_per_cell[index] as usize;
                    let zstart_block = bytes_per_cell[..index]
                        .iter()
                        .map(|&x| x as usize)
                        .sum::<usize>();
                    let mut octree_bytes: Vec<u8> = vec![0_u8; zread_size];
                    let blockvar_slice = slice_ds(&blockvariable, zstart_block, zread_size);
                    self.read_variable_into::<u8>(None, Some(blockvar_slice), &mut octree_bytes);

                    let octree_state =
                        parse_octree_state(&octree_bytes).expect("OCTREE decode error");
                    let octree_core = &mut octree_bytes[octree_state.read_index..];
                    let mut decompressed_vdf = Array4::<T>::zeros([
                        octree_state.bbox_shape[0],
                        octree_state.bbox_shape[1],
                        octree_state.bbox_shape[2],
                        1,
                    ]);

                    #[cfg(not(no_octree))]
                    unsafe {
                        uncompress_with_toctree_method(
                            decompressed_vdf.as_mut_ptr() as *mut f32,
                            octree_state.bbox_shape[0],
                            octree_state.bbox_shape[1],
                            octree_state.bbox_shape[2],
                            octree_core.as_mut_ptr(),
                            octree_core.len() as u64,
                        );
                    }

                    let (nvx, nvy, nvz) = self.get_vspace_mesh_bbox(pop)?;
                    let extents = self.get_vspace_mesh_extents(pop)?;

                    let dvx = (extents.3 - extents.0) / nvx as f64;
                    let dvy = (extents.4 - extents.1) / nvy as f64;
                    let dvz = (extents.5 - extents.2) / nvz as f64;

                    let start_i = ((octree_state.bbox_lims[0] - extents.0) / dvx).round() as usize;
                    let start_j = ((octree_state.bbox_lims[1] - extents.1) / dvy).round() as usize;
                    let start_k = ((octree_state.bbox_lims[2] - extents.2) / dvz).round() as usize;
                    let blocks_per_dim_x = nvx / wid;
                    let blocks_per_dim_y = nvy / wid;

                    for (coord, &val) in decompressed_vdf.indexed_iter() {
                        let (i, j, k, _) = coord;
                        let gi = i + start_i;
                        let gj = j + start_j;
                        let gk = k + start_k;
                        let block_i = gi / wid;
                        let block_j = gj / wid;
                        let block_k = gk / wid;
                        let li = gi % wid;
                        let lj = gj % wid;
                        let lk = gk % wid;
                        let linear_block_idx =
                            block_i + blocks_per_dim_x * (block_j + blocks_per_dim_y * block_k);
                        let local_id = li + wid * (lj + wid * lk);
                        vdf_map.insert(local_id + (linear_block_idx * wid3), val);
                    }
                }

                #[cfg(not(no_nn))]
                CompressionMethod::MLP | CompressionMethod::MLPMULTI => {
                    let ntasks = self.get_writting_tasks()?;
                    let mlp_bytes_per_rank_dset = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .mlp_bytes_per_rank
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mlp_clusters_per_rank_dset = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .mlp_clusters_per_rank
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mut mlp_bytes_per_rank: Vec<u64> =
                        vec![0; mlp_bytes_per_rank_dset.arraysize];
                    let mut mlp_clusters_per_rank: Vec<u64> =
                        vec![0; mlp_clusters_per_rank_dset.arraysize];

                    self.read_variable_into::<u64>(
                        None,
                        Some(mlp_bytes_per_rank_dset),
                        &mut mlp_bytes_per_rank,
                    );
                    self.read_variable_into::<u64>(
                        None,
                        Some(mlp_clusters_per_rank_dset),
                        &mut mlp_clusters_per_rank,
                    );
                    let nmlps: u64 = mlp_clusters_per_rank.iter().sum();
                    let mut current_offset: u64 = 0;
                    let scan_bytes_per_cell: Vec<u64> = mlp_bytes_per_rank
                        .iter()
                        .map(|&b| {
                            let prev = current_offset;
                            current_offset += b as u64;
                            prev
                        })
                        .collect();

                    let mut mlp_headers = vec![Header::default(); nmlps as usize];
                    let mut nbytes_multi_mlp_case = Vec::new();
                    let mut cnt = 0;
                    let blockvariable = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .blockvariable
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    for i in 0..ntasks {
                        let mut offset = 0;
                        for _cluster in 0..mlp_clusters_per_rank[i] {
                            let target_offset = scan_bytes_per_cell[i] + offset;
                            let mut target_ds = blockvariable.clone();
                            target_ds.offset += target_offset as usize;
                            target_ds.arraysize = std::mem::size_of::<Header>();
                            self.read_variable_into::<u8>(
                                None,
                                Some(target_ds),
                                bytemuck::cast_slice_mut(&mut mlp_headers[cnt..cnt + 1]),
                            );
                            if nmlps > ntasks as u64 {
                                nbytes_multi_mlp_case
                                    .push(mlp_headers[cnt].total_size.try_into().unwrap());
                            }
                            offset += mlp_headers[cnt].total_size as u64;
                            cnt += 1;
                        }
                    }

                    let mut scan_bytes_per_cell = scan_bytes_per_cell;
                    let mut current_nbytes = vec![0; mlp_bytes_per_rank.len()];
                    if nmlps > ntasks as u64 {
                        current_nbytes = nbytes_multi_mlp_case;
                        let mut current_offset: u64 = 0;
                        scan_bytes_per_cell = current_nbytes
                            .iter()
                            .map(|&b| {
                                let prev = current_offset;
                                current_offset += b as u64;
                                prev
                            })
                            .collect();
                    }

                    type CellID = u64;
                    let mut mlp_cids: Vec<Vec<CellID>> = vec![Vec::new(); nmlps as usize];
                    let mut cnt = 0;

                    for i in 0..ntasks {
                        for _cluster in 0..mlp_clusters_per_rank[i] {
                            let cols = mlp_headers[cnt].cols;
                            mlp_cids[cnt].resize(cols, 0 as CellID);
                            let target_offset =
                                scan_bytes_per_cell[cnt] + std::mem::size_of::<Header>() as u64;

                            let mut target_ds = blockvariable.clone();
                            target_ds.offset += target_offset as usize;
                            target_ds.arraysize = cols * std::mem::size_of::<CellID>();
                            self.read_variable_into::<CellID>(
                                None,
                                Some(target_ds),
                                &mut mlp_cids[cnt],
                            );

                            cnt += 1;
                        }
                    }

                    let cellswithblocks = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .cellswithblocks
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mut cids_with_blocks: Vec<usize> = vec![0; cellswithblocks.arraysize];
                    self.read_variable_into::<usize>(
                        None,
                        Some(cellswithblocks),
                        &mut cids_with_blocks,
                    );

                    let target_mlp: usize = mlp_cids
                        .iter()
                        .position(|cand_vec| cand_vec.contains(&(cid as u64)))
                        .expect("cid not found in any mlp");
                    let fourier_order =
                        self.read_scalar_parameter("FOURIER_ORDER").unwrap() as usize;
                    let mlp_arch_dset = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .mlp_arch
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;
                    let mut mlp_arch: Vec<i64> = vec![0; mlp_arch_dset.arraysize];
                    self.read_variable_into::<i64>(None, Some(mlp_arch_dset), &mut mlp_arch);
                    let mlp_arch_usize: Vec<usize> = mlp_arch.iter().map(|&x| x as usize).collect();
                    let mut target_ds = blockvariable.clone();
                    target_ds.offset += scan_bytes_per_cell[target_mlp] as usize;
                    target_ds.arraysize = mlp_headers[target_mlp].total_size;
                    let mut mlp_bytes = vec![u8::zero(); mlp_headers[target_mlp].total_size];
                    self.read_variable_into::<u8>(None, Some(target_ds), &mut mlp_bytes);
                    let mut phasespace = PhaseSpaceUnion::<f32>::new_from_buffer(&mlp_bytes)
                        .expect("Could deserialize phasespace");
                    phasespace.decompress(&mlp_arch_usize, fourier_order);
                    phasespace.unnormalize_and_unscale(<f32 as NumCast>::from(sparse).unwrap());
                    let (nvx, nvy, nvz) = self.get_vspace_mesh_bbox(pop).unwrap();
                    let global_extents = self.get_vspace_mesh_extents(pop).unwrap();
                    let dvx = (global_extents.3.to_f64().unwrap()
                        - global_extents.0.to_f64().unwrap())
                        / nvx as f64;
                    let dvy = (global_extents.4.to_f64().unwrap()
                        - global_extents.1.to_f64().unwrap())
                        / nvy as f64;
                    let dvz = (global_extents.5.to_f64().unwrap()
                        - global_extents.2.to_f64().unwrap())
                        / nvz as f64;

                    let start_i = ((phasespace.v_limits[0] as f64
                        - global_extents.0.to_f64().unwrap())
                        / dvx)
                        .round() as usize;
                    let start_j = ((phasespace.v_limits[1] as f64
                        - global_extents.1.to_f64().unwrap())
                        / dvy)
                        .round() as usize;
                    let start_k = ((phasespace.v_limits[2] as f64
                        - global_extents.2.to_f64().unwrap())
                        / dvz)
                        .round() as usize;
                    let vdf = phasespace.reconstruct_vdf_dense(cid, dvx as f32);
                    let vdf_t = vdf.mapv(|val| T::from(val).unwrap());
                    let blocks_per_row = nvx / wid;
                    let blocks_per_layer = (nvx / wid) * (nvy / wid);
                    for (coord, &val) in vdf_t.indexed_iter() {
                        let (i, j, k, _) = coord;
                        let gi = i + start_i;
                        let gj = j + start_j;
                        let gk = k + start_k;
                        let block_i = gi / wid;
                        let block_j = gj / wid;
                        let block_k = gk / wid;
                        let li = gi % wid;
                        let lj = gj % wid;
                        let lk = gk % wid;
                        let bid =
                            block_i + (block_j * blocks_per_row) + (block_k * blocks_per_layer);
                        let local_id = li + wid * (lj + wid * lk);
                        vdf_map.insert(local_id + (bid * wid3), val);
                    }
                }

                #[cfg(no_nn)]
                CompressionMethod::MLP | CompressionMethod::MLPMULTI => {
                    panic!("Compiled without MLP support")
                }
            }

            Some(vdf_map)
        }

        pub fn get_hermite<T>(&self, cid: usize, pop: &str) -> Option<HermiteState>
        where
            T: Pod
                + Zero
                + Num
                + NumCast
                + std::iter::Sum
                + Default
                + TypeTag
                + Float
                + ndarray::ScalarOperand
                + std::cmp::PartialOrd
                + std::fmt::Debug,
        {
            let blockspercell = self.get_blocks_per_cell_dataset(pop)?;
            let cellswithblocks = self.get_cells_with_blocks_dataset(pop)?;
            let blockvariable = self.get_block_variable_dataset(pop)?;

            let mut cids_with_blocks: Vec<usize> = vec![0; cellswithblocks.arraysize];
            let mut blocks_per_cell: Vec<u32> = vec![0; blockspercell.arraysize];
            self.read_variable_into::<usize>(None, Some(cellswithblocks), &mut cids_with_blocks);
            self.read_variable_into::<u32>(None, Some(blockspercell), &mut blocks_per_cell);
            fn slice_ds(ds: &VlsvDataset, elem_offset: usize, elem_count: usize) -> VlsvDataset {
                let mut sub = ds.clone();
                sub.offset = ds.offset + elem_offset * ds.vectorsize * ds.datasize;
                sub.arraysize = elem_count;
                sub
            }

            let compression_used = blockvariable
                .compression
                .clone()
                .unwrap_or(CompressionMethod::NONE);

            match compression_used {
                CompressionMethod::HERMITE => {
                    let vdf_byte_size =
                        self.read_scalar_parameter("VDF_BYTE_SIZE").unwrap() as usize;
                    if vdf_byte_size != std::mem::size_of::<T>() {
                        panic!(
                            "This reader will not work for this combo of T and compressed VDF BYTE SIZE"
                        );
                    }
                    let bytespercell = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .bytespercell
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;
                    let mut bytes_per_cell: Vec<u64> = vec![0; bytespercell.arraysize];
                    self.read_variable_into::<u64>(None, Some(bytespercell), &mut bytes_per_cell);
                    let index = cids_with_blocks.iter().position(|&v| v == cid)?;
                    let read_size = bytes_per_cell[index] as usize;
                    let read_offset = bytes_per_cell[..index]
                        .iter()
                        .map(|&x| x as usize)
                        .sum::<usize>();
                    let mut hermite_bytes: Vec<u8> = vec![0_u8; read_size];
                    let blockvar_slice = slice_ds(&blockvariable, read_offset, read_size);
                    self.read_variable_into::<u8>(None, Some(blockvar_slice), &mut hermite_bytes);
                    let hermite_state = parse_hermite_state(&hermite_bytes);
                    Some(hermite_state)
                }
                _ => {
                    panic!(
                        "This file is not using Hermite compression so the hermite state could not be read in!"
                    );
                }
            }
        }

        pub fn read_vdf<T>(&self, cid: usize, pop: &str) -> Option<Array4<T>>
        where
            T: Pod
                + Zero
                + Num
                + NumCast
                + std::iter::Sum
                + Default
                + TypeTag
                + Float
                + ndarray::ScalarOperand
                + std::cmp::PartialOrd
                + std::fmt::Debug,
        {
            let blockspercell = self.get_blocks_per_cell_dataset(pop)?;
            let cellswithblocks = self.get_cells_with_blocks_dataset(pop)?;
            let blockvariable = self.get_block_variable_dataset(pop)?;
            let wid = self.get_wid(pop)?;
            let wid3 = wid.pow(3);
            let (nvx, nvy, nvz) = self.get_vspace_mesh_bbox(pop)?;
            let (mx, my, mz) = (nvx / wid, nvy / wid, nvz / wid);

            let mut cids_with_blocks: Vec<usize> = vec![0; cellswithblocks.arraysize];
            let mut blocks_per_cell: Vec<u32> = vec![0; blockspercell.arraysize];
            self.read_variable_into::<usize>(None, Some(cellswithblocks), &mut cids_with_blocks);
            self.read_variable_into::<u32>(None, Some(blockspercell), &mut blocks_per_cell);
            let index_res = cids_with_blocks.iter().position(|&v| v == cid);
            let index = if let Some(v) = index_res {
                v
            } else {
                println!("Available cids with blocks{:?}", cids_with_blocks);
                panic!("CID DOES NOT CONTAINS VDF!");
            };
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

            let id2ijk = |id: usize| -> (usize, usize, usize) {
                let plane = mx * my;
                debug_assert!(id < plane * mz, "GID out of bounds");
                let k = id / plane;
                let rem = id % plane;
                let j = rem / mx;
                let i = rem % mx;
                (i, j, k)
            };

            // Read block data (T)
            let vsamples = read_size * wid3;
            let mut blocks: Vec<T> = vec![];
            let compression_used = &blockvariable
                .compression
                .clone()
                .unwrap_or(CompressionMethod::NONE);

            let sparse: T = self
                .read_sparsity(pop, cid)
                .unwrap_or(T::from(1e-16).unwrap());
            match compression_used {
                CompressionMethod::NONE => {
                    let blockids = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .blockids
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mut block_ids: Vec<u32> = vec![0; read_size * blockids.vectorsize];
                    let blockids_slice = slice_ds(&blockids, start_block, read_size);
                    self.read_variable_into::<u32>(None, Some(blockids_slice), &mut block_ids);

                    let blockvar_slice = slice_ds(&blockvariable, start_block, read_size);
                    blocks.resize(vsamples, T::default());
                    self.read_variable_into::<T>(None, Some(blockvar_slice), &mut blocks);
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
                                    let val = block_buf[local_id];
                                    if val >= sparse {
                                        vdf[(gi, gj, gk, 0)] = block_buf[local_id];
                                    }
                                }
                            }
                        }
                    }
                    Some(vdf)
                }
                CompressionMethod::ZFP => {
                    let vdf_byte_size =
                        self.read_scalar_parameter("VDF_BYTE_SIZE").unwrap() as usize;
                    if vdf_byte_size != std::mem::size_of::<T>() {
                        panic!(
                            "This reader will not work for this combo of T and compressed VDF BYTE SIZE"
                        );
                    }
                    let blockids = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .blockids
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mut block_ids: Vec<u32> = vec![0; read_size * blockids.vectorsize];
                    let blockids_slice = slice_ds(&blockids, start_block, read_size);
                    self.read_variable_into::<u32>(None, Some(blockids_slice), &mut block_ids);

                    let bytespercell = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .bytespercell
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;
                    let mut bytes_per_cell: Vec<u64> = vec![0; bytespercell.arraysize];
                    self.read_variable_into::<u64>(None, Some(bytespercell), &mut bytes_per_cell);
                    let zindex = cids_with_blocks.iter().position(|&v| v == cid)?;
                    let zread_size = bytes_per_cell[zindex] as usize;
                    let zstart_block = bytes_per_cell[..zindex]
                        .iter()
                        .map(|&x| x as usize)
                        .sum::<usize>();
                    let mut zblocks: Vec<u8> = vec![0_u8; zread_size];
                    let zblockvar_slice = slice_ds(&blockvariable, zstart_block, zread_size);
                    self.read_variable_into::<u8>(None, Some(zblockvar_slice), &mut zblocks);
                    let retval = unsafe {
                        zfp_decompress_1d_f32(
                            &zblocks,
                            vsamples,
                            sparse.to_f64().expect("Failed to cast sparse value to f64"),
                        )
                        .unwrap()
                    };
                    blocks.resize(retval.len(), T::zeroed());
                    blocks.copy_from_slice(bytemuck::cast_slice(retval.as_slice()));
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
                                    let val = block_buf[local_id];
                                    if val >= sparse {
                                        vdf[(gi, gj, gk, 0)] = block_buf[local_id];
                                    }
                                }
                            }
                        }
                    }
                    Some(vdf)
                }
                //No blockids here
                #[cfg(no_octree)]
                CompressionMethod::OCTREE => {
                    panic!("Octree compression not supported");
                }
                #[cfg(not(no_octree))]
                CompressionMethod::OCTREE => {
                    use core::ffi::{c_float, c_ulonglong};
                    use std::os::raw::c_uchar;
                    type VdfReal = c_float;
                    #[cfg(not(no_octree))]
                    #[link(name = "toctree_compressor")]
                    unsafe extern "C" {
                        pub fn uncompress_with_toctree_method(
                            buffer: *mut VdfReal,
                            Nx: usize,
                            Ny: usize,
                            Nz: usize,
                            serialized_buffer: *mut c_uchar,
                            serialized_buffer_size: c_ulonglong,
                        );
                    }
                    let vdf_byte_size =
                        self.read_scalar_parameter("VDF_BYTE_SIZE").unwrap() as usize;
                    if vdf_byte_size != std::mem::size_of::<T>() {
                        panic!(
                            "This reader will not work for this combo of T and compressed VDF BYTE SIZE"
                        );
                    }
                    let bytespercell = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .bytespercell
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;
                    let mut bytes_per_cell: Vec<u64> = vec![0; bytespercell.arraysize];
                    self.read_variable_into::<u64>(None, Some(bytespercell), &mut bytes_per_cell);
                    let index = cids_with_blocks.iter().position(|&v| v == cid)?;
                    let read_size = bytes_per_cell[index] as usize;
                    let read_offset = bytes_per_cell[..index]
                        .iter()
                        .map(|&x| x as usize)
                        .sum::<usize>();
                    let mut octree_bytes: Vec<u8> = vec![0_u8; read_size];
                    let blockvar_slice = slice_ds(&blockvariable, read_offset, read_size);
                    self.read_variable_into::<u8>(None, Some(blockvar_slice), &mut octree_bytes);
                    let octree_state: OctreeState = parse_octree_state(&octree_bytes)
                        .expect("ERROR: Could not decode OCTREE State");

                    let read_index = octree_state.read_index;
                    let octree_core = &mut octree_bytes[read_index..];
                    let len = octree_core.len();
                    let mut decompressed_vdf = Array4::<T>::zeros([
                        octree_state.bbox_shape[0],
                        octree_state.bbox_shape[1],
                        octree_state.bbox_shape[2],
                        1,
                    ]);

                    unsafe {
                        uncompress_with_toctree_method(
                            decompressed_vdf.as_mut_ptr() as *mut f32,
                            octree_state.bbox_shape[0],
                            octree_state.bbox_shape[1],
                            octree_state.bbox_shape[2],
                            octree_core.as_mut_ptr(),
                            len as u64,
                        );
                    }
                    decompressed_vdf.iter_mut().for_each(|val| {
                        if *val < sparse {
                            *val = T::zero();
                        }
                    });
                    Some(decompressed_vdf)
                }
                CompressionMethod::HERMITE => {
                    let vdf_byte_size =
                        self.read_scalar_parameter("VDF_BYTE_SIZE").unwrap() as usize;
                    if vdf_byte_size != std::mem::size_of::<T>() {
                        panic!(
                            "This reader will not work for this combo of T and compressed VDF BYTE SIZE"
                        );
                    }
                    let bytespercell = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .bytespercell
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;
                    let mut bytes_per_cell: Vec<u64> = vec![0; bytespercell.arraysize];
                    self.read_variable_into::<u64>(None, Some(bytespercell), &mut bytes_per_cell);
                    let index = cids_with_blocks.iter().position(|&v| v == cid)?;
                    let read_size = bytes_per_cell[index] as usize;
                    let read_offset = bytes_per_cell[..index]
                        .iter()
                        .map(|&x| x as usize)
                        .sum::<usize>();
                    let mut hermite_bytes: Vec<u8> = vec![0_u8; read_size];
                    let blockvar_slice = slice_ds(&blockvariable, read_offset, read_size);
                    self.read_variable_into::<u8>(None, Some(blockvar_slice), &mut hermite_bytes);
                    let hermite_state = parse_hermite_state(&hermite_bytes);
                    let vdf = reconstruct_vdf::<f32>(&hermite_state);
                    let mut vdf_t = vdf.mapv(|val| T::from(val).unwrap());
                    let scale = T::from(0.1).unwrap();
                    let factor = scale * sparse;
                    let base = T::from(10.0).unwrap();
                    let max_exponent = T::from(38.0).unwrap();
                    vdf_t.map_inplace(|x| {
                        let clamped_x = (*x).min(max_exponent);
                        *x = factor * base.powf(clamped_x);
                        if *x < sparse {
                            *x = T::zero();
                        }
                    });
                    Some(vdf_t)
                }
                #[cfg(not(no_nn))]
                CompressionMethod::MLP | CompressionMethod::MLPMULTI => {
                    let ntasks = self.get_writting_tasks()?;
                    let mlp_bytes_per_rank_dset = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .mlp_bytes_per_rank
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mlp_clusters_per_rank_dset = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .mlp_clusters_per_rank
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mut mlp_bytes_per_rank: Vec<u64> =
                        vec![0; mlp_bytes_per_rank_dset.arraysize];
                    let mut mlp_clusters_per_rank: Vec<u64> =
                        vec![0; mlp_clusters_per_rank_dset.arraysize];

                    self.read_variable_into::<u64>(
                        None,
                        Some(mlp_bytes_per_rank_dset),
                        &mut mlp_bytes_per_rank,
                    );
                    self.read_variable_into::<u64>(
                        None,
                        Some(mlp_clusters_per_rank_dset),
                        &mut mlp_clusters_per_rank,
                    );
                    let nmlps: u64 = mlp_clusters_per_rank.iter().sum();
                    let mut current_offset: u64 = 0;
                    let scan_bytes_per_cell: Vec<u64> = mlp_bytes_per_rank
                        .iter()
                        .map(|&b| {
                            let prev = current_offset;
                            current_offset += b as u64;
                            prev
                        })
                        .collect();

                    let mut mlp_headers = vec![Header::default(); nmlps as usize];
                    let mut nbytes_multi_mlp_case = Vec::new();
                    let mut cnt = 0;
                    let blockvariable = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .blockvariable
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    for i in 0..ntasks {
                        let mut offset = 0;
                        for _cluster in 0..mlp_clusters_per_rank[i] {
                            let target_offset = scan_bytes_per_cell[i] + offset;
                            let mut target_ds = blockvariable.clone();
                            target_ds.offset += target_offset as usize;
                            target_ds.arraysize = std::mem::size_of::<Header>();
                            self.read_variable_into::<u8>(
                                None,
                                Some(target_ds),
                                bytemuck::cast_slice_mut(&mut mlp_headers[cnt..cnt + 1]),
                            );
                            if nmlps > ntasks as u64 {
                                nbytes_multi_mlp_case
                                    .push(mlp_headers[cnt].total_size.try_into().unwrap());
                            }
                            offset += mlp_headers[cnt].total_size as u64;
                            cnt += 1;
                        }
                    }

                    let mut scan_bytes_per_cell = scan_bytes_per_cell;
                    let mut current_nbytes = vec![0; mlp_bytes_per_rank.len()];
                    if nmlps > ntasks as u64 {
                        current_nbytes = nbytes_multi_mlp_case;
                        let mut current_offset: u64 = 0;
                        scan_bytes_per_cell = current_nbytes
                            .iter()
                            .map(|&b| {
                                let prev = current_offset;
                                current_offset += b as u64;
                                prev
                            })
                            .collect();
                    }

                    type CellID = u64;
                    let mut mlp_cids: Vec<Vec<CellID>> = vec![Vec::new(); nmlps as usize];
                    let mut cnt = 0;

                    for i in 0..ntasks {
                        for _cluster in 0..mlp_clusters_per_rank[i] {
                            let cols = mlp_headers[cnt].cols;
                            mlp_cids[cnt].resize(cols, 0 as CellID);
                            let target_offset =
                                scan_bytes_per_cell[cnt] + std::mem::size_of::<Header>() as u64;

                            let mut target_ds = blockvariable.clone();
                            target_ds.offset += target_offset as usize;
                            target_ds.arraysize = cols * std::mem::size_of::<CellID>();
                            self.read_variable_into::<CellID>(
                                None,
                                Some(target_ds),
                                &mut mlp_cids[cnt],
                            );

                            cnt += 1;
                        }
                    }

                    let cellswithblocks = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .cellswithblocks
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;

                    let mut cids_with_blocks: Vec<usize> = vec![0; cellswithblocks.arraysize];
                    self.read_variable_into::<usize>(
                        None,
                        Some(cellswithblocks),
                        &mut cids_with_blocks,
                    );

                    let target_mlp: usize = mlp_cids
                        .iter()
                        .position(|cand_vec| cand_vec.contains(&(cid as u64)))
                        .expect("cid not found in any mlp");

                    let fourier_order =
                        self.read_scalar_parameter("FOURIER_ORDER").unwrap() as usize;
                    let mlp_arch_dset = TryInto::<VlsvDataset>::try_into(
                        self.root()
                            .mlp_arch
                            .as_ref()?
                            .iter()
                            .find(|v| v.name.as_deref() == Some(pop))?,
                    )
                    .ok()?;
                    let mut mlp_arch: Vec<i64> = vec![0; mlp_arch_dset.arraysize];
                    self.read_variable_into::<i64>(None, Some(mlp_arch_dset), &mut mlp_arch);
                    let mlp_arch_usize: Vec<usize> = mlp_arch.iter().map(|&x| x as usize).collect();
                    let mut target_ds = blockvariable.clone();
                    target_ds.offset += scan_bytes_per_cell[target_mlp] as usize;
                    target_ds.arraysize = mlp_headers[target_mlp].total_size;
                    let mut mlp_bytes = vec![u8::zero(); mlp_headers[target_mlp].total_size];
                    self.read_variable_into::<u8>(None, Some(target_ds), &mut mlp_bytes);
                    let mut phasespace = PhaseSpaceUnion::<f32>::new_from_buffer(&mlp_bytes)
                        .expect("Could deserialize phasespace");
                    phasespace.decompress(&mlp_arch_usize, fourier_order);
                    phasespace.unnormalize_and_unscale(<f32 as NumCast>::from(sparse).unwrap());
                    let bbox = self.get_vspace_mesh_bbox(pop).unwrap();
                    let extent = self.get_vspace_mesh_extents(pop).unwrap();
                    let dv = (extent.3 - extent.0) / bbox.0 as f64;
                    let vdf = phasespace.reconstruct_vdf_dense(cid, dv as f32);
                    let mut vdf_t = vdf.mapv(|val| T::from(val).unwrap());
                    vdf_t.iter_mut().for_each(|val| {
                        if *val < sparse {
                            *val = T::zero();
                        }
                    });
                    Some(vdf_t)
                }
                #[cfg(no_nn)]
                CompressionMethod::MLP | CompressionMethod::MLPMULTI => {
                    panic!("Compiled without MLP support");
                }
            }
        }

        pub fn read_vdf_into<T>(
            &self,
            cid: usize,
            pop: &str,
            target: &mut Array4<T>,
            target_extent: (f64, f64, f64, f64, f64, f64),
        ) -> Option<()>
        where
            T: Pod
                + Zero
                + Num
                + NumCast
                + std::iter::Sum
                + Default
                + TypeTag
                + std::cmp::PartialOrd
                + Float
                + ndarray::ScalarOperand
                + std::fmt::Debug,
        {
            let vdf: Array4<T> = self.read_vdf::<T>(cid, pop)?;
            let src_extent = self.get_vspace_mesh_extents(pop)?;
            // remesh_trilinear(&vdf, src_extent, target, target_extent);
            remesh_conservative(&vdf, src_extent, target, target_extent);
            Some(())
        }

        pub fn read_vdf_zoom<T>(
            &self,
            cid: usize,
            pop: &str,
            scale_factor: f64,
        ) -> Option<Array4<T>>
        where
            T: Pod
                + Zero
                + Num
                + NumCast
                + std::iter::Sum
                + Default
                + TypeTag
                + std::cmp::PartialOrd
                + Float
                + ndarray::ScalarOperand
                + std::fmt::Debug,
        {
            let dst_extents = self.get_vspace_mesh_extents(pop)?;

            let (nx, ny, nz) = {
                let (nx0, ny0, nz0) = self.get_vspace_mesh_bbox(pop).unwrap();
                (
                    ((nx0 as f64) / scale_factor).round() as usize,
                    ((ny0 as f64) / scale_factor).round() as usize,
                    ((nz0 as f64) / scale_factor).round() as usize,
                )
            };
            let mut vdf: Array4<T> = Array4::<T>::zeros((nx, ny, nz, 1));
            self.read_vdf_into(cid, pop, &mut vdf, dst_extents);
            Some(vdf)
        }

        pub fn read_sparsity<
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag + std::cmp::PartialOrd,
        >(
            &self,
            name: &str,
            cid: usize,
        ) -> Option<T> {
            let v = self.read_vg_variable_at_as_ref_dyn::<T>(name, &[cid], &mut [0])?[0];
            let mut k: f32 = f32::zero();
            for chunk in v.chunks_exact(std::mem::size_of::<f32>()) {
                k = pod_read_unaligned::<f32>(chunk);
            }
            T::from(k)
        }

        pub fn read_variable<
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag + std::cmp::PartialOrd,
        >(
            &self,
            name: &str,
            op: Option<i32>,
        ) -> Option<ndarray::Array4<T>> {
            self.read_fsgrid_variable::<T>(name, op)
                .or_else(|| self.read_vg_variable_as_fg::<T>(name, op))
        }

        pub fn read_variable_zoom<
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag + std::cmp::PartialOrd,
        >(
            &self,
            name: &str,
            op: Option<i32>,
            scale_factor: f64,
        ) -> Option<ndarray::Array4<T>> {
            let mesh = self
                .read_fsgrid_variable::<T>(name, op)
                .or_else(|| self.read_vg_variable_as_fg::<T>(name, op))?;

            let vector_dim = mesh.dim().3;
            let dst_extents = self.get_spatial_mesh_extents()?;
            let (nx, ny, nz) = {
                let (nx0, ny0, nz0) = self.get_spatial_mesh_bbox().unwrap();
                (
                    ((nx0 as f64) / scale_factor).round() as usize,
                    ((ny0 as f64) / scale_factor).round() as usize,
                    ((nz0 as f64) / scale_factor).round() as usize,
                )
            };
            let mut remesh: Array4<T> = Array4::<T>::zeros((nx, ny, nz, vector_dim));
            remesh_conservative(&mesh, dst_extents, &mut remesh, dst_extents);
            Some(remesh)
        }

        pub fn read_vg_variable_at<
            T: Pod + Zero + Num + NumCast + std::iter::Sum + Default + TypeTag + std::cmp::PartialOrd,
        >(
            &self,
            name: &str,
            cid: &[usize],
        ) -> Option<Vec<T>> {
            let mut info = self.get_dataset(name)?;
            if info.grid.clone()? != VlasiatorGrid::SPATIALGRID {
                panic!("This method only supports reading in VG variables");
            }
            let cellid_ds = self.get_dataset("CellID")?;
            let mut cell_ids = Vec::<u64>::with_capacity(cellid_ds.arraysize);
            unsafe { cell_ids.set_len(cellid_ds.arraysize) };
            self.read_variable_into::<u64>(None, Some(cellid_ds), &mut cell_ids);
            let indices = cid
                .iter()
                .map(|cand| {
                    cell_ids
                        .iter()
                        .position(|x| *x == *cand as u64)
                        .expect("Failed to find cellid {cand}")
                })
                .collect::<Vec<usize>>();

            let base_byte_offset = info.offset;
            let v_size = info.vectorsize;
            let total_elements = indices.len() * v_size;
            let mut retval = Vec::<T>::with_capacity(total_elements);
            unsafe { retval.set_len(total_elements) };

            retval
                .chunks_mut(v_size)
                .zip(indices)
                .for_each(|(slice, index)| {
                    info.offset = base_byte_offset + (index * v_size) * info.datasize;
                    info.arraysize = 1;
                    info.vectorsize = v_size;
                    self.read_variable_into::<T>(None, Some(info.clone()), slice);
                });
            Some(retval)
        }

        pub fn get_hints_for_cids_const<const N: usize>(&self, cids: &[usize; N]) -> [usize; N] {
            let cellid_ds = self.get_dataset("CellID").expect("CellIDs not found!");
            let cell_id_bytes = &self.memorymap()
                [cellid_ds.offset..cellid_ds.offset + cellid_ds.datasize * cellid_ds.arraysize];
            let cell_ids: &[u64] = bytemuck::try_cast_slice(cell_id_bytes)
                .expect("CELLIDS misaligned or wrong length");
            std::array::from_fn(|i| {
                let cand = cids[i] as u64;
                cell_ids
                    .iter()
                    .position(|x| *x == cand)
                    .unwrap_or_else(|| panic!("Failed to find cellid {cand}"))
            })
        }

        pub fn get_hints_for_cids(&self, cids: &[usize]) -> Vec<usize> {
            let cellid_ds = self.get_dataset("CellID").expect("CellIDs not found!");
            let cell_id_bytes = &self.memorymap()
                [cellid_ds.offset..cellid_ds.offset + cellid_ds.datasize * cellid_ds.arraysize];
            let cell_ids: &[u64] = bytemuck::try_cast_slice(cell_id_bytes)
                .expect("CELLIDS misaligned or wrong length");
            cids.iter()
                .map(|cand| {
                    cell_ids
                        .iter()
                        .position(|x| *x == *cand as u64)
                        .expect("Failed to find cellid {cand}")
                })
                .collect::<Vec<usize>>()
        }

        pub fn read_vg_variable_at_as_ref_dyn<'a, T>(
            &'a self,
            name: &str,
            cid: &[usize],
            hint: &mut [usize],
        ) -> Option<Vec<&'a [u8]>>
        where
            T: bytemuck::AnyBitPattern,
        {
            let info = self.get_dataset(name)?;
            if info.grid.clone()? != VlasiatorGrid::SPATIALGRID {
                panic!("This method only supports reading in VG variables");
            }

            assert_eq!(
                cid.len(),
                hint.len(),
                "CIDs and hint must have the same length."
            );

            if info.datasize != core::mem::size_of::<T>() {
                panic!(
                    "Size mismatch: dataset has datasize {}, function expects {}",
                    info.datasize,
                    core::mem::size_of::<T>()
                );
            }

            let cellid_ds = self.get_dataset("CellID")?;
            let cell_id_bytes = &self.memorymap()
                [cellid_ds.offset..cellid_ds.offset + cellid_ds.datasize * cellid_ds.arraysize];
            let cell_ids: &[u64] = bytemuck::try_cast_slice(cell_id_bytes)
                .expect("CELLIDS misaligned or wrong length");

            let mut indices = Vec::with_capacity(cid.len());
            for (i, &c) in cid.iter().enumerate() {
                let idx = find_near_with_hint(cell_ids, c as u64, hint[i])
                    .unwrap_or_else(|| panic!("Failed to find cellid {c}"));
                indices.push(idx);
            }
            hint.copy_from_slice(&indices);
            let stride_bytes = info.datasize * info.vectorsize;
            let mut retval = Vec::with_capacity(indices.len());
            for idx in indices {
                let off = info.offset + idx * stride_bytes;
                retval.push(&self.memorymap()[off..off + stride_bytes]);
            }
            Some(retval)
        }

        pub fn read_vg_variable_at_as_ref_const<'a, T, const N: usize>(
            &'a self,
            name: &str,
            cid: &[usize; N],
            hint: &mut [usize; N],
        ) -> Option<[&'a [u8]; N]>
        where
            T: bytemuck::AnyBitPattern,
        {
            let info = self.get_dataset(name)?;
            if info.grid.clone()? != VlasiatorGrid::SPATIALGRID {
                panic!("This method only supports reading in VG variables");
            }

            if info.datasize != core::mem::size_of::<T>() {
                panic!(
                    "Size mismatch: dataset has datasize {}, function expects {}",
                    info.datasize,
                    core::mem::size_of::<T>()
                );
            }

            let cellid_ds = self.get_dataset("CellID")?;
            let cell_id_bytes = &self.memorymap()
                [cellid_ds.offset..cellid_ds.offset + cellid_ds.datasize * cellid_ds.arraysize];
            let cell_ids: &[u64] = bytemuck::try_cast_slice(cell_id_bytes)
                .expect("CELLIDS misaligned or wrong length");

            let indices: [usize; N] = core::array::from_fn(|i| {
                let target = cid[i] as u64;
                let h = hint[i];
                find_near_with_hint(cell_ids, target, h)
                    .unwrap_or_else(|| panic!("Failed to find cellid {target}"))
            });

            hint.copy_from_slice(&indices);
            let stride_bytes = info.datasize * info.vectorsize;
            let out: [&'a [u8]; N] = core::array::from_fn(|i| {
                let idx = indices[i];
                let off = info.offset + idx * stride_bytes;
                &self.memorymap()[off..off + stride_bytes]
            });
            Some(out)
        }
    }

    #[derive(Debug)]
    pub struct HermiteState {
        pub n_hermite_harmonic: i32,
        pub vth: f32,
        pub u: [f32; 3],
        pub spectrum: Vec<f32>,
        pub v_limits: [f64; 6],
        pub shape: [usize; 3],
    }

    pub fn parse_hermite_state(bytes: &[u8]) -> HermiteState {
        fn take<const N: usize>(bytes: &[u8], i: &mut usize) -> [u8; N] {
            let out: [u8; N] = bytes[*i..*i + N].try_into().unwrap();
            *i += N;
            out
        }
        let mut i = 0usize;

        let n_hermite_harmonic = i32::from_le_bytes(take::<4>(bytes, &mut i));
        let vth = f32::from_le_bytes(take::<4>(bytes, &mut i));

        let u = [
            f32::from_le_bytes(take::<4>(bytes, &mut i)),
            f32::from_le_bytes(take::<4>(bytes, &mut i)),
            f32::from_le_bytes(take::<4>(bytes, &mut i)),
        ];

        let size = u64::from_le_bytes(take::<8>(bytes, &mut i)) as usize;

        let mut spectrum = Vec::with_capacity(size);
        for _ in 0..size {
            spectrum.push(f32::from_le_bytes(take::<4>(bytes, &mut i)));
        }

        let mut v_limits = [0.0f64; 6];
        for k in 0..6 {
            v_limits[k] = f64::from_le_bytes(take::<8>(bytes, &mut i));
        }

        let shape = [
            usize::from_le_bytes(take::<8>(bytes, &mut i)),
            usize::from_le_bytes(take::<8>(bytes, &mut i)),
            usize::from_le_bytes(take::<8>(bytes, &mut i)),
        ];
        assert!(i == bytes.len(), "trailing bytes: {}", bytes.len() - i);
        HermiteState {
            n_hermite_harmonic,
            vth,
            u,
            spectrum,
            v_limits,
            shape,
        }
    }

    fn factorial_f64(n: usize) -> f64 {
        (1..=n).fold(1.0f64, |acc, k| acc * (k as f64))
    }

    fn linspace_inclusive(a: f64, b: f64, n: usize) -> Vec<f64> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![a];
        }
        let step = (b - a) / ((n - 1) as f64);
        (0..n).map(|i| a + step * (i as f64)).collect()
    }

    fn get_hermite_axis(
        shape: [usize; 3],
        v_limits: [f64; 6],
        order: usize,
        vth: f32,
        u: [f32; 3],
        axis: usize,
    ) -> Vec<f32> {
        let npts = shape[axis];
        let vmin = v_limits[axis] as f32;
        let vmax = v_limits[axis + 3] as f32;
        let u_ax = u[axis];

        let mut herm = vec![0f32; order * npts];
        if order == 0 || npts == 0 {
            return herm;
        }

        let inv_sqrt_vth_sqrt_pi = (vth * std::f32::consts::PI.sqrt()).sqrt().recip();

        for i in 0..npts {
            let v = if npts > 1 {
                vmin + (vmax - vmin) * (i as f32 / (npts - 1) as f32)
            } else {
                vmin
            };
            let x = (v - u_ax) / vth;
            let mut h_prev = 0.0f32;
            let mut h_curr = inv_sqrt_vth_sqrt_pi;
            herm[0 * npts + i] = h_curr;

            for n in 1..order {
                let n_f = n as f32;
                let h_next =
                    (x * 2.0f32.sqrt() * h_curr - (n_f - 1.0).sqrt() * h_prev) / n_f.sqrt();
                h_prev = h_curr;
                h_curr = h_next;
                herm[n * npts + i] = h_curr;
            }
        }
        herm
    }

    pub fn reconstruct_vdf<T>(state: &HermiteState) -> Array4<T>
    where
        T: Zero + FromPrimitive + Copy,
    {
        let order = state.n_hermite_harmonic as usize;
        let [nxv, nyv, nzv] = state.shape;
        let mut decompressed_vdf = Array4::<T>::zeros([nxv, nyv, nzv, 1]);
        let hx = get_hermite_axis(state.shape, state.v_limits, order, state.vth, state.u, 0);
        let hy = get_hermite_axis(state.shape, state.v_limits, order, state.vth, state.u, 1);
        let hz = get_hermite_axis(state.shape, state.v_limits, order, state.vth, state.u, 2);
        let o2 = order * order;
        let mut tmp_xy = vec![0f32; order * order];
        let mut tmp_x = vec![0f32; order];

        for vz in 0..nzv {
            for nx in 0..order {
                for ny in 0..order {
                    let base = nx * o2 + ny * order;
                    let mut acc = 0f32;
                    for nz in 0..order {
                        acc += state.spectrum[base + nz] * hz[nz * nzv + vz];
                    }
                    tmp_xy[nx * order + ny] = acc;
                }
            }

            for vy in 0..nyv {
                for nx in 0..order {
                    let mut acc = 0f32;
                    for ny in 0..order {
                        acc += tmp_xy[nx * order + ny] * hy[ny * nyv + vy];
                    }
                    tmp_x[nx] = acc;
                }

                for vx in 0..nxv {
                    let mut acc = 0f32;
                    for nx in 0..order {
                        acc += tmp_x[nx] * hx[nx * nxv + vx];
                    }
                    decompressed_vdf[[vx, vy, vz, 0]] = T::from_f32(acc).expect("failed to cast");
                }
            }
        }
        decompressed_vdf
    }

    #[derive(Debug, Clone)]
    struct OctreeState {
        blocks_to_ignore: Vec<u32>,
        bbox_shape: [usize; 3],
        bbox_lims: [f64; 6],
        read_index: usize,
    }

    fn parse_octree_state(octree_state: &[u8]) -> Option<OctreeState> {
        fn read_f64_le(buf: &[u8], idx: &mut usize) -> Option<f64> {
            let end = *idx + 8;
            let bytes = buf.get(*idx..end)?;
            *idx = end;
            Some(f64::from_le_bytes(bytes.try_into().ok()?))
        }
        fn read_array_f64_6(buf: &[u8], idx: &mut usize) -> Option<[f64; 6]> {
            Some([
                read_f64_le(buf, idx)?,
                read_f64_le(buf, idx)?,
                read_f64_le(buf, idx)?,
                read_f64_le(buf, idx)?,
                read_f64_le(buf, idx)?,
                read_f64_le(buf, idx)?,
            ])
        }
        fn read_usize_le(buf: &[u8], idx: &mut usize) -> Option<usize> {
            let end = *idx + 8;
            let bytes = buf.get(*idx..end)?;
            *idx = end;
            Some(u64::from_le_bytes(bytes.try_into().ok()?) as usize)
        }
        fn read_u32_le(buf: &[u8], idx: &mut usize) -> Option<u32> {
            let end = *idx + 4;
            let bytes = buf.get(*idx..end)?;
            *idx = end;
            Some(u32::from_le_bytes(bytes.try_into().ok()?))
        }
        fn read_array_usize_3(buf: &[u8], idx: &mut usize) -> Option<[usize; 3]> {
            Some([
                read_usize_le(buf, idx)?,
                read_usize_le(buf, idx)?,
                read_usize_le(buf, idx)?,
            ])
        }

        let mut read_index = 0;
        let n_ignored_blocks = read_usize_le(octree_state, &mut read_index)?;
        let mut blocks_to_ignore = Vec::with_capacity(n_ignored_blocks);
        for _ in 0..n_ignored_blocks {
            blocks_to_ignore.push(read_u32_le(octree_state, &mut read_index)?);
        }
        let bbox_shape = read_array_usize_3(octree_state, &mut read_index)?;
        let bbox_lims = read_array_f64_6(octree_state, &mut read_index)?;
        Some(OctreeState {
            blocks_to_ignore,
            bbox_shape,
            bbox_lims,
            read_index,
        })
    }

    pub unsafe fn zfp_decompress_1d_f32(
        bytes: &[u8],
        nx: usize,
        sparse: f64,
    ) -> Result<Vec<f32>, String> {
        use std::ffi::c_void;
        use std::os::raw::{c_double, c_uint};

        if bytes.is_empty() {
            return Err("ERROR: no bytes to decompress".into());
        }
        if nx == 0 {
            return Ok(Vec::new());
        }
        if nx > (u32::MAX as usize) {
            return Err("ERROR: nx too large ".into());
        }

        #[repr(C)]
        struct Bitstream {
            _private: [u8; 0],
        }
        #[repr(C)]
        struct ZfpStream {
            _private: [u8; 0],
        }
        #[repr(C)]
        struct ZfpField {
            _private: [u8; 0],
        }

        #[repr(i32)]
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone)]
        enum zfp_type {
            zfp_type_none = 0,
            zfp_type_int32 = 1,
            zfp_type_int64 = 2,
            zfp_type_float = 3,
            zfp_type_double = 4,
        }

        #[link(name = "zfp")]
        unsafe extern "C" {
            fn stream_open(buffer: *mut c_void, bytes: usize) -> *mut Bitstream;
            fn stream_close(stream: *mut Bitstream);

            fn zfp_stream_open(stream: *mut Bitstream) -> *mut ZfpStream;
            fn zfp_stream_close(stream: *mut ZfpStream);
            fn zfp_stream_rewind(stream: *mut ZfpStream);
            fn zfp_stream_set_accuracy(stream: *mut ZfpStream, tolerance: c_double) -> c_double;

            fn zfp_field_1d(pointer: *mut c_void, ty: zfp_type, nx: c_uint) -> *mut ZfpField;
            fn zfp_field_free(field: *mut ZfpField);

            fn zfp_decompress(stream: *mut ZfpStream, field: *mut ZfpField) -> usize;
        }

        struct BsGuard(*mut Bitstream);
        impl Drop for BsGuard {
            fn drop(&mut self) {
                unsafe {
                    if !self.0.is_null() {
                        stream_close(self.0);
                    }
                }
            }
        }

        struct ZfpGuard(*mut ZfpStream);
        impl Drop for ZfpGuard {
            fn drop(&mut self) {
                unsafe {
                    if !self.0.is_null() {
                        zfp_stream_close(self.0);
                    }
                }
            }
        }

        struct FieldGuard(*mut ZfpField);
        impl Drop for FieldGuard {
            fn drop(&mut self) {
                unsafe {
                    if !self.0.is_null() {
                        zfp_field_free(self.0);
                    }
                }
            }
        }

        let mut out = vec![0f32; nx];

        unsafe {
            let bs = stream_open(bytes.as_ptr() as *mut c_void, bytes.len());
            if bs.is_null() {
                return Err("ERROR: failed to open ZFP bitstream".into());
            }
            let _bs_guard = BsGuard(bs);

            let zfp = zfp_stream_open(bs);
            if zfp.is_null() {
                return Err("ERROR: failed to open ZFP stream".into());
            }
            let _zfp_guard = ZfpGuard(zfp);
            let _actual_tol: c_double = zfp_stream_set_accuracy(zfp, sparse as c_double);
            zfp_stream_rewind(zfp);
            let field = zfp_field_1d(
                out.as_mut_ptr() as *mut c_void,
                zfp_type::zfp_type_float,
                nx as c_uint,
            );
            if field.is_null() {
                return Err("ERROR: ZFP failed to init field metadata".into());
            }
            let _field_guard = FieldGuard(field);

            let written = zfp_decompress(zfp, field);
            if written == 0 {
                return Err("ERROR: ZFP decompression failed ".into());
            }
        }

        Ok(out)
    }

    //Galloping on top of cellids hints
    #[inline(always)]
    fn find_near_with_hint(cell_ids: &[u64], target: u64, hint: usize) -> Option<usize> {
        let n = cell_ids.len();
        if n == 0 {
            return None;
        }
        let h = hint.min(n - 1);
        if cell_ids[h] == target {
            return Some(h);
        }

        let mut prev_w = 0usize;
        let mut w: usize = 128;

        loop {
            let left_prev = h.saturating_sub(prev_w);
            let right_prev = (h + prev_w).min(n - 1);
            let left_now = h.saturating_sub(w);
            let right_now = (h + w).min(n - 1);
            let mut i = left_now;
            while i < left_prev {
                if cell_ids[i] == target {
                    return Some(i);
                }
                i += 1;
            }
            let mut j = right_prev.saturating_add(1);
            while j <= right_now {
                if cell_ids[j] == target {
                    return Some(j);
                }
                j += 1;
            }
            if left_now == 0 && right_now == n - 1 {
                break;
            }
            prev_w = w;
            if w >= n {
                break;
            }
            w = w.saturating_mul(2);
        }
        None
    }

    //Instead of native .position() cause here the compiler seems to use some ILP nicely
    #[inline(always)]
    fn find_cell_index_unrolled(cell_ids: &[u64], target: u64) -> Option<usize> {
        let n = cell_ids.len();
        let mut i = 0;
        while i + 8 <= n {
            let a = cell_ids[i];
            let b = cell_ids[i + 1];
            let c = cell_ids[i + 2];
            let d = cell_ids[i + 3];
            let e = cell_ids[i + 4];
            let f = cell_ids[i + 5];
            let g = cell_ids[i + 6];
            let h = cell_ids[i + 7];
            if a == target {
                return Some(i);
            }
            if b == target {
                return Some(i + 1);
            }
            if c == target {
                return Some(i + 2);
            }
            if d == target {
                return Some(i + 3);
            }
            if e == target {
                return Some(i + 4);
            }
            if f == target {
                return Some(i + 5);
            }
            if g == target {
                return Some(i + 6);
            }
            if h == target {
                return Some(i + 7);
            }
            i += 8;
        }
        while i < n {
            if cell_ids[i] == target {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    pub fn read_tag(
        xml: &str,
        tag: &str,
        mesh: Option<&str>,
        name: Option<&str>,
    ) -> Option<Variable> {
        let re_attr = RE_ATTR.get_or_init(|| Regex::new(r#"(\w+)\s*=\s*"([^"]*)""#).unwrap());
        thread_local! {
            static CACHE: RefCell<HashMap<String, (Regex, Regex)>> = RefCell::new(HashMap::with_capacity(16));
        }
        let (re_normal, re_self) = CACHE.with(|cache_cell| {
            let mut cache = cache_cell.borrow_mut();
            if let Some(pair) = cache.get(tag) {
                return pair.clone();
            }

            let t = regex::escape(tag);
            let normal = Regex::new(&format!(r#"(?s)<{t}\b([^>]*)>([^<]*)</{t}>"#, t = t)).unwrap();
            let self_closing = Regex::new(&format!(r#"(?s)<{t}\b([^>]*)/>"#, t = t)).unwrap();

            let pair = (normal, self_closing);
            cache.insert(tag.to_string(), pair.clone());
            pair
        });
        let parse_match = |attrs_str: &str, inner_text: Option<&str>| -> Option<Variable> {
            let mut attrs: HashMap<&str, &str> = HashMap::with_capacity(8);
            for cap in re_attr.captures_iter(attrs_str) {
                attrs.insert(
                    cap.get(1).map(|m| m.as_str()).unwrap_or(""),
                    cap.get(2).map(|m| m.as_str()).unwrap_or(""),
                );
            }
            if mesh.is_some() && attrs.get("mesh").copied() != mesh {
                return None;
            }
            if name.is_some() && attrs.get("name").copied() != name {
                return None;
            }
            Some(Variable {
                arraysize: attrs.get("arraysize").map(|s| s.to_string()),
                datasize: attrs.get("datasize").map(|s| s.to_string()),
                datatype: attrs.get("datatype").map(|s| s.to_string()),
                mesh: attrs.get("mesh").map(|s| s.to_string()),
                name: attrs
                    .get("name")
                    .map(|s| s.to_string())
                    .or_else(|| Some(tag.to_string())),
                vectorsize: attrs.get("vectorsize").map(|s| s.to_string()),
                compression: attrs.get("compression").map(|s| s.to_string()),
                max_refinement_level: attrs.get("max_refinement_level").map(|s| s.to_string()),
                unit: attrs.get("unit").map(|s| s.to_string()),
                unit_conversion: attrs.get("unitConversion").map(|s| s.to_string()),
                unit_latex: attrs.get("unitLaTeX").map(|s| s.to_string()),
                variable_latex: attrs.get("variableLaTeX").map(|s| s.to_string()),
                offset: inner_text.map(|s| s.trim().to_string()),
            })
        };

        for caps in re_normal.captures_iter(xml) {
            if let Some(v) = parse_match(
                caps.get(1).unwrap().as_str(),
                caps.get(2).map(|m| m.as_str()),
            ) {
                return Some(v);
            }
        }

        for caps in re_self.captures_iter(xml) {
            if let Some(v) = parse_match(caps.get(1).unwrap().as_str(), None) {
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
            0..=3 => {}
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

    pub fn remesh_trilinear<T>(
        src: &Array4<T>,
        src_extent: (f64, f64, f64, f64, f64, f64),
        dst: &mut Array4<T>,
        dst_extent: (f64, f64, f64, f64, f64, f64),
    ) where
        T: Copy + Zero + Num + NumCast + Default,
    {
        fn trilinear_sample<T>(src: &Array4<T>, ux: f64, uy: f64, uz: f64, chan: usize) -> f64
        where
            T: Num + NumCast + Copy,
        {
            let (sx, sy, sz, _sc) = src.dim();
            let in_bounds = |i: isize, n: usize| i >= 0 && (i as usize) < n;
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

            let f = |i, j, k| -> f64 { NumCast::from(src[(i, j, k, chan)]).unwrap_or(0.0) };
            let c000 = f(ix0 as usize, iy0 as usize, iz0 as usize);
            let c100 = f((ix0 + 1) as usize, iy0 as usize, iz0 as usize);
            let c010 = f(ix0 as usize, (iy0 + 1) as usize, iz0 as usize);
            let c110 = f((ix0 + 1) as usize, (iy0 + 1) as usize, iz0 as usize);
            let c001 = f(ix0 as usize, iy0 as usize, (iz0 + 1) as usize);
            let c101 = f((ix0 + 1) as usize, iy0 as usize, (iz0 + 1) as usize);
            let c011 = f(ix0 as usize, (iy0 + 1) as usize, (iz0 + 1) as usize);
            let c111 = f((ix0 + 1) as usize, (iy0 + 1) as usize, (iz0 + 1) as usize);

            let c00 = c000 * (1.0 - fx) + c100 * fx;
            let c10 = c010 * (1.0 - fx) + c110 * fx;
            let c01 = c001 * (1.0 - fx) + c101 * fx;
            let c11 = c011 * (1.0 - fx) + c111 * fx;

            let c0 = c00 * (1.0 - fy) + c10 * fy;
            let c1 = c01 * (1.0 - fy) + c11 * fy;

            c0 * (1.0 - fz) + c1 * fz
        }

        let (sx, sy, sz, sc) = src.dim();
        let (tx, ty, tz, tc) = dst.dim();
        debug_assert!(sc == tc, "ERROR: different vectorsizes found");

        let (sxmin, symin, szmin, sxmax, symax, szmax) = src_extent;
        let (txmin, tymin, tzmin, txmax, tymax, tzmax) = dst_extent;

        let sdx = (sxmax - sxmin) / (sx as f64);
        let sdy = (symax - symin) / (sy as f64);
        let sdz = (szmax - szmin) / (sz as f64);

        let tdx = (txmax - txmin) / (tx as f64);
        let tdy = (tymax - tymin) / (ty as f64);
        let tdz = (tzmax - tzmin) / (tz as f64);

        let to_src_u = |x_t: f64, xmin_s: f64, sd: f64| -> f64 { (x_t - xmin_s) / sd - 0.5 };
        dst.fill(T::zero());

        let c_use = sc.min(tc);

        for ic in 0..tc {
            for iz in 0..tz {
                let zc = tzmin + (iz as f64 + 0.5) * tdz;
                let uz = to_src_u(zc, szmin, sdz);

                for iy in 0..ty {
                    let yc = tymin + (iy as f64 + 0.5) * tdy;
                    let uy = to_src_u(yc, symin, sdy);
                    for ix in 0..tx {
                        let xc = txmin + (ix as f64 + 0.5) * tdx;
                        let ux = to_src_u(xc, sxmin, sdx);

                        let v = trilinear_sample(src, ux, uy, uz, ic);
                        let v_t: T = NumCast::from(v).unwrap_or_else(T::zero);

                        for c in 0..c_use {
                            dst[(ix, iy, iz, c)] = v_t;
                        }
                    }
                }
            }
        }
    }

    pub fn remesh_conservative<T>(
        src: &Array4<T>,
        src_extent: (f64, f64, f64, f64, f64, f64),
        dst: &mut Array4<T>,
        dst_extent: (f64, f64, f64, f64, f64, f64),
    ) where
        T: Copy + Zero + Num + NumCast + Default,
    {
        fn overlap_1d(a_min: f64, a_max: f64, b_min: f64, b_max: f64) -> f64 {
            let lo = a_min.max(b_min);
            let hi = a_max.min(b_max);
            (hi - lo).max(0.0)
        }
        let (sx, sy, sz, sc) = src.dim();
        let (tx, ty, tz, tc) = dst.dim();
        debug_assert!(sc == tc, "ERROR: different vectorsizes found");

        let (sxmin, symin, szmin, sxmax, symax, szmax) = src_extent;
        let (txmin, tymin, tzmin, txmax, tymax, tzmax) = dst_extent;

        let sdx = (sxmax - sxmin) / sx as f64;
        let sdy = (symax - symin) / sy as f64;
        let sdz = (szmax - szmin) / sz as f64;

        let tdx = (txmax - txmin) / tx as f64;
        let tdy = (tymax - tymin) / ty as f64;
        let tdz = (tzmax - tzmin) / tz as f64;

        let dst_cell_vol = tdx * tdy * tdz;

        dst.fill(T::zero());

        for ic in 0..tc {
            for iz in 0..tz {
                let tz0 = tzmin + iz as f64 * tdz;
                let tz1 = tz0 + tdz;

                let sz_start = (((tz0 - szmin) / sdz).floor() as isize - 1).max(0) as usize;
                let sz_end = (((tz1 - szmin) / sdz).ceil() as usize + 1).min(sz);

                for iy in 0..ty {
                    let ty0 = tymin + iy as f64 * tdy;
                    let ty1 = ty0 + tdy;
                    let sy_start = (((ty0 - symin) / sdy).floor() as isize - 1).max(0) as usize;
                    let sy_end = (((ty1 - symin) / sdy).ceil() as usize + 1).min(sy);

                    for ix in 0..tx {
                        let tx0 = txmin + ix as f64 * tdx;
                        let tx1 = tx0 + tdx;
                        let sx_start = (((tx0 - sxmin) / sdx).floor() as isize - 1).max(0) as usize;
                        let sx_end = (((tx1 - sxmin) / sdx).ceil() as usize + 1).min(sx);

                        let mut accum = 0.0f64;

                        for kz in sz_start..sz_end {
                            let sz0 = szmin + kz as f64 * sdz;
                            let sz1 = sz0 + sdz;
                            let wz = overlap_1d(sz0, sz1, tz0, tz1);

                            if wz == 0.0 {
                                continue;
                            }

                            for ky in sy_start..sy_end {
                                let sy0 = symin + ky as f64 * sdy;
                                let sy1 = sy0 + sdy;
                                let wy = overlap_1d(sy0, sy1, ty0, ty1);
                                if wy == 0.0 {
                                    continue;
                                }

                                for kx in sx_start..sx_end {
                                    let sx0 = sxmin + kx as f64 * sdx;
                                    let sx1 = sx0 + sdx;
                                    let wx = overlap_1d(sx0, sx1, tx0, tx1);
                                    if wx == 0.0 {
                                        continue;
                                    }

                                    // overlap volume
                                    let w = wx * wy * wz;
                                    let val: f64 =
                                        NumCast::from(src[(kx, ky, kz, ic)]).unwrap_or(0.0);
                                    accum += val * w;
                                }
                            }
                        }

                        let avg = if dst_cell_vol > 0.0 {
                            accum / dst_cell_vol
                        } else {
                            0.0
                        };
                        dst[(ix, iy, iz, ic)] = NumCast::from(avg).unwrap_or_else(T::zero);
                    }
                }
            }
        }
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

    pub fn vg_variable_to_fg<T: bytemuck::Pod + Copy + Default>(
        cell_ids: &[u64],
        vg_rows: &[T],
        vecsz: usize,
        x0: usize,
        y0: usize,
        z0: usize,
        lmax: u32,
    ) -> ndarray::Array4<T> {
        let (fx, fy, fz) = (x0 << lmax, y0 << lmax, z0 << lmax);
        let mut fg = Array4::<T>::default((fx, fy, fz, vecsz));
        assert_eq!(vg_rows.len(), cell_ids.len() * vecsz);

        for (idx, &cid) in cell_ids.iter().enumerate() {
            let lvl = amr_level(cid, x0, y0, z0, lmax).expect("bad CellID/levels");
            let (sx, sy, sz) = cid2fineijk(cid, lvl, lmax, x0, y0, z0).unwrap();
            let scale = 1usize << ((lmax - lvl) as usize);
            let row = ArrayView1::from(&vg_rows[idx * vecsz..(idx + 1) * vecsz]);
            let mut block = fg.slice_mut(s![sx..sx + scale, sy..sy + scale, sz..sz + scale, ..]);
            let row_b = row.broadcast(block.raw_dim()).unwrap();
            block.assign(&row_b);
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
        #[serde(rename = "compression")]
        pub compression: Option<String>,
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

    //  def get_sparsity_for_cid(self,cellid,pop):
    // if self.check_variable('MinValue') == True:
    //    val = self.read_variable('MinValue',cellid)
    //    logging.info("Found a vlsv file MinValue of "+str(val))
    // elif self.check_variable(pop+"/EffectiveSparsityThreshold") == True:
    //    val = self.read_variable(pop+"/EffectiveSparsityThreshold",cellid)
    //    logging.info("Found a vlsv file value "+pop+"/EffectiveSparsityThreshold"+" of "+str(val))
    // elif self.check_variable(pop+"/vg_effectivesparsitythreshold") == True:
    //    val = self.read_variable(pop+"/vg_effectivesparsitythreshold",cellid)
    //    logging.info("Found a vlsv file value "+pop+"/vg_effectivesparsitythreshold"+" of "+str(val))
    // else:
    //    logging.warning("Unable to find a MinValue or EffectiveSparsityThreshold value from the .vlsv file.")
    //    logging.info("Using a default value of 1.e-16. Override with val=value.")
    //    val = 1.e-16
    // return val

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

        #[serde(rename = "BYTESPERCELL")]
        pub bytespercell: Option<Vec<Variable>>,

        #[serde(rename = "BLOCKVARIABLE")]
        pub blockvariable: Option<Vec<Variable>>,

        #[serde(rename = "CELLSWITHBLOCKS")]
        pub cellswithblocks: Option<Vec<Variable>>,

        #[serde(rename = "MLP_BYTES_PER_RANK")]
        pub mlp_bytes_per_rank: Option<Vec<Variable>>,

        #[serde(rename = "MLP_ARCH")]
        pub mlp_arch: Option<Vec<Variable>>,

        #[serde(rename = "MLP_CLUSTERS_PER_RANK")]
        pub mlp_clusters_per_rank: Option<Vec<Variable>>,

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
                    .map_err(|e| format!("Invalid offset: {e}"))?,

                arraysize: var
                    .arraysize
                    .as_deref()
                    .ok_or("Missing arraysize")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid arraysize: {e}"))?,

                vectorsize: var
                    .vectorsize
                    .as_deref()
                    .unwrap_or("1")
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid vectorsize: {e}"))?,

                datasize: var
                    .datasize
                    .as_deref()
                    .ok_or("Missing datasize")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid datasize: {e}"))?,

                datatype: var
                    .datatype
                    .as_deref()
                    .ok_or("Missing datatype")?
                    .parse::<DataType>()
                    .map_err(|e| e.to_string())?,
                grid: g,
                compression: Some(
                    var.compression
                        .as_deref()
                        .unwrap_or("NONE")
                        .parse::<CompressionMethod>()
                        .map_err(|e| format!("Invalid datasize: {e}"))?,
                ),
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
                    .map_err(|e| format!("Invalid offset: {e}"))?,

                arraysize: var
                    .arraysize
                    .as_ref()
                    .ok_or("Missing arraysize")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid arraysize: {e}"))?,

                vectorsize: var
                    .vectorsize
                    .as_ref()
                    .unwrap_or(&"1".to_string())
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid vectorsize: {e}"))?,

                datasize: var
                    .datasize
                    .as_ref()
                    .ok_or("Missing datasize")?
                    .parse::<usize>()
                    .map_err(|e| format!("Invalid datasize: {e}"))?,

                datatype: var
                    .datatype
                    .as_deref()
                    .ok_or("Missing datatype")?
                    .parse::<DataType>()
                    .map_err(|e| e.to_string())?,
                grid: g,
                compression: Some(
                    var.compression
                        .as_deref()
                        .unwrap_or("NONE")
                        .parse::<CompressionMethod>()
                        .map_err(|e| format!("Invalid datasize: {e}"))?,
                ),
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
                _ => Ok(VlasiatorGrid::VMESH),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum DataType {
        Float,
        Int,
        Uint,
        U8,
    }

    impl std::str::FromStr for DataType {
        type Err = String;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "float" => Ok(DataType::Float),
                "int" => Ok(DataType::Int),
                "uint" => Ok(DataType::Uint),
                "u8" => Ok(DataType::U8),
                other => Err(format!("Unknown datatype: {other}")),
            }
        }
    }

    #[repr(C)]
    #[derive(Default, Clone, Copy, Debug, Pod, Zeroable)]
    pub struct Header {
        pub key: usize,
        pub total_size: usize,
        pub rows: usize,
        pub cols: usize,
        pub n_weights: usize,
        pub type_size: usize,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub enum CompressionMethod {
        NONE,
        ZFP,
        OCTREE,
        MLP,
        MLPMULTI,
        HERMITE,
    }

    impl std::str::FromStr for CompressionMethod {
        type Err = String;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "" | "NONE" => Ok(CompressionMethod::NONE),
                "ZFP" => Ok(CompressionMethod::ZFP),
                "OCTREE" => Ok(CompressionMethod::OCTREE),
                "HERMITE" => Ok(CompressionMethod::HERMITE),
                "MLP" => Ok(CompressionMethod::MLP),
                "MLPMULTI" => Ok(CompressionMethod::MLPMULTI),
                other => Err(format!("Unknown compression scheme: {other}")),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct VlsvDataset {
        pub offset: usize,
        pub arraysize: usize,
        pub vectorsize: usize,
        pub datasize: usize,
        pub datatype: DataType,
        grid: Option<VlasiatorGrid>,
        pub compression: Option<CompressionMethod>,
    }

    pub trait TypeTag {
        fn type_name() -> &'static str;
        fn data_type() -> DataType;
    }

    impl TypeTag for f32 {
        fn type_name() -> &'static str {
            "float"
        }
        fn data_type() -> DataType {
            DataType::Float
        }
    }
    impl TypeTag for f64 {
        fn type_name() -> &'static str {
            "float"
        }
        fn data_type() -> DataType {
            DataType::Float
        }
    }
    impl TypeTag for u32 {
        fn type_name() -> &'static str {
            "uint"
        }
        fn data_type() -> DataType {
            DataType::Uint
        }
    }
    impl TypeTag for i32 {
        fn type_name() -> &'static str {
            "int"
        }
        fn data_type() -> DataType {
            DataType::Int
        }
    }
    impl TypeTag for u64 {
        fn type_name() -> &'static str {
            "uint"
        }
        fn data_type() -> DataType {
            DataType::Uint
        }
    }
    impl TypeTag for i64 {
        fn type_name() -> &'static str {
            "int"
        }
        fn data_type() -> DataType {
            DataType::Int
        }
    }
    impl TypeTag for u8 {
        fn type_name() -> &'static str {
            "u8"
        }
        fn data_type() -> DataType {
            DataType::U8
        }
    }
    impl TypeTag for usize {
        fn type_name() -> &'static str {
            "uint"
        }
        fn data_type() -> DataType {
            DataType::Uint
        }
    }
}

#[cfg(feature = "vlsv_ptr")]
pub mod mod_vlsv_tracing {
    use bytemuck::Pod;
    use ndarray::Array4;
    use ndarray::ArrayView1;
    use ndarray::s;
    use num_traits::Float;
    use num_traits::{Num, NumCast};
    use rand::Rng;
    use rand_distr::Normal;
    use std::f64::consts::PI;
    use std::io::Write;
    extern crate libc;
    use super::mod_vlsv_reader::*;
    const DELTA: f64 = 1.0e3;

    pub mod physical_constants {
        pub mod f64 {
            pub const C: f64 = 299792458.0; // m/s
            pub const C2: f64 = C * C; // m/s
            pub const PROTON_MASS: f64 = 1.67262192e-27; // kg
            pub const PROTON_CHARGE: f64 = 1.602e-19; // C
            pub const ELECTRON_MASS: f64 = 9.1093837e-31; // kg
            pub const ELECTRON_CHARGE: f64 = -PROTON_CHARGE; // C
            pub const JOULE_TO_KEV: f64 = 6.242e+15;
            pub const JOULE_TO_EV: f64 = 6.242e+18;
            pub const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0_f64;
            pub const RAD_TO_DEG: f64 = 180.0_f64 / std::f64::consts::PI;
            pub const EV_TO_JOULE: f64 = 1_f64 / JOULE_TO_EV;
            pub const EARTH_RE: f64 = 6378137.0;
            pub const OUTER_LIM: f64 = 30.0 * EARTH_RE;
            pub const INNER_LIM: f64 = 5.0 * EARTH_RE;
            pub const TOL: f64 = 5e-5;
            pub const PRECIPITATION_RE: f64 = 1.2 * EARTH_RE;
            pub const MAX_STEPS: usize = 10000000;
            pub const DIPOLE_MOMENT: f64 = 8.0e15;
        }
        pub mod f32 {
            pub const C: f32 = 299792458.0; // m/s
            pub const C2: f32 = C * C; // m/s
            pub const PROTON_MASS: f32 = 1.67262192e-27; // kg
            pub const PROTON_CHARGE: f32 = 1.602e-19; // C
            pub const ELECTRON_MASS: f32 = 9.1093837e-31; // kg
            pub const ELECTRON_CHARGE: f32 = -PROTON_CHARGE; // C
            pub const JOULE_TO_KEV: f32 = 6.242e+15;
            pub const JOULE_TO_EV: f32 = 6.242e+18;
            pub const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0_f32;
            pub const RAD_TO_DEG: f32 = 180.0_f32 / std::f32::consts::PI;
            pub const EV_TO_JOULE: f32 = 1_f32 / JOULE_TO_EV;
            pub const EARTH_RE: f32 = 6378137.0;
            pub const OUTER_LIM: f32 = 30.0 * EARTH_RE;
            pub const INNER_LIM: f32 = 5.0 * EARTH_RE;
            pub const TOL: f32 = 5e-5;
            pub const PRECIPITATION_RE: f32 = 1.2 * EARTH_RE;
            pub const MAX_STEPS: usize = 10000000;
            pub const DIPOLE_MOMENT: f32 = 8.0e15;
        }
    }

    pub trait PtrTrait:
        Float
        + Pod
        + Send
        + Sync
        + Sized
        + std::fmt::Debug
        + std::fmt::Display
        + num_traits::ToBytes
        + std::iter::Sum
        + TypeTag
        + std::default::Default
        + Num
        + NumCast
    {
    }

    impl<T> PtrTrait for T where
        T: Float
            + Pod
            + Send
            + Sync
            + Sized
            + std::fmt::Debug
            + std::fmt::Display
            + num_traits::ToBytes
            + std::default::Default
            + std::iter::Sum
            + TypeTag
            + Num
            + NumCast
    {
    }

    pub trait Field<T: PtrTrait> {
        fn get_fields_at(&self, time: T, x: T, y: T, z: T) -> Option<[T; 6]>;
        fn ds(&self) -> T;
    }

    pub struct DipoleField<T: PtrTrait> {
        pub moment: T,
    }

    impl<T: PtrTrait> DipoleField<T> {
        pub fn new(moment: T) -> Self {
            DipoleField { moment }
        }
    }

    pub struct UniformField<T: PtrTrait> {
        pub mag: T,
        pub axis: i32,
    }

    impl<T: PtrTrait> UniformField<T> {
        pub fn new(mag: T, axis: i32) -> Self {
            UniformField { mag, axis }
        }
    }

    pub struct VlsvStaticField<T: PtrTrait> {
        b: Array4<T>,
        e: Array4<T>,
        extents: [T; 6],
        periodic: [bool; 3],
        ds: T,
    }

    pub struct VlsvDynamicField<T: PtrTrait> {
        //time,field
        timeline: Vec<(T, VlsvStaticField<T>)>,
        ds: T,
    }

    impl<T: PtrTrait> VlsvStaticField<T> {
        pub fn new(filename: &String, periodic: [bool; 3]) -> Self {
            let f = VlsvFile::new(&filename).unwrap();
            let extents: [T; 6] = [
                T::from(f.read_scalar_parameter("xmin").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("ymin").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("zmin").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("xmax").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("ymax").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("zmax").unwrap()).unwrap(),
            ];
            let b = f.read_variable::<T>("vg_b_vol", None).unwrap();
            let e = f.read_variable::<T>("vg_e_vol", None).unwrap();
            let ds = (extents[3] - extents[0]) / T::from(b.dim().0).unwrap();
            VlsvStaticField {
                b,
                e,
                extents,
                periodic,
                ds,
            }
        }

        fn real2mesh(&self, x: T, y: T, z: T, periodic: [bool; 3]) -> Option<([usize; 3], [T; 3])> {
            let dims = self.e.dim();
            let coords = [x, y, z];
            let grid_dims = [dims.0, dims.1, dims.2];

            let mut indices = [0usize; 3];
            let mut weights = [T::zero(); 3];

            for i in 0..3 {
                let dim = grid_dims[i];
                let size = T::from(dim).unwrap() * self.ds;
                let mut val = coords[i] - self.extents[i];

                if periodic[i] {
                    val = ((val % size) + size) % size;
                } else {
                    if val < T::zero() || val >= size - self.ds {
                        return None;
                    }
                }

                let norm = val / self.ds;
                let mut idx = norm.floor().to_usize().unwrap();
                if periodic[i] {
                    if idx >= dim {
                        idx = 0;
                    }
                    indices[i] = idx;
                } else {
                    indices[i] = idx.min(dim - 2);
                }
                weights[i] = norm - T::from(indices[i]).unwrap();
            }

            Some((indices, weights))
        }

        // https://en.wikipedia.org/wiki/Trilinear_interpolation#Formulation
        fn trilerp(&self, grid_point: [usize; 3], weights: [T; 3], field: &Array4<T>) -> [T; 3] {
            let [x0, y0, z0] = grid_point;
            let [xd, yd, zd] = weights;
            let (nx, ny, nz, _) = field.dim();

            fn lerp<T: Float>(a: &ArrayView1<T>, b: &ArrayView1<T>, t: T) -> [T; 3] {
                [
                    a[0] * (T::one() - t) + b[0] * t,
                    a[1] * (T::one() - t) + b[1] * t,
                    a[2] * (T::one() - t) + b[2] * t,
                ]
            }

            if nx > 1 && ny > 1 && nz > 1 {
                let c000 = field.slice(s![x0, y0, z0, ..]);
                let c001 = field.slice(s![x0, y0, z0 + 1, ..]);
                let c010 = field.slice(s![x0, y0 + 1, z0, ..]);
                let c011 = field.slice(s![x0, y0 + 1, z0 + 1, ..]);
                let c100 = field.slice(s![x0 + 1, y0, z0, ..]);
                let c101 = field.slice(s![x0 + 1, y0, z0 + 1, ..]);
                let c110 = field.slice(s![x0 + 1, y0 + 1, z0, ..]);
                let c111 = field.slice(s![x0 + 1, y0 + 1, z0 + 1, ..]);

                let c00 = lerp(&c000, &c100, xd);
                let c01 = lerp(&c001, &c101, xd);
                let c10 = lerp(&c010, &c110, xd);
                let c11 = lerp(&c011, &c111, xd);

                let c0 = [
                    c00[0] * (T::one() - yd) + c10[0] * yd,
                    c00[1] * (T::one() - yd) + c10[1] * yd,
                    c00[2] * (T::one() - yd) + c10[2] * yd,
                ];
                let c1 = [
                    c01[0] * (T::one() - yd) + c11[0] * yd,
                    c01[1] * (T::one() - yd) + c11[1] * yd,
                    c01[2] * (T::one() - yd) + c11[2] * yd,
                ];

                return [
                    c0[0] * (T::one() - zd) + c1[0] * zd,
                    c0[1] * (T::one() - zd) + c1[1] * zd,
                    c0[2] * (T::one() - zd) + c1[2] * zd,
                ];
            }

            if nx > 1 && ny > 1 && nz == 1 {
                let c00 = field.slice(s![x0, y0, 0, ..]);
                let c10 = field.slice(s![x0 + 1, y0, 0, ..]);
                let c01 = field.slice(s![x0, y0 + 1, 0, ..]);
                let c11 = field.slice(s![x0 + 1, y0 + 1, 0, ..]);

                let c0 = lerp(&c00, &c10, xd);
                let c1 = lerp(&c01, &c11, xd);

                return [
                    c0[0] * (T::one() - yd) + c1[0] * yd,
                    c0[1] * (T::one() - yd) + c1[1] * yd,
                    c0[2] * (T::one() - yd) + c1[2] * yd,
                ];
            }

            if nx > 1 && ny == 1 && nz == 1 {
                let c0 = field.slice(s![x0, 0, 0, ..]);
                let c1 = field.slice(s![x0 + 1, 0, 0, ..]);
                return lerp(&c0, &c1, xd);
            }

            let c: ArrayView1<T> = field.slice(s![0, 0, 0, ..]);
            [c[0usize], c[1usize], c[2usize]]
        }
    }

    impl<T: PtrTrait> VlsvDynamicField<T> {
        pub fn new(dir: &str, periodic: [bool; 3]) -> Self {
            use indicatif::{ProgressBar, ProgressStyle};
            use rayon::prelude::*;
            use std::fs;
            let mut files: Vec<String> = fs::read_dir(dir)
                .unwrap()
                .filter_map(|entry| {
                    let path = entry.unwrap().path();
                    if path.extension().map(|e| e == "vlsv").unwrap_or(false) {
                        Some(path.to_string_lossy().to_string())
                    } else {
                        None
                    }
                })
                .collect();
            files.sort();

            let num_files = files.len();
            let num_threads = rayon::current_num_threads();
            println!("Loading {num_files} VLSV files using {num_threads} threads...");

            let pb = ProgressBar::new(num_files as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("##-"),
            );

            let mut timeline: Vec<(T, VlsvStaticField<T>)> = files
                .par_iter()
                .map(|filename| {
                    let f = VlsvFile::new(&filename).unwrap();
                    let t: T = T::from(f.read_scalar_parameter("time").unwrap()).unwrap();
                    let fields = VlsvStaticField::new(filename, periodic);
                    pb.inc(1);
                    (t, fields)
                })
                .collect();

            pb.finish_with_message("All files loaded.");
            timeline.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let ds = if let Some(first) = timeline.first() {
                first.1.ds()
            } else {
                T::zero()
            };
            Self { timeline, ds }
        }

        pub fn new_partial(dir: &str, periodic: [bool; 3], tmin: T, tmax: T) -> Self {
            use indicatif::{ProgressBar, ProgressStyle};
            use rayon::prelude::*;
            use std::fs;
            let mut files: Vec<String> = fs::read_dir(dir)
                .unwrap()
                .filter_map(|entry| {
                    let path = entry.unwrap().path();
                    if path.extension().map(|e| e == "vlsv").unwrap_or(false) {
                        Some(path.to_string_lossy().to_string())
                    } else {
                        None
                    }
                })
                .collect();
            files.sort();

            let num_files = files.len();
            let num_threads = rayon::current_num_threads();
            println!("Loading {num_files} VLSV files using {num_threads} threads...");

            let pb = ProgressBar::new(num_files as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("##-"),
            );

            let mut timeline: Vec<(T, VlsvStaticField<T>)> = files
                .par_iter()
                .filter_map(|filename| {
                    let f = VlsvFile::new(&filename).unwrap();
                    let time = T::from(f.read_scalar_parameter("time").unwrap()).unwrap();
                    if time < tmin || time > tmax {
                        pb.inc(1);
                        return None;
                    }
                    let fields = VlsvStaticField::new(filename, periodic);
                    pb.inc(1);

                    Some((time, fields))
                })
                .collect();

            pb.finish_with_message("All files loaded.");
            timeline.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let ds = if let Some(first) = timeline.first() {
                first.1.ds()
            } else {
                T::zero()
            };
            Self { timeline, ds }
        }

        pub fn temporal_range(&self) -> (T, T) {
            return (
                self.timeline.first().unwrap().0,
                self.timeline.last().unwrap().0,
            );
        }
    }

    impl<T: PtrTrait> Field<T> for VlsvDynamicField<T> {
        fn get_fields_at(&self, time: T, x: T, y: T, z: T) -> Option<[T; 6]> {
            if self.timeline.is_empty() {
                return None;
            }
            if self.timeline.len() == 1 {
                return self.timeline[0].1.get_fields_at(time, x, y, z);
            }
            let mut i = 0;
            while i + 1 < self.timeline.len() && self.timeline[i + 1].0 <= time {
                i += 1;
            }
            if i + 1 == self.timeline.len() {
                return self.timeline[i].1.get_fields_at(time, x, y, z);
            }
            let (tmin, tmax) = self.temporal_range();
            if time < tmin || time > tmax {
                panic!("Time {time} is outside of directory temporal range {tmin} - {tmax}!");
            }

            let (t0, ref f0) = self.timeline[i];
            let (t1, ref f1) = self.timeline[i + 1];
            let frac = (time - t0) / (t1 - t0);
            let fields0 = f0.get_fields_at(time, x, y, z)?;
            let fields1 = f1.get_fields_at(time, x, y, z)?;
            //Temporal lerp
            Some([
                fields0[0] * (T::one() - frac) + fields1[0] * frac,
                fields0[1] * (T::one() - frac) + fields1[1] * frac,
                fields0[2] * (T::one() - frac) + fields1[2] * frac,
                fields0[3] * (T::one() - frac) + fields1[3] * frac,
                fields0[4] * (T::one() - frac) + fields1[4] * frac,
                fields0[5] * (T::one() - frac) + fields1[5] * frac,
            ])
        }
        fn ds(&self) -> T {
            return self.ds;
        }
    }

    pub fn earth_dipole<T: PtrTrait>(x: T, y: T, z: T) -> [T; 6] {
        let position_mag = (x * x + y * y + z * z).sqrt();
        let m = T::from(-7800e+12).unwrap();
        let mut b = [T::zero(), T::zero(), T::zero()];
        b[0] = (T::from(3.0).unwrap() * m * x * z) / position_mag.powi(5);
        b[1] = (T::from(3.0).unwrap() * m * y * z) / position_mag.powi(5);
        b[2] = (m / position_mag.powi(3))
            * ((T::from(3.0).unwrap() * z * z) / position_mag.powi(2) - T::one());
        [b[0], b[1], b[2], T::zero(), T::zero(), T::zero()]
    }

    impl<T: PtrTrait> Field<T> for DipoleField<T> {
        fn get_fields_at(&self, _time: T, x: T, y: T, z: T) -> Option<[T; 6]> {
            return Some(earth_dipole::<T>(x, y, z));
        }

        fn ds(&self) -> T {
            return T::from(DELTA).unwrap();
        }
    }

    impl<T: PtrTrait> Field<T> for UniformField<T> {
        fn get_fields_at(&self, _time: T, _x: T, _y: T, _z: T) -> Option<[T; 6]> {
            match self.axis {
                0 => Some([
                    self.mag,
                    T::zero(),
                    T::zero(),
                    T::zero(),
                    T::zero(),
                    T::zero(),
                ]),
                1 => Some([
                    T::zero(),
                    self.mag,
                    T::zero(),
                    T::zero(),
                    T::zero(),
                    T::zero(),
                ]),
                2 => Some([
                    T::zero(),
                    T::zero(),
                    self.mag,
                    T::zero(),
                    T::zero(),
                    T::zero(),
                ]),
                _ => panic!(),
            }
        }

        fn ds(&self) -> T {
            return T::from(DELTA).unwrap();
        }
    }

    impl<T: PtrTrait> Field<T> for VlsvStaticField<T> {
        fn get_fields_at(&self, _time: T, x: T, y: T, z: T) -> Option<[T; 6]> {
            let (grid_point, weights) = self.real2mesh(x, y, z, self.periodic)?;
            let e_field = self.trilerp(grid_point, weights, &self.e);
            let b_field = self.trilerp(grid_point, weights, &self.b);
            Some([
                b_field[0], b_field[1], b_field[2], e_field[0], e_field[1], e_field[2],
            ])
        }

        fn ds(&self) -> T {
            return self.ds;
        }
    }
    pub fn mag<T>(x: T, y: T, z: T) -> T
    where
        T: PtrTrait,
    {
        T::sqrt(x * x + y * y + z * z)
    }

    pub fn mag2<T>(x: T, y: T, z: T) -> T
    where
        T: PtrTrait,
    {
        x * x + y * y + z * z
    }

    fn dot<T: PtrTrait>(a: &[T; 3], b: &[T; 3]) -> T {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    pub fn gamma<T>(vx: T, vy: T, vz: T) -> T
    where
        T: PtrTrait,
    {
        let term1: T = T::one();
        let term2: T = T::sqrt(T::one() - (mag2(vx, vy, vz) / T::from(3.0e8 * 3.0e8).unwrap()));
        term1 / term2
    }

    #[derive(Debug)]
    pub struct ParticlePopulation<T: PtrTrait> {
        pub x: Vec<T>,
        pub y: Vec<T>,
        pub z: Vec<T>,
        pub vx: Vec<T>,
        pub vy: Vec<T>,
        pub vz: Vec<T>,
        pub alive: Vec<bool>,
        pub mass: T,
        pub charge: T,
    }

    pub struct ParticleView<'a, T: PtrTrait> {
        pub x: &'a T,
        pub y: &'a T,
        pub z: &'a T,
        pub vx: &'a T,
        pub vy: &'a T,
        pub vz: &'a T,
        pub alive: &'a bool,
    }

    pub struct ParticleIter<'a, T: PtrTrait> {
        population: &'a ParticlePopulation<T>,
        index: usize,
    }

    impl<'a, T: PtrTrait> ParticleIter<'a, T> {
        pub fn new(population: &'a ParticlePopulation<T>) -> Self {
            ParticleIter {
                population,
                index: 0,
            }
        }
    }
    impl<T: PtrTrait> ParticlePopulation<T> {
        pub fn count_alive(&self) -> usize {
            self.alive.iter().copied().filter(|&alive| alive).count()
        }
    }

    impl<'a, T: PtrTrait> Iterator for ParticleIter<'a, T> {
        type Item = ParticleView<'a, T>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.index >= self.population.x.len() {
                return None;
            }

            let i = self.index;
            self.index += 1;

            Some(ParticleView {
                x: &self.population.x[i],
                y: &self.population.y[i],
                z: &self.population.z[i],
                vx: &self.population.vx[i],
                vy: &self.population.vy[i],
                vz: &self.population.vz[i],
                alive: &self.population.alive[i],
            })
        }
    }

    impl<T: PtrTrait> ParticlePopulation<T> {
        pub fn new(n: usize, mass: T, charge: T) -> Self {
            Self {
                x: Vec::<T>::with_capacity(n),
                y: Vec::<T>::with_capacity(n),
                z: Vec::<T>::with_capacity(n),
                vx: Vec::<T>::with_capacity(n),
                vy: Vec::<T>::with_capacity(n),
                vz: Vec::<T>::with_capacity(n),
                alive: Vec::<bool>::with_capacity(n),
                mass,
                charge,
            }
        }

        pub fn iter(&self) -> ParticleIter<'_, T> {
            ParticleIter {
                population: &self,
                index: 0,
            }
        }

        pub fn save(&self, filename: &str) {
            let size = self.size();
            let datasize = std::mem::size_of::<T>();
            let cap = size * std::mem::size_of::<T>() * 6;
            let mut data: Vec<u8> = Vec::with_capacity(cap);
            let bytes: [u8; std::mem::size_of::<usize>()] = size.to_ne_bytes();
            data.extend_from_slice(&bytes);
            let bytes: [u8; std::mem::size_of::<usize>()] = datasize.to_ne_bytes();
            data.extend_from_slice(&bytes);
            //X
            for i in 0..size {
                let bytes = self.x[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //Y
            for i in 0..size {
                let bytes = self.y[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //Z
            for i in 0..size {
                let bytes = self.z[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //VX
            for i in 0..size {
                let bytes = self.vx[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //VY
            for i in 0..size {
                let bytes = self.vy[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //VZ
            for i in 0..size {
                let bytes = self.vz[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            println!(
                "\tWriting {}/{} bytes to {}",
                data.len(),
                cap + 8 + 8,
                filename
            );
            let mut file = std::fs::File::create(filename).expect("Failed to create file");
            file.write_all(&data)
                .expect("Failed to write state file  to file!");
        }

        pub fn new_with_energy_at_Lshell(n: usize, mass: T, charge: T, kev: T, L: T) -> Self {
            let mut pop = Self::new(n, mass, charge);
            let c = T::from(3.0e8).unwrap();
            let ke_joules = kev * T::from(1.602e-16).unwrap();

            let rest_energy = mass * c * c;
            let total_energy = ke_joules + rest_energy;

            // Relativistic speed
            let v = c * (T::one() - (rest_energy / total_energy).powi(2)).sqrt();
            let _pitch_angle_dist = Normal::new(90.0, 5.0).unwrap();

            for _ in 0..n {
                let pitch_angle_deg = T::from(45.0).unwrap(); //
                // T::from(pitch_angle_dist.sample(&mut rng).clamp(0.0, 180.0)).unwrap();
                let pitch_angle_rad =
                    pitch_angle_deg * T::from(PI).unwrap() / T::from(180.0).unwrap();

                let v_par = v * pitch_angle_rad.cos();
                let v_perp = v * pitch_angle_rad.sin();

                // Random phase
                let gyro_phase = T::zero() * T::from(rand::random::<f64>() * 2.0 * PI).unwrap();
                let vx = v_perp * gyro_phase.cos();
                let vy = v_perp * gyro_phase.sin();
                let vz = v_par;
                let _theta = rand::rng().random_range(0.0..2.0 * PI);
                let x = L; //T::from(L.to_f64().unwrap() * theta.cos()).unwrap();
                let y = T::zero();
                // T::from(L.to_f64().unwrap() * theta.sin()).unwrap();
                let _z = T::zero();

                pop.add_particle(
                    [
                        x,
                        y,
                        T::zero(),
                        T::from(vx).unwrap(),
                        T::from(vy).unwrap(),
                        T::from(vz).unwrap(),
                    ],
                    true,
                );
            }

            pop
        }
        pub fn add_particle(&mut self, state: [T; 6], status: bool) {
            self.x.push(state[0]);
            self.y.push(state[1]);
            self.z.push(state[2]);
            self.vx.push(state[3]);
            self.vy.push(state[4]);
            self.vz.push(state[5]);
            self.alive.push(status);
        }

        pub fn size(&self) -> usize {
            self.x.len()
        }

        pub fn get_temp_particle(&self, id: usize) -> Particle<T> {
            Particle {
                x: self.x[id],
                y: self.y[id],
                z: self.z[id],
                vx: self.vx[id],
                vy: self.vy[id],
                vz: self.vz[id],
                alive: self.alive[id],
            }
        }

        pub fn take_temp_particle(&mut self, p: &Particle<T>, id: usize) {
            self.x[id] = p.x;
            self.y[id] = p.y;
            self.z[id] = p.z;
            self.vx[id] = p.vx;
            self.vy[id] = p.vy;
            self.vz[id] = p.vz;
            self.alive[id] = p.alive;
        }
    }

    #[derive(Debug, Clone)]
    pub struct Particle<T: PtrTrait> {
        pub x: T,
        pub y: T,
        pub z: T,
        pub vx: T,
        pub vy: T,
        pub vz: T,
        pub alive: bool,
    }

    impl<T: PtrTrait> Particle<T> {
        pub fn new(x: T, y: T, z: T, vx: T, vy: T, vz: T, alive: bool) -> Self {
            Self {
                x,
                y,
                z,
                vx,
                vy,
                vz,
                alive,
            }
        }
    }
    pub fn boris<T: PtrTrait>(p: &mut Particle<T>, e: &[T], b: &[T], dt: T, m: T, c: T) {
        let mut v_minus: [T; 3] = [T::zero(); 3];
        let mut v_prime: [T; 3] = [T::zero(); 3];
        let mut v_plus: [T; 3] = [T::zero(); 3];
        let mut t: [T; 3] = [T::zero(); 3];
        let mut s: [T; 3] = [T::zero(); 3];

        let g = gamma(p.vx, p.vy, p.vz);
        let cm = c / m;
        let half_dt = T::from(0.5).unwrap() * dt;

        v_minus[0] = p.vx + cm * e[0] * half_dt;
        v_minus[1] = p.vy + cm * e[1] * half_dt;
        v_minus[2] = p.vz + cm * e[2] * half_dt;

        t[0] = cm * b[0] * half_dt / g;
        t[1] = cm * b[1] * half_dt / g;
        t[2] = cm * b[2] * half_dt / g;

        let t_mag2 = t[0].powi(2) + t[1].powi(2) + t[2].powi(2);
        let s_factor = T::from(2.0).unwrap() / (T::one() + t_mag2);

        s[0] = t[0] * s_factor;
        s[1] = t[1] * s_factor;
        s[2] = t[2] * s_factor;

        v_prime[0] = v_minus[0] + (v_minus[1] * t[2] - v_minus[2] * t[1]);
        v_prime[1] = v_minus[1] + (v_minus[2] * t[0] - v_minus[0] * t[2]);
        v_prime[2] = v_minus[2] + (v_minus[0] * t[1] - v_minus[1] * t[0]);

        v_plus[0] = v_minus[0] + (v_prime[1] * s[2] - v_prime[2] * s[1]);
        v_plus[1] = v_minus[1] + (v_prime[2] * s[0] - v_prime[0] * s[2]);
        v_plus[2] = v_minus[2] + (v_prime[0] * s[1] - v_prime[1] * s[0]);

        p.vx = v_plus[0] + cm * e[0] * half_dt;
        p.vy = v_plus[1] + cm * e[1] * half_dt;
        p.vz = v_plus[2] + cm * e[2] * half_dt;
        p.x = p.x + p.vx * dt;
        p.y = p.y + p.vy * dt;
        p.z = p.z + p.vz * dt;
    }

    pub fn larmor_radius<T: PtrTrait>(particle: &Particle<T>, b: &[T; 3], mass: T, charge: T) -> T {
        let b_mag = mag(b[0], b[1], b[2]);
        let v = [particle.vx, particle.vy, particle.vz];
        let dot_vb = dot(&v, b);
        let v_parallel_mag = dot_vb / b_mag;
        let b_unit = [b[0] / b_mag, b[1] / b_mag, b[2] / b_mag];
        let v_parallel = [
            b_unit[0] * v_parallel_mag,
            b_unit[1] * v_parallel_mag,
            b_unit[2] * v_parallel_mag,
        ];

        let v_perp = [
            v[0] - v_parallel[0],
            v[1] - v_parallel[1],
            v[2] - v_parallel[2],
        ];
        let v_perp_mag = mag(v_perp[0], v_perp[1], v_perp[2]);
        let numerator = mass * v_perp_mag;
        let denominator = charge.abs() * b_mag;
        numerator / denominator
    }

    pub fn boris_adaptive<T: PtrTrait, F: Field<T> + std::marker::Sync>(
        p: &mut Particle<T>,
        f: &F,
        dt: &mut T,
        t0: T,
        t1: T,
        mass: T,
        charge: T,
    ) {
        let mut t = t0;
        while t < t1 {
            if t + *dt > t1 {
                *dt = t1 - t;
            }

            let fields_at_p = match f.get_fields_at(t, p.x, p.y, p.z) {
                Some(fields) => fields,
                None => {
                    p.alive = false;
                    return;
                }
            };

            let mut p1 = p.clone();
            boris(
                &mut p1,
                &fields_at_p[3..6],
                &fields_at_p[0..3],
                *dt,
                mass,
                charge,
            );

            let mut p2 = p.clone();
            boris(
                &mut p2,
                &fields_at_p[3..6],
                &fields_at_p[0..3],
                *dt / T::from(2.0).unwrap(),
                mass,
                charge,
            );

            let fields_at_p2 =
                match f.get_fields_at(t + *dt / T::from(2.0).unwrap(), p2.x, p2.y, p2.z) {
                    Some(fields) => fields,
                    None => {
                        p.alive = false;
                        return;
                    }
                };

            boris(
                &mut p2,
                &fields_at_p2[3..6],
                &fields_at_p2[0..3],
                *dt / T::from(2.0).unwrap(),
                mass,
                charge,
            );

            let error = [
                T::from(100.0).unwrap() * (p2.x - p1.x).abs(),
                T::from(100.0).unwrap() * (p2.y - p1.y).abs(),
                T::from(100.0).unwrap() * (p2.z - p1.z).abs(),
            ]
            .iter()
            .copied()
            .fold(T::neg_infinity(), T::max);

            let b = [fields_at_p[0], fields_at_p[1], fields_at_p[2]];
            let larmor = larmor_radius(p, &b, mass, charge);
            let tol = T::from(larmor / T::from(360.0).unwrap()).unwrap();

            let new_dt = T::from(0.9).unwrap()
                * *dt
                * T::min(
                    T::max(
                        (tol / (T::from(2.0).unwrap() * error + T::epsilon())).sqrt(),
                        T::from(0.3).unwrap(),
                    ),
                    T::from(2.0).unwrap(),
                );

            if error < tol {
                *p = p2;
                t = t + *dt;
                *dt = new_dt;
            } else {
                *dt = new_dt;
            }

            if !p.alive {
                break;
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct GuidingCenter2D<T: PtrTrait> {
        pub x: T,
        pub y: T,
        pub vpar: T,
        pub vperp: T,
        pub mu: T,
        pub alive: bool,
        pub dt: T,
    }

    #[derive(Debug, Clone)]
    pub struct GCPopulation<T: PtrTrait> {
        pub particles: Vec<GuidingCenter2D<T>>,
        pub mass: T,
        pub charge: T,
    }

    impl<T: PtrTrait> GCPopulation<T> {
        pub fn new(mass: T, charge: T) -> Self {
            Self {
                particles: Vec::new(),
                mass,
                charge,
            }
        }

        pub fn add(&mut self, gc: GuidingCenter2D<T>) {
            self.particles.push(gc);
        }

        pub fn size(&self) -> usize {
            self.particles.len()
        }

        //Write in the same way we write full particles
        pub fn save(&self, filename: &str) {
            let size = self.size();
            let datasize = std::mem::size_of::<T>();
            let cap = size * datasize * 6 + size;
            let mut data: Vec<u8> = Vec::with_capacity(cap);
            let size_bytes: [u8; std::mem::size_of::<usize>()] = size.to_ne_bytes();
            data.extend_from_slice(&size_bytes);
            let datasize_bytes: [u8; std::mem::size_of::<usize>()] = datasize.to_ne_bytes();
            data.extend_from_slice(&datasize_bytes);

            // x
            for i in 0..size {
                let bytes = self.particles[i].x.to_ne_bytes();
                data.extend_from_slice(bytes.as_ref());
            }
            // y
            for i in 0..size {
                let bytes = self.particles[i].y.to_ne_bytes();
                data.extend_from_slice(bytes.as_ref());
            }
            // z = 0
            for _ in 0..size {
                let bytes = T::zero().to_ne_bytes();
                data.extend_from_slice(bytes.as_ref());
            }
            // vx = vpar
            for i in 0..size {
                let bytes = self.particles[i].vpar.to_ne_bytes();
                data.extend_from_slice(bytes.as_ref());
            }
            // vy = mu
            for i in 0..size {
                let bytes = self.particles[i].mu.to_ne_bytes();
                data.extend_from_slice(bytes.as_ref());
            }
            // vz = 0
            for i in 0..size {
                let bytes = self.particles[i].vperp.to_ne_bytes();
                data.extend_from_slice(bytes.as_ref());
            }
            // alive flags
            for i in 0..size {
                data.push(self.particles[i].alive as u8);
            }

            println!(
                "\tWriting {}/{} bytes to {}",
                data.len(),
                cap + 8 + 8,
                filename
            );
            let mut file = std::fs::File::create(filename).expect("Failed to create file");
            file.write_all(&data)
                .expect("Failed to write GC state to file!");
        }
    }
    pub fn euler_gc_step<T: PtrTrait, F: Field<T>>(
        gc: &GuidingCenter2D<T>,
        f: &F,
        t: T,
        dt: T,
        charge: T,
        mass: T,
    ) -> GuidingCenter2D<T> {
        let die = || {
            let mut dead = gc.clone();
            dead.alive = false;
            dead
        };

        let fields_opt = f.get_fields_at(t, gc.x, gc.y, T::zero());
        let (b, e) = match fields_opt {
            Some(fields) => {
                let b = [fields[0], fields[1], fields[2]];
                let e = [fields[3], fields[4], fields[5]];
                (b, e)
            }
            None => return die(),
        };

        let bmag = mag(b[0], b[1], b[2]);
        if bmag <= T::epsilon() {
            return die();
        }

        //ExB
        let inv_b2 = T::one() / (bmag * bmag);
        let v_exb = [
            (e[1] * b[2] - e[2] * b[1]) * inv_b2,
            (e[2] * b[0] - e[0] * b[2]) * inv_b2,
        ];

        //GradB
        let delta = f.ds() / T::from(4.0).unwrap();
        let (bpx, bmx, bpy, bmy) = match (
            f.get_fields_at(t, gc.x + delta, gc.y, T::zero()),
            f.get_fields_at(t, gc.x - delta, gc.y, T::zero()),
            f.get_fields_at(t, gc.x, gc.y + delta, T::zero()),
            f.get_fields_at(t, gc.x, gc.y - delta, T::zero()),
        ) {
            (Some(px), Some(mx), Some(py), Some(my)) => (px, mx, py, my),
            _ => return die(),
        };

        let grad_b = [
            (mag(bpx[0], bpx[1], bpx[2]) - mag(bmx[0], bmx[1], bmx[2]))
                / (T::from(2.0).unwrap() * delta),
            (mag(bpy[0], bpy[1], bpy[2]) - mag(bmy[0], bmy[1], bmy[2]))
                / (T::from(2.0).unwrap() * delta),
            T::zero(),
        ];

        let v_gradb_3d = [
            (b[1] * grad_b[2] - b[2] * grad_b[1]),
            (b[2] * grad_b[0] - b[0] * grad_b[2]),
            (b[0] * grad_b[1] - b[1] * grad_b[0]),
        ];

        //Curvature
        let b_hat = [b[0] / bmag, b[1] / bmag, b[2] / bmag];
        let (fwd, back) = match (
            f.get_fields_at(
                t,
                gc.x + b_hat[0] * delta,
                gc.y + b_hat[1] * delta,
                b_hat[2] * delta,
            ),
            f.get_fields_at(
                t,
                gc.x - b_hat[0] * delta,
                gc.y - b_hat[1] * delta,
                -b_hat[2] * delta,
            ),
        ) {
            (Some(f), Some(b)) => (f, b),
            _ => return die(),
        };

        let db_ds = [
            (fwd[0] - back[0]) / (T::from(2.0).unwrap() * delta),
            (fwd[1] - back[1]) / (T::from(2.0).unwrap() * delta),
            (fwd[2] - back[2]) / (T::from(2.0).unwrap() * delta),
        ];

        let b_cross_bgrad = [
            b[1] * db_ds[2] - b[2] * db_ds[1],
            b[2] * db_ds[0] - b[0] * db_ds[2],
            b[0] * db_ds[1] - b[1] * db_ds[0],
        ];

        let mu = gc.mu;
        let vperp = ((T::from(2.0).unwrap() * mu * bmag) / mass).sqrt();
        let vpar2 = gc.vpar * gc.vpar;

        let cb2 = charge * bmag * bmag;
        let vx = v_exb[0]
            + v_gradb_3d[0] * (mu / cb2)
            + b_cross_bgrad[0] * (mass * vpar2 / (cb2 * bmag));

        let vy = v_exb[1]
            + v_gradb_3d[1] * (mu / cb2)
            + b_cross_bgrad[1] * (mass * vpar2 / (cb2 * bmag));

        let mut out = gc.clone();
        out.x = out.x + vx * dt;
        out.y = out.y + vy * dt;
        out.vperp = vperp;
        out
    }

    pub fn gc_adaptive<T: PtrTrait, F: Field<T> + Sync>(
        gc: &mut GuidingCenter2D<T>,
        f: &F,
        dt: &mut T,
        t0: T,
        t1: T,
        mass: T,
        charge: T,
    ) {
        if !gc.alive {
            return;
        }
        let mut t = t0;
        let min_dt = T::from(1e-5).unwrap();
        const MAX_ITER: usize = 1000000;

        let mut iter = 0;
        while t < t1 && gc.alive && iter < MAX_ITER {
            iter += 1;
            if t + *dt > t1 {
                *dt = t1 - t;
            }

            //1st order
            let p_full = euler_gc_step(gc, f, t, *dt, charge, mass);

            // 2nd order
            let p_mid = euler_gc_step(gc, f, t, *dt / T::from(2.0).unwrap(), charge, mass);
            let p_half = euler_gc_step(
                &p_mid,
                f,
                t + *dt / T::from(2.0).unwrap(),
                *dt / T::from(2.0).unwrap(),
                charge,
                mass,
            );

            // error
            let ex = (p_half.x - p_full.x).abs();
            let ey = (p_half.y - p_full.y).abs();
            let error = if ex > ey { ex } else { ey };

            // tolerance
            let v_fields_opt = f.get_fields_at(t, gc.x, gc.y, T::zero());
            let v_fields = match v_fields_opt {
                Some(v) => v,
                None => {
                    gc.alive = false;
                    return;
                }
            };
            let b = [v_fields[0], v_fields[1], v_fields[2]];
            let e = [v_fields[3], v_fields[4], v_fields[5]];
            let bmag = mag(b[0], b[1], b[2]);
            let inv_b2 = T::one() / (bmag * bmag);
            let v_exb = [
                (e[1] * b[2] - e[2] * b[1]) * inv_b2,
                (e[2] * b[0] - e[0] * b[2]) * inv_b2,
            ];
            let vmag = mag(v_exb[0], v_exb[1], T::zero());
            let tol = T::from(1e-3).unwrap() * vmag * *dt + T::from(1e-6).unwrap();

            if error < tol || *dt <= min_dt {
                *gc = p_half.clone();
                t = t + *dt;
                *dt = T::from(1.1).unwrap() * *dt;
            } else {
                *dt = *dt * T::from(0.5).unwrap();
            }
        }

        if iter >= MAX_ITER {
            eprintln!("WARNING: gc_adaptive hit MAX_ITER without finishing interval");
        }
    }
}

pub mod mod_vlsv_c_exports {
    use super::mod_vlsv_reader::VlsvFile;
    use ndarray::Array4;
    use std::ffi::{CStr, c_void};
    use std::os::raw::c_char;

    #[repr(C)]
    pub struct GenericGrid {
        nx: usize,
        ny: usize,
        nz: usize,
        nc: usize,
        xmin: f64,
        ymin: f64,
        zmin: f64,
        xmax: f64,
        ymax: f64,
        zmax: f64,
        data: *mut c_void,
        datasize: usize,
    }

    #[repr(C)]
    pub struct Grid<T> {
        nx: usize,
        ny: usize,
        nz: usize,
        nc: usize,
        xmin: f64,
        ymin: f64,
        zmin: f64,
        xmax: f64,
        ymax: f64,
        zmax: f64,
        data: *mut T,
    }

    impl<T> Grid<T> {
        pub fn new(
            meshsize: (usize, usize, usize, usize),
            extents: (f64, f64, f64, f64, f64, f64),
            data: *mut T,
        ) -> Self {
            Self {
                nx: meshsize.0,
                ny: meshsize.1,
                nz: meshsize.2,
                nc: meshsize.3,
                xmin: extents.0,
                ymin: extents.1,
                zmin: extents.2,
                xmax: extents.3,
                ymax: extents.4,
                zmax: extents.5,
                data,
            }
        }
    }

    impl GenericGrid {
        pub fn new(
            meshsize: (usize, usize, usize, usize),
            extents: (f64, f64, f64, f64, f64, f64),
            data: *mut c_void,
            datasize: usize,
        ) -> Self {
            Self {
                nx: meshsize.0,
                ny: meshsize.1,
                nz: meshsize.2,
                nc: meshsize.3,
                xmin: extents.0,
                ymin: extents.1,
                zmin: extents.2,
                xmax: extents.3,
                ymax: extents.4,
                zmax: extents.5,
                data,
                datasize,
            }
        }
    }

    /************************* C Bindings *********************************/
    #[unsafe(export_name = "get_wid")]
    pub unsafe fn get_wid(filename: *const c_char, popname: *const c_char) -> usize {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let pop = unsafe { CStr::from_ptr(popname).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        return f.get_wid(pop).expect("ERROR: could not get WID for {name}");
    }

    #[unsafe(export_name = "read_var")]
    pub unsafe fn read_var(
        filename: *const c_char,
        varname: *const c_char,
        op: i32,
    ) -> GenericGrid {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let ds = f.get_dataset(var).expect("Variable not found");
        let retval = match ds.datasize {
            4 => {
                let var: Array4<f32> = f.read_variable::<f32>(var, Some(op)).unwrap();
                let dims = var.dim();
                let mut vec = var.into_raw_vec_and_offset().0;
                let ptr = vec.as_mut_ptr();
                std::mem::forget(vec);
                GenericGrid::new(
                    dims,
                    f.get_spatial_mesh_extents().unwrap(),
                    ptr as *mut c_void,
                    4,
                )
            }
            8 => {
                let var: Array4<f64> = f.read_variable::<f64>(var, Some(op)).unwrap();
                let dims = var.dim();
                let mut vec = var.into_raw_vec_and_offset().0;
                let ptr = vec.as_mut_ptr();
                std::mem::forget(vec);
                GenericGrid::new(
                    dims,
                    f.get_spatial_mesh_extents().unwrap(),
                    ptr as *mut c_void,
                    8,
                )
            }
            _ => {
                panic!("Unable to create generic grid with this type")
            }
        };
        retval
    }

    #[unsafe(export_name = "read_var_32")]
    pub unsafe fn read_var_32(
        filename: *const c_char,
        varname: *const c_char,
        op: i32,
    ) -> Grid<f32> {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let var: Array4<f32> = f.read_variable::<f32>(var, Some(op)).unwrap();
        let dims = var.dim();
        let mut vec = var.into_raw_vec_and_offset().0;
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        Grid::<f32>::new(dims, f.get_spatial_mesh_extents().unwrap(), ptr)
    }

    #[unsafe(export_name = "read_var_64")]
    pub unsafe fn read_var_64(
        filename: *const c_char,
        varname: *const c_char,
        op: i32,
    ) -> Grid<f64> {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let var: Array4<f64> = f.read_variable::<f64>(var, Some(op)).unwrap();
        let dims = var.dim();
        let mut vec = var.into_raw_vec_and_offset().0;
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        Grid::<f64>::new(dims, f.get_spatial_mesh_extents().unwrap(), ptr)
    }

    #[unsafe(export_name = "read_var_zoom_32")]
    pub unsafe fn read_var_zoom_32(
        filename: *const c_char,
        varname: *const c_char,
        op: i32,
        scale_factor: f64,
    ) -> Grid<f32> {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let var: Array4<f32> = f
            .read_variable_zoom::<f32>(var, Some(op), scale_factor)
            .unwrap();
        let dims = var.dim();
        let mut vec = var.into_raw_vec_and_offset().0;
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        Grid::<f32>::new(dims, f.get_spatial_mesh_extents().unwrap(), ptr)
    }

    #[unsafe(export_name = "read_var_zoom_64")]
    pub unsafe fn read_var_zoom_64(
        filename: *const c_char,
        varname: *const c_char,
        op: i32,
        scale_factor: f64,
    ) -> Grid<f64> {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let var: Array4<f64> = f
            .read_variable_zoom::<f64>(var, Some(op), scale_factor)
            .unwrap();
        let dims = var.dim();
        let mut vec = var.into_raw_vec_and_offset().0;
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        Grid::<f64>::new(dims, f.get_spatial_mesh_extents().unwrap(), ptr)
    }

    #[unsafe(export_name = "read_vdf")]
    pub unsafe fn read_vdf(filename: *const c_char, pop: *const c_char, cid: usize) -> GenericGrid {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let pop = unsafe { CStr::from_ptr(pop).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        match f.get_vdf_float_datasize(pop).unwrap() {
            4 => {
                let var: Array4<f32> = f.read_vdf::<f32>(cid, pop).unwrap();
                let dims = var.dim();
                let mut vec = var.into_raw_vec_and_offset().0;
                let ptr = vec.as_mut_ptr();
                std::mem::forget(vec);
                GenericGrid::new(
                    dims,
                    f.get_vspace_mesh_extents(pop).unwrap(),
                    ptr as *mut c_void,
                    4,
                )
            }
            8 => {
                let var: Array4<f64> = f.read_vdf::<f64>(cid, pop).unwrap();
                let dims = var.dim();
                let mut vec = var.into_raw_vec_and_offset().0;
                let ptr = vec.as_mut_ptr();
                std::mem::forget(vec);
                GenericGrid::new(
                    dims,
                    f.get_vspace_mesh_extents(pop).unwrap(),
                    ptr as *mut c_void,
                    8,
                )
            }
            _ => {
                panic!("Unable to create generic grid with this type")
            }
        }
    }

    #[unsafe(export_name = "read_vdf_32")]
    pub unsafe fn read_vdf_32(
        filename: *const c_char,
        pop: *const c_char,
        cid: usize,
    ) -> Grid<f32> {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let pop = unsafe { CStr::from_ptr(pop).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let var: Array4<f32> = f.read_vdf::<f32>(cid, pop).unwrap();
        let dims = var.dim();
        let mut vec = var.into_raw_vec_and_offset().0;
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        Grid::<f32>::new(dims, f.get_vspace_mesh_extents(pop).unwrap(), ptr)
    }

    #[unsafe(export_name = "read_vdf_64")]
    pub unsafe fn read_vdf_64(
        filename: *const c_char,
        pop: *const c_char,
        cid: usize,
    ) -> Grid<f64> {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let pop = unsafe { CStr::from_ptr(pop).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let var: Array4<f64> = f.read_vdf::<f64>(cid, pop).unwrap();
        let dims = var.dim();
        let mut vec = var.into_raw_vec_and_offset().0;
        let ptr = vec.as_mut_ptr();
        std::mem::forget(vec);
        Grid::<f64>::new(dims, f.get_vspace_mesh_extents(pop).unwrap(), ptr)
    }

    #[unsafe(export_name = "read_vdf_into_32")]
    pub unsafe fn read_vdf_into_32(
        filename: *const c_char,
        pop: *const c_char,
        cid: usize,
        target: *mut Grid<f32>,
    ) {
        debug_assert!(!target.is_null(), "target Grid is NULL");
        let target: &mut Grid<f32> = unsafe { &mut *target };
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let pop = unsafe { CStr::from_ptr(pop).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let mut vdf: Array4<f32> = Array4::<f32>::zeros((target.nx, target.ny, target.nz, 1));
        let new_extents = (
            target.xmin,
            target.ymin,
            target.zmin,
            target.xmax,
            target.ymax,
            target.zmax,
        );
        f.read_vdf_into(cid, pop, &mut vdf, new_extents);
        (target.nx, target.ny, target.nz, _) = vdf.dim();
        let (mut vec, _) = vdf.into_raw_vec_and_offset();
        let ptr = vec.as_mut_ptr();
        target.data = ptr;
        std::mem::forget(vec);
    }

    #[unsafe(export_name = "read_vdf_into_64")]
    pub unsafe fn read_vdf_into_64(
        filename: *const c_char,
        pop: *const c_char,
        cid: usize,
        target: *mut Grid<f64>,
    ) {
        debug_assert!(!target.is_null(), "target Grid is NULL");
        let target: &mut Grid<f64> = unsafe { &mut *target };
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let pop = unsafe { CStr::from_ptr(pop).to_str().unwrap() };
        let f = VlsvFile::new(name).unwrap();
        let mut vdf: Array4<f64> = Array4::<f64>::zeros((target.nx, target.ny, target.nz, 1));
        let new_extents = (
            target.xmin,
            target.ymin,
            target.zmin,
            target.xmax,
            target.ymax,
            target.zmax,
        );
        f.read_vdf_into(cid, pop, &mut vdf, new_extents);
        (target.nx, target.ny, target.nz, _) = vdf.dim();
        let (mut vec, _) = vdf.into_raw_vec_and_offset();
        let ptr = vec.as_mut_ptr();
        target.data = ptr;
        std::mem::forget(vec);
    }

    #[unsafe(export_name = "read_scalar_parameter")]
    pub unsafe fn read_scalar_parameter(filename: *const c_char, parameter: *const c_char) -> f64 {
        let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
        let parameter = unsafe { CStr::from_ptr(parameter).to_str().unwrap() };
        VlsvFile::new(name)
            .unwrap()
            .read_scalar_parameter(parameter)
            .expect("Could not read parameter {parameter} in {name}")
    }
}

#[cfg(feature = "with_bindings")]
pub mod mod_vlsv_py_exports {

    use super::mod_vlsv_reader::*;
    use bytemuck::pod_read_unaligned;
    use ndarray::Array2;
    use ndarray::Array4;
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4};
    use pyfunction;
    use pyo3::exceptions::{PyIOError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::wrap_pyfunction;
    //********************* Python Bindings **************************

    fn map_opt<T, E>(o: Option<T>, msg: E) -> PyResult<T>
    where
        E: std::fmt::Display,
    {
        o.ok_or_else(|| PyValueError::new_err(msg.to_string()))
    }

    #[pyclass(name = "VlsvFile")]
    pub struct PyVlsvFile {
        inner: VlsvFile,
    }
    use pyo3::prelude::*;

    #[pyclass]
    #[derive(Debug, Clone)]
    pub struct HermiteStateOut {
        #[pyo3(get)]
        pub n_hermite_harmonic: i32,
        #[pyo3(get)]
        pub vth: f32,
        #[pyo3(get)]
        pub u: [f32; 3],
        #[pyo3(get)]
        pub spectrum: Vec<f32>,
        #[pyo3(get)]
        pub v_limits: [f64; 6],
        #[pyo3(get)]
        pub shape: [usize; 3],
    }

    #[pymethods]
    impl PyVlsvFile {
        #[new]
        fn new(path: &str) -> PyResult<Self> {
            VlsvFile::new(path)
                .map(|inner| Self { inner })
                .map_err(|e| PyIOError::new_err(format!("Failed to open '{}': {}", path, e)))
        }

        fn __repr__(&self) -> String {
            format!("VlsvFile(filename='{}')", self.inner.filename)
        }

        fn list_variables(&self) -> Vec<String> {
            self.inner.variables().keys().cloned().collect()
        }

        fn read_scalar_parameter(&self, name: &str) -> Option<f64> {
            self.inner.read_scalar_parameter(name)
        }

        fn get_wid(&self, pop: &str) -> Option<usize> {
            self.inner.get_wid(pop)
        }

        fn get_hermite(&self, cid: usize, pop: &str) -> Option<HermiteStateOut> {
            let raw_hermite = self
                .inner
                .get_hermite::<f32>(cid, pop)
                .expect("Could not read Hermite");
            Some(HermiteStateOut {
                n_hermite_harmonic: raw_hermite.n_hermite_harmonic,
                vth: raw_hermite.vth,
                u: raw_hermite.u,
                spectrum: raw_hermite.spectrum,
                v_limits: raw_hermite.v_limits,
                shape: raw_hermite.shape,
            })
        }

        fn get_global_wid(&self) -> Option<usize> {
            self.inner.get_global_wid()
        }

        fn get_spatial_mesh_bbox(&self) -> Option<(usize, usize, usize)> {
            self.inner.get_spatial_mesh_bbox()
        }

        fn get_spatial_mesh_extents(&self) -> Option<(f64, f64, f64, f64, f64, f64)> {
            self.inner.get_spatial_mesh_extents()
        }

        fn get_vspace_mesh_bbox(&self, pop: &str) -> Option<(usize, usize, usize)> {
            self.inner.get_vspace_mesh_bbox(pop)
        }

        fn get_vspace_mesh_extents(&self, pop: &str) -> Option<(f64, f64, f64, f64, f64, f64)> {
            self.inner.get_vspace_mesh_extents(pop)
        }

        fn read_variable<'py>(
            &self,
            py: Python<'py>,
            variable: &str,
            op: Option<i32>,
        ) -> PyResult<PyObject> {
            let ds = self
                .inner
                .get_dataset(variable)
                .expect("Variable not found");
            match ds.datasize {
                4 => {
                    let arr: Array4<f32> = map_opt(
                        self.inner.read_variable::<f32>(variable, op),
                        format!("variable '{}' not found", variable),
                    )?;
                    Ok(arr.into_pyarray(py).to_owned().into())
                }
                8 => {
                    let arr: Array4<f64> = map_opt(
                        self.inner.read_variable::<f64>(variable, op),
                        format!("variable '{}' not found", variable),
                    )?;
                    Ok(arr.into_pyarray(py).to_owned().into())
                }
                _ => {
                    panic!("Type not recognized!")
                }
            }
        }
        fn read_variable_f32<'py>(
            &self,
            py: Python<'py>,
            variable: &str,
            op: Option<i32>,
        ) -> PyResult<Py<PyArray4<f32>>> {
            let arr: Array4<f32> = map_opt(
                self.inner.read_variable::<f32>(variable, op),
                format!("variable '{}' not found", variable),
            )?;
            Ok(arr.into_pyarray(py).to_owned().into())
        }

        fn read_variable_f64<'py>(
            &self,
            py: Python<'py>,
            variable: &str,
            op: Option<i32>,
        ) -> PyResult<Py<PyArray4<f64>>> {
            let arr: Array4<f64> = map_opt(
                self.inner.read_variable::<f64>(variable, op),
                format!("variable '{}' not found", variable),
            )?;
            Ok(arr.into_pyarray(py).to_owned().into())
        }

        fn read_vg_variable_at<'py>(
            &self,
            py: Python<'py>,
            variable: &str,
            cid: Vec<usize>,
        ) -> PyResult<PyObject> {
            let ds = self
                .inner
                .get_dataset(variable)
                .expect("Variable not found");
            match ds.datasize {
                4 => {
                    let vals: Vec<f32> = self
                        .inner
                        .read_vg_variable_at::<f32>(variable, &cid)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "Variable '{}' or CellIDs {:?} not found",
                                variable, cid
                            ))
                        })?;
                    Ok(PyArray1::from_vec(py, vals).to_owned().into())
                }
                8 => {
                    let vals: Vec<f64> = self
                        .inner
                        .read_vg_variable_at::<f64>(variable, &cid)
                        .ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "Variable '{}' or CellIDs {:?} not found",
                                variable, cid
                            ))
                        })?;
                    Ok(PyArray1::from_vec(py, vals).to_owned().into())
                }
                _ => {
                    panic!("Type not recognized!")
                }
            }
        }

        fn read_vg_variable_at_f32<'py>(
            &self,
            py: Python<'py>,
            variable: &str,
            cid: Vec<usize>,
        ) -> PyResult<Py<PyArray1<f32>>> {
            let vals: Vec<f32> = self
                .inner
                .read_vg_variable_at::<f32>(variable, &cid)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Variable '{}' or CellIDs {:?} not found",
                        variable, cid
                    ))
                })?;
            Ok(PyArray1::from_vec(py, vals).to_owned().into())
        }

        fn read_vg_variable_at_f64<'py>(
            &self,
            py: Python<'py>,
            variable: &str,
            cid: Vec<usize>,
        ) -> PyResult<Py<PyArray1<f64>>> {
            let vals: Vec<f64> = self
                .inner
                .read_vg_variable_at::<f64>(variable, &cid)
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "Variable '{}' or CellIDs {:?} not found",
                        variable, cid
                    ))
                })?;
            Ok(PyArray1::from_vec(py, vals).to_owned().into())
        }

        fn read_vdf<'py>(&self, py: Python<'py>, cid: usize, pop: &str) -> PyResult<PyObject> {
            let dtype = self.inner.get_vdf_float_datasize(pop).unwrap();
            match dtype {
                4 => {
                    let arr: Array4<f32> = map_opt(
                        self.inner.read_vdf::<f32>(cid, pop),
                        format!("VDF not found for cid={} pop='{}'", cid, pop),
                    )?;
                    Ok(arr.into_pyarray(py).to_owned().into())
                }
                8 => {
                    let arr: Array4<f64> = map_opt(
                        self.inner.read_vdf::<f64>(cid, pop),
                        format!("VDF not found for cid={} pop='{}'", cid, pop),
                    )?;
                    Ok(arr.into_pyarray(py).to_owned().into())
                }
                _ => {
                    panic!("VDF type not recognized!")
                }
            }
        }

        fn read_vdf_f32<'py>(
            &self,
            py: Python<'py>,
            cid: usize,
            pop: &str,
        ) -> PyResult<Py<PyArray4<f32>>> {
            let arr: Array4<f32> = map_opt(
                self.inner.read_vdf::<f32>(cid, pop),
                format!("VDF not found for cid={} pop='{}'", cid, pop),
            )?;
            Ok(arr.into_pyarray(py).to_owned().into())
        }

        fn read_vdf_f64<'py>(
            &self,
            py: Python<'py>,
            cid: usize,
            pop: &str,
        ) -> PyResult<Py<PyArray4<f64>>> {
            let arr: Array4<f64> = map_opt(
                self.inner.read_vdf::<f64>(cid, pop),
                format!("VDF not found for cid={} pop='{}'", cid, pop),
            )?;
            Ok(arr.into_pyarray(py).to_owned().into())
        }

        fn read_vdf_f32_zoom<'py>(
            &self,
            py: Python<'py>,
            cid: usize,
            pop: &str,
            scale_factor: f64,
        ) -> PyResult<Py<PyArray4<f32>>> {
            debug_assert!(scale_factor > 0.0, "scale_factor must be > 0");

            let new_extents = self.inner.get_vspace_mesh_extents(pop).unwrap();

            let (nx, ny, nz) = {
                let (nx0, ny0, nz0) = self.inner.get_vspace_mesh_bbox(pop).unwrap();
                (
                    ((nx0 as f64) / scale_factor).round() as usize,
                    ((ny0 as f64) / scale_factor).round() as usize,
                    ((nz0 as f64) / scale_factor).round() as usize,
                )
            };

            let mut vdf: Array4<f32> = Array4::<f32>::zeros((nx, ny, nz, 1));
            self.inner.read_vdf_into(cid, pop, &mut vdf, new_extents);
            Ok(vdf.into_pyarray(py).to_owned().into())
        }

        fn read_vdf_f64_zoom<'py>(
            &self,
            py: Python<'py>,
            cid: usize,
            pop: &str,
            scale_factor: f64,
        ) -> PyResult<Py<PyArray4<f64>>> {
            debug_assert!(scale_factor > 0.0, "scale_factor must be > 0");

            let new_extents = self.inner.get_vspace_mesh_extents(pop).unwrap();

            let (nx, ny, nz) = {
                let (nx0, ny0, nz0) = self.inner.get_vspace_mesh_bbox(pop).unwrap();
                (
                    ((nx0 as f64) / scale_factor).round() as usize,
                    ((ny0 as f64) / scale_factor).round() as usize,
                    ((nz0 as f64) / scale_factor).round() as usize,
                )
            };

            let mut vdf: Array4<f64> = Array4::<f64>::zeros((nx, ny, nz, 1));
            self.inner.read_vdf_into(cid, pop, &mut vdf, new_extents);
            Ok(vdf.into_pyarray(py).to_owned().into())
        }

        fn read_vdf_into_f32<'py>(
            &self,
            py: Python<'py>,
            cid: usize,
            pop: &str,
            nx: usize,
            ny: usize,
            nz: usize,
            vxmin: f64,
            vymin: f64,
            vzmin: f64,
            vxmax: f64,
            vymax: f64,
            vzmax: f64,
        ) -> PyResult<Py<PyArray4<f32>>> {
            let mut vdf: Array4<f32> = Array4::<f32>::zeros((nx, ny, nz, 1));
            let new_extents = (vxmin, vymin, vzmin, vxmax, vymax, vzmax);
            self.inner.read_vdf_into(cid, pop, &mut vdf, new_extents);
            Ok(vdf.into_pyarray(py).to_owned().into())
        }

        fn read_vdf_into_f64<'py>(
            &self,
            py: Python<'py>,
            cid: usize,
            pop: &str,
            nx: usize,
            ny: usize,
            nz: usize,
            vxmin: f64,
            vymin: f64,
            vzmin: f64,
            vxmax: f64,
            vymax: f64,
            vzmax: f64,
        ) -> PyResult<Py<PyArray4<f64>>> {
            let mut vdf: Array4<f64> = Array4::<f64>::zeros((nx, ny, nz, 1));
            let new_extents = (vxmin, vymin, vzmin, vxmax, vymax, vzmax);
            self.inner.read_vdf_into(cid, pop, &mut vdf, new_extents);
            Ok(vdf.into_pyarray(py).to_owned().into())
        }

        fn read_vdf_sparse(&self, py: Python<'_>, cid: usize, pop: &str) -> PyResult<PyObject> {
            let dtype = self.inner.get_vdf_float_datasize(pop).unwrap();
            match dtype {
                4 => {
                    let map = self.inner.read_vdf_dict::<f32>(cid, pop).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "VDF not found for cid={} pop='{}'",
                            cid, pop
                        ))
                    })?;
                    Ok(map.into_pyobject(py)?.into_any().unbind())
                }
                8 => {
                    let map = self.inner.read_vdf_dict::<f64>(cid, pop).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "VDF not found for cid={} pop='{}'",
                            cid, pop
                        ))
                    })?;
                    Ok(map.into_pyobject(py)?.into_any().unbind())
                }
                _ => {
                    panic!("VDF type not recognized!")
                }
            }
        }

        fn read_vdf_sparse_f32(
            &self,
            cid: usize,
            pop: &str,
        ) -> PyResult<std::collections::HashMap<usize, f32>> {
            let map = self.inner.read_vdf_dict::<f32>(cid, pop).ok_or_else(|| {
                PyValueError::new_err(format!("VDF not found for cid={} pop='{}'", cid, pop))
            })?;
            Ok(map)
        }

        fn read_vdf_sparse_f64(
            &self,
            cid: usize,
            pop: &str,
        ) -> PyResult<std::collections::HashMap<usize, f64>> {
            let map = self.inner.read_vdf_dict::<f64>(cid, pop).ok_or_else(|| {
                PyValueError::new_err(format!("VDF not found for cid={} pop='{}'", cid, pop))
            })?;
            Ok(map)
        }
    }

    #[pyfunction]
    fn read_variable_f64(
        py: Python<'_>,
        filename: &str,
        variable: &str,
        op: Option<i32>,
    ) -> PyResult<Py<PyArray4<f64>>> {
        let f = VlsvFile::new(filename)
            .map_err(|e| PyIOError::new_err(format!("open '{}': {}", filename, e)))?;
        let arr = map_opt(
            f.read_fsgrid_variable::<f64>(variable, op),
            format!("variable '{}' not found", variable),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }

    #[pyfunction]
    fn read_variable_f32(
        py: Python<'_>,
        filename: &str,
        variable: &str,
        op: Option<i32>,
    ) -> PyResult<Py<PyArray4<f32>>> {
        let f = VlsvFile::new(filename)
            .map_err(|e| PyIOError::new_err(format!("open '{}': {}", filename, e)))?;
        let arr = map_opt(
            f.read_fsgrid_variable::<f32>(variable, op),
            format!("variable '{}' not found", variable),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }

    #[pyfunction]
    fn read_vdf_f32(
        py: Python<'_>,
        filename: &str,
        cid: usize,
        pop: &str,
    ) -> PyResult<Py<PyArray4<f32>>> {
        let f = VlsvFile::new(filename)
            .map_err(|e| PyIOError::new_err(format!("open '{}': {}", filename, e)))?;
        let arr = map_opt(
            f.read_vdf(cid, pop),
            format!("VDF not found for cid={} pop='{}'", cid, pop),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }

    #[pyfunction]
    fn read_vdf_f64(
        py: Python<'_>,
        filename: &str,
        cid: usize,
        pop: &str,
    ) -> PyResult<Py<PyArray4<f64>>> {
        let f = VlsvFile::new(filename)
            .map_err(|e| PyIOError::new_err(format!("open '{}': {}", filename, e)))?;
        let arr = map_opt(
            f.read_vdf(cid, pop),
            format!("VDF not found for cid={} pop='{}'", cid, pop),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }

    //With concurency here to open many files from 1 thread
    #[cfg(all(feature = "uring", target_os = "linux"))]
    fn open_vlsv_files_with_uring(filenames: Vec<String>) -> Vec<VlsvFile> {
        use futures::stream::StreamExt;
        use futures::stream::{self};
        const IO_OPS_LIMIT: usize = 1024;
        tokio_uring::start(async move {
            let results = stream::iter(
                filenames
                    .into_iter()
                    .map(|path| async move { VlsvFile::new_uring(&path).await }),
            )
            .buffered(IO_OPS_LIMIT)
            .collect::<Vec<_>>()
            .await;
            let mut out = Vec::with_capacity(results.len());
            for r in results {
                out.push(r.unwrap());
            }
            out
        })
    }

    fn open_vlsv_files_with_threads(filenames: Vec<String>, max_threads: usize) -> Vec<VlsvFile> {
        use rayon::prelude::*;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads)
            .build()
            .unwrap();
        pool.install(|| {
            filenames
                .into_par_iter()
                .map(|f| VlsvFile::new(&f).unwrap())
                .collect::<Vec<VlsvFile>>()
        })
    }

    fn open_vlsv_files_fast(filenames: Vec<String>) -> Vec<VlsvFile> {
        #[cfg(all(feature = "uring", target_os = "linux"))]
        {
            // println!("Opening files using Uring");
            return open_vlsv_files_with_uring(filenames);
        }

        #[cfg(not(all(feature = "uring")))]
        {
            const THREAD_LIMIT: usize = 16;
            // println!("Opening files using mini Thread Pool");
            return open_vlsv_files_with_threads(filenames, THREAD_LIMIT);
        }
    }

    #[pyfunction]
    fn get_timeline_f32(
        py: Python<'_>,
        filenames: Vec<String>,
        var: &str,
        cids: Vec<usize>,
    ) -> PyResult<Py<PyArray2<f32>>> {
        let n_files = filenames.len();
        let n_cids = cids.len();
        let fptrs = py.allow_threads(|| open_vlsv_files_fast(filenames));
        let mut hints = fptrs.first().unwrap().get_hints_for_cids(&cids);
        let mut flat = vec![0f32; n_cids * n_files];
        for (fi, f) in fptrs.iter().enumerate() {
            let refs = f
                .read_vg_variable_at_as_ref_dyn::<f32>(var, &cids, &mut hints)
                .expect("PEBKAC");
            assert_eq!(refs.len(), n_cids);

            for (ci, bytes) in refs.iter().enumerate() {
                let v = pod_read_unaligned::<f32>(bytes);
                flat[ci * n_files + fi] = v;
            }
        }
        let a = Array2::from_shape_vec((n_cids, n_files), flat).map_err(|_| {
            PyValueError::new_err("I really do not know the conversion I copied from an example :)")
        })?;
        Ok(a.into_pyarray(py).to_owned().into())
    }

    #[pyfunction]
    fn get_timeline_f64(
        py: Python<'_>,
        filenames: Vec<String>,
        var: &str,
        cids: Vec<usize>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        let n_files = filenames.len();
        let n_cids = cids.len();
        let fptrs = py.allow_threads(|| open_vlsv_files_fast(filenames));
        let mut hints = fptrs.first().unwrap().get_hints_for_cids(&cids);
        let mut flat = vec![0f64; n_cids * n_files];
        for (fi, f) in fptrs.iter().enumerate() {
            let refs = f
                .read_vg_variable_at_as_ref_dyn::<f64>(var, &cids, &mut hints)
                .expect("PEBKAC");
            assert_eq!(refs.len(), n_cids);

            for (ci, bytes) in refs.iter().enumerate() {
                let v = pod_read_unaligned::<f64>(bytes);
                flat[ci * n_files + fi] = v;
            }
        }
        let a = Array2::from_shape_vec((n_cids, n_files), flat).map_err(|_| {
            PyValueError::new_err("I really do not know the conversion I copied from an example :)")
        })?;
        Ok(a.into_pyarray(py).to_owned().into())
    }

    // -------------------- module --------------------
    #[pymodule(name = "vlsvrs")]
    fn vlsvrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyVlsvFile>()?;
        m.add_function(wrap_pyfunction!(read_variable_f32, m)?)?;
        m.add_function(wrap_pyfunction!(read_variable_f64, m)?)?;
        m.add_function(wrap_pyfunction!(read_vdf_f32, m)?)?;
        m.add_function(wrap_pyfunction!(read_vdf_f64, m)?)?;
        m.add_function(wrap_pyfunction!(get_timeline_f32, m)?)?;
        m.add_function(wrap_pyfunction!(get_timeline_f64, m)?)?;
        Ok(())
    }
}
