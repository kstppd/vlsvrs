pub mod vlsv_reader;
use crate::vlsv_reader::vlsv_reader::VlsvFile;
use ndarray::Array4;
use numpy::{IntoPyArray, PyArray4};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::ffi::{CStr, c_void};
use std::os::raw::c_char;
extern crate libc;
use crate::pyfunction;

/************************* C Bindings *********************************/
#[unsafe(export_name = "read_vg_as_fg_32")]
pub unsafe fn read_vg_as_fg_32(
    filename: *const c_char,
    varname: *const c_char,
    nx: *mut usize,
    ny: *mut usize,
    nz: *mut usize,
    nc: *mut usize,
) -> *mut f32 {
    let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
    let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
    println!("Reading in {} from {}", name, var);
    let var: Array4<f32> = VlsvFile::new(name)
        .unwrap()
        .read_vg_variable_as_fg::<f32>(var, None)
        .unwrap();
    unsafe {
        (*nx, *ny, *nz, *nc) = var.dim();
    }
    let mut vec = var.into_raw_vec_and_offset().0;
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[unsafe(export_name = "read_vg_as_fg_64")]
pub unsafe fn read_vg_as_fg_64(
    filename: *const c_char,
    varname: *const c_char,
    nx: *mut usize,
    ny: *mut usize,
    nz: *mut usize,
    nc: *mut usize,
) -> *mut f64 {
    let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
    let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
    println!("Reading in {} from {}", name, var);
    let var: Array4<f64> = VlsvFile::new(name)
        .unwrap()
        .read_vg_variable_as_fg::<f64>(var, None)
        .unwrap();
    unsafe {
        (*nx, *ny, *nz, *nc) = var.dim();
    }
    let mut vec = var.into_raw_vec_and_offset().0;
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}
#[unsafe(export_name = "read_fg_32")]
pub unsafe fn read_fg_32(
    filename: *const c_char,
    varname: *const c_char,
    nx: *mut usize,
    ny: *mut usize,
    nz: *mut usize,
    nc: *mut usize,
) -> *mut f32 {
    let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
    let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
    println!("Reading in {} from {}", name, var);
    let var: Array4<f32> = VlsvFile::new(name)
        .unwrap()
        .read_fsgrid_variable::<f32>(var, None)
        .unwrap();
    unsafe {
        (*nx, *ny, *nz, *nc) = var.dim();
    }
    let mut vec = var.into_raw_vec_and_offset().0;
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}
#[unsafe(export_name = "read_fg_64")]
pub unsafe fn read_fg_64(
    filename: *const c_char,
    varname: *const c_char,
    nx: *mut usize,
    ny: *mut usize,
    nz: *mut usize,
    nc: *mut usize,
) -> *mut f64 {
    let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
    let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
    println!("Reading in {} from {}", name, var);
    let var: Array4<f64> = VlsvFile::new(name)
        .unwrap()
        .read_fsgrid_variable::<f64>(var, None)
        .unwrap();
    unsafe {
        (*nx, *ny, *nz, *nc) = var.dim();
    }
    let mut vec = var.into_raw_vec_and_offset().0;
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}
#[unsafe(export_name = "read_vdf_32")]
pub unsafe fn read_vdf_32(
    filename: *const c_char,
    pop: *const c_char,
    cid: usize,
    nx: *mut usize,
    ny: *mut usize,
    nz: *mut usize,
) -> *mut f32 {
    let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
    let pop = unsafe { CStr::from_ptr(pop).to_str().unwrap() };
    let var: Array4<f32> = VlsvFile::new(name).unwrap().read_vdf(cid, pop).unwrap();
    unsafe {
        (*nx, *ny, *nz, _) = var.dim();
    }
    let mut vec = var.into_raw_vec_and_offset().0;
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[unsafe(export_name = "vlsvreader_free")]
pub unsafe fn vlsvreader_free(ptr: *mut f32) {
    unsafe { libc::free(ptr.cast::<c_void>()) };
}

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
        self.inner.variables.keys().cloned().collect()
    }

    fn read_scalar_parameter(&self, name: &str) -> Option<f64> {
        self.inner.read_scalar_parameter(name)
    }

    fn get_wid(&self) -> Option<usize> {
        self.inner.get_wid()
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

    fn read_fg_variable_f32<'py>(
        &self,
        py: Python<'py>,
        variable: &str,
        op: Option<i32>,
    ) -> PyResult<Py<PyArray4<f32>>> {
        let arr: Array4<f32> = map_opt(
            self.inner.read_fsgrid_variable::<f32>(variable, op),
            format!("variable '{}' not found", variable),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }

    fn read_fg_variable_f64<'py>(
        &self,
        py: Python<'py>,
        variable: &str,
        op: Option<i32>,
    ) -> PyResult<Py<PyArray4<f64>>> {
        let arr: Array4<f64> = map_opt(
            self.inner.read_fsgrid_variable::<f64>(variable, op),
            format!("variable '{}' not found", variable),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }

    fn read_vg_variable_as_fg_f32<'py>(
        &self,
        py: Python<'py>,
        variable: &str,
        op: Option<i32>,
    ) -> PyResult<Py<PyArray4<f32>>> {
        let arr: Array4<f32> = map_opt(
            self.inner.read_vg_variable_as_fg::<f32>(variable, op),
            format!("variable '{}' not found", variable),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }

    fn read_vg_variable_as_fg_f64<'py>(
        &self,
        py: Python<'py>,
        variable: &str,
        op: Option<i32>,
    ) -> PyResult<Py<PyArray4<f64>>> {
        let arr: Array4<f64> = map_opt(
            self.inner.read_vg_variable_as_fg::<f64>(variable, op),
            format!("variable '{}' not found", variable),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }

    fn read_vdf_f32<'py>(
        &self,
        py: Python<'py>,
        cid: usize,
        pop: &str,
    ) -> PyResult<Py<PyArray4<f32>>> {
        let arr: Array4<f32> = map_opt(
            self.inner.read_vdf(cid, pop),
            format!("VDF not found for cid={} pop='{}'", cid, pop),
        )?;
        Ok(arr.into_pyarray(py).to_owned().into())
    }
}

#[pyfunction]
fn read_fg_variable_f32(
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
fn read_fg_variable_f64(
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
fn read_vg_variable_as_fg_f32(
    py: Python<'_>,
    filename: &str,
    variable: &str,
    op: Option<i32>,
) -> PyResult<Py<PyArray4<f32>>> {
    let f = VlsvFile::new(filename)
        .map_err(|e| PyIOError::new_err(format!("open '{}': {}", filename, e)))?;
    let arr = map_opt(
        f.read_vg_variable_as_fg::<f32>(variable, op),
        format!("variable '{}' not found", variable),
    )?;
    Ok(arr.into_pyarray(py).to_owned().into())
}

#[pyfunction]
fn read_vg_variable_as_fg_f64(
    py: Python<'_>,
    filename: &str,
    variable: &str,
    op: Option<i32>,
) -> PyResult<Py<PyArray4<f64>>> {
    let f = VlsvFile::new(filename)
        .map_err(|e| PyIOError::new_err(format!("open '{}': {}", filename, e)))?;
    let arr = map_opt(
        f.read_vg_variable_as_fg::<f64>(variable, op),
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

// // -------------------- module --------------------

#[pymodule]
fn vlsvrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVlsvFile>()?;
    m.add_function(wrap_pyfunction!(read_fg_variable_f32, m)?)?;
    m.add_function(wrap_pyfunction!(read_fg_variable_f64, m)?)?;
    m.add_function(wrap_pyfunction!(read_vg_variable_as_fg_f32, m)?)?;
    m.add_function(wrap_pyfunction!(read_vg_variable_as_fg_f64, m)?)?;
    m.add_function(wrap_pyfunction!(read_vdf_f32, m)?)?;
    Ok(())
}
