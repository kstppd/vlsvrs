pub mod vlsv_reader;
use ndarray::Array4;
use std::ffi::{CStr, c_void};
use std::os::raw::c_char;
use vlsv_reader::vlsv_reader::VlsvFile;
extern crate libc;

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
        .read_vg_variable_as_fg::<f32>(var)
        .unwrap();
    unsafe {
        (*nx, *ny, *nz, *nc) = var.dim();
    }
    let mut vec = var.into_raw_vec();
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
        .read_vg_variable_as_fg::<f64>(var)
        .unwrap();
    unsafe {
        (*nx, *ny, *nz, *nc) = var.dim();
    }
    let mut vec = var.into_raw_vec();
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
        .read_fsgrid_variable::<f32>(var)
        .unwrap();
    unsafe {
        (*nx, *ny, *nz, *nc) = var.dim();
    }
    let mut vec = var.into_raw_vec();
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
        .read_fsgrid_variable::<f64>(var)
        .unwrap();
    unsafe {
        (*nx, *ny, *nz, *nc) = var.dim();
    }
    let mut vec = var.into_raw_vec();
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
    let mut vec = var.into_raw_vec();
    let ptr = vec.as_mut_ptr();
    std::mem::forget(vec);
    ptr
}

#[unsafe(export_name = "vlsvreader_free")]
pub unsafe fn vlsvreader_free(ptr: *mut f32) {
    unsafe { libc::free(ptr.cast::<c_void>()) };
}
