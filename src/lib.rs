pub mod vlsv_reader;
use libc::malloc;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr;
use vlsv_reader::vlsv_reader::VlsvFile;

#[repr(C)]
pub struct GridInfo {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn read_fsgrid_variable_f32(
    filename: *const c_char,
    varname: *const c_char,
    out_ptr: *mut *mut f32,
) -> GridInfo {
    let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
    let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
    println!("Reading {} from {}", var, name);
    let _var = VlsvFile::new(&String::from(name))
        .unwrap()
        .read_fsgrid_variable::<f32>(&String::from(var))
        .unwrap();
    let size = _var.len() * std::mem::size_of::<f32>();
    unsafe {
        assert!((*out_ptr).is_null());
        *out_ptr = malloc(size) as *mut f32;
        if (*out_ptr).is_null() {
            panic!("Failed to allocate memory");
        }
        ptr::copy_nonoverlapping(_var.as_ptr(), *out_ptr, _var.len());
    }
    let s = _var.shape();
    let retval = GridInfo {
        nx: s[0],
        ny: s[1],
        nz: s[2],
    };
    retval
}
#[unsafe(no_mangle)]
pub extern "C" fn read_fsgrid_variable_f64(
    filename: *const c_char,
    varname: *const c_char,
    out_ptr: *mut *mut f64,
) -> GridInfo {
    let name = unsafe { CStr::from_ptr(filename).to_str().unwrap() };
    let var = unsafe { CStr::from_ptr(varname).to_str().unwrap() };
    println!("Reading {} from {}", var, name);
    let _var = VlsvFile::new(&String::from(name))
        .unwrap()
        .read_fsgrid_variable::<f64>(&String::from(var))
        .unwrap();
    let size = _var.len() * std::mem::size_of::<f64>();
    unsafe {
        assert!((*out_ptr).is_null());
        *out_ptr = malloc(size) as *mut f64;
        if (*out_ptr).is_null() {
            panic!("Failed to allocate memory");
        }
        ptr::copy_nonoverlapping(_var.as_ptr(), *out_ptr, _var.len());
    }
    let s = _var.shape();
    let retval = GridInfo {
        nx: s[0],
        ny: s[1],
        nz: s[2],
    };
    retval
}
