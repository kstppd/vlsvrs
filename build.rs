use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(no_octree)");
    println!("cargo:rerun-if-env-changed=VDF_COMPRESSION_DIR");
    match env::var("VDF_COMPRESSION_DIR") {
        Ok(base_dir_str) => {
            let base = PathBuf::from(base_dir_str);
            let lib_dir = base.join("lib");
            let lib_file = lib_dir.join("libtoctree_compressor.a");

            println!(
                "cargo:warning=Checking for static lib at: {}",
                lib_file.display()
            );

            if lib_file.exists() {
                println!(
                    "cargo:warning=Found static lib, linking: {}",
                    lib_file.display()
                );
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
                println!("cargo:rustc-link-lib=static=toctree_compressor");
                println!("cargo:rustc-link-lib=zfp");
                println!("cargo:rustc-link-lib=stdc++");

                println!("cargo:rerun-if-changed={}", lib_file.display());
            } else {
                println!(
                    "cargo:warning=Static library not found at {}, setting 'no_octree' flag.",
                    lib_file.display()
                );
                println!("cargo:rustc-cfg=no_octree");
            }
        }
        Err(_) => {
            println!("cargo:warning=VDF_COMPRESSION_DIR not set, setting 'no_octree' flag.");
            println!("cargo:rustc-cfg=no_octree");
        }
    }
}
