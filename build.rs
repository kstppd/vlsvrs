use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(no_octree)");
    println!("cargo:rerun-if-env-changed=VDF_COMPRESSION_DIR");

    // ---- NN shared lib ----
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let nn_name = "libvlasiator_vdf_compressor_nn.so";

    let mut candidates = Vec::new();

    if let Ok(base_dir_str) = env::var("VDF_COMPRESSION_DIR") {
        let base = PathBuf::from(base_dir_str);
        candidates.push(base.join("lib").join(nn_name));
        candidates.push(base.join(nn_name));
    }

    candidates.push(manifest_dir.join(nn_name));
    candidates.push(manifest_dir.join("lib").join(nn_name));

    let mut found_nn = None;

    for full in candidates {
        println!(
            "cargo:warning=Checking for NN shared lib at: {}",
            full.display()
        );
        if full.exists() {
            let dir = full.parent().unwrap().to_path_buf();
            found_nn = Some((dir, full));
            break;
        }
    }

    if let Some((dir, full)) = found_nn {
        println!("cargo:warning=Found NN shared lib: {}", full.display());
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=vlasiator_vdf_compressor_nn");

        // runtime loader
        if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }

        println!("cargo:rerun-if-changed={}", full.display());
    } else {
        println!("cargo:warning=NN shared lib not found, NN decompression will fail to link");
    }

    // ---- OCTREE static lib ----
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
                println!("cargo:warning=Static library not found, setting no_octree");
                println!("cargo:rustc-cfg=no_octree");
            }
        }
        Err(_) => {
            println!("cargo:warning=VDF_COMPRESSION_DIR not set, setting no_octree");
            println!("cargo:rustc-cfg=no_octree");
        }
    }
}
