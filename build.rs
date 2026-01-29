use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(no_octree)");
    println!("cargo:rustc-check-cfg=cfg(no_nn)");
    println!("cargo:rerun-if-env-changed=OCTREE_COMPRESSION_DIR");
    println!("cargo:rerun-if-env-changed=MLP_COMPRESSION_DIR");
    println!("cargo:rerun-if-env-changed=WITHOUT_NN");

    let skip_nn = env::var("WITHOUT_NN").is_ok();
    let mut found_nn = None;

    if !skip_nn {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let nn_name = "libvlasiator_vdf_compressor_nn.so";
        let mut candidates = Vec::new();

        if let Ok(base_dir_str) = env::var("MLP_COMPRESSION_DIR") {
            let base = PathBuf::from(base_dir_str);
            candidates.push(base.join("lib").join(nn_name));
            candidates.push(base.join(nn_name));
        }

        candidates.push(manifest_dir.join(nn_name));
        candidates.push(manifest_dir.join("lib").join(nn_name));

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
    }

    if let Some((dir, full)) = found_nn {
        println!("cargo:warning=Found NN shared lib: {}", full.display());
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=vlasiator_vdf_compressor_nn");

        if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
        println!("cargo:rerun-if-changed={}", full.display());
    } else {
        let reason = if skip_nn {
            "WITHOUT_NN is set"
        } else {
            "NN shared lib not found"
        };
        println!("cargo:warning={}, setting no_nn", reason);
        println!("cargo:rustc-cfg=no_nn");
    }

    match env::var("OCTREE_COMPRESSION_DIR") {
        Ok(base_dir_str) => {
            let base = PathBuf::from(base_dir_str);
            let lib_dir = base.join("lib");
            let lib_file = lib_dir.join("libtoctree_compressor.a");

            if lib_file.exists() {
                println!("cargo:rustc-link-search=native={}", lib_dir.display());
                println!("cargo:rustc-link-lib=static=toctree_compressor");
                println!("cargo:rustc-link-lib=zfp");
                println!("cargo:rustc-link-lib=stdc++");
                println!("cargo:rerun-if-changed={}", lib_file.display());
            } else {
                println!("cargo:rustc-cfg=no_octree");
            }
        }
        Err(_) => {
            println!("cargo:rustc-cfg=no_octree");
        }
    }
}
