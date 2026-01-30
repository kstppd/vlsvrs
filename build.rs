use std::{env, path::PathBuf};

fn main() {
    for cfg in ["no_nn", "no_octree"] {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }
    for k in ["MLP_COMPRESSION_DIR", "OCTREE_COMPRESSION_DIR"] {
        println!("cargo:rerun-if-env-changed={k}");
    }

    let linux = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");

    // MLP
    link_so(
        "MLP_COMPRESSION_DIR",
        "libvlasiator_vdf_compressor_nn.so",
        "no_nn",
        linux,
        |_dir| println!("cargo:rustc-link-lib=vlasiator_vdf_compressor_nn"),
    );

    // Octree
    link_so(
        "OCTREE_COMPRESSION_DIR",
        "libtoctree_compressor.so",
        "no_octree",
        linux,
        |_dir| {
            println!("cargo:rustc-link-lib=toctree_compressor");
            println!("cargo:rustc-link-lib=zfp");
            println!("cargo:rustc-link-lib=stdc++");
        },
    );
}

fn link_so(
    env_key: &str,
    so_name: &str,
    cfg_if_missing: &str,
    linux: bool,
    emit_libs: impl FnOnce(&PathBuf),
) {
    let Some(base) = env::var_os(env_key).map(PathBuf::from) else {
        println!("cargo:rustc-cfg={cfg_if_missing}");
        return;
    };

    let candidates = [
        base.join("lib").join(so_name),
        base.join(so_name),
        base.join("build").join(so_name),
    ];

    let Some(so) = candidates.into_iter().find(|p| p.exists()) else {
        println!("cargo:rustc-cfg={cfg_if_missing}");
        return;
    };

    let dir = so.parent().unwrap().to_path_buf();
    println!("cargo:rustc-link-search=native={}", dir.display());
    emit_libs(&dir);

    if linux {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
    println!("cargo:rerun-if-changed={}", so.display());
}
