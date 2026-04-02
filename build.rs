use std::path::Path;
use std::process::Command;
use std::{env, path::PathBuf};

fn main() {
    let skip_optional = env::var("VLSVRS_SKIP_OPTIONAL").is_ok();

    let zfp_dst = cmake::Config::new("external/zfp")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();

    if skip_optional {
        println!("cargo:warning=VLSVRS_SKIP_OPTIONAL set: skipping Octree and MLP builds.");
        println!("cargo:rustc-cfg=no_octree");
        println!("cargo:rustc-cfg=no_nn");
        for cfg in ["no_nn", "no_octree"] {
            println!("cargo:rustc-check-cfg=cfg({cfg})");
        }
        let zfp_lib_dir = zfp_dst.join("lib");
        let zfp_lib64_dir = zfp_dst.join("lib64");
        if zfp_lib64_dir.exists() {
            println!("cargo:rustc-link-search=native={}", zfp_lib64_dir.display());
        } else {
            println!("cargo:rustc-link-search=native={}", zfp_lib_dir.display());
        }
        println!("cargo:rustc-link-lib=static=zfp");
        println!("cargo:rerun-if-env-changed=VLSVRS_SKIP_OPTIONAL");
        return;
    }
    let eigen_dst = cmake::Config::new("external/eigen")
        .define("CMAKE_POLICY_VERSION_MINIMUM", "3.5")
        .build();

    let octree_src = Path::new("external/tucker-octree");
    let cpp_file = octree_src.join("toctree.cpp");
    let mut octree_compiled = false;

    let patch_1 = Command::new("sed")
        .args(&["-i", "s/ColMajor/RowMajor/g", cpp_file.to_str().unwrap()])
        .status();

    let patch_2 = Command::new("sed")
        .args(&[
            "-i",
            "s/Eigen::Vector<T, Eigen::Dynamic>/Eigen::Matrix<T, Eigen::Dynamic, 1>/g",
            cpp_file.to_str().unwrap(),
        ])
        .status();

    if patch_1.map_or(false, |s| s.success()) && patch_2.map_or(false, |s| s.success()) {
        let octree_dst_result = cmake::Config::new(octree_src)
            .define("TOCTREE_L2ERROR", "true")
            .define("CMAKE_BUILD_TYPE", "Release")
            .define("BUILD_SHARED_LIBS", "OFF")
            .define("zfp_DIR", format!("{}/lib/cmake/zfp", zfp_dst.display()))
            .define(
                "Eigen3_DIR",
                format!("{}/share/eigen3/cmake", eigen_dst.display()),
            )
            .build();

        let lib_path = octree_dst_result.join("lib/libtoctree_compressor.a");
        if lib_path.exists() {
            octree_compiled = true;
            setup_octree_linking(&octree_dst_result, &zfp_dst);
        }
    }

    if !octree_compiled {
        println!("cargo:warning=Octree build failed. Setting no_octree.");
        println!("cargo:rustc-cfg=no_octree");
    }

    let mut ml_compiled = false;
    let asterix_dir = Path::new("external/asterix");
    let mlp_src = asterix_dir.join("src/vdf_compressor_nn.cu");
    let include_path = asterix_dir.join("include");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ml_lib_name = "vlasiator_vdf_compressor_nn";
    let ml_lib_path = out_dir.join(format!("lib{}.so", ml_lib_name));

    let compiler = if is_program_in_path("nvcc") {
        Some("nvcc")
    } else if is_program_in_path("hipcc") {
        Some("hipcc")
    } else {
        None
    };

    if let Some(cc) = compiler {
        println!("cargo:rerun-if-changed={}", mlp_src.display());
        let mut gpu_cmd = Command::new(cc);
        if cc == "nvcc" {
            gpu_cmd.args(&[
                mlp_src.to_str().unwrap(),
                "--std=c++20",
                "-DTINYAI_MEMORY_GB=4",
                "-DNOPROFILE",
                &format!("-I{}", include_path.display()),
                "--shared",
                "-o",
                ml_lib_path.to_str().unwrap(),
                "-Xcompiler=-fPIC",
                "-lcublas",
                "-lblas",
            ]);
        } else if cc == "hipcc" {
            gpu_cmd.args(&[
                mlp_src.to_str().unwrap(),
                "--std=c++20",
                "-I/opt/rocm/include/hipblas",
                "-DTINYAI_MEMORY_GB=4",
                "-DNOPROFILE",
                &format!("-I{}", include_path.display()),
                "--shared",
                "-o",
                ml_lib_path.to_str().unwrap(),
                "-fPIC",
                "-lhipblas",
                "-DSKIP_HOSTBLAS",
            ]);
        }

        if gpu_cmd.status().map(|s| s.success()).unwrap_or(false) {
            ml_compiled = true;
            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=dylib={}", ml_lib_name);

            if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_dir.display());
            }
            if cc == "nvcc" {
                println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
                println!("cargo:rustc-link-lib=cudart");
                println!("cargo:rustc-link-lib=cublas");
            } else if cc == "hipcc" {
                println!("cargo:rustc-link-lib=hipblas");
            }
        } else {
            println!("cargo:warning=MLP build failed. Setting no_nn.");
            println!("cargo:rustc-cfg=no_nn");
        }
    } else {
        println!("cargo:rustc-cfg=no_nn");
    }

    for cfg in ["no_nn", "no_octree"] {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }

    println!("cargo:rerun-if-env-changed=VLSVRS_SKIP_OPTIONAL");
    println!("cargo:rerun-if-env-changed=MLP_COMPRESSION_DIR");
    println!("cargo:rerun-if-changed=external/tucker-octree/toctree.cpp");

    let ml_msg = if ml_compiled { "OK" } else { "FAILED (no_nn)" };
    let octree_msg = if octree_compiled {
        "OK"
    } else {
        "FAILED (no_octree)"
    };
    let zfp_lib_dir = zfp_dst.join("lib");
    let zfp_lib64_dir = zfp_dst.join("lib64");

    if zfp_lib64_dir.exists() {
        println!("cargo:rustc-link-search=native={}", zfp_lib64_dir.display());
    } else {
        println!("cargo:rustc-link-search=native={}", zfp_lib_dir.display());
    }
    println!("cargo:warning=Octree: {} | MLP: {}", octree_msg, ml_msg);
}

fn setup_octree_linking(octree_dst: &PathBuf, zfp_dst: &PathBuf) {
    let octree_lib_dir = octree_dst.join("lib");
    let zfp_lib_dir = zfp_dst.join("lib");
    let zfp_lib64_dir = zfp_dst.join("lib64");

    if zfp_lib64_dir.exists() {
        println!("cargo:rustc-link-search=native={}", zfp_lib64_dir.display());
    } else {
        println!("cargo:rustc-link-search=native={}", zfp_lib_dir.display());
    }

    println!(
        "cargo:rustc-link-search=native={}",
        octree_lib_dir.display()
    );
    println!("cargo:rustc-link-lib=static=toctree_compressor");
    println!("cargo:rustc-link-lib=static=zfp");
    println!("cargo:rustc-link-lib=stdc++");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-lstdc++");
    }
}

fn is_program_in_path(program: &str) -> bool {
    Command::new("which")
        .arg(program)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
