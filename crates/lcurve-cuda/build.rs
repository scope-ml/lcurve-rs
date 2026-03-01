use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Locate nvcc
    let nvcc = env::var("NVCC_PATH").unwrap_or_else(|_| "nvcc".to_string());

    // CUDA kernel source
    let kernel_src = PathBuf::from("kernels/flux_kernel.cu");
    let kernel_obj = out_dir.join("flux_kernel.o");

    println!("cargo:rerun-if-changed=kernels/flux_kernel.cu");

    // Multi-arch compilation: Maxwell (sm_50) through Hopper (sm_90)
    let archs = [
        ("compute_50", "sm_50"),
        ("compute_60", "sm_60"),
        ("compute_70", "sm_70"),
        ("compute_80", "sm_80"),
        ("compute_90", "sm_90"),
    ];

    let mut nvcc_cmd = Command::new(&nvcc);
    nvcc_cmd
        .arg("-c")
        .arg(kernel_src.to_str().unwrap())
        .arg("-o")
        .arg(kernel_obj.to_str().unwrap())
        .arg("-O3")
        .arg("--compiler-options")
        .arg("-fPIC");

    for (virt_arch, real_arch) in &archs {
        nvcc_cmd.arg(format!(
            "--generate-code=arch={},code={}",
            virt_arch, real_arch
        ));
    }

    let status = nvcc_cmd
        .status()
        .expect("Failed to execute nvcc. Is CUDA installed? Set NVCC_PATH if nvcc is not in PATH.");

    if !status.success() {
        panic!("nvcc compilation failed with exit code: {:?}", status.code());
    }

    // Pack into static library
    let lib_path = out_dir.join("liblcurve_cuda_kernels.a");
    let ar_status = Command::new("ar")
        .args(["rcs"])
        .arg(lib_path.to_str().unwrap())
        .arg(kernel_obj.to_str().unwrap())
        .status()
        .expect("Failed to run ar");

    if !ar_status.success() {
        panic!("ar failed to create static library");
    }

    // Link the static kernel library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=lcurve_cuda_kernels");

    // Link CUDA runtime
    if let Ok(cuda_lib_path) = env::var("CUDA_LIB_PATH") {
        println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    } else {
        // Common default locations
        for path in &["/usr/local/cuda/lib64", "/usr/lib64"] {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-search=native={}", path);
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
