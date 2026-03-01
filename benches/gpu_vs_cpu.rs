//! Benchmark: GPU vs CPU chisq_batch
//!
//! Run with:
//!   NVCC_PATH=/usr/local/cuda/bin/nvcc CUDA_LIB_PATH=/usr/local/cuda/lib64 \
//!     cargo run --release --bin gpu_bench --features cuda

use std::time::Instant;
use lcurve::types::{Datum, Data};

/// Build synthetic data with nonzero weight and ferr.
fn make_data(ntime: usize) -> Data {
    let time1 = -0.2;
    let time2 = 1.2;
    (0..ntime)
        .map(|i| {
            let time = time1 + (time2 - time1) * i as f64 / (ntime - 1) as f64;
            Datum {
                time,
                expose: 0.001,
                flux: 1.0e-9,
                ferr: 1.0e-11,
                weight: 1.0,
                ndiv: 1,
            }
        })
        .collect()
}

fn main() {
    let model_path = "test_data/test_model.dat";

    let base = lcurve::model::Model::from_file(model_path)
        .expect("Failed to load model");

    let ntime = 500;
    let data = make_data(ntime);

    println!("=== GPU vs CPU Benchmark: chisq_batch ===");
    println!("Model: {}", model_path);
    println!("Data:  {} time points (synthetic)", ntime);
    println!("GPU:   Tesla P100-PCIE-12GB (sm_60)");
    println!();

    // Verify base model works and report grid size
    let test = lcurve::orchestration::light_curve_comp(&base, &data, false, true);
    match &test {
        Ok(r) => println!("Base model chisq = {:.4e}", r.chisq),
        Err(e) => {
            println!("Base model FAILS: {}", e);
            return;
        }
    }

    // Report grid sizes
    {
        use lcurve_roche::Star;
        let s1f = lcurve::grid::set_star_grid(&base, Star::Primary, true).unwrap();
        let s2f = lcurve::grid::set_star_grid(&base, Star::Secondary, true).unwrap();
        let s1c = lcurve::grid::set_star_grid(&base, Star::Primary, false).unwrap();
        let s2c = lcurve::grid::set_star_grid(&base, Star::Secondary, false).unwrap();
        println!("Star1 fine: {} pts, coarse: {} pts", s1f.len(), s1c.len());
        println!("Star2 fine: {} pts, coarse: {} pts", s2f.len(), s2c.len());
        println!("Total fine points: {}", s1f.len() + s2f.len());
    }
    println!();

    let param_names: Vec<&str> = vec!["q", "iangle"];
    let ndim = param_names.len();
    let base_q = base.q.value;
    let base_iangle = base.iangle.value;

    // ---- Profile: where does time go in GPU path? ----
    println!("--- Profiling GPU path (single model) ---");
    {
        let mut ctx = lcurve_cuda::CudaContext::new(0).expect("CUDA init failed");
        let values = vec![base_q, base_iangle];

        // Grid setup time
        let t0 = Instant::now();
        for _ in 0..10 {
            let _ = lcurve_cuda::chisq_batch_gpu(
                &mut ctx, &base, &data, &param_names, &values, false,
            );
        }
        let total_per_model = t0.elapsed().as_secs_f64() / 10.0;

        // Just CPU light_curve_comp for comparison
        let t0 = Instant::now();
        for _ in 0..10 {
            let _ = lcurve::orchestration::light_curve_comp(&base, &data, false, true);
        }
        let cpu_per_model = t0.elapsed().as_secs_f64() / 10.0;

        println!("GPU total per model: {:.3}s", total_per_model);
        println!("CPU total per model: {:.3}s", cpu_per_model);
        println!("GPU/CPU single model ratio: {:.2}x", total_per_model / cpu_per_model);
    }
    println!();

    // ---- Batch benchmark ----
    println!("--- Batch benchmark (CPU uses Rayon parallelism) ---");
    println!("{:>6} {:>10} {:>10} {:>10} {:>14} {:>14}", "N", "CPU(s)", "GPU(s)", "Speedup", "max_rel_err", "mean_rel_err");
    println!("{}", "-".repeat(78));

    for &n_sets in &[32, 64, 128, 256, 512] {
        let mut param_values = Vec::with_capacity(n_sets * ndim);
        for i in 0..n_sets {
            let frac = (i as f64 / (n_sets - 1).max(1) as f64) - 0.5;
            param_values.push(base_q + 0.02 * frac);
            param_values.push(base_iangle + 2.0 * frac);
        }

        // Verify no NaN
        let check = lcurve::orchestration::chisq_batch(
            &base, &data, &param_names, &param_values, false,
        );
        let nan_count = check.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            println!("{:>6} | SKIP: {}/{} CPU results are NaN", n_sets, nan_count, n_sets);
            continue;
        }

        // CPU
        let _ = lcurve::orchestration::chisq_batch(&base, &data, &param_names, &param_values, false);
        let n_repeats = 3;
        let mut cpu_times = Vec::new();
        for _ in 0..n_repeats {
            let t0 = Instant::now();
            let r = lcurve::orchestration::chisq_batch(&base, &data, &param_names, &param_values, false);
            cpu_times.push(t0.elapsed().as_secs_f64());
            std::hint::black_box(&r);
        }
        let cpu_mean = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;

        // GPU
        let mut ctx = lcurve_cuda::CudaContext::new(0).expect("CUDA init failed");
        let _ = lcurve_cuda::chisq_batch_gpu(&mut ctx, &base, &data, &param_names, &param_values, false);

        let mut gpu_times = Vec::new();
        for _ in 0..n_repeats {
            let t0 = Instant::now();
            let r = lcurve_cuda::chisq_batch_gpu(&mut ctx, &base, &data, &param_names, &param_values, false)
                .expect("GPU failed");
            gpu_times.push(t0.elapsed().as_secs_f64());
            std::hint::black_box(&r);
        }
        let gpu_mean = gpu_times.iter().sum::<f64>() / gpu_times.len() as f64;

        // Correctness
        let cpu_result = lcurve::orchestration::chisq_batch(&base, &data, &param_names, &param_values, false);
        let gpu_result = lcurve_cuda::chisq_batch_gpu(&mut ctx, &base, &data, &param_names, &param_values, false).unwrap();

        let mut max_rel = 0.0f64;
        let mut sum_rel = 0.0f64;
        let mut n_cmp = 0;
        for (c, g) in cpu_result.iter().zip(gpu_result.iter()) {
            if c.is_nan() || g.is_nan() || *c == 0.0 { continue; }
            let rel = ((c - g) / c).abs();
            max_rel = max_rel.max(rel);
            sum_rel += rel;
            n_cmp += 1;
        }
        let mean_rel = if n_cmp > 0 { sum_rel / n_cmp as f64 } else { 0.0 };

        let speedup = cpu_mean / gpu_mean;
        println!(
            "{:>6} {:>10.3} {:>10.3} {:>9.2}x {:>14.2e} {:>14.2e}",
            n_sets, cpu_mean, gpu_mean, speedup, max_rel, mean_rel
        );
    }

    println!();
    println!("Note: GPU path runs grid setup (Roche geometry, brightness) sequentially on CPU.");
    println!("The kernel-only speedup is masked by the grid setup overhead.");
    println!("Speedup improves with larger grid sizes (more points per star).");
}
