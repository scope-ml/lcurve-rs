//! GPU vs CPU correctness test for chisq_batch.
//!
//! Compares GPU chi-squared values against CPU (Rayon) results for a range
//! of parameter perturbations. Requires a CUDA device and test_data/test_model.dat.

use lcurve::model::Model;
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

#[test]
fn gpu_matches_cpu_chisq_batch() {
    // Load model from workspace-relative path
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_data/test_model.dat");
    let base = Model::from_file(model_path)
        .expect("Failed to load test model");

    let ntime = 200;
    let data = make_data(ntime);

    let param_names: Vec<&str> = vec!["q", "iangle"];
    let ndim = param_names.len();
    let base_q = base.q.value;
    let base_iangle = base.iangle.value;

    // Generate 64 parameter sets with small perturbations
    let n_sets = 64;
    let mut param_values = Vec::with_capacity(n_sets * ndim);
    for i in 0..n_sets {
        let frac = (i as f64 / (n_sets - 1).max(1) as f64) - 0.5;
        param_values.push(base_q + 0.02 * frac);
        param_values.push(base_iangle + 2.0 * frac);
    }

    // CPU reference
    let cpu_result = lcurve::orchestration::chisq_batch(
        &base, &data, &param_names, &param_values, false,
    );

    // GPU result
    let mut ctx = lcurve_cuda::CudaContext::new(0)
        .expect("CUDA init failed — is a GPU available?");
    let gpu_result = lcurve_cuda::chisq_batch_gpu(
        &mut ctx, &base, &data, &param_names, &param_values, false,
    ).expect("GPU chisq_batch failed");

    assert_eq!(cpu_result.len(), gpu_result.len());

    let mut max_rel = 0.0f64;
    let mut n_compared = 0;
    let mut n_nan_cpu = 0;
    let mut n_nan_gpu = 0;

    for (i, (c, g)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        if c.is_nan() {
            n_nan_cpu += 1;
            continue;
        }
        if g.is_nan() {
            n_nan_gpu += 1;
            continue;
        }
        if *c == 0.0 {
            continue;
        }

        let rel = ((c - g) / c).abs();
        if rel > max_rel {
            max_rel = rel;
        }
        n_compared += 1;

        assert!(
            rel < 1e-3,
            "Model {} relative error too large: CPU={:.6e}, GPU={:.6e}, rel={:.6e}",
            i, c, g, rel
        );
    }

    eprintln!(
        "GPU correctness: compared {}/{} models, max_rel_err={:.2e}, NaN(cpu)={}, NaN(gpu)={}",
        n_compared, n_sets, max_rel, n_nan_cpu, n_nan_gpu
    );

    // At least 90% of models should produce valid results
    assert!(
        n_compared >= n_sets * 9 / 10,
        "Too few valid comparisons: {}/{}",
        n_compared, n_sets
    );

    // Max relative error should be well under 1e-3 (typically ~1e-5 from f32 grid data)
    assert!(
        max_rel < 1e-3,
        "Max relative error {:.2e} exceeds tolerance 1e-3",
        max_rel
    );
}

#[test]
fn gpu_matches_cpu_with_scale() {
    let model_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../test_data/test_model.dat");
    let base = Model::from_file(model_path)
        .expect("Failed to load test model");

    let ntime = 100;
    let data = make_data(ntime);

    let param_names: Vec<&str> = vec!["q"];
    let base_q = base.q.value;

    let n_sets = 16;
    let mut param_values = Vec::with_capacity(n_sets);
    for i in 0..n_sets {
        let frac = (i as f64 / (n_sets - 1).max(1) as f64) - 0.5;
        param_values.push(base_q + 0.01 * frac);
    }

    let cpu_result = lcurve::orchestration::chisq_batch(
        &base, &data, &param_names, &param_values, true,
    );

    let mut ctx = lcurve_cuda::CudaContext::new(0)
        .expect("CUDA init failed");
    let gpu_result = lcurve_cuda::chisq_batch_gpu(
        &mut ctx, &base, &data, &param_names, &param_values, true,
    ).expect("GPU chisq_batch with scale failed");

    for (i, (c, g)) in cpu_result.iter().zip(gpu_result.iter()).enumerate() {
        if c.is_nan() || g.is_nan() || *c == 0.0 {
            continue;
        }
        let rel = ((c - g) / c).abs();
        assert!(
            rel < 1e-3,
            "Scaled model {} relative error too large: CPU={:.6e}, GPU={:.6e}, rel={:.6e}",
            i, c, g, rel
        );
    }
}
