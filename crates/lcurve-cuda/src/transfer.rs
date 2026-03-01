use lcurve::types::Point;

/// Maximum number of eclipse pairs per surface element.
/// In practice, eclipses have at most 2-3 pairs (primary/secondary/disc).
pub const MAX_ECL: usize = 3;

/// GPU-transferable surface element — flattened version of `Point`.
///
/// All coordinates stored as f32 (sufficient for ~1e-5 relative flux accuracy).
/// Eclipse data flattened from `Vec<(f64,f64)>` to fixed arrays.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuPoint {
    /// Outward normal: x, y, z
    pub dirn_x: f32,
    pub dirn_y: f32,
    pub dirn_z: f32,
    /// Position: x, y, z (for beam/lensing calculations)
    pub posn_x: f32,
    pub posn_y: f32,
    pub posn_z: f32,
    /// Brightness * area
    pub flux: f32,
    /// Number of eclipse pairs (0..MAX_ECL)
    pub n_ecl: i32,
    /// Ingress phases for each eclipse pair
    pub ecl_ing: [f32; MAX_ECL],
    /// Egress phases for each eclipse pair
    pub ecl_eg: [f32; MAX_ECL],
}
// Size: 3*4 + 3*4 + 4 + 4 + 3*4 + 3*4 = 12+12+4+4+12+12 = 56 bytes
// Padded to 64 bytes for alignment:
const _: () = {
    // Static assert that GpuPoint is a reasonable size
    assert!(std::mem::size_of::<GpuPoint>() <= 64);
};

/// Scalar model parameters passed to the GPU kernel via constant memory.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuFluxParams {
    /// cos(inclination)
    pub cosi: f64,
    /// sin(inclination)
    pub sini: f64,
    /// Centre of mass offset: q / (1+q)
    pub xcofm: f64,
    /// Velocity factor: vscale / (c/1e3)
    pub vfac: f64,

    // LDC for star 1
    pub ldc1_1: f64,
    pub ldc1_2: f64,
    pub ldc1_3: f64,
    pub ldc1_4: f64,
    pub mucrit1: f64,
    /// 0 = Poly, 1 = Claret
    pub ltype1: i32,

    // LDC for star 2
    pub ldc2_1: f64,
    pub ldc2_2: f64,
    pub ldc2_3: f64,
    pub ldc2_4: f64,
    pub mucrit2: f64,
    pub ltype2: i32,

    // Beam factors
    pub beam_factor1: f64,
    pub beam_factor2: f64,
    pub spin1: f64,
    pub spin2: f64,

    // Disc limb darkening
    pub lin_limb_disc: f64,
    pub quad_limb_disc: f64,

    // Gravitational lensing
    pub has_glens: i32,
    pub rlens1: f64,

    // Ginterp phase boundaries and scale factors
    pub gint_phase1: f64,
    pub gint_phase2: f64,
    pub gint_scale11: f64,
    pub gint_scale12: f64,
    pub gint_scale21: f64,
    pub gint_scale22: f64,

    // Grid sizes (number of points in each component)
    pub n_star1f: i32,
    pub n_star1c: i32,
    pub n_star2f: i32,
    pub n_star2c: i32,
    pub n_disc: i32,
    pub n_edge: i32,
    pub n_spot: i32,

    /// Padding to keep alignment clean
    pub _pad: [i32; 1],
}

/// Per-time-point data uploaded once per batch.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuDatum {
    /// Orbital phase (pre-computed on CPU including pdot/Roemer corrections)
    pub phase: f64,
    /// Exposure in phase units
    pub expose: f64,
    /// Observed flux
    pub flux_obs: f64,
    /// Flux uncertainty
    pub ferr: f64,
    /// Weight (0 for bad points)
    pub weight: f64,
    /// Number of exposure sub-divisions
    pub ndiv: i32,
    /// Padding
    pub _pad: i32,
}

/// Polynomial fudge factor per time point (pre-computed on CPU).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuFudge {
    /// Slope/polynomial factor: 1 + frac*(slope + frac*(quad + frac*cube))
    pub slfac: f64,
    /// Third light component
    pub third: f64,
}

/// Convert a slice of `Point`s to a Vec of `GpuPoint`s.
pub fn flatten_points(points: &[Point]) -> Vec<GpuPoint> {
    points
        .iter()
        .map(|pt| {
            let n_ecl = pt.eclipse.len().min(MAX_ECL) as i32;
            let mut ecl_ing = [0.0f32; MAX_ECL];
            let mut ecl_eg = [0.0f32; MAX_ECL];
            for (i, &(ing, eg)) in pt.eclipse.iter().take(MAX_ECL).enumerate() {
                ecl_ing[i] = ing as f32;
                ecl_eg[i] = eg as f32;
            }
            GpuPoint {
                dirn_x: pt.dirn.x as f32,
                dirn_y: pt.dirn.y as f32,
                dirn_z: pt.dirn.z as f32,
                posn_x: pt.posn.x as f32,
                posn_y: pt.posn.y as f32,
                posn_z: pt.posn.z as f32,
                flux: pt.flux,
                n_ecl,
                ecl_ing,
                ecl_eg,
            }
        })
        .collect()
}
