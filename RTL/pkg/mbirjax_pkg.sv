// =============================================================================
// mbirjax_pkg.sv
//
// SystemVerilog translation of mbirjax_types.h and shared constants.
//
// Provides:
//   - IEEE-754 single-precision (fp32) type alias
//   - Geometry parameters as localparams
//   - Shared structs: SinoShape, ReconShape, QGGMRFParams, ProjParams
//   - Math helper functions implemented as automatic functions
//   - Neighbour offset table (26-point 3-D neighbourhood)
//
// All floating-point uses IEEE 754-2008 single precision (32-bit).
// Hardware targets: FPGA (Xilinx/Intel) or ASIC.
// =============================================================================

package mbirjax_pkg;

  // ---------------------------------------------------------------------------
  // IEEE-754 single-precision type
  // ---------------------------------------------------------------------------
  typedef logic [31:0] fp32;   // raw 32-bit; treated as real in functions

  // ---------------------------------------------------------------------------
  // Default algorithm parameters  (mirrors C #defines)
  // ---------------------------------------------------------------------------
  localparam real DEF_P             = 1.2;
  localparam real DEF_Q             = 2.0;
  localparam real DEF_T             = 1.0;
  localparam real DEF_SHARPNESS     = 0.0;
  localparam real DEF_SNR_DB        = 30.0;
  localparam real DEF_SIGMA_X       = 0.0;
  localparam real DEF_SIGMA_Y       = 0.0;
  localparam real DEF_DELTA_DET_CH  = 1.0;
  localparam real DEF_DELTA_DET_ROW = 1.0;
  localparam real DEF_DELTA_VOXEL   = 1.0;
  localparam real DEF_MAX_OVERRELAX = 1.5;
  localparam real MJ_EPS            = 1.192093e-7;
  localparam real MJ_PI             = 3.14159265358979323846;

  localparam int  DEF_GRANULARITY   = 16;
  localparam int  N_NEIGHBOURS      = 26;

  // ---------------------------------------------------------------------------
  // Geometry type tag
  // ---------------------------------------------------------------------------
  typedef enum logic [2:0] {
    GEOM_NONE        = 3'd0,
    GEOM_PARALLEL    = 3'd1,
    GEOM_CONE        = 3'd2,
    GEOM_TRANSLATION = 3'd3,
    GEOM_MULTIAXIS   = 3'd4
  } geom_type_e;

  // ---------------------------------------------------------------------------
  // Shape descriptors
  // ---------------------------------------------------------------------------
  typedef struct packed {
    logic [15:0] views;
    logic [15:0] rows;
    logic [15:0] channels;
  } sino_shape_t;

  typedef struct packed {
    logic [15:0] rows;
    logic [15:0] cols;
    logic [15:0] slices;
  } recon_shape_t;

  // ---------------------------------------------------------------------------
  // qGGMRF prior parameters
  // ---------------------------------------------------------------------------
  typedef struct packed {
    logic [31:0] sigma_x;    // fp32
    logic [31:0] p;          // fp32
    logic [31:0] q;          // fp32
    logic [31:0] T;          // fp32
    // b[26] stored as flat packed array: 26*32 = 832 bits
    logic [31:0] b [0:25];   // fp32 neighbour weights (unpacked for SV)
  } qggmrf_params_t;

  // ---------------------------------------------------------------------------
  // Projector geometry parameters  (mirrors ProjParams struct)
  // ---------------------------------------------------------------------------
  typedef struct packed {
    sino_shape_t  sino;
    recon_shape_t recon;
    logic [31:0]  delta_det_channel;   // fp32
    logic [31:0]  delta_det_row;       // fp32
    logic [31:0]  det_row_offset;      // fp32
    logic [31:0]  det_channel_offset;  // fp32
    logic [31:0]  delta_voxel;         // fp32
    logic [31:0]  source_iso_dist;     // fp32 (cone beam; 0 for parallel)
    logic [31:0]  source_det_dist;     // fp32 (cone beam; 0 for parallel)
  } proj_params_t;

  // ---------------------------------------------------------------------------
  // Reconstruction statistics (returned by the VCD controller)
  // ---------------------------------------------------------------------------
  typedef struct {
    int unsigned  num_iterations;
    real          fm_rmse;
    real          prior_loss;
  } recon_stats_t;

  // ---------------------------------------------------------------------------
  // 26-neighbour offset table  (mirrors static g_offs[] in vcd_utils.c)
  //
  // Stored as signed byte triples (dz, dy, dx).
  // Index order: dz=-1..1, dy=-1..1, dx=-1..1, skip (0,0,0).
  // ---------------------------------------------------------------------------
  typedef struct {
    logic signed [1:0] dz;
    logic signed [1:0] dy;
    logic signed [1:0] dx;
  } offset3_t;

  // Pre-computed neighbour table as a function returning an array.
  function automatic void get_nbr_offsets(output offset3_t offs[0:25]);
    int k;
    k = 0;
    for (int dz = -1; dz <= 1; dz++)
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) begin
          if (dz == 0 && dy == 0 && dx == 0) continue;
          offs[k].dz = dz[1:0];
          offs[k].dy = dy[1:0];
          offs[k].dx = dx[1:0];
          k++;
        end
  endfunction

  // Pre-computed neighbour weights b[k] = 1/sqrt(dz²+dy²+dx²)
  function automatic real nbr_weight(int dz, int dy, int dx);
    real d2;
    d2 = real'(dz*dz + dy*dy + dx*dx);
    return (d2 > 0.0) ? 1.0 / $sqrt(d2) : 0.0;
  endfunction

  // ---------------------------------------------------------------------------
  // Scalar math helpers  (automatic functions = synthesisable in SV when
  //                        called from always_comb or other automatic contexts)
  // ---------------------------------------------------------------------------

  function automatic real mj_absf(input real x);
    return (x < 0.0) ? -x : x;
  endfunction

  function automatic real mj_minf(input real a, input real b);
    return (a < b) ? a : b;
  endfunction

  function automatic real mj_maxf(input real a, input real b);
    return (a > b) ? a : b;
  endfunction

  function automatic real mj_clipf(input real x, input real lo, input real hi);
    return mj_maxf(lo, mj_minf(hi, x));
  endfunction

  function automatic int mj_clamp(input int v, input int lo, input int hi);
    return (v < lo) ? lo : ((v > hi) ? hi : v);
  endfunction

  // ---------------------------------------------------------------------------
  // IEEE-754 fp32 ↔ real conversion helpers
  // ---------------------------------------------------------------------------

  // Convert a real (double) to IEEE-754 single-precision bits.
  // Used to initialise register constants from parameter values.
  function automatic fp32 real_to_fp32(input real r);
    logic [63:0] d;
    logic        sign;
    logic [10:0] exp64;
    logic [51:0] mant64;
    logic [7:0]  exp32;
    logic [22:0] mant32;
    int          exp_val;

    // Handle zero
    if (r == 0.0) return 32'h0000_0000;

    // Handle special: NaN / Inf inputs map to fp32 equivalents
    sign = (r < 0.0);
    if (sign) r = -r;

    // Extract exponent and mantissa using $realtobits
    d = $realtobits(r);
    exp64  = d[62:52];
    mant64 = d[51:0];

    // Rebias exponent from double (1023) to single (127)
    exp_val = int'(exp64) - 1023 + 127;
    if (exp_val <= 0)  exp_val = 0;
    if (exp_val >= 255) exp_val = 254;
    exp32  = exp_val[7:0];
    mant32 = mant64[51:29];  // take top 23 bits of double mantissa

    return {sign, exp32, mant32};
  endfunction

  // Convert IEEE-754 single-precision bits to real.
  function automatic real fp32_to_real(input fp32 f);
    logic        sign;
    logic [7:0]  exp8;
    logic [22:0] mant;
    real         result;
    int          e;

    sign = f[31];
    exp8 = f[30:23];
    mant = f[22:0];

    if (exp8 == 8'hFF) begin
      // NaN or Inf
      return 0.0;
    end else if (exp8 == 8'h00) begin
      // Denormal — treat as zero
      return 0.0;
    end else begin
      // Normal number: (-1)^sign * 2^(exp-127) * (1 + mant/2^23)
      e = int'(exp8) - 127;
      result = 1.0 + real'(mant) / real'(1 << 23);
      // Scale by 2^e
      if (e >= 0)
        for (int i = 0; i < e; i++) result *= 2.0;
      else
        for (int i = 0; i < -e; i++) result /= 2.0;
      return sign ? -result : result;
    end
  endfunction

  // ---------------------------------------------------------------------------
  // qGGMRF potential first derivative  (automatic — called from modules)
  //
  // ρ'(δ) = sign(δ)/σx · |u|^(p-1) · [1/D − (p-q)/p · |v|^(p-q)/D²]
  //   where D = 1 + |v|^(p-q),  u = δ/σx,  v = δ/(T·σx)
  // ---------------------------------------------------------------------------
  function automatic real qggmrf_rho_prime(
    input real delta,
    input real sigma_x,
    input real p,
    input real q,
    input real T
  );
    real u, absu, v, absv, upq, D, sgn, corr;

    if (sigma_x <= 0.0) return 0.0;
    u    = delta / sigma_x;
    absu = mj_absf(u);
    if (absu < 1e-30) return 0.0;

    v    = delta / (T * sigma_x);
    absv = mj_absf(v);

    upq  = absv ** (p - q);
    D    = 1.0 + upq;
    sgn  = (delta > 0.0) ? 1.0 : -1.0;
    corr = (p - q) / p * upq / (D * D);

    return sgn / sigma_x * (absu ** (p - 1.0)) * (1.0 / D - corr);
  endfunction

  // qGGMRF second derivative via central finite-difference
  function automatic real qggmrf_rho_dprime(
    input real delta,
    input real sigma_x,
    input real p,
    input real q,
    input real T
  );
    real h, fp, fm;
    h  = 1e-4 * (mj_absf(delta) + sigma_x * 1e-4 + 1e-8);
    fp = qggmrf_rho_prime(delta + h, sigma_x, p, q, T);
    fm = qggmrf_rho_prime(delta - h, sigma_x, p, q, T);
    return (fp - fm) / (2.0 * h);
  endfunction

endpackage : mbirjax_pkg
