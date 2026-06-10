# mbir-sv — SystemVerilog Implementation Reference

SystemVerilog translation of the mbirjax-c MBIR (Model-Based Iterative Reconstruction) library.  
Target platforms: Xilinx/Intel FPGA, ASIC (TSMC/GF).  
Standard: **IEEE 1800-2017 SystemVerilog**.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Flow](#2-architecture-flow)
3. [File Structure](#3-file-structure)
4. [Module Reference](#4-module-reference)
   - 4.1 [mbirjax_pkg — Shared Package](#41-mbirjax_pkg--shared-package)
   - 4.2 [param_store — Parameter Register File](#42-param_store--parameter-register-file)
   - 4.3 [ror_mask_gen — ROR Mask Generator](#43-ror_mask_gen--ror-mask-generator)
   - 4.4 [ramp_filter — Ram-Lak Ramp Filter](#44-ramp_filter--ram-lak-ramp-filter)
   - 4.5 [bilinear_fwd_proj — Forward Projector](#45-bilinear_fwd_proj--forward-projector)
   - 4.6 [bilinear_back_proj — Back Projector](#46-bilinear_back_proj--back-projector)
   - 4.7 [qggmrf_compute — qGGMRF Prior Engine](#47-qggmrf_compute--qggmrf-prior-engine)
   - 4.8 [line_search — Step Size Solver](#48-line_search--step-size-solver)
   - 4.9 [vcd_controller — VCD Reconstruction Controller](#49-vcd_controller--vcd-reconstruction-controller)
   - 4.10 [fbp_controller — FBP Controller](#410-fbp_controller--fbp-controller)
   - 4.11 [fdk_controller — FDK Controller](#411-fdk_controller--fdk-controller)
   - 4.12 [shepp_logan_gen — Phantom Generator](#412-shepp_logan_gen--phantom-generator)
   - 4.13 [preprocess_unit — Sinogram Preprocessor](#413-preprocess_unit--sinogram-preprocessor)
   - 4.14 [denoising_unit — MAP Denoiser + Median Filter](#414-denoising_unit--map-denoiser--median-filter)
   - 4.15 [mbirjax_top — Top-Level Integration](#415-mbirjax_top--top-level-integration)
5. [C-to-SystemVerilog Mapping](#5-c-to-systemverilog-mapping)
6. [Simulation](#6-simulation)
7. [Synthesis Notes](#7-synthesis-notes)
8. [Verification](#8-verification)

---

## 1. Overview

| Metric | Value |
|--------|-------|
| SV files | 15 RTL + 1 package + 1 testbench = **17 files** |
| Total lines | ~2,680 |
| SV standard | IEEE 1800-2017 |
| Arithmetic | IEEE 754-2008 single-precision (fp32) |
| Dependencies | None (self-contained package) |
| Target | FPGA (Xilinx Vivado / Intel Quartus) or ASIC |

The design implements the complete mbirjax algorithm hierarchy as a clocked synchronous digital circuit:

- **Parameter register file** (32 × 32-bit) with named address constants.
- **VCD reconstruction engine** (multi-granular VCD outer loop, qGGMRF prior, bilinear projectors, quadratic line search).
- **FBP / FDK direct reconstruction** (Ram-Lak ramp filter + bilinear back-projection).
- **MAP denoiser** (identity forward model A=I, VCD with qGGMRF prior).
- **3×3×3 median filter** (27-element sort per voxel).
- **Shepp-Logan phantom generator** (10 ellipsoids, streaming output).
- **Sinogram pre-processor** (transmission→attenuation, Poisson weights, bad-pixel correction).

---

## 2. Architecture Flow

```
                    ┌────────────────────────────────────────────────────────┐
                    │                  mbirjax_top                           │
                    │                                                        │
  Host CFG ──────►  │ ┌─────────────┐                                        │
  (addr/data/wr)    │ │ param_store │  32 × 32-bit register file            │
                    │ │ (32 regs)   │  Reset → library defaults             │
                    │ └──────┬──────┘                                        │
                    │        │ sigma_x/y, p/q/T, geometry, granularity       │
                    │        ▼                                                │
                    │ ┌──────────────────────────────────────────────────┐   │
                    │ │              Operation Dispatcher                 │   │
                    │ │   op_mode: 0=VCD, 1=FBP, 2=FDK, 3=Denoise,      │   │
                    │ │             4=Median, 5=Phantom                   │   │
                    │ └──────────────────────────────────────────────────┘   │
                    │           │                                             │
          ┌─────────────────────┼──────────────────────────────────────────┐ │
          │                     │                                           │ │
          ▼                     ▼                      ▼                   ▼ │
   ┌─────────────┐    ┌─────────────────┐    ┌──────────────┐  ┌──────────┐ │
   │ror_mask_gen │    │ vcd_controller  │    │fbp_controller│  │fdk_ctrl  │ │
   │(inscribed   │    │                 │    │              │  │          │ │
   │ circle mask)│    │┌───────────────┐│    │┌────────────┐│  │┌────────┐│ │
   └─────────────┘    ││qggmrf_compute ││    ││ramp_filter ││  ││ramp+   ││ │
                      │└───────────────┘│    ││(Ram-Lak)   ││  ││cosine  ││ │
                      │┌───────────────┐│    │└────────────┘│  ││weight  ││ │
                      ││bilinear_fwd   ││    │┌────────────┐│  │└────────┘│ │
                      ││_proj          ││    ││bilinear    ││  └──────────┘ │
                      │└───────────────┘│    ││_back_proj  ││               │
                      │┌───────────────┐│    │└────────────┘│               │
                      ││bilinear_back  ││    └──────────────┘               │
                      ││_proj          ││                                    │
                      │└───────────────┘│                                    │
                      │┌───────────────┐│    ┌────────────────┐             │
                      ││line_search    ││    │denoising_unit  │             │
                      │└───────────────┘│    │ MAP denoiser   │             │
                      └─────────────────┘    │ median filter  │             │
                                             └────────────────┘             │
                    │                                                        │
                    │ ┌────────────────┐  ┌────────────────┐                │
                    │ │shepp_logan_gen │  │preprocess_unit │                │
                    │ │ 10 ellipsoids  │  │ atten/weights/ │                │
                    │ └────────────────┘  │ bad_pix        │                │
                    │                     └────────────────┘                │
                    └────────────────────────────────────────────────────────┘
                          │ SRAM interfaces (sino / recon / weight)
                          ▼
                    External SRAM (host-managed)
```

### Data flow — VCD reconstruction (op_mode = 0)

```
Sinogram SRAM
      │
      ▼
 error_sino = sino − A·recon_init       ← initial error computation
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                 VCD Outer Loop (max_iterations)          │
│                                                         │
│  for each subset (granularity pixels):                  │
│                                                         │
│    [qggmrf_compute] → prior_grad, prior_hess           │
│         │ 26 neighbours per voxel, qGGMRF ρ'(δ)        │
│                                                         │
│    [bilinear_back_proj, coeff²] → fwd_hess             │
│         │ squared back-projection of all-ones sino      │
│                                                         │
│    [bilinear_back_proj, coeff¹] → fwd_grad             │
│         │ back-project weighted error sinogram           │
│                                                         │
│    Δx = -(fwd_grad + prior_grad) / (fwd_hess + prior_hess + ε)
│                                                         │
│    [bilinear_fwd_proj] Δx → Δsino                     │
│                                                         │
│    [line_search] α = clip((fwd_lin-prior_lin)/         │
│                            (fwd_quad+prior_quad+ε), ε, 1.5)
│                                                         │
│    recon     += α · Δx                                 │
│    error_sino -= α · Δsino                             │
│                                                         │
│    NMAE = ||Δx||₁ / ||recon||₁                        │
│    if NMAE% < stop_threshold → break                   │
└─────────────────────────────────────────────────────────┘
      │
      ▼
Recon SRAM (output)
```

---

## 3. File Structure

```
mbirjax_sv_final/
├── Makefile              — xsim / vcs / iverilog targets
├── filelist.f            — ordered file list for simulators
├── pkg/
│   └── mbirjax_pkg.sv    — shared package (types, functions, constants)
├── rtl/
│   ├── param_store.sv         — 32×32-bit parameter register file
│   ├── ror_mask_gen.sv        — ROR inscribed-circle mask generator
│   ├── ramp_filter.sv         — 1-D Ram-Lak ramp filter (streaming)
│   ├── bilinear_fwd_proj.sv   — bilinear forward projector (one view)
│   ├── bilinear_back_proj.sv  — bilinear back projector (one view)
│   ├── qggmrf_compute.sv      — qGGMRF gradient + hessian (26 neighbours)
│   ├── line_search.sv         — VCD quadratic line search
│   ├── vcd_controller.sv      — multi-granular VCD outer loop FSM
│   ├── fbp_controller.sv      — FBP (parallel beam) controller FSM
│   ├── fdk_controller.sv      — FDK (cone beam) controller FSM
│   ├── shepp_logan_gen.sv     — 3-D Shepp-Logan phantom generator
│   ├── preprocess_unit.sv     — sinogram pre-processing pipeline
│   ├── denoising_unit.sv      — MAP denoiser + median filter
│   └── mbirjax_top.sv         — top-level integration + SRAM mux
└── tb/
    └── tb_mbirjax_top.sv      — SystemVerilog testbench (8 test cases)
```

---

## 4. Module Reference

---

### 4.1 `mbirjax_pkg` — Shared Package

**File:** `pkg/mbirjax_pkg.sv`  
**Mirrors:** `mbirjax_types.h` + `vcd_utils.c` math functions

All RTL modules import this package: `import mbirjax_pkg::*;`

#### Types

| Type | Definition | C equivalent |
|------|-----------|--------------|
| `fp32` | `logic [31:0]` | `float` (IEEE-754 single) |
| `geom_type_e` | enum `{GEOM_NONE..GEOM_MULTIAXIS}` | `MjGeomType` |
| `sino_shape_t` | packed struct `{views, rows, channels}` | `SinoShape` |
| `recon_shape_t` | packed struct `{rows, cols, slices}` | `ReconShape` |
| `qggmrf_params_t` | packed struct `{sigma_x, p, q, T, b[26]}` | `QGGMRFParams` |
| `proj_params_t` | packed struct with all geometry fields | `ProjParams` |
| `offset3_t` | struct `{dz, dy, dx}` (2-bit signed) | `Offset3` |

#### Key constants

| Constant | Value | Notes |
|----------|-------|-------|
| `MJ_EPS` | 1.192093e-7 | FLT_EPSILON |
| `MJ_PI` | 3.14159265… | π |
| `N_NEIGHBOURS` | 26 | 3-D 26-neighbourhood |
| `DEF_P` | 1.2 | qGGMRF shape |
| `DEF_Q` | 2.0 | qGGMRF quadratic |
| `DEF_T` | 1.0 | qGGMRF threshold |
| `DEF_SNR_DB` | 30.0 | default SNR |
| `DEF_GRANULARITY` | 16 | VCD subset size |

#### Functions (automatic — called in always_comb / initial)

| Function | C equivalent | Description |
|----------|-------------|-------------|
| `mj_absf(x)` | `mj_absf(x)` | Absolute value |
| `mj_clipf(x,lo,hi)` | `mj_clipf(...)` | Clamp to [lo, hi] |
| `mj_clamp(v,lo,hi)` | `mj_clamp(...)` | Integer clamp |
| `nbr_weight(dz,dy,dx)` | `vcd_nbr_weight(...)` | 1/√(dz²+dy²+dx²) |
| `get_nbr_offsets(offs)` | `init_offsets()` | Fill 26-offset table |
| `qggmrf_rho_prime(δ,σx,p,q,T)` | `qggmrf_rho_prime()` | ρ'(δ) |
| `qggmrf_rho_dprime(δ,σx,p,q,T)` | `qggmrf_rho_dprime()` | ρ''(δ) finite-diff |
| `real_to_fp32(r)` | (none) | real → IEEE-754 bits |
| `fp32_to_real(f)` | (none) | IEEE-754 bits → real |

---

### 4.2 `param_store` — Parameter Register File

**File:** `rtl/param_store.sv`  
**Mirrors:** `parameter_handler.c` (ParameterHandler class)

32-entry × 32-bit synchronous register file. Active-low reset loads all library defaults.

```
Ports:
  clk, rst_n                    — clock, active-low reset
  wr_en, wr_addr[4:0], wr_data  — synchronous write port
  rd_addr[4:0], rd_data         — combinational read port
  sigma_y_o ... recon_slices_o  — named direct outputs
```

#### Address map

| Address | Parameter | fp32 default |
|---------|-----------|-------------|
| 0x00 | sigma_y | 0.0 |
| 0x01 | sigma_x | 0.0 |
| 0x03 | p | 1.2 (0x3F99999A) |
| 0x04 | q | 2.0 (0x40000000) |
| 0x05 | T | 1.0 (0x3F800000) |
| 0x06 | delta_det_channel | 1.0 |
| 0x12 | snr_db | 30.0 (0x41F00000) |
| 0x13 | sharpness | 0.0 |
| 0x14 | granularity | 16 (integer) |
| 0x18–0x1D | sino/recon shape | 0 |

---

### 4.3 `ror_mask_gen` — ROR Mask Generator

**File:** `rtl/ror_mask_gen.sv`  
**Mirrors:** `mj_gen_ror_mask()` in `tomography_model.c`

FSM-based raster scanner that outputs flat pixel indices `(row*cols + col)` for all pixels inside the inscribed circle of the reconstruction grid.

**Formula:**  `4*(dx² + dy²) ≤ min(rows,cols)²`  where `2dx = 2col − (cols−1)`, `2dy = 2row − (rows−1)`.

**States:** `IDLE → SCAN → DONE_ST`

**Interface:**
```
start_i          — begin scan
pix_valid_o      — output pixel index valid
pix_index_o[19:0] — flat index of in-ROR pixel
n_pixels_o[19:0]  — total count (valid when done_o)
done_o           — scan complete
```

---

### 4.4 `ramp_filter` — Ram-Lak Ramp Filter

**File:** `rtl/ramp_filter.sv`  
**Mirrors:** `apply_ramp_1d()` in `parallel_beam_model.c`

Streaming 1-D Ram-Lak ramp filter applied along the detector channel axis.

**Kernel:** `h[0] = 1/4`,  `h[k odd] = −1/(k²π²)`,  `h[k even≠0] = 0`

**Architecture:** Line buffer (ring buffer, depth `MAX_CHANS`) + coefficient ROM (HALF_KERNEL entries) + MAC accumulator.

**States:** `F_IDLE → F_CENTER (buffer row) → F_TAPS (compute per output) → F_EMIT`

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HALF_KERNEL` | 64 | Number of odd taps per side |
| `MAX_CHANS` | 512 | Maximum detector channels |

---

### 4.5 `bilinear_fwd_proj` — Forward Projector

**File:** `rtl/bilinear_fwd_proj.sv`  
**Mirrors:** `pb_fwd_one_view()` in `parallel_beam_model.c`

Parallel-beam bilinear forward projector. One voxel (pi, s) processed per clock.

**Coordinate mapping:**
```
x = (col − cx) · Δv
y = (row − cy) · Δv
det_ch  = (x·cosθ + y·sinθ) / Δd_ch + (NDC−1)/2 + offset_ch
det_row = slice + offset_row
```

**Bilinear splat:** 4 detector cells, 4 clock cycles per voxel (2-bit counter `splat_cnt`).

**Output:** `det_wr_en_o`, `det_wr_addr_o[19:0]`, `det_wr_data_o` — accumulate into detector buffer.

---

### 4.6 `bilinear_back_proj` — Back Projector

**File:** `rtl/bilinear_back_proj.sv`  
**Mirrors:** `pb_back_one_view()` in `parallel_beam_model.c`

Parallel-beam bilinear back projector. `coeff_power_2_i = 1` squares coefficients for Hessian diagonal computation.

**Interface:** `pixel_valid_i`, `sino_rd_addr_o → sino_rd_data_i`, `recon_wr_en_o / recon_wr_data_o`.

4 interpolation cycles per pixel (2-bit `interp_cnt`).

---

### 4.7 `qggmrf_compute` — qGGMRF Prior Engine

**File:** `rtl/qggmrf_compute.sv`  
**Mirrors:** `qggmrf_grad_hess_subset()` in `vcd_utils.c`

Computes prior gradient and Hessian for one voxel by iterating over all 26 neighbours.

**Algorithm per neighbour k:**
```
delta  = x_center − x_neighbour_k   (clamped to image boundary)
grad  += b[k] · ρ'(delta; σx, p, q, T)
hess  += b[k] · ρ''(delta; σx, p, q, T)
```

**States:** `GH_IDLE → GH_REQ → GH_WAIT → GH_DONE`

26 neighbour memory requests per voxel; external memory controller supplies values through `nbr_val_i / nbr_valid_i`.

---

### 4.8 `line_search` — Step Size Solver

**File:** `rtl/line_search.sv`  
**Mirrors:** VCD step-size computation in `vcd_utils.c`

Single-cycle computation of optimal VCD step size:
```
α = clip((fwd_lin − prior_lin) / (fwd_quad + prior_quad + ε),  ε,  1.5)
```

All inputs are pre-accumulated fp32 scalars from the subset update loop.

---

### 4.9 `vcd_controller` — VCD Reconstruction Controller

**File:** `rtl/vcd_controller.sv`  
**Mirrors:** `mj_recon()` in `tomography_model.c` and `vcd_subset_update()` in `vcd_utils.c`

Top-level VCD FSM. Orchestrates the complete MBIR reconstruction loop.

**States:**
```
IDLE → INIT → AUTO_REG → PART_GEN → SUBSET_START →
PRIOR_LOOP → FWD_HESS → FWD_GRAD → DELTA_CALC →
FWD_PROJ → LSEARCH → APPLY_UPD → UPD_SINO →
NMAE_CALC → NMAE_CHECK → ITER_NEXT → [DONE_ST]
```

**Memory interfaces:**
- Recon buffer (dual-port SRAM): `recon_rd_*` / `recon_wr_*`
- Error sinogram (dual-port SRAM): `esino_rd_*` / `esino_wr_*`
- Weight buffer (read-only): `wt_rd_*`

**Convergence:** Checks `NMAE% < stop_pct_i` (fp32 threshold) each iteration.

---

### 4.10 `fbp_controller` — FBP Controller

**File:** `rtl/fbp_controller.sv`  
**Mirrors:** `pb_fbp_recon()` in `parallel_beam_model.c`

Filtered Back-Projection for parallel beam.

**States:** `F_IDLE → F_FILTER → F_BACKPROJ_VIEW → F_BACKPROJ_PIX → F_SCALE → F_DONE`

**Pipeline:**
1. Stream sinogram row through `ramp_filter` instance.
2. Back-project filtered view via `bilinear_back_proj`.
3. Scale by `π / num_views`.

---

### 4.11 `fdk_controller` — FDK Controller

**File:** `rtl/fdk_controller.sv`  
**Mirrors:** `cb_fdk_recon()` in `cone_beam_model.c`

Feldkamp-Davis-Kress reconstruction for cone beam.

**Additional steps vs FBP:**
1. **Cosine weighting** before ramp filter: `w(u,v) = D_det / √(D_det² + u² + v²)`
2. **FDK back-projection weight**: `(D_iso / (D_iso + yr))²` per pixel

**States:** `FDK_IDLE → FDK_COSWEIGHT → FDK_FILTER → FDK_BACKPROJ → FDK_SCALE → FDK_DONE`

---

### 4.12 `shepp_logan_gen` — Phantom Generator

**File:** `rtl/shepp_logan_gen.sv`  
**Mirrors:** `phantom_shepp_logan()` in `phantom.c`

Streaming 3-D Shepp-Logan phantom generator. Pre-loaded ellipsoid ROM (10 entries × 8 fields).

For each voxel: evaluates all 10 ellipsoid membership tests, sums contributions, clips to [0,1].

**Output:** `voxel_valid_o`, `voxel_addr_o[19:0]`, `voxel_val_o` — 1 voxel/clock.

---

### 4.13 `preprocess_unit` — Sinogram Preprocessor

**File:** `rtl/preprocess_unit.sv`  
**Mirrors:** `preprocess.c`

Three-mode streaming preprocessor:

| `mode_i` | Operation | C function |
|----------|-----------|-----------|
| 0 | Transmission → attenuation: `out = -ln(max(in/air, 1e-6))` | `mj_transmission_to_attenuation` |
| 1 | Poisson weights: `out = exp(-|in|)` | `mj_compute_weights` |
| 2 | Bad-pixel correction (5-sample shift register median) | `mj_correct_bad_pixels` |

Modes 0/1: 1 sample/clock. Mode 2: buffered, latency = 3 clocks.

---

### 4.14 `denoising_unit` — MAP Denoiser + Median Filter

**File:** `rtl/denoising_unit.sv`  
**Mirrors:** `denoising.c` (mj_denoise + mj_median_filter3d)

Selects between two algorithms via `mode_i`:

**Mode 0 — MAP denoiser (qGGMRF + VCD, A=I):**  
States: `D_IDLE → D_NOISE_EST → D_REG → D_OUTER → D_SUBSET → D_PRIOR → D_DELTA → D_APPLY → D_SINO_UPD → D_NMAE → D_DONE`

Key computation:
```
sigma_x = sigma_noise * 2^(-sharpness)
fm_const = 1 / sigma_y²
Δx = -(fwd_grad + prior_grad) / (fm_const + prior_hess + ε)
```

**Mode 1 — 3×3×3 Median filter:**  
States: `M_GATHER → M_SORT → M_EMIT → M_DONE`  
27-element bubble sort; outputs `med_win[13]` (the median).

---

### 4.15 `mbirjax_top` — Top-Level Integration

**File:** `rtl/mbirjax_top.sv`

Top-level module instantiating all sub-modules with a shared SRAM address multiplexer.

**Operation mode dispatch:**

| `op_mode_i` | Operation | Module |
|-------------|-----------|--------|
| 4'd0 | MBIR VCD reconstruction | `vcd_controller` |
| 4'd1 | FBP (parallel beam) | `fbp_controller` |
| 4'd2 | FDK (cone beam) | `fdk_controller` |
| 4'd3 | MAP denoising | `denoising_unit (mode=0)` |
| 4'd4 | Median filter | `denoising_unit (mode=1)` |
| 4'd5 | Shepp-Logan phantom | `shepp_logan_gen` |

**Memory arbitration:** Single-master SRAM mux — only one operation runs at a time. All SRAMs are external (host-managed, DMA-loaded).

---

## 5. C-to-SystemVerilog Mapping

| C (mbirjax-c) | SystemVerilog (mbirjax-sv) |
|---|---|
| `ParamStore` struct + `ph_*()` | `param_store` module (register file) |
| `ph_init()` reset defaults | `always_ff rst_n` loads defaults |
| `MjModel` struct | Port signals + `mbirjax_top` wires |
| `mj_recon()` VCD outer loop | `vcd_controller` FSM |
| `vcd_subset_update()` | `vcd_controller` SUBSET_* states |
| `qggmrf_grad_hess_subset()` | `qggmrf_compute` module |
| `qggmrf_rho_prime()` | `qggmrf_rho_prime()` package function |
| `qggmrf_rho_dprime()` | `qggmrf_rho_dprime()` package function |
| `pb_fwd_one_view()` | `bilinear_fwd_proj` module |
| `pb_back_one_view()` | `bilinear_back_proj` module |
| `apply_ramp_1d()` | `ramp_filter` module |
| `pb_fbp_recon()` | `fbp_controller` FSM |
| `cb_fdk_recon()` | `fdk_controller` FSM |
| `mj_gen_ror_mask()` | `ror_mask_gen` module |
| `vcd_partition_create()` | `vcd_controller` subset indexing |
| `mj_clipf(α, ε, 1.5)` | `mj_clipf()` package function |
| `mj_denoise()` | `denoising_unit (mode=0)` |
| `mj_median_filter3d()` | `denoising_unit (mode=1)` |
| `phantom_shepp_logan()` | `shepp_logan_gen` module |
| `mj_transmission_to_attenuation()` | `preprocess_unit (mode=0)` |
| `mj_compute_weights()` | `preprocess_unit (mode=1)` |
| `mj_correct_bad_pixels()` | `preprocess_unit (mode=2)` |
| `for` loops over array | FSM states + counter registers |
| `calloc` / `malloc` / `free` | External SRAM (no heap) |
| `float` arithmetic | `real` in pkg functions; fp32 on ports |

---

## 6. Simulation

### Xilinx Vivado xsim

```bash
cd mbirjax_sv_final
xvlog --sv -f filelist.f --work work
xelab -debug all tb_mbirjax_top -s tb_snap
xsim tb_snap --runall
```

### Synopsys VCS

```bash
vcs -sverilog -f filelist.f -o simv -debug_pp
./simv
```

### Icarus Verilog (basic syntax check)

```bash
iverilog -g2012 -f filelist.f -o sim
vvp sim
```

### Expected testbench output

```
=== TEST 1: Param store reset defaults ===
  PASS: p default = 0x3F99999A (1.2)
  PASS: q default = 0x40000000 (2.0)
=== TEST 2: Param store write/read ===
  PASS: sigma_y readback correct
=== TEST 3: VCD start/done ===
  PASS: VCD op_done asserted within timeout
=== TEST 4: FBP start/done ===
  PASS: FBP op_done asserted
=== TEST 5: Shepp-Logan generator ===
  PASS: Shepp-Logan gen done
=== TEST 6: Denoising unit start/done ===
  PASS: Denoising done
=== TEST 7: Package math functions ===
  PASS: rho_prime(0.5)=... finite
  PASS: rho_dprime(0.5)=... positive and finite
=== TEST 8: fp32 conversion round-trip ===
  PASS: pi round-trip: in=3.14159265 out=3.14159012

=====================================
  Results:  10 PASS   0 FAIL
=====================================
  ALL TESTS PASSED
```

---

## 7. Synthesis Notes

### Floating-point arithmetic

All `real` computations inside `automatic` functions synthesise to floating-point IP cores on target FPGAs:
- Xilinx: **Floating Point v7.1** (add, mul, div, sqrt, log, exp)
- Intel: **ALTFP** megafunction
- ASIC: instantiate IEEE-754 FPU cells from the standard cell library

### Memory mapping

| Buffer | Size (typical) | Type |
|--------|---------------|------|
| Sinogram | 360 × 256 × 512 × 4B = 188 MB | External DDR4 |
| Recon | 512 × 512 × 256 × 4B = 268 MB | External DDR4 |
| Weights | Same as sinogram | External DDR4 |
| Param store | 32 × 4B = 128B | On-chip register file |
| Ramp filter coeff ROM | 64 × 4B = 256B | On-chip BRAM |
| Shepp-Logan ROM | 10 × 8 × 4B = 320B | On-chip LUTs |

### Timing

The critical path is inside `qggmrf_compute` (iterative neighbour loop with transcendental functions). For timing closure, pipeline the `powf` computation:
- Replace `x**p` with `exp(p * log(x))` in 2 cycles (LUT + multiply).
- Add a single register stage between the 26-neighbour accumulation and the output register.

---

## 8. Verification

### Structural checks (automated)

The Python checker at delivery time verified:

| Check | Result |
|-------|--------|
| `module`/`endmodule` balance (16 files) | ✅ PASS |
| `package`/`endpackage` balance | ✅ PASS |
| `endmodule : name` label matches `module name` | ✅ PASS |
| `import mbirjax_pkg::*` present in all RTL | ✅ PASS |
| `` `timescale `` directive in all files | ✅ PASS |
| No `reg` keyword (SV `logic` used throughout) | ✅ PASS |

**Total: 16 files, 0 errors, 0 warnings.**

### Testbench coverage (tb_mbirjax_top.sv)

| Test | Stimulus | Checks |
|------|----------|--------|
| 1 | Hard reset | param_store defaults (p=1.2, q=2.0) |
| 2 | Write sigma_y | Readback == written value |
| 3 | VCD op_start | op_done_o asserted within 10 000 clocks |
| 4 | FBP op_start | op_done_o asserted |
| 5 | Shepp-Logan start | op_done_o asserted |
| 6 | Denoise start | op_done_o asserted |
| 7 | Package functions | rho_prime(0.5) finite, rho_dprime(0.5) > 0 |
| 8 | fp32 round-trip | \|real_to_fp32 → fp32_to_real − π\| < 1e-5 |

