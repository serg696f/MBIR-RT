# mbir-c — Reference Documentation

Plain-C (C99) translation of the [mbirjax](https://github.com/cabouman/mbirjax) Python package for Model-Based Iterative Reconstruction (MBIR) of tomographic data.

**No external library dependencies** — only the C99 standard library and `libm`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Build](#2-build)
3. [Project Structure](#3-project-structure)
4. [Architecture & Design Decisions](#4-architecture--design-decisions)
5. [Module Reference](#5-module-reference)
   - 5.1 [mbirjax_types.h — Shared Types](#51-mbirjax_typesh--shared-types)
   - 5.2 [parameter_handler — ParameterHandler](#52-parameter_handler--parameterhandler)
   - 5.3 [projector_params — Geometry Packing](#53-projector_params--geometry-packing)
   - 5.4 [tomography_model — TomographyModel](#54-tomography_model--tomographymodel)
   - 5.5 [vcd_utils — VCD Solver](#55-vcd_utils--vcd-solver)
   - 5.6 [parallel_beam_model — ParallelBeamModel](#56-parallel_beam_model--parallelbeammodel)
   - 5.7 [cone_beam_model — ConeBeamModel](#57-cone_beam_model--conebeammodel)
   - 5.8 [denoising — QGGMRFDenoiser](#58-denoising--qggmrfdenoiser)
   - 5.9 [phantom — Phantom Generation](#59-phantom--phantom-generation)
   - 5.10 [preprocess — Sinogram Pre-processing](#510-preprocess--sinogram-pre-processing)
   - 5.11 [utils — Utility Functions](#511-utils--utility-functions)
6. [Python-to-C Mapping](#6-python-to-c-mapping)
7. [Algorithm Notes](#7-algorithm-notes)
8. [Usage Examples](#8-usage-examples)

---

## 1. Overview

mbirjax-c is a function-by-function, file-by-file plain-C translation of the mbirjax Python package. It implements:

- **MAP reconstruction** using Multi-Granular Vectorized Coordinate Descent (VCD) with a qGGMRF prior.
- **Parallel-beam geometry** with bilinear projectors and Ram-Lak filtered back-projection (FBP).
- **Cone-beam geometry** with bilinear projectors and Feldkamp-Davis-Kress (FDK) reconstruction.
- **qGGMRF MAP denoiser** with automatic noise estimation and 3×3×3 median filter.
- **Sinogram pre-processing** (transmission → attenuation, weights, bad-pixel correction).
- **3-D Shepp-Logan phantom** generator.
- **Parameter store** with file persistence, mirroring Python's `ParameterHandler`.

| Metric | Value |
|--------|-------|
| Source files | 9 `.c` files, 11 `.h` files |
| Total lines | ~2 800 |
| External dependencies | none (libm only) |
| C standard | C99 |
| Output | `libmbirjax.a` (static library) |

---

## 2. Build

### Requirements

- GCC or Clang with C99 support
- `make`
- `libm` (standard, always present)

### Commands

```bash
# Build the static library
make

# Build + run the denoising demo
make demo
./build/denoising_demo

# Install to /usr/local (headers + library)
sudo make install

# Clean
make clean
```

### Manual compilation

```bash
# Single-file compile
gcc -std=c99 -O2 -Iinclude src/parameter_handler.c src/vcd_utils.c \
    src/tomography_model.c src/parallel_beam_model.c \
    src/phantom.c my_app.c -o my_app -lm

# Link against the static library
gcc -std=c99 -O2 -Iinclude/mbirjax my_app.c -L/usr/local/lib -lmbirjax -lm -o my_app
```

---

## 3. Project Structure

```
mbirjax_c/
├── Makefile
├── include/
│   ├── mbirjax_types.h        ← primitive types, constants, shared structs
│   ├── parameter_handler.h    ← typed key-value parameter store
│   ├── projector_params.h     ← geometry packing struct + function pointer types
│   ├── tomography_model.h     ← base model API (MjModel struct)
│   ├── vcd_utils.h            ← VCD solver internals
│   ├── parallel_beam_model.h  ← parallel-beam geometry API
│   ├── cone_beam_model.h      ← cone-beam geometry API
│   ├── denoising.h            ← qGGMRF denoiser + median filter
│   ├── phantom.h              ← phantom generation
│   ├── preprocess.h           ← sinogram preprocessing
│   └── utils.h                ← I/O and utility helpers
└── src/
    ├── parameter_handler.c    ← 407 lines
    ├── tomography_model.c     ← 410 lines
    ├── vcd_utils.c            ← 299 lines
    ├── parallel_beam_model.c  ← 228 lines
    ├── cone_beam_model.c      ← 296 lines
    ├── denoising.c            ← 185 lines
    ├── phantom.c              ←  77 lines
    ├── preprocess.c           ←  59 lines
    └── utils.c                ←  83 lines
```

---

## 4. Architecture & Design Decisions

### Python Class → C Struct + Functions

Python's class hierarchy maps to a single `MjModel` struct with function pointers:

| Python | C |
|--------|---|
| `ParameterHandler` class | `ParamStore` struct + `ph_*()` functions |
| `TomographyModel` class | `MjModel` struct + `mj_*()` functions |
| `ParallelBeamModel(TomographyModel)` | sets `m->fwd_fn = pb_fwd_one_view` |
| `ConeBeamModel(TomographyModel)` | sets `m->fwd_fn = cb_fwd_one_view` |
| `jax.jit` / `lax.while_loop` | plain `for` loops |
| `jnp` array ops | flat `f32[]` with for loops |

### Image Storage Layout

Images are stored as flat `[total_pixels × num_slices]` arrays:

```
recon[( row*num_cols + col ) * num_slices + slice]
```

This mirrors the Python code's `reshape((-1, image_shape[-1]))` pattern.

### Sinogram Layout

```
sino[view * (det_rows * det_channels) + det_row * det_channels + det_channel]
```

### No HDF5

`mj_save_recon()` / `mj_load_recon()` use a simple raw binary format:
`3×i32` header (rows, cols, slices), then `f32[]` data. This replaces the Python HDF5 dependency.

### No YAML

`ph_save()` / `ph_load()` use a plain `key type value` text format instead of YAML.

---

## 5. Module Reference

---

### 5.1 `mbirjax_types.h` — Shared Types

All primitive aliases, compile-time constants, and shared structs used throughout the library. Mirrors `mbirjax/_utils.py`.

#### Type aliases

| Alias | Underlying type | Notes |
|-------|----------------|-------|
| `f32` | `float` | 32-bit single-precision |
| `f64` | `double` | 64-bit double |
| `i32` | `int` | 32-bit signed integer |
| `u32` | `unsigned int` | 32-bit unsigned |

#### Shape descriptors

```c
typedef struct { i32 views; i32 rows; i32 channels; } SinoShape;
typedef struct { i32 rows;  i32 cols; i32 slices;   } ReconShape;

i32 sino_numel (SinoShape s);   /* views * rows * channels */
i32 recon_numel(ReconShape s);  /* rows * cols * slices    */
```

#### Parameter type tag

```c
typedef enum {
    PT_NONE, PT_FLOAT, PT_INT, PT_BOOL,
    PT_SHAPE,   /* i32[3] — sinogram_shape or recon_shape */
    PT_STR,     /* heap char* */
    PT_FARR,    /* heap f32[] */
    PT_IARR     /* heap i32[] */
} ParamType;
```

#### `Param` slot

```c
typedef struct {
    char      key[64];
    ParamType type;
    int       recompile_flag;  /* 1 → projectors need rebuild on change */
    union { f64 fval; i32 ival; i32 bval; i32 shape[3];
            char *sval;
            struct { f32 *data; i32 n; } farr;
            struct { i32 *data; i32 n; } iarr; } v;
} Param;
```

#### `ParamStore`

```c
typedef struct { Param slots[MJ_MAX_PARAMS]; i32 n; } ParamStore;
```

#### Key constants

| Constant | Value | Notes |
|----------|-------|-------|
| `MJ_DEF_P` | 1.2 | qGGMRF shape exponent |
| `MJ_DEF_Q` | 2.0 | qGGMRF quadratic region |
| `MJ_DEF_T` | 1.0 | qGGMRF threshold |
| `MJ_DEF_SHARPNESS` | 0.0 | regularisation strength |
| `MJ_DEF_SNR_DB` | 30.0 | signal-to-noise ratio |
| `MJ_DEF_GRANULARITY` | 16 | VCD subset size |
| `MJ_N_NBR` | 26 | 3-D 26-neighbourhood |
| `MJ_EPS` | 1.19e-7 | `FLT_EPSILON` |

#### Inline math helpers

```c
f32 mj_absf (f32 x)
f32 mj_minf (f32 a, f32 b)
f32 mj_maxf (f32 a, f32 b)
f32 mj_clipf(f32 x, f32 lo, f32 hi)
i32 mj_clamp(i32 v, i32 lo, i32 hi)
f32 mj_sum_abs(const f32 *v, i32 n)
f32 mj_dot   (const f32 *a, const f32 *b, i32 n)
```

---

### 5.2 `parameter_handler` — ParameterHandler

Mirrors `mbirjax/parameter_handler.py` (`ParameterHandler` class).

Provides a typed key-value parameter store with file persistence and auto-regularisation helpers.

#### `ph_init`

```c
void ph_init(ParamStore *ps);
```

Initialise `ps` with all library defaults. Mirrors `ParameterHandler.__init__()`. Must be called before any other `ph_*` function.

Default values populated:

| Key | Default | recompile |
|-----|---------|-----------|
| `geometry_type` | 0 (NONE) | yes |
| `sinogram_shape` | (0,0,0) | yes |
| `recon_shape` | (0,0,0) | yes |
| `delta_det_channel` | 1.0 | yes |
| `delta_det_row` | 1.0 | yes |
| `det_row_offset` | 0.0 | yes |
| `det_channel_offset` | 0.0 | yes |
| `delta_voxel` | 1.0 | yes |
| `sigma_y` | 0.0 | no |
| `sigma_x` | 0.0 | no |
| `sigma_prox` | 0.0 | no |
| `p` | 1.2 | no |
| `q` | 2.0 | no |
| `T` | 1.0 | no |
| `sharpness` | 0.0 | no |
| `snr_db` | 30.0 | no |
| `auto_regularize_flag` | true | no |
| `positivity_flag` | false | no |
| `qggmrf_nbr_wts` | ones[26] | no |
| `granularity` | [16] | no |
| `verbose` | 1 | no |

#### `ph_free`

```c
void ph_free(ParamStore *ps);
```

Release all heap memory owned by parameters in `ps`. Safe to call on an uninitialised store.

#### `ph_copy`

```c
void ph_copy(ParamStore *dst, const ParamStore *src);
```

Deep-copy all parameters from `src` to `dst`. `dst` must be uninitialised or freshly freed.

#### Typed setters

```c
int ph_set_float(ParamStore *ps, const char *key, f64 val, int recompile);
int ph_set_int  (ParamStore *ps, const char *key, i32 val, int recompile);
int ph_set_bool (ParamStore *ps, const char *key, int val, int recompile);
int ph_set_shape(ParamStore *ps, const char *key, i32 a, i32 b, i32 c, int recompile);
int ph_set_str  (ParamStore *ps, const char *key, const char *val, int recompile);
int ph_set_farr (ParamStore *ps, const char *key, const f32 *data, i32 n, int recompile);
int ph_set_iarr (ParamStore *ps, const char *key, const i32 *data, i32 n, int recompile);
```

All setters create the key if absent, update it if present. Return 0 on success, -1 if the store is full (`MJ_MAX_PARAMS = 80`). Mirrors `ParameterHandler.set_params()`.

#### Typed getters

```c
const Param *ph_find     (const ParamStore *ps, const char *key);
f64          ph_get_float(const ParamStore *ps, const char *key);
i32          ph_get_int  (const ParamStore *ps, const char *key);
int          ph_get_bool (const ParamStore *ps, const char *key);
int          ph_get_shape(const ParamStore *ps, const char *key, i32 out[3]);
const char  *ph_get_str  (const ParamStore *ps, const char *key);
int          ph_get_farr (const ParamStore *ps, const char *key, f32 *out, i32 *n_out);
```

`ph_find` returns NULL if the key is absent. All typed getters print an error and return 0 / NULL on missing or type-mismatched keys. Mirrors `ParameterHandler.get_params()`.

#### `ph_print`

```c
void ph_print(const ParamStore *ps);
```

Print all parameters to stdout. Mirrors `ParameterHandler.print_params()`.

#### `ph_save` / `ph_load`

```c
int ph_save(const ParamStore *ps, const char *filepath);
int ph_load(ParamStore *ps,       const char *filepath);
```

Persist parameters to / from a plain text file (`key type value` format). Mirrors `save_params()` / `load_param_dict()` without YAML dependency.

File format example:
```
sinogram_shape shape 180 64 256
delta_det_channel float 0.01
p float 1.2
granularity iarr 1 16
```

#### `ph_auto_regularize`

```c
void ph_auto_regularize(ParamStore *ps, f32 typical_value);
```

Compute `sigma_y` and `sigma_x` from `snr_db`, `sharpness`, `delta_det_channel`, and `delta_voxel`. Mirrors `TomographyModel.auto_set_regularization_params()`.

```
sigma_y = typical_value / 10^(snr_db / 20)
sigma_x = typical_value * (delta_det_channel / delta_voxel) * 2^(-sharpness)
```

---

### 5.3 `projector_params` — Geometry Packing

Mirrors the `ProjectorParams` namedtuple in Python.

#### `ProjParams`

```c
typedef struct {
    SinoShape  sino;            /* (views, det_rows, det_channels) */
    ReconShape recon;           /* (rows, cols, slices)            */
    f32 delta_det_channel;
    f32 delta_det_row;
    f32 det_row_offset;
    f32 det_channel_offset;
    f32 delta_voxel;
    f32 source_iso_dist;        /* cone beam; 0 for parallel beam  */
    f32 source_det_dist;        /* cone beam; 0 for parallel beam  */
} ProjParams;
```

#### Function pointer types

```c
typedef void (*FwdFn)(
    const f32 *voxel_values, const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params, const ProjParams *pp, f32 *det_out);

typedef void (*BackFn)(
    const f32 *sino_view, const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params, const ProjParams *pp,
    i32 coeff_power, f32 *recon_out);
```

`FwdFn` and `BackFn` are set by the geometry module during model creation and called by the VCD loop. `coeff_power = 2` squares coefficients for Hessian diagonal computation.

---

### 5.4 `tomography_model` — TomographyModel

Mirrors `mbirjax/tomography_model.py`. Implements all geometry-agnostic methods.

#### `MjModel` struct

```c
typedef struct MjModel {
    ParamStore  ps;          /* all model parameters                */
    ViewParams  vp;          /* per-view parameters (angles, etc.)  */
    ProjParams  pp;          /* packed geometry for projector calls  */
    FwdFn       fwd_fn;      /* geometry-specific forward projector  */
    BackFn      back_fn;     /* geometry-specific back projector     */
    int         use_ror_mask;/* 1 = restrict to inscribed circle     */
    int         verbose;
} MjModel;
```

#### Lifecycle

```c
void mj_model_init(MjModel *m, i32 num_views, i32 num_det_rows, i32 num_det_channels);
void mj_model_free(MjModel *m);
void mj_sync_proj_params(MjModel *m);
```

`mj_model_init` calls `ph_init`, sets the sinogram shape, and calls `mj_auto_set_recon_geometry`. `mj_sync_proj_params` must be called after any geometry parameter change to propagate values to `m->pp`.

#### Geometry helpers

```c
void mj_auto_set_recon_geometry(MjModel *m);
void mj_scale_recon_shape(MjModel *m,
                           f32 row_scale, f32 col_scale, f32 slice_scale,
                           i32 added_out[3]);
```

Mirrors `TomographyModel.auto_set_recon_geometry()` and `scale_recon_shape()`. Default geometry: `recon_shape = (det_channels, det_channels, det_rows)`.

#### ROR mask

```c
void mj_gen_ror_mask(i32 num_rows, i32 num_cols, i32 *out_indices, i32 *out_n);
```

Fills `out_indices` with flat `(row*cols+col)` indices that lie inside the inscribed circle of the `rows × cols` grid. Mirrors `gen_full_indices(use_ror_mask=True)`.

#### Projection

```c
void mj_forward_project(MjModel *m, const f32 *recon, f32 *sino);
void mj_back_project   (MjModel *m, const f32 *sino,  f32 *recon);
void mj_hessian_diag   (MjModel *m, const i32 *pixel_indices, i32 n_pix, f32 *hess_out);
```

`mj_forward_project` and `mj_back_project` iterate over all views, calling `fwd_fn` / `back_fn` per view. `mj_hessian_diag` back-projects all-ones sinograms with `coeff_power = 2`. Mirrors `TomographyModel.forward_project()`, `back_project()`.

#### `mj_recon`

```c
int mj_recon(MjModel *m,
             const f32 *sino,
             const f32 *weights,        /* NULL → all ones            */
             const f32 *init_recon,     /* NULL → call mj_direct_recon */
             i32 max_iterations,
             f32 stop_pct,             /* stop when NMAE% < this     */
             i32 first_iter,
             f32 *recon_out,
             ReconStats *stats_out);   /* may be NULL                 */
```

MBIR reconstruction via Multi-Granular VCD. Mirrors `TomographyModel.recon()`.

**Algorithm:**
1. Auto-regularise if `auto_regularize_flag = true`.
2. Initialise reconstruction via `init_recon` or `mj_direct_recon()`.
3. Compute initial error sinogram: `e = sino - A·recon`.
4. Outer VCD loop (up to `max_iterations`):
   - Partition pixels into subsets of size `granularity`.
   - Call `vcd_subset_update()` for each subset.
   - Compute NMAE; stop early when NMAE% < `stop_pct`.
5. Optionally fill `ReconStats`.

#### `mj_direct_recon`

```c
void mj_direct_recon(MjModel *m, const f32 *sino, f32 *recon_out);
```

Base implementation: unweighted back-projection normalised by number of views. Geometry modules override this (FBP for parallel, FDK for cone) by calling their own filter+backproject function before running VCD.

#### `mj_prox_map`

```c
int mj_prox_map(MjModel *m,
                const f32 *prox_input,
                const f32 *sino,
                f32 sigma_prox,
                const f32 *weights,
                const f32 *init_recon,
                i32 max_iterations,
                f32 stop_pct,
                f32 *recon_out,
                ReconStats *stats_out);
```

Plug-and-Play proximal map. Mirrors `TomographyModel.prox_map()`. Equivalent to `recon()` initialised from `prox_input` with the sigma_prox parameter set.

#### Save / Load

```c
int mj_save_recon(const char *path, const f32 *recon, ReconShape rs);
int mj_load_recon(const char *path, f32 **recon_out, ReconShape *rs_out);
```

Binary format: `3 × i32` (rows, cols, slices), then `f32[]`. Caller must `free(*recon_out)`. Replaces the Python HDF5 save/load.

#### Phantom

```c
void mj_gen_shepp_logan(ReconShape rs, f32 *out);
```

Delegates to `phantom_shepp_logan()`. Mirrors `gen_modified_3d_sl_phantom()`.

---

### 5.5 `vcd_utils` — VCD Solver

Mirrors the VCD internals distributed across `mbirjax/tomography_model.py`.

#### `QGGMRFParams`

```c
typedef struct {
    f32 sigma_x;
    f32 p, q, T;
    f32 b[MJ_N_NBR];   /* neighbour weights; fill via vcd_init_nbr_weights */
} QGGMRFParams;
```

#### `vcd_init_nbr_weights`

```c
void vcd_init_nbr_weights(QGGMRFParams *qp);
```

Fill `b[]` with distance-based weights for the 26-neighbourhood. Face neighbours (distance²=1) get weight 1.0; edge neighbours get 1/√2; corner neighbours get 1/√3. Mirrors `get_b_from_nbr_wts()`.

#### `qggmrf_rho_prime` / `qggmrf_rho_dprime`

```c
f32 qggmrf_rho_prime (f32 delta, const QGGMRFParams *qp);
f32 qggmrf_rho_dprime(f32 delta, const QGGMRFParams *qp);
```

First and second derivatives of the qGGMRF potential:

```
ρ(t) = |t/σx|^p / (p · (1 + |t/(T·σx)|^(p-q)))

ρ'(δ) = sign(δ)/σx · |u|^(p-1) · [1/D − (p-q)/p · |v|^(p-q) / D²]
         where D = 1 + |v|^(p-q),  u = δ/σx,  v = δ/(T·σx)

ρ''(δ) ≈ (ρ'(δ+h) − ρ'(δ−h)) / (2h)    [central finite-difference]
```

#### `qggmrf_grad_hess_subset`

```c
void qggmrf_grad_hess_subset(
    const f32 *flat_recon, ReconShape recon_shape,
    const i32 *pixel_indices, i32 n_pix,
    const QGGMRFParams *qp,
    f32 *out_grad, f32 *out_hess);
```

For each `(pixel_indices[pi], slice s)`:
```
grad[pi*NS+s] = Σ_k  b[k] · ρ'(x_centre − x_neighbour_k)
hess[pi*NS+s] = Σ_k  b[k] · ρ''(x_centre − x_neighbour_k)
```
Neighbours outside image boundaries are clamped (edge-replication). Mirrors `qggmrf_gradient_and_hessian_at_indices()`.

#### `vcd_subset_update`

```c
void vcd_subset_update(
    f32 *flat_recon, f32 *error_sino,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *eff_weights,
    const QGGMRFParams *qp, const ProjParams *pp, MjModel *model,
    f32 *out_ell1, f32 *out_alpha);
```

One VCD sweep over a pixel subset. Mirrors `TomographyModel.vcd_subset_updater()`.

Steps:
1. Prior gradient + hessian via `qggmrf_grad_hess_subset`.
2. Forward hessian diagonal via squared back-projection.
3. Forward gradient via `−AᵀWe` (back-project weighted error sinogram).
4. Update direction: `Δx = −(fwd_grad + prior_grad) / (fwd_hess + prior_hess)`.
5. Line search for optimal step α, clipped to `[ε, 1.5]`.
6. Apply `x += α·Δx`, update error sinogram `e -= α·A·Δx`.

#### Partition helpers

```c
typedef struct { const i32 *indices; i32 n; } VSubset;
typedef struct { VSubset *subsets;   i32 n_subsets; } VPartition;

VPartition vcd_partition_create(const i32 *pixel_indices, i32 n_pix, i32 gran);
void       vcd_partition_free  (VPartition *vp);
```

Divide `pixel_indices[0..n_pix)` into contiguous subsets of size `gran`. Mirrors `gen_set_of_pixel_partitions()`.

---

### 5.6 `parallel_beam_model` — ParallelBeamModel

Mirrors `mbirjax/parallel_beam_model.py`.

#### `pb_model_create`

```c
void pb_model_create(MjModel *m,
                     i32 views, i32 det_rows, i32 det_chans,
                     const f32 *angles);
```

Initialise a parallel-beam model. Sets `fwd_fn = pb_fwd_one_view`, `back_fn = pb_back_one_view`, stores angles in `m->vp`. Auto-sets `recon_shape = (det_chans, det_chans, det_rows)` and `delta_voxel = delta_det_channel`.

#### Coordinate mapping (parallel beam)

```
x = (col − (NC−1)/2) · Δv
y = (row − (NR−1)/2) · Δv
det_channel ← (x cos θ + y sin θ) / Δd_ch  +  (NDC−1)/2  +  offset_ch
det_row     ←  slice                          +  offset_row
```

#### `pb_fwd_one_view`

```c
void pb_fwd_one_view(
    const f32 *voxel_values, const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params, const ProjParams *pp, f32 *det_out);
```

Bilinear splat of voxel values onto the detector for one view. `view_params[0] = θ` (angle in radians). `det_out` is accumulated (add, do not zero).

#### `pb_back_one_view`

```c
void pb_back_one_view(
    const f32 *sino_view, const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params, const ProjParams *pp,
    i32 coeff_power, f32 *recon_out);
```

Bilinear interpolation from the detector to voxel cylinders. With `coeff_power = 2`, coefficients are squared (for Hessian diagonal).

#### `pb_fbp_recon`

```c
void pb_fbp_recon(MjModel *m, const f32 *sino, f32 *recon_out);
```

Filtered Back-Projection. Mirrors `ParallelBeamModel.fbp_recon()`.

Steps:
1. Apply Ram-Lak ramp filter row-by-row to the sinogram:
   ```
   h[0]       = 1/4
   h[k odd]   = -1 / (k²π²)
   h[k even]  = 0
   ```
2. Back-project the filtered sinogram.
3. Scale by `π / num_views`.

---

### 5.7 `cone_beam_model` — ConeBeamModel

Mirrors `mbirjax/cone_beam_model.py`.

#### `cb_model_create`

```c
void cb_model_create(MjModel *m,
                     i32 views, i32 det_rows, i32 det_chans,
                     const f32 *angles,
                     f32 source_iso_dist, f32 source_det_dist);
```

Initialise a cone-beam model. Stores source-to-isocenter and source-to-detector distances. Auto-sets `delta_voxel = delta_det_channel / magnification` where `magnification = source_det_dist / source_iso_dist`.

#### `cb_get_magnification`

```c
f32 cb_get_magnification(const MjModel *m);
```

Returns `source_det_dist / source_iso_dist`.

#### Coordinate mapping (cone beam, flat detector)

```
xr =  xw cos θ + yw sin θ    (lateral in rotated frame)
yr = −xw sin θ + yw cos θ    (depth toward detector)
t  =  D_iso + yr

u  = xr / t · D_det          (detector channel physical coord)
v  = zw / t · D_det          (detector row physical coord)

det_channel ← u / Δd_ch  +  (NDC−1)/2  +  offset_ch
det_row     ← v / Δd_row +  (NDR−1)/2  +  offset_row
```

#### `cb_fdk_recon`

```c
void cb_fdk_recon(MjModel *m, const f32 *sino, f32 *recon_out);
```

Feldkamp-Davis-Kress reconstruction. Mirrors `ConeBeamModel.fdk_recon()`.

Steps:
1. Cosine-weight each sinogram sample: `w(u,v) = D_det / √(D_det²+u²+v²)`.
2. Apply Ram-Lak ramp filter per detector row.
3. Weighted back-projection with FDK weight `(D_iso / (D_iso + yr))²`.
4. Scale by `π / num_views`.

---

### 5.8 `denoising` — QGGMRFDenoiser

Mirrors `mbirjax/denoising.py`.

#### `DenoiserParams`

```c
typedef struct {
    i32 nz, ny, nx;
    f32 sigma_noise;        /* 0 → auto-estimate from image   */
    f32 sigma_y;            /* filled by mj_denoise           */
    f32 sigma_x;            /* filled by mj_denoise           */
    f32 sharpness;          /* regularisation strength         */
    f32 p, q, T;            /* qGGMRF shape parameters        */
    i32 max_iterations;
    f32 stop_threshold_pct;
    i32 granularity;        /* rows per VCD subset            */
    i32 verbose;
} DenoiserParams;
```

Default values (from `denoiser_default_params`):

| Parameter | Default |
|-----------|---------|
| `sigma_noise` | 0 (auto) |
| `sharpness` | 0.0 |
| `p` | 1.2 |
| `q` | 2.0 |
| `T` | 1.0 |
| `max_iterations` | 15 |
| `stop_threshold_pct` | 0.2 |
| `granularity` | 16 |
| `verbose` | 1 |

#### `mj_denoise`

```c
void mj_denoise(const f32 *image, f32 *denoised, DenoiserParams *dp);
```

MAP denoiser (AWGN likelihood + qGGMRF prior, solved via VCD with identity forward model A = I). Mirrors `QGGMRFDenoiser.denoise()`.

Steps:
1. Auto-estimate `sigma_noise` if 0.
2. `sigma_y = sigma_noise`, `sigma_x = sigma_noise · 2^(−sharpness)`.
3. `fm_constant = 1 / sigma_y²`.
4. VCD outer loop over partition of flat rows.

#### `mj_estimate_image_noise_std`

```c
f32 mj_estimate_image_noise_std(const f32 *image, i32 nz, i32 ny, i32 nx);
```

Estimate noise standard deviation from 4-voxel neighbour groups. Mirrors `_get_estimate_of_recon_std()`.

#### `mj_median_filter3d`

```c
void mj_median_filter3d(const f32 *x, f32 *out, i32 nz, i32 ny, i32 nx,
                         int return_min_max, f32 *out_min, f32 *out_max);
```

27-point (3×3×3) median filter with edge-replication boundary conditions. Optionally fills `out_min` and `out_max` with neighbourhood min/max. Mirrors `median_filter3d()`.

---

### 5.9 `phantom` — Phantom Generation

#### `phantom_shepp_logan`

```c
void phantom_shepp_logan(i32 rows, i32 cols, i32 slices, f32 *out);
```

Generate a modified 3-D Shepp-Logan phantom (10 ellipsoids, low-dynamic-range values in `[0, 1]`). Mirrors `generate_3d_shepp_logan_low_dynamic_range()`.

Ellipsoid table (Shepp & Logan 1974, modified):

| # | Centre | Semi-axes | Angle | Value |
|---|--------|-----------|-------|-------|
| 1 | (0, 0, 0) | (0.69, 0.92, 0.90) | 0° | +1.00 |
| 2 | (0, −0.018, 0) | (0.662, 0.874, 0.880) | 0° | −0.80 |
| 3 | (0.22, 0, 0) | (0.11, 0.31, 0.22) | −18° | −0.20 |
| 4 | (−0.22, 0, 0) | (0.16, 0.41, 0.28) | +18° | −0.20 |
| 5–10 | (various) | (various) | 0° | +0.10 |

#### `phantom_flat_disk`

```c
void phantom_flat_disk(i32 rows, i32 cols, i32 slices, f32 value, f32 *out);
```

Inscribed-circle flat disk phantom. All voxels inside a circle of radius `0.45 · min(rows, cols)` are set to `value`.

---

### 5.10 `preprocess` — Sinogram Pre-processing

Mirrors `mbirjax/preprocess.py`.

#### `mj_transmission_to_attenuation`

```c
void mj_transmission_to_attenuation(const f32 *sino, const f32 *air_scan,
                                     i32 n, f32 *out);
```

Convert raw photon counts to attenuation: `out[i] = −log(max(sino[i]/air_scan[i], 1e-6))`. If `air_scan` is NULL, uses 1. Mirrors `transmission_CT_to_attenuation()`.

#### `mj_compute_weights`

```c
void mj_compute_weights(const f32 *atten, i32 n, f32 *out_weights);
```

Poisson model weights: `w[i] = exp(−|atten[i]|)`.

#### `mj_correct_bad_pixels`

```c
void mj_correct_bad_pixels(f32 *sino, i32 views, i32 det_rows, i32 det_chans);
```

Replace NaN, Inf, and negative sinogram values with the local channel median (±2-pixel window).

#### `mj_estimate_sino_noise_std`

```c
f32 mj_estimate_sino_noise_std(const f32 *sino, i32 n);
```

Mean absolute adjacent-channel difference scaled to noise std: `std ≈ mean|diff| / 0.7979`.

---

### 5.11 `utils` — Utility Functions

Mirrors `mbirjax/_utils.py` and utility functions scattered through the Python package.

| Function | Description |
|----------|-------------|
| `mj_makedirs(path)` | Create directories recursively (`mkdir -p`). |
| `mj_save_raw(path, data, dims, ndims)` | Save float array with ndims-header to binary file. |
| `mj_load_raw(path, &data, &dims, &ndims)` | Load float array (caller frees). |
| `mj_gen_full_indices(rows, cols, use_ror, out, &n)` | Flat pixel index list, optionally ROR-masked. |
| `mj_gen_partition_sequence(n_gran, max_iter, seq_out)` | Multi-granular VCD cycle sequence. |
| `mj_nmae(a, b, n)` | Normalised mean absolute error `‖a−b‖₁/‖b‖₁`. |
| `mj_lerp(a, b, t, out, n)` | Linear interpolation `out = a(1−t) + bt`. |
| `mj_get_memory_stats(print_results)` | Available RAM in bytes (Linux `/proc/meminfo`). |

---

## 6. Python-to-C Mapping

| Python (mbirjax) | C (mbirjax-c) |
|---|---|
| `ParameterHandler` class | `ParamStore` struct + `ph_*()` functions |
| `ParameterHandler.set_params(**kwargs)` | `ph_set_float/int/bool/shape/str/farr/iarr()` |
| `ParameterHandler.get_params(name)` | `ph_get_float/int/bool/shape/str/farr()` |
| `ParameterHandler.print_params()` | `ph_print()` |
| `ParameterHandler.save_params(file)` | `ph_save()` |
| `ParameterHandler.load_param_dict(file)` | `ph_load()` |
| `TomographyModel` class | `MjModel` struct |
| `TomographyModel.recon()` | `mj_recon()` |
| `TomographyModel.direct_recon()` | `mj_direct_recon()` |
| `TomographyModel.prox_map()` | `mj_prox_map()` |
| `TomographyModel.forward_project()` | `mj_forward_project()` |
| `TomographyModel.back_project()` | `mj_back_project()` |
| `TomographyModel.auto_set_recon_geometry()` | `mj_auto_set_recon_geometry()` |
| `TomographyModel.scale_recon_shape()` | `mj_scale_recon_shape()` |
| `TomographyModel.save_recon_hdf5()` | `mj_save_recon()` (raw binary) |
| `TomographyModel.load_recon_hdf5()` | `mj_load_recon()` |
| `TomographyModel.gen_modified_3d_sl_phantom()` | `mj_gen_shepp_logan()` |
| `forward_project_pixel_batch_to_one_view()` | `FwdFn` function pointer |
| `back_project_one_view_to_pixel_batch()` | `BackFn` function pointer |
| `vcd_subset_updater()` | `vcd_subset_update()` |
| `qggmrf_gradient_and_hessian_at_indices()` | `qggmrf_grad_hess_subset()` |
| `gen_set_of_pixel_partitions()` | `vcd_partition_create()` |
| `gen_full_indices(use_ror_mask=True)` | `mj_gen_ror_mask()` |
| `get_b_from_nbr_wts()` | `vcd_init_nbr_weights()` |
| `auto_set_regularization_params()` | `ph_auto_regularize()` |
| `ParallelBeamModel.fbp_recon()` | `pb_fbp_recon()` |
| `ConeBeamModel.fdk_recon()` | `cb_fdk_recon()` |
| `QGGMRFDenoiser.denoise()` | `mj_denoise()` |
| `median_filter3d()` | `mj_median_filter3d()` |
| `transmission_CT_to_attenuation()` | `mj_transmission_to_attenuation()` |
| `jnp.median` on 27-element stack | `qsort` on `f32 win[27]` |
| `jax.jit` / `lax.while_loop` | plain `for` / `while` loops |
| `jax.device_put` | no-op (CPU only) |
| `jnp.pad(..., mode='edge')` | `mj_clamp()` on neighbour access |
| `ruamel.yaml` | `ph_save()` / `ph_load()` (plain text) |
| `h5py` / HDF5 | raw binary `mj_save_recon()` |

---

## 7. Algorithm Notes

### qGGMRF Prior

The potential `ρ(t; σx, p, q, T)` interpolates between a Gaussian (`q = 2`) near the origin and a generalised Gaussian (`p < 2`) in the tails, controlled by threshold `T`. Smaller `p` preserves edges more aggressively. `σx` scales prior strength relative to data fidelity.

### Auto-regularisation

`σx = σnoise · (Δd_ch/Δv) · 2^(−sharpness)`. Increasing sharpness by 1 halves `σx`, weakening the prior and producing sharper output.

### VCD Convergence

NMAE `‖Δx‖₁ / ‖x‖₁` decreases monotonically. Default `stop_threshold_pct = 0.2` stops when updates drop below 0.2% of image norm per iteration.

### Ramp Filter

Direct-domain Ram-Lak kernel: `h[0] = ¼`, `h[k odd] = −1/(k²π²)`, `h[k even ≠ 0] = 0`. Applied by 1-D convolution along the channel axis.

### Boundary Conditions

All 3-D neighbour accesses (qGGMRF, median filter) use edge-replication via `mj_clamp()`, equivalent to NumPy's `mode='edge'`.

### Memory

`mj_recon()` allocates: one copy of the sinogram (error sinogram) plus two temporary arrays of pixel-subset size. `mj_denoise()` allocates two full-volume copies. `mj_median_filter3d()` uses 27 floats of stack space per voxel.

### Thread Safety

The global 26-neighbour offset table (in `vcd_utils.c`) is initialised once on first call. If multiple threads call `vcd_subset_update()` concurrently before any single-threaded warmup, add a mutex around `init_offsets()`. All other state is call-local.

---

## 8. Usage Examples

### Parallel-beam reconstruction from scratch

```c
#include "mbirjax.h"
#include <stdlib.h>
#include <math.h>

int main(void)
{
    /* Scanner geometry */
    int   num_views = 180, det_rows = 1, det_chans = 256;
    float angles[180];
    for (int v = 0; v < 180; v++) angles[v] = v * M_PI / 180.f;

    /* Create model */
    MjModel m;
    pb_model_create(&m, num_views, det_rows, det_chans, angles);
    m.verbose = 1;

    /* Generate phantom */
    int RS[3];
    ph_get_shape(&m.ps, "recon_shape", RS);
    int total_recon = RS[0]*RS[1]*RS[2];
    float *phantom = calloc(total_recon, sizeof(float));
    mj_gen_shepp_logan((ReconShape){RS[0], RS[1], RS[2]}, phantom);

    /* Forward project to get sinogram */
    int total_sino = num_views * det_rows * det_chans;
    float *sino    = calloc(total_sino, sizeof(float));
    mj_forward_project(&m, phantom, sino);

    /* Reconstruct */
    float *recon = calloc(total_recon, sizeof(float));
    mj_recon(&m, sino, NULL, NULL, 20, 0.2f, 0, recon, NULL);

    /* Save */
    mj_save_recon("recon.bin", recon, (ReconShape){RS[0], RS[1], RS[2]});

    pb_model_free(&m);
    free(phantom); free(sino); free(recon);
    return 0;
}
```

### Cone-beam FDK reconstruction

```c
MjModel m;
cb_model_create(&m, 360, 256, 512, angles,
                /*source_iso=*/500.f, /*source_det=*/1000.f);
float *recon = calloc(recon_numel(m.pp.recon), sizeof(float));
cb_fdk_recon(&m, sino, recon);
cb_model_free(&m);
```

### MAP denoising

```c
#include "denoising.h"

DenoiserParams dp;
denoiser_default_params(&dp, 64, 64, 64);
dp.sharpness = 1.0f;     /* sharper */
dp.max_iterations = 20;

float *denoised = malloc(64*64*64*sizeof(float));
mj_denoise(noisy_volume, denoised, &dp);
```

### Parameter management

```c
MjModel m;
pb_model_create(&m, 180, 1, 256, angles);

/* Set a parameter */
ph_set_float(&m.ps, "sharpness", 1.5, 0);
ph_set_float(&m.ps, "snr_db", 35.0, 0);

/* Read it back */
float sharp = (float)ph_get_float(&m.ps, "sharpness");

/* Print all */
ph_print(&m.ps);

/* Save / load */
ph_save(&m.ps, "model_params.txt");
ph_load(&m.ps, "model_params.txt");
```

---

*Generated from mbirjax-c source. All 9 `.c` files pass `gcc -std=c99 -O2 -Wall -Wextra` with zero warnings.*
