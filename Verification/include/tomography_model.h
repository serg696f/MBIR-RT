/**
 * tomography_model.h
 *
 * Plain-C translation of mbirjax/tomography_model.py (TomographyModel class).
 *
 * MjModel is the central struct; geometry-specific modules (parallel_beam,
 * cone_beam) set the fwd_fn / back_fn function pointers and call
 * mj_model_init() with the sinogram shape.
 *
 * Public API mirrors every public method in TomographyModel:
 *   mj_model_init()             — __init__
 *   mj_model_free()             — cleanup
 *   mj_auto_set_recon_geometry()— auto_set_recon_geometry
 *   mj_scale_recon_shape()      — scale_recon_shape
 *   mj_forward_project()        — forward_project
 *   mj_back_project()           — back_project
 *   mj_recon()                  — recon  (Multi-Granular VCD)
 *   mj_direct_recon()           — direct_recon  (geometry-specific FBP/FDK)
 *   mj_prox_map()               — prox_map
 *   mj_save_recon()             — save_recon_hdf5 (simplified binary)
 *   mj_load_recon()             — load_recon_hdf5 (simplified binary)
 *   mj_gen_shepp_logan()        — gen_modified_3d_sl_phantom
 *   mj_gen_ror_mask()           — gen_full_indices with use_ror_mask
 */
#ifndef TOMOGRAPHY_MODEL_H
#define TOMOGRAPHY_MODEL_H

#include "mbirjax_types.h"
#include "parameter_handler.h"
#include "projector_params.h"

/* =========================================================================
 * View parameters  (angles array for parallel/cone beam)
 * ========================================================================= */
typedef struct {
    f32 *data;          /* [n_views * params_per_view] row-major */
    i32  n_views;
    i32  params_per_view;
} ViewParams;

/* =========================================================================
 * MjModel  — mirrors the TomographyModel class
 * ========================================================================= */
typedef struct MjModel {
    ParamStore  ps;          /* all set_params/get_params data              */
    ViewParams  vp;          /* per-view parameters (angles, source pos, …) */
    ProjParams  pp;          /* packed geometry for projector calls          */
    FwdFn       fwd_fn;      /* geometry-specific forward projector          */
    BackFn      back_fn;     /* geometry-specific back projector             */
    int         use_ror_mask;/* 1 = restrict to inscribed circle             */
    int         verbose;
} MjModel;

/* ── Lifecycle ─────────────────────────────────────────────────────────── */
void mj_model_init(MjModel *m,
                   i32 num_views, i32 num_det_rows, i32 num_det_channels);
void mj_model_free(MjModel *m);

/** Sync ProjParams from ParamStore (call after any geometry param change). */
void mj_sync_proj_params(MjModel *m);

/* ── Geometry helpers ─────────────────────────────────────────────────── */
/** auto_set_recon_geometry: default square grid = (det_channels, det_channels, det_rows). */
void mj_auto_set_recon_geometry(MjModel *m);
/** scale_recon_shape by per-axis factors; optionally return pixels added. */
void mj_scale_recon_shape(MjModel *m,
                           f32 row_scale, f32 col_scale, f32 slice_scale,
                           i32 added_out[3]);

/* ── ROR mask ──────────────────────────────────────────────────────────── */
/**
 * Generate flat pixel indices inside the inscribed circle.
 * Mirrors gen_full_indices(use_ror_mask=True).
 *
 * @param num_rows, num_cols  grid dimensions
 * @param out_indices  pre-allocated [num_rows*num_cols]; filled with in-ROR indices
 * @param out_n        receives count
 */
void mj_gen_ror_mask(i32 num_rows, i32 num_cols,
                     i32 *out_indices, i32 *out_n);

/* ── Projection ────────────────────────────────────────────────────────── */
/** forward_project: recon → sinogram (full field of view). */
void mj_forward_project(MjModel *m, const f32 *recon, f32 *sino);
/** back_project: sinogram → recon (full field of view). */
void mj_back_project   (MjModel *m, const f32 *sino,  f32 *recon);
/** Compute Hessian diagonal for a pixel subset (coeff_power=2 backprojection). */
void mj_hessian_diag   (MjModel *m,
                         const i32 *pixel_indices, i32 n_pix,
                         f32 *hess_out);

/* ── Reconstruction ───────────────────────────────────────────────────── */
/**
 * recon: MBIR reconstruction via Multi-Granular VCD.
 *
 * @param sino          [views * det_rows * det_channels]
 * @param weights       same shape as sino; NULL → all ones
 * @param init_recon    [rows * cols * slices]; NULL → call mj_direct_recon
 * @param max_iterations
 * @param stop_pct      stop when NMAE % change < this
 * @param first_iter    starting iteration index (for restart)
 * @param recon_out     [rows * cols * slices] — caller allocated
 * @param stats_out     iteration statistics — may be NULL
 * @return 0 on success
 */
int mj_recon(MjModel *m,
             const f32 *sino,
             const f32 *weights,
             const f32 *init_recon,
             i32 max_iterations,
             f32 stop_pct,
             i32 first_iter,
             f32 *recon_out,
             ReconStats *stats_out);

/**
 * direct_recon: geometry-specific FBP/FDK (base = plain backprojection).
 * Geometry modules override by calling their own filter+backproject.
 */
void mj_direct_recon(MjModel *m, const f32 *sino, f32 *recon_out);

/**
 * prox_map: Plug-and-Play proximal map.
 * Mirrors TomographyModel.prox_map().
 */
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

/* ── Save / Load ─────────────────────────────────────────────────────── */
/** Binary format: 3×i32 dims header, then f32[] data. */
int mj_save_recon(const char *path, const f32 *recon, ReconShape rs);
int mj_load_recon(const char *path, f32 **recon_out,  ReconShape *rs_out);

/* ── Phantom ─────────────────────────────────────────────────────────── */
/** gen_modified_3d_sl_phantom: fills out[rows * cols * slices]. */
void mj_gen_shepp_logan(ReconShape rs, f32 *out);

#endif /* TOMOGRAPHY_MODEL_H */
