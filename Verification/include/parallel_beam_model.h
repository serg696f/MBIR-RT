/**
 * parallel_beam_model.h
 *
 * Plain-C translation of mbirjax/parallel_beam_model.py (ParallelBeamModel).
 *
 * Geometry: parallel X-ray beams at angle θ(v) per view.
 *   Detector rows align with the rotation axis (z/slice direction).
 *   Bilinear interpolation on the detector.
 *   One view parameter per view: angle (radians).
 *
 * Public API:
 *   pb_model_create()  — init MjModel for parallel beam
 *   pb_model_free()    — alias for mj_model_free
 *   pb_fwd_one_view()  — forward projector (FwdFn)
 *   pb_back_one_view() — back projector    (BackFn)
 *   pb_fbp_recon()     — Filtered Back-Projection (FBP)
 *   pb_auto_set_recon_geometry() — recon_shape from sino + angles
 */
#ifndef PARALLEL_BEAM_MODEL_H
#define PARALLEL_BEAM_MODEL_H

#include "mbirjax_types.h"
#include "tomography_model.h"

/**
 * Initialise a parallel-beam MjModel.
 *
 * @param m          uninitialised MjModel
 * @param views      sinogram dimension 0
 * @param det_rows   sinogram dimension 1
 * @param det_chans  sinogram dimension 2
 * @param angles     [views] rotation angles in radians
 */
void pb_model_create(MjModel *m,
                     i32 views, i32 det_rows, i32 det_chans,
                     const f32 *angles);

/** Free; wraps mj_model_free. */
void pb_model_free(MjModel *m);

/** Parallel-beam forward projector (one view). FwdFn signature. */
void pb_fwd_one_view(
    const f32 *voxel_values,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params,        /* [0] = angle */
    const ProjParams *pp,
    f32 *det_out
);

/** Parallel-beam back projector (one view). BackFn signature. */
void pb_back_one_view(
    const f32 *sino_view,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params,
    const ProjParams *pp,
    i32 coeff_power,
    f32 *recon_out
);

/**
 * Filtered Back-Projection.
 * Mirrors ParallelBeamModel.fbp_recon().
 * Uses Ram-Lak ramp filter (direct-domain convolution).
 */
void pb_fbp_recon(MjModel *m, const f32 *sino, f32 *recon_out);

#endif /* PARALLEL_BEAM_MODEL_H */
