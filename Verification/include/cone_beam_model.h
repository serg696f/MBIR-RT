/**
 * cone_beam_model.h
 *
 * Plain-C translation of mbirjax/cone_beam_model.py (ConeBeamModel).
 *
 * Flat-detector, circular-orbit cone-beam geometry.
 *   Source rotates around the isocenter at distance source_iso_dist.
 *   Detector is at distance source_det_dist from the source.
 *   One view parameter: rotation angle θ (radians).
 *
 * Public API:
 *   cb_model_create()  — init MjModel for cone beam
 *   cb_model_free()    — alias for mj_model_free
 *   cb_fwd_one_view()  — forward projector (FwdFn)
 *   cb_back_one_view() — back projector    (BackFn)
 *   cb_fdk_recon()     — Feldkamp-Davis-Kress (FDK) reconstruction
 *   cb_get_magnification() — source_det_dist / source_iso_dist
 */
#ifndef CONE_BEAM_MODEL_H
#define CONE_BEAM_MODEL_H

#include "mbirjax_types.h"
#include "tomography_model.h"

/**
 * @param views, det_rows, det_chans  sinogram shape
 * @param angles                      [views] rotation angles (radians)
 * @param source_iso_dist             source-to-isocenter distance (ALU)
 * @param source_det_dist             source-to-detector distance (ALU)
 */
void cb_model_create(MjModel *m,
                     i32 views, i32 det_rows, i32 det_chans,
                     const f32 *angles,
                     f32 source_iso_dist,
                     f32 source_det_dist);

void cb_model_free(MjModel *m);

void cb_fwd_one_view(
    const f32 *voxel_values,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params,
    const ProjParams *pp,
    f32 *det_out
);

void cb_back_one_view(
    const f32 *sino_view,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params,
    const ProjParams *pp,
    i32 coeff_power,
    f32 *recon_out
);

/**
 * FDK reconstruction.
 * Mirrors ConeBeamModel.fdk_recon().
 */
void cb_fdk_recon(MjModel *m, const f32 *sino, f32 *recon_out);

/** Return source_det_dist / source_iso_dist. */
f32 cb_get_magnification(const MjModel *m);

#endif /* CONE_BEAM_MODEL_H */
