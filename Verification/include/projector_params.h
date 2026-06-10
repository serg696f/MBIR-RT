/**
 * projector_params.h
 *
 * Geometry parameters passed to every forward/back projector function.
 * Shared by parallel_beam_model.c and cone_beam_model.c.
 *
 * This replaces the Python "projector_params" namedtuple built in
 * TomographyModel.create_projectors().
 */
#ifndef PROJECTOR_PARAMS_H
#define PROJECTOR_PARAMS_H

#include "mbirjax_types.h"

/**
 * All geometric parameters needed by one projector call.
 *
 * Python equivalent:
 *   ProjectorParams = namedtuple('ProjectorParams',
 *       ['sinogram_shape','recon_shape','get_geometry_params'])
 */
typedef struct {
    SinoShape  sino;        /* (views, det_rows, det_channels)  */
    ReconShape recon;       /* (rows, cols, slices)             */
    f32 delta_det_channel;  /* detector channel spacing  [ALU]  */
    f32 delta_det_row;      /* detector row spacing      [ALU]  */
    f32 det_row_offset;     /* row centre offset         [pixels] */
    f32 det_channel_offset; /* channel centre offset     [pixels] */
    f32 delta_voxel;        /* voxel size                [ALU]  */
    /* Cone-beam extras (both 0 for parallel beam) */
    f32 source_iso_dist;    /* source-to-isocenter distance     */
    f32 source_det_dist;    /* source-to-detector distance      */
} ProjParams;

/**
 * Function pointer: forward project a pixel batch into one detector view.
 *
 * Python: TomographyModel.forward_project_pixel_batch_to_one_view()
 *
 * @param voxel_values  [n_pix * slices]  values for the pixel batch
 * @param pixel_indices [n_pix]           flat (row*cols + col) indices
 * @param n_pix         batch size
 * @param view_params   [params_per_view] view-specific parameters
 * @param pp            packed geometry
 * @param det_out       [det_rows * det_channels]  ADD to this buffer
 */
typedef void (*FwdFn)(
    const f32 *voxel_values,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params,
    const ProjParams *pp,
    f32 *det_out
);

/**
 * Function pointer: back-project one sinogram view to a pixel batch.
 *
 * Python: TomographyModel.back_project_one_view_to_pixel_batch()
 *
 * @param sino_view     [det_rows * det_channels]  (NULL → use all-ones for hessian)
 * @param pixel_indices [n_pix]
 * @param n_pix
 * @param view_params
 * @param pp
 * @param coeff_power   1 = normal,  2 = hessian diagonal (squares coefficients)
 * @param recon_out     [n_pix * slices]  ADD to this buffer
 */
typedef void (*BackFn)(
    const f32 *sino_view,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params,
    const ProjParams *pp,
    i32 coeff_power,
    f32 *recon_out
);

#endif /* PROJECTOR_PARAMS_H */
