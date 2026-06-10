/**
 * denoising.h
 * Plain-C translation of mbirjax/denoising.py (QGGMRFDenoiser + median_filter3d).
 */
#ifndef DENOISING_H
#define DENOISING_H
#include "mbirjax_types.h"
typedef struct {
    i32 nz, ny, nx;
    f32 sigma_noise;   /* 0 = auto-estimate    */
    f32 sigma_y;       /* set by mj_denoise    */
    f32 sigma_x;       /* set by mj_denoise    */
    f32 sharpness;
    f32 p, q, T;
    i32 max_iterations;
    f32 stop_threshold_pct;
    i32 granularity;
    i32 verbose;
} DenoiserParams;
void denoiser_default_params(DenoiserParams *dp, i32 nz, i32 ny, i32 nx);
void mj_denoise(const f32 *image, f32 *denoised, DenoiserParams *dp);
f32  mj_estimate_image_noise_std(const f32 *image, i32 nz, i32 ny, i32 nx);
void mj_median_filter3d(const f32 *x, f32 *out, i32 nz, i32 ny, i32 nx,
                         int return_min_max, f32 *out_min, f32 *out_max);
#endif
