/**
 * preprocess.h
 * Plain-C translation of mbirjax/preprocess.py
 */
#ifndef PREPROCESS_H
#define PREPROCESS_H
#include "mbirjax_types.h"
/** -log(max(sino/air_scan, 1e-6)); air_scan may be NULL (->ones). */
void mj_transmission_to_attenuation(const f32 *sino, const f32 *air_scan, i32 n, f32 *out);
/** Poisson weights: w[i] = exp(-atten[i]). */
void mj_compute_weights(const f32 *atten, i32 n, f32 *out_weights);
/** Replace NaN/Inf/negative pixels with local channel median. */
void mj_correct_bad_pixels(f32 *sino, i32 views, i32 det_rows, i32 det_chans);
/** Estimate noise std from adjacent-channel differences. */
f32  mj_estimate_sino_noise_std(const f32 *sino, i32 n);
#endif
