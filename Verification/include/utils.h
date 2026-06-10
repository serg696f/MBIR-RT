/**
 * utils.h
 * Plain-C translation of mbirjax/_utils.py and utility helpers.
 */
#ifndef UTILS_H
#define UTILS_H
#include "mbirjax_types.h"
/** mkdir -p equivalent. */
int  mj_makedirs(const char *path);
/** Save float array with ndims-header to binary file. */
int  mj_save_raw(const char *path, const f32 *data, const i32 *dims, i32 ndims);
/** Load float array saved with mj_save_raw. Caller frees *data and *dims. */
int  mj_load_raw(const char *path, f32 **data, i32 **dims, i32 *ndims);
/** Generate flat pixel indices; use_ror_mask=1 → inscribed circle only. */
void mj_gen_full_indices(i32 rows, i32 cols, int use_ror, i32 *out, i32 *n_out);
/** Multi-granular partition sequence for VCD. seq_out[max_iter]. */
void mj_gen_partition_sequence(i32 n_gran, i32 max_iter, i32 *seq_out);
/** NMAE: ||a-b||_1 / ||b||_1. */
f32  mj_nmae(const f32 *a, const f32 *b, i32 n);
/** Linear interp: out = a*(1-t) + b*t. */
void mj_lerp(const f32 *a, const f32 *b, f32 t, f32 *out, i32 n);
/** Print/return approximate available memory in bytes. */
long mj_get_memory_stats(int print_results);
#endif
