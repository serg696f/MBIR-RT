/**
 * phantom.h
 * Plain-C translation of mbirjax phantom generation utilities.
 * Provides 3-D Shepp-Logan and flat-disk phantom generators.
 */
#ifndef PHANTOM_H
#define PHANTOM_H
#include "mbirjax_types.h"
/** 10-ellipsoid modified Shepp-Logan, low-dynamic-range. out[rows*cols*slices]. */
void phantom_shepp_logan(i32 rows, i32 cols, i32 slices, f32 *out);
/** Flat disk: all voxels inside inscribed circle = value, outside = 0. */
void phantom_flat_disk(i32 rows, i32 cols, i32 slices, f32 value, f32 *out);
#endif
