/**
 * vcd_utils.h
 *
 * Plain-C translation of the VCD (Voxel Coordinate Descent) internals
 * in mbirjax/tomography_model.py and the internal _vcd helpers.
 *
 * Key functions (all mirrors of Python methods):
 *   vcd_init_nbr_weights()        — fill b[] from distance weights
 *   qggmrf_rho_prime()            — ρ'(δ)  first derivative of prior
 *   qggmrf_rho_dprime()           — ρ''(δ) (central finite-difference)
 *   qggmrf_grad_hess_subset()     — gradient + hessian for a pixel subset
 *   vcd_subset_update()           — one VCD sweep over a pixel subset
 *   vcd_partition_create/free()   — partition helpers
 */
#ifndef VCD_UTILS_H
#define VCD_UTILS_H

#include "mbirjax_types.h"
#include "projector_params.h"

/* Forward-declare MjModel to avoid circular includes */
struct MjModel;

/* =========================================================================
 * qGGMRF prior parameters
 * ========================================================================= */
typedef struct {
    f32 sigma_x;
    f32 p, q, T;
    f32 b[MJ_N_NBR];   /* neighbour weights, filled by vcd_init_nbr_weights */
} QGGMRFParams;

/** Fill b[] from distance-based weights (face=1, edge=1/√2, corner=1/√3). */
void vcd_init_nbr_weights(QGGMRFParams *qp);

/* ── qGGMRF potential derivatives ─────────────────────────────────────── */
/** First derivative of ρ(δ) w.r.t. δ. */
f32 qggmrf_rho_prime (f32 delta, const QGGMRFParams *qp);
/** Second derivative (central finite-difference of rho_prime). */
f32 qggmrf_rho_dprime(f32 delta, const QGGMRFParams *qp);

/* =========================================================================
 * Gradient + Hessian for a pixel subset
 *
 * Mirrors TomographyModel.qggmrf_gradient_and_hessian_at_indices()
 * ========================================================================= */
/**
 * @param flat_recon   [total_pixels * slices] — reconstruction in flat layout
 * @param recon_shape  (rows, cols, slices)
 * @param pixel_indices [n_pix] flat (row*cols+col) indices
 * @param n_pix
 * @param qp
 * @param out_grad     [n_pix * slices]   (caller allocated, zeroed)
 * @param out_hess     [n_pix * slices]   (caller allocated, zeroed)
 */
void qggmrf_grad_hess_subset(
    const f32        *flat_recon,
    ReconShape        recon_shape,
    const i32        *pixel_indices, i32 n_pix,
    const QGGMRFParams *qp,
    f32 *out_grad, f32 *out_hess
);

/* =========================================================================
 * VCD subset update
 *
 * Mirrors TomographyModel.vcd_subset_updater()
 * ========================================================================= */
/**
 * Perform one VCD update over a pixel subset.
 *
 * @param flat_recon   [rows*cols * slices]  updated in place
 * @param error_sino   [views * dr * dc]     updated in place
 * @param pixel_indices [n_pix]
 * @param n_pix
 * @param eff_weights  [views * dr * dc]   = weights * fm_constant
 * @param qp
 * @param pp           projector geometry
 * @param model        for fwd_fn / back_fn / vp
 * @param out_ell1     L1 norm of the update (output)
 * @param out_alpha    step size used (output)
 */
void vcd_subset_update(
    f32 *flat_recon,
    f32 *error_sino,
    const i32 *pixel_indices, i32 n_pix,
    const f32  *eff_weights,
    const QGGMRFParams *qp,
    const ProjParams   *pp,
    struct MjModel     *model,
    f32 *out_ell1, f32 *out_alpha
);

/* =========================================================================
 * Partition helpers  (mirrors gen_set_of_pixel_partitions)
 * ========================================================================= */
typedef struct { const i32 *indices; i32 n; } VSubset;
typedef struct { VSubset *subsets; i32 n_subsets; } VPartition;

/** Divide pixel_indices into contiguous subsets of size `gran`. */
VPartition vcd_partition_create(const i32 *pixel_indices, i32 n_pix, i32 gran);
void       vcd_partition_free(VPartition *vp);

#endif /* VCD_UTILS_H */
