/**
 * vcd_utils.c
 *
 * Plain-C translation of the VCD (Voxel Coordinate Descent) internals
 * from mbirjax/tomography_model.py.
 *
 * Function-by-function mapping:
 *   qggmrf_rho_prime   ← _qggmrf_potential_prime
 *   qggmrf_rho_dprime  ← central finite-difference
 *   qggmrf_grad_hess_subset ← qggmrf_gradient_and_hessian_at_indices
 *   vcd_subset_update  ← vcd_subset_updater
 *   vcd_partition_*    ← gen_set_of_pixel_partitions
 */

#include "vcd_utils.h"
#include "tomography_model.h"   /* for MjModel */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── 26-neighbour direction table ─────────────────────────────────────── */
typedef struct { i32 dz,dy,dx; } Offset3;
static Offset3  g_offs[MJ_N_NBR];
static int      g_offs_init = 0;

static void init_offsets(void)
{
    if (g_offs_init) return;
    i32 k = 0;
    for (i32 dz=-1; dz<=1; dz++)
    for (i32 dy=-1; dy<=1; dy++)
    for (i32 dx=-1; dx<=1; dx++) {
        if (dz==0 && dy==0 && dx==0) continue;
        g_offs[k].dz=dz; g_offs[k].dy=dy; g_offs[k].dx=dx; k++;
    }
    g_offs_init = 1;
}

/* ── Neighbour weights ────────────────────────────────────────────────── */

void vcd_init_nbr_weights(QGGMRFParams *qp)
{
    init_offsets();
    for (i32 k = 0; k < MJ_N_NBR; k++) {
        i32 d2 = g_offs[k].dz*g_offs[k].dz
                +g_offs[k].dy*g_offs[k].dy
                +g_offs[k].dx*g_offs[k].dx;
        qp->b[k] = 1.f / sqrtf((f32)d2);
    }
}

/* ── qGGMRF potential ─────────────────────────────────────────────────── */
/*
 * ρ(t) = |t/σx|^p / (p · (1 + |t/(T·σx)|^(p-q)))
 *
 * ρ'(δ) = sign(δ)/σx · |u|^(p-1) · [1/D - (p-q)/p · |v|^(p-q) / D²]
 *   where D = 1 + |v|^(p-q),  u = δ/σx,  v = δ/(T·σx)
 */
f32 qggmrf_rho_prime(f32 delta, const QGGMRFParams *qp)
{
    f32 sx = qp->sigma_x;
    if (sx <= 0.f) return 0.f;

    f32 u    = delta / sx;
    f32 absu = mj_absf(u);
    if (absu < 1e-30f) return 0.f;

    f32 p = qp->p, q = qp->q, T = qp->T;
    f32 v    = delta / (T * sx);
    f32 absv = mj_absf(v);

    f32 upq   = powf(absv, p - q);
    f32 D     = 1.f + upq;
    f32 sign  = (delta > 0.f) ? 1.f : -1.f;
    f32 corr  = (p - q) / p * upq / (D * D);
    return sign / sx * powf(absu, p - 1.f) * (1.f / D - corr);
}

f32 qggmrf_rho_dprime(f32 delta, const QGGMRFParams *qp)
{
    /* Central finite-difference of rho_prime */
    f32 h  = 1e-4f * (mj_absf(delta) + qp->sigma_x * 1e-4f + 1e-8f);
    f32 fp = qggmrf_rho_prime(delta + h, qp);
    f32 fm = qggmrf_rho_prime(delta - h, qp);
    return (fp - fm) / (2.f * h);
}

/* ── Prior gradient + hessian for a pixel subset ──────────────────────── */
/*
 * For each voxel (pixel_indices[pi], slice s):
 *   grad[pi*NS+s] = Σ_k  b[k] · ρ'(x_center − x_nbr_k)
 *   hess[pi*NS+s] = Σ_k  b[k] · ρ''(x_center − x_nbr_k)
 *
 * Neighbours are clamped to image boundaries (edge-replication).
 *
 * Layout: flat_recon[(row*NC + col) * NS + slice]
 */
void qggmrf_grad_hess_subset(
    const f32 *flat_recon, ReconShape rs,
    const i32 *pixel_indices, i32 n_pix,
    const QGGMRFParams *qp,
    f32 *out_grad, f32 *out_hess)
{
    init_offsets();
    i32 NR=rs.rows, NC=rs.cols, NS=rs.slices;

    for (i32 pi = 0; pi < n_pix; pi++) {
        i32 flat = pixel_indices[pi];
        i32 col  = flat % NC;
        i32 row  = flat / NC;

        for (i32 s = 0; s < NS; s++) {
            f32 center = flat_recon[flat * NS + s];
            f32 g = 0.f, h = 0.f;

            for (i32 k = 0; k < MJ_N_NBR; k++) {
                /* The neighbour offset (dz,dy,dx) maps to (drow, dcol, dslice) */
                /* Python uses (nz,ny,nx)=(slices,rows,cols) convention.         */
                /* In our flat layout: pixel dim = (row,col), depth = slice.     */
                /* Offset: g_offs[k].dz → dslice, dy → drow, dx → dcol          */
                i32 nrow  = mj_clamp(row + g_offs[k].dy, 0, NR-1);
                i32 ncol  = mj_clamp(col + g_offs[k].dx, 0, NC-1);
                i32 nsl   = mj_clamp(s   + g_offs[k].dz, 0, NS-1);
                i32 nflat = nrow * NC + ncol;
                f32 nbr   = flat_recon[nflat * NS + nsl];
                f32 delta = center - nbr;
                f32 bk    = qp->b[k];
                g += bk * qggmrf_rho_prime (delta, qp);
                h += bk * qggmrf_rho_dprime(delta, qp);
            }
            out_grad[pi * NS + s] = g;
            out_hess[pi * NS + s] = h;
        }
    }
}

/* ── VCD subset update ────────────────────────────────────────────────── */
/*
 * Python reference:  TomographyModel.vcd_subset_updater()
 *
 * Steps:
 *  1. Compute prior gradient + hessian via qggmrf_grad_hess_subset.
 *  2. Compute forward hessian diagonal via squared backprojection.
 *  3. Back-project weighted error sinogram → forward gradient.
 *  4. Compute update direction delta_recon = -(fwd_grad + prior_grad) /
 *                                             (fwd_hess + prior_hess)
 *  5. Line-search for optimal step alpha.
 *  6. Apply x += alpha * delta_recon.
 *  7. Update error sinogram: e -= alpha * A * delta_recon.
 */
void vcd_subset_update(
    f32 *flat_recon,
    f32 *error_sino,
    const i32 *pixel_indices, i32 n_pix,
    const f32  *eff_weights,
    const QGGMRFParams *qp,
    const ProjParams   *pp,
    MjModel            *model,
    f32 *out_ell1, f32 *out_alpha)
{
    i32 NS         = pp->recon.slices;
    i32 num_views  = pp->sino.views;
    i32 num_dr     = pp->sino.rows;
    i32 num_dc     = pp->sino.channels;
    i32 view_stride = num_dr * num_dc;
    i32 n_vox      = n_pix * NS;

    f32 *prior_grad = (f32*)calloc(n_vox, sizeof(f32));
    f32 *prior_hess = (f32*)calloc(n_vox, sizeof(f32));
    f32 *fwd_hess   = (f32*)calloc(n_vox, sizeof(f32));
    f32 *fwd_grad   = (f32*)calloc(n_vox, sizeof(f32));
    f32 *delta_recon= (f32*)malloc(n_vox * sizeof(f32));
    f32 *delta_sino = (f32*)calloc(num_views * view_stride, sizeof(f32));
    f32 *wt_err_view= (f32*)malloc(view_stride * sizeof(f32));

    /* Step 1: prior grad + hess */
    qggmrf_grad_hess_subset(flat_recon, pp->recon,
                             pixel_indices, n_pix, qp,
                             prior_grad, prior_hess);

    /* Step 2: forward hessian diagonal — back-project all-ones with coeff^2 */
    if (model->back_fn) {
        f32 *ones = (f32*)malloc(view_stride * sizeof(f32));
        for (i32 i = 0; i < view_stride; i++) ones[i] = 1.f;
        for (i32 v = 0; v < num_views; v++) {
            const f32 *vpar = model->vp.data
                            ? model->vp.data + v * model->vp.params_per_view
                            : NULL;
            model->back_fn(ones, pixel_indices, n_pix, vpar, pp, 2, fwd_hess);
        }
        free(ones);
    }
    /* Scale forward hessian by fm_constant (= eff_weights[0] representative) */
    f32 fm = eff_weights[0];
    for (i32 i = 0; i < n_vox; i++) fwd_hess[i] *= fm;

    /* Step 3: forward gradient = -Aᵀ W e  (back-project weighted error) */
    if (model->back_fn) {
        for (i32 v = 0; v < num_views; v++) {
            const f32 *es = error_sino + v * view_stride;
            const f32 *ws = eff_weights + v * view_stride;
            for (i32 i = 0; i < view_stride; i++) wt_err_view[i] = ws[i] * es[i];
            const f32 *vpar = model->vp.data
                            ? model->vp.data + v * model->vp.params_per_view
                            : NULL;
            model->back_fn(wt_err_view, pixel_indices, n_pix, vpar, pp, 1, fwd_grad);
        }
    }
    /* fwd_grad = -AᵀWe  → negate */
    for (i32 i = 0; i < n_vox; i++) fwd_grad[i] = -fwd_grad[i];

    /* Step 4: update direction */
    for (i32 i = 0; i < n_vox; i++) {
        f32 denom = fwd_hess[i] + prior_hess[i] + MJ_EPS;
        delta_recon[i] = -(fwd_grad[i] + prior_grad[i]) / denom;
    }

    /* Step 5: forward project delta_recon → delta_sino */
    if (model->fwd_fn) {
        for (i32 v = 0; v < num_views; v++) {
            const f32 *vpar = model->vp.data
                            ? model->vp.data + v * model->vp.params_per_view
                            : NULL;
            model->fwd_fn(delta_recon, pixel_indices, n_pix, vpar, pp,
                          delta_sino + v * view_stride);
        }
    }

    /* Step 5b: compute optimal alpha via quadratic line search */
    f32 prior_lin = mj_dot(prior_grad, delta_recon, n_vox);
    f32 prior_quad = 0.f;
    for (i32 i = 0; i < n_vox; i++)
        prior_quad += prior_hess[i] * delta_recon[i] * delta_recon[i];

    f32 fwd_lin = 0.f, fwd_quad = 0.f;
    for (i32 v = 0; v < num_views; v++) {
        const f32 *ds = delta_sino + v * view_stride;
        const f32 *es = error_sino + v * view_stride;
        const f32 *ws = eff_weights + v * view_stride;
        for (i32 i = 0; i < view_stride; i++) {
            fwd_lin  += ws[i] * es[i] * ds[i];
            fwd_quad += ws[i] * ds[i] * ds[i];
        }
    }
    f32 alpha_num = fwd_lin - prior_lin;
    f32 alpha_den = fwd_quad + prior_quad + MJ_EPS;
    f32 alpha = mj_clipf(alpha_num / alpha_den, MJ_EPS, 1.5f);

    /* Step 6: apply update */
    f32 ell1 = 0.f;
    for (i32 pi = 0; pi < n_pix; pi++) {
        i32 flat = pixel_indices[pi];
        for (i32 s = 0; s < NS; s++) {
            i32 i = pi * NS + s;
            f32 dr = alpha * delta_recon[i];
            flat_recon[flat * NS + s] += dr;
            ell1 += mj_absf(dr);
        }
    }

    /* Step 7: update error sinogram */
    for (i32 v = 0; v < num_views; v++) {
        f32 *es = error_sino + v * view_stride;
        const f32 *ds = delta_sino + v * view_stride;
        for (i32 i = 0; i < view_stride; i++) es[i] -= alpha * ds[i];
    }

    *out_ell1  = ell1;
    *out_alpha = alpha;

    free(wt_err_view); free(delta_sino); free(delta_recon);
    free(fwd_grad); free(fwd_hess); free(prior_hess); free(prior_grad);
}

/* ── Partition helpers ────────────────────────────────────────────────── */

VPartition vcd_partition_create(const i32 *pixel_indices, i32 n_pix, i32 gran)
{
    i32 ns = (n_pix + gran - 1) / gran;
    VPartition vp;
    vp.n_subsets = ns;
    vp.subsets   = (VSubset*)malloc(ns * sizeof(VSubset));
    /* subsets point directly into the pixel_indices array (no copy) */
    for (i32 s = 0; s < ns; s++) {
        i32 start = s * gran;
        i32 end   = start + gran < n_pix ? start + gran : n_pix;
        vp.subsets[s].indices = pixel_indices + start;
        vp.subsets[s].n       = end - start;
    }
    return vp;
}

void vcd_partition_free(VPartition *vp)
{
    free(vp->subsets);
    vp->subsets   = NULL;
    vp->n_subsets = 0;
}
