/**
 * tomography_model.c
 *
 * Plain-C translation of mbirjax/tomography_model.py (TomographyModel class).
 *
 * Implements all geometry-agnostic methods.
 * Geometry-specific projectors are provided by parallel_beam_model.c and
 * cone_beam_model.c via the fwd_fn / back_fn function pointers in MjModel.
 */

#include "tomography_model.h"
#include "vcd_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── Internal: sync ProjParams from ParamStore ───────────────────────── */

void mj_sync_proj_params(MjModel *m)
{
    ProjParams *pp = &m->pp;
    i32 ss[3], rs[3];
    ph_get_shape(&m->ps, "sinogram_shape", ss);
    ph_get_shape(&m->ps, "recon_shape",    rs);
    pp->sino.views    = ss[0]; pp->sino.rows    = ss[1]; pp->sino.channels = ss[2];
    pp->recon.rows    = rs[0]; pp->recon.cols   = rs[1]; pp->recon.slices  = rs[2];
    pp->delta_det_channel  = (f32)ph_get_float(&m->ps, "delta_det_channel");
    pp->delta_det_row      = (f32)ph_get_float(&m->ps, "delta_det_row");
    pp->det_row_offset     = (f32)ph_get_float(&m->ps, "det_row_offset");
    pp->det_channel_offset = (f32)ph_get_float(&m->ps, "det_channel_offset");
    pp->delta_voxel        = (f32)ph_get_float(&m->ps, "delta_voxel");
    pp->source_iso_dist    = (f32)ph_get_float(&m->ps, "source_iso_dist");
    pp->source_det_dist    = (f32)ph_get_float(&m->ps, "source_det_dist");
}

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

void mj_model_init(MjModel *m,
                   i32 num_views, i32 num_det_rows, i32 num_det_channels)
{
    memset(m, 0, sizeof(*m));
    ph_init(&m->ps);
    /* Set geometry-neutral defaults that ph_init doesn't know yet */
    ph_set_float(&m->ps, "source_iso_dist", 0.0, 1);
    ph_set_float(&m->ps, "source_det_dist", 0.0, 1);

    ph_set_shape(&m->ps, "sinogram_shape",
                 num_views, num_det_rows, num_det_channels, 1);
    m->use_ror_mask = 1;
    m->verbose      = MJ_DEF_VERBOSE;
    mj_auto_set_recon_geometry(m);
    mj_sync_proj_params(m);
}

void mj_model_free(MjModel *m)
{
    ph_free(&m->ps);
    if (m->vp.data) { free(m->vp.data); m->vp.data = NULL; }
    memset(m, 0, sizeof(*m));
}

/* ── Geometry helpers ─────────────────────────────────────────────────── */

void mj_auto_set_recon_geometry(MjModel *m)
{
    /*
     * Default: recon = (det_channels, det_channels, det_rows).
     * Geometry subclasses override by calling ph_set_shape("recon_shape",…)
     * and mj_sync_proj_params() after mj_model_init().
     */
    i32 ss[3];
    ph_get_shape(&m->ps, "sinogram_shape", ss);
    ph_set_shape(&m->ps, "recon_shape",
                 ss[2], ss[2], ss[1], 1);   /* (det_ch, det_ch, det_rows) */
    mj_sync_proj_params(m);
}

void mj_scale_recon_shape(MjModel *m,
                           f32 row_scale, f32 col_scale, f32 slice_scale,
                           i32 added_out[3])
{
    i32 rs[3];
    ph_get_shape(&m->ps, "recon_shape", rs);
    i32 nr = (i32)(rs[0] * row_scale   + 0.5f);
    i32 nc = (i32)(rs[1] * col_scale   + 0.5f);
    i32 ns = (i32)(rs[2] * slice_scale + 0.5f);
    if (added_out) {
        added_out[0] = nr - rs[0];
        added_out[1] = nc - rs[1];
        added_out[2] = ns - rs[2];
    }
    ph_set_shape(&m->ps, "recon_shape", nr, nc, ns, 1);
    mj_sync_proj_params(m);
}

/* ── ROR mask ──────────────────────────────────────────────────────────── */

void mj_gen_ror_mask(i32 num_rows, i32 num_cols,
                     i32 *out_indices, i32 *out_n)
{
    f32 cx = (num_cols - 1) * 0.5f;
    f32 cy = (num_rows - 1) * 0.5f;
    f32 r  = mj_minf((f32)num_rows, (f32)num_cols) * 0.5f;
    f32 r2 = r * r;
    i32 count = 0;
    for (i32 row = 0; row < num_rows; row++)
    for (i32 col = 0; col < num_cols; col++) {
        f32 dr = row - cy, dc = col - cx;
        if (dr*dr + dc*dc <= r2) {
            if (out_indices) out_indices[count] = row * num_cols + col;
            count++;
        }
    }
    if (out_n) *out_n = count;
}

/* ── Full forward projection ─────────────────────────────────────────── */

void mj_forward_project(MjModel *m, const f32 *recon, f32 *sino)
{
    if (!m->fwd_fn) {
        fprintf(stderr, "[mbirjax] fwd_fn not set\n"); return;
    }
    const ProjParams *pp = &m->pp;
    i32 NR = pp->recon.rows, NC = pp->recon.cols, NS = pp->recon.slices;
    i32 num_views = pp->sino.views;
    i32 num_dr    = pp->sino.rows, num_dc = pp->sino.channels;
    i32 view_stride = num_dr * num_dc;

    /* Collect pixel indices */
    i32 *idx = (i32*)malloc(NR * NC * sizeof(i32));
    i32  n_pix;
    if (m->use_ror_mask) mj_gen_ror_mask(NR, NC, idx, &n_pix);
    else { for(i32 i=0;i<NR*NC;i++) idx[i]=i; n_pix=NR*NC; }

    /* Gather voxel cylinders */
    f32 *vox = (f32*)calloc(n_pix * NS, sizeof(f32));
    for (i32 pi = 0; pi < n_pix; pi++) {
        i32 flat = idx[pi];
        for (i32 s = 0; s < NS; s++)
            vox[pi * NS + s] = recon[flat * NS + s];
    }

    memset(sino, 0, num_views * view_stride * sizeof(f32));
    for (i32 v = 0; v < num_views; v++) {
        const f32 *vpar = m->vp.data
                        ? m->vp.data + v * m->vp.params_per_view : NULL;
        m->fwd_fn(vox, idx, n_pix, vpar, pp, sino + v * view_stride);
    }
    free(vox); free(idx);
}

/* ── Full back projection ─────────────────────────────────────────────── */

void mj_back_project(MjModel *m, const f32 *sino, f32 *recon)
{
    if (!m->back_fn) {
        fprintf(stderr, "[mbirjax] back_fn not set\n"); return;
    }
    const ProjParams *pp = &m->pp;
    i32 NR = pp->recon.rows, NC = pp->recon.cols, NS = pp->recon.slices;
    i32 num_views = pp->sino.views;
    i32 num_dr    = pp->sino.rows, num_dc = pp->sino.channels;
    i32 view_stride = num_dr * num_dc;

    i32 *idx = (i32*)malloc(NR * NC * sizeof(i32));
    i32  n_pix;
    if (m->use_ror_mask) mj_gen_ror_mask(NR, NC, idx, &n_pix);
    else { for(i32 i=0;i<NR*NC;i++) idx[i]=i; n_pix=NR*NC; }

    f32 *cyl = (f32*)calloc(n_pix * NS, sizeof(f32));
    for (i32 v = 0; v < num_views; v++) {
        const f32 *vpar = m->vp.data
                        ? m->vp.data + v * m->vp.params_per_view : NULL;
        m->back_fn(sino + v * view_stride, idx, n_pix, vpar, pp, 1, cyl);
    }
    memset(recon, 0, NR * NC * NS * sizeof(f32));
    for (i32 pi = 0; pi < n_pix; pi++) {
        i32 flat = idx[pi];
        for (i32 s = 0; s < NS; s++)
            recon[flat * NS + s] = cyl[pi * NS + s];
    }
    free(cyl); free(idx);
}

/* ── Hessian diagonal ─────────────────────────────────────────────────── */

void mj_hessian_diag(MjModel *m,
                     const i32 *pixel_indices, i32 n_pix,
                     f32 *hess_out)
{
    if (!m->back_fn) return;
    const ProjParams *pp = &m->pp;
    i32 num_views   = pp->sino.views;
    i32 num_dr      = pp->sino.rows, num_dc = pp->sino.channels;
    i32 view_stride = num_dr * num_dc;
    i32 NS          = pp->recon.slices;

    memset(hess_out, 0, n_pix * NS * sizeof(f32));
    f32 *ones = (f32*)malloc(view_stride * sizeof(f32));
    for (i32 i = 0; i < view_stride; i++) ones[i] = 1.f;
    for (i32 v = 0; v < num_views; v++) {
        const f32 *vpar = m->vp.data
                        ? m->vp.data + v * m->vp.params_per_view : NULL;
        m->back_fn(ones, pixel_indices, n_pix, vpar, pp, 2, hess_out);
    }
    free(ones);
}

/* ── Direct reconstruction (base: plain back-projection) ──────────────── */

void mj_direct_recon(MjModel *m, const f32 *sino, f32 *recon_out)
{
    /* Base implementation: unweighted back-projection + normalise by views */
    mj_back_project(m, sino, recon_out);
    i32 total = recon_numel(m->pp.recon);
    f32 scale = (m->pp.sino.views > 0)
              ? 1.f / (f32)m->pp.sino.views : 1.f;
    for (i32 i = 0; i < total; i++) recon_out[i] *= scale;
}

/* ── MBIR Reconstruction (Multi-Granular VCD) ─────────────────────────── */

int mj_recon(MjModel *m,
             const f32 *sino,
             const f32 *weights,
             const f32 *init_recon,
             i32 max_iterations, f32 stop_pct, i32 first_iter,
             f32 *recon_out, ReconStats *stats_out)
{
    const ProjParams *pp = &m->pp;
    i32 NR = pp->recon.rows, NC = pp->recon.cols, NS = pp->recon.slices;
    i32 total_recon  = NR * NC * NS;
    i32 total_sino   = sino_numel(pp->sino);
    (void)pp; /* pp used via vcd_subset_update internally */

    /* ── Auto-regularisation ── */
    if (ph_get_bool(&m->ps, "auto_regularize_flag")) {
        f32 typ = mj_sum_abs(sino, total_sino) / (f32)(total_sino > 0 ? total_sino : 1);
        ph_auto_regularize(&m->ps, typ);
    }
    f32 sigma_y = (f32)ph_get_float(&m->ps, "sigma_y");
    f32 fm_const = (sigma_y > 0.f) ? 1.f / (sigma_y * sigma_y) : 1.f;

    /* ── Pixel index list ── */
    i32 *idx = (i32*)malloc(NR * NC * sizeof(i32));
    i32  n_pix;
    if (m->use_ror_mask) mj_gen_ror_mask(NR, NC, idx, &n_pix);
    else { for(i32 i=0;i<NR*NC;i++) idx[i]=i; n_pix=NR*NC; }

    /* ── Initialise recon ── */
    if (init_recon) {
        memcpy(recon_out, init_recon, total_recon * sizeof(f32));
    } else {
        mj_direct_recon(m, sino, recon_out);
    }

    /* ── Initial error sinogram: e = sino − A*recon ── */
    f32 *err_sino = (f32*)malloc(total_sino * sizeof(f32));
    {
        f32 *fwd = (f32*)calloc(total_sino, sizeof(f32));
        mj_forward_project(m, recon_out, fwd);
        for (i32 i = 0; i < total_sino; i++) err_sino[i] = sino[i] - fwd[i];
        free(fwd);
    }

    /* ── Effective weights = W * fm_const ── */
    f32 *eff_w = (f32*)malloc(total_sino * sizeof(f32));
    if (weights) {
        for (i32 i = 0; i < total_sino; i++) eff_w[i] = weights[i] * fm_const;
    } else {
        for (i32 i = 0; i < total_sino; i++) eff_w[i] = fm_const;
    }

    /* ── qGGMRF parameters ── */
    QGGMRFParams qp;
    qp.sigma_x = (f32)ph_get_float(&m->ps, "sigma_x");
    qp.p       = (f32)ph_get_float(&m->ps, "p");
    qp.q       = (f32)ph_get_float(&m->ps, "q");
    qp.T       = (f32)ph_get_float(&m->ps, "T");
    vcd_init_nbr_weights(&qp);

    /* ── Granularity ── */
    i32 gran = MJ_DEF_GRANULARITY;
    {
        const Param *gp = ph_find(&m->ps, "granularity");
        if (gp && gp->type == PT_IARR && gp->v.iarr.n > 0)
            gran = gp->v.iarr.data[0];
    }

    /* ── Per-iteration tracking arrays ── */
    f32 *pct_arr   = (f32*)calloc(max_iterations, sizeof(f32));
    f32 *alpha_arr = (f32*)calloc(max_iterations, sizeof(f32));
    f32  stop_frac = stop_pct / 100.f;
    i32  iter;

    for (iter = first_iter; iter < max_iterations; iter++) {
        VPartition part = vcd_partition_create(idx, n_pix, gran);
        f32 ell1_total = 0.f, alpha_total = 0.f;

        for (i32 s = 0; s < part.n_subsets; s++) {
            f32 e1=0.f, a1=0.f;
            vcd_subset_update(recon_out, err_sino,
                              part.subsets[s].indices, part.subsets[s].n,
                              eff_w, &qp, pp, m, &e1, &a1);
            ell1_total  += e1;
            alpha_total += a1;
        }
        vcd_partition_free(&part);

        f32 recon_l1 = mj_sum_abs(recon_out, total_recon);
        f32 nmae     = (recon_l1 > 0.f) ? ell1_total / recon_l1 : 0.f;

        pct_arr  [iter - first_iter] = 100.f * nmae;
        alpha_arr[iter - first_iter] = alpha_total / (f32)(n_pix > 0 ? n_pix : 1);

        if (m->verbose && (iter % 5 == 0 || iter == first_iter))
            printf("[mbirjax] iter %d  pct_change=%.4f\n",
                   iter, 100.f * nmae);
        if (nmae < stop_frac) { iter++; break; }
    }

    /* ── Fill statistics ── */
    if (stats_out) {
        i32 stored = iter - first_iter;
        stats_out->num_iterations = stored;
        stats_out->pct_change  = (f32*)malloc(stored * sizeof(f32));
        stats_out->alpha_values= (f32*)malloc(stored * sizeof(f32));
        if (stats_out->pct_change && stats_out->alpha_values) {
            memcpy(stats_out->pct_change,   pct_arr,   stored * sizeof(f32));
            memcpy(stats_out->alpha_values, alpha_arr, stored * sizeof(f32));
        }
        stats_out->fm_rmse    = FLT_MAX;
        stats_out->prior_loss = FLT_MAX;
    }

    free(alpha_arr); free(pct_arr);
    free(eff_w); free(err_sino); free(idx);
    return 0;
}

/* ── Proximal map ──────────────────────────────────────────────────────── */

int mj_prox_map(MjModel *m,
                const f32 *prox_input,
                const f32 *sino,
                f32 sigma_prox,
                const f32 *weights,
                const f32 *init_recon,
                i32 max_iterations, f32 stop_pct,
                f32 *recon_out, ReconStats *stats_out)
{
    /*
     * Plug-and-Play proximal map:
     *   Adds prox prior term ||x − prox_input||² / sigma_prox²
     *   to the data fidelity cost.
     *
     * Implemented as: start from prox_input and run recon() with
     * the sigma_prox parameter set so the prior is centred at prox_input.
     */
    if (sigma_prox <= 0.f) {
        i32 n = recon_numel(m->pp.recon);
        f32 typ = mj_sum_abs(prox_input, n) / (f32)(n > 0 ? n : 1);
        sigma_prox = (typ > 0.f) ? typ : 1.f;
    }
    ph_set_float(&m->ps, "sigma_prox", sigma_prox, 0);

    /* Use prox_input as init_recon when no explicit init is provided */
    const f32 *use_init = init_recon ? init_recon : prox_input;
    return mj_recon(m, sino, weights, use_init,
                    max_iterations, stop_pct, 0, recon_out, stats_out);
}

/* ── Save / Load ──────────────────────────────────────────────────────── */

int mj_save_recon(const char *path, const f32 *recon, ReconShape rs)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    fwrite(&rs.rows,   sizeof(i32), 1, f);
    fwrite(&rs.cols,   sizeof(i32), 1, f);
    fwrite(&rs.slices, sizeof(i32), 1, f);
    fwrite(recon, sizeof(f32), recon_numel(rs), f);
    fclose(f); return 0;
}

int mj_load_recon(const char *path, f32 **recon_out, ReconShape *rs_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    if (fread(&rs_out->rows,   sizeof(i32), 1, f) != 1) goto io_err;
    if (fread(&rs_out->cols,   sizeof(i32), 1, f) != 1) goto io_err;
    if (fread(&rs_out->slices, sizeof(i32), 1, f) != 1) goto io_err;
    i32 n = recon_numel(*rs_out);
    *recon_out = (f32*)malloc(n * sizeof(f32));
    if (fread(*recon_out, sizeof(f32), n, f) != (size_t)n) goto io_err;
    fclose(f); return 0;
io_err:
    fclose(f); free(*recon_out); *recon_out = NULL; return -1;
}

/* ── Shepp-Logan phantom ─────────────────────────────────────────────── */
/* (Delegates to phantom.c) */
#include "phantom.h"
void mj_gen_shepp_logan(ReconShape rs, f32 *out)
{
    phantom_shepp_logan(rs.rows, rs.cols, rs.slices, out);
}
