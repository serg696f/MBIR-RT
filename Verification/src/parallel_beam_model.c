/**
 * parallel_beam_model.c
 *
 * Plain-C translation of mbirjax/parallel_beam_model.py (ParallelBeamModel).
 *
 * Implements:
 *   pb_model_create()   — sets fwd_fn, back_fn, view_params (angles)
 *   pb_fwd_one_view()   — bilinear forward projection at angle θ
 *   pb_back_one_view()  — bilinear back projection at angle θ
 *   pb_fbp_recon()      — Ram-Lak filtered back-projection
 *
 * Layout conventions (match Python):
 *   sino  [view, det_row, det_channel] → flat index v*(R*C)+r*C+c
 *   recon [(row*cols+col)*slices+slice]
 *
 * Coordinate system (same as mbirjax):
 *   x = (col − (NC−1)/2) * Δv
 *   y = (row − (NR−1)/2) * Δv
 *   det_channel ← (x cos θ + y sin θ) / Δd_ch  + (NDC−1)/2 + offset
 *   det_row     ← slice                + (NDR−1)/2 + offset   (1:1 mapping)
 */

#include "parallel_beam_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Internal: sync ProjParams (bypasses include cycle) ─────────────── */
static void pb_sync(MjModel *m)
{
    mj_sync_proj_params(m);   /* defined in tomography_model.c */
}

/* ── Coordinate mapping ─────────────────────────────────────────────── */

static inline void voxel_to_det_pb(
    i32 row, i32 col, i32 slice,
    f32 cos_t, f32 sin_t,
    const ProjParams *pp,
    f32 *det_ch_out, f32 *det_row_out)
{
    f32 cx = (pp->recon.cols - 1) * 0.5f;
    f32 cy = (pp->recon.rows - 1) * 0.5f;
    f32 x  = (col - cx) * pp->delta_voxel;
    f32 y  = (row - cy) * pp->delta_voxel;

    *det_ch_out  = (x * cos_t + y * sin_t) / pp->delta_det_channel
                 + (pp->sino.channels - 1) * 0.5f
                 + pp->det_channel_offset;
    *det_row_out = (f32)slice + pp->det_row_offset;
}

/* ── Model creation ─────────────────────────────────────────────────── */

void pb_model_create(MjModel *m,
                     i32 views, i32 det_rows, i32 det_chans,
                     const f32 *angles)
{
    mj_model_init(m, views, det_rows, det_chans);
    ph_set_int(&m->ps, "geometry_type", MJ_GEOM_PARALLEL, 1);
    ph_set_str(&m->ps, "view_params_name", "angles", 1);

    /* recon_shape = (det_chans, det_chans, det_rows) */
    ph_set_shape(&m->ps, "recon_shape", det_chans, det_chans, det_rows, 1);
    /* delta_voxel = delta_det_channel */
    f32 ddc = (f32)ph_get_float(&m->ps, "delta_det_channel");
    ph_set_float(&m->ps, "delta_voxel", ddc, 1);

    /* Store angles as view parameters */
    m->vp.n_views         = views;
    m->vp.params_per_view = 1;
    m->vp.data = (f32*)malloc(views * sizeof(f32));
    memcpy(m->vp.data, angles, views * sizeof(f32));

    m->fwd_fn  = pb_fwd_one_view;
    m->back_fn = pb_back_one_view;
    pb_sync(m);
}

void pb_model_free(MjModel *m) { mj_model_free(m); }

/* ── Forward projector ──────────────────────────────────────────────── */

void pb_fwd_one_view(
    const f32 *voxel_values,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params,
    const ProjParams *pp,
    f32 *det_out)
{
    if (!view_params) return;
    f32 theta = view_params[0];
    f32 cos_t = cosf(theta), sin_t = sinf(theta);
    i32 NC  = pp->recon.cols, NS = pp->recon.slices;
    i32 NDR = pp->sino.rows,  NDC = pp->sino.channels;

    for (i32 pi = 0; pi < n_pix; pi++) {
        i32 flat = pixel_indices[pi];
        i32 col  = flat % NC, row = flat / NC;
        for (i32 s = 0; s < NS; s++) {
            f32 val = voxel_values[pi * NS + s];
            if (val == 0.f) continue;
            f32 dch, drow;
            voxel_to_det_pb(row, col, s, cos_t, sin_t, pp, &dch, &drow);
            /* Bilinear splat */
            i32 r0=(i32)drow, c0=(i32)dch;
            f32 wr=drow-r0,   wc=dch-c0;
            for(i32 dr=0;dr<=1;dr++) for(i32 dc=0;dc<=1;dc++) {
                i32 ri=r0+dr, ci=c0+dc;
                if(ri<0||ri>=NDR||ci<0||ci>=NDC) continue;
                f32 w=(dr?wr:1.f-wr)*(dc?wc:1.f-wc);
                det_out[ri*NDC+ci] += w * val * pp->delta_voxel;
            }
        }
    }
}

/* ── Back projector ─────────────────────────────────────────────────── */

void pb_back_one_view(
    const f32 *sino_view,
    const i32 *pixel_indices, i32 n_pix,
    const f32 *view_params,
    const ProjParams *pp,
    i32 coeff_power,
    f32 *recon_out)
{
    if (!view_params) return;
    f32 theta = view_params[0];
    f32 cos_t = cosf(theta), sin_t = sinf(theta);
    i32 NC  = pp->recon.cols, NS = pp->recon.slices;
    i32 NDR = pp->sino.rows,  NDC = pp->sino.channels;

    for (i32 pi = 0; pi < n_pix; pi++) {
        i32 flat = pixel_indices[pi];
        i32 col  = flat % NC, row = flat / NC;
        for (i32 s = 0; s < NS; s++) {
            f32 dch, drow;
            voxel_to_det_pb(row, col, s, cos_t, sin_t, pp, &dch, &drow);
            i32 r0=(i32)drow, c0=(i32)dch;
            f32 wr=drow-r0,   wc=dch-c0;
            f32 sum = 0.f;
            for(i32 dr=0;dr<=1;dr++) for(i32 dc=0;dc<=1;dc++) {
                i32 ri=r0+dr, ci=c0+dc;
                if(ri<0||ri>=NDR||ci<0||ci>=NDC) continue;
                f32 w=(dr?wr:1.f-wr)*(dc?wc:1.f-wc);
                f32 coeff = w * pp->delta_voxel;
                f32 sino_val = sino_view ? sino_view[ri*NDC+ci] : 1.f;
                sum += (coeff_power==2) ? coeff*coeff*sino_val : coeff*sino_val;
            }
            recon_out[pi*NS+s] += sum;
        }
    }
}

/* ── Ram-Lak ramp filter ───────────────────────────────────────────── */
/*
 * Direct-domain convolution kernel (analytical ramp):
 *   h[0]     = 1/4
 *   h[k odd] = -1 / (k²π²)
 *   h[k even, k≠0] = 0
 *
 * Applied row-by-row to the sinogram.
 */
static void apply_ramp_1d(f32 *row_data, i32 n, f32 dc)
{
    f32 *tmp = (f32*)calloc(n, sizeof(f32));
    for (i32 i = 0; i < n; i++) {
        f32 acc = row_data[i] * 0.25f;
        for (i32 k = 1; k < n; k += 2) {
            f32 h = -1.f / ((f32)(k*k) * (f32)(M_PI*M_PI));
            if (i-k >= 0) acc += h * row_data[i-k];
            if (i+k <  n) acc += h * row_data[i+k];
        }
        tmp[i] = acc / dc;
    }
    memcpy(row_data, tmp, n * sizeof(f32));
    free(tmp);
}

/* ── FBP ────────────────────────────────────────────────────────────── */

void pb_fbp_recon(MjModel *m, const f32 *sino, f32 *recon_out)
{
    pb_sync(m);
    const ProjParams *pp = &m->pp;
    i32 num_views = pp->sino.views;
    i32 NDR = pp->sino.rows, NDC = pp->sino.channels;
    i32 view_stride = NDR * NDC;

    /* Step 1: filter each sinogram row */
    f32 *fsino = (f32*)malloc(num_views * view_stride * sizeof(f32));
    memcpy(fsino, sino, num_views * view_stride * sizeof(f32));
    for (i32 v = 0; v < num_views; v++)
    for (i32 r = 0; r < NDR; r++)
        apply_ramp_1d(fsino + v*view_stride + r*NDC, NDC, pp->delta_det_channel);

    /* Step 2: back-project the filtered sinogram */
    i32 NR=pp->recon.rows, NC=pp->recon.cols, NS=pp->recon.slices;
    i32 *idx = (i32*)malloc(NR*NC*sizeof(i32));
    i32  n_pix;
    if (m->use_ror_mask) mj_gen_ror_mask(NR, NC, idx, &n_pix);
    else { for(i32 i=0;i<NR*NC;i++) idx[i]=i; n_pix=NR*NC; }

    f32 *cyl = (f32*)calloc(n_pix*NS, sizeof(f32));
    for (i32 v = 0; v < num_views; v++) {
        const f32 *vpar = m->vp.data + v;
        pb_back_one_view(fsino + v*view_stride, idx, n_pix,
                         vpar, pp, 1, cyl);
    }

    /* Step 3: scatter and scale by π/num_views */
    i32 total = NR*NC*NS;
    memset(recon_out, 0, total * sizeof(f32));
    f32 scale = (f32)(M_PI) / (f32)num_views;
    for (i32 pi = 0; pi < n_pix; pi++) {
        i32 flat = idx[pi];
        for (i32 s = 0; s < NS; s++)
            recon_out[flat*NS+s] = cyl[pi*NS+s] * scale;
    }

    free(cyl); free(idx); free(fsino);
}
