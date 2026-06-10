/**
 * cone_beam_model.c
 *
 * Plain-C translation of mbirjax/cone_beam_model.py (ConeBeamModel).
 *
 * Projection geometry (flat-detector cone beam):
 *   World frame: x = along detector channel axis (right),
 *                y = beam direction (source → iso → detector),
 *                z = detector row axis (up).
 *
 *   Source position at angle θ: S = (0, −D_iso, 0) rotated by −θ.
 *   Voxel position: (xv, yv, zv) rotated into source frame.
 *   Detector coordinates:
 *     u  = xv_rot / (D_iso + yv_rot) * D_det   [channel]
 *     v  = zv      / (D_iso + yv_rot) * D_det   [row]
 *   Detector indices:
 *     det_ch  = u / Δd_ch + (NDC−1)/2 + offset_ch
 *     det_row = v / Δd_row + (NDR−1)/2 + offset_row
 *
 * FDK:
 *   1. Cosine-weight the sinogram: w(u,v) = D_det / sqrt(D_det²+u²+v²)
 *   2. Apply 1-D ramp filter along u (Ram-Lak) per detector row.
 *   3. Weighted back-projection: each pixel accumulates
 *       (D_iso / (D_iso + yv_rot))²  × bilinear_sample(filtered_sino)
 *   4. Scale by π / num_views.
 */

#include "cone_beam_model.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Sync helper ─────────────────────────────────────────────────────── */
static void cb_sync(MjModel *m) { mj_sync_proj_params(m); }

/* ── Voxel → detector mapping ────────────────────────────────────────── */

static inline void voxel_to_det_cb(
    i32 row, i32 col, i32 slice,
    f32 cos_t, f32 sin_t,
    const ProjParams *pp,
    f32 *det_ch_out, f32 *det_row_out)
{
    f32 D_iso = pp->source_iso_dist;
    f32 D_det = pp->source_det_dist;
    f32 dv    = pp->delta_voxel;

    f32 cx = (pp->recon.cols  - 1) * 0.5f;
    f32 cy = (pp->recon.rows  - 1) * 0.5f;
    f32 cz = (pp->recon.slices- 1) * 0.5f;

    /* Physical voxel coordinates */
    f32 xw = (col   - cx) * dv;
    f32 yw = (row   - cy) * dv;
    /* Slice maps to z through detector row spacing / magnification */
    f32 magnif = (D_iso > 0.f) ? D_det / D_iso : 1.f;
    f32 zw = (slice - cz) * pp->delta_det_row / magnif;

    /* Rotate source/voxel system: source along -y at angle 0 */
    /* At rotation angle θ, the source is at (D_iso*sin θ, -D_iso*cos θ, 0) */
    /* Voxel in rotated frame (source at origin, beam along +y): */
    f32 xr =  xw * cos_t + yw * sin_t;   /* lateral           */
    f32 yr = -xw * sin_t + yw * cos_t;   /* depth toward det  */

    f32 t = D_iso + yr;
    if (fabsf(t) < 1e-10f) t = 1e-10f;

    f32 u = xr / t * D_det;
    f32 v = zw / t * D_det;

    *det_ch_out  = u / pp->delta_det_channel
                 + (pp->sino.channels - 1) * 0.5f + pp->det_channel_offset;
    *det_row_out = v / pp->delta_det_row
                 + (pp->sino.rows     - 1) * 0.5f + pp->det_row_offset;
}

/* ── Model creation ─────────────────────────────────────────────────── */

void cb_model_create(MjModel *m,
                     i32 views, i32 det_rows, i32 det_chans,
                     const f32 *angles,
                     f32 source_iso_dist, f32 source_det_dist)
{
    mj_model_init(m, views, det_rows, det_chans);
    ph_set_int  (&m->ps, "geometry_type",   MJ_GEOM_CONE, 1);
    ph_set_str  (&m->ps, "view_params_name","angles",     1);
    ph_set_float(&m->ps, "source_iso_dist", source_iso_dist, 1);
    ph_set_float(&m->ps, "source_det_dist", source_det_dist, 1);

    f32 magnif = (source_iso_dist > 0.f) ? source_det_dist / source_iso_dist : 1.f;
    f32 ddc    = (f32)ph_get_float(&m->ps, "delta_det_channel");
    f32 dv     = ddc / magnif;
    ph_set_float(&m->ps, "delta_voxel", dv, 1);

    /* recon_shape: field-of-view at isocenter */
    ph_set_shape(&m->ps, "recon_shape", det_chans, det_chans, det_rows, 1);

    m->vp.n_views         = views;
    m->vp.params_per_view = 1;
    m->vp.data = (f32*)malloc(views * sizeof(f32));
    memcpy(m->vp.data, angles, views * sizeof(f32));

    m->fwd_fn  = cb_fwd_one_view;
    m->back_fn = cb_back_one_view;
    cb_sync(m);
}

void cb_model_free(MjModel *m) { mj_model_free(m); }

f32 cb_get_magnification(const MjModel *m)
{
    f32 iso = (f32)ph_get_float(&m->ps, "source_iso_dist");
    f32 det = (f32)ph_get_float(&m->ps, "source_det_dist");
    return (iso > 0.f) ? det / iso : 1.f;
}

/* ── Forward projector ──────────────────────────────────────────────── */

void cb_fwd_one_view(
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
            f32 val = voxel_values[pi*NS+s];
            if (val == 0.f) continue;
            f32 dch, drow;
            voxel_to_det_cb(row, col, s, cos_t, sin_t, pp, &dch, &drow);
            i32 r0=(i32)drow, c0=(i32)dch;
            f32 wr=drow-r0, wc=dch-c0;
            for(i32 dr=0;dr<=1;dr++) for(i32 dc=0;dc<=1;dc++){
                i32 ri=r0+dr, ci=c0+dc;
                if(ri<0||ri>=NDR||ci<0||ci>=NDC) continue;
                f32 w=(dr?wr:1.f-wr)*(dc?wc:1.f-wc);
                det_out[ri*NDC+ci] += w * val * pp->delta_voxel;
            }
        }
    }
}

/* ── Back projector ─────────────────────────────────────────────────── */

void cb_back_one_view(
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
            voxel_to_det_cb(row, col, s, cos_t, sin_t, pp, &dch, &drow);
            i32 r0=(i32)drow, c0=(i32)dch;
            f32 wr=drow-r0, wc=dch-c0;
            f32 sum = 0.f;
            for(i32 dr=0;dr<=1;dr++) for(i32 dc=0;dc<=1;dc++){
                i32 ri=r0+dr, ci=c0+dc;
                if(ri<0||ri>=NDR||ci<0||ci>=NDC) continue;
                f32 w=(dr?wr:1.f-wr)*(dc?wc:1.f-wc);
                f32 coeff = w * pp->delta_voxel;
                f32 sv    = sino_view ? sino_view[ri*NDC+ci] : 1.f;
                sum += (coeff_power==2) ? coeff*coeff*sv : coeff*sv;
            }
            recon_out[pi*NS+s] += sum;
        }
    }
}

/* ── Ramp filter (shared with parallel beam) ─────────────────────────── */

static void apply_ramp_cb(f32 *row_data, i32 n, f32 dc)
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

/* ── FDK reconstruction ─────────────────────────────────────────────── */

void cb_fdk_recon(MjModel *m, const f32 *sino, f32 *recon_out)
{
    cb_sync(m);
    const ProjParams *pp = &m->pp;
    i32 num_views   = pp->sino.views;
    i32 NDR = pp->sino.rows, NDC = pp->sino.channels;
    i32 view_stride = NDR * NDC;
    f32 D_iso = pp->source_iso_dist;
    f32 D_det = pp->source_det_dist;

    /* Step 1: cosine-weight + ramp-filter per view per row */
    f32 *fsino = (f32*)malloc(num_views * view_stride * sizeof(f32));
    memcpy(fsino, sino, num_views * view_stride * sizeof(f32));

    for (i32 v = 0; v < num_views; v++) {
        f32 *view = fsino + v * view_stride;
        for (i32 r = 0; r < NDR; r++) {
            f32 *row = view + r * NDC;
            /* Physical detector row coordinate */
            f32 vphys = ((f32)r - (NDR-1)*0.5f) * pp->delta_det_row + pp->det_row_offset;
            for (i32 c = 0; c < NDC; c++) {
                f32 uphys = ((f32)c - (NDC-1)*0.5f) * pp->delta_det_channel + pp->det_channel_offset;
                f32 dist  = sqrtf(D_det*D_det + uphys*uphys + vphys*vphys);
                row[c] *= D_det / dist;
            }
            apply_ramp_cb(row, NDC, pp->delta_det_channel);
        }
    }

    /* Step 2: weighted back-projection */
    i32 NR=pp->recon.rows, NC=pp->recon.cols, NS=pp->recon.slices;
    i32 *idx = (i32*)malloc(NR*NC*sizeof(i32));
    i32  n_pix;
    if (m->use_ror_mask) mj_gen_ror_mask(NR, NC, idx, &n_pix);
    else { for(i32 i=0;i<NR*NC;i++) idx[i]=i; n_pix=NR*NC; }

    f32 *cyl = (f32*)calloc(n_pix*NS, sizeof(f32));
    f32 cx = (NC-1)*0.5f, cy = (NR-1)*0.5f;
    f32 dv = pp->delta_voxel;

    for (i32 v = 0; v < num_views; v++) {
        f32 theta = m->vp.data[v];
        f32 cos_t = cosf(theta), sin_t = sinf(theta);
        const f32 *fview = fsino + v * view_stride;

        for (i32 pi = 0; pi < n_pix; pi++) {
            i32 flat = idx[pi];
            i32 col  = flat % NC, row = flat / NC;
            /* FDK weight = (D_iso / (D_iso + yr))² */
            f32 xw  = (col-cx)*dv, yw = (row-cy)*dv;
            f32 yr  = -xw*sin_t + yw*cos_t;
            f32 t   = D_iso + yr;
            if (fabsf(t) < 1e-10f) t=1e-10f;
            f32 fdk_w = (D_iso / t) * (D_iso / t);

            for (i32 s = 0; s < NS; s++) {
                f32 dch, drow;
                voxel_to_det_cb(row, col, s, cos_t, sin_t, pp, &dch, &drow);
                i32 r0=(i32)drow, c0=(i32)dch;
                f32 wr=drow-r0, wc=dch-c0;
                f32 sum=0.f;
                for(i32 dr=0;dr<=1;dr++) for(i32 dc=0;dc<=1;dc++){
                    i32 ri=r0+dr, ci=c0+dc;
                    if(ri<0||ri>=NDR||ci<0||ci>=NDC) continue;
                    f32 w=(dr?wr:1.f-wr)*(dc?wc:1.f-wc);
                    sum += w * fview[ri*NDC+ci];
                }
                cyl[pi*NS+s] += fdk_w * sum;
            }
        }
    }

    /* Step 3: scatter + scale */
    memset(recon_out, 0, NR*NC*NS*sizeof(f32));
    f32 scale = (f32)(M_PI) / (f32)num_views;
    for (i32 pi=0; pi<n_pix; pi++) {
        i32 flat=idx[pi];
        for (i32 s=0;s<NS;s++)
            recon_out[flat*NS+s] = cyl[pi*NS+s] * scale;
    }
    free(cyl); free(idx); free(fsino);
}
