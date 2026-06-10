/**
 * phantom.c — Plain-C translation of mbirjax phantom generation utilities.
 *
 * phantom_shepp_logan(): 3-D Shepp-Logan (10 ellipsoids, low-dynamic-range).
 * phantom_flat_disk():   Simple inscribed-circle flat phantom.
 *
 * Shepp-Logan ellipsoid table from:
 *   Shepp & Logan, IEEE Trans. Nucl. Sci. 21(3):21-43, 1974.
 *   "Low-dynamic-range" values scaled to [0,1] as used in mbirjax demos.
 */
#include "phantom.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { f32 cx,cy,cz, ax,ay,az, phi, val; } Ellipsoid;

/* Modified Shepp-Logan table (mbirjax low-dynamic-range version) */
static const Ellipsoid SL[10] = {
/*    cx      cy       cz     ax    ay    az    phi_deg   val  */
{  0.00f,  0.00f,  0.00f, 0.69f,0.92f,0.90f,   0.0f,  1.00f },
{  0.00f, -0.0184f,0.00f,0.6624f,0.874f,0.880f, 0.0f,-0.80f },
{  0.22f,  0.00f,  0.00f, 0.11f,0.31f,0.22f, -18.0f,-0.20f },
{ -0.22f,  0.00f,  0.00f, 0.16f,0.41f,0.28f,  18.0f,-0.20f },
{  0.00f,  0.35f,  0.00f, 0.21f,0.25f,0.41f,   0.0f, 0.10f },
{  0.00f,  0.10f,  0.00f,0.046f,0.046f,0.05f,  0.0f, 0.10f },
{  0.00f, -0.10f,  0.00f,0.046f,0.046f,0.05f,  0.0f, 0.10f },
{ -0.08f, -0.605f, 0.00f,0.046f,0.023f,0.05f,  0.0f, 0.10f },
{  0.00f, -0.606f, 0.00f,0.023f,0.023f,0.02f,  0.0f, 0.10f },
{  0.06f, -0.605f, 0.00f,0.023f,0.046f,0.02f,  0.0f, 0.10f }
};

void phantom_shepp_logan(i32 rows, i32 cols, i32 slices, f32 *out)
{
    i32 total = rows * cols * slices;
    memset(out, 0, total * sizeof(f32));
    for (i32 r = 0; r < rows; r++) {
        f32 y = 2.f*(r+0.5f)/(f32)rows - 1.f;
        for (i32 c = 0; c < cols; c++) {
            f32 x = 2.f*(c+0.5f)/(f32)cols - 1.f;
            for (i32 s = 0; s < slices; s++) {
                f32 z = 2.f*(s+0.5f)/(f32)slices - 1.f;
                f32 v = 0.f;
                for (i32 e = 0; e < 10; e++) {
                    f32 phi = SL[e].phi * (f32)(M_PI/180.0);
                    f32 cp  = cosf(phi), sp = sinf(phi);
                    f32 dx  = x - SL[e].cx, dy = y - SL[e].cy, dz = z - SL[e].cz;
                    f32 dxr = dx*cp + dy*sp, dyr = -dx*sp + dy*cp;
                    f32 ax = SL[e].ax, ay = SL[e].ay, az = SL[e].az;
                    if (ax<=0||ay<=0||az<=0) continue;
                    if ((dxr/ax)*(dxr/ax)+(dyr/ay)*(dyr/ay)+(dz/az)*(dz/az)<=1.f)
                        v += SL[e].val;
                }
                if (v < 0.f) v = 0.f;
                if (v > 1.f) v = 1.f;
                out[(r*cols+c)*slices+s] = v;
            }
        }
    }
}

void phantom_flat_disk(i32 rows, i32 cols, i32 slices, f32 value, f32 *out)
{
    i32 total = rows * cols * slices;
    memset(out, 0, total * sizeof(f32));
    f32 cx=(cols-1)*0.5f, cy=(rows-1)*0.5f;
    f32 r = mj_minf((f32)rows,(f32)cols)*0.45f, r2=r*r;
    for (i32 row=0;row<rows;row++)
    for (i32 col=0;col<cols;col++) {
        f32 dx=col-cx, dy=row-cy;
        if (dx*dx+dy*dy<=r2)
            for (i32 s=0;s<slices;s++) out[(row*cols+col)*slices+s]=value;
    }
}
