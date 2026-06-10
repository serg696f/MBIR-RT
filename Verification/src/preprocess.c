#define _POSIX_C_SOURCE 200809L
#include "preprocess.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

void mj_transmission_to_attenuation(const f32 *sino, const f32 *air, i32 n, f32 *out)
{
    const f32 MIN_VAL = 1e-6f;
    for (i32 i = 0; i < n; i++) {
        f32 a = air ? air[i] : 1.f;
        if (a <= 0.f) a = 1.f;
        f32 ratio = sino[i] / a;
        if (ratio < MIN_VAL) ratio = MIN_VAL;
        out[i] = -logf(ratio);
    }
}

void mj_compute_weights(const f32 *atten, i32 n, f32 *out)
{
    for (i32 i = 0; i < n; i++) out[i] = expf(-mj_absf(atten[i]));
}

void mj_correct_bad_pixels(f32 *sino, i32 views, i32 det_rows, i32 det_chans)
{
    i32 total = views * det_rows * det_chans;
    f32 *tmp = (f32*)malloc(total * sizeof(f32));
    memcpy(tmp, sino, total * sizeof(f32));
    for (i32 v=0;v<views;v++)
    for (i32 r=0;r<det_rows;r++)
    for (i32 c=0;c<det_chans;c++) {
        i32 idx=(v*det_rows+r)*det_chans+c;
        f32 val=sino[idx];
        int bad=(val!=val)||(val>=FLT_MAX)||(val<0.f);
        if (!bad) continue;
        f32 nbr[5]; i32 nn=0;
        for (i32 dc=-2;dc<=2;dc++) {
            i32 c2=c+dc;
            if(dc==0||c2<0||c2>=det_chans) continue;
            f32 nv=sino[(v*det_rows+r)*det_chans+c2];
            if(nv==nv&&nv<FLT_MAX&&nv>=0.f) nbr[nn++]=nv;
        }
        if (nn==0) { tmp[idx]=0.f; continue; }
        for(i32 i=0;i<nn-1;i++) for(i32 j=i+1;j<nn;j++)
            if(nbr[j]<nbr[i]){f32 t=nbr[i];nbr[i]=nbr[j];nbr[j]=t;}
        tmp[idx]=nbr[nn/2];
    }
    memcpy(sino, tmp, total*sizeof(f32));
    free(tmp);
}

f32 mj_estimate_sino_noise_std(const f32 *sino, i32 n)
{
    if (n < 2) return 0.f;
    f32 sum=0.f;
    for(i32 i=0;i<n-1;i++) sum+=mj_absf(sino[i+1]-sino[i]);
    return (sum/(f32)(n-1))/0.7979f;
}
