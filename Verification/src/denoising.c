#include "denoising.h"
#include "vcd_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

void denoiser_default_params(DenoiserParams *dp, i32 nz, i32 ny, i32 nx)
{
    dp->nz=nz; dp->ny=ny; dp->nx=nx;
    dp->sigma_noise=0.f; dp->sigma_y=0.f; dp->sigma_x=0.f;
    dp->sharpness=0.f;
    dp->p=MJ_DEF_P; dp->q=MJ_DEF_Q; dp->T=MJ_DEF_T;
    dp->max_iterations=15; dp->stop_threshold_pct=0.2f;
    dp->granularity=16; dp->verbose=1;
}

f32 mj_estimate_image_noise_std(const f32 *image, i32 nz, i32 ny, i32 nx)
{
    f32 sum=0.f; i32 cnt=0;
    for(i32 iz=1;iz<nz;iz++)
    for(i32 iy=1;iy<ny;iy++)
    for(i32 ix=1;ix<nx;ix++){
        f32 v0=image[(iz*ny+iy)*nx+ix];
        f32 v1=image[((iz-1)*ny+iy)*nx+ix];
        f32 v2=image[(iz*ny+(iy-1))*nx+ix];
        f32 v3=image[(iz*ny+iy)*nx+(ix-1)];
        f32 mean=(v0+v1+v2+v3)*0.25f;
        f32 var=(v0-mean)*(v0-mean)+(v1-mean)*(v1-mean)
               +(v2-mean)*(v2-mean)+(v3-mean)*(v3-mean);
        sum+=sqrtf(var*0.25f); cnt++;
    }
    return (cnt>0)?sum/(f32)cnt:0.f;
}

/* Denoising VCD subset update (identity forward model A=I) */
static void denoise_subset(
    f32 *image, f32 *error,
    const i32 *rows, i32 nr,
    f32 fm_const, const QGGMRFParams *qp,
    i32 nz, i32 ny, i32 nx,
    f32 *out_ell1, f32 *out_alpha)
{
    i32 NS=nx, n_vox=nr*NS;
    f32 *pg=(f32*)calloc(n_vox,sizeof(f32));
    f32 *ph=(f32*)calloc(n_vox,sizeof(f32));
    f32 *dr=(f32*)malloc(n_vox*sizeof(f32));

    /* Prior grad + hess over 26 neighbours */
    typedef struct{i32 dz,dy,dx;} Off;
    Off offs[26]; i32 noff=0;
    for(i32 dz=-1;dz<=1;dz++) for(i32 dy=-1;dy<=1;dy++) for(i32 dx=-1;dx<=1;dx++)
        if(!(dz==0&&dy==0&&dx==0)){offs[noff].dz=dz;offs[noff].dy=dy;offs[noff].dx=dx;noff++;}

    for(i32 pi=0;pi<nr;pi++){
        i32 flat_row=rows[pi];
        i32 iy=flat_row%ny, iz=flat_row/ny;
        for(i32 ix=0;ix<NS;ix++){
            f32 ctr=image[flat_row*NS+ix];
            f32 g=0.f,h=0.f;
            for(i32 k=0;k<noff;k++){
                i32 nz2=mj_clamp(iz+offs[k].dz,0,nz-1);
                i32 ny2=mj_clamp(iy+offs[k].dy,0,ny-1);
                i32 nx2=mj_clamp(ix+offs[k].dx,0,nx-1);
                i32 nf=nz2*ny+ny2;
                f32 nbr=image[nf*NS+nx2];
                f32 delta=ctr-nbr;
                i32 d2=offs[k].dz*offs[k].dz+offs[k].dy*offs[k].dy+offs[k].dx*offs[k].dx;
                f32 bk=1.f/sqrtf((f32)d2);
                g+=bk*qggmrf_rho_prime(delta,qp);
                h+=bk*qggmrf_rho_dprime(delta,qp);
            }
            pg[pi*NS+ix]=g; ph[pi*NS+ix]=h;
        }
    }

    /* Update direction */
    for(i32 pi=0;pi<nr;pi++){
        i32 row=rows[pi];
        for(i32 ix=0;ix<NS;ix++){
            i32 i=pi*NS+ix;
            f32 fgrad=-fm_const*error[row*NS+ix];
            f32 denom=fm_const+ph[i]+MJ_EPS;
            dr[i]=-(fgrad+pg[i])/denom;
        }
    }

    /* Line search */
    f32 pl=mj_dot(pg,dr,n_vox), pq=0.f;
    for(i32 i=0;i<n_vox;i++) pq+=ph[i]*dr[i]*dr[i];
    f32 fl=0.f, fq=0.f;
    for(i32 pi=0;pi<nr;pi++){
        i32 row=rows[pi];
        for(i32 ix=0;ix<NS;ix++){
            i32 i=pi*NS+ix;
            f32 e=error[row*NS+ix];
            fl+=fm_const*e*dr[i]; fq+=fm_const*dr[i]*dr[i];
        }
    }
    f32 alpha=mj_clipf((fl-pl)/(fq+pq+MJ_EPS),MJ_EPS,1.5f);

    f32 ell1=0.f;
    for(i32 pi=0;pi<nr;pi++){
        i32 row=rows[pi];
        for(i32 ix=0;ix<NS;ix++){
            i32 i=pi*NS+ix;
            f32 step=alpha*dr[i];
            image[row*NS+ix]+=step;
            error[row*NS+ix]-=step;
            ell1+=mj_absf(step);
        }
    }
    *out_ell1=ell1; *out_alpha=alpha;
    free(dr); free(ph); free(pg);
}

void mj_denoise(const f32 *image, f32 *denoised, DenoiserParams *dp)
{
    i32 nz=dp->nz, ny=dp->ny, nx=dp->nx;
    i32 total=nz*ny*nx, flat_rows=nz*ny;

    if(dp->sigma_noise<=0.f){
        dp->sigma_noise=mj_estimate_image_noise_std(image,nz,ny,nx);
        if(dp->verbose) printf("[mbirjax] sigma_noise=%.6f\n",dp->sigma_noise);
    }
    dp->sigma_y=dp->sigma_noise;
    dp->sigma_x=mj_maxf(dp->sigma_noise*powf(2.f,-dp->sharpness),1e-10f);

    f32 fm=1.f/(dp->sigma_y*dp->sigma_y+MJ_EPS);
    QGGMRFParams qp;
    qp.sigma_x=dp->sigma_x; qp.p=dp->p; qp.q=dp->q; qp.T=dp->T;
    vcd_init_nbr_weights(&qp);

    f32 *flat=(f32*)malloc(total*sizeof(f32));
    memcpy(flat,image,total*sizeof(f32));
    f32 *err=(f32*)calloc(total,sizeof(f32));

    i32 gran=dp->granularity;
    i32 ns=(flat_rows+gran-1)/gran;
    i32 iter;
    f32 stop=dp->stop_threshold_pct/100.f;

    for(iter=0;iter<dp->max_iterations;iter++){
        f32 ell1=0.f, alpha=0.f;
        for(i32 s=0;s<ns;s++){
            i32 start=s*gran, end=start+gran<flat_rows?start+gran:flat_rows;
            i32 cnt=end-start;
            i32 *idx=(i32*)malloc(cnt*sizeof(i32));
            for(i32 k=0;k<cnt;k++) idx[k]=start+k;
            f32 e1=0.f,a1=0.f;
            denoise_subset(flat,err,idx,cnt,fm,&qp,nz,ny,nx,&e1,&a1);
            ell1+=e1; alpha+=a1; free(idx);
        }
        f32 l1=mj_sum_abs(flat,total);
        f32 nmae=(l1>0.f)?ell1/l1:0.f;
        if(dp->verbose&&iter%5==0) printf("[mbirjax] denoise iter %d  pct=%.4f\n",iter,100.f*nmae);
        if(nmae<stop){iter++;break;}
    }
    if(dp->verbose) printf("[mbirjax] Denoising done (%d iters)\n",iter);
    memcpy(denoised,flat,total*sizeof(f32));
    free(err); free(flat);
}

static int cmp_f32(const void *a, const void *b)
{ f32 fa=*(const f32*)a, fb=*(const f32*)b; return (fa>fb)-(fa<fb); }

void mj_median_filter3d(const f32 *x, f32 *out, i32 nz, i32 ny, i32 nx,
                         int rmm, f32 *omin, f32 *omax)
{
    f32 win[27];
    for(i32 iz=0;iz<nz;iz++) for(i32 iy=0;iy<ny;iy++) for(i32 ix=0;ix<nx;ix++){
        i32 k=0;
        for(i32 dz=-1;dz<=1;dz++) for(i32 dy=-1;dy<=1;dy++) for(i32 dx=-1;dx<=1;dx++){
            i32 jz=mj_clamp(iz+dz,0,nz-1);
            i32 jy=mj_clamp(iy+dy,0,ny-1);
            i32 jx=mj_clamp(ix+dx,0,nx-1);
            win[k++]=x[(jz*ny+jy)*nx+jx];
        }
        qsort(win,27,sizeof(f32),cmp_f32);
        i32 lin=(iz*ny+iy)*nx+ix;
        out[lin]=win[13];
        if(rmm){if(omin)omin[lin]=win[0];if(omax)omax[lin]=win[26];}
    }
}
