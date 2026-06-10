/**
 * parameter_handler.c
 *
 * Plain-C translation of mbirjax/parameter_handler.py (ParameterHandler class).
 *
 * Every public method maps 1-to-1 to a Python method:
 *   __init__           → ph_init
 *   set_params         → ph_set_{float,int,bool,shape,str,farr,iarr}
 *   get_params         → ph_get_{float,int,bool,shape,str,farr}
 *   print_params       → ph_print
 *   save_params        → ph_save
 *   load_param_dict    → ph_load
 *   auto_set_regularization_params → ph_auto_regularize
 */

#define _POSIX_C_SOURCE 200809L
#include "parameter_handler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Internal helpers ───────────────────────────────────────────────────── */

static void param_free_heap(Param *p)
{
    if (p->type == PT_STR  && p->v.sval)       { free(p->v.sval);       p->v.sval = NULL; }
    if (p->type == PT_FARR && p->v.farr.data)  { free(p->v.farr.data);  p->v.farr.data = NULL; p->v.farr.n = 0; }
    if (p->type == PT_IARR && p->v.iarr.data)  { free(p->v.iarr.data);  p->v.iarr.data = NULL; p->v.iarr.n = 0; }
}

static Param *ph_find_or_alloc(ParamStore *ps, const char *key)
{
    for (i32 i = 0; i < ps->n; i++)
        if (strncmp(ps->slots[i].key, key, 63) == 0)
            return &ps->slots[i];
    if (ps->n >= MJ_MAX_PARAMS) {
        fprintf(stderr, "[mbirjax] ParamStore full, cannot add '%s'\n", key);
        return NULL;
    }
    Param *p = &ps->slots[ps->n++];
    memset(p, 0, sizeof(*p));
    strncpy(p->key, key, 63);
    return p;
}

/* ── Lifecycle ─────────────────────────────────────────────────────────── */

void ph_init(ParamStore *ps)
{
    memset(ps, 0, sizeof(*ps));

    /* Geometry / meta */
    ph_set_int  (ps, "geometry_type",      MJ_GEOM_NONE,         1);
    ph_set_str  (ps, "file_format",        "binary",             0);

    /* Sinogram shape: (views, det_rows, det_channels) */
    ph_set_shape(ps, "sinogram_shape",     0, 0, 0,              1);

    /* Detector geometry */
    ph_set_float(ps, "delta_det_channel",  MJ_DEF_DELTA_DET_CHAN, 1);
    ph_set_float(ps, "delta_det_row",      MJ_DEF_DELTA_DET_ROW,  1);
    ph_set_float(ps, "det_row_offset",     MJ_DEF_DET_ROW_OFF,    1);
    ph_set_float(ps, "det_channel_offset", MJ_DEF_DET_CHAN_OFF,   1);

    /* Reconstruction geometry */
    ph_set_shape(ps, "recon_shape",        0, 0, 0,              1);
    ph_set_float(ps, "delta_voxel",        MJ_DEF_DELTA_VOXEL,   1);

    /* Regularisation */
    ph_set_float(ps, "sigma_y",            MJ_DEF_SIGMA_Y,       0);
    ph_set_float(ps, "sigma_x",            MJ_DEF_SIGMA_X,       0);
    ph_set_float(ps, "sigma_prox",         MJ_DEF_SIGMA_PROX,    0);
    ph_set_float(ps, "p",                  MJ_DEF_P,             0);
    ph_set_float(ps, "q",                  MJ_DEF_Q,             0);
    ph_set_float(ps, "T",                  MJ_DEF_T,             0);
    ph_set_float(ps, "sharpness",          MJ_DEF_SHARPNESS,     0);
    ph_set_float(ps, "snr_db",             MJ_DEF_SNR_DB,        0);
    ph_set_bool (ps, "auto_regularize_flag", MJ_DEF_AUTO_REG,    0);
    ph_set_bool (ps, "positivity_flag",    MJ_DEF_POSITIVITY,    0);

    /* qGGMRF 26-neighbour weights — ones by default */
    {
        f32 nbr[MJ_N_NBR];
        for (i32 k = 0; k < MJ_N_NBR; k++) nbr[k] = 1.f;
        ph_set_farr(ps, "qggmrf_nbr_wts", nbr, MJ_N_NBR, 0);
    }

    /* ALU */
    ph_set_str  (ps, "alu_unit",           "mm",                 0);
    ph_set_float(ps, "alu_value",          1.0,                  0);

    /* VCD / solver */
    {
        i32 gran[1] = { MJ_DEF_GRANULARITY };
        ph_set_iarr(ps, "granularity", gran, 1, 0);
        i32 pseq[1] = { 0 };
        ph_set_iarr(ps, "partition_sequence", pseq, 1, 0);
    }
    ph_set_float(ps, "max_overrelaxation", MJ_DEF_MAX_OVERRELAX, 0);

    /* Misc */
    ph_set_int  (ps, "verbose",            MJ_DEF_VERBOSE,       0);
    ph_set_str  (ps, "use_gpu",            "none",               0);
    ph_set_str  (ps, "view_params_name",   "angles",             1);
}

void ph_free(ParamStore *ps)
{
    for (i32 i = 0; i < ps->n; i++) param_free_heap(&ps->slots[i]);
    memset(ps, 0, sizeof(*ps));
}

void ph_copy(ParamStore *dst, const ParamStore *src)
{
    memset(dst, 0, sizeof(*dst));
    for (i32 i = 0; i < src->n; i++) {
        const Param *s = &src->slots[i];
        switch (s->type) {
            case PT_FLOAT: ph_set_float(dst,s->key,s->v.fval,s->recompile_flag); break;
            case PT_INT:   ph_set_int  (dst,s->key,s->v.ival,s->recompile_flag); break;
            case PT_BOOL:  ph_set_bool (dst,s->key,s->v.bval,s->recompile_flag); break;
            case PT_SHAPE: ph_set_shape(dst,s->key,s->v.shape[0],s->v.shape[1],s->v.shape[2],s->recompile_flag); break;
            case PT_STR:   ph_set_str  (dst,s->key,s->v.sval,s->recompile_flag); break;
            case PT_FARR:  ph_set_farr (dst,s->key,s->v.farr.data,s->v.farr.n,s->recompile_flag); break;
            case PT_IARR:  ph_set_iarr (dst,s->key,s->v.iarr.data,s->v.iarr.n,s->recompile_flag); break;
            default: break;
        }
    }
}

/* ── Typed setters ──────────────────────────────────────────────────────── */

int ph_set_float(ParamStore *ps, const char *key, f64 val, int recompile)
{
    Param *p = ph_find_or_alloc(ps, key);
    if (!p) return -1;
    param_free_heap(p);
    p->type = PT_FLOAT; p->recompile_flag = recompile; p->v.fval = val;
    return 0;
}

int ph_set_int(ParamStore *ps, const char *key, i32 val, int recompile)
{
    Param *p = ph_find_or_alloc(ps, key);
    if (!p) return -1;
    param_free_heap(p);
    p->type = PT_INT; p->recompile_flag = recompile; p->v.ival = val;
    return 0;
}

int ph_set_bool(ParamStore *ps, const char *key, int val, int recompile)
{
    Param *p = ph_find_or_alloc(ps, key);
    if (!p) return -1;
    param_free_heap(p);
    p->type = PT_BOOL; p->recompile_flag = recompile; p->v.bval = val ? 1 : 0;
    return 0;
}

int ph_set_shape(ParamStore *ps, const char *key, i32 a, i32 b, i32 c, int recompile)
{
    Param *p = ph_find_or_alloc(ps, key);
    if (!p) return -1;
    param_free_heap(p);
    p->type = PT_SHAPE; p->recompile_flag = recompile;
    p->v.shape[0]=a; p->v.shape[1]=b; p->v.shape[2]=c;
    return 0;
}

int ph_set_str(ParamStore *ps, const char *key, const char *val, int recompile)
{
    Param *p = ph_find_or_alloc(ps, key);
    if (!p) return -1;
    param_free_heap(p);
    p->type = PT_STR; p->recompile_flag = recompile;
    p->v.sval = strdup(val ? val : "");
    return 0;
}

int ph_set_farr(ParamStore *ps, const char *key, const f32 *data, i32 n, int recompile)
{
    Param *p = ph_find_or_alloc(ps, key);
    if (!p) return -1;
    param_free_heap(p);
    p->type = PT_FARR; p->recompile_flag = recompile;
    p->v.farr.n = n;
    p->v.farr.data = (f32*)malloc(n * sizeof(f32));
    if (!p->v.farr.data) return -1;
    memcpy(p->v.farr.data, data, n * sizeof(f32));
    return 0;
}

int ph_set_iarr(ParamStore *ps, const char *key, const i32 *data, i32 n, int recompile)
{
    Param *p = ph_find_or_alloc(ps, key);
    if (!p) return -1;
    param_free_heap(p);
    p->type = PT_IARR; p->recompile_flag = recompile;
    p->v.iarr.n = n;
    p->v.iarr.data = (i32*)malloc(n * sizeof(i32));
    if (!p->v.iarr.data) return -1;
    memcpy(p->v.iarr.data, data, n * sizeof(i32));
    return 0;
}

/* ── Typed getters ──────────────────────────────────────────────────────── */

const Param *ph_find(const ParamStore *ps, const char *key)
{
    for (i32 i = 0; i < ps->n; i++)
        if (strncmp(ps->slots[i].key, key, 63) == 0)
            return &ps->slots[i];
    return NULL;
}

f64 ph_get_float(const ParamStore *ps, const char *key)
{
    const Param *p = ph_find(ps, key);
    if (!p) { fprintf(stderr,"[mbirjax] get_float: '%s' not found\n",key); return 0.0; }
    if (p->type == PT_FLOAT) return p->v.fval;
    if (p->type == PT_INT)   return (f64)p->v.ival;
    if (p->type == PT_BOOL)  return (f64)p->v.bval;
    fprintf(stderr,"[mbirjax] get_float: '%s' type mismatch\n",key); return 0.0;
}

i32 ph_get_int(const ParamStore *ps, const char *key)
{
    const Param *p = ph_find(ps, key);
    if (!p) { fprintf(stderr,"[mbirjax] get_int: '%s' not found\n",key); return 0; }
    if (p->type == PT_INT)   return p->v.ival;
    if (p->type == PT_FLOAT) return (i32)p->v.fval;
    if (p->type == PT_BOOL)  return p->v.bval;
    fprintf(stderr,"[mbirjax] get_int: '%s' type mismatch\n",key); return 0;
}

int ph_get_bool(const ParamStore *ps, const char *key)
{
    const Param *p = ph_find(ps, key);
    if (!p) { fprintf(stderr,"[mbirjax] get_bool: '%s' not found\n",key); return 0; }
    if (p->type == PT_BOOL)  return p->v.bval;
    if (p->type == PT_INT)   return p->v.ival != 0;
    if (p->type == PT_FLOAT) return p->v.fval != 0.0;
    return 0;
}

int ph_get_shape(const ParamStore *ps, const char *key, i32 out[3])
{
    const Param *p = ph_find(ps, key);
    if (!p || p->type != PT_SHAPE) return -1;
    out[0]=p->v.shape[0]; out[1]=p->v.shape[1]; out[2]=p->v.shape[2];
    return 0;
}

const char *ph_get_str(const ParamStore *ps, const char *key)
{
    const Param *p = ph_find(ps, key);
    if (!p || p->type != PT_STR) return NULL;
    return p->v.sval;
}

int ph_get_farr(const ParamStore *ps, const char *key, f32 *out, i32 *n_out)
{
    const Param *p = ph_find(ps, key);
    if (!p || p->type != PT_FARR) return -1;
    if (n_out) *n_out = p->v.farr.n;
    if (out)   memcpy(out, p->v.farr.data, p->v.farr.n * sizeof(f32));
    return 0;
}

/* ── Print  (mirrors print_params) ─────────────────────────────────────── */

void ph_print(const ParamStore *ps)
{
    printf("---- Parameters (%d) ----\n", ps->n);
    for (i32 i = 0; i < ps->n; i++) {
        const Param *p = &ps->slots[i];
        printf("  %-30s = ", p->key);
        switch (p->type) {
        case PT_FLOAT: printf("%.7g\n",  p->v.fval); break;
        case PT_INT:   printf("%d\n",    p->v.ival); break;
        case PT_BOOL:  printf("%s\n",    p->v.bval ? "true" : "false"); break;
        case PT_STR:   printf("\"%s\"\n",p->v.sval ? p->v.sval : ""); break;
        case PT_SHAPE: printf("(%d,%d,%d)\n",p->v.shape[0],p->v.shape[1],p->v.shape[2]); break;
        case PT_FARR: {
            printf("[");
            for (i32 k=0;k<p->v.farr.n;k++) printf(k?",%.4g":"%.4g",p->v.farr.data[k]);
            printf("]\n"); break;
        }
        case PT_IARR: {
            printf("[");
            for (i32 k=0;k<p->v.iarr.n;k++) printf(k?",%d":"%d",p->v.iarr.data[k]);
            printf("]\n"); break;
        }
        default: printf("<unknown>\n");
        }
    }
    printf("----\n");
}

/* ── File I/O  (simple "key type value..." text format) ────────────────── */
/*
 * Mirrors save_params() / load_param_dict() but without YAML dependency.
 * Format per line:
 *   key float 1.234567e+00
 *   key int 42
 *   key bool 1
 *   key shape 64 64 32
 *   key str hello
 *   key farr 26  0.1 0.2 ...
 *   key iarr  3  4 8 16
 */

int ph_save(const ParamStore *ps, const char *filepath)
{
    FILE *f = fopen(filepath, "w");
    if (!f) { perror(filepath); return -1; }
    for (i32 i = 0; i < ps->n; i++) {
        const Param *p = &ps->slots[i];
        switch (p->type) {
        case PT_FLOAT: fprintf(f,"%s float %.10g\n",p->key,p->v.fval); break;
        case PT_INT:   fprintf(f,"%s int %d\n",p->key,p->v.ival); break;
        case PT_BOOL:  fprintf(f,"%s bool %d\n",p->key,p->v.bval); break;
        case PT_STR:   fprintf(f,"%s str %s\n",p->key,p->v.sval?p->v.sval:""); break;
        case PT_SHAPE: fprintf(f,"%s shape %d %d %d\n",p->key,p->v.shape[0],p->v.shape[1],p->v.shape[2]); break;
        case PT_FARR:
            fprintf(f,"%s farr %d",p->key,p->v.farr.n);
            for(i32 k=0;k<p->v.farr.n;k++) fprintf(f," %.10g",p->v.farr.data[k]);
            fprintf(f,"\n"); break;
        case PT_IARR:
            fprintf(f,"%s iarr %d",p->key,p->v.iarr.n);
            for(i32 k=0;k<p->v.iarr.n;k++) fprintf(f," %d",p->v.iarr.data[k]);
            fprintf(f,"\n"); break;
        default: break;
        }
    }
    fclose(f); return 0;
}

int ph_load(ParamStore *ps, const char *filepath)
{
    FILE *f = fopen(filepath, "r");
    if (!f) { perror(filepath); return -1; }
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        char key[64], type[16];
        if (sscanf(line, "%63s %15s", key, type) != 2) continue;
        /* Advance past "key type " */
        char *rest = line;
        while (*rest && *rest != ' ') rest++;  /* skip key */
        while (*rest == ' ') rest++;
        while (*rest && *rest != ' ') rest++;  /* skip type */
        while (*rest == ' ') rest++;

        if (!strcmp(type,"float")) {
            double v; sscanf(rest,"%lf",&v); ph_set_float(ps,key,v,1);
        } else if (!strcmp(type,"int")) {
            int v; sscanf(rest,"%d",&v); ph_set_int(ps,key,v,1);
        } else if (!strcmp(type,"bool")) {
            int v; sscanf(rest,"%d",&v); ph_set_bool(ps,key,v,1);
        } else if (!strcmp(type,"str")) {
            char s[512]; sscanf(rest,"%511s",s); ph_set_str(ps,key,s,1);
        } else if (!strcmp(type,"shape")) {
            int a,b,c; sscanf(rest,"%d %d %d",&a,&b,&c); ph_set_shape(ps,key,a,b,c,1);
        } else if (!strcmp(type,"farr")) {
            int n; sscanf(rest,"%d",&n);
            f32 *arr = (f32*)malloc(n*sizeof(f32));
            char *p2 = rest; while(*p2&&*p2!=' ')p2++; /* skip n */
            for(int k=0;k<n;k++){ double v; sscanf(p2," %lf",&v); arr[k]=(f32)v; while(*p2==' ')p2++; while(*p2&&*p2!=' ')p2++; }
            ph_set_farr(ps,key,arr,n,1); free(arr);
        } else if (!strcmp(type,"iarr")) {
            int n; sscanf(rest,"%d",&n);
            i32 *arr = (i32*)malloc(n*sizeof(i32));
            char *p2 = rest; while(*p2&&*p2!=' ')p2++;
            for(int k=0;k<n;k++){ int v; sscanf(p2," %d",&v); arr[k]=v; while(*p2==' ')p2++; while(*p2&&*p2!=' ')p2++; }
            ph_set_iarr(ps,key,arr,n,1); free(arr);
        }
    }
    fclose(f); return 0;
}

/* ── Auto-regularisation ─────────────────────────────────────────────── */
/*
 * Mirrors TomographyModel.auto_set_regularization_params():
 *
 *   sigma_y = typical_value / 10^(snr_db/20)
 *   sigma_x = typical_value * (delta_det_channel/delta_voxel) * 2^(-sharpness)
 */
void ph_auto_regularize(ParamStore *ps, f32 typical_value)
{
    f64 snr_db    = ph_get_float(ps, "snr_db");
    f64 sharpness = ph_get_float(ps, "sharpness");
    f64 ddc       = ph_get_float(ps, "delta_det_channel");
    f64 dv        = ph_get_float(ps, "delta_voxel");

    if (typical_value <= 0.f) typical_value = 1.f;

    f64 sigma_y = typical_value / pow(10.0, snr_db / 20.0);
    if (sigma_y < 1e-10) sigma_y = 1e-10;

    f64 voxel_factor = (dv > 0.0 && ddc > 0.0) ? ddc / dv : 1.0;
    f64 sigma_x = typical_value * voxel_factor * pow(2.0, -sharpness);
    if (sigma_x < 1e-10) sigma_x = 1e-10;

    ph_set_float(ps, "sigma_y", sigma_y, 0);
    ph_set_float(ps, "sigma_x", sigma_x, 0);
}
