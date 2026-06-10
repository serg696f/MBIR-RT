/**
 * mbirjax_types.h
 *
 * Primitive type aliases, compile-time constants, and shared structs
 * for the plain-C MBIRJAX library.
 *
 * Mirrors: mbirjax/_utils.py  (Param namedtuple, default parameter table)
 *
 * Dependency: only <math.h> for inline helpers.
 */
#ifndef MBIRJAX_TYPES_H
#define MBIRJAX_TYPES_H

#include <math.h>
#include <stddef.h>
#include <float.h>

/* =========================================================================
 * Primitive aliases
 * ========================================================================= */
typedef float          f32;
typedef double         f64;
typedef int            i32;
typedef unsigned int   u32;

/* =========================================================================
 * Geometry tag  (mirrors geometry_type string in Python)
 * ========================================================================= */
typedef enum {
    MJ_GEOM_NONE        = 0,
    MJ_GEOM_PARALLEL    = 1,
    MJ_GEOM_CONE        = 2,
    MJ_GEOM_TRANSLATION = 3,
    MJ_GEOM_MULTIAXIS   = 4
} MjGeomType;

/* =========================================================================
 * Shape descriptors
 * ========================================================================= */
/** sinogram_shape = (num_views, num_det_rows, num_det_channels) */
typedef struct { i32 views; i32 rows; i32 channels; } SinoShape;

/** recon_shape = (num_rows, num_cols, num_slices) */
typedef struct { i32 rows; i32 cols; i32 slices; } ReconShape;

static inline i32 sino_numel (SinoShape  s){ return s.views * s.rows * s.channels; }
static inline i32 recon_numel(ReconShape s){ return s.rows * s.cols * s.slices; }

/* =========================================================================
 * Parameter store
 *
 * Mirrors ParameterHandler.params  (dict[str, Param]).
 * Each Param holds a value and a recompile_flag.
 * Values are typed via a union; type tag is stored separately.
 * ========================================================================= */
typedef enum {
    PT_NONE   = 0,
    PT_FLOAT  = 1,   /* f64 scalar                       */
    PT_INT    = 2,   /* i32 scalar                       */
    PT_BOOL   = 3,   /* i32  0/1                         */
    PT_SHAPE  = 4,   /* i32[3] (tuple of 3 ints)         */
    PT_STR    = 5,   /* heap char*                       */
    PT_FARR   = 6,   /* heap f32[], length in farr.n     */
    PT_IARR   = 7    /* heap i32[], length in iarr.n     */
} ParamType;

typedef struct {
    char      key[64];
    ParamType type;
    int       recompile_flag;   /* 1 → projectors need rebuild on change */
    union {
        f64  fval;
        i32  ival;
        i32  bval;
        i32  shape[3];
        char *sval;             /* owned heap string  */
        struct { f32 *data; i32 n; } farr;
        struct { i32 *data; i32 n; } iarr;
    } v;
} Param;

#define MJ_MAX_PARAMS 80

typedef struct {
    Param slots[MJ_MAX_PARAMS];
    i32   n;
} ParamStore;

/* =========================================================================
 * Iteration statistics  (mirrors ReconParams namedtuple)
 * ========================================================================= */
typedef struct {
    i32   num_iterations;
    f32  *pct_change;      /* [num_iterations] heap */
    f32  *alpha_values;    /* [num_iterations] heap */
    f32   fm_rmse;
    f32   prior_loss;
} ReconStats;

/* =========================================================================
 * Default parameter values  (mirrors get_default_params() in _utils.py)
 * ========================================================================= */
#define MJ_DEF_P                1.2f
#define MJ_DEF_Q                2.0f
#define MJ_DEF_T                1.0f
#define MJ_DEF_SIGMA_X          0.0f
#define MJ_DEF_SIGMA_Y          0.0f
#define MJ_DEF_SIGMA_PROX       0.0f
#define MJ_DEF_SHARPNESS        0.0f
#define MJ_DEF_SNR_DB          30.0f
#define MJ_DEF_AUTO_REG         1
#define MJ_DEF_POSITIVITY       0
#define MJ_DEF_GRANULARITY      16
#define MJ_DEF_MAX_OVERRELAX    1.0f
#define MJ_DEF_VERBOSE          1
#define MJ_DEF_DELTA_DET_CHAN   1.0f
#define MJ_DEF_DELTA_DET_ROW    1.0f
#define MJ_DEF_DET_ROW_OFF      0.0f
#define MJ_DEF_DET_CHAN_OFF     0.0f
#define MJ_DEF_DELTA_VOXEL      1.0f
#define MJ_N_NBR               26    /* 3-D 26-neighbourhood */
#define MJ_EPS                  1.192093e-7f  /* FLT_EPSILON */
#define MJ_MAX_PATH             512

/* =========================================================================
 * Inline scalar math helpers  (replace jnp / np primitives)
 * ========================================================================= */
static inline f32 mj_absf (f32 x)            { return x < 0.f ? -x : x; }
static inline f32 mj_minf (f32 a, f32 b)     { return a < b ? a : b; }
static inline f32 mj_maxf (f32 a, f32 b)     { return a > b ? a : b; }
static inline f32 mj_clipf(f32 x,f32 lo,f32 hi){ return mj_maxf(lo,mj_minf(hi,x)); }
static inline i32 mj_clamp(i32 v,i32 lo,i32 hi){ return v<lo?lo:(v>hi?hi:v); }

static inline f32 mj_sum_abs(const f32 *v, i32 n){
    f32 s=0.f; for(i32 i=0;i<n;i++) s+=mj_absf(v[i]); return s;
}
static inline f32 mj_dot(const f32 *a,const f32 *b,i32 n){
    f32 s=0.f; for(i32 i=0;i<n;i++) s+=a[i]*b[i]; return s;
}

#endif /* MBIRJAX_TYPES_H */
