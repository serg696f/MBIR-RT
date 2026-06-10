/**
 * parameter_handler.h
 *
 * Plain-C translation of mbirjax/parameter_handler.py  (ParameterHandler class)
 *
 * Provides a typed key-value parameter store with:
 *   ph_init()              — populate with library defaults
 *   ph_free()              — release heap memory
 *   ph_set_{float,int,bool,shape,str,farr,iarr}()  — typed setters
 *   ph_get_{float,int,bool,shape,str,farr}()        — typed getters
 *   ph_find()              — look up a Param slot by name (NULL = not found)
 *   ph_print()             — print all params to stdout
 *   ph_save() / ph_load()  — simple text file persistence
 *   ph_copy()              — deep copy
 *   ph_auto_regularize()   — compute sigma_x / sigma_y from meta-params
 *
 * Mirrors ParameterHandler.set_params / get_params / save_params / load_param_dict.
 */
#ifndef PARAMETER_HANDLER_H
#define PARAMETER_HANDLER_H

#include "mbirjax_types.h"
#include <stdio.h>

/* ── Lifecycle ─────────────────────────────────────────────────────────── */
void ph_init(ParamStore *ps);
void ph_free(ParamStore *ps);
void ph_copy(ParamStore *dst, const ParamStore *src);

/* ── Typed setters  (return 0 on success, -1 on overflow) ─────────────── */
int ph_set_float(ParamStore *ps, const char *key, f64 val, int recompile);
int ph_set_int  (ParamStore *ps, const char *key, i32 val, int recompile);
int ph_set_bool (ParamStore *ps, const char *key, int val, int recompile);
int ph_set_shape(ParamStore *ps, const char *key, i32 a, i32 b, i32 c, int recompile);
int ph_set_str  (ParamStore *ps, const char *key, const char *val, int recompile);
int ph_set_farr (ParamStore *ps, const char *key, const f32 *data, i32 n, int recompile);
int ph_set_iarr (ParamStore *ps, const char *key, const i32 *data, i32 n, int recompile);

/* ── Typed getters ────────────────────────────────────────────────────── */
/** Return NULL if key not present. */
const Param *ph_find(const ParamStore *ps, const char *key);

f64         ph_get_float(const ParamStore *ps, const char *key);
i32         ph_get_int  (const ParamStore *ps, const char *key);
int         ph_get_bool (const ParamStore *ps, const char *key);
/** Fill out[3]; return 0 on success, -1 if missing/wrong type. */
int         ph_get_shape(const ParamStore *ps, const char *key, i32 out[3]);
const char *ph_get_str  (const ParamStore *ps, const char *key);
/** Fill out[] with at most *n_out values; set *n_out to actual count. */
int         ph_get_farr (const ParamStore *ps, const char *key, f32 *out, i32 *n_out);

/* ── I/O ──────────────────────────────────────────────────────────────── */
void ph_print(const ParamStore *ps);
int  ph_save (const ParamStore *ps, const char *filepath);
int  ph_load (ParamStore *ps, const char *filepath);

/* ── Auto-regularisation ──────────────────────────────────────────────── */
/**
 * Compute sigma_y and sigma_x from snr_db, sharpness, delta_det_channel,
 * delta_voxel, and a representative image magnitude.
 * Mirrors TomographyModel.auto_set_regularization_params().
 */
void ph_auto_regularize(ParamStore *ps, f32 typical_value);

#endif /* PARAMETER_HANDLER_H */
