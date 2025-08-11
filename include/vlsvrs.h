#pragma once
#include "stddef.h"
/*
WARNING
The ownership of the pointers returned is passed to the c callsite
So it is the user's responsibillity to free the pointers!!!
*/
float *read_var_32(const char *filename, const char *varname, size_t *nx,
                   size_t *ny, size_t *nz, size_t *nc, int op);

double *read_var_64(const char *filename, const char *varname, size_t *nx,
                    size_t *ny, size_t *nz, size_t *nc, int op);

float *read_vdf_32(const char *filename, const char *population, size_t cid,
                   size_t *nx, size_t *ny, size_t *nz);

double *read_vdf_64(const char *filename, const char *population, size_t cid,
                   size_t *nx, size_t *ny, size_t *nz);
