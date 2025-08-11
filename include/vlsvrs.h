#pragma once
#include "stddef.h"
/*
WARNING
The ownership of the pointers returned is passed to the c callsite
So it is the user's responsibillity to free the pointers!!!
*/

typedef struct {
  size_t nx;
  size_t ny;
  size_t nz;
  size_t nc;
  double xmin;
  double ymin;
  double zmin;
  double xmax;
  double ymax;
  double zmax;
  float *data;
}Grid32;

typedef struct {
  size_t nx;
  size_t ny;
  size_t nz;
  size_t nc;
  double xmin;
  double ymin;
  double zmin;
  double xmax;
  double ymax;
  double zmax;
  float *data;
}Grid64;

Grid32 read_var_32(const char *fname, const char *varname, int op);
Grid64 read_var_64(const char *fname, const char *varname, int op);
Grid32 read_vdf_32(const char *fname, const char *pop, size_t cid);
Grid64 read_vdf_64(const char *fname, const char *pop, size_t cid);
void read_vdf_into_32(const char *fname, const char *pop, size_t cid,
                      Grid32 *target);
void read_vdf_into_64(const char *fname, const char *pop, size_t cid,
                      Grid64 *target);
double read_scalar_parameter(const char *fname, const char *parameter);
