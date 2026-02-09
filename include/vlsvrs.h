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
  void *data;
  size_t datasize;  
} VLSVRS_GenericGrid;

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
} VLSVRS_Grid32;

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
  double *data;
} VLSVRS_Grid64;

VLSVRS_GenericGrid read_var(const char *fname, const char *varname, int op);
VLSVRS_Grid32 read_var_32(const char *fname, const char *varname, int op);
VLSVRS_Grid64 read_var_64(const char *fname, const char *varname, int op);
VLSVRS_GenericGrid read_vdf(const char *fname, const char *pop, size_t cid);
VLSVRS_Grid32 read_vdf_32(const char *fname, const char *pop, size_t cid);
VLSVRS_Grid64 read_vdf_64(const char *fname, const char *pop, size_t cid);
void read_vdf_into_32(const char *fname, const char *pop, size_t cid,
                      VLSVRS_Grid32 *target);
void read_vdf_into_64(const char *fname, const char *pop, size_t cid,
                      VLSVRS_Grid64 *target);
double read_scalar_parameter(const char *fname, const char *parameter);
size_t get_wid(const char *fname, const char *pop);

#ifdef VLSVRS_STRIP_PREFIX
#define VLSVRS_Grid32 Grid32
#define VLSVRS_Grid64 Grid64
#endif
