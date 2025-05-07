#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t nx;
    size_t ny;
    size_t nz;
} GridInfo;

GridInfo read_fsgrid_variable_f32(const char*filename ,const char* var,float**data);


int main(int argc, char** argv){

  const char* file=argv[1];
  float* data=NULL;
  printf("Data = %p\n",data);
  GridInfo g = read_fsgrid_variable_f32(file,"fg_e",&data);
  printf("Data = %p with gridsize [%zu,%zu,%zu]\n",data,g.nx,g.ny,g.nz);
  free(data);
    
}
