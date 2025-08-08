#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

float *read_vg_as_fg_32(const char *, const char *, size_t *, size_t *,
                        size_t *, size_t *);
void vlsvreader_free(float *);

int main(int argc, char **argv) {

  const char *file = argv[1];
  const char *variable = "proton/vg_rho";
  size_t nx = {0};
  size_t ny = {0};
  size_t nz = {0};
  size_t nc = {0};
  while (1 == 1) {
    float *data = read_vg_as_fg_32(file, variable, &nx, &ny, &nz, &nc);
    printf("Got back data [%zu,%zu,%zu] with memory allocated at %p\n", nx, ny,
           nz, data);
    vlsvreader_free(data);
  }
}
