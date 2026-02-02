# VLSVRS

Motivation:
I hate all available methods of reading in VLSV files.
FsGrid is dumped on disk unordered.
SpatialGrid is hard to read because it has AMR.
VDFs are hard to read because they are sparse.
No more! So vlsvrs is a set of tools written mainly for fun but also for
some projects in Vlasiator (Asterix, Faiser...).
A very very nice thing here is that we can actually read
in a VDF into a dense mesh (we can also remap the VDF to a target mesh)
which is handy for training neural nets. And it can just read all you need
with a simple call. And it is not python! (**This did not really age well since most 
people who now use vlsvrs actually only use the python bindings (me included :D )**)

This package is written in rust, so you will need [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).

## C Bindings

To install the C bindings system-wide (headers and `vlsvrs` library):

```bash
./install.sh
```
And now you can use:

```c
/*
WARNING
The ownership of the pointers returned is passed to the c callsite
So it is the user's responsibillity to free the pointers!!!
*/
Grid32 read_var_32(const char *filename, const char *varname, int op);
Grid64 read_var_64(const char *filename, const char *varname, int op);
Grid32 read_vdf_32(const char *filename, const char *population, size_t cid);
Grid64 read_vdf_64(const char *filename, const char *population, size_t cid);
```
Example usage in C:
```c
/* gcc main.c -Wall -Wextra -O3 -lvlsvrs -o bin && ./bin
Output:
  VDF with shape [100,100,100] extents[-3000000.000000,-3000000.000000,-3000000.000000,3000000.000000,3000000.000000,3000000.000000] @0x7c41f502f010
  
  rho with shape [12,8,1] extents[-5250000.000000,-3500000.000000,-437500.000000,5250000.000000,3500000.000000,437500.000000] @0x62ee34227490
  
  velocity with shape [12,8,1] extents[-5250000.000000,-3500000.000000,-437500.000000,5250000.000000,3500000.000000,437500.000000] @0x62ee3420fdf0
  
  Velocity Block Width = 4
  
  Simulation time = 1.019220
*/
#include "stdlib.h"
#include "vlsvrs.h"
#include <stdio.h>

int main(int argc, char **argv) {
  (void)argc;

  // Reading in VDFs
  VLSVRS_Grid32 vdf = read_vdf_32(argv[1], "proton", 32);
  printf("VDF with shape [%zu,%zu,%zu] extents[%f,%f,%f,%f,%f,%f] @%p\n",
         vdf.nx, vdf.ny, vdf.nz, vdf.xmin, vdf.ymin, vdf.zmin, vdf.xmax,
         vdf.ymax, vdf.zmax, vdf.data);

  // Reading in Variables
  VLSVRS_Grid32 rho = read_var_32(argv[1], "proton/vg_rho", 0);
  read_vdf_32(argv[1], "proton", 32);
  printf("rho with shape [%zu,%zu,%zu] extents[%f,%f,%f,%f,%f,%f] @%p\n",
         rho.nx, rho.ny, rho.nz, rho.xmin, rho.ymin, rho.zmin, rho.xmax,
         rho.ymax, rho.zmax, rho.data);

  // Reading in Vy
  VLSVRS_Grid32 velocity = read_var_32(argv[1], "proton/vg_v", 1);
  read_vdf_32(argv[1], "proton", 32);
  printf("velocity with shape [%zu,%zu,%zu] extents[%f,%f,%f,%f,%f,%f] @%p\n",
         velocity.nx, velocity.ny, velocity.nz, velocity.xmin, velocity.ymin,
         velocity.zmin, velocity.xmax, velocity.ymax, velocity.zmax,
         velocity.data);

  // Read WID 
  size_t WID = get_wid(argv[1], "proton");
  printf("Velocity Block Width = %zu \n", WID);

  // Read in a scalar parameter
  double time = read_scalar_parameter(argv[1], "time");
  printf("Simulation time = %f \n", time);

  // RAII?? GG...
  free(vdf.data);
  free(rho.data);
  free(velocity.data);
}
```

## FORTRAN bindings

To install:

```{bash}
./install.sh 
gfortran vlsvrs.f90 -c -O3
```

### Example

```{fortran}
PROGRAM main
    USE vlsvrs
    use iso_c_binding, only : c_null_char
    IMPLICIT NONE
    type(Grid32) :: data
    integer(8) :: i
    data = read_var_32(TRIM("tsi.vlsv")//c_null_char, TRIM("vg_b_vol")//c_null_char, 0)

    WRITE (*, *) data%nx, data%ny, data%nz

    i = 256
    data = read_vdf_32(TRIM("tsi.vlsv")//c_null_char, TRIM("proton")//c_null_char, i)
    WRITE (*, *) data%nx, data%ny, data%nz
END PROGRAM
```

Which can be compiled via:

```{bash}
gfortran vlsvrs.mod main.f90 -Wall -Wextra -Wno-conversion -Wno-c-binding-type -lvlsvrs -o bin
```


The module is built into the `./fortran_bindings` folder. Note the signatures: integer kind is 8 for cell id, and strings in fortran vs c are a bit of black magic, requiring null chars and trim. 

## Python Bindings

With pip:
```bash
git clone https://github.com/kstppd/vlsvrs
cd vlsvrs/
pip install .
```

Or:
```bash
pip install maturin,numpy
git clone https://github.com/kstppd/vlsvrs
cd vlsvrs/
maturin develop -F with_bindings --release
#If you are building on Linux and your kernel (check with ```uname -r```) version is 5.1+ then enable io uring 
maturin develop -F with_bindings,uring --release
```
Now you can do:
```python
import vlsvrs
f=vlsvrs.VlsvFile("bulk.vlsv")
```

## EXAMPLES
```rust
let f = VlsvFile::new("bulk.vlsv").unwrap();
//OP: vec->scalar reduction into first component with  0|1->x(noop) 2->y 3->z 4->magnitude
let OP = 0;
let data:Array4<_> = f.read_variable::<f32>(&varname, Some(OP)).unwrap()
let data:Array4<_> = f.read_vg_variable_as_fg::<f32>(&varname, Some(OP)).unwrap()
let data:Array4<_> = f.read_fsgrid_variable::<f32>(&varname, Some(OP)).unwrap()
let data:Array4<_> = f.read_vdf::<f32>(256, "proton")).unwrap();
```

## 1) MOD_VLSV_READER
  Reads VLSV files and metadata.  
  Can read ordered fsgrid variables.  
  Can read vg variables as fg.  
  Can read dense vdfs and up/down scale them.  
  Has much smaller memory footprint than analysator.
    **Keywords:**
    read_scalar_parameter, read_config, read_version, read_variable_into, get_wid, get_vspace_mesh_bbox, get_spatial_mesh_extents, get_vspace_mesh_extents, get_domain_decomposition, get_max_amr_refinement, get_writting_tasks, get_spatial_mesh_bbox, get_dataset, read_vg_variable_as_fg, read_fsgrid_variable, read_vdf, read_vdf_into, read_variable, read_tag, vg_variable_to_fg

## 2) MOD_VLSV_TRACING
  Particle tracing routines using fields from Vlasiator.
    **Keywords:**
    get_fields_at, new_with_energy_at_Lshell, boris, larmor_radius, borris_adaptive
    ![Demo](video.gif?v=1.1) 

## 3) MOD_VLSV_EXPORTS
  Creates C and Python interfaces for VLSV_READER.
    **Keywords:**
    read_variable_f64, read_variable_f32, read_vdf_f32, read_vdf_f64
