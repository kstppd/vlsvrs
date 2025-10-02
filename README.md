# VLSV Tools

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
with a simple call. And it is not python!

This package is written in rust, so you will need [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).

## C Bindings

Toinstall the C bindings system-wide (headers and `vlsvrs` library):

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
// gcc main.c -Wall -Wextra -O3 -lvlsvrs -o bin && ./bin
#include "stdlib.h"
#include "vlsvrs.h"
#include <stdio.h>

int main() {
  Grid32 vdf1 =
      read_vdf_32("tsi.vlsv", "proton", 256);
  Grid64 vdf2 =
      read_vdf_64("tsi.vlsv", "proton", 256);

  printf("VDF1 with shape [%zu,%zu,%zu] extents[%f,%f,%f,%f,%f,%f] @%p\n",
         vdf1.nx, vdf1.ny, vdf1.nz, vdf1.xmin, vdf1.ymin, vdf1.zmin, vdf1.xmax,
         vdf1.ymax, vdf1.zmax, vdf1.data);
  printf("VDF2 with shape [%zu,%zu,%zu] extents[%f,%f,%f,%f,%f,%f] @%p\n",
         vdf2.nx, vdf2.ny, vdf2.nz, vdf2.xmin, vdf2.ymin, vdf2.zmin, vdf2.xmax,
         vdf2.ymax, vdf2.zmax, vdf2.data);
  printf("Hello World!\n");
  free(vdf1.data);
  free(vdf2.data);
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
    ![Demo](video.gif) 

## 3) MOD_VLSV_EXPORTS
  Creates C and Python interfaces for VLSV_READER.
    **Keywords:**
    read_variable_f64, read_variable_f32, read_vdf_f32, read_vdf_f64
