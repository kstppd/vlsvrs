# VLSVRS

![Demo](video.gif?v=1.1)

Motivation: I hate all available methods of reading in VLSV files. FsGrid is
dumped on disk unordered. SpatialGrid is hard to read because it has AMR. VDFs
are hard to read because they are sparse. No more! So vlsvrs is a set of tools
written mainly for fun but also for some projects in Vlasiator (Asterix,
Faiser...). A very very nice thing here is that we can actually read in a VDF
into a dense mesh (we can also remap the VDF to a target mesh) which is handy
for training neural nets. And it can just read all you need with a simple call.
And it is not python! (**This did not really age well since most people who now
use vlsvrs actually only use the python bindings (me included :D )**)

This package is written in rust, so you will need [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html).

## Build instructions

Clone the repository:

```bash
git clone --recurse-submodules  https://github.com/kstppd/vlsvrs.git 
cd vlsvrs
```

If you do not need to read compressed VDFs with Asterix, you can skip the
optional dependencies to speed up compilation:

```bash
export VLSVRS_SKIP_OPTIONAL=1
```

### Python Bindings

If you only want to use `vlsvrs` from Python, create a virtual environment and
install the package:

```bash
python3 -m venv env
source env/bin/activate
pip install .
```

### Rust binaries

For more control, `vlsvrs` provides these binaries. These can be built
individually by enabling the corresponding Cargo feature.

| Binary | Feature | Source | Description |
|---|---|---|---|
| `vlsv_dump` | `vlsv_dump` | `src/vlsv_dump.rs` | CLI tool to see what is inside vlsv files |
| `vlsv_diff` | `vlsv_dump` | `src/vlsv_diff.rs` | diffs vlsv files |
| `vlsv_particle_sampler` | `vlsv_ptr` | `src/restart_particle_sampler.rs` | samples particles from VDFs |
| `vlsv_ptr_gc2d` | `vlsv_ptr` | `src/vlsv_ptr_gc2d.rs` | GC particle tracer|
| `vlsv_tracer` | `vlsv_ptr` | `src/vlsv_tracer.rs` | full 3D particle tracer |
| `vlsv_field_line_tracer` | `vlsv_ptr` | `src/vlsv_field_line_tracer.rs` | 3D field line tracer |
| `vlsv_view` | `vlsv_view` | `src/vlsv_view.rs` | cli tool to quickly visualize vlsv files using raylib|

For example, to build `vlsv_tracer`:

```bash
cargo build --release --features vlsv_ptr --bin vlsv_tracer
```

Or to build `vlsv_view`:

```bash
cargo build --release --features vlsv_view --bin vlsv_view
```

To build all available binaries, enable all features:

```bash
cargo build --release --all-features --bins
```

To build the library as well:

```bash
cargo build --release --all-features
```

### C and FORTRAN bindings

#### C Bindings

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
/* gcc main.c -Wall -Wextra -O3 -lvlsvrs -o bin && ./bin Output: VDF with shape
[100,100,100]
extents[-3000000.000000,-3000000.000000,-3000000.000000,3000000.000000,3000000.000000,3000000.000000]
@0x7c41f502f010

rho with shape [12,8,1]
extents[-5250000.000000,-3500000.000000,-437500.000000,5250000.000000,3500000.000000,437500.000000]
@0x62ee34227490

velocity with shape [12,8,1]
extents[-5250000.000000,-3500000.000000,-437500.000000,5250000.000000,3500000.000000,437500.000000]
@0x62ee3420fdf0

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

#### FORTRAN bindings

To install:

```text {bash}
./install.sh 
gfortran vlsvrs.f90 -c -O3
```

### Example

```text {fortran}
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

```text {bash}
gfortran vlsvrs.mod main.f90 -Wall -Wextra -Wno-conversion -Wno-c-binding-type
-lvlsvrs -o bin
```

The module is built into the `./fortran_bindings` folder. Note the signatures:
integer kind is 8 for cell id, and strings in fortran vs c are a bit of black
magic, requiring null chars and trim.
