# VLSV Tools

This is a set of tools written mainly for fun but also for
some projects in Vlasiator (Asterix, Faiser...).
A very very nice thing here is that we can actually read
in a VDF into a dense mesh (we can also remap the VDF to a target mesh)
which is handy for training neural nets.

## C Bindings

Toinstall the C bindings system-wide (headers and `vlsvrs` library):

```bash
./install.sh
cc main.c $(pkg-config --cflags --libs vlsvrs)
```
And now you can use:
```c
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
```

## Python Bindings
Install [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin build --release --features with_bindings
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

## 3) MOD_VLSV_EXPORTS
  Creates C and Python interfaces for VLSV_READER.
    **Keywords:**
    read_variable_f64, read_variable_f32, read_vdf_f32, read_vdf_f64
