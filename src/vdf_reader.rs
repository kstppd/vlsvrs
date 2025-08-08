mod vlsv_reader;
use ndarray::{Array3, s};
use ndarray_npy::write_npy;
use vlsv_reader::vlsv_reader::VlsvFile;

pub fn bulk_velocity(
    vdf: &Array3<f32>,
    mesh: (f64, f64, f64, f64, f64, f64),
    sparse: f32,
) -> Option<([f64; 3], [usize; 3])> {
    #[inline]
    fn trap_weight(idx: usize, n: usize) -> f64 {
        if idx == 0 || idx + 1 == n { 0.5 } else { 1.0 }
    }

    let (vxmin, vymin, vzmin, vxmax, vymax, vzmax) = mesh;
    let (nx, ny, nz) = vdf.dim();
    if nx < 2 || ny < 2 || nz < 2 {
        return None;
    }

    let hx = (vxmax - vxmin) / (nx as f64 - 1.0);
    let hy = (vymax - vymin) / (ny as f64 - 1.0);
    let hz = (vzmax - vzmin) / (nz as f64 - 1.0);

    let mut m0 = 0.0f64;
    let mut mx = 0.0f64;
    let mut my = 0.0f64;
    let mut mz = 0.0f64;

    for i in 0..nx {
        let wx = trap_weight(i, nx);
        let vx = vxmin + (i as f64) * hx;
        for j in 0..ny {
            let wy = trap_weight(j, ny);
            let vy = vymin + (j as f64) * hy;
            let wxy = wx * wy;
            for k in 0..nz {
                let wz = trap_weight(k, nz);
                let w = wxy * wz;
                let vz = vzmin + (k as f64) * hz;

                let mut f = vdf[(i, j, k)] as f64;
                if (f as f32) < sparse {
                    f = 0.0;
                }

                m0 += w * f;
                mx += w * vx * f;
                my += w * vy * f;
                mz += w * vz * f;
            }
        }
    }
    let scale = hx * hy * hz;
    m0 *= scale;
    mx *= scale;
    my *= scale;
    mz *= scale;

    if !m0.is_finite() || m0 == 0.0 {
        return None;
    }

    let bvx = mx / m0;
    let bvy = my / m0;
    let bvz = mz / m0;
    let ix = ((bvx - vxmin) / hx).clamp(0.0, (nx - 1) as f64).round() as usize;
    let iy = ((bvy - vymin) / hy).clamp(0.0, (ny - 1) as f64).round() as usize;
    let iz = ((bvz - vzmin) / hz).clamp(0.0, (nz - 1) as f64).round() as usize;
    Some(([bvx, bvy, bvz], [ix, iy, iz]))
}

pub fn vdf_bounding_box(vdf: &Array3<f32>, sparse: f32) -> Option<[usize; 6]> {
    let (nx, ny, nz) = vdf.dim();

    let mut min_x = nx;
    let mut min_y = ny;
    let mut min_z = nz;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut max_z = 0;

    let mut found = false;

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if vdf[(x, y, z)] > sparse {
                    if x < min_x {
                        min_x = x;
                    }
                    if y < min_y {
                        min_y = y;
                    }
                    if z < min_z {
                        min_z = z;
                    }
                    if x > max_x {
                        max_x = x;
                    }
                    if y > max_y {
                        max_y = y;
                    }
                    if z > max_z {
                        max_z = z;
                    }
                    found = true;
                }
            }
        }
    }

    if found {
        Some([min_x, min_y, min_z, max_x, max_y, max_z])
    } else {
        None
    }
}

fn main() {
    let args = std::env::args()
        .into_iter()
        .skip(1)
        .collect::<Vec<String>>();
    let f = VlsvFile::new(&args[0]).unwrap();
    // let vdf = f.read_vdf(256, "proton").expect("No VDF in CellID");
    // let mesh = f.get_vspace_mesh_extents("proton").unwrap();
    // let (bulk_v, loc) = bulk_velocity(&vdf, mesh, 1e-16).expect("bulk velocity failed");
    // if let Some(bbox) = vdf_bounding_box(&vdf, 1e-16) {
    //     println!("Bounding box: {:?}", bbox);
    // } else {
    //     println!("No values above sparsity threshold");
    // }

    // println!("bulk v = {:?}, at indices {:?}", bulk_v, loc);
    // println!("vdf size = {}", vdf.len());
    // ndarray_npy::write_npy("vdf.npy", &vdf).unwrap();
    // println!("{},", f.read_scalar_parameter("time").unwrap());
    let a = f.read_vg_variable_as_fg::<f32>("proton/vg_rho").unwrap();
    ndarray_npy::write_npy("vg.npy", &a).unwrap();
    println!("{:?}", a.dim());
}
