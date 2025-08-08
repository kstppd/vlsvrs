pub mod vlsv_reader {
    use crate::vlsv_reader;
    use bytemuck::Pod;
    use ndarray::{Array4, s};
    use num_traits::Float;
    use vlsv_reader::vlsv_reader::VlsvFile;

    pub trait PtrTrait:
        Float
        + Pod
        + Send
        + Sync
        + Sized
        + std::fmt::Debug
        + std::fmt::Display
        + num_traits::ToBytes
    {
    }

    impl<T> PtrTrait for T where
        T: Float
            + Pod
            + Send
            + Sync
            + Sized
            + std::fmt::Debug
            + std::fmt::Display
            + num_traits::ToBytes
    {
    }

    pub trait Field<T: PtrTrait> {
        fn get_fields_at(&self, time: T, x: T, y: T, z: T) -> Option<[T; 6]>;
    }

    pub struct DipoleField<T: PtrTrait> {
        pub moment: T,
    }

    impl<T: PtrTrait> DipoleField<T> {
        pub fn new(moment: T) -> Self {
            DipoleField { moment: moment }
        }
    }

    pub struct VlsvStaticField<T: PtrTrait> {
        b: Array4<T>,
        e: Array4<T>,
        extents: [T; 6],
        ds: T,
    }

    impl<T: PtrTrait> VlsvStaticField<T> {
        pub fn new(filename: &String) -> Self {
            let f = VlsvFile::new(&filename).unwrap();
            let extents: [T; 6] = [
                T::from(f.read_scalar_parameter("xmin").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("ymin").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("zmin").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("xmax").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("ymax").unwrap()).unwrap(),
                T::from(f.read_scalar_parameter("zmax").unwrap()).unwrap(),
            ];
            let b = f.read_fsgrid_variable::<T>("fg_b").unwrap();
            let e = f.read_fsgrid_variable::<T>("fg_e").unwrap();
            let ds = (extents[3] - extents[0]) / T::from(b.dim().0).unwrap();
            VlsvStaticField { b, e, extents, ds }
        }

        fn real2mesh(&self, x: T, y: T, z: T) -> Option<([usize; 3], [T; 3])> {
            if x < self.extents[0]
                || x > self.extents[3]
                || y < self.extents[1]
                || y > self.extents[4]
                || z < self.extents[2]
                || z > self.extents[5]
            {
                // eprintln!(
                //     "ERROR: Tried to probe fields outside mesh at location [{:?},{:?},{:?}]. Mesh extents are {:?}!",
                //     x, y, z, self.extents
                // );
                return None;
            }

            let dims = self.e.dim();
            let x_norm = (x - self.extents[0]) / self.ds;
            let y_norm = (y - self.extents[1]) / self.ds;
            let z_norm = (z - self.extents[2]) / self.ds;
            let x0 = x_norm.floor().to_usize()?;
            let y0 = y_norm.floor().to_usize()?;
            let z0 = z_norm.floor().to_usize()?;
            let x0 = x0.min(dims.0 - 2);
            let y0 = y0.min(dims.1 - 2);
            let z0 = z0.min(dims.2 - 2);
            let xd = x_norm - T::from(x0).unwrap();
            let yd = y_norm - T::from(y0).unwrap();
            let zd = z_norm - T::from(z0).unwrap();
            Some(([x0, y0, z0], [xd, yd, zd]))
        }

        // https://en.wikipedia.org/wiki/Trilinear_interpolation#Formulation
        fn trilerp(&self, grid_point: [usize; 3], weights: [T; 3], field: &Array4<T>) -> [T; 3] {
            let [x0, y0, z0] = grid_point;
            let [xd, yd, zd] = weights;

            // Collect 3D neighborhood
            let c000 = &field.slice(s![x0, y0, z0, ..]);
            let c001 = &field.slice(s![x0, y0, z0 + 1, ..]);
            let c010 = &field.slice(s![x0, y0 + 1, z0, ..]);
            let c011 = &field.slice(s![x0, y0 + 1, z0 + 1, ..]);
            let c100 = &field.slice(s![x0 + 1, y0, z0, ..]);
            let c101 = &field.slice(s![x0 + 1, y0, z0 + 1, ..]);
            let c110 = &field.slice(s![x0 + 1, y0 + 1, z0, ..]);
            let c111 = &field.slice(s![x0 + 1, y0 + 1, z0 + 1, ..]);

            // Lerps upcoming
            let c00 = [
                c000[0] * (T::one() - xd) + c100[0] * xd,
                c000[1] * (T::one() - xd) + c100[1] * xd,
                c000[2] * (T::one() - xd) + c100[2] * xd,
            ];
            let c01 = [
                c001[0] * (T::one() - xd) + c101[0] * xd,
                c001[1] * (T::one() - xd) + c101[1] * xd,
                c001[2] * (T::one() - xd) + c101[2] * xd,
            ];
            let c10 = [
                c010[0] * (T::one() - xd) + c110[0] * xd,
                c010[1] * (T::one() - xd) + c110[1] * xd,
                c010[2] * (T::one() - xd) + c110[2] * xd,
            ];
            let c11 = [
                c011[0] * (T::one() - xd) + c111[0] * xd,
                c011[1] * (T::one() - xd) + c111[1] * xd,
                c011[2] * (T::one() - xd) + c111[2] * xd,
            ];

            let c0 = [
                c00[0] * (T::one() - yd) + c10[0] * yd,
                c00[1] * (T::one() - yd) + c10[1] * yd,
                c00[2] * (T::one() - yd) + c10[2] * yd,
            ];
            let c1 = [
                c01[0] * (T::one() - yd) + c11[0] * yd,
                c01[1] * (T::one() - yd) + c11[1] * yd,
                c01[2] * (T::one() - yd) + c11[2] * yd,
            ];

            //One more lerp and we there!
            [
                c0[0] * (T::one() - zd) + c1[0] * zd,
                c0[1] * (T::one() - zd) + c1[1] * zd,
                c0[2] * (T::one() - zd) + c1[2] * zd,
            ]
        }
    }

    pub fn earth_dipole<T: PtrTrait>(x: T, y: T, z: T) -> [T; 6] {
        let position_mag = (x * x + y * y + z * z).sqrt();
        let m = T::from(-7800e+12).unwrap();
        let mut b = [T::zero(), T::zero(), T::zero()];
        b[0] = (T::from(3.0).unwrap() * m * x * z) / position_mag.powi(5);
        b[1] = (T::from(3.0).unwrap() * m * y * z) / position_mag.powi(5);
        b[2] = (m / position_mag.powi(3))
            * ((T::from(3.0).unwrap() * z * z) / position_mag.powi(2) - T::one());
        [b[0], b[1], b[2], T::zero(), T::zero(), T::zero()]
    }

    impl<T: PtrTrait> Field<T> for DipoleField<T> {
        fn get_fields_at(&self, _time: T, x: T, y: T, z: T) -> Option<[T; 6]> {
            return Some(earth_dipole::<T>(x, y, z));
        }
    }

    impl<T: PtrTrait> Field<T> for VlsvStaticField<T> {
        fn get_fields_at(&self, _time: T, x: T, y: T, z: T) -> Option<[T; 6]> {
            let (grid_point, weights) = self.real2mesh(x, y, z)?;
            let e_field = self.trilerp(grid_point, weights, &self.e);
            let b_field = self.trilerp(grid_point, weights, &self.b);
            Some([
                b_field[0], b_field[1], b_field[2], e_field[0], e_field[1], e_field[2],
            ])
        }
    }
}
