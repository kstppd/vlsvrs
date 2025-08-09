pub mod tracer_particles {
    use crate::tracer_fields;
    use crate::tracer_fields::vlsv_reader::PtrTrait;
    use rand::Rng;
    use rand_distr::Normal;
    use std::f64::consts::PI;
    use std::io::Write;
    use tracer_fields::vlsv_reader::Field;

    pub fn mag<T>(x: T, y: T, z: T) -> T
    where
        T: PtrTrait,
    {
        T::sqrt(x * x + y * y + z * z)
    }

    pub fn mag2<T>(x: T, y: T, z: T) -> T
    where
        T: PtrTrait,
    {
        x * x + y * y + z * z
    }

    fn dot<T: PtrTrait>(a: &[T; 3], b: &[T; 3]) -> T {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    pub fn gamma<T>(vx: T, vy: T, vz: T) -> T
    where
        T: PtrTrait,
    {
        let term1: T = T::one();
        let term2: T = T::sqrt(T::one() - (mag2(vx, vy, vz) / T::from(3.0e8 * 3.0e8).unwrap()));
        term1 / term2
    }

    pub struct ParticlePopulation<T: PtrTrait> {
        pub x: Vec<T>,
        pub y: Vec<T>,
        pub z: Vec<T>,
        pub vx: Vec<T>,
        pub vy: Vec<T>,
        pub vz: Vec<T>,
        pub alive: Vec<bool>,
        pub mass: T,
        pub charge: T,
    }

    pub struct ParticleView<'a, T: PtrTrait> {
        pub x: &'a T,
        pub y: &'a T,
        pub z: &'a T,
        pub vx: &'a T,
        pub vy: &'a T,
        pub vz: &'a T,
        pub alive: &'a bool,
    }

    pub struct ParticleIter<'a, T: PtrTrait> {
        population: &'a ParticlePopulation<T>,
        index: usize,
    }

    impl<'a, T: PtrTrait> ParticleIter<'a, T> {
        pub fn new(population: &'a ParticlePopulation<T>) -> Self {
            ParticleIter {
                population,
                index: 0,
            }
        }
    }

    impl<'a, T: PtrTrait> Iterator for ParticleIter<'a, T> {
        type Item = ParticleView<'a, T>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.index >= self.population.x.len() {
                return None;
            }

            let i = self.index;
            self.index += 1;

            Some(ParticleView {
                x: &self.population.x[i],
                y: &self.population.y[i],
                z: &self.population.z[i],
                vx: &self.population.vx[i],
                vy: &self.population.vy[i],
                vz: &self.population.vz[i],
                alive: &self.population.alive[i],
            })
        }
    }

    impl<T: PtrTrait> ParticlePopulation<T> {
        pub fn new(n: usize, mass: T, charge: T) -> Self {
            Self {
                x: Vec::<T>::with_capacity(n),
                y: Vec::<T>::with_capacity(n),
                z: Vec::<T>::with_capacity(n),
                vx: Vec::<T>::with_capacity(n),
                vy: Vec::<T>::with_capacity(n),
                vz: Vec::<T>::with_capacity(n),
                alive: Vec::<bool>::with_capacity(n),
                mass,
                charge,
            }
        }

        pub fn iter(&self) -> ParticleIter<'_, T> {
            ParticleIter {
                population: &self,
                index: 0,
            }
        }

        pub fn save(&self, filename: &str) {
            let size = self.size();
            let datasize = std::mem::size_of::<T>();
            let cap = size * std::mem::size_of::<T>() * 6;
            let mut data: Vec<u8> = Vec::with_capacity(cap);
            let bytes: [u8; std::mem::size_of::<usize>()] = size.to_ne_bytes();
            data.extend_from_slice(&bytes);
            let bytes: [u8; std::mem::size_of::<usize>()] = datasize.to_ne_bytes();
            data.extend_from_slice(&bytes);
            //X
            for i in 0..size {
                let bytes = self.x[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //Y
            for i in 0..size {
                let bytes = self.y[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //Z
            for i in 0..size {
                let bytes = self.z[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //VX
            for i in 0..size {
                let bytes = self.vx[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //VY
            for i in 0..size {
                let bytes = self.vy[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            //VZ
            for i in 0..size {
                let bytes = self.vz[i].to_ne_bytes();
                data.extend_from_slice(&bytes.as_ref());
            }
            println!(
                "\tWriting {}/{} bytes to {}",
                data.len(),
                cap + 8 + 8,
                filename
            );
            let mut file = std::fs::File::create(filename).expect("Failed to create file");
            file.write_all(&data)
                .expect("Failed to write state file  to file!");
        }

        pub fn new_with_energy_at_Lshell(n: usize, mass: T, charge: T, kev: T, L: T) -> Self {
            let mut pop = Self::new(n, mass, charge);
            let c = T::from(3.0e8).unwrap();
            let ke_joules = kev * T::from(1.602e-16).unwrap();

            let rest_energy = mass * c * c;
            let total_energy = ke_joules + rest_energy;

            // Relativistic speed
            let v = c * (T::one() - (rest_energy / total_energy).powi(2)).sqrt();
            let _pitch_angle_dist = Normal::new(90.0, 5.0).unwrap();

            for _ in 0..n {
                let pitch_angle_deg = T::from(45.0).unwrap(); //
                // T::from(pitch_angle_dist.sample(&mut rng).clamp(0.0, 180.0)).unwrap();
                let pitch_angle_rad =
                    pitch_angle_deg * T::from(PI).unwrap() / T::from(180.0).unwrap();

                let v_par = v * pitch_angle_rad.cos();
                let v_perp = v * pitch_angle_rad.sin();

                // Random phase
                let gyro_phase = T::zero() * T::from(rand::random::<f64>() * 2.0 * PI).unwrap();
                let vx = v_perp * gyro_phase.cos();
                let vy = v_perp * gyro_phase.sin();
                let vz = v_par;
                let _theta = rand::rng().random_range(0.0..2.0 * PI);
                let x = L; //T::from(L.to_f64().unwrap() * theta.cos()).unwrap();
                let y = T::zero();
                // T::from(L.to_f64().unwrap() * theta.sin()).unwrap();
                let _z = T::zero();

                pop.add_particle(
                    [
                        x,
                        y,
                        T::zero(),
                        T::from(vx).unwrap(),
                        T::from(vy).unwrap(),
                        T::from(vz).unwrap(),
                    ],
                    true,
                );
            }

            pop
        }
        pub fn add_particle(&mut self, state: [T; 6], status: bool) {
            self.x.push(state[0]);
            self.y.push(state[1]);
            self.z.push(state[2]);
            self.vx.push(state[3]);
            self.vy.push(state[4]);
            self.vz.push(state[5]);
            self.alive.push(status);
        }

        pub fn size(&self) -> usize {
            self.x.len()
        }

        pub fn get_temp_particle(&self, id: usize) -> Particle<T> {
            Particle {
                x: self.x[id],
                y: self.y[id],
                z: self.z[id],
                vx: self.vx[id],
                vy: self.vy[id],
                vz: self.vz[id],
                alive: self.alive[id],
            }
        }

        pub fn take_temp_particle(&mut self, p: &Particle<T>, id: usize) {
            self.x[id] = p.x;
            self.y[id] = p.y;
            self.z[id] = p.z;
            self.vx[id] = p.vx;
            self.vy[id] = p.vy;
            self.vz[id] = p.vz;
            self.alive[id] = p.alive;
        }
    }

    #[derive(Debug, Clone)]
    pub struct Particle<T: PtrTrait> {
        pub x: T,
        pub y: T,
        pub z: T,
        pub vx: T,
        pub vy: T,
        pub vz: T,
        pub alive: bool,
    }

    impl<T: PtrTrait> Particle<T> {
        pub fn new(x: T, y: T, z: T, vx: T, vy: T, vz: T, alive: bool) -> Self {
            Self {
                x,
                y,
                z,
                vx,
                vy,
                vz,
                alive,
            }
        }
    }
    pub fn boris<T: PtrTrait>(p: &mut Particle<T>, e: &[T], b: &[T], dt: T, m: T, c: T) {
        // println!("b={:?},e={:?}", b, e);
        // panic!();
        let mut v_minus: [T; 3] = [T::zero(); 3];
        let mut v_prime: [T; 3] = [T::zero(); 3];
        let mut v_plus: [T; 3] = [T::zero(); 3];
        let mut t: [T; 3] = [T::zero(); 3];
        let mut s: [T; 3] = [T::zero(); 3];
        let g = gamma(p.vx, p.vy, p.vz);
        let cm = c / m;
        t[0] = cm * b[0] * T::from(0.5).unwrap() * dt / g;
        t[1] = cm * b[1] * T::from(0.5).unwrap() * dt / g;
        t[2] = cm * b[2] * T::from(0.5).unwrap() * dt / g;

        let t_mag2 = t[0].powi(2) + t[1].powi(2) + t[2].powi(2);

        s[0] = T::from(2.0).unwrap() * t[0] / (T::one() + t_mag2);
        s[1] = T::from(2.0).unwrap() * t[1] / (T::one() + t_mag2);
        s[2] = T::from(2.0).unwrap() * t[2] / (T::one() + t_mag2);

        v_minus[0] = p.vx + cm * e[0] * T::from(0.5).unwrap() * dt;
        v_minus[1] = p.vy + cm * e[1] * T::from(0.5).unwrap() * dt;
        v_minus[2] = p.vz + cm * e[2] * T::from(0.5).unwrap() * dt;

        v_prime[0] = v_minus[0] + v_minus[1] * t[2] - v_minus[2] * t[1];
        v_prime[1] = v_minus[1] - v_minus[0] * t[2] + v_minus[2] * t[0];
        v_prime[2] = v_minus[2] + v_minus[0] * t[1] - v_minus[1] * t[0];

        v_plus[0] = v_minus[0] + v_prime[1] * s[2] - v_prime[2] * s[1];
        v_plus[1] = v_minus[1] - v_prime[0] * s[2] + v_prime[2] * s[0];
        v_plus[2] = v_minus[2] + v_prime[0] * s[1] - v_prime[1] * s[0];

        p.vx = v_plus[0] + cm * e[0] * T::from(0.5).unwrap() * dt;
        p.vy = v_plus[1] + cm * e[1] * T::from(0.5).unwrap() * dt;
        p.vz = v_plus[2] + cm * e[2] * T::from(0.5).unwrap() * dt;

        p.x = p.x + p.vx * dt;
        p.y = p.y + p.vy * dt;
        p.z = p.z + p.vz * dt;
    }

    pub fn larmor_radius<T: PtrTrait>(particle: &Particle<T>, b: &[T; 3], mass: T, charge: T) -> T {
        let b_mag = mag(b[0], b[1], b[2]);
        let v = [particle.vx, particle.vy, particle.vz];
        let dot_vb = dot(&v, b);
        let v_parallel_mag = dot_vb / b_mag;
        let b_unit = [b[0] / b_mag, b[1] / b_mag, b[2] / b_mag];
        let v_parallel = [
            b_unit[0] * v_parallel_mag,
            b_unit[1] * v_parallel_mag,
            b_unit[2] * v_parallel_mag,
        ];

        let v_perp = [
            v[0] - v_parallel[0],
            v[1] - v_parallel[1],
            v[2] - v_parallel[2],
        ];
        let v_perp_mag = mag(v_perp[0], v_perp[1], v_perp[2]);
        let numerator = mass * v_perp_mag;
        let denominator = charge.abs() * b_mag;
        numerator / denominator
    }
    pub fn borris_adaptive<T: PtrTrait, F: Field<T> + std::marker::Sync>(
        p: &mut Particle<T>,
        f: &F,
        dt: &mut T,
        t0: T,
        t1: T,
        mass: T,
        charge: T,
    ) {
        let mut t = t0;
        while t < t1 {
            //Do not go over t1
            if t + *dt > t1 {
                *dt = t1 - t;
            }

            let mut p1 = p.clone();
            let mut p2 = p.clone();
            //1st order step
            let fields = f.get_fields_at(t, p.x, p.y, p.z).unwrap();
            boris(&mut p1, &fields[3..6], &fields[0..3], *dt, mass, charge);

            let fields = f.get_fields_at(t, p.x, p.y, p.z).unwrap();
            boris(
                &mut p2,
                &fields[3..6],
                &fields[0..3],
                *dt / T::from(2).unwrap(),
                mass,
                charge,
            );

            //Get error
            let error = [
                T::from(100.0).unwrap() * (p2.x - p1.x).abs(),
                T::from(100.0).unwrap() * (p2.y - p1.y).abs(),
                T::from(100.0).unwrap() * (p2.z - p1.z).abs(),
            ]
            .iter()
            .copied()
            .fold(T::neg_infinity(), T::max);

            //Calc new dt
            let b = [fields[0], fields[1], fields[2]];
            let larmor = larmor_radius(p, &b, mass, charge);
            let tol = T::from(larmor / T::from(100).unwrap()).unwrap();
            let new_dt = T::from(0.9).unwrap()
                * *dt
                * T::min(
                    T::max(
                        (tol / (T::from(2.0).unwrap() * error)).sqrt(),
                        T::from(0.3).unwrap(),
                    ),
                    T::from(2.0).unwrap(),
                );

            //Accept step
            if error < tol {
                *p = p1;
                t = t + new_dt;
                *dt = new_dt;
            }
        }
    }
}
