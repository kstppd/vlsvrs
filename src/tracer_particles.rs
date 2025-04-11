pub mod tracer_particles {
    use crate::constants::physical_constants;
    use bytemuck::Pod;
    use num_traits::Float;
    use rand::Rng;
    use rand::thread_rng;
    use rand_distr::{Distribution, Normal};
    use std::f64::consts::PI;
    use std::io::Write;

    pub fn mag<T>(x: T, y: T, z: T) -> T
    where
        T: Float + Pod + Send + Sized + std::marker::Sync + std::fmt::Debug + num_traits::ToBytes,
    {
        T::sqrt(x * x + y * y + z * z)
    }

    pub fn mag2<T>(x: T, y: T, z: T) -> T
    where
        T: Float + Pod + Send + Sized + std::marker::Sync + std::fmt::Debug + num_traits::ToBytes,
    {
        x * x + y * y + z * z
    }

    pub fn gamma<T>(vx: T, vy: T, vz: T) -> T
    where
        T: Float + Pod + Send + Sized + std::marker::Sync + std::fmt::Debug + num_traits::ToBytes,
    {
        let term1: T = T::one();
        let term2: T = T::sqrt(T::one() - (mag2(vx, vy, vz) / T::from(3.0e8 * 3.0e8).unwrap()));
        term1 / term2
    }

    pub struct ParticlePopulation<
        T: Float + Pod + Send + Sized + std::marker::Sync + std::fmt::Debug + num_traits::ToBytes,
    > {
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

    impl<T: Float + Pod + Send + Sized + std::marker::Sync + std::fmt::Debug + num_traits::ToBytes>
        ParticlePopulation<T>
    {
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
            println!("\tWriting {}/{} bytes to {}", data.len(), cap + 8, filename);
            let mut file = std::fs::File::create(filename).expect("Failed to create file");
            file.write_all(&data)
                .expect("Failed to write state file  to file!");
        }

        pub fn new_with_energy_at_Lshell(n: usize, mass: T, charge: T, kev: T, L: T) -> Self {
            let mut pop = Self::new(n, mass, charge);
            let mut rng = thread_rng();
            let c = T::from(3.0e8).unwrap();
            let ke_joules = kev * T::from(1.602e-16).unwrap();

            let rest_energy = mass * c * c;
            let total_energy = ke_joules + rest_energy;

            // Relativistic speed
            let v = c * (T::one() - (rest_energy / total_energy).powi(2)).sqrt();
            let pitch_angle_dist = Normal::new(90.0, 5.0).unwrap();

            for _ in 0..n {
                let pitch_angle_deg =
                    T::from(pitch_angle_dist.sample(&mut rng).clamp(0.0, 180.0)).unwrap();
                let pitch_angle_rad =
                    pitch_angle_deg * T::from(PI).unwrap() / T::from(180.0).unwrap();

                let v_par = v * pitch_angle_rad.cos();
                let v_perp = v * pitch_angle_rad.sin();

                // Random phase
                let gyro_phase = T::from(rand::random::<f64>() * 2.0 * PI).unwrap();
                let vx = v_perp * gyro_phase.cos();
                let vy = v_perp * gyro_phase.sin();
                let vz = v_par;
                let theta = rand::thread_rng().gen_range(0.0..2.0 * PI);
                let x = T::from(L.to_f64().unwrap() * theta.cos()).unwrap();
                let y = T::from(L.to_f64().unwrap() * theta.sin()).unwrap();
                let z = 0.0;

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

    pub struct Particle<
        T: Float + Pod + Send + Sized + std::marker::Sync + std::fmt::Debug + num_traits::ToBytes,
    > {
        pub x: T,
        pub y: T,
        pub z: T,
        pub vx: T,
        pub vy: T,
        pub vz: T,
        pub alive: bool,
    }

    impl<T: Float + Pod + Send + Sized + std::marker::Sync + std::fmt::Debug + num_traits::ToBytes>
        Particle<T>
    {
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
}
