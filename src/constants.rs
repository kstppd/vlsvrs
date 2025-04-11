#[allow(dead_code)]
pub mod physical_constants {
    pub mod f64 {
        pub const C: f64 = 299792458.0; // m/s
        pub const C2: f64 = C * C; // m/s
        pub const PROTON_MASS: f64 = 1.67262192e-27; // kg
        pub const PROTON_CHARGE: f64 = 1.602e-19; // C
        pub const ELECTRON_MASS: f64 = 9.1093837e-31; // kg
        pub const ELECTRON_CHARGE: f64 = -PROTON_CHARGE; // C
        pub const JOULE_TO_KEV: f64 = 6.242e+15;
        pub const JOULE_TO_EV: f64 = 6.242e+18;
        pub const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0_f64;
        pub const RAD_TO_DEG: f64 = 180.0_f64 / std::f64::consts::PI;
        pub const EV_TO_JOULE: f64 = 1_f64 / JOULE_TO_EV;
        pub const EARTH_RE: f64 = 6378137.0;
        pub const OUTER_LIM: f64 = 30.0 * EARTH_RE;
        pub const INNER_LIM: f64 = 5.0 * EARTH_RE;
        pub const TOL: f64 = 5e-5;
        pub const PRECIPITATION_RE: f64 = 1.2 * EARTH_RE;
        pub const MAX_STEPS: usize = 10000000;
        pub const DIPOLE_MOMENT: f64 = 8.0e15;
    }
    pub mod f32 {
        pub const C: f32 = 299792458.0; // m/s
        pub const C2: f32 = C * C; // m/s
        pub const PROTON_MASS: f32 = 1.67262192e-27; // kg
        pub const PROTON_CHARGE: f32 = 1.602e-19; // C
        pub const ELECTRON_MASS: f32 = 9.1093837e-31; // kg
        pub const ELECTRON_CHARGE: f32 = -PROTON_CHARGE; // C
        pub const JOULE_TO_KEV: f32 = 6.242e+15;
        pub const JOULE_TO_EV: f32 = 6.242e+18;
        pub const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0_f32;
        pub const RAD_TO_DEG: f32 = 180.0_f32 / std::f32::consts::PI;
        pub const EV_TO_JOULE: f32 = 1_f32 / JOULE_TO_EV;
        pub const EARTH_RE: f32 = 6378137.0;
        pub const OUTER_LIM: f32 = 30.0 * EARTH_RE;
        pub const INNER_LIM: f32 = 5.0 * EARTH_RE;
        pub const TOL: f32 = 5e-5;
        pub const PRECIPITATION_RE: f32 = 1.2 * EARTH_RE;
        pub const MAX_STEPS: usize = 10000000;
        pub const DIPOLE_MOMENT: f32 = 8.0e15;
    }
}
