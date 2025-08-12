mod vlsv_reader;
use ndarray::Zip;
use raylib::prelude::*;
use vlsv_reader::mod_vlsv_reader::VlsvFile;

#[derive(Copy, Clone)]
struct RGB {
    r: f32,
    g: f32,
    b: f32,
}
const VIRIDIS_STOPS: [RGB; 7] = [
    RGB {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    },
    RGB {
        r: 0.283,
        g: 0.141,
        b: 0.458,
    },
    RGB {
        r: 0.254,
        g: 0.265,
        b: 0.530,
    },
    RGB {
        r: 0.207,
        g: 0.372,
        b: 0.553,
    },
    RGB {
        r: 0.164,
        g: 0.471,
        b: 0.558,
    },
    RGB {
        r: 0.477,
        g: 0.717,
        b: 0.441,
    },
    RGB {
        r: 0.993,
        g: 0.906,
        b: 0.144,
    },
];

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
fn lerp_rgb(a: RGB, b: RGB, t: f32) -> RGB {
    RGB {
        r: lerp(a.r, b.r, t),
        g: lerp(a.g, b.g, t),
        b: lerp(a.b, b.b, t),
    }
}

fn build_viridis_256() -> Vec<Color> {
    let mut lut = Vec::with_capacity(256);
    for i in 0..256 {
        let t = (i as f32) / 255.0;
        let n = VIRIDIS_STOPS.len() as f32 - 1.0;
        let pos = t * n;
        let idx = pos.floor() as usize;
        let frac = (pos - pos.floor()).clamp(0.0, 1.0);
        let (a, b) = if idx >= VIRIDIS_STOPS.len() - 1 {
            (
                VIRIDIS_STOPS[VIRIDIS_STOPS.len() - 2],
                VIRIDIS_STOPS[VIRIDIS_STOPS.len() - 1],
            )
        } else {
            (VIRIDIS_STOPS[idx], VIRIDIS_STOPS[idx + 1])
        };
        let c = lerp_rgb(a, b, frac);
        lut.push(Color::new(
            (c.r.clamp(0.0, 1.0) * 255.0) as u8,
            (c.g.clamp(0.0, 1.0) * 255.0) as u8,
            (c.b.clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ));
    }
    lut
}

fn main() {
    let file = std::env::args()
        .skip(1)
        .collect::<Vec<String>>()
        .first()
        .cloned()
        .expect("No file provided!");

    let var = std::env::args()
        .skip(2)
        .collect::<Vec<String>>()
        .first()
        .cloned()
        .expect("No variable provided!");

    let f = VlsvFile::new(&file).unwrap();
    let mut needs_log_scaling = false;
    let mut data = f
        .read_variable::<f32>(&var, Some(4))
        .or_else(|| {
            needs_log_scaling = true;
            f.read_vdf::<f32>(var.parse::<usize>().unwrap(), "proton")
        })
        .unwrap();

    if needs_log_scaling {
        let eps = 1e-30;
        data.mapv_inplace(|v| (v + eps).log10());
    }
    let max = data.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = data.fold(f32::INFINITY, |a, &b| a.min(b));
    let range = max - min;
    Zip::from(&mut data).for_each(|val| {
        *val = (*val - min) / range;
    });

    let (mut rl, thread) = raylib::init()
        .size(2 * 640, 2 * 480)
        .title("Run Dashboard")
        .build();

    rl.set_trace_log(TraceLogLevel::LOG_NONE);
    let mut xcut = data.dim().0 / 2;
    let mut ycut = data.dim().1 / 2;
    let mut zcut = data.dim().2 / 2;
    let mut mode = 2;
    let palette = build_viridis_256();
    while !rl.window_should_close() {
        if rl.is_key_down(KeyboardKey::KEY_UP) {
            match mode {
                0 => xcut = xcut + 1,
                1 => ycut = ycut + 1,
                2 => zcut = zcut + 1,
                _ => panic!(),
            }
        }
        if rl.is_key_down(KeyboardKey::KEY_DOWN) {
            match mode {
                0 => xcut = xcut - 1,
                1 => ycut = ycut - 1,
                2 => zcut = zcut - 1,
                _ => panic!(),
            }
        }

        xcut = xcut.clamp(0, data.dim().0 - 1);
        ycut = ycut.clamp(0, data.dim().1 - 1);
        zcut = zcut.clamp(0, data.dim().2 - 1);

        if rl.is_key_pressed(KeyboardKey::KEY_ONE) {
            mode = 0;
            xcut = data.dim().0 / 2;
            ycut = data.dim().1 / 2;
            zcut = data.dim().2 / 2;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_TWO) {
            mode = 1;
            xcut = data.dim().0 / 2;
            ycut = data.dim().1 / 2;
            zcut = data.dim().2 / 2;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_THREE) {
            mode = 2;
            xcut = data.dim().0 / 2;
            ycut = data.dim().1 / 2;
            zcut = data.dim().2 / 2;
        }

        let text = format!(
            "X={}% | Y={}% | Z= {}% -- [UP/DN to move] [1/2/3 to change slice]",
            100 * xcut / data.dim().0,
            100 * ycut / data.dim().1,
            100 * zcut / data.dim().2,
        );
        let mut image = match mode {
            0 => Image::gen_image_color(data.dim().1 as i32, data.dim().2 as i32, Color::RED),
            1 => Image::gen_image_color(data.dim().0 as i32, data.dim().2 as i32, Color::RED),
            2 => Image::gen_image_color(data.dim().0 as i32, data.dim().1 as i32, Color::RED),
            _ => panic!(),
        };
        let width = image.width();
        let height = image.height();
        for y in 0..height {
            for x in 0..width {
                let val = match mode {
                    0 => data[(xcut, x as usize, y as usize, 0)],
                    1 => data[(x as usize, ycut, y as usize, 0)],
                    2 => data[(x as usize, y as usize, zcut, 0)],
                    _ => panic!(),
                };
                let idx = (val * 255.0).round().clamp(0.0, 255.0) as usize;
                image.draw_pixel(x, y, palette[idx]);
            }
        }
        image.resize(rl.get_screen_width(), rl.get_screen_height());
        let tex = rl.load_texture_from_image(&thread, &image).unwrap();
        let mut d = rl.begin_drawing(&thread);
        d.clear_background(Color::BLACK);
        d.draw_texture(&tex, 0, 0, Color::WHITE);
        d.draw_text(text.as_str(), 0, 0, 20, Color::RED);
    }
}
