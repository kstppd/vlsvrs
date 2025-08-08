mod vlsv_reader;
use raylib::prelude::*;
use vlsv_reader::vlsv_reader::VlsvFile;

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
        .expect("No var provided!");

    let mut var = VlsvFile::new(&file)
        .unwrap()
        .read_vg_variable_as_fg::<f32>(&var)
        .unwrap();
    println!("a={:?}", var.dim());
    let max = var.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = var.fold(f32::INFINITY, |a, &b| a.min(b));
    let (mut rl, thread) = raylib::init()
        .size(2 * 640, 2 * 480)
        .title("Run Dashboard")
        .build();

    rl.set_trace_log(TraceLogLevel::LOG_NONE);
    let mut xcut = var.dim().0 / 2;
    let mut ycut = var.dim().1 / 2;
    let mut zcut = var.dim().2 / 2;
    let mut mode = 2;
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

        xcut = xcut.clamp(0, var.dim().0 - 1);
        ycut = ycut.clamp(0, var.dim().1 - 1);
        zcut = zcut.clamp(0, var.dim().2 - 1);

        if rl.is_key_pressed(KeyboardKey::KEY_ONE) {
            mode = 0;
            xcut = var.dim().0 / 2;
            ycut = var.dim().1 / 2;
            zcut = var.dim().2 / 2;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_TWO) {
            mode = 1;
            xcut = var.dim().0 / 2;
            ycut = var.dim().1 / 2;
            zcut = var.dim().2 / 2;
        }
        if rl.is_key_pressed(KeyboardKey::KEY_THREE) {
            mode = 2;
            xcut = var.dim().0 / 2;
            ycut = var.dim().1 / 2;
            zcut = var.dim().2 / 2;
        }

        let text = format!(
            "X={}% | Y={}% | Z= {}% -- [UP/DN to move] [1/2/3 to change slice]",
            100 * xcut / var.dim().0,
            100 * ycut / var.dim().1,
            100 * zcut / var.dim().2,
        );
        let mut image = match mode {
            0 => Image::gen_image_color(var.dim().1 as i32, var.dim().2 as i32, Color::RED),
            1 => Image::gen_image_color(var.dim().0 as i32, var.dim().2 as i32, Color::RED),
            2 => Image::gen_image_color(var.dim().0 as i32, var.dim().1 as i32, Color::RED),
            _ => panic!(),
        };
        let width = image.width();
        let height = image.height();
        for y in 0..height {
            for x in 0..width {
                let mut val = match mode {
                    0 => var[(xcut, x as usize, y as usize, 0)],
                    1 => var[(x as usize, ycut, y as usize, 0)],
                    2 => var[(x as usize, y as usize, zcut, 0)],
                    _ => panic!(),
                };
                val = (val - min) / (max - min);
                image.draw_pixel(
                    x,
                    y,
                    color::Color::color_from_normalized(Vector4 {
                        x: val as f32,
                        y: val as f32,
                        z: val as f32,
                        w: 1.0_f32,
                    }),
                );
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
