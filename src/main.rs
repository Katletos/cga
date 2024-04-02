mod app;
mod bresenham;

use crate::app::DickDrawingApp;
use obj::{load_obj, Obj};
use std::fs::File;
use std::io::BufReader;

fn main() {
    let file = File::open("./obj.obj").expect("Can't open file");
    let input = BufReader::new(file);
    let obj: Obj = load_obj(input).expect("Can't load obj");

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_decorations(false)
            .with_fullscreen(true),
        ..Default::default()
    };

    eframe::run_native(
        "eframe template",
        native_options,
        Box::new(|cc| Box::new(DickDrawingApp::new(cc, obj))),
    )
    .unwrap();
}
